from collections.abc import Sequence
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from ..shell import _harmonic
from ..masks import EllModeMask
from ._modes import sampling_mask
from ._spectrum import shell_values

if TYPE_CHECKING:
    from ..field import ShellField
    from ..shell import Shell
    from ..spectrum import AngularSpectrum


class ShellGRFSampler(eqx.Module):
    """Isotropic Gaussian random fields on a Shell, from an angular spectrum C_ell.

    Harmonic-space white noise: one iid N(0, 1) latent per real dof of the
    kept modes, f_l0 = sqrt(C_l) u and f_lm = sqrt(C_l / 2) (u_re + i u_im),
    so <|f_lm|^2> = C_l. The monopole is never a dof; ``restrictions`` cut
    further multipoles (e.g. EllModeMask((2, None)) drops the dipole).
    """

    restrictions: tuple[EllModeMask, ...] = eqx.field(static=True, default=())

    def __init__(self, restrictions: EllModeMask | Sequence[EllModeMask] = ()):
        if isinstance(restrictions, EllModeMask):
            restrictions = (restrictions,)
        self.restrictions = tuple(restrictions)
        self._validate_static()

    def get_randoms(self, seed: int, shell: "Shell") -> dict:
        key = jax.random.PRNGKey(seed)
        n = self.get_n_dof(shell)["u"]
        return {"u": jax.random.normal(key, shape=(n,))}

    def sample(
        self,
        randoms: dict,
        shell: "Shell",
        cl: "AngularSpectrum",
    ) -> "ShellField":
        from ..field import ShellField

        u = jnp.asarray(randoms["u"])
        mask = self._get_mask(shell)
        n_dof = _harmonic.n_dof(mask)
        if u.shape != (n_dof,):
            raise ValueError(f"expected randoms['u'] shape ({n_dof},); got {u.shape}.")
        if jnp.iscomplexobj(u):
            raise TypeError("randoms['u'] must be real.")
        u = eqx.error_if(u, jnp.any(~jnp.isfinite(u)), "randoms['u'] must be finite.")
        raw = _harmonic.pack(u, mask)
        return ShellField(
            data=raw * self._multiplier(shell, mask, cl), shell=shell, has_hat=True
        )

    def sample_from_seed(
        self,
        seed: int,
        shell: "Shell",
        cl: "AngularSpectrum",
    ) -> "ShellField":
        return self.sample(self.get_randoms(seed, shell), shell, cl)

    def randoms_from_grf(
        self,
        field: "ShellField",
        shell: "Shell",
        cl: "AngularSpectrum",
    ) -> dict:
        """Recover latents on retained modes with positive power; zero elsewhere.

        Exact for harmonic fields sampled with positive retained power. A pixel
        field goes through ``sht`` and is as accurate as that transform.
        """
        if field.shell != shell:
            raise ValueError("field must live on the supplied shell.")
        mask = self._get_mask(shell)
        multiplier = self._multiplier(shell, mask, cl)
        safe = jnp.where(multiplier > 0, multiplier, 1.0)
        raw = jnp.where(multiplier > 0, field.sht().data / safe, 0.0)
        return {"u": _harmonic.unpack(raw, mask)}

    def get_n_dof(self, shell: "Shell") -> dict:
        return {"u": _harmonic.n_dof(self._get_mask(shell))}

    # ------------------
    # --- INFERENCE ----
    # ------------------

    def get_dist(self) -> dict:
        return {"u": dist.Normal(0.0, 1.0)}

    def get_randoms_numpyro(self, shell: "Shell", prefix: str = "") -> dict:
        return {
            key_name: numpyro.sample(
                f"{prefix}{key_name}",
                self.get_dist()[key_name].expand((n,)).to_event(1),
            )
            for key_name, n in self.get_n_dof(shell).items()
        }

    # ------------------
    # --- UTILITIES ----
    # ------------------

    def _multiplier(self, shell: "Shell", mask, cl: "AngularSpectrum") -> jax.Array:
        """sqrt(C_l / w_m) on kept modes, w_m = 1 at m = 0 and 2 at m > 0."""
        values = shell_values(cl, shell, ())
        keep = jnp.any(mask, axis=-1)
        values = jnp.where(keep, values, 0.0)
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values) | (values < 0)),
            "Auto-power must be finite and nonnegative at retained multipoles.",
        )
        w = jnp.where(shell.m_weights > 0, shell.m_weights, 1.0)
        # Excluded entries use a positive argument before sqrt, keeping their
        # gradients finite even when their input spectra are undefined.
        variance = jnp.where(mask, values[:, None] / w, 1.0)
        return jnp.where(mask, jnp.sqrt(variance), 0.0)

    def _get_mask(self, shell: "Shell"):
        return sampling_mask(shell, self.restrictions)

    # ---------------------
    # --- SERIALIZATION ---
    # ---------------------

    def to_dict(self) -> dict:
        from . import _serialize

        return _serialize.grf_to_dict(self)

    @classmethod
    def from_dict(cls, config: dict) -> "ShellGRFSampler":
        from . import _serialize

        return _serialize.grf_from_dict(config, cls)

    def to_yaml(self) -> str:
        import yaml

        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_yaml(cls, s: str) -> "ShellGRFSampler":
        import yaml

        return cls.from_dict(yaml.safe_load(s))

    # ------------------
    # --- VALIDATION ---
    # ------------------

    def _validate_static(self) -> None:
        if not all(isinstance(r, EllModeMask) for r in self.restrictions):
            raise TypeError("restrictions must be EllModeMask instances.")
