from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from ..shell import _harmonic
from ..shell.modemask import EllModeMask
from ._modes import sampling_mask

if TYPE_CHECKING:
    from ..field import ShellField
    from ..shell import Shell


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
        cl_fn: Callable[[jax.Array], jax.Array],
    ) -> "ShellField":
        from ..field import ShellField

        u = randoms["u"]
        mask = self._get_mask(shell)
        n_dof = _harmonic.n_dof(mask)
        if u.shape != (n_dof,):
            raise ValueError(f"expected randoms['u'] shape ({n_dof},); got {u.shape}.")
        raw = _harmonic.pack(u, mask)
        return ShellField(
            data=raw * self._multiplier(shell, mask, cl_fn), shell=shell, has_hat=True
        )

    def sample_from_seed(
        self,
        seed: int,
        shell: "Shell",
        cl_fn: Callable[[jax.Array], jax.Array],
    ) -> "ShellField":
        return self.sample(self.get_randoms(seed, shell), shell, cl_fn)

    def randoms_from_grf(
        self,
        field: "ShellField",
        shell: "Shell",
        cl_fn: Callable[[jax.Array], jax.Array],
    ) -> dict:
        """Inverse of ``sample``: exact for a harmonic-space field; a pixel-space
        field goes through ``sht`` and is as accurate as that transform."""
        mask = self._get_mask(shell)
        multiplier = self._multiplier(shell, mask, cl_fn)
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

    def _multiplier(self, shell: "Shell", mask, cl_fn) -> jax.Array:
        """sqrt(C_l / w_m) on kept modes, w_m = 1 at m = 0 and 2 at m > 0."""
        ell = shell.ell_axis.astype(float)
        cl = jnp.where(ell > 0, cl_fn(ell), 0.0)
        w = jnp.where(shell.m_weights > 0, shell.m_weights, 1.0)
        return jnp.where(mask, jnp.sqrt(cl / w), 0.0)

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
