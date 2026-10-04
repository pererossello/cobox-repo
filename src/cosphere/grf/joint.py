"""Correlated Gaussian real scalar fields on a common shell."""

from collections.abc import Callable, Sequence
from numbers import Integral
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from ..shell import _harmonic
from ..shell.modemask import EllModeMask
from ._covariance import covariance_factor
from ._modes import sampling_mask

if TYPE_CHECKING:
    from ..field import ShellField
    from ..shell import Shell


class JointShellGRFSampler(eqx.Module):
    """Draw n_fields correlated real scalar skies from C_ell[i, j].

    cl_fn receives shell.ell_1D, shape (L,), and must return a real symmetric
    covariance of shape (L, n_fields, n_fields). Field order is preserved in
    the returned tuple. Signed cross-spectra and singular PSD covariances are
    valid. The same restrictions apply to every field; the monopole is omitted.

    randoms['u'] is one flat standard-normal vector, grouped by field, then by
    ShellGRFSampler's mode convention. All fields have the same Shell. Mixing
    acts on the field axis at each ell, with the usual 1/sqrt(2) for m > 0.
    The factor is the symmetric PSD square root; no diagonal noise is added.

    First derivatives of sampling are well-defined for positive-definite kept
    covariances, including repeated eigenvalues. At rank changes a square root
    is not generally differentiable; do not infer regular gradients there.
    get_n_dof counts latent coordinates, not covariance rank: null directions
    still have allocated randoms but contribute nothing to the sampled fields.
    """

    n_fields: int = eqx.field(static=True)
    restrictions: tuple[EllModeMask, ...] = eqx.field(static=True)

    def __init__(
        self, n_fields: int, restrictions: EllModeMask | Sequence[EllModeMask] = ()
    ):
        if (
            isinstance(n_fields, bool)
            or not isinstance(n_fields, Integral)
            or n_fields < 1
        ):
            raise ValueError("n_fields must be a positive integer.")
        self.n_fields = int(n_fields)
        if isinstance(restrictions, EllModeMask):
            restrictions = (restrictions,)
        self.restrictions = tuple(restrictions)
        if not all(isinstance(r, EllModeMask) for r in self.restrictions):
            raise TypeError("restrictions must be EllModeMask instances.")

    def get_n_dof(self, shell: "Shell") -> dict:
        return {"u": self.n_fields * _harmonic.n_dof(self._get_mask(shell))}

    def get_randoms(self, seed: int, shell: "Shell") -> dict:
        return {
            "u": jax.random.normal(
                jax.random.PRNGKey(seed), (self.get_n_dof(shell)["u"],)
            )
        }

    def sample(
        self,
        randoms: dict,
        shell: "Shell",
        cl_fn: Callable[[jax.Array], jax.Array],
    ) -> tuple["ShellField", ...]:
        """Transform independent latent coordinates into jointly distributed fields."""
        from ..field import ShellField

        mask = self._get_mask(shell)
        n_modes = _harmonic.n_dof(mask)
        u = jnp.asarray(randoms["u"])
        expected = (self.n_fields * n_modes,)
        if u.shape != expected:
            raise ValueError(f"expected randoms['u'] shape {expected}; got {u.shape}.")
        if jnp.iscomplexobj(u):
            raise TypeError("randoms['u'] must be real.")
        u = eqx.error_if(u, jnp.any(~jnp.isfinite(u)), "randoms['u'] must be finite.")
        raw = jax.vmap(_harmonic.pack, in_axes=(0, None))(
            u.reshape(self.n_fields, n_modes), mask
        )
        factor = covariance_factor(shell, mask, cl_fn, self.n_fields)
        mixed = jnp.einsum(
            "lij,jlm->ilm", factor, raw, precision=jax.lax.Precision.HIGHEST
        )
        mode_weight = jnp.where(shell.m_weights > 0, shell.m_weights, 1)
        mixed = jnp.where(mask[None], mixed / jnp.sqrt(mode_weight)[None], 0)
        return tuple(
            ShellField(mixed[i], shell, has_hat=True) for i in range(self.n_fields)
        )

    def sample_from_seed(
        self, seed: int, shell: "Shell", cl_fn: Callable
    ) -> tuple["ShellField", ...]:
        return self.sample(self.get_randoms(seed, shell), shell, cl_fn)

    def randoms_from_grf(
        self,
        fields: Sequence["ShellField"],
        shell: "Shell",
        cl_fn: Callable,
    ) -> dict:
        """Recover latents for full rank, or minimum-norm latents for singular power.

        Uses a pseudoinverse with relative cutoff n_fields * machine epsilon on
        the square-root factor. Null-space randoms are unrecoverable; sampling
        the recovered coordinates reproduces supported field components.
        Pixel fields require the optional harmonic transform backend. Excluded
        modes are ignored, as in ShellGRFSampler.randoms_from_grf.
        """
        fields = tuple(fields)
        if len(fields) != self.n_fields or any(f.shell != shell for f in fields):
            raise ValueError(
                "fields must contain n_fields fields on the supplied shell."
            )
        mask = self._get_mask(shell)
        factor = covariance_factor(shell, mask, cl_fn, self.n_fields)
        cutoff = self.n_fields * jnp.finfo(factor.dtype).eps
        inverse = jnp.linalg.pinv(factor, rtol=cutoff, hermitian=True)
        data = jnp.stack([f.sht().data for f in fields])
        data = jnp.where(mask[None], data, 0) * jnp.sqrt(shell.m_weights)[None]
        raw = jnp.einsum(
            "lij,jlm->ilm", inverse, data, precision=jax.lax.Precision.HIGHEST
        )
        u = jax.vmap(_harmonic.unpack, in_axes=(0, None))(raw, mask)
        return {"u": u.reshape(-1)}

    def get_dist(self) -> dict:
        return {"u": dist.Normal(0.0, 1.0)}

    def get_randoms_numpyro(self, shell: "Shell", prefix: str = "") -> dict:
        n = self.get_n_dof(shell)["u"]
        return {
            "u": numpyro.sample(
                f"{prefix}u", self.get_dist()["u"].expand((n,)).to_event(1)
            )
        }

    def _get_mask(self, shell: "Shell"):
        return sampling_mask(shell, self.restrictions)

    def to_dict(self) -> dict:
        return {
            "n_fields": self.n_fields,
            "restrictions": [r.to_dict() for r in self.restrictions],
        }

    @classmethod
    def from_dict(cls, config: dict) -> "JointShellGRFSampler":
        unknown = set(config) - {"n_fields", "restrictions"}
        if unknown:
            raise ValueError(
                f"unknown keys in JointShellGRFSampler config: {sorted(unknown)}."
            )
        if "n_fields" not in config:
            raise ValueError("JointShellGRFSampler config requires n_fields.")
        return cls(
            config["n_fields"],
            tuple(EllModeMask.from_dict(r) for r in config.get("restrictions", ())),
        )

    def to_yaml(self) -> str:
        import yaml

        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_yaml(cls, s: str) -> "JointShellGRFSampler":
        import yaml

        return cls.from_dict(yaml.safe_load(s))
