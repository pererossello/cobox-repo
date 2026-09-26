from typing import Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

import equinox as eqx

from ..background.cosmology import Cosmology
from ._transfer import TransferKindLiteral, TRANSFER_DISPATCH
from .linear_growth import LinearGrowthKindLiteral, LINEAR_GROWTH_DISPATCH


class LinearMatterPowSpec(eqx.Module):
    transfer_kind: TransferKindLiteral
    growth_kind: LinearGrowthKindLiteral
    cosmology: Optional[Cosmology] = None

    def pk_shape(
        self, k: ArrayLike, cosmology: Optional[Cosmology] = None
    ) -> jax.Array:
        c = self._resolve_cosmology(cosmology)
        return TRANSFER_DISPATCH[self.transfer_kind](c, k)

    def linear_growth(
        self, a: ArrayLike, cosmology: Optional[Cosmology] = None
    ) -> jax.Array:
        """Linear growth factor D(a) (unnormalized; D -> a at early times)."""
        c = self._resolve_cosmology(cosmology)
        return LINEAR_GROWTH_DISPATCH[self.growth_kind](c, a)

    def pk(
        self,
        k: ArrayLike,
        a: ArrayLike = 1.0,
        cosmology: Optional[Cosmology] = None,
    ) -> jax.Array:
        c = self._resolve_cosmology(cosmology)
        growth_fn = LINEAR_GROWTH_DISPATCH[self.growth_kind]
        return self.pk_shape(k, c) * growth_fn(c, a) ** 2

    def delta_sq(
        self,
        k: ArrayLike,
        a: ArrayLike = 1.0,
        cosmology: Optional[Cosmology] = None,
    ) -> jax.Array:
        """Dimensionless power spectrum
        Delta^2(k, a) = k^3 P(k, a) / (2 pi^2)."""
        k = jnp.asarray(k)
        return k**3 * self.pk(k, a, cosmology) / (2.0 * jnp.pi**2)

    def sigma_sq_R(
        self,
        R: ArrayLike,
        a: ArrayLike = 1.0,
        cosmology: Optional[Cosmology] = None,
        *,
        k_min: float = 1e-4,
        k_max: float = 1e2,
        n_k: int = 2048,
    ) -> jax.Array:
        """Variance of the density field smoothed with a 3D spherical top-hat of
        radius R [Mpc/h]
        sigma^2(R) = 1/(2 pi^2) int P(k) W^2(kR) k^3 dln k."""

        from cobox.window.window import win_hat_iso

        c = self._resolve_cosmology(cosmology)
        R = jnp.asarray(R)
        lnk = jnp.linspace(jnp.log(k_min), jnp.log(k_max), n_k)
        k = jnp.exp(lnk)
        # Put the integration axis first; broadcast k against R's shape.
        kc = k.reshape((-1,) + (1,) * R.ndim)
        # scale = diameter = 2R; normalized=True gives the bare shape W(kR), W(0)=1.
        W = win_hat_iso("radial_tophat", 2.0 * R, kc, 3, True)
        integrand = self.pk(k, a, c).reshape((-1,) + (1,) * R.ndim) * W**2 * kc**3
        return jnp.trapezoid(integrand, lnk, axis=0) / (2.0 * jnp.pi**2)

    def sigma_R(
        self,
        R: ArrayLike,
        a: ArrayLike = 1.0,
        cosmology: Optional[Cosmology] = None,
        **kw,
    ) -> jax.Array:
        """RMS density fluctuation in a top-hat sphere of radius R [Mpc/h]."""
        return jnp.sqrt(self.sigma_sq_R(R, a, cosmology, **kw))

    def sigma8(
        self, a: ArrayLike = 1.0, cosmology: Optional[Cosmology] = None, **kw
    ) -> jax.Array:
        """sigma_8 = sigma(R = 8 Mpc/h), the standard amplitude normalization."""
        return self.sigma_R(8.0, a, cosmology, **kw)

    def integrate(
        self,
        k_range: tuple[float, float],
        a: ArrayLike = 1.0,
        cosmology: Optional[Cosmology] = None,
        n_k: int = 2048,
        k_power: int = 3,
    ):
        c = self._resolve_cosmology(cosmology)
        lnk = jnp.linspace(jnp.log(k_range[0]), jnp.log(k_range[1]), n_k)
        k = jnp.exp(lnk)
        integrand = self.pk(k, a, c) * k**k_power
        return jnp.trapezoid(integrand, lnk, axis=0) / (2.0 * jnp.pi**2)

    def __call__(
        self,
        k: ArrayLike,
        a: ArrayLike = 1.0,
        cosmology: Optional[Cosmology] = None,
    ) -> jax.Array:
        return self.pk(k, a, cosmology)

    def _resolve_cosmology(self, cosmology: Optional[Cosmology]) -> Cosmology:
        c = cosmology if cosmology is not None else self.cosmology
        if c is None:
            raise ValueError(
                "a cosmology must be provided, either at construction or per call"
            )
        return c

    # ---------------------
    # --- SERIALIZATION ---
    # ---------------------

    def to_dict(self) -> dict:
        from . import _serialize

        return _serialize.linear_power_to_dict(self)

    @classmethod
    def from_dict(
        cls, config: dict, cosmology: Optional[Cosmology] = None
    ) -> "LinearMatterPowSpec":
        from . import _serialize

        return _serialize.linear_power_from_dict(config, cls, cosmology)

    def to_yaml(self) -> str:
        from . import _serialize

        return _serialize.linear_power_to_yaml(self)

    @classmethod
    def from_yaml(
        cls, s: str, cosmology: Optional[Cosmology] = None
    ) -> "LinearMatterPowSpec":
        from . import _serialize

        return _serialize.linear_power_from_yaml(s, cls, cosmology)
