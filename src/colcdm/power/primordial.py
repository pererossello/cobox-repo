"""Primordial curvature spectrum; parameters come from PrimordialCosmo per call."""

from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from cobox.spectrum import PowerSpectrum

from ..cosmology.primordial import PrimordialCosmo

PrimordialKindLiteral = Literal["power_law"]
PRIMORDIAL_KINDS = ("power_law",)


def delta_sq_power_law(primordial: PrimordialCosmo, k: ArrayLike) -> jax.Array:
    """Delta^2_R(k) = As (k / k_pivot)^(n_s - 1); n_s = 1 is Harrison-Zel'dovich."""
    k = jnp.asarray(k)
    return primordial.As * (k / primordial.k_pivot) ** (primordial.n_s - 1.0)


PRIMORDIAL_DISPATCH: dict[
    PrimordialKindLiteral, Callable[[PrimordialCosmo, ArrayLike], jax.Array]
] = {
    "power_law": delta_sq_power_law,
}


class PrimordialSpectrum(PowerSpectrum):
    """Primordial curvature power P_R(k, primordial) = 2 pi^2 Delta^2_R(k) / k^3.

    k is physical, in 1/Mpc, and P_R in Mpc^3.
    """

    kind: PrimordialKindLiteral = eqx.field(static=True, default="power_law")

    def __check_init__(self):
        if self.kind not in PRIMORDIAL_DISPATCH:
            raise ValueError(
                f"unknown primordial kind: {self.kind!r}; expected one of {PRIMORDIAL_KINDS}."
            )

    def __call__(self, k: ArrayLike, primordial: PrimordialCosmo) -> jax.Array:
        k = jnp.asarray(k, dtype=float)
        return 2.0 * jnp.pi**2 / k**3 * self.delta_sq(k, primordial)

    def delta_sq(self, k: ArrayLike, primordial: PrimordialCosmo) -> jax.Array:
        return PRIMORDIAL_DISPATCH[self.kind](primordial, k)

    def to_dict(self) -> dict:
        return {"kind": self.kind}

    @classmethod
    def from_dict(cls, config: dict) -> "PrimordialSpectrum":
        unknown = set(config) - {"kind"}
        if unknown:
            raise ValueError(
                f"unknown keys in PrimordialSpectrum config: {sorted(unknown)}."
            )
        return cls(**config)
