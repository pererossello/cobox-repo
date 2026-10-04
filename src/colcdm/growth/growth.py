"""Reusable linear growth model; cosmological parameters are supplied per call."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..cosmology import BackgroundCosmo
from ._linear import (
    LINEAR_GROWTH_DISPATCH,
    LINEAR_GROWTH_KINDS,
    LinearGrowthKindLiteral,
)


class Growth(eqx.Module):
    """Linear growth approximation with explicit normalization.

    D(a) = g(a) / g(1), where unnormalized(a) returns the original g(a).
    f(a) = d ln D / d ln a = d ln g / d ln a. symbolic_pofk and hypergeometric
    require flat geometry; hypergeometric additionally requires Lambda dark energy.
    ode integrates the growth equation of the background (any curvature and CPL
    dark energy) and requires a <= 1; its table is rebuilt on each call.
    No cosmology or computed tables are retained on the model.
    """

    kind: LinearGrowthKindLiteral = eqx.field(static=True, default="symbolic_pofk")

    def __check_init__(self):
        if self.kind not in LINEAR_GROWTH_DISPATCH:
            raise ValueError(
                f"unknown linear growth kind: {self.kind!r}; expected one of {LINEAR_GROWTH_KINDS}."
            )

    def unnormalized(self, a: ArrayLike, background: BackgroundCosmo) -> jax.Array:
        """Original growth amplitude g(a); retain g(1) when normalizing power."""
        a = jnp.asarray(a, dtype=float)
        a = eqx.error_if(
            a, jnp.any(~jnp.isfinite(a) | (a <= 0)), "a must be finite and positive."
        )
        return LINEAR_GROWTH_DISPATCH[self.kind](background, a)

    def D(self, a: ArrayLike, background: BackgroundCosmo) -> jax.Array:
        """Growth normalized to D(1)=1; ordinary array broadcasting applies."""
        return self.unnormalized(a, background) / self.unnormalized(1.0, background)

    def f(self, a: ArrayLike, background: BackgroundCosmo) -> jax.Array:
        """Pointwise logarithmic growth rate, including for arrays of times."""
        a = jnp.asarray(a, dtype=float)
        a = eqx.error_if(
            a, jnp.any(~jnp.isfinite(a) | (a <= 0)), "a must be finite and positive."
        )
        ln_a = jnp.log(a)
        _, slope = jax.jvp(
            lambda x: jnp.log(self.unnormalized(jnp.exp(x), background)),
            (ln_a,),
            (jnp.ones_like(ln_a),),
        )
        return slope

    def to_dict(self) -> dict:
        return {"kind": self.kind}

    @classmethod
    def from_dict(cls, config: dict) -> "Growth":
        unknown = set(config) - {"kind"}
        if unknown:
            raise ValueError(f"unknown keys in Growth config: {sorted(unknown)}.")
        return cls(**config)


def growth_factor(
    a: ArrayLike,
    background: BackgroundCosmo,
    kind: LinearGrowthKindLiteral = "symbolic_pofk",
    normalized: bool = True,
) -> jax.Array:
    """Convenience function delegating to Growth; no separate implementation."""
    growth = Growth(kind)
    return growth.D(a, background) if normalized else growth.unnormalized(a, background)
