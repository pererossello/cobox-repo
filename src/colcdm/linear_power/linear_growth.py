from typing import Literal, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..background.cosmology import Cosmology
from ._symbolic_pofk import growth_correction_R

LinearGrowthKindLiteral = Literal["symbolic_pofk", "hypergeometric"]
LINEAR_GROWTH_KINDS = ("symbolic_pofk", "hypergeometric")


def growth_symbolic_pofk(cosmology: Cosmology, a: ArrayLike) -> jax.Array:

    c = cosmology
    a = jnp.asarray(a, dtype=float)
    a = eqx.error_if(
        a,
        ~c.is_flat,
        "The symbolic_pofk backend supports only flat cosmologies (Omega_k = 0).",
    )
    matter = c.Omega_m * a**-3
    dark_energy = (
        (1.0 - c.Omega_m) * a ** (-3 * (1 + c.w0 + c.wa)) * jnp.exp(-3 * c.wa * (1 - a))
    )
    total = matter + dark_energy
    Om_a = matter / total
    OL_a = dark_energy / total
    D1 = 2.5 * a * Om_a / (Om_a ** (4 / 7) - OL_a + (1 + Om_a / 2) * (1 + OL_a / 70))
    R = growth_correction_R(
        As=None,  # dummy argument
        Om=c.Omega_m,
        Ob=c.Omega_b,
        h=c.h,
        ns=None,  # dummy argument
        mnu=0.0,
        w0=c.w0,
        wa=c.wa,
        a=a,
    )

    return D1 * jnp.sqrt(R)  # type: ignore


def growth_hypergeometric(cosmology: Cosmology, a: ArrayLike) -> jax.Array:
    """
    Linear growth factor D1(a) via the symbolic hypergeometric approximation of
    Bartlett & Pandey 2025 (arXiv:2510.18749), Eq. 2.4.
    """
    c = cosmology
    a = jnp.asarray(a)
    a = eqx.error_if(a, ~c.is_flat, "Only works for flat cosmologies")
    a = eqx.error_if(
        a, ~c.is_de_Lambda, "Only works for cosmological constant (no w0wa)"
    )
    b0, b1 = 0.723, 1.204
    x = a**3 * (c.Omega_m - 1.0) / c.Omega_m
    return a * jnp.sqrt(b0 ** (2 / 3) + b1) / jnp.sqrt((b0 - x) ** (2 / 3) + b1)  # type: ignore


LINEAR_GROWTH_DISPATCH: dict[
    LinearGrowthKindLiteral, Callable[[Cosmology, ArrayLike], jax.Array]
] = {
    "symbolic_pofk": growth_symbolic_pofk,
    "hypergeometric": growth_hypergeometric,
}


def growth_factor(
    a: ArrayLike,
    cosmology: Cosmology,
    kind: LinearGrowthKindLiteral = "symbolic_pofk",
    normalized: bool = True,
) -> jax.Array:
    """Linear growth factor D(a); normalized=True returns D(a) / D(1)."""
    if kind not in LINEAR_GROWTH_DISPATCH:
        raise ValueError(
            f"unknown linear growth kind: {kind!r}; expected one of {LINEAR_GROWTH_KINDS}."
        )
    growth = LINEAR_GROWTH_DISPATCH[kind]
    D = growth(cosmology, a)
    return D / growth(cosmology, 1.0) if normalized else D
