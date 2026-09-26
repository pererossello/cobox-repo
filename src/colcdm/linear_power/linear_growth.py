from typing import Literal, Callable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..background.cosmology import Cosmology
from ._symbolic_pofk import growth_correction_R

LinearGrowthKindLiteral = Literal["symbolic_pofk", "hypergeometric"]
LINEAR_GROWTH_KINDS = ("symbolic_pofk", "hypergeometric")


def growth_symbolic_pofk(cosmology: Cosmology, a: ArrayLike) -> jax.Array:

    c = cosmology
    if not c.is_flat:
        raise ValueError(
            "The symbolic_pofk backend supports only flat cosmologies (Omega_k = 0)."
        )

    # Massless limit of get_approximate_D, with the (1 + z_eq) factors
    # cancelled analytically. The massive-neutrino formula is singular at
    # f_cb = 1 and produces NaN parameter gradients in float32.
    a = jnp.asarray(a, dtype=float)
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

    return D1 * jnp.sqrt(R)


def growth_hypergeometric(cosmology: Cosmology, a: ArrayLike) -> jax.Array:
    """
    Linear growth factor D1(a) via the symbolic hypergeometric approximation of
    Bartlett & Pandey 2025 (arXiv:2510.18749), Eq. 2.4.
    """
    c = cosmology
    if not c.is_flat:
        raise ValueError("Only works for flat cosmologies")
    if not c.is_de_Lambda:
        raise ValueError("Only works for cosmological constant (no w0wa)")

    a = jnp.asarray(a)
    b0, b1 = 0.723, 1.204
    x = a**3 * (c.Omega_m - 1.0) / c.Omega_m
    return a * jnp.sqrt(b0 ** (2 / 3) + b1) / jnp.sqrt((b0 - x) ** (2 / 3) + b1)


LINEAR_GROWTH_DISPATCH: dict[
    LinearGrowthKindLiteral, Callable[[Cosmology, ArrayLike], jax.Array]
] = {
    "symbolic_pofk": growth_symbolic_pofk,
    "hypergeometric": growth_hypergeometric,
}
