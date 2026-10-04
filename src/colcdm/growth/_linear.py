from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..cosmology.background import BackgroundCosmo
from ._ode import A_INIT, N_STEPS, check_a_range, hermite_interp, ode_table
from ._symbolic_pofk import growth_correction_R

LinearGrowthKindLiteral = Literal["symbolic_pofk", "hypergeometric", "ode"]
LINEAR_GROWTH_KINDS = ("symbolic_pofk", "hypergeometric", "ode")


def growth_symbolic_pofk(background: BackgroundCosmo, a: ArrayLike) -> jax.Array:

    c = background
    a = jnp.asarray(a, dtype=float)
    a = eqx.error_if(
        a,
        ~c.is_flat,
        "The symbolic_pofk backend supports only flat cosmologies (Omega_k = 0).",
    )
    matter = c.Omega_m * a**-3
    dark_energy = (
        (1.0 - c.Omega_m) * a ** (-3 * (1 + c.w0 + c.wa)) * jnp.exp(-3 * c.wa * (1 - a))
    )  # type: ignore
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


def growth_hypergeometric(background: BackgroundCosmo, a: ArrayLike) -> jax.Array:
    """
    Linear growth factor D1(a) via the symbolic hypergeometric approximation of
    Bartlett & Pandey 2025 (arXiv:2510.18749), Eq. 2.4.
    """
    c = background
    a = jnp.asarray(a)
    a = eqx.error_if(a, ~c.is_flat, "Only works for flat cosmologies")
    a = eqx.error_if(
        a, ~c.is_de_Lambda, "Only works for cosmological constant (no w0wa)"
    )
    b0, b1 = 0.723, 1.204
    x = a**3 * (c.Omega_m - 1.0) / c.Omega_m
    return a * jnp.sqrt(b0 ** (2 / 3) + b1) / jnp.sqrt((b0 - x) ** (2 / 3) + b1)  # type: ignore


def growth_ode(
    background: BackgroundCosmo,
    a: ArrayLike,
    *,
    a_init: float = A_INIT,
    n_steps: int = N_STEPS,
) -> jax.Array:
    """
    Linear growth factor D1(a) from the growth ODE of the background
    (matter, curvature and CPL dark energy; no radiation), for a_init <= a <= 1.
    Growing mode D1 -> a at early times, as for the other kinds.
    """
    a = check_a_range(jnp.asarray(a, dtype=float), a_init)
    grid, D, F = ode_table(background, a_init=a_init, n_steps=n_steps)
    log_D, _ = hermite_interp(grid, jnp.log(D), F / D, jnp.log(a))
    return jnp.exp(log_D)


LINEAR_GROWTH_DISPATCH: dict[
    LinearGrowthKindLiteral, Callable[[BackgroundCosmo, ArrayLike], jax.Array]
] = {
    "symbolic_pofk": growth_symbolic_pofk,
    "hypergeometric": growth_hypergeometric,
    "ode": growth_ode,
}
