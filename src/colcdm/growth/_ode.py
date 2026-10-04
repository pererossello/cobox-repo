"""Growth ODE utilities: RK4 tables of the background's growth equation, and interpolation.

Tables are RK4 on a uniform grid in ln a, from an EdS start at a_init to a = 1,
and are rebuilt on each call (under jit, repeated solves are deduplicated).
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from ..cosmology.background import BackgroundCosmo

A_INIT = 1e-5
N_STEPS = 512


def check_a_range(a: jax.Array, a_init: float = A_INIT) -> jax.Array:
    """Require a_init <= a <= 1, up to exp(log(a)) roundoff at the ends."""
    tol = 8 * jnp.finfo(a.dtype).eps
    return eqx.error_if(
        a,
        jnp.any((a < a_init * (1 - tol)) | (a > 1.0 + tol)),
        f"ODE growth requires {a_init} <= a <= 1.",
    )


def linear_rhs(c: BackgroundCosmo, ln_a, y):
    """State y = [D, F],  F = dD/dln a. D and F may be arrays of modes.

    D'' + (2 + dlnH/dlna) D' - 3/2 Omega_m(a) D = 0, with ' = d/dln a.
    """
    D, F = y
    a = jnp.exp(ln_a)
    Om = c.Omega_m_of_a(a)
    drag = 2.0 + c.dlnH_dlna(a)
    return jnp.array([F, 1.5 * Om * D - drag * F])


def rk4_table(
    rhs: Callable[[jax.Array, jax.Array], jax.Array],
    y0: jax.Array,
    *,
    a_init: float = A_INIT,
    n_steps: int = N_STEPS,
) -> tuple[jax.Array, jax.Array]:
    """ln a grid (ascending, ending at 0) and the states on it, ys[i] at grid[i]."""
    grid = jnp.linspace(jnp.log(a_init), 0.0, n_steps + 1)
    h = grid[1] - grid[0]

    def step(y, t):
        k1 = rhs(t, y)
        k2 = rhs(t + h / 2, y + h / 2 * k1)
        k3 = rhs(t + h / 2, y + h / 2 * k2)
        k4 = rhs(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return y, y

    _, ys = jax.lax.scan(step, y0, grid[:-1])
    return grid, jnp.concatenate([y0[None, :], ys], axis=0)


def ode_table(
    background: BackgroundCosmo,
    *,
    a_init: float = A_INIT,
    n_steps: int = N_STEPS,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """ln a grid with D1 and F1 = dD1/dln a on it."""
    grid, ys = rk4_table(
        lambda t, y: linear_rhs(background, t, y),
        jnp.array([a_init, a_init]),  # EdS growing mode
        a_init=a_init,
        n_steps=n_steps,
    )
    return grid, ys[:, 0], ys[:, 1]


def hermite_interp(grid, values, slopes, x) -> tuple[jax.Array, jax.Array]:
    """Cubic Hermite interpolant through (values, slopes) at nodes, and its slope.

    grid is ascending (not necessarily uniform); x outside it extrapolates the end cubics.
    """
    n = grid.shape[0] - 1
    index = jnp.clip(jnp.searchsorted(grid, x, side="right") - 1, 0, n - 1)
    width = grid[index + 1] - grid[index]
    offset = x - grid[index]
    y0, y1 = values[index], values[index + 1]
    m0, m1 = slopes[index], slopes[index + 1]
    secant = (y1 - y0) / width
    quadratic = (3 * secant - 2 * m0 - m1) / width
    cubic = (m0 + m1 - 2 * secant) / width**2
    value = y0 + offset * (m0 + offset * (quadratic + offset * cubic))
    slope = m0 + offset * (2 * quadratic + 3 * offset * cubic)
    return value, slope
