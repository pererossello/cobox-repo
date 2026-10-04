"""Real spherical Bessel functions for radial projection kernels.

Recurrences and derivatives: https://dlmf.nist.gov/10.51
Recurrence stability: https://dlmf.nist.gov/10.74#iv
"""

from functools import partial
from math import ceil
from operator import index

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def _series(x: jax.Array, ell_max: int) -> jax.Array:
    """Power series for |x| <= 1; no division by x, including at the origin."""
    ell = jnp.arange(ell_max + 1, dtype=x.dtype).reshape((-1,) + (1,) * x.ndim)
    leading = jnp.concatenate(
        [jnp.ones_like(x)[None], jnp.cumprod(x / (2 * ell[1:] + 1), axis=0)]
    )

    def step(m, state):
        term, total = state
        term = -term * x**2 / (2 * m * (2 * ell + 2 * m + 1))
        return term, total + term

    # The first omitted correction is < 1 / 27! at |x| = 1, ell = 0.
    _, total = jax.lax.fori_loop(
        1, 13, step, (jnp.ones_like(leading), jnp.ones_like(leading))
    )
    return leading * total


def _j01(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Analytic anchors, evaluated only on safe arguments |x| >= 1."""
    j0 = jnp.sin(x) / x
    return j0, (j0 - jnp.cos(x)) / x


def _upward(x: jax.Array, ell_max: int) -> jax.Array:
    """Ascending recurrence, used only when x > ell_max and x >= 1."""
    j0, j1 = _j01(x)

    def step(pair, ell):
        previous, current = pair
        following = (2 * ell + 1) * current / x - previous
        return (current, following), following

    _, tail = jax.lax.scan(step, (j0, j1), jnp.arange(1, ell_max))
    return jnp.concatenate([j0[None], j1[None], tail])


def _downward(x: jax.Array, ell_max: int, n_start: int) -> jax.Array:
    """Scaled Miller recurrence; 1 <= x <= ell_max, n_start > ell_max.

    Rescale the pair at each step. Store each rescaling separately and restore
    the requested orders using cumulative products, avoiding both overflow and
    subtraction of large logarithms. Normalize against j0 AND j1 so that zeros
    of either anchor are harmless.
    """

    def step(pair, ell):
        current, following = pair
        previous = (2 * ell + 1) * current / x - following
        scale = jnp.maximum(1.0, jnp.maximum(jnp.abs(previous), jnp.abs(current)))
        return (previous / scale, current / scale), (current, 1 / scale)

    def warmup(i, pair):
        return step(pair, n_start - i)[0]

    pair = jax.lax.fori_loop(
        0, n_start - ell_max, warmup, (jnp.ones_like(x), jnp.zeros_like(x))
    )
    (u0, u1), (values, inverse_scales) = jax.lax.scan(
        step, pair, jnp.arange(ell_max, 0, -1)
    )
    tail = values[::-1] * jnp.cumprod(inverse_scales[::-1], axis=0)
    j0, j1 = _j01(x)
    norm = (u0 * j0 + u1 * j1) / (u0**2 + u1**2)
    result = jnp.concatenate([u0[None], tail]) * norm
    # Use the analytic anchors themselves for the two lowest orders.
    return result.at[0].set(j0).at[1].set(j1)


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _spherical_jn(x: jax.Array, ell_max: int) -> jax.Array:
    magnitude = jnp.abs(x)
    # Keep unselected branches finite too, including under jit/vmap.
    series = _series(jnp.where(magnitude < 1, x, 0.0), ell_max)
    if ell_max == 0:
        safe_x = jnp.maximum(magnitude, 1.0)
        return jnp.where(magnitude < 1, series, (jnp.sin(safe_x) / safe_x)[None])

    upward = _upward(jnp.maximum(magnitude, float(ell_max)), ell_max)
    # Extra orders grow with the width of the turning region (~ell^(1/3)).
    n_start = ell_max + ceil(32 + 8 * (ell_max + 1) ** (1 / 3))
    downward = _downward(jnp.clip(magnitude, 1.0, float(ell_max)), ell_max, n_start)
    values = jnp.where(magnitude > ell_max, upward, downward)
    ell = jnp.arange(ell_max + 1).reshape((-1,) + (1,) * x.ndim)
    parity = jnp.where((x < 0) & (ell % 2 == 1), -1.0, 1.0)
    return jnp.where(magnitude < 1, series, parity * values)


@_spherical_jn.defjvp
def _spherical_jn_jvp(ell_max, primals, tangents):
    (x,), (x_dot,) = primals, tangents
    extended = _spherical_jn(x, ell_max + 1)
    ell = jnp.arange(ell_max + 1, dtype=x.dtype).reshape((-1,) + (1,) * x.ndim)
    previous = jnp.concatenate(
        [jnp.zeros_like(x)[None], extended[:ell_max]]  # type:ignore
    )
    # Equivalent to j_(ell-1) - (ell+1) j_ell/x, but regular at x = 0.
    derivative = (ell * previous - (ell + 1) * extended[1:]) / (  # type:ignore
        2 * ell + 1
    )
    return _spherical_jn(x, ell_max), derivative * x_dot


def spherical_jn(x: ArrayLike, ell_max: int) -> jax.Array:
    """Return j_ell(x) for every integer 0 <= ell <= ell_max.

    Parameters
    ----------
    x : array_like
        Finite real arguments, with any shape. Negative arguments use parity.
        Floating inputs retain their precision, with a minimum of float32.
    ell_max : int
        Nonnegative maximum order, static under JAX transformations.

    Returns
    -------
    jax.Array
        Shape ``(ell_max + 1, *x.shape)``; the first axis is the order. JIT,
        vmap, and differentiation in x are supported, including at x = 0.

    Notes
    -----
    Uses a series below |x| = 1, upward recurrence for |x| > ell_max, and
    scaled Miller recurrence otherwise. The starting-order margin is chosen
    conservatively and checked against SciPy through ell_max = 2048. Values
    below the dtype's normal range may underflow. Float64 is recommended for
    accurate projection integrals; enable it with JAX's usual configuration.
    Complex arguments are not supported. This API returns all orders and uses
    argument order ``(x, ell_max)``, unlike SciPy's ``(n, z)``.
    """
    if isinstance(ell_max, bool):
        raise TypeError("ell_max must be a nonnegative integer.")
    ell_max = index(ell_max)
    if ell_max < 0:
        raise ValueError("ell_max must be nonnegative.")
    x = jnp.asarray(x)
    if jnp.issubdtype(x.dtype, jnp.complexfloating):
        raise TypeError("spherical_jn supports real arguments only.")
    x = x.astype(jnp.result_type(x.dtype, jnp.float32))
    return _spherical_jn(x, ell_max)
