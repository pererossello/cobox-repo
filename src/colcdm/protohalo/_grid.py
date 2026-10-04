"""Periodic windowed reductions on node grids.

Fixed shapes and a traced window width: each is compiled once per grid
shape, whatever the radius.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax


def _lexmax(a, b):
    """Lexicographic max of (key, tag) pairs a and b."""
    wins = (b[0] > a[0]) | ((b[0] == a[0]) & (b[1] > a[1]))
    return jnp.where(wins, b[0], a[0]), jnp.where(wins, b[1], a[1])


def _roll(pair, shift, axis):
    return tuple(jnp.roll(x, shift, axis) for x in pair)


@partial(jax.jit, static_argnames="axis")
def _window_lexmax(
    key: jax.Array, tag: jax.Array, half_width: int, axis: int
) -> tuple[jax.Array, jax.Array]:
    """Lexicographic max of (key, tag) over shifts |s| <= half_width along axis.

    Doubling: after j steps, run(i) is the max over [i, i + 2^j); two runs
    then cover the window [i - w, i + w].
    """
    width = 2 * half_width + 1
    steps = jnp.floor(jnp.log2(width)).astype(int)
    span = 2**steps

    def double(j, run):
        return _lexmax(run, _roll(run, -(2**j), axis))

    run = lax.fori_loop(0, steps, double, (key, tag))
    return _lexmax(
        _roll(run, half_width, axis), _roll(run, span - 1 - half_width, axis)
    )


@partial(jax.jit, static_argnames="axis")
def _window_min_parabola(
    f: jax.Array, half_width: int, spacing: float, axis: int
) -> jax.Array:
    """min over shifts |s| <= half_width of f(x - s) + (s spacing)^2 along axis."""

    def body(s, m):
        shifted = jnp.minimum(jnp.roll(f, s, axis), jnp.roll(f, -s, axis))
        return jnp.minimum(m, shifted + (s * spacing) ** 2)

    return lax.fori_loop(1, half_width + 1, body, f)


def cube_argmax(key: jax.Array, tag: jax.Array, half_width: int) -> jax.Array:
    """tag of the lexicographic max of (key, tag) in the periodic cube of
    half-width half_width (nodes) around each node. Unique tags make it exact."""
    for axis in range(key.ndim):
        key, tag = _window_lexmax(key, tag, half_width, axis)
    return tag


def distance_to(mask: jax.Array, half_width: int, spacing: float) -> jax.Array:
    """Periodic Euclidean distance from each node to the nearest True node.

    Separable lower envelope of parabolas, one axis at a time. Exact for
    distances below (half_width + 1) * spacing; never below the exact value.
    """
    f = jnp.where(mask, 0.0, jnp.inf)
    for axis in range(mask.ndim):
        f = _window_min_parabola(f, half_width, spacing, axis)
    return jnp.sqrt(f)
