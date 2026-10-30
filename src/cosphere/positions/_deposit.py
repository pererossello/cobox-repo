"""Put values at angular positions onto HEALPix pixels."""

from math import log
from typing import TYPE_CHECKING, Literal, get_args

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ..field import ShellField
from ..shell import Shell
from ..shell._discs import disc_pixels, ring_span
from ..shell._stencil import StencilMethod, real_finite, stencil

if TYPE_CHECKING:
    from .angular import AngularPositions

Reduce = Literal["sum", "mean"]


def deposit(
    points: "AngularPositions",
    shell: Shell,
    values: ArrayLike | None,
    method: StencilMethod,
    reduce: Reduce,
) -> ShellField:
    _check_reduce(reduce)
    v = _values(points, values)
    pixels, fractions = stencil(shell, points.theta, points.phi, method=method)  # (K, N)
    total = _scatter(shell, pixels, fractions * v)
    weight = _scatter(shell, pixels, fractions) if reduce == "mean" else None
    return _finish(shell, total, weight)


def deposit_gaussian(
    points: "AngularPositions",
    shell: Shell,
    sigma: ArrayLike,
    values: ArrayLike | None,
    truncate: float,
    reduce: Reduce,
) -> ShellField:
    _check_reduce(reduce)
    v = _values(points, values)
    s = np.broadcast_to(np.asarray(sigma, dtype=float), (len(points),))
    if not np.all(np.isfinite(s) & (s > 0)):
        raise ValueError("sigma must be finite and positive.")
    if not truncate > 0:
        raise ValueError(f"truncate must be positive; got {truncate!r}.")

    total = jnp.zeros(shell.SHAPE)
    weight = jnp.zeros(shell.SHAPE) if reduce == "mean" else None
    for index in _chunks(shell.nside, s, truncate):
        chunk = type(points)(points.theta[index], points.phi[index])
        pixels, w = _gaussian_weights(shell, chunk, jnp.asarray(s[index]), truncate)
        total = total + _scatter(shell, pixels, w * v[index][:, None])
        if weight is not None:
            weight = weight + _scatter(shell, pixels, w)
    return _finish(shell, total, weight)


def _gaussian_weights(
    shell: Shell, points: "AngularPositions", sigma: jax.Array, truncate: float
) -> tuple[jax.Array, jax.Array]:
    """Pixels and weights per point, shape (n, M), weights summing to 1 per point.

    Pixels within truncate * sigma get exp(-d^2 / 2 sigma^2). The containing pixel
    is an extra entry, weighted only if it lies beyond the cut; if every weight
    underflows it takes all of it, so sigma -> 0 is the ngp deposit.
    """
    n = len(points)
    radius = truncate * sigma
    pixels, valid = disc_pixels(shell.nside, points.theta, points.phi, radius)
    pixels, valid = pixels.reshape(n, -1), valid.reshape(n, -1)
    centre = shell.ang2pix(points.theta, points.phi)
    pixels = jnp.concatenate([pixels, centre[:, None]], axis=1)

    u = points.unit_vectors  # (3, n)
    d = _angle(shell.pix2vec(pixels), u[:, :, None])  # (n, M + 1)
    inside = jnp.concatenate([valid, (d[:, -1] > radius)[:, None]], axis=1)
    w = jnp.where(inside, jnp.exp(-0.5 * (d / sigma[:, None]) ** 2), 0.0)
    norm = jnp.sum(w, axis=1, keepdims=True)
    ngp = jnp.zeros_like(w).at[:, -1].set(1.0)
    return pixels, jnp.where(norm > 0, w / jnp.where(norm > 0, norm, 1), ngp)


def _angle(a: jax.Array, b: jax.Array) -> jax.Array:
    """Angle between unit vectors along axis 0, accurate at small angles."""
    chord = jnp.sqrt(jnp.sum((a - b) ** 2, axis=0))
    return 2 * jnp.arcsin(jnp.clip(chord / 2, 0, 1))


def _chunks(
    nside: int, sigma: np.ndarray, truncate: float, ratio: float = 1.5, budget: int = 2**22
) -> list[np.ndarray]:
    """Point indices grouped by sigma, in log bins of width ratio, then split so
    each group's (points, rings, run length) table has about budget entries.

    Binning only sizes the tables: every point keeps its own sigma, so a few
    large-sigma points do not inflate the tables of all the others.
    """
    if len(sigma) == 0:
        return []
    bins = np.floor(np.log(sigma / sigma.min()) / log(ratio)).astype(int)
    groups = []
    for b in np.unique(bins):
        index = np.flatnonzero(bins == b)
        span = ring_span(nside, truncate * float(sigma[index].max()))
        per_point = 2 * span**2  # rings x run length; runs near a pole reach ~2x span
        size = max(1, budget // per_point)
        groups += [index[i : i + size] for i in range(0, len(index), size)]
    return groups


def _values(points: "AngularPositions", values: ArrayLike | None) -> jax.Array:
    if values is None:
        return jnp.ones(len(points))
    v = real_finite(values, "values")
    if v.shape != (len(points),):
        raise ValueError(f"values must have shape ({len(points)},); got {v.shape}.")
    return v


def _check_reduce(reduce: str) -> None:
    if reduce not in get_args(Reduce):
        raise ValueError(f"reduce must be one of {get_args(Reduce)}; got {reduce!r}.")


def _scatter(shell: Shell, pixels: jax.Array, weights: jax.Array) -> jax.Array:
    """Add weights into their pixels; pixels and weights have the same shape."""
    return jnp.zeros(shell.SHAPE).at[pixels.reshape(-1)].add(weights.reshape(-1))


def _finish(shell: Shell, total: jax.Array, weight: jax.Array | None) -> ShellField:
    """Sum map, or total / weight per pixel (NaN where empty) when weight is given."""
    if weight is not None:
        total = jnp.where(weight > 0, total / jnp.where(weight > 0, weight, 1), jnp.nan)
    return ShellField(total, shell)
