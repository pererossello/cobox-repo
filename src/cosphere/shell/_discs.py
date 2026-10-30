"""Pixel runs of discs on a RING map, their pixels, and map sums over them.

Run arrays have shape (N, K): directions, candidate rings.

On ring i the pixel centres within angular radius R of (theta0, phi0) are one
cyclic run of consecutive pixels, |phi - phi0| <= dphi with
cos(dphi) = (cos R - z z0) / (sin(theta) sin(theta0)). This is the closed
pixel-centre membership of healpy.query_disc(inclusive=False), without healpy.
"""

from math import ceil, pi

import jax
import jax.numpy as jnp

from . import _pixels


def ring_span(nside: int, radius: float) -> int:
    """Candidate rings per disc: d(ring)/d(theta) <= 1.5 nside, plus margins."""
    return min(ceil(3 * nside * radius) + 3, 4 * nside - 1)


def _ring_coordinate(nside: int, theta: jax.Array) -> jax.Array:
    """Continuous ring number of colatitude theta (ring i sits at i), increasing."""
    z = jnp.cos(theta)
    cap = nside * jnp.sqrt(3 * (1 - jnp.abs(z)))
    ring = jnp.where(jnp.abs(z) > 2 / 3, cap, nside * (2 - 1.5 * z))
    return jnp.where(z < -2 / 3, 4 * nside - cap, ring)


def disc_runs(
    nside: int, theta: jax.Array, phi: jax.Array, radius: float | jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Ring first pixel, ring size, run offset in [0, size) and run length per (direction, ring).

    radius is one value for all discs or one per disc, shape (N,); the number of
    candidate rings is set by the largest. A run may wrap past the ring's last
    pixel back to its first. Rings a disc misses have length 0.
    """
    n_ring = 4 * nside - 1
    largest = float(radius) if jnp.ndim(radius) == 0 else float(jnp.max(radius))
    radius = jnp.broadcast_to(radius, theta.shape)
    first = jnp.floor(_ring_coordinate(nside, jnp.clip(theta - radius, 0, pi)))
    ring = jnp.maximum(first.astype(int) - 1, 1)[:, None] + jnp.arange(
        ring_span(nside, largest)
    )
    exists = ring <= n_ring
    z, sin_theta, size, start, phi0 = _pixels.ring_geometry(
        nside, jnp.minimum(ring, n_ring)
    )

    # At a pole only the sign of sin(theta0) ~ 0 decides whole-ring-in vs ring-out.
    # float32(pi) exceeds pi, so sin would be negative and flip every ring; a
    # positive floor keeps the sign right and avoids 0/0 on a ring at the edge.
    sin_theta0 = jnp.maximum(jnp.sin(theta), jnp.finfo(theta.dtype).tiny)
    x = (jnp.cos(radius)[:, None] - z * jnp.cos(theta)[:, None]) / (
        sin_theta * sin_theta0[:, None]
    )
    dphi = jnp.arccos(jnp.clip(x, -1, 1))
    width = 2 * pi / size
    relative = phi[:, None] - phi0
    offset = jnp.ceil((relative - dphi) / width).astype(int)
    length = jnp.floor((relative + dphi) / width).astype(int) - offset + 1
    length = jnp.where((x <= -1) | (length >= size), size, jnp.maximum(length, 0))
    length = jnp.where((x > 1) | ~exists, 0, length)
    return start, size, offset % size, length


def disc_pixels(
    nside: int, theta: jax.Array, phi: jax.Array, radius: float | jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Pixels of each disc and which entries are real, both shape (N, K, W).

    W is the longest run, so the table size depends on the data (host-side).
    Padding entries point at pixel 0 and are False in the second array.
    """
    start, size, offset, length = disc_runs(nside, theta, phi, radius)
    step = jnp.arange(max(int(jnp.max(length, initial=0)), 1))
    valid = step < length[..., None]
    pixels = start[..., None] + (offset[..., None] + step) % size[..., None]
    return jnp.where(valid, pixels, 0), valid


def _before(cumsum: jax.Array, k: jax.Array) -> jax.Array:
    """Sum of values[:k], from cumsum = jnp.cumsum(values) (no leading zero to copy)."""
    return jnp.where(k > 0, cumsum[k - 1], 0)


def run_sums(
    cumsum: jax.Array,
    start: jax.Array,
    size: jax.Array,
    offset: jax.Array,
    length: jax.Array,
) -> jax.Array:
    """Sum of a map over each direction's runs from disc_runs, shape (N,).

    cumsum is jnp.cumsum of the map in RING order. A run that wraps past the
    end of its ring adds the whole ring and subtracts the gap.
    """
    end = offset + length
    wraps = end > size
    ring_total = _before(cumsum, start + size) - _before(cumsum, start)
    runs = (
        _before(cumsum, start + end - wraps * size)
        - _before(cumsum, start + offset)
        + wraps * ring_total
    )
    return runs.sum(axis=1)
