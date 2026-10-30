"""HEALPix RING interpolation and assignment stencils; arrays have shape (neighbours, N).

Bilinear weights interpolate in longitude on the two bracketing rings and
then in colatitude between them. Beyond the first/last ring, blend towards
equal weights on the four polar-ring pixels. No full-sky neighbour search
or external HEALPix dependency is needed.
"""

from math import pi
from typing import TYPE_CHECKING, Literal, get_args

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from . import _pixels

if TYPE_CHECKING:
    from .shell import Shell

StencilMethod = Literal["ngp", "bilinear"]


def _ring_pair(
    rings: dict[str, jax.Array], ring: jax.Array, phi: jax.Array
) -> tuple[jax.Array, jax.Array]:
    size = rings["npix"][ring]
    coordinate = (phi - rings["phi0"][ring]) * size / (2 * pi)
    left = jnp.floor(coordinate).astype(int)
    fraction = coordinate - left
    pixels = rings["start"][ring] + jnp.stack((left % size, (left + 1) % size))
    return pixels, jnp.stack((1 - fraction, fraction))


def bilinear(
    nside: int, theta: jax.Array, phi: jax.Array
) -> tuple[jax.Array, jax.Array]:
    rings = _pixels.rings(nside)
    angles = rings["theta"]
    above = jnp.searchsorted(angles, theta, side="right") - 1
    upper = jnp.clip(above, 0, len(angles) - 2)
    lower = upper + 1
    p_up, w_up = _ring_pair(rings, upper, phi)
    p_lo, w_lo = _ring_pair(rings, lower, phi)
    fraction = jnp.clip((theta - angles[upper]) / (angles[lower] - angles[upper]), 0, 1)
    pixels = jnp.concatenate((p_up, p_lo), axis=0)
    weights = jnp.concatenate(((1 - fraction) * w_up, fraction * w_lo), axis=0)

    # The polar limit must be independent of longitude. The nearest ring has
    # four pixels: two bracketing phi plus their two opposite pixels.
    north, south = theta < angles[0], theta > angles[-1]
    polar_ring = jnp.where(north, 0, len(angles) - 1)
    pair, pair_weights = _ring_pair(rings, polar_ring, phi)
    start = rings["start"][polar_ring]
    opposite = start + (pair - start + 2) % 4
    polar_pixels = jnp.concatenate((pair, opposite), axis=0)
    distance = jnp.where(north, theta, pi - theta)
    extent = jnp.where(north, angles[0], pi - angles[-1])
    towards_ring = jnp.clip(distance / extent, 0, 1)
    polar_weights = (1 - towards_ring) / 4 + towards_ring * jnp.concatenate(
        (pair_weights, jnp.zeros_like(pair_weights)), axis=0
    )
    polar = north | south
    return jnp.where(polar, polar_pixels, pixels), jnp.where(
        polar, polar_weights, weights
    )


def real_finite(value: ArrayLike, name: str) -> jax.Array:
    value = jnp.asarray(value)
    if jnp.iscomplexobj(value):
        raise ValueError(f"{name} must be real.")
    value = value.astype(float)
    return eqx.error_if(value, jnp.any(~jnp.isfinite(value)), f"{name} must be finite.")


def stencil(
    shell: "Shell", theta: ArrayLike, phi: ArrayLike, *, method: StencilMethod
) -> tuple[jax.Array, jax.Array]:
    """Validate angular queries and return shared (neighbours, N) geometry."""
    if method not in get_args(StencilMethod):
        raise ValueError(
            f"method must be one of {get_args(StencilMethod)}; got {method!r}."
        )
    t, p = real_finite(theta, "theta"), real_finite(phi, "phi")
    if t.ndim != 1 or p.shape != t.shape:
        raise ValueError("theta and phi must be 1D arrays of equal length.")
    t = eqx.error_if(t, jnp.any((t < 0) | (t > pi)), "theta must be in [0, pi].")
    p = jnp.mod(p, 2 * pi)
    if method == "ngp":
        return shell.ang2pix(t, p)[None, :], jnp.ones_like(t)[None, :]
    return bilinear(shell.nside, t, p)
