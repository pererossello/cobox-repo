"""HEALPix RING-scheme geometry in closed form (Gorski et al. 2005).

Rings are counted i = 1 .. 4 nside - 1 from the north pole. Polar-cap ring i
(i < nside, mirrored in the south) has 4 i pixels; equatorial rings
(nside <= i <= 3 nside) have 4 nside. Pixels are numbered ring by ring,
west to east, from 0 at the north pole.
"""

from __future__ import annotations

from math import pi

import jax
import jax.numpy as jnp


def _isqrt(x: jax.Array) -> jax.Array:
    """floor(sqrt(x)) for non-negative integers, exact."""
    r = jnp.floor(jnp.sqrt(x.astype(float))).astype(x.dtype)
    r = jnp.where(r * r > x, r - 1, r)
    return jnp.where((r + 1) * (r + 1) <= x, r + 1, r)


def pix2ang(nside: int, pix: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Pixel-centre (theta, phi) in radians for RING pixel indices."""
    pix = jnp.asarray(pix)
    npix = 12 * nside**2
    ncap = 2 * nside * (nside - 1)
    fact2 = 4.0 / npix

    # north cap
    ring_n = (1 + _isqrt(1 + 2 * pix)) // 2
    iphi_n = pix + 1 - 2 * ring_n * (ring_n - 1)
    tmp_n = ring_n**2 * fact2  # 1 - z
    phi_n = (iphi_n - 0.5) * (0.5 * pi) / jnp.maximum(ring_n, 1)

    # equatorial belt
    ip = pix - ncap
    ring_e = ip // (4 * nside) + nside
    iphi_e = ip % (4 * nside) + 1
    fodd = jnp.where((ring_e + nside) % 2 == 1, 1.0, 0.5)
    z_e = (2 * nside - ring_e) * (2 * nside * fact2)
    phi_e = (iphi_e - fodd) * pi / (2 * nside)

    # south cap
    ip_s = npix - pix
    ring_s = (1 + _isqrt(jnp.maximum(2 * ip_s - 1, 0))) // 2
    iphi_s = 4 * ring_s + 1 - (ip_s - 2 * ring_s * (ring_s - 1))
    tmp_s = ring_s**2 * fact2  # 1 + z
    phi_s = (iphi_s - 0.5) * (0.5 * pi) / jnp.maximum(ring_s, 1)

    north = pix < ncap
    south = pix >= npix - ncap
    z = jnp.where(north, 1.0 - tmp_n, jnp.where(south, tmp_s - 1.0, z_e))
    one_minus_abs_z = jnp.where(north, tmp_n, jnp.where(south, tmp_s, 1.0 - jnp.abs(z_e)))
    phi = jnp.where(north, phi_n, jnp.where(south, phi_s, phi_e))

    # sin(theta) from 1 - |z| keeps theta accurate near the poles
    sin_theta = jnp.sqrt(one_minus_abs_z * (2.0 - one_minus_abs_z))
    return jnp.arctan2(sin_theta, z), phi


def zphi2pix(
    nside: int, z: jax.Array, sin_theta: jax.Array, phi: jax.Array
) -> jax.Array:
    """RING pixel containing the direction with cos(theta) = z, sin(theta), and phi."""
    npix = 12 * nside**2
    ncap = 2 * nside * (nside - 1)
    za = jnp.abs(z)
    tt = jnp.mod(phi, 2 * pi) / (0.5 * pi)  # in [0, 4)

    # equatorial belt, |z| <= 2/3: count the edge lines crossed
    temp1 = nside * (0.5 + tt)
    temp2 = nside * z * 0.75
    jp = jnp.floor(temp1 - temp2).astype(int)  # ascending edge line
    jm = jnp.floor(temp1 + temp2).astype(int)  # descending edge line
    ir = nside + 1 + jp - jm  # ring counted from z = 2/3, in 1 .. 2 nside + 1
    kshift = 1 - (ir % 2)
    ip = jnp.mod((jp + jm - nside + kshift + 1) // 2, 4 * nside)
    pix_e = ncap + (ir - 1) * 4 * nside + ip

    # polar caps: sqrt(3 (1 - |z|)) written with sin(theta), accurate near the poles
    tp = tt - jnp.floor(tt)
    tmp = nside * sin_theta / jnp.sqrt((1.0 + za) / 3.0)
    jp = jnp.floor(tp * tmp).astype(int)
    jm = jnp.floor((1.0 - tp) * tmp).astype(int)
    ir = jp + jm + 1  # ring counted from the nearer pole
    ip = jnp.mod(jnp.floor(tt * ir).astype(int), 4 * ir)
    pix_p = jnp.where(z > 0, 2 * ir * (ir - 1) + ip, npix - 2 * ir * (ir + 1) + ip)

    return jnp.where(za <= 2.0 / 3.0, pix_e, pix_p)


def rings(nside: int) -> dict[str, jax.Array]:
    """Per-ring arrays (length 4 nside - 1): theta, npix, start pixel, phi of the first pixel."""
    npix = 12 * nside**2
    ncap = 2 * nside * (nside - 1)
    i = jnp.arange(1, 4 * nside)
    i_cap = jnp.minimum(i, 4 * nside - i)  # ring number counted from the nearer pole

    north, south = i < nside, i > 3 * nside
    ring_npix = 4 * jnp.minimum(i_cap, nside)
    start = jnp.where(
        north,
        2 * i * (i - 1),
        jnp.where(south, npix - 2 * i_cap * (i_cap + 1), ncap + (i - nside) * 4 * nside),
    )
    theta, phi0 = pix2ang(nside, start)
    return {"theta": theta, "npix": ring_npix, "start": start, "phi0": phi0}
