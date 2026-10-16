"""Numerical helpers for Euclidean coordinates, with components first."""

from math import pi

import equinox as eqx
import jax.numpy as jnp


def real_array(value, name):
    """Convert real, finite input using JAX's default floating precision."""
    value = jnp.asarray(value)
    if jnp.iscomplexobj(value):
        raise ValueError(f"{name} must be real.")
    value = value.astype(float)
    return eqx.error_if(value, jnp.any(~jnp.isfinite(value)), f"{name} must be finite.")


def radius(x):
    return jnp.linalg.norm(x, axis=0)


def nonzero_radius(x):
    r = radius(x)
    return eqx.error_if(r, jnp.any(r == 0), "Direction is undefined at the origin.")


def unit_vectors(x):
    return x / nonzero_radius(x)


def colatitude(direction):
    return jnp.arctan2(jnp.hypot(direction[0], direction[1]), direction[2])


def longitude(direction):
    x, y, _ = direction
    # Longitude at either pole is conventionally zero.
    polar = (x == 0) & (y == 0)
    return jnp.mod(jnp.arctan2(y, jnp.where(polar, 1.0, x)), 2 * pi)


def spherical_data(radius, theta, phi):
    radius, theta, phi = jnp.broadcast_arrays(
        real_array(radius, "radius"),
        real_array(theta, "theta"),
        real_array(phi, "phi"),
    )
    if radius.ndim > 1:
        raise ValueError("radius, theta and phi must be scalars or 1D arrays.")
    radius, theta, phi = map(jnp.atleast_1d, (radius, theta, phi))
    radius = eqx.error_if(radius, jnp.any(radius < 0), "radius must be nonnegative.")
    theta = eqx.error_if(
        theta, jnp.any((theta < 0) | (theta > pi)), "theta must be in [0, pi]."
    )
    phi = jnp.mod(phi, 2 * pi)
    return jnp.stack((radius, theta, phi))


def cartesian(radius, theta, phi):
    sin_theta = jnp.sin(theta)
    return radius * jnp.stack(
        (sin_theta * jnp.cos(phi), sin_theta * jnp.sin(phi), jnp.cos(theta))
    )
