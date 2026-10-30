"""Profiles per steradian; Gaussian denotes the spherical heat kernel."""

from math import pi

import jax
import jax.numpy as jnp


def cap_area(radius):
    return 4 * pi * jnp.sin(radius / 2) ** 2


def gaussian_profile(theta, sigma, ell_max):
    # Inverse Legendre series of exp[-l(l+1) sigma^2/2], explicitly truncated.
    t = 2 * jnp.sin(theta / 2) ** 2
    one = jnp.ones_like(theta)

    def step(ell, state):
        previous, difference, total = state
        next_difference = ((ell - 1) * difference - (2 * ell - 1) * t * previous) / ell
        current = previous + next_difference
        weight = (2 * ell + 1) * jnp.exp(-0.5 * ell * (ell + 1) * sigma**2)
        return current, next_difference, total + weight * current

    return jax.lax.fori_loop(1, ell_max + 1, step, (one, jnp.zeros_like(theta), one))[
        -1
    ] / (4 * pi)


def compact_regions(
    kind: str, scale: jax.Array, outer_radius: jax.Array | None
) -> tuple[tuple[jax.Array | None, jax.Array, int], ...]:
    """(inner, outer, coefficient) regions; None means a closed central disc."""
    disc = (None, scale, 1)
    if kind == "disc":
        return (disc,)
    if outer_radius is None:
        raise ValueError(f"{kind} windows need an outer_radius.")
    annulus = (scale, outer_radius, 1)
    if kind == "annulus":
        return (annulus,)
    if kind == "aperture":
        return (disc, (scale, outer_radius, -1))
    raise ValueError("Only disc, annulus and aperture have compact regions.")


def profile(kind, scale, outer_radius, theta, ell_max):
    if kind == "gaussian":
        return gaussian_profile(theta, scale, ell_max)
    result = jnp.zeros_like(theta)
    for inner, outer, coefficient in compact_regions(kind, scale, outer_radius):
        if inner is None:
            inside = theta <= outer
            area = cap_area(outer)
        else:
            inside = (theta > inner) & (theta <= outer)
            area = 4 * pi * jnp.sin((outer + inner) / 2) * jnp.sin((outer - inner) / 2)
        result = result + jnp.where(inside, coefficient / area, 0.0)
    return result
