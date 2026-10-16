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


def profile(kind, scale, outer_radius, theta, ell_max):
    if kind == "gaussian":
        return gaussian_profile(theta, scale, ell_max)
    disc = jnp.where(theta <= scale, 1 / cap_area(scale), 0.0)
    if kind == "disc":
        return disc
    area = (
        4
        * pi
        * jnp.sin((outer_radius + scale) / 2)
        * jnp.sin((outer_radius - scale) / 2)
    )
    annulus = jnp.where((theta > scale) & (theta <= outer_radius), 1 / area, 0.0)
    return annulus if kind == "annulus" else disc - annulus
