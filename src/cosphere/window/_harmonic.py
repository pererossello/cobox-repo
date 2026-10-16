"""Axisymmetric harmonic responses using stable Legendre recurrences."""

import jax
import jax.numpy as jnp


def _band_average(midpoint, half_width, ell):
    # Average P_l(x) on [x_mid-h, x_mid+h], x_mid = 1-midpoint.
    # Divided differences avoid subtracting two nearly equal cap integrals.
    # Evolve P_n-P_(n-1) explicitly to retain small-angle accuracy.
    def step(n, state):
        mean, difference, derivative_difference, derivative, result = state
        ratio = n / (n + 1)
        factor = (2 * n + 1) / (n + 1)
        next_difference = ratio * difference - factor * (
            midpoint * mean - half_width**2 * derivative
        )
        next_derivative_difference = ratio * derivative_difference + factor * (
            mean - midpoint * derivative
        )
        next_derivative = derivative + next_derivative_difference
        average = (next_derivative_difference + derivative_difference) / (2 * n + 1)
        return (
            mean + next_difference,
            next_difference,
            next_derivative_difference,
            next_derivative,
            jnp.where(ell == n, average, result),
        )

    one = jnp.ones_like(midpoint)
    initial = (
        one - midpoint,
        -midpoint,
        one,
        one,
        jnp.ones_like(ell, dtype=float),
    )
    maximum = jnp.max(ell, initial=0)
    return jax.lax.fori_loop(1, maximum + 1, step, initial)[-1]


@jax.custom_jvp
def band_average(midpoint, half_width, ell):
    return _band_average(midpoint, half_width, ell)


@band_average.defjvp
def _band_average_jvp(primals, tangents):
    midpoint, half_width, ell = primals
    dmid, dwidth, _ = tangents

    # Compute partial derivatives independently of the incoming tangent. This
    # permits reverse-mode differentiation despite the dynamic multipole bound.
    def evaluate(mid, width):
        return _band_average(mid, width, ell)

    one, zero = jnp.ones_like(midpoint), jnp.zeros_like(midpoint)
    value, partial_mid = jax.jvp(evaluate, (midpoint, half_width), (one, zero))
    _, partial_width = jax.jvp(evaluate, (midpoint, half_width), (zero, one))
    return value, partial_mid * dmid + partial_width * dwidth


def cap_response(radius, ell):
    half_width = jnp.sin(radius / 2) ** 2
    return band_average(half_width, half_width, ell)


def annulus_response(inner, outer, ell):
    midpoint = jnp.sin(inner / 2) ** 2 + jnp.sin(outer / 2) ** 2
    half_width = jnp.sin((outer + inner) / 2) * jnp.sin((outer - inner) / 2)
    return band_average(midpoint, half_width, ell)


def response(kind, scale, outer_radius, ell):
    if kind == "gaussian":
        return jnp.exp(-0.5 * ell * (ell + 1) * scale**2)
    if kind == "disc":
        return cap_response(scale, ell)
    annulus = annulus_response(scale, outer_radius, ell)
    if kind == "annulus":
        return annulus
    return cap_response(scale, ell) - annulus
