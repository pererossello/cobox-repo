"""Angular power from projection kernels of one common source field."""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from ..spectrum import AngularSpectrum
from .kernel import ProjectionKernel


def angular_power(
    power_fn: Callable[[jax.Array], jax.Array],
    kernel: ProjectionKernel,
    other: ProjectionKernel | None = None,
) -> AngularSpectrum:
    """C_ell = 4 pi integral dln k Delta^2(k) Delta^i_ell(k) Delta^j_ell(k).

    Delta^2 = k^3 P0(k) / (2 pi^2), so this equals
    (2/pi) integral dk k^2 P0(k) Delta^i_ell(k) Delta^j_ell(k).
    other=None gives the auto-spectrum of kernel. Both kernels project the
    SAME source field, whose auto-power is power_fn(k) = P0(k): real, finite,
    nonnegative, scalar or of shape k.shape. The kernels' amplitudes must use
    the convention of that source (e.g. primordial curvature or matter today).

    Trapezoid rule in ln k on the kernels' common grid, which must be positive
    and strictly increasing; there are no tails beyond its ends. The kernels
    oscillate in k, so the grid controls the accuracy of C_ell: check that
    refining it and extending its ends leaves the result unchanged.
    """
    other = kernel if other is None else other
    values = kernel._matching(other)
    k = kernel.k
    if k.size < 2:
        raise ValueError("angular_power needs a k grid with at least two points.")
    k = eqx.error_if(
        k,
        jnp.any(k <= 0) | jnp.any(jnp.diff(k) <= 0),
        "k must be positive and strictly increasing to integrate in ln k.",
    )
    power = jnp.asarray(power_fn(k))
    if jnp.iscomplexobj(power):
        raise TypeError("power_fn must return the real auto-power P0(k).")
    power = jnp.broadcast_to(power, k.shape)
    power = eqx.error_if(
        power,
        jnp.any(~jnp.isfinite(power) | (power < 0)),
        "P0(k) must be finite and nonnegative.",
    )
    integrand = (2 / jnp.pi) * k**3 * power * values * other.values
    return AngularSpectrum(jnp.trapezoid(integrand, jnp.log(k), axis=-1))
