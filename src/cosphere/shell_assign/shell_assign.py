"""Deposit angular point samples as sums of weights per HEALPix pixel."""

from typing import Literal

import jax.numpy as jnp
from jax.typing import ArrayLike

from ..field import ShellField
from ..shell import Shell
from ..shell._stencil import real_finite, stencil


def shell_assign(
    shell: Shell,
    theta: ArrayLike,
    phi: ArrayLike,
    weights: ArrayLike | None = None,
    *,
    method: Literal["ngp", "bilinear"] = "ngp",
) -> ShellField:
    """Assign directions to a HEALPix RING shell, conserving total weight.

    theta and phi are equal-length 1D arrays in radians, with theta in [0, pi].
    weights is a real finite array of the same shape (signed weights allowed);
    None gives one count per point. The result contains sums per pixel, with
    no division by pixel area, mean density or total weight.

    ngp assigns the containing pixel. bilinear scatters the four interpolation
    weights from the bracketing rings, with equal sharing at each pole. It is
    continuous and piecewise differentiable in angles; derivatives at stencil
    boundaries and Cartesian pole coordinates need not exist. NGP has no
    useful positional gradient.

    Bilinear conserves particle weights but its effective pixel areas are not
    exactly equal: a uniform continuum sky need not give uniform pixel counts.
    It is an interpolation-based assignment, not integration of a finite cloud
    over pixel boundaries. Radial selection and weights belong to the caller.

    JIT-compatible with a static method; work/storage scale with the number
    of points and shell pixels, never their product. No optional dependency.
    """
    pixels, fractions = stencil(shell, theta, phi, method=method)
    shape = (pixels.shape[1],)
    if weights is None:
        weights = jnp.ones(shape)
    else:
        weights = real_finite(weights, "weights")
        if weights.shape != shape:
            raise ValueError(f"weights must have shape {shape}; got {weights.shape}.")

    data = (
        jnp.zeros(shell.SHAPE)
        .at[pixels.reshape(-1)]
        .add((fractions * weights).reshape(-1))
    )
    return ShellField(data, shell)
