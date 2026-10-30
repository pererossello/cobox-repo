"""Interpolation of shell pixel-centre values at angular directions."""

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ...shell._stencil import StencilMethod, stencil

if TYPE_CHECKING:
    from ..shell_field import ShellField


def interpolate_at(
    field: "ShellField",
    theta: ArrayLike,
    phi: ArrayLike,
    *,
    method: StencilMethod = "bilinear",
) -> jax.Array:
    """Gather pixel values using the same geometry as shell assignment."""
    pixels, fractions = stencil(field.shell, theta, phi, method=method)
    field = field.isht()
    return jnp.sum(field.data[pixels] * fractions, axis=0)
