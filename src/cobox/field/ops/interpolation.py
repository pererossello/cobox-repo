from __future__ import annotations
from typing import Literal, TYPE_CHECKING

import jax
import jax.numpy as jnp

from .nufft import evaluate_spectral

if TYPE_CHECKING:
    from ..scalar import ScalarField
    from ..vector import VectorField

InterpolationMethodLiteral = Literal["linear", "spectral"]

# -------------------------
# --- LINEAR PRIMITIVES ---
# -------------------------


def interp_linear_1d_uniform(
    values: jax.Array,
    x_lo: float,
    dx: float,
    x_query: jax.Array,
) -> jax.Array:
    """Linear interpolation on a uniform 1D grid, clamped at the boundaries."""
    N = values.shape[0]
    u = (x_query - x_lo) / dx
    i0 = jnp.clip(jnp.floor(u).astype(int), 0, N - 2)
    frac = jnp.clip(u - i0, 0.0, 1.0)
    return (1.0 - frac) * values[i0] + frac * values[i0 + 1]


def interp_linear_1d_periodic(
    values: jax.Array,
    scaled_pos: jax.Array,
) -> jax.Array:
    """Linear interpolation on a periodic 1D grid, in index units."""
    N = values.shape[0]
    i0 = jnp.floor(scaled_pos).astype(int)
    frac = scaled_pos - i0
    i0_wrapped = i0 % N
    i1_wrapped = (i0 + 1) % N
    return (1.0 - frac) * values[i0_wrapped] + frac * values[i1_wrapped]


def interp_linear_2d_periodic(
    values: jax.Array,
    scaled_pos: jax.Array,
) -> jax.Array:
    """Bilinear interpolation on a periodic 2D grid, in index units."""
    values = jnp.asarray(values)
    scaled_pos = jnp.asarray(scaled_pos)
    Nx, Ny = values.shape
    x, y = scaled_pos[0], scaled_pos[1]
    i = jnp.floor(x).astype(int)
    j = jnp.floor(y).astype(int)
    fx = x - i
    fy = y - j
    i0, i1 = i % Nx, (i + 1) % Nx
    j0, j1 = j % Ny, (j + 1) % Ny
    v00 = values[i0, j0]
    v10 = values[i1, j0]
    v01 = values[i0, j1]
    v11 = values[i1, j1]
    return (
        (1.0 - fx) * (1.0 - fy) * v00
        + fx * (1.0 - fy) * v10
        + (1.0 - fx) * fy * v01
        + fx * fy * v11
    )


def interp_linear_3d_periodic(
    values: jax.Array,
    scaled_pos: jax.Array,
) -> jax.Array:
    """Trilinear interpolation on a periodic 3D grid, in index units."""
    values = jnp.asarray(values)
    scaled_pos = jnp.asarray(scaled_pos)
    Nx, Ny, Nz = values.shape
    x, y, z = scaled_pos[0], scaled_pos[1], scaled_pos[2]
    i = jnp.floor(x).astype(int)
    j = jnp.floor(y).astype(int)
    k = jnp.floor(z).astype(int)
    fx = x - i
    fy = y - j
    fz = z - k
    i0, i1 = i % Nx, (i + 1) % Nx
    j0, j1 = j % Ny, (j + 1) % Ny
    k0, k1 = k % Nz, (k + 1) % Nz
    c00 = values[i0, j0, k0] * (1.0 - fx) + values[i1, j0, k0] * fx
    c10 = values[i0, j1, k0] * (1.0 - fx) + values[i1, j1, k0] * fx
    c01 = values[i0, j0, k1] * (1.0 - fx) + values[i1, j0, k1] * fx
    c11 = values[i0, j1, k1] * (1.0 - fx) + values[i1, j1, k1] * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz


# --------------------
# --- FIELD-LEVEL ----
# --------------------


def _interp_on_values(
    values: jax.Array,
    scaled_pos: jax.Array,
    *,
    method: InterpolationMethodLiteral,
    eps: float,
) -> jax.Array:
    """Dispatch the per-D primitive for one ``(*SHAPE)`` array of values."""
    D = values.ndim
    sp = scaled_pos[0] if D == 1 else scaled_pos
    if method == "spectral":
        return evaluate_spectral(values, scaled_pos, eps=eps)
    if method == "linear":
        if D == 1:
            return interp_linear_1d_periodic(values, sp)
        if D == 2:
            return interp_linear_2d_periodic(values, sp)
        if D == 3:
            return interp_linear_3d_periodic(values, sp)
        raise ValueError(f"unsupported D={D} (expected 1, 2 or 3).")


def _prepare_field(field, positions: jax.Array) -> tuple:
    """Shared validation + scaled-position computation for the field-level
    interpolation entry points.
    """

    if field.has_hat:
        field = field.ifft()

    D = field.box.D
    positions = jnp.asarray(positions)
    if positions.shape[0] != D:
        raise ValueError(
            f"positions must have leading axis of length D = {D}; got shape "
            f"{tuple(positions.shape)}."
        )
    scaled_pos = positions / field.box.R  # (D, ...) in index units
    return field, scaled_pos


def scalar_interpolate_at(
    field: "ScalarField",
    positions: jax.Array,
    *,
    method: InterpolationMethodLiteral = "linear",
    eps: float = 1e-8,
) -> jax.Array:
    """Interpolate a ScalarField at world-coordinate positions."""
    field, scaled_pos = _prepare_field(field, positions)
    return _interp_on_values(field.data, scaled_pos, method=method, eps=eps)


def vector_interpolate_at(
    field: "VectorField",
    positions: jax.Array,
    *,
    method: InterpolationMethodLiteral = "linear",
    eps: float = 1e-8,
) -> jax.Array:
    """Interpolate each component of a VectorField at world-coordinate positions."""
    field, scaled_pos = _prepare_field(field, positions)
    D = field.box.D
    return jnp.stack(
        [
            _interp_on_values(field.data[a], scaled_pos, method=method, eps=eps)
            for a in range(D)
        ],
        axis=0,
    )
