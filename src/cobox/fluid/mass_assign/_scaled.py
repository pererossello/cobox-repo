from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import jax
import jax.numpy as jnp

from ._common import F1, F2, F3, alignment_shift_scaled, scatter
from ._native import deposit_native

if TYPE_CHECKING:
    from ..particles import MeshConventionLiteral

_CLOUD = {
    "cic": (F1, 0.5, 1),
    "tsc": (F2, 1.0, 2),
    "pcs": (F3, 1.5, 3),
}


def _cloud_1d(
    idx_pos_v: jax.Array,
    scale_in_cells: float,
    N: int,
    K_max: int,
    antideriv,
    half_width_factor: float,
):
    half = half_width_factor * scale_in_cells
    j_start = jnp.floor(idx_pos_v - half).astype(jnp.int32)
    offsets = jnp.arange(K_max, dtype=jnp.int32)
    j_cells = j_start[None, :] + offsets[:, None]  # (K_max, n)
    u_lower = (j_cells - idx_pos_v[None, :]) / scale_in_cells
    u_upper = (j_cells + 1.0 - idx_pos_v[None, :]) / scale_in_cells
    weights = antideriv(u_upper) - antideriv(u_lower)
    indices = j_cells % N
    return indices, weights


def deposit_scaled(
    idx_pos: jax.Array,
    particle_weights: Optional[jax.Array],
    kind: str,
    shape: tuple,
    mesh_convention: MeshConventionLiteral,
    scale_in_cells: float,
    K_max: int,
) -> jax.Array:

    if kind == "ngp":
        return deposit_native(idx_pos, particle_weights, "ngp", shape, mesh_convention)
    if kind not in _CLOUD:
        raise ValueError(
            f"deposit_scaled supports 'ngp', {tuple(_CLOUD)}; got {kind!r}."
        )
    if K_max < 1:
        raise ValueError(f"K_max must be >= 1; got {K_max!r}.")

    N = shape[0]
    D = len(shape)
    antideriv, half_width_factor, _ = _CLOUD[kind]
    idx_pos_v = idx_pos + alignment_shift_scaled(mesh_convention)

    indices_per_axis = []
    weights_per_axis = []
    for a in range(D):
        ix, w = _cloud_1d(
            idx_pos_v[a], scale_in_cells, N, K_max, antideriv, half_width_factor
        )
        indices_per_axis.append(ix)
        weights_per_axis.append(w)
    return scatter(indices_per_axis, weights_per_axis, particle_weights, shape)


def k_max_for(kind: str, scale_in_cells: float) -> int:
    """Static upper bound on cells touched per axis by one particle's cloud."""
    import math

    p_cloud = _CLOUD[kind][2]
    return int(math.ceil(p_cloud * scale_in_cells)) + 1
