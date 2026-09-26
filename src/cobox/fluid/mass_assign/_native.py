from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import jax
import jax.numpy as jnp

from ._common import alignment_shift_native, scatter

if TYPE_CHECKING:
    from ..particles import MeshConventionLiteral


def _ngp_1d(idx_pos_kernel: jax.Array, N: int):
    i = jnp.floor(idx_pos_kernel + 0.5).astype(jnp.int32) % N
    return i[None, :], None


def _cic_1d(idx_pos_kernel: jax.Array, N: int):
    i0 = jnp.floor(idx_pos_kernel).astype(jnp.int32)
    frac = idx_pos_kernel - i0
    indices = jnp.stack([i0 % N, (i0 + 1) % N], axis=0)
    weights = jnp.stack([1.0 - frac, frac], axis=0)
    return indices, weights


def _tsc_1d(idx_pos_kernel: jax.Array, N: int):
    i_center = jnp.floor(idx_pos_kernel + 0.5).astype(jnp.int32)
    e = idx_pos_kernel - i_center  # in [-0.5, 0.5)
    indices = jnp.stack([(i_center - 1) % N, i_center % N, (i_center + 1) % N], axis=0)
    weights = jnp.stack(
        [0.5 * (0.5 - e) ** 2, 0.75 - e**2, 0.5 * (0.5 + e) ** 2], axis=0
    )
    return indices, weights


def _pcs_1d(idx_pos_kernel: jax.Array, N: int):
    i0 = jnp.floor(idx_pos_kernel).astype(jnp.int32)
    frac = idx_pos_kernel - i0
    indices = jnp.stack(
        [(i0 - 1) % N, i0 % N, (i0 + 1) % N, (i0 + 2) % N], axis=0
    )
    weights = jnp.stack(
        [
            (1.0 - frac) ** 3,
            4.0 - 6.0 * frac**2 + 3.0 * frac**3,
            1.0 + 3.0 * frac + 3.0 * frac**2 - 3.0 * frac**3,
            frac**3,
        ],
        axis=0,
    ) / 6.0
    return indices, weights


_NATIVE_PRIMITIVE = {"ngp": _ngp_1d, "cic": _cic_1d, "tsc": _tsc_1d, "pcs": _pcs_1d}


def deposit_native(
    idx_pos: jax.Array,
    particle_weights: Optional[jax.Array],
    kind: str,
    shape: tuple,
    mesh_convention: MeshConventionLiteral,
) -> jax.Array:
    if kind not in _NATIVE_PRIMITIVE:
        raise ValueError(
            f"deposit_native supports {tuple(_NATIVE_PRIMITIVE)}; got {kind!r}."
        )
    N = shape[0]
    D = len(shape)
    primitive = _NATIVE_PRIMITIVE[kind]
    idx_pos_kernel = idx_pos + alignment_shift_native(mesh_convention)

    indices_per_axis = []
    weights_per_axis = []
    for a in range(D):
        ix, w = primitive(idx_pos_kernel[a], N)
        indices_per_axis.append(ix)
        weights_per_axis.append(w)
    return scatter(indices_per_axis, weights_per_axis, particle_weights, shape)
