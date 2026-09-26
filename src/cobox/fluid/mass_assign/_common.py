from __future__ import annotations
from typing import TYPE_CHECKING
import itertools
from typing import Optional

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from ..particles import MeshConventionLiteral


def alignment_shift_native(mesh_convention: MeshConventionLiteral) -> float:
    if mesh_convention == "node":
        return 0.0
    if mesh_convention == "cell":
        return -0.5


def alignment_shift_scaled(mesh_convention: MeshConventionLiteral) -> float:
    if mesh_convention == "node":
        return 0.5
    if mesh_convention == "cell":
        return 0.0


# ---------------------------
# --- B-SPLINE ANTIDERIVS ---
# ---------------------------


def F1(u: jax.Array) -> jax.Array:
    """Antiderivative of the CIC cloud (top-hat on [-0.5, 0.5])."""
    return jnp.clip(u + 0.5, 0.0, 1.0)


def F2(u: jax.Array) -> jax.Array:
    """Antiderivative of the TSC cloud (triangle on [-1, 1])."""
    return jnp.where(
        u <= -1.0,
        0.0,
        jnp.where(
            u <= 0.0,
            0.5 * (1.0 + u) ** 2,
            jnp.where(u <= 1.0, 1.0 - 0.5 * (1.0 - u) ** 2, 1.0),
        ),
    )


def F3(u: jax.Array) -> jax.Array:
    """Antiderivative of the PCS cloud (quadratic B-spline on [-1.5, 1.5])."""
    return jnp.where(
        u <= -1.5,
        0.0,
        jnp.where(
            u <= -0.5,
            (u + 1.5) ** 3 / 6.0,
            jnp.where(
                u <= 0.5,
                0.5 + 0.75 * u - u**3 / 3.0,
                jnp.where(u <= 1.5, 1.0 - (1.5 - u) ** 3 / 6.0, 1.0),
            ),
        ),
    )


def scatter(
    indices_per_axis: list,
    weights_per_axis: list,
    particle_weights: Optional[jax.Array],
    shape: tuple,
) -> jax.Array:

    D = len(shape)
    K = indices_per_axis[0].shape[0]
    out = jnp.zeros(shape)
    for offset in itertools.product(range(K), repeat=D):
        ix = tuple(indices_per_axis[a][offset[a]] for a in range(D))
        w = particle_weights
        for a in range(D):
            wa = weights_per_axis[a]
            if wa is not None:
                w = wa[offset[a]] if w is None else w * wa[offset[a]]
        out = out.at[ix].add(1.0 if w is None else w)
    return out
