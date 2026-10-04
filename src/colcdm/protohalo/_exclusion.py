"""Exclusion rules: collapsed nodes -> protohalo centres and radii.

Rules start empty on a box, are updated one radius at a time from large to
small, and finish with the node indices (D, n) and radii (n,) they accepted.
State lives on node grids of fixed shape.
"""

from dataclasses import replace
from math import ceil

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from cobox.box import Box

from ._grid import cube_argmax, distance_to
from ._patches import Patch


class FullExclusion(eqx.Module):
    """Full exclusion (Mesinger & Furlanetto 2007): accepted spheres never overlap.

    clearance(q) = min_j (|q - q_j| - R_j) over accepted spheres (periodic,
    Mpc/h); node q can host a sphere of radius R iff clearance(q) >= R.
    It only needs to be exact where |q - q_j| < 2 R_j: farther nodes clear
    every later, smaller radius. At each radius, candidates are accepted
    greedily by decreasing delta_R (ties: larger flat index first). This runs
    in rounds: a round accepts every candidate that outranks all other
    candidates in the cube of half-width 2R around it, which yields the
    same set as the sequential greedy order.
    """

    box: Box = eqx.field(static=True)
    clearance: jax.Array  # (*SHAPE)
    host: jax.Array  # (*SHAPE) int: index into radii of the sphere centred there, or -1
    radii: tuple[float, ...] = eqx.field(static=True, default=())

    @classmethod
    def start(cls, box: Box) -> "FullExclusion":
        return cls(
            box=box,
            clearance=jnp.full(box.SHAPE, jnp.inf),
            host=jnp.full(box.SHAPE, -1, dtype=jnp.int32),
        )

    def update(self, patch: Patch, collapsed: jax.Array) -> "FullExclusion":
        R = patch.R
        half_width = ceil(2.0 * R / self.box.R) - 1  # |offset| < 2R in nodes
        clearance, host = _sweep(
            collapsed,
            patch.delta,
            self.clearance,
            self.host,
            len(self.radii),
            R,
            half_width,
            self.box.R,
        )
        return replace(self, clearance=clearance, host=host, radii=self.radii + (R,))

    def finish(self) -> tuple[jax.Array, jax.Array]:
        """Node indices (D, n) and radii (n,), by decreasing radius."""
        idx = jnp.stack(jnp.nonzero(self.host >= 0))
        k = self.host[tuple(idx)]
        order = jnp.argsort(k, stable=True)
        return idx[:, order], jnp.asarray(self.radii)[k[order]]


@jax.jit
def _sweep(collapsed, delta, clearance, host, k, R, half_width, spacing):
    """Rounds of greedy acceptance at one radius, until no candidate is left."""
    index = jnp.arange(delta.size, dtype=jnp.int32).reshape(delta.shape)

    def candidates(clearance):
        return collapsed & (clearance >= R)

    def round_(state):
        clearance, host = state
        c = candidates(clearance)
        key = jnp.where(c, delta, -jnp.inf)
        accepted = c & (cube_argmax(key, jnp.where(c, index, -1), half_width) == index)
        distance = distance_to(accepted, half_width, spacing)
        return jnp.minimum(clearance, distance - R), jnp.where(accepted, k, host)

    return lax.while_loop(
        lambda state: jnp.any(candidates(state[0])), round_, (clearance, host)
    )


EXCLUSION_DISPATCH = {"full": FullExclusion}
