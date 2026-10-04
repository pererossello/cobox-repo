"""Lagrangian protohalo catalogue."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..cosmology.background import BackgroundCosmo

if TYPE_CHECKING:
    from cobox.box import Box


class ProtohaloCatalog(eqx.Module):
    """Protohalos in Lagrangian space, by decreasing mass.

    q: centres in Mpc/h, (D, n). M: masses in M_sun/h, (n,). a: scale factor
    each protohalo was tested at, (n,); on a light cone, a(q) at its centre.
    """

    q: jax.Array
    M: jax.Array
    a: jax.Array

    def __check_init__(self):
        n = self.q.shape[-1]
        if self.q.ndim != 2 or self.M.shape != (n,) or self.a.shape != (n,):
            raise ValueError(
                f"expected q (D, n), M (n,) and a (n,); got {self.q.shape}, "
                f"{self.M.shape} and {self.a.shape}."
            )

    def __len__(self) -> int:
        return self.M.shape[0]

    def R_lag(self, background: BackgroundCosmo) -> jax.Array:
        """Lagrangian radii in Mpc/h."""
        return background.R_lag_of_M(self.M)

    # -----------------
    # --- SELECTION ---
    # -----------------

    def select(self, mask: ArrayLike) -> ProtohaloCatalog:
        """Protohalos where mask is True. Changes the size, so it runs on the
        host, not under jit."""
        mask = jnp.asarray(mask)
        if mask.dtype != bool or mask.shape != (len(self),):
            raise ValueError(f"mask must be a boolean array of shape ({len(self)},).")
        return replace(self, q=self.q[:, mask], M=self.M[mask], a=self.a[mask])

    def M_slice(self, M_min: float, M_max: float) -> ProtohaloCatalog:
        """Protohalos with M_min <= M < M_max [M_sun/h] (half-open, so adjacent
        slices do not share protohalos). Host-side, like select."""
        if not M_min < M_max:
            raise ValueError(f"need M_min < M_max; got {M_min} and {M_max}.")
        return self.select((self.M >= M_min) & (self.M < M_max))

    def R_lag_slice(
        self, R_min: float, R_max: float, background: BackgroundCosmo
    ) -> ProtohaloCatalog:
        """Protohalos with Lagrangian radius R_min <= R_lag < R_max [Mpc/h]."""
        if not R_min < R_max:
            raise ValueError(f"need R_min < R_max; got {R_min} and {R_max}.")
        R = self.R_lag(background)
        return self.select((R >= R_min) & (R < R_max))

    # ------------
    # --- PLOT ---
    # ------------

    def paint(self, box: Box, background=None, ax=None, **kwargs):
        """Slab of centres or Lagrangian spheres; see protohalo._paint.paint."""
        from ._paint import paint

        return paint(self, box, background, ax=ax, **kwargs)
