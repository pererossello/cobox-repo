"""Protohalo catalogue: Lagrangian centres, masses and test epochs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar

import jax
from jax.typing import ArrayLike

from ..catalog import _AbstractCatalog

if TYPE_CHECKING:
    from cobox.field.ops.interpolation import InterpolationMethodLiteral

    from ...cosmology.background import BackgroundCosmo
    from ...lpt.lpt import LPTBasis
    from ..catalog import HaloCatalog


class ProtohaloCatalog(_AbstractCatalog):
    """Protohalos in Lagrangian space, by decreasing mass.

    q: centres in Mpc/h, (D, n). M: masses in M_sun/h, (n,). a: scale factor
    each protohalo was tested at, (n,); on a light cone, a(q) at its centre.
    """

    q: jax.Array
    M: jax.Array
    a: jax.Array

    COORD: ClassVar[str] = "q"

    def __check_init__(self):
        n = self.q.shape[-1]
        if self.q.ndim != 2 or self.M.shape != (n,) or self.a.shape != (n,):
            raise ValueError(
                f"expected q (D, n), M (n,) and a (n,); got {self.q.shape}, "
                f"{self.M.shape} and {self.a.shape}."
            )

    @property
    def positions(self) -> jax.Array:
        return self.q

    def select(self, mask: ArrayLike) -> ProtohaloCatalog:
        mask = self._mask(mask)
        return replace(self, q=self.q[:, mask], M=self.M[mask], a=self.a[mask])

    # ------------------
    # --- DISPLACING ---
    # ------------------

    def displace(
        self,
        basis: LPTBasis,
        background: BackgroundCosmo,
        *,
        observer: tuple[float, ...] | None = None,
        R_grid: Sequence[float] | None = None,
        interpolation: InterpolationMethodLiteral = "linear",
        n_iter: int = 3,
    ) -> HaloCatalog:
        """Halos at x = q + Psi_h(a), with Psi_h the LPT displacement averaged
        over each protohalo's Lagrangian sphere of radius R_lag(M): its
        centre-of-mass displacement. R_grid=None reads it at q instead.

        The shapes of basis are averaged once; Psi_h(a) = sum_n c_n(a) <s_n>_h
        is then exact at any a. observer=None places the halos at self.a.
        observer (box fractions, as LPTBasis.get_a_lc) solves each halo's own
        light-cone crossing, chi(a) = chi(|q + Psi_h(a) - observer|), by
        n_iter Newton steps; its mass stays the one tested at self.a. The halos
        then carry the observer (Mpc/h), for HaloCatalog.to_redshift_catalog.

        R_grid (Mpc/h, any order) are the radii at which spheres are averaged;
        each R_lag blends its two neighbours in ln R, exact on the grid (e.g.
        ProtohaloFinder.radii). Every R_lag must lie within it. Centres off the
        basis grid (another box) want interpolation="spectral".
        v = a H(a) / h dPsi_h / d ln a in km/s.
        """
        from ._displace import displace

        return displace(
            self,
            basis,
            background,
            observer=observer,
            R_grid=R_grid,
            interpolation=interpolation,
            n_iter=n_iter,
        )
