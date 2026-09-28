"""LPT for one initial density field, with a reusable spatial basis."""

from __future__ import annotations

from dataclasses import replace
from numbers import Integral
from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from cobox.field import ScalarField
from cobox.field.ops.field_product import OutputN

from ..background._constants import C_LIGHT
from ..linear_power.linear_growth import LINEAR_GROWTH_DISPATCH, LinearGrowthKindLiteral
from ._basis import _build_basis, _LPTBasis
from ._growth import LPTGrowthKindLiteral, coeffs, rates
from ._lightcone import LightconeMethodLiteral, crossing_a
from ._observer import get_distance_and_n_los
from ._rsd import get_dpsi_r_dlna_and_n_los

if TYPE_CHECKING:
    from cobox.box import Box
    from cobox.field import VectorField

    from ..background.cosmology import Cosmology

LPTOrderLiteral = Literal[1, 2, 3]


class LPT(eqx.Module):
    """Build spatial shapes once; evaluate them at scalar or per-point times.

    The basis starts in Fourier space. Use .ifft() to keep it in real space
    for repeated evaluations at per-point times.
    """

    _basis: _LPTBasis
    order: LPTOrderLiteral = eqx.field(static=True)
    dealias: bool = eqx.field(static=True)
    transverse: bool = eqx.field(static=True)
    linear_growth_kind: LinearGrowthKindLiteral = eqx.field(static=True)
    lpt_growth_kind: LPTGrowthKindLiteral = eqx.field(static=True)

    def __init__(
        self,
        delta0: ScalarField,
        order: LPTOrderLiteral,
        *,
        dealias: bool = True,
        transverse: bool = True,
        linear_growth_kind: LinearGrowthKindLiteral = "symbolic_pofk",
        lpt_growth_kind: LPTGrowthKindLiteral = "fit",
        N_iso: int | None = None,
        out_N: OutputN = None,
    ):
        """Construct the basis for delta0.

        N_iso declares |k_idx| < N_iso/2; it does not filter the input.
        out_N=None keeps the input grid; 'full' keeps all generated modes.
        """
        self.order = order
        self.dealias = dealias
        self.transverse = transverse
        self.linear_growth_kind = linear_growth_kind
        self.lpt_growth_kind = lpt_growth_kind
        self._validate()
        self._basis = _build_basis(
            delta0,
            order,
            dealias=dealias,
            transverse=transverse,
            N_iso=N_iso,
            out_N=out_N,
        )

    @property
    def box(self) -> Box:
        """Output box shared by the stored shapes."""
        return self._basis.s_1.box

    def fft(self) -> LPT:
        """Return an LPT instance with its basis in Fourier space."""
        return eqx.tree_at(lambda lpt: lpt._basis, self, self._basis.fft())

    def ifft(self) -> LPT:
        """Return an LPT instance with its basis in real space."""
        return eqx.tree_at(lambda lpt: lpt._basis, self, self._basis.ifft())

    def coeffs(self, a: ArrayLike, cosmology: Cosmology) -> dict:
        return coeffs(
            a,
            cosmology,
            self.linear_growth_kind,
            self.lpt_growth_kind,
            order=self.order,
        )

    def rates(self, a: ArrayLike, cosmology: Cosmology) -> dict:
        return rates(
            a,
            cosmology,
            self.linear_growth_kind,
            self.lpt_growth_kind,
            order=self.order,
        )

    def get_psi(
        self,
        a: ArrayLike,
        cosmology: Cosmology,
    ) -> VectorField:
        """Evaluate at a scalar time or one time per Lagrangian grid point.

        Scalar a preserves the stored space; a grid of times returns real space.
        """
        weights = self.coeffs(a, cosmology)
        return self._space_for(a)._combine(weights)

    def get_psi_and_dpsi_dlna(
        self,
        a: ArrayLike,
        cosmology: Cosmology,
    ) -> tuple[VectorField, VectorField]:
        """Real-space psi and dpsi/dln a, both in Mpc/h."""
        basis = self._space_for(a)
        c = self.coeffs(a, cosmology)
        f = self.rates(a, cosmology)
        psi = basis._combine(c).ifft()

        if self.order == 1:
            # First order: no additional inverse FFT.
            dpsi_dlna = replace(psi, data=psi.data * f["1"])
        else:
            derivative_weights = {key: c[key] * f[key] for key in c}
            dpsi_dlna = basis._combine(derivative_weights).ifft()
        return psi, dpsi_dlna

    # -----------
    # --- RSD ---
    # -----------

    def get_psi_rsd(
        self,
        a: ArrayLike,
        cosmology: Cosmology,
        *,
        observer: tuple[float, ...],
    ) -> VectorField:
        """Radial redshift-space displacement at a scalar or per-point a."""
        psi, dpsi_dlna = self.get_psi_and_dpsi_dlna(a, cosmology)
        dpsi_r_dlna, n_los = get_dpsi_r_dlna_and_n_los(psi, dpsi_dlna, observer)
        return replace(
            psi,
            data=psi.data + dpsi_r_dlna.data[None, ...] * n_los.data,
        )

    def get_dpsi_dlna_and_n_los(
        self,
        a: ArrayLike,
        cosmology: Cosmology,
        *,
        observer: tuple[float, ...],
    ) -> tuple[VectorField, VectorField]:
        """Return dPsi/dln a in Mpc/h and the unit line of sight."""
        psi, dpsi_dlna = self.get_psi_and_dpsi_dlna(a, cosmology)
        r_vec = psi.box.vec_from_point(observer) + psi.data
        _, n_los = get_distance_and_n_los(r_vec)
        return dpsi_dlna, replace(psi, data=n_los)

    def get_z_rsd(
        self,
        a: ArrayLike,
        cosmology: Cosmology,
        *,
        observer: tuple[float, ...],
    ) -> ScalarField:
        """Non-relativistic peculiar redshift increment."""
        dpsi_dlna, n_los = self.get_dpsi_dlna_and_n_los(
            a,
            cosmology,
            observer=observer,
        )
        dpsi_r_dlna = jnp.sum(dpsi_dlna.data * n_los.data, axis=0)

        delta_z = cosmology.H(a) / (cosmology.h * C_LIGHT) * dpsi_r_dlna
        return ScalarField(
            delta_z,
            box=dpsi_dlna.box,
        )

    # -----------------
    # --- LIGHTCONE ---
    # -----------------

    def get_a_lc(
        self,
        cosmology: Cosmology,
        *,
        observer: tuple[float, ...],
        method: LightconeMethodLiteral = "newton",
        n_iter: int = 3,
    ) -> jax.Array:
        """Past light-cone crossing scale factor, one per grid point."""
        return crossing_a(
            self,
            cosmology,
            observer=observer,
            method=method,
            n_iter=n_iter,
        )

    def get_psi_lc(
        self,
        cosmology: Cosmology,
        *,
        observer: tuple[float, ...],
        rsd: bool = False,
        method: LightconeMethodLiteral = "newton",
        n_iter: int = 3,
    ) -> tuple[jax.Array, VectorField]:
        """Crossing scale factors and displacement, optionally with radial RSD."""
        lpt = self.ifft()
        a = lpt.get_a_lc(cosmology, observer=observer, method=method, n_iter=n_iter)
        psi = (
            lpt.get_psi_rsd(a, cosmology, observer=observer)
            if rsd
            else lpt.get_psi(a, cosmology)
        )
        return a, psi

    # -------------
    # --- UTILS ---
    # -------------

    def _space_for(self, a: ArrayLike) -> _LPTBasis:
        """Per-point weights only make sense in real space."""
        a_shape = jnp.shape(a)
        if a_shape == ():
            return self._basis
        if a_shape != self.box.SHAPE:
            raise ValueError(
                f"a must be scalar or have shape {self.box.SHAPE}; got {a_shape}."
            )
        return self._basis.ifft()

    # ------------------
    # --- VALIDATION ---
    # ------------------

    def _validate(self) -> None:
        if (
            isinstance(self.order, bool)
            or not isinstance(self.order, Integral)
            or self.order not in (1, 2, 3)
        ):
            raise ValueError("order must be 1, 2, or 3.")
        if not isinstance(self.dealias, bool) or not isinstance(self.transverse, bool):
            raise TypeError("dealias and transverse must be bools.")
        if self.linear_growth_kind not in LINEAR_GROWTH_DISPATCH:
            raise ValueError(
                f"Unknown linear growth kind: {self.linear_growth_kind!r}."
            )
        if self.lpt_growth_kind not in ("eds", "fit", "ode"):
            raise ValueError(f"Unknown LPT growth kind: {self.lpt_growth_kind!r}.")
