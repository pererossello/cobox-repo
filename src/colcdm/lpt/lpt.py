"""LPT on one working grid, with projection only after the basis is complete."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from numbers import Integral
from typing import TYPE_CHECKING, Literal, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from cobox.box._fourier import conj_reverse
from cobox.field.ops.field_product import (
    OutputN,
    _full_product_N,
    _input_mode_bounds,
    _resolve_output_N,
)

from ..linear_power.linear_growth import LINEAR_GROWTH_DISPATCH, LinearGrowthKindLiteral
from ._growth import LPTGrowthKindLiteral, coeffs, rates
from ._psi import psi_1, psi_2, psi_3a, psi_3b, psi_3c
from ._rsd import radial_rsd

if TYPE_CHECKING:
    from cobox.box import Box
    from cobox.field import ScalarField, VectorField
    from ..background.cosmology import Cosmology

LPTOrderLiteral = Literal[1, 2, 3]


class LPT(eqx.Module):
    order: LPTOrderLiteral = eqx.field(static=True)
    dealias: bool = eqx.field(static=True)
    transverse: bool = eqx.field(static=True)
    linear_growth_kind: LinearGrowthKindLiteral = eqx.field(static=True)
    lpt_growth_kind: LPTGrowthKindLiteral = eqx.field(static=True)

    def __init__(
        self,
        order: LPTOrderLiteral,
        *,
        dealias: bool = True,
        transverse: bool = True,
        linear_growth_kind: LinearGrowthKindLiteral = "symbolic_pofk",
        lpt_growth_kind: LPTGrowthKindLiteral = "fit",
    ):
        self.order = order
        self.dealias = dealias
        self.transverse = transverse
        self.linear_growth_kind = linear_growth_kind
        self.lpt_growth_kind = lpt_growth_kind
        self._validate()

    def get_psi_basis(
        self,
        delta0: ScalarField,
        *,
        N_iso: int | None = None,
        out_N: OutputN = None,
    ) -> LPTBasis:
        """N_iso declares |k_idx| < N_iso/2; None allows the full FFT box.

        out_N=None keeps the input N; 'full' keeps all generated modes.
        Dealiasing preserves all intermediates. A declared cutoff is not a filter.
        """
        box = delta0.box
        if self.order > 1 and box.D != 3:
            raise ValueError("LPT orders above 1 require D == 3.")
        work_N, output_N = _grid_sizes(box, self.order, self.dealias, N_iso, out_N)
        work_box = box if work_N == box.N else replace(box, N=work_N)
        output_box = box if output_N == box.N else replace(box, N=output_N)

        d0 = delta0.fft()
        if work_N != box.N:
            d0 = replace(d0, data=box.pad_rfft(d0.data, work_N), box=work_box)

        s1 = psi_1(d0)
        s2 = s3a = s3b = s3c = None
        if self.order >= 2:
            m1 = s1.jacobian(claim_symmetric=True).ifft()
            s2 = psi_2(m1)
            if self.order == 3:
                m2 = s2.jacobian(claim_symmetric=True).ifft()
                s3a = psi_3a(m1)
                s3b = psi_3b(m1, m2)
                if self.transverse:
                    s3c = psi_3c(m1, m2)

        def finish(field: VectorField | None) -> VectorField | None:
            return None if field is None else _resize_vector(field, output_box)

        return LPTBasis(
            s_1=_resize_vector(s1, output_box),
            s_2=finish(s2),
            s_3a=finish(s3a),
            s_3b=finish(s3b),
            s_3c=finish(s3c),
        )

    def get_psi(
        self,
        delta0: ScalarField,
        a: ArrayLike,
        cosmology: Cosmology,
        *,
        N_iso: int | None = None,
        out_N: OutputN = None,
    ) -> VectorField:
        basis = self.get_psi_basis(delta0, N_iso=N_iso, out_N=out_N)
        return basis.get_psi(
            a, cosmology, self.linear_growth_kind, self.lpt_growth_kind
        )

    def get_psi_rsd(
        self,
        delta0: ScalarField,
        a: ArrayLike,
        cosmology: Cosmology,
        *,
        observer: Tuple[float, ...],
        N_iso: int | None = None,
        out_N: OutputN = None,
    ) -> VectorField:

        basis = self.get_psi_basis(delta0, N_iso=N_iso, out_N=out_N)
        return basis.get_psi_rsd(
            a,
            cosmology,
            observer=observer,
            linear_growth_kind=self.linear_growth_kind,
            lpt_growth_kind=self.lpt_growth_kind,
        )

    def coeffs(self, a: ArrayLike, cosmology: Cosmology) -> dict:
        return coeffs(a, cosmology, self.linear_growth_kind, self.lpt_growth_kind)

    def rates(self, a: ArrayLike, cosmology: Cosmology) -> dict:
        return rates(a, cosmology, self.linear_growth_kind, self.lpt_growth_kind)

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


class LPTBasis(eqx.Module):
    """Time-independent shapes on one box, all stored in Fourier space."""

    s_1: VectorField
    s_2: VectorField | None = None
    s_3a: VectorField | None = None
    s_3b: VectorField | None = None
    s_3c: VectorField | None = None

    def __post_init__(self):
        self._validate()

    def get_psi(
        self,
        a: ArrayLike,
        cosmology: Cosmology,
        linear_growth_kind: LinearGrowthKindLiteral = "symbolic_pofk",
        lpt_growth_kind: LPTGrowthKindLiteral = "fit",
    ) -> VectorField:
        """Combine the basis at a single scale factor."""
        if jnp.ndim(a) != 0:
            raise ValueError("get_psi expects a scalar scale factor.")
        return self._combine(coeffs(a, cosmology, linear_growth_kind, lpt_growth_kind))

    def get_psi_rsd(
        self,
        a: ArrayLike,
        cosmology: Cosmology,
        *,
        observer: Tuple[float, ...],
        linear_growth_kind: LinearGrowthKindLiteral = "symbolic_pofk",
        lpt_growth_kind: LPTGrowthKindLiteral = "fit",
    ) -> VectorField:

        if jnp.ndim(a) != 0:
            raise ValueError("get_psi_rsd expects a scalar scale factor.")

        c = coeffs(a, cosmology, linear_growth_kind, lpt_growth_kind)
        f = rates(a, cosmology, linear_growth_kind, lpt_growth_kind)
        psi = self._combine(c).ifft()

        if self.s_2 is None:
            vel = psi * f["1"]  # First order: no additional inverse FFT.
        else:
            velocity_weights = {key: c[key] * f[key] for key in c}
            vel = self._combine(velocity_weights).ifft()

        return radial_rsd(psi, vel, observer)

    def _combine(self, weights: Mapping[str, ArrayLike]) -> VectorField:
        out = self.s_1 * weights["1"]
        for key, field in (
            ("2", self.s_2),
            ("3a", self.s_3a),
            ("3b", self.s_3b),
            ("3c", self.s_3c),
        ):
            if field is not None:
                out = out + field * weights[key]
        return out

    # ------------------
    # --- VALIDATION ---
    # ------------------

    def _validate(self) -> None:
        fields = (self.s_1, self.s_2, self.s_3a, self.s_3b, self.s_3c)
        for field in fields:
            if field is not None and (field.box != self.s_1.box or not field.has_hat):
                raise ValueError(
                    "Basis fields must share a box and be in Fourier space."
                )
        if self.s_2 is None and any(f is not None for f in fields[2:]):
            raise ValueError("Third-order shapes require the second-order shape.")
        if (self.s_3a is None) != (self.s_3b is None):
            raise ValueError("Provide both longitudinal third-order shapes.")
        if self.s_3c is not None and self.s_3a is None:
            raise ValueError(
                "The transverse shape requires third-order longitudinal shapes."
            )


def _grid_sizes(
    box: Box,
    order: LPTOrderLiteral,
    dealias: bool,
    N_iso: int | None,
    out_N: OutputN,
) -> tuple[int, int]:
    bound = box.N // 2 if N_iso is None else _input_mode_bounds(box, (N_iso,), 1)[0]
    full_N = _full_product_N((bound,) * order)
    if not dealias and out_N == "full":
        raise ValueError("out_N='full' requires dealias=True.")
    output_N = _resolve_output_N(out_N, box.N, full_N)
    work_N = max(box.N, full_N) if dealias else box.N
    return work_N, output_N


def _resize_vector(field: VectorField, box: Box) -> VectorField:
    """Fourier projection/resampling, with canonical rFFT boundary planes."""
    f = field.fft()
    data = f.data
    if box.N < f.box.N:
        data = f.box.crop_rfft(data, box.N)
    elif box.N > f.box.N:
        data = f.box.pad_rfft(data, box.N)
    for boundary in (slice(0, 1), slice(-1, None)):
        plane = data[..., boundary]
        partner = jax.vmap(conj_reverse)(plane.conj())
        data = data.at[..., boundary].set(0.5 * (plane + partner))
    return replace(f, data=data, box=box)
