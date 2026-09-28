"""Private spatial basis construction and storage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

import equinox as eqx
from jax.typing import ArrayLike

from cobox.field.ops.field_product import OutputN

from ._growth import SHAPE_KEYS
from ._psi import psi_1, psi_2, psi_3a, psi_3b, psi_3c
from ._utils import _grid_sizes, _resize_vector

if TYPE_CHECKING:
    from cobox.field import ScalarField, VectorField

    from .lpt import LPTOrderLiteral


class _LPTBasis(eqx.Module):
    """Time-independent shapes on one box, all in the same space."""

    s_1: VectorField
    s_2: VectorField | None = None
    s_3a: VectorField | None = None
    s_3b: VectorField | None = None
    s_3c: VectorField | None = None

    def __post_init__(self):
        self._validate()

    def fft(self) -> _LPTBasis:
        return self._map(lambda field: field.fft())

    def ifft(self) -> _LPTBasis:
        return self._map(lambda field: field.ifft())

    def _combine(self, weights: Mapping[str, ArrayLike]) -> VectorField:
        """sum_n w_n S_n, in the basis space."""
        data = sum(
            field.data * weights[key]
            for key, field in zip(SHAPE_KEYS, self._fields)
            if field is not None
        )
        return replace(self.s_1, data=data)

    @property
    def _fields(self) -> tuple[VectorField | None, ...]:
        return (self.s_1, self.s_2, self.s_3a, self.s_3b, self.s_3c)

    def _map(self, fn) -> _LPTBasis:
        return _LPTBasis(
            fn(self.s_1), *(None if f is None else fn(f) for f in self._fields[1:])
        )

    # ------------------
    # --- VALIDATION ---
    # ------------------

    def _validate(self) -> None:
        fields = self._fields
        for field in fields:
            if field is not None and (
                field.box != self.s_1.box or field.has_hat != self.s_1.has_hat
            ):
                raise ValueError("Basis fields must share a box and a space.")
        if self.s_2 is None and any(f is not None for f in fields[2:]):
            raise ValueError("Third-order shapes require the second-order shape.")
        if (self.s_3a is None) != (self.s_3b is None):
            raise ValueError("Provide both longitudinal third-order shapes.")
        if self.s_3c is not None and self.s_3a is None:
            raise ValueError(
                "The transverse shape requires third-order longitudinal shapes."
            )


def _build_basis(
    delta0: ScalarField,
    order: LPTOrderLiteral,
    *,
    dealias: bool,
    transverse: bool,
    N_iso: int | None,
    out_N: OutputN,
) -> _LPTBasis:
    """Build on one working grid, then project the complete basis."""
    box = delta0.box
    if order > 1 and box.D != 3:
        raise ValueError("LPT orders above 1 require D == 3.")
    work_N, output_N = _grid_sizes(box, order, dealias, N_iso, out_N)
    work_box = box if work_N == box.N else replace(box, N=work_N)
    output_box = box if output_N == box.N else replace(box, N=output_N)

    d0 = delta0.fft()
    if work_N != box.N:
        d0 = replace(d0, data=box.pad_rfft(d0.data, work_N), box=work_box)

    s1 = psi_1(d0)
    s2 = s3a = s3b = s3c = None
    if order >= 2:
        m1 = s1.jacobian(claim_symmetric=True).ifft()
        s2 = psi_2(m1)
        if order == 3:
            m2 = s2.jacobian(claim_symmetric=True).ifft()
            s3a = psi_3a(m1)
            s3b = psi_3b(m1, m2)
            if transverse:
                s3c = psi_3c(m1, m2)

    def finish(field: VectorField | None) -> VectorField | None:
        return None if field is None else _resize_vector(field, output_box)

    return _LPTBasis(
        s_1=_resize_vector(s1, output_box),
        s_2=finish(s2),
        s_3a=finish(s3a),
        s_3b=finish(s3b),
        s_3c=finish(s3c),
    )
