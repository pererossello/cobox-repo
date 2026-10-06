"""Lagrangian bias operators up to third order, as data.

Basis: Desjacques, Jeong & Schmidt (2018), Eq. (2.61): all scalar contractions
of the symmetric distortion tensors M^(n) with total order n, excluding
tr[M^(n)] for n > 1 (reducible by the equations of motion). Counts 1, 2, 4.
This is the D = 3 basis; in lower D some entries are degenerate.

An operator is a product of traces; a trace is a tuple of M orders:
((1, 1), (1,)) is tr[M1 M1] tr[M1]. M^(n) are the time-independent shapes
built from delta0 (M1 = grad s_1, M2 = grad s_2), so tr(1) = -delta0; their
growth factors are constants absorbed by the free bias coefficients.

Plus the leading higher-derivative operator lap tr(1) = -lap delta0, from the
expansion in derivatives, independent of the perturbative order.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cobox.field.ops.field_product import naive_product

if TYPE_CHECKING:
    from cobox.field import ScalarField

TraceSpec = tuple[int, ...]


@dataclass(frozen=True)
class OperatorSpec:
    key: str
    traces: tuple[TraceSpec, ...]
    laplacian: bool = False  # lap_q applied to the product of traces

    @property
    def order(self) -> int:
        """Perturbative order: total order of all M factors."""
        return sum(sum(t) for t in self.traces)


OPERATORS: tuple[OperatorSpec, ...] = (
    # 1st
    OperatorSpec("tr(1)", ((1,),)),
    OperatorSpec("lap tr(1)", ((1,),), laplacian=True),
    # 2nd
    OperatorSpec("tr(11)", ((1, 1),)),
    OperatorSpec("tr(1)^2", ((1,), (1,))),
    # 3rd
    OperatorSpec("tr(111)", ((1, 1, 1),)),
    OperatorSpec("tr(11)tr(1)", ((1, 1), (1,))),
    OperatorSpec("tr(1)^3", ((1,), (1,), (1,))),
    OperatorSpec("tr(12)", ((1, 2),)),
)


def specs_for(order: int) -> tuple[OperatorSpec, ...]:
    """All registered operators up to ``order``, in registry order."""
    return tuple(op for op in OPERATORS if op.order <= order)


def evaluate(
    spec: OperatorSpec, trace_of: Callable[[TraceSpec], ScalarField]
) -> ScalarField:
    """Product of the spec's traces, pointwise on the caller's grid.

    trace_of returns cached traces, so repeated traces are one object and
    naive_product evaluates them once.
    """
    fields = [trace_of(t) for t in spec.traces]
    out = fields[0] if len(fields) == 1 else naive_product(fields)
    return out.laplacian() if spec.laplacian else out
