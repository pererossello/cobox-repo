"""Private construction of the Lagrangian bias operators from delta0."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from cobox.field.ops._grids import OutputN, product_grid_sizes
from cobox.field.ops.tensor_utils import trace, trace_of_products

from ..lpt._psi import psi_1, psi_2
from ._operators import TraceSpec, evaluate, specs_for

if TYPE_CHECKING:
    from cobox.field import ScalarField, TensorField


def _build_operators(
    delta0: ScalarField,
    order: int,
    *,
    dealias: bool,
    out_N: OutputN,
    subtract_mean: bool,
) -> dict[str, ScalarField]:
    """Every operator up to ``order``, built on one work grid, then projected.

    LPT's grid rule: with dealias, delta0 is padded so that products of up to
    ``order`` linear factors fit (tr(12) has support b + 2b, as a cubic).
    Every operator is then an exact pointwise product there, and each result
    is resampled onto the output box, the box LPT uses for the same out_N.
    """
    box = delta0.box
    if order > 1 and box.D != 3:
        raise ValueError("Bias operators above first order require D == 3.")
    work_N, output_N = product_grid_sizes(
        box, (delta0.support,) * order, dealias=dealias, out_N=out_N
    )

    d0 = delta0.resample(work_N)
    M: dict[int, TensorField] = {1: psi_1(d0).jacobian(claim_symmetric=True).ifft()}
    if order >= 3:
        M[2] = psi_2(M[1]).jacobian(claim_symmetric=True).ifft()

    cache: dict[TraceSpec, ScalarField] = {}

    def trace_of(t: TraceSpec) -> ScalarField:
        if t not in cache:
            tensors = [M[n] for n in t]
            cache[t] = (
                trace(tensors[0])
                if len(t) == 1
                else trace_of_products(tensors, return_hat=False)
            )
        return cache[t]

    operators = {}
    for spec in specs_for(order):
        field = evaluate(spec, trace_of).resample(output_N)
        if subtract_mean:
            field = replace(field, data=field.data.at[(0,) * box.D].set(0.0))
        operators[spec.key] = field
    return operators
