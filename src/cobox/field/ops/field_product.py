from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from ...box._fourier import conj_reverse
from ...box.support import ModeSupport
from ._grids import OutputN, product_grid_sizes

if TYPE_CHECKING:
    from jax import Array

    from ...box import Box
    from ..scalar import ScalarField


# ------------------
# --- VALIDATION ---
# ------------------


def _common_box(operands: Sequence[ScalarField]) -> Box:
    if not operands:
        raise ValueError("At least one operand is required.")

    box = operands[0].box
    if any(f.box != box for f in operands[1:]):
        raise ValueError("Product operands live on different boxes.")
    return box


# ------------------
# --- PRIMITIVES ---
# ------------------


def naive_product(operands: Sequence[ScalarField]) -> ScalarField:
    """Pointwise multiplication on the original grid.

    The result carries the summed support when it fits the grid (the product
    is then exact); otherwise it aliased and carries none.
    """
    from ..scalar import ScalarField

    box = _common_box(operands)
    values: dict[int, Array] = {}

    for f in operands:
        key = id(f)
        if key not in values:
            values[key] = f.ifft().data

    result = values[id(operands[0])]
    for f in operands[1:]:
        result = result * values[id(f)]

    support = ModeSupport.of_product(*(f.support for f in operands))
    return ScalarField(data=result, box=box, has_hat=False, support=support)


def dealiased_product(
    operands: Sequence[ScalarField],
    *,
    out_N: OutputN = None,
    return_hat: bool = True,
) -> ScalarField:
    """Compute the full spectral product, then project or resample it.

    Declared supports are trusted, not checked. Missing declarations use the
    full input-grid bound for padding. This gives a dealiased product of the
    represented grid fields; it cannot repair aliasing already in the inputs.

    out_N=None uses the original N. 'full' uses the smallest sufficient
    even N with all possible product modes strictly below Nyquist.
    An explicit positive even integer requests exactly that resolution.
    L and D are unchanged. return_hat=True returns Fourier coefficients;
    False returns real-space values. The result carries the summed support,
    normalized on the output box (None if projected below it or any input
    declaration was missing).

    Cropping retains both signs of each output Nyquist boundary mode
    before they coincide on the output grid. Its compressed Nyquist plane
    is made Hermitian so the spectrum is valid without an inverse FFT.

    out_N and return_hat must be static configuration under JAX.
    """
    from ..scalar import ScalarField

    if not isinstance(return_hat, bool):
        raise TypeError("return_hat must be a bool.")

    box = _common_box(operands)
    supports = tuple(f.support for f in operands)
    support = ModeSupport.of_product(*supports)
    work_N, output_N = product_grid_sizes(box, supports, out_N=out_N)
    work_box = box if work_N == box.N else replace(box, N=work_N)

    # Repeated operands, as in square/cube, share their padded values.
    values: dict[int, Array] = {}
    for f in operands:
        key = id(f)
        if key not in values:
            if work_N == box.N:
                values[key] = f.ifft().data
            else:
                padded = box.pad_rfft(f.fft().data, work_N)
                values[key] = work_box.ifft(padded)

    result = values[id(operands[0])]
    for f in operands[1:]:
        result = result * values[id(f)]

    # Preserve the original box object when returning its resolution.
    output_box = box if output_N == box.N else replace(box, N=output_N)
    if output_N == work_N and not return_hat:
        return ScalarField(data=result, box=output_box, has_hat=False, support=support)

    spectrum = work_box.fft(result)
    if output_N < work_N:
        spectrum = work_box.crop_rfft(spectrum, output_N)
        # Retain the explicit Hermitian projection; it is idempotent with
        # the boundary combination already performed by crop_rfft.
        boundary = spectrum[..., -1:]
        boundary = 0.5 * (boundary + conj_reverse(boundary.conj()))
        spectrum = spectrum.at[..., -1:].set(boundary)
    elif output_N > work_N:
        spectrum = work_box.pad_rfft(spectrum, output_N)

    return ScalarField(
        data=spectrum if return_hat else output_box.ifft(spectrum),
        box=output_box,
        has_hat=return_hat,
        support=support,
    )


# ----------------
# --- DISPATCH ---
# ----------------


def _dispatch_product(
    operands: Sequence[ScalarField],
    *,
    dealias: bool,
    out_N: OutputN,
    return_hat: bool | None,
) -> ScalarField:
    if not isinstance(dealias, bool):
        raise TypeError("dealias must be a bool.")
    if return_hat is None:
        return_hat = dealias
    elif not isinstance(return_hat, bool):
        raise TypeError("return_hat must be a bool or None.")

    if not dealias:
        if out_N is not None:
            raise ValueError("out_N requires dealias=True.")
        result = naive_product(operands)
        return result.fft() if return_hat else result

    return dealiased_product(operands, out_N=out_N, return_hat=return_hat)


# -------------------------
# --- PUBLIC OPERATIONS ---
# -------------------------


def product(
    f1: ScalarField,
    f2: ScalarField,
    *,
    dealias: bool = False,
    out_N: OutputN = None,
    return_hat: bool | None = None,
) -> ScalarField:
    """Multiply two fields; dealiasing reads both supports.

    return_hat=None means Fourier output when dealiased, real otherwise.
    """
    return _dispatch_product(
        (f1, f2), dealias=dealias, out_N=out_N, return_hat=return_hat
    )


def tri_product(
    f1: ScalarField,
    f2: ScalarField,
    f3: ScalarField,
    *,
    dealias: bool = False,
    out_N: OutputN = None,
    return_hat: bool | None = None,
) -> ScalarField:
    """Multiply three fields in one step, without intermediate projection."""
    return _dispatch_product(
        (f1, f2, f3), dealias=dealias, out_N=out_N, return_hat=return_hat
    )


def square(
    f: ScalarField,
    *,
    dealias: bool = False,
    out_N: OutputN = None,
    return_hat: bool | None = None,
) -> ScalarField:
    """Square a field."""
    return _dispatch_product(
        (f, f), dealias=dealias, out_N=out_N, return_hat=return_hat
    )


def cube(
    f: ScalarField,
    *,
    dealias: bool = False,
    out_N: OutputN = None,
    return_hat: bool | None = None,
) -> ScalarField:
    """Cube a field in one step, without intermediate projection."""
    return _dispatch_product(
        (f, f, f), dealias=dealias, out_N=out_N, return_hat=return_hat
    )
