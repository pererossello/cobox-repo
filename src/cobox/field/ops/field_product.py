from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from numbers import Integral
from typing import TYPE_CHECKING, Literal

from ...box._fourier import conj_reverse

if TYPE_CHECKING:
    from jax import Array

    from ...box import Box
    from ..scalar import ScalarField


OutputN = int | Literal["full"] | None

# Retained only for existing imports in vector.py and vector_utils.py.
# The scalar operations below accept a boolean dealias, not this string.
DealiasLiteral = Literal["ozarg"]


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


def _input_mode_bounds(
    box: Box,
    N_iso: Sequence[int] | None,
    count: int,
) -> tuple[int, ...]:
    """Coordinate bounds for declared supports |k_idx| < N_iso / 2.

    N_iso is a static integer cutoff, not a storage resolution, so odd
    values are allowed. N_iso == box.N excludes input Nyquist modes.
    """
    if N_iso is None:
        raise ValueError("Dealiasing requires one N_iso per operand.")
    if len(N_iso) != count:
        raise ValueError(f"Expected {count} N_iso values, got {len(N_iso)}.")

    bounds = []
    for n in N_iso:
        if isinstance(n, bool) or not isinstance(n, Integral):
            raise TypeError("Each N_iso must be a static integer.")
        n = int(n)
        if not 0 < n <= box.N:
            raise ValueError("Each N_iso must satisfy 0 < N_iso <= box.N.")
        bounds.append((n - 1) // 2)

    return tuple(bounds)


# ------------------------
# --- GRID RESOLUTIONS ---
# ------------------------


def _full_product_N(bounds: Sequence[int]) -> int:
    """Smallest even N keeping all possible product modes below Nyquist."""
    return 2 * (sum(bounds) + 1)


def _resolve_output_N(
    out_N: OutputN,
    original_N: int,
    full_N: int,
) -> int:
    if out_N is None:
        return original_N

    if isinstance(out_N, str):
        if out_N == "full":
            return full_N
        raise ValueError("out_N must be None, 'full', or a positive even integer.")

    if isinstance(out_N, bool) or not isinstance(out_N, Integral):
        raise TypeError("out_N must be None, 'full', or a positive even integer.")

    out_N = int(out_N)
    if out_N <= 0 or out_N % 2:
        raise ValueError("An integer out_N must be positive and even.")
    return out_N


# ------------------
# --- PRIMITIVES ---
# ------------------


def naive_product(operands: Sequence[ScalarField]) -> ScalarField:
    """Pointwise multiplication on the original grid."""
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

    return ScalarField(data=result, box=box, has_hat=False)


def dealiased_product(
    operands: Sequence[ScalarField],
    *,
    N_iso: Sequence[int],
    out_N: OutputN = None,
    return_hat: bool = True,
) -> ScalarField:
    """Compute the full spectral product, then project or resample it.

    Each operand must already vanish outside |k_idx| < N_iso / 2.
    These support declarations are trusted, not checked or enforced.

    out_N=None uses the original N. 'full' uses the smallest sufficient
    even N with all possible product modes strictly below Nyquist.
    An explicit positive even integer requests exactly that resolution.
    L and D are unchanged. return_hat=True returns Fourier coefficients;
    False returns real-space values.

    Cropping retains both signs of each output Nyquist boundary mode
    before they coincide on the output grid. Its compressed Nyquist plane
    is made Hermitian so the spectrum is valid without an inverse FFT.

    N_iso, out_N and return_hat must be static configuration under JAX.
    """
    from ..scalar import ScalarField

    if not isinstance(return_hat, bool):
        raise TypeError("return_hat must be a bool.")

    box = _common_box(operands)
    bounds = _input_mode_bounds(box, N_iso, len(operands))
    full_N = _full_product_N(bounds)
    output_N = _resolve_output_N(out_N, box.N, full_N)

    # Avoid shrinking inputs before multiplication.
    work_N = max(box.N, full_N)
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
        return ScalarField(data=result, box=output_box, has_hat=False)

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
    )


# ----------------
# --- DISPATCH ---
# ----------------


def _dispatch_product(
    operands: Sequence[ScalarField],
    *,
    dealias: bool,
    N_iso: Sequence[int] | None,
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
        if N_iso is not None or out_N is not None:
            raise ValueError("N_iso and out_N require dealias=True.")
        result = naive_product(operands)
        return result.fft() if return_hat else result

    if N_iso is None:
        raise ValueError("dealias=True requires N_iso.")
    return dealiased_product(operands, N_iso=N_iso, out_N=out_N, return_hat=return_hat)


# -------------------------
# --- PUBLIC OPERATIONS ---
# -------------------------


def product(
    f1: ScalarField,
    f2: ScalarField,
    *,
    dealias: bool = False,
    N_iso: tuple[int, int] | None = None,
    out_N: OutputN = None,
    return_hat: bool | None = None,
) -> ScalarField:
    """Multiply two fields; N_iso follows operand order when dealiased.

    return_hat=None means Fourier output when dealiased, real otherwise.
    """
    return _dispatch_product(
        (f1, f2),
        dealias=dealias,
        N_iso=N_iso,
        out_N=out_N,
        return_hat=return_hat,
    )


def tri_product(
    f1: ScalarField,
    f2: ScalarField,
    f3: ScalarField,
    *,
    dealias: bool = False,
    N_iso: tuple[int, int, int] | None = None,
    out_N: OutputN = None,
    return_hat: bool | None = None,
) -> ScalarField:
    """Multiply three fields in one step, without intermediate projection."""
    return _dispatch_product(
        (f1, f2, f3),
        dealias=dealias,
        N_iso=N_iso,
        out_N=out_N,
        return_hat=return_hat,
    )


def square(
    f: ScalarField,
    *,
    dealias: bool = False,
    N_iso: int | None = None,
    out_N: OutputN = None,
    return_hat: bool | None = None,
) -> ScalarField:
    """Square a field; N_iso describes the input support."""
    cutoffs = None if N_iso is None else (N_iso, N_iso)
    return _dispatch_product(
        (f, f),
        dealias=dealias,
        N_iso=cutoffs,
        out_N=out_N,
        return_hat=return_hat,
    )


def cube(
    f: ScalarField,
    *,
    dealias: bool = False,
    N_iso: int | None = None,
    out_N: OutputN = None,
    return_hat: bool | None = None,
) -> ScalarField:
    """Cube a field in one step; N_iso describes the input support."""
    cutoffs = None if N_iso is None else (N_iso,) * 3
    return _dispatch_product(
        (f, f, f),
        dealias=dealias,
        N_iso=cutoffs,
        out_N=out_N,
        return_hat=return_hat,
    )
