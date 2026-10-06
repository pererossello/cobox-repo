"""Shared Cartesian grid policy for polynomial field operations."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import TYPE_CHECKING, Literal

from ...box.support import ModeSupport

if TYPE_CHECKING:
    from ...box import Box

OutputN = int | Literal["full"] | None


def resolve_out_N(out_N: OutputN, original_N: int, full_N: int) -> int:
    """Output resolution policy shared by products and Lagrangian builders.

    None keeps original_N, 'full' gives full_N (the smallest grid holding the
    whole product), and a positive even integer is used as given.
    """
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


def product_grid_sizes(
    box: Box,
    supports: Sequence[ModeSupport | None],
    *,
    dealias: bool = True,
    out_N: OutputN = None,
) -> tuple[int, int]:
    """Return (work_N, output_N) for a product on a common input box.

    Bounds are per-axis mode indices, not spherical radii. A missing declaration
    uses the input Nyquist bound N//2, including boundary modes. This computes
    products of the represented grid fields; it cannot undo earlier aliasing.
    The inferred bounds size the grid only and do not alter field metadata.

    Repeat a support p times for a polynomial of degree p. Dealiasing never
    shrinks the working grid. Without dealiasing, the work grid stays unchanged;
    an explicit output size resamples that result, whereas 'full' is rejected.
    """
    supports = tuple(supports)
    if not supports:
        raise ValueError("At least one operand support is required.")
    if not isinstance(dealias, bool):
        raise TypeError("dealias must be a bool.")
    bound = sum(box.N // 2 if s is None else s.bound for s in supports)
    full_N = ModeSupport(bound).N_min
    if not dealias and out_N == "full":
        raise ValueError("out_N='full' requires dealias=True.")
    output_N = resolve_out_N(out_N, box.N, full_N)
    work_N = max(box.N, full_N) if dealias else box.N
    return work_N, output_N
