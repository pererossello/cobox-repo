from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import jax

from cobox.box._fourier import conj_reverse
from cobox.field.ops.field_product import (
    OutputN,
    _full_product_N,
    _input_mode_bounds,
    _resolve_output_N,
)

if TYPE_CHECKING:
    from cobox.box import Box
    from cobox.field import VectorField

    from .lpt import LPTOrderLiteral


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
