from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import jax

from cobox.box._fourier import conj_reverse

if TYPE_CHECKING:
    from cobox.box import Box
    from cobox.field import VectorField


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
