"""Fourier-mode operations shared by ScalarField, VectorField and TensorField."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, TypeVar

import jax.numpy as jnp

from ...box.support import ModeSupport

if TYPE_CHECKING:
    from ...box import ModeMask
    from ..scalar import ScalarField
    from ..vector import VectorField
    from ..tensor import TensorField

F = TypeVar("F", "ScalarField", "VectorField", "TensorField")


def resample(field: F, N: int) -> F:
    """Fourier crop or zero-pad onto an N grid with the same L and D.

    Returns Fourier space. Exact when the field's modes fit the new grid (a
    support that fits it); otherwise modes above the new Nyquist are dropped
    and the support is normalized away.
    """
    f = field.fft()
    if N == f.box.N:
        return f
    data = f.box.crop_rfft(f.data, N) if N < f.box.N else f.box.pad_rfft(f.data, N)
    return replace(f, data=data, box=replace(f.box, N=N))


def restrict(field: F, mask: ModeMask) -> F:
    """Zero every mode outside ``mask``, in Fourier space.

    The mask is shared by all components. The support tightens to the smaller
    of the field's own and the mask's.
    """
    f = field.fft()
    return replace(
        f,
        data=jnp.where(mask.mask(f.box), f.data, 0.0),
        support=ModeSupport.of_intersection(f.support, mask.support(f.box)),
    )
