from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Optional

import jax
import jax.numpy as jnp

from ...window import Window, Windows

if TYPE_CHECKING:
    from ..scalar import ScalarField
    from ..vector import VectorField


# ------------------
# --- PRIMITIVES ---
# ------------------


def _convolve_data(data: jax.Array, win_hat: jax.Array) -> jax.Array:
    return data * win_hat


def _deconvolve_data(
    data: jax.Array, win_hat: jax.Array, regularize: bool = False, eps: float = 1e-6
) -> jax.Array:
    if regularize:  # Wiener inverse: stable where win_hat ~ 0 (b-splines near Nyquist)
        return data * jnp.conj(win_hat) / (jnp.abs(win_hat) ** 2 + eps**2)
    return data / win_hat


# --------------
# --- SCALAR ---
# --------------


def convolve_scalar(field: "ScalarField", window: Window) -> "ScalarField":
    f = field.fft()
    win_hat = window.win_hat_on_box(f.box)
    return replace(f, data=_convolve_data(f.data, win_hat))


def deconvolve_scalar(
    field: "ScalarField",
    window: Window,
    regularize: bool = False,
    eps: float = 1e-5,
) -> "ScalarField":
    f = field.fft()
    win_hat = window.win_hat_on_box(f.box)
    data = _deconvolve_data(f.data, win_hat, regularize=regularize, eps=eps)
    return replace(f, data=data)


# --------------
# --- VECTOR ---
# --------------


def convolve_vector(field: "VectorField", window: Window) -> "VectorField":
    v = field.fft()
    win_hat = window.win_hat_on_box(
        v.box
    )  # (*KSHAPE) broadcasts over the component axis
    return replace(v, data=_convolve_data(v.data, win_hat))


def deconvolve_vector(
    field: "VectorField",
    window: Window,
    regularize: bool = False,
    eps: float = 1e-5,
) -> "VectorField":
    v = field.fft()
    win_hat = window.win_hat_on_box(v.box)
    data = _deconvolve_data(v.data, win_hat, regularize=regularize, eps=eps)
    return replace(v, data=data)
