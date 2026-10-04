"""Collapse thresholds and criteria, and their dispatch."""

from __future__ import annotations

from collections.abc import Callable
from math import pi
from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..cosmology.background import BackgroundCosmo
from ._patches import Patch

if TYPE_CHECKING:
    from .collapse import Collapse

CollapseKindLiteral = Literal["spherical"]
COLLAPSE_KINDS = ("spherical",)

DeltaCLiteral = Literal["eds", "lcdm"]
DELTA_C_KINDS = ("eds", "lcdm")

# Linear overdensity of a top-hat at collapse in Einstein-de Sitter, 1.68647.
DELTA_C_EDS = 3.0 / 20.0 * (12.0 * pi) ** (2.0 / 3.0)


# ------------------
# --- THRESHOLDS ---
# ------------------


def delta_c_eds(a: ArrayLike, background: BackgroundCosmo) -> jax.Array:
    return jnp.full(jnp.shape(a), DELTA_C_EDS)


def delta_c_lcdm(a: ArrayLike, background: BackgroundCosmo) -> jax.Array:
    """delta_c (1 + 0.0123 log10 Omega_m(a)) (Kitayama & Suto 1996, flat LCDM)."""
    c = background
    a = jnp.asarray(a, dtype=float)
    a = eqx.error_if(
        a,
        ~(c.is_flat & c.is_de_Lambda),
        "delta_c 'lcdm' is calibrated for flat LCDM only.",
    )
    return DELTA_C_EDS * (1.0 + 0.0123 * jnp.log10(c.Omega_m_of_a(a)))


DELTA_C_DISPATCH: dict[
    DeltaCLiteral, Callable[[ArrayLike, BackgroundCosmo], jax.Array]
] = {
    "eds": delta_c_eds,
    "lcdm": delta_c_lcdm,
}


# ----------------
# --- CRITERIA ---
# ----------------


def collapsed_spherical(
    collapse: Collapse,
    patch: Patch,
    a: ArrayLike,
    D: ArrayLike,
    background: BackgroundCosmo,
) -> jax.Array:
    """delta_R D(a) >= delta_c(a), node by node."""
    delta_c = DELTA_C_DISPATCH[collapse.delta_c](a, background)
    return patch.delta * D >= delta_c


COLLAPSE_DISPATCH: dict[CollapseKindLiteral, Callable[..., jax.Array]] = {
    "spherical": collapsed_spherical,
}

COLLAPSE_NEEDS: dict[CollapseKindLiteral, tuple[str, ...]] = {
    "spherical": ("delta",),
}
