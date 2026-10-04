"""Collapse criterion for spheres of the linear density field."""

import equinox as eqx
import jax
from jax.typing import ArrayLike

from ..cosmology.background import BackgroundCosmo
from ._collapse import (
    COLLAPSE_DISPATCH,
    COLLAPSE_KINDS,
    COLLAPSE_NEEDS,
    DELTA_C_KINDS,
    CollapseKindLiteral,
    DeltaCLiteral,
)
from ._patches import Patch

_COLLAPSE_FIELDS = ("kind", "delta_c")


class Collapse(eqx.Module):
    """Collapse criterion; configuration only.

    spherical: the sphere of radius R around node q has collapsed by a(q)
    when delta_R(q) D(a(q)) >= delta_c(a(q)), with delta_R its mean linear
    overdensity at a = 1. delta_c: eds (1.686) or lcdm (weak Omega_m(a)
    dependence, flat LCDM only).
    """

    kind: CollapseKindLiteral = eqx.field(static=True, default="spherical")
    delta_c: DeltaCLiteral = eqx.field(static=True, default="eds")

    def __check_init__(self):
        if self.kind not in COLLAPSE_DISPATCH:
            raise ValueError(
                f"unknown collapse kind: {self.kind!r}; expected one of {COLLAPSE_KINDS}."
            )
        if self.delta_c not in DELTA_C_KINDS:
            raise ValueError(
                f"unknown delta_c: {self.delta_c!r}; expected one of {DELTA_C_KINDS}."
            )

    @property
    def needs(self) -> tuple[str, ...]:
        """Patch statistics read by this criterion."""
        return COLLAPSE_NEEDS[self.kind]

    def collapsed(
        self,
        patch: Patch,
        a: ArrayLike,
        D: ArrayLike,
        background: BackgroundCosmo,
    ) -> jax.Array:
        """Boolean (*SHAPE): nodes whose sphere of radius patch.R has collapsed by a.

        a is a scalar or one value per node; D = D(a), normalized to D(1) = 1.
        """
        return COLLAPSE_DISPATCH[self.kind](self, patch, a, D, background)

    # ---------------------
    # --- SERIALIZATION ---
    # ---------------------

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in _COLLAPSE_FIELDS}

    @classmethod
    def from_dict(cls, config: dict) -> "Collapse":
        unknown = set(config) - set(_COLLAPSE_FIELDS)
        if unknown:
            raise ValueError(f"unknown keys in Collapse config: {sorted(unknown)}.")
        return cls(**config)
