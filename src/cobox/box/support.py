"""Declared band limits of fields on periodic grids."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .box import Box


@dataclass(frozen=True)
class ModeSupport:
    """Declared band limit: every mode has |k_idx,i| <= bound on each axis.

    bound is in units of the fundamental 2 pi / L, so it is invariant under
    resampling, which changes N but not L. A declaration is trusted, not
    checked against the data. A field carries either None (no band limit below
    its grid's own) or a support that fits its box; see ``on``.
    """

    bound: int

    def __post_init__(self):
        if isinstance(self.bound, bool) or not isinstance(self.bound, Integral):
            raise TypeError(f"bound must be an integer; got {self.bound!r}.")
        if self.bound < 0:
            raise ValueError(f"bound must be >= 0; got {self.bound}.")
        object.__setattr__(self, "bound", int(self.bound))

    @property
    def N_min(self) -> int:
        """Smallest even N holding every mode strictly below Nyquist."""
        return 2 * (self.bound + 1)

    def fits(self, box: "Box") -> bool:
        """Every mode lies strictly below the box's Nyquist on each axis."""
        return self.bound < box.N // 2

    def on(self, box: "Box") -> Optional["ModeSupport"]:
        """This support as carried by a field on box: None unless it fits."""
        return self if self.fits(box) else None

    @staticmethod
    def of_product(*supports: Optional["ModeSupport"]) -> Optional["ModeSupport"]:
        """Support of a pointwise product: bounds add; None if any is None."""
        if not supports:
            raise ValueError("of_product needs at least one support.")
        if any(s is None for s in supports):
            return None
        return ModeSupport(sum(s.bound for s in supports))

    @staticmethod
    def of_intersection(
        *supports: Optional["ModeSupport"],
    ) -> Optional["ModeSupport"]:
        """Support of a field restricted by several cuts: smallest bound.

        Here None means "no limit" and is ignored; None only if all are None.
        """
        if not supports:
            raise ValueError("of_intersection needs at least one support.")
        bounds = [s.bound for s in supports if s is not None]
        return ModeSupport(min(bounds)) if bounds else None

    @staticmethod
    def of_sum(*supports: Optional["ModeSupport"]) -> Optional["ModeSupport"]:
        """Support of a linear combination: largest bound; None if any is None."""
        if not supports:
            raise ValueError("of_sum needs at least one support.")
        if any(s is None for s in supports):
            return None
        return ModeSupport(max(s.bound for s in supports))
