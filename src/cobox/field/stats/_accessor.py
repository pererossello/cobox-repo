"""``ScalarField.stats``: stateless shortcuts to the free functions.

Wire it in scalar.py as

    @property
    def stats(self) -> "StatsAccessor":
        from .stats._accessor import StatsAccessor
        return StatsAccessor(self)

No caching: call ``delta.fft()`` once and ``fft()`` is a no-op afterwards.
Multi-field statistics read better as functions: ``stats.cross_spectrum(a, b)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from . import one_point, two_point

if TYPE_CHECKING:
    from ..scalar import ScalarField
    from .one_point import PDF, Moments
    from .two_point import BinsArg, CrossSpectrumEstimate, PowerSpectrumEstimate


class StatsAccessor:
    def __init__(self, field: "ScalarField"):
        self._field = field

    def moments(self) -> "Moments":
        return one_point.moments(self._field)

    def pdf(self, **kwargs) -> "PDF":
        return one_point.pdf(self._field, **kwargs)

    def power_spectrum(
        self,
        other: Optional["ScalarField"] = None,
        *,
        bins: "BinsArg" = 30,
        shot_noise: float = 0.0,
    ) -> "PowerSpectrumEstimate":
        return two_point.power_spectrum(
            self._field, other, bins=bins, shot_noise=shot_noise
        )

    def cross_spectrum(
        self,
        other: "ScalarField",
        *,
        bins: "BinsArg" = 30,
        shot_noise: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "CrossSpectrumEstimate":
        return two_point.cross_spectrum(
            self._field, other, bins=bins, shot_noise=shot_noise
        )

    def plot_modes(self, other: Optional["ScalarField"] = None, **kwargs):
        return two_point.plot_modes(self._field, other, **kwargs)
