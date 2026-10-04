"""Stateless ShellField.stats shortcuts to the free estimators."""

from . import two_point


class StatsAccessor:
    def __init__(self, field):
        self._field = field

    def power_spectrum(self, other=None, *, iter=3):
        return two_point.power_spectrum(self._field, other, iter=iter)

    def cross_spectrum(self, other, *, iter=3):
        return two_point.cross_spectrum(self._field, other, iter=iter)
