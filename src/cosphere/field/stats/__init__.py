"""Full-sky scalar statistics measured from ShellFields; no cosmology.

    auto = field.stats.power_spectrum()
    comparison = cross_spectrum(field, other)
    comparison.plot("r")

power_spectrum(a, b) returns AB alone. cross_spectrum(a, b) returns AA/BB/AB,
matching cobox.field.stats. Result objects are JAX PyTrees; plotting is host-only.
"""

from .two_point import (
    AngularCrossSpectrumEstimate,
    AngularSpectrumEstimate,
    cross_spectrum,
    power_spectrum,
)

__all__ = [
    "AngularCrossSpectrumEstimate",
    "AngularSpectrumEstimate",
    "cross_spectrum",
    "power_spectrum",
]
