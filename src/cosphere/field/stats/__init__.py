"""Full-sky scalar statistics measured from ShellFields; no cosmology.

    auto = field.stats.power_spectrum()
    comparison = cross_spectrum(field, other)
    comparison.correlation[:, 0, 1]

power_spectrum(a, b) returns AB alone, values of shape (L,). cross_spectrum(a, b)
returns [[AA, AB], [AB, BB]], values of shape (L, 2, 2). Both are
cosphere.spectrum.AngularSpectrum objects, the same container as predictions.
"""

from .two_point import cross_spectrum, power_spectrum

__all__ = ["cross_spectrum", "power_spectrum"]
