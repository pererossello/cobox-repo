"""Field statistics as free functions of ScalarFields.

Lives in cobox/field/stats/.

Estimators take ScalarFields; everything else composes into one first
(``convolve`` for smoothing, ``div``/``curl`` for vectors, ``deposit`` for
particles, ``deconvolve`` for mass-assignment windows). Theory enters only
as callables of k. Auto statistics are the ``b=None`` case of cross ones.

    from cobox.field import stats
    d, g = delta.fft(), other.fft()          # FFT once; estimators reuse it
    bins = stats.KBins.log(d.box, 30)

    ps = stats.power_spectrum(d, bins=bins)
    ps.plot(ratio_to=plin)

    cc = stats.cross_spectrum(d, g, bins=bins)
    cc.plot("r")
    stats.stack([cc_seed0, cc_seed1, ...]).plot("r")   # r from mean spectra

    d.stats.power_spectrum(bins=bins)        # accessor shortcut, same thing
"""

from ._binning import KBins
from ._results import Binned, stack
from .one_point import PDF, Moments, moments, pdf
from .two_point import (
    CrossSpectrumEstimate,
    PowerSpectrumEstimate,
    cross_spectrum,
    mode_coherence,
    mode_power,
    plot_modes,
    power_spectrum,
)

__all__ = [
    "KBins",
    "Binned",
    "stack",
    "Moments",
    "PDF",
    "moments",
    "pdf",
    "PowerSpectrumEstimate",
    "CrossSpectrumEstimate",
    "mode_power",
    "mode_coherence",
    "power_spectrum",
    "cross_spectrum",
    "plot_modes",
]
