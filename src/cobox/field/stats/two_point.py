"""Two-point statistics. One primitive, ``mode_power(a, b)``, reduced over
|k| bins; auto spectra are the ``b=None`` case, and ``CrossSpectrumEstimate`` is a
bundle of three spectra with r, T, and error power derived from them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Sequence, Union

import jax
import jax.numpy as jnp

from ._binning import KBins, bin_average
from ._plot import PaintResult, get_ax
from ._results import Binned, stack

if TYPE_CHECKING:
    from ...box import Box
    from ..scalar import ScalarField
    from ._results import Reference

BinsArg = Union[KBins, int]
CrossPlotLiteral = Literal["r", "transfer", "error_power", "spectra"]


# ---------------
# --- RESULTS ---
# ---------------


@dataclass(frozen=True, eq=False, kw_only=True)
class PowerSpectrumEstimate(Binned):
    """Binned auto or cross power spectrum, shot noise already subtracted."""

    cross: bool = False
    shot_noise: float = 0.0

    @property
    def delta_sq(self) -> jax.Array:
        """Dimensionless Delta^2(k) = k^3 P(k) / (2 pi^2), at k_eff."""
        return self.k_eff**3 * self.values / (2.0 * jnp.pi**2)

    @property
    def gaussian_error(self) -> jax.Array:
        """Per-realization cosmic-variance error of an auto spectrum,
        (P + N) sqrt(2 / counts); counts / 2 modes are independent.
        Cross spectra: use ``CrossSpectrumEstimate.ab_error``."""
        if self.cross:
            raise ValueError("use CrossSpectrumEstimate.ab_error for cross spectra.")
        return (self.values + self.shot_noise) * jnp.sqrt(2.0 / self.counts)

    def plot(
        self,
        ax=None,
        *,
        ratio_to: Optional["Reference"] = None,
        dimless: bool = False,
        loglog: bool = True,
        figsize: tuple = (6.5, 4.4),
        **kwargs,
    ) -> PaintResult:
        if ratio_to is not None:
            y = self.ratio(ratio_to)
        elif dimless:
            y = self.delta_sq
        else:
            y = self.values
        fig, ax = get_ax(ax, figsize)
        ax.stairs(y, self.edges, baseline=None, **kwargs)
        if loglog:
            ax.set_xscale("log")
            if ratio_to is None:
                ax.set_yscale("log")
        return PaintResult(fig=fig, ax=ax)


@dataclass(frozen=True, eq=False)
class CrossSpectrumEstimate:
    """P_aa, P_bb, P_ab on shared bins; everything else is derived."""

    aa: PowerSpectrumEstimate
    bb: PowerSpectrumEstimate
    ab: PowerSpectrumEstimate

    def __post_init__(self):
        self.aa.check_compatible(self.bb)
        self.aa.check_compatible(self.ab)

    # --- shared binning ---

    @property
    def bins(self) -> KBins:
        return self.ab.bins

    @property
    def edges(self) -> jax.Array:
        return self.ab.edges

    @property
    def k_eff(self) -> jax.Array:
        return self.ab.k_eff

    @property
    def counts(self) -> jax.Array:
        return self.ab.counts

    # --- derived ---

    @property
    def r(self) -> jax.Array:
        """Cross-correlation coefficient P_ab / sqrt(P_aa P_bb)."""
        return self.ab.values / jnp.sqrt(self.aa.values * self.bb.values)

    @property
    def transfer(self) -> jax.Array:
        """Transfer function / propagator T = P_ab / P_bb (a ~ T b)."""
        return self.ab.values / self.bb.values

    @property
    def error_power(self) -> jax.Array:
        """Power of a - T b: P_aa - P_ab^2 / P_bb = P_aa (1 - r^2)."""
        return self.aa.values - self.ab.values**2 / self.bb.values

    @property
    def ab_error(self) -> jax.Array:
        """Per-realization Gaussian error of P_ab,
        sqrt((P_ab^2 + P_aa P_bb) / (counts / 2)), shot noise included."""
        p_aa = self.aa.values + self.aa.shot_noise
        p_bb = self.bb.values + self.bb.shot_noise
        p_ab = self.ab.values + self.ab.shot_noise
        return jnp.sqrt(2.0 * (p_ab**2 + p_aa * p_bb) / self.counts)

    # --- plot ---

    def plot(
        self,
        which: CrossPlotLiteral = "r",
        ax=None,
        *,
        figsize: tuple = (6.5, 4.4),
        **kwargs,
    ) -> PaintResult:
        fig, ax = get_ax(ax, figsize)
        if which == "spectra":
            for name, ps in (("aa", self.aa), ("bb", self.bb), ("ab", self.ab)):
                ax.stairs(ps.values, self.edges, baseline=None, label=name, **kwargs)
            ax.set_yscale("log")
            ax.legend()
        else:
            y = {
                "r": self.r,
                "transfer": self.transfer,
                "error_power": self.error_power,
            }[which]
            ax.stairs(y, self.edges, baseline=None, **kwargs)
            if which == "error_power":
                ax.set_yscale("log")
        ax.set_xscale("log")
        return PaintResult(fig=fig, ax=ax)

    # --- stacking ---

    @classmethod
    def _stack(cls, results: Sequence["CrossSpectrumEstimate"]) -> "CrossSpectrumEstimate":
        return cls(
            aa=stack([c.aa for c in results]),
            bb=stack([c.bb for c in results]),
            ab=stack([c.ab for c in results]),
        )


# ---------------------------
# --- PER-MODE PRIMITIVES ---
# ---------------------------


def _hats(a: "ScalarField", b: Optional["ScalarField"]):
    if b is not None and a.box != b.box:
        raise ValueError(
            "fields live on different boxes; crop explicitly with crop_rfft."
        )
    a_hat = a.fft().data
    return a_hat, (a_hat if b is None else b.fft().data)


def mode_power(a: "ScalarField", b: Optional["ScalarField"] = None) -> jax.Array:
    """(KSHAPE) Re(a b*) / V per rfft mode; |a|^2 / V when b is None."""
    a_hat, b_hat = _hats(a, b)
    return jnp.real(a_hat * jnp.conj(b_hat)) * a.box.INV_V


def mode_coherence(a: "ScalarField", b: "ScalarField") -> jax.Array:
    """(KSHAPE) per-mode normalized cross, cos(phi_a - phi_b); 0 where
    either mode vanishes. Binned r is its amplitude-weighted average."""
    a_hat, b_hat = _hats(a, b)
    prod = a_hat * jnp.conj(b_hat)
    norm = jnp.abs(prod)
    return jnp.where(norm > 0, jnp.real(prod) / jnp.where(norm > 0, norm, 1.0), 0.0)


# ------------------
# --- ESTIMATORS ---
# ------------------


def _resolve_bins(bins: BinsArg, box: "Box") -> KBins:
    return KBins.log(box, bins) if isinstance(bins, int) else bins


def power_spectrum(
    a: "ScalarField",
    b: Optional["ScalarField"] = None,
    *,
    bins: BinsArg = 30,
    shot_noise: float = 0.0,
) -> PowerSpectrumEstimate:
    """Binned P_ab(k); the auto spectrum when ``b`` is None.

    ``bins`` is a KBins (share it across results) or an int (log bins).
    ``shot_noise`` is subtracted: 1/nbar for an auto spectrum of a
    deposit, 0 for a cross between independent tracers.
    """
    bins = _resolve_bins(bins, a.box)
    values, counts, k_eff = bin_average(mode_power(a, b), bins, a.box)
    return PowerSpectrumEstimate(
        bins=bins,
        values=values - shot_noise,
        counts=counts,
        k_eff=k_eff,
        box=a.box,
        cross=b is not None and b is not a,
        shot_noise=shot_noise,
    )


def cross_spectrum(
    a: "ScalarField",
    b: "ScalarField",
    *,
    bins: BinsArg = 30,
    shot_noise: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> CrossSpectrumEstimate:
    """P_aa, P_bb, P_ab on shared bins. ``shot_noise = (N_aa, N_bb, N_ab)``;
    N_ab is nonzero only when a and b share the same discrete tracers."""
    a, b = a.fft(), b.fft()  # FFT once; the three spectra reuse it
    bins = _resolve_bins(bins, a.box)
    n_aa, n_bb, n_ab = shot_noise
    return CrossSpectrumEstimate(
        aa=power_spectrum(a, bins=bins, shot_noise=n_aa),
        bb=power_spectrum(b, bins=bins, shot_noise=n_bb),
        ab=power_spectrum(a, b, bins=bins, shot_noise=n_ab),
    )


# -------------
# --- PLOTS ---
# -------------


def plot_modes(
    a: "ScalarField",
    b: Optional["ScalarField"] = None,
    ax=None,
    *,
    coherence: bool = False,
    dimless: bool = False,
    k_max: Optional[float] = None,
    loglog: bool = True,
    figsize: tuple = (6.5, 4.4),
    **kwargs,
) -> PaintResult:
    """Scatter per-mode power (or, with ``coherence=True``, cos of the phase
    difference to ``b``) against |k|, one point per stored rfft mode."""
    box = a.box
    if coherence:
        if b is None:
            raise ValueError("coherence=True needs a second field.")
        y = mode_coherence(a, b).ravel()
    else:
        y = mode_power(a, b).ravel()
    k = box.k.ravel()
    keep = k > 0 if k_max is None else (k > 0) & (k <= k_max)
    k, y = k[keep], y[keep]
    if dimless and not coherence:
        y = k**3 * y / (2.0 * jnp.pi**2)

    fig, ax = get_ax(ax, figsize)
    ax.scatter(k, y, lw=0.0, **kwargs)
    ax.set_xlabel("$k$")
    ax.grid(True, alpha=0.3)
    if loglog:
        ax.set_xscale("log")
        if not coherence:
            ax.set_yscale("log")
    return PaintResult(fig=fig, ax=ax)
