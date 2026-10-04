"""Full-sky scalar angular spectra measured from ShellFields."""

from numbers import Integral
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from ._plot import PaintResult, get_ax

if TYPE_CHECKING:
    from ...shell import Shell
    from ..shell_field import ShellField


class AngularSpectrumEstimate(eqx.Module):
    """One measured auto/cross spectrum, including ell=0 without mean removal.

    values has shape (shell.L,). ell and counts are derived from the common
    Shell: ell = 0..L-1, counts = 2*ell+1. Counts describe full-sky multipoles,
    not a correction for masks or modes zeroed before measurement. No binning,
    beam deconvolution or noise subtraction is implicit. This is an immutable
    JAX PyTree containing measured data, not a cosmological theory model.
    """

    values: jax.Array
    shell: "Shell" = eqx.field(static=True)
    cross: bool = eqx.field(static=True, default=False)

    def __init__(self, values, shell: "Shell", *, cross: bool = False):
        self.values = jnp.asarray(values)
        if self.values.shape != (shell.L,) or jnp.iscomplexobj(self.values):
            raise ValueError("values must be real with shape (shell.L,).")
        if not isinstance(cross, bool):
            raise TypeError("cross must be a static bool.")
        self.shell = shell
        self.cross = cross

    @property
    def ell(self) -> jax.Array:
        return self.shell.ell_1D

    @property
    def counts(self) -> jax.Array:
        return 2 * self.ell + 1

    @property
    def scaled(self) -> jax.Array:
        """ell*(ell+1)*C_ell/(2*pi), a plotting convention, not a new estimator."""
        return self.ell * (self.ell + 1) * self.values / (2 * jnp.pi)

    def check_compatible(self, other: "AngularSpectrumEstimate") -> None:
        if self.shell != other.shell:
            raise ValueError("estimates live on different shells.")

    def ratio(self, reference) -> jax.Array:
        """Divide by a compatible estimate, callable of ell, array or scalar.

        Returns NaN wherever the reference is zero (including 0/0).
        """
        if isinstance(reference, AngularSpectrumEstimate):
            self.check_compatible(reference)
            reference = reference.values
        elif callable(reference):
            reference = reference(self.ell)
        reference = jnp.asarray(reference)
        if reference.shape not in ((), self.values.shape) or jnp.iscomplexobj(
            reference
        ):
            raise ValueError("reference must be real and scalar or match values.shape.")
        valid = reference != 0
        return jnp.where(valid, self.values / jnp.where(valid, reference, 1), jnp.nan)

    def plot(
        self,
        ax=None,
        *,
        ratio_to=None,
        scaled=False,
        loglog=False,
        figsize=(6.5, 4.4),
        **kwargs,
    ) -> PaintResult:
        """Plot raw C_ell, scaled C_ell, or a ratio; return the shared PaintResult.

        Linear axes by default preserve negative cross-power and the monopole.
        For explicitly logarithmic plots, nonpositive points are masked rather
        than plotted as absolute values. ratio_to takes precedence over scaled.
        """
        import numpy as np

        if ratio_to is not None:
            y, ylabel = self.ratio(ratio_to), "Measured / reference"
        elif scaled:
            y, ylabel = self.scaled, r"$\ell(\ell+1)\widehat C_\ell/(2\pi)$"
        else:
            y, ylabel = self.values, r"$\widehat C_\ell$"
        fig, ax = get_ax(ax, figsize)
        x, y = np.asarray(self.ell), np.asarray(y)
        if loglog:
            y = np.where((x > 0) & (y > 0), y, np.nan)
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.plot(x, y, **kwargs)
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(ylabel)
        return PaintResult(fig=fig, ax=ax)


class AngularCrossSpectrumEstimate(eqx.Module):
    """Measured AA, BB and AB spectra on the same shell; r is derived."""

    aa: AngularSpectrumEstimate
    bb: AngularSpectrumEstimate
    ab: AngularSpectrumEstimate

    def __check_init__(self):
        for estimate in (self.aa, self.bb, self.ab):
            if not isinstance(estimate, AngularSpectrumEstimate):
                raise TypeError(
                    "aa, bb and ab must be AngularSpectrumEstimate objects."
                )
        self.aa.check_compatible(self.bb)
        self.aa.check_compatible(self.ab)

    @property
    def ell(self) -> jax.Array:
        return self.ab.ell

    @property
    def counts(self) -> jax.Array:
        return self.ab.counts

    @property
    def r(self) -> jax.Array:
        """AB/sqrt(AA*BB); NaN if either auto-power vanishes. No clipping."""
        valid = (self.aa.values > 0) & (self.bb.values > 0)
        denominator = jnp.sqrt(jnp.where(valid, self.aa.values, 1)) * jnp.sqrt(
            jnp.where(valid, self.bb.values, 1)
        )
        return jnp.where(valid, self.ab.values / denominator, jnp.nan)

    def plot(
        self, which="r", ax=None, *, scaled=False, figsize=(6.5, 4.4), **kwargs
    ) -> PaintResult:
        """Plot the correlation coefficient or all three spectra on linear axes."""
        if which not in ("r", "spectra"):
            raise ValueError("which must be 'r' or 'spectra'.")
        fig, ax = get_ax(ax, figsize)
        if which == "r":
            ax.plot(self.ell, self.r, **kwargs)
            ax.set_xlabel(r"$\ell$")
            ax.set_ylabel(r"$r_\ell$")
        else:
            for name, estimate in (("AA", self.aa), ("BB", self.bb), ("AB", self.ab)):
                options = {"label": name, **kwargs}
                estimate.plot(ax=ax, scaled=scaled, **options)
            ax.legend()
        return PaintResult(fig=fig, ax=ax)


def _hats(a, b, iterations):
    from ..shell_field import ShellField

    if not isinstance(a, ShellField) or (
        b is not None and not isinstance(b, ShellField)
    ):
        raise TypeError("estimators require ShellField inputs.")
    if b is not None and a.shell != b.shell:
        raise ValueError("fields live on different shells; resample explicitly first.")
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, Integral)
        or iterations < 0
    ):
        raise ValueError("iter must be a nonnegative integer.")
    a_hat = _checked_harmonics(a.sht(iter=iterations))
    b_hat = a_hat if b is None or b is a else _checked_harmonics(b.sht(iter=iterations))
    return a_hat, b_hat


def _checked_harmonics(field):
    # Entries above the triangular domain are storage, not modes. Ignore them
    # explicitly, so even NaNs there cannot leak through a zero multiplier.
    data = jnp.asarray(field.data, dtype=jnp.result_type(field.data, jnp.float32))
    data = jnp.where(field.shell.is_mode, data, 0)
    data = eqx.error_if(
        data, jnp.any(~jnp.isfinite(data)), "Harmonic modes must be finite."
    )
    real_dtype = jnp.result_type(jnp.real(data), jnp.float32)
    tolerance = 32 * jnp.finfo(real_dtype).eps * jnp.abs(data[:, 0])
    data = eqx.error_if(
        data,
        jnp.any(jnp.abs(jnp.imag(data[:, 0])) > tolerance),
        "Real scalar fields require real m=0 coefficients.",
    )
    # Remove only tolerated imaginary roundoff in m=0.
    return data.at[:, 0].set(jnp.real(data[:, 0]))


def _estimate(a_hat, b_hat, shell, *, cross):
    mode_power = jnp.real(a_hat * jnp.conj(b_hat))
    values = jnp.sum(mode_power * shell.m_weights, axis=-1) / (2 * shell.ell_1D + 1)
    return AngularSpectrumEstimate(values, shell, cross=cross)


def power_spectrum(
    a: "ShellField", b: "ShellField | None" = None, *, iter: int = 3
) -> AngularSpectrumEstimate:
    """Full-sky C_ell = sum_m Re(a_lm conj(b_lm))/(2*ell+1); auto if b=None.

    Stored m>0 modes count twice, m=0 once. Returns every ell including zero,
    without mean subtraction. Inputs must be real scalar fields on the same
    Shell. Harmonic inputs are reused; pixel inputs use .sht(iter=iter) and
    require the optional transform backend. A masked input gives a pseudo-Cl,
    not an unbiased full-sky spectrum; no mask/noise/beam correction is applied.
    """
    a_hat, b_hat = _hats(a, b, iter)
    return _estimate(a_hat, b_hat, a.shell, cross=b is not None and b is not a)


def cross_spectrum(
    a: "ShellField", b: "ShellField", *, iter: int = 3
) -> AngularCrossSpectrumEstimate:
    """AA, BB, AB bundle, reusing each input's harmonic transform once."""
    a_hat, b_hat = _hats(a, b, iter)
    return AngularCrossSpectrumEstimate(
        aa=_estimate(a_hat, a_hat, a.shell, cross=False),
        bb=_estimate(b_hat, b_hat, a.shell, cross=False),
        ab=_estimate(a_hat, b_hat, a.shell, cross=b is not a),
    )
