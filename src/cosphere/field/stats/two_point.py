"""Full-sky scalar angular spectra measured from ShellFields."""

from numbers import Integral
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

from ...spectrum import AngularSpectrum

if TYPE_CHECKING:
    from ..shell_field import ShellField


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
    data = jnp.asarray(field.data)
    data = data.astype(complex if jnp.iscomplexobj(data) else float)
    data = jnp.where(field.shell.is_mode, data, 0)
    data = eqx.error_if(
        data, jnp.any(~jnp.isfinite(data)), "Harmonic modes must be finite."
    )
    # m=0 coefficients of a real field are real; drop transform roundoff.
    return data.at[:, 0].set(jnp.real(data[:, 0]))


def _estimate(a_hat, b_hat, shell):
    mode_power = jnp.real(a_hat * jnp.conj(b_hat))
    return jnp.sum(mode_power * shell.m_weights, axis=-1) / (2 * shell.ell_1D + 1)


def power_spectrum(
    a: "ShellField", b: "ShellField | None" = None, *, iter: int = 3
) -> AngularSpectrum:
    """Full-sky C_ell = sum_m Re(a_lm conj(b_lm))/(2*ell+1); auto if b=None.

    Stored m>0 modes count twice, m=0 once. Returns every ell including zero,
    without mean subtraction. Inputs must be real scalar fields on the same
    Shell. Harmonic inputs are reused; pixel inputs use .sht(iter=iter) and
    require the optional transform backend. A masked input gives a pseudo-Cl,
    not an unbiased full-sky spectrum; no mask/noise/beam correction is applied.
    """
    a_hat, b_hat = _hats(a, b, iter)
    return AngularSpectrum(_estimate(a_hat, b_hat, a.shell))


def cross_spectrum(
    a: "ShellField", b: "ShellField", *, iter: int = 3
) -> AngularSpectrum:
    """[[AA, AB], [AB, BB]] as values of shape (L, 2, 2); see .correlation.

    Reuses each input's harmonic transform once.
    """
    a_hat, b_hat = _hats(a, b, iter)
    ab = _estimate(a_hat, b_hat, a.shell)
    aa, bb = _estimate(a_hat, a_hat, a.shell), _estimate(b_hat, b_hat, a.shell)
    return AngularSpectrum(jnp.moveaxis(jnp.array([[aa, ab], [ab, bb]]), -1, 0))
