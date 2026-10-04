"""Theory power spectra and their isotropic three-dimensional statistics."""

from typing import Any

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .spectrum import Spectrum


class PowerSpectrum(Spectrum):
    """A theoretical isotropic 3D auto- or cross-power spectrum P(k, *args).

    Subclasses document units; radii use the inverse units of k. For a
    dimensionless field, delta_sq is dimensionless power. For a dimensional
    field it retains the squared field units. Cross-spectra yield covariances,
    possibly negative; sigma methods only describe nonnegative auto-variances.

    This callable model is distinct from the measured, binned result
    cobox.field.stats.PowerSpectrum.
    """

    def delta_sq(self, k: ArrayLike, *args: Any, **kwargs: Any) -> jax.Array:
        """Power per logarithmic interval, k^3 P(k) / (2 pi^2)."""
        k = jnp.asarray(k, dtype=float)
        return k**3 * self(k, *args, **kwargs) / (2.0 * jnp.pi**2)

    def variance_R(
        self,
        R: ArrayLike,
        *args,
        R_prime: ArrayLike | None = None,
        **kw,
    ) -> jax.Array:
        """int Delta^2(k) W(kR) W(kR') dln k for spherical top-hats of radii R, R'.

        R_prime=None uses the same radius for both fields. Auto-power then
        gives a variance; cross-power or distinct radii give a covariance.
        """
        from ..window.window import win_hat_iso

        R = jnp.asarray(R)
        R_prime = R if R_prime is None else jnp.asarray(R_prime)
        ndim = len(jnp.broadcast_shapes(R.shape, R_prime.shape))

        def weight(k):
            k = k.reshape((-1,) + (1,) * ndim)
            # scale = diameter = 2R; normalized=True gives the bare shape W(kR), W(0)=1.
            W = win_hat_iso("radial_tophat", 2.0 * R, k, 3, True)
            W_prime = win_hat_iso("radial_tophat", 2.0 * R_prime, k, 3, True)
            return k**3 * W * W_prime / (2.0 * jnp.pi**2)

        return self.integrate(*args, weight=weight, **kw)

    def sigma_R(self, R: ArrayLike, *args, **kw) -> jax.Array:
        """RMS fluctuation in a top-hat sphere of radius R."""
        return jnp.sqrt(self.variance_R(R, *args, **kw))

    def sigma8(self, *args, **kw) -> jax.Array:
        """sigma_R at R = 8 in inverse-k units (8 Mpc/h if k is in h/Mpc)."""
        return self.sigma_R(8.0, *args, **kw)
