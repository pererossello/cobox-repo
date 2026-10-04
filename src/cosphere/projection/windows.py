"""Radial integration weights, distinct from angular or Cartesian smoothing."""

import abc
from functools import lru_cache
from operator import index

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf
from jax.typing import ArrayLike


def _real_array(value: ArrayLike) -> jax.Array:
    value = jnp.asarray(value)
    if jnp.issubdtype(value.dtype, jnp.complexfloating):
        raise TypeError("Radial windows require real values.")
    return value.astype(jnp.result_type(value.dtype, jnp.float32))


def _scalar(value: ArrayLike, name: str) -> jax.Array:
    value = _real_array(value)
    if value.ndim != 0:
        raise ValueError(f"{name} must be scalar.")
    return eqx.error_if(value, ~jnp.isfinite(value), f"{name} must be finite.")


def _check_normalize(normalize: bool) -> None:
    if not isinstance(normalize, bool):
        raise TypeError("normalize must be a static bool.")


@lru_cache(maxsize=32)
def _legendre(order: int) -> tuple[np.ndarray, np.ndarray]:
    # Cache only host constants; never put traced physical arrays in a cache.
    return np.polynomial.legendre.leggauss(order)


def _check_n_r(n_r: int) -> int:
    if isinstance(n_r, bool):
        raise TypeError("n_r must be an integer >= 2.")
    n_r = index(n_r)
    if n_r < 2:
        raise ValueError("n_r must be >= 2.")
    return n_r


class RadialWindow(eqx.Module):
    """A radial integration measure, with ordinary profiles as a special case.

    Implement support and quadrature to define a new window. Quadrature weights
    already include W(r) dr: consumers evaluate sum(weights * f(r)), without
    evaluating a pointwise profile. This also covers Dirac distributions.
    Physical parameters remain differentiable PyTree leaves.
    """

    @property
    @abc.abstractmethod
    def support(self) -> tuple[jax.Array, jax.Array]:
        """Lower and upper radial bounds, possibly equal for a thin shell."""

    @abc.abstractmethod
    def quadrature(self, n_r: int = 128) -> tuple[jax.Array, jax.Array]:
        """Return 1D radii and matching weights for integral W(r) f(r) dr.

        n_r is a static resolution request, >= 2; the actual node count may
        differ. Radii are finite and nonnegative. Weights include amplitude
        and normalization and may be signed. Exact discrete measures may
        ignore the requested resolution. Ordinary profiles can additionally
        implement __call__(r), which is not part of this base interface.
        """


class TopHatWindow(RadialWindow):
    """A constant radial weight on [r_min, r_max], zero outside.

    normalize=True gives unit integral in dr. Otherwise the height is one,
    so the integral is r_max - r_min. Distances are nonnegative, in any
    consistent units. Bounds are differentiable PyTree leaves.
    """

    r_min: jax.Array
    r_max: jax.Array
    normalize: bool = eqx.field(static=True)

    def __init__(self, r_min: ArrayLike, r_max: ArrayLike, normalize: bool = True):
        _check_normalize(normalize)
        self.r_min = _scalar(r_min, "r_min")
        self.r_max = _scalar(r_max, "r_max")
        self.r_min = eqx.error_if(
            self.r_min,
            (self.r_min < 0) | (self.r_max <= self.r_min),
            "Require 0 <= r_min < r_max.",
        )
        self.normalize = normalize

    @property
    def support(self) -> tuple[jax.Array, jax.Array]:
        return self.r_min, self.r_max

    def __call__(self, r: ArrayLike) -> jax.Array:
        r = jnp.asarray(r)
        height = 1 / (self.r_max - self.r_min) if self.normalize else 1.0
        return jnp.where((r >= self.r_min) & (r <= self.r_max), height, 0.0)

    def quadrature(self, n_r: int = 128) -> tuple[jax.Array, jax.Array]:
        """n_r Gauss-Legendre nodes over the interval, including normalization."""
        nodes, weights = _legendre(_check_n_r(n_r))
        width = self.r_max - self.r_min
        nodes = jnp.asarray(nodes, dtype=width.dtype)
        weights = jnp.asarray(weights, dtype=width.dtype)
        r = self.r_min + width * (nodes + 1) / 2
        measure = weights / 2 if self.normalize else weights * width / 2
        return r, measure


class GaussianWindow(RadialWindow):
    """Gaussian radial weight with finite, nonnegative support.

    The profile is exp(-0.5 * ((r - center) / sigma)**2), truncated to
    [max(0, center - truncate * sigma), center + truncate * sigma].
    normalize=True gives unit integral in dr over that actual interval;
    otherwise the peak height is one. center >= 0, sigma > 0 and truncate > 0
    are differentiable leaves. sigma describes the underlying Gaussian,
    not the standard deviation after truncation. Derivatives are piecewise
    smooth where the lower bound switches between zero and center - truncate*sigma.
    """

    center: jax.Array
    sigma: jax.Array
    truncate: jax.Array
    normalize: bool = eqx.field(static=True)

    def __init__(
        self,
        center: ArrayLike,
        sigma: ArrayLike,
        truncate: ArrayLike = 5.0,
        normalize: bool = True,
    ):
        _check_normalize(normalize)
        self.center = _scalar(center, "center")
        self.sigma = _scalar(sigma, "sigma")
        self.truncate = _scalar(truncate, "truncate")
        self.center = eqx.error_if(
            self.center, self.center < 0, "center must be nonnegative."
        )
        self.sigma = eqx.error_if(
            self.sigma, self.sigma <= 0, "sigma must be positive."
        )
        self.truncate = eqx.error_if(
            self.truncate, self.truncate <= 0, "truncate must be positive."
        )
        self.normalize = normalize
        lower, upper = self.support
        self.sigma = eqx.error_if(
            self.sigma,
            ~jnp.isfinite(upper) | (upper <= lower),
            "Gaussian support must have finite, positive width at this precision.",
        )

    @property
    def support(self) -> tuple[jax.Array, jax.Array]:
        half_width = self.truncate * self.sigma
        return jnp.maximum(0, self.center - half_width), self.center + half_width

    def _scaled_bounds(self) -> tuple[jax.Array, jax.Array]:
        return -jnp.minimum(self.center / self.sigma, self.truncate), self.truncate

    def _scaled_integral(self) -> jax.Array:
        lower, upper = self._scaled_bounds()
        return jnp.sqrt(jnp.pi / 2) * (
            erf(upper / jnp.sqrt(2.0)) - erf(lower / jnp.sqrt(2.0))
        )

    def __call__(self, r: ArrayLike) -> jax.Array:
        r = jnp.asarray(r)
        lower, upper = self.support
        profile = jnp.exp(-0.5 * ((r - self.center) / self.sigma) ** 2)
        if self.normalize:
            profile = profile / (self.sigma * self._scaled_integral())
        return jnp.where((r >= lower) & (r <= upper), profile, 0.0)

    def quadrature(self, n_r: int = 128) -> tuple[jax.Array, jax.Array]:
        """Gauss-Legendre quadrature in (r - center) / sigma.

        Increase n_r for large truncation ranges or oscillatory integrands.
        """
        nodes, weights = _legendre(_check_n_r(n_r))
        lower, upper = self._scaled_bounds()
        dtype = jnp.result_type(self.center, self.sigma, self.truncate)
        fraction = (jnp.asarray(nodes, dtype=dtype) + 1) / 2
        width = upper - lower
        z = lower + width * fraction
        r_min, r_max = self.support
        r = r_min + (r_max - r_min) * fraction
        measure = jnp.asarray(weights, dtype=dtype) * width / 2 * jnp.exp(-0.5 * z**2)
        if self.normalize:
            measure = measure / self._scaled_integral()
        else:
            measure = measure * self.sigma
        return r, measure


class TabulatedWindow(RadialWindow):
    """Piecewise-linear radial weight, zero outside [r[0], r[-1]].

    r must be a strictly increasing, nonnegative 1D array with at least two
    knots. values has the same shape; both remain differentiable leaves.
    Raw values are stored. normalize=True divides by their integral in dr;
    zero or numerically cancelling integrals are rejected. Use normalize=False
    for compensated profiles or to preserve a signed weight's amplitude.
    """

    r: jax.Array
    values: jax.Array
    normalize: bool = eqx.field(static=True)

    def __init__(self, r: ArrayLike, values: ArrayLike, normalize: bool = True):
        _check_normalize(normalize)
        r, values = _real_array(r), _real_array(values)
        if r.ndim != 1 or r.size < 2 or values.shape != r.shape:
            raise ValueError("r and values must be 1D arrays of equal length >= 2.")
        self.r = eqx.error_if(
            r,
            jnp.any(~jnp.isfinite(r)) | (r[0] < 0) | jnp.any(jnp.diff(r) <= 0),
            "r must be finite, nonnegative and strictly increasing.",
        )
        self.values = eqx.error_if(
            values, jnp.any(~jnp.isfinite(values)), "values must be finite."
        )
        self.normalize = normalize
        if normalize:
            integral = jnp.trapezoid(self.values, self.r)
            scale = jnp.trapezoid(jnp.abs(self.values), self.r)
            tolerance = 16 * jnp.finfo(integral.dtype).eps * scale
            self.values = eqx.error_if(
                self.values,
                ~jnp.isfinite(integral) | (jnp.abs(integral) <= tolerance),
                "Cannot normalize a zero or numerically cancelling integral; "
                "use normalize=False for compensated weights.",
            )

    @property
    def support(self) -> tuple[jax.Array, jax.Array]:
        return self.r[0], self.r[-1]

    def _profile_values(self) -> jax.Array:
        if self.normalize:
            return self.values / jnp.trapezoid(self.values, self.r)
        return self.values

    def __call__(self, r: ArrayLike) -> jax.Array:
        return jnp.interp(
            jnp.asarray(r), self.r, self._profile_values(), left=0.0, right=0.0
        )

    def quadrature(self, n_r: int = 128) -> tuple[jax.Array, jax.Array]:
        """Split at knots, using at least two Gauss-Legendre nodes per segment.

        Each segment gets ceil(n_r / n_segments) nodes, with a minimum of two;
        the total may therefore exceed n_r. Weights include the linear profile.
        """
        n_r = _check_n_r(n_r)
        n_segments = self.r.size - 1
        order = max(2, (n_r + n_segments - 1) // n_segments)
        nodes, weights = _legendre(order)
        dtype = jnp.result_type(self.r, self.values)
        fraction = (jnp.asarray(nodes, dtype=dtype) + 1) / 2
        weights = jnp.asarray(weights, dtype=dtype)
        width = jnp.diff(self.r)[:, None]
        r = self.r[:-1, None] + width * fraction
        values = self._profile_values()
        profile = values[:-1, None] * (1 - fraction) + values[1:, None] * fraction
        measure = width * weights / 2 * profile
        return r.reshape(-1), measure.reshape(-1)


class ThinShellWindow(RadialWindow):
    """The distribution amplitude * delta_D(r - radius).

    Its integral is amplitude (default one), which may be signed. A Dirac
    delta has no pointwise profile: use radial_kernel to evaluate its analytic
    kernel amplitude * t(k, radius) * j_ell(k * radius). Its quadrature rule
    is exact: one node and one weight. Radius and amplitude remain
    differentiable leaves.
    """

    radius: jax.Array
    amplitude: jax.Array

    def __init__(self, radius: ArrayLike, amplitude: ArrayLike = 1.0):
        self.radius = _scalar(radius, "radius")
        self.radius = eqx.error_if(
            self.radius, self.radius < 0, "radius must be nonnegative."
        )
        self.amplitude = _scalar(amplitude, "amplitude")

    @property
    def support(self) -> tuple[jax.Array, jax.Array]:
        return self.radius, self.radius

    def quadrature(self, n_r: int = 128) -> tuple[jax.Array, jax.Array]:
        """One exact node at radius, with weight amplitude, independent of n_r."""
        _check_n_r(n_r)
        return self.radius[None], self.amplitude[None]
