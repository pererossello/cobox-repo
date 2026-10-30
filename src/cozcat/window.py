"""Redshift selections W(z), with parameters stored as JAX array leaves."""

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


class RedshiftWindow(eqx.Module):
    """A redshift weight W(z), distinct from an angular window or a theory kernel.

    Subclasses return finite nonnegative weights with the same shape as z. Store
    numerical parameters as array fields so JAX can transform the whole window.
    """

    @abstractmethod
    def __call__(self, z: ArrayLike) -> jax.Array:
        """Evaluate W(z), without normalizing its integral."""
        raise NotImplementedError


class GaussianWindow(RedshiftWindow):
    """W(z) = exp(-0.5 ((z - center) / sigma)**2), with unit peak.

    center and sigma are finite scalars; sigma must be positive. Both parameters
    and z are differentiable. center need not equal the selected bin's mean z.
    """

    center: jax.Array
    sigma: jax.Array

    def __init__(self, center: ArrayLike, sigma: ArrayLike):
        self.center = _scalar(center, "center")
        sigma = _scalar(sigma, "sigma")
        self.sigma = eqx.error_if(sigma, sigma <= 0, "sigma must be positive.")

    def __call__(self, z: ArrayLike) -> jax.Array:
        z = _real(z, "z")
        return jnp.exp(-0.5 * ((z - self.center) / self.sigma) ** 2)


class TopHatWindow(RedshiftWindow):
    """W(z) = 1 for z_min <= z < z_max, otherwise 0.

    Finite scalar bounds must satisfy z_min < z_max. Hard membership changes are
    discontinuous: autodiff gives zero away from the edges, not useful gradients
    for fitting the cut boundaries. Use a smooth window for that purpose.
    """

    z_min: jax.Array
    z_max: jax.Array

    def __init__(self, z_min: ArrayLike, z_max: ArrayLike):
        self.z_min = _scalar(z_min, "z_min")
        z_max = _scalar(z_max, "z_max")
        self.z_max = eqx.error_if(
            z_max, z_max <= self.z_min, "require z_min < z_max."
        )

    def __call__(self, z: ArrayLike) -> jax.Array:
        z = _real(z, "z")
        return ((z >= self.z_min) & (z < self.z_max)).astype(z.dtype)


def _real(value: ArrayLike, name: str) -> jax.Array:
    value = jnp.asarray(value)
    if jnp.iscomplexobj(value):
        raise ValueError(f"{name} must be real.")
    value = value.astype(float)
    return eqx.error_if(value, jnp.any(~jnp.isfinite(value)), f"{name} must be finite.")


def _scalar(value: ArrayLike, name: str) -> jax.Array:
    value = _real(value, name)
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar.")
    return value
