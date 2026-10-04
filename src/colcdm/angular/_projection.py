"""Private adapters from cosmological conventions to numerical projection inputs."""

from collections.abc import Callable
from operator import index

import equinox as eqx
import jax
import jax.numpy as jnp

from ..cosmology import BackgroundCosmo, Cosmology
from ..cosmology._distances import a_of_chi
from ..power import PrimordialSpectrum


def static_integer(value: int, name: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer >= {minimum}.")
    value = index(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return value


class PrimordialPower(eqx.Module):
    """Primordial P_R in (Mpc/h)^3, evaluated at k in h/Mpc."""

    model: PrimordialSpectrum
    cosmology: Cosmology

    def __call__(self, k):
        h = self.cosmology.background.h
        h = eqx.error_if(
            jnp.asarray(h),
            ~jnp.isfinite(h) | (h <= 0),
            "h must be finite and positive.",
        )
        return h**3 * self.model(h * k, self.cosmology.primordial)


class LightConeTransfer(eqx.Module):
    """Bind a physical transfer to a background and one shared distance table."""

    model: Callable
    background: BackgroundCosmo
    a_grid: jax.Array
    chi_grid: jax.Array

    def __call__(self, k, chi):
        a = a_of_chi(chi, self.a_grid, self.chi_grid)
        return self.model(k, a, self.background)
