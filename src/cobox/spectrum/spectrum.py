"""Spectrum: base class for functions of the wavenumber magnitude k."""

import abc
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


class Spectrum(eqx.Module):
    """A function f(k, *args) of the wavenumber magnitude k.

    Subclasses define __call__ and to_dict. Integration and plotting pass
    positional arguments after k (parameters, times) through to __call__.
    No power-spectrum interpretation is assumed; power statistics belong to
    the PowerSpectrum subclass.
    """

    @abc.abstractmethod
    def __call__(self, k: ArrayLike, *args: Any, **kwargs: Any) -> jax.Array: ...

    @abc.abstractmethod
    def to_dict(self) -> dict:
        """Model configuration; parameters are never included."""

    def integrate(
        self,
        *args,
        weight=None,
        k_min: float = 1e-4,
        k_max: float = 1e2,
        n_k: int = 2048,
    ) -> jax.Array:
        """int f(k) w(k) dln k on a log grid in k.

        weight(k) receives the 1D grid and may return extra trailing axes.
        """
        lnk = jnp.linspace(jnp.log(k_min), jnp.log(k_max), n_k)
        k = jnp.exp(lnk)
        values = self(k, *args)
        if weight is not None:
            w = weight(k)
            values = values.reshape((-1,) + (1,) * (w.ndim - 1)) * w
        return jnp.trapezoid(values, lnk, axis=0)

    def plot(self, k: ArrayLike, *args, ax=None, **kwargs):
        """Log-log plot of f(k, *args); kwargs go to Axes.loglog."""
        from ..field.stats._plot import PaintResult, get_ax

        fig, ax = get_ax(ax)
        ax.loglog(k, self(jnp.asarray(k, dtype=float), *args), **kwargs)
        ax.set_xlabel("k")
        return PaintResult(fig=fig, ax=ax)
