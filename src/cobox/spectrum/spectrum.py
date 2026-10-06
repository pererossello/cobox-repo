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

    def bind(self, **kwargs: Any) -> eqx.Partial:
        """This spectrum as a function of k alone, the other arguments fixed.

        Arguments bind by keyword, e.g. power.bind(a=1.0, cosmology=cosmo).
        The result is a PyTree, so bound parameters stay traceable under jit.
        """
        return eqx.Partial(self, **kwargs)

    def integrate(
        self,
        *args,
        weight=None,
        k_min: float = 1e-4,
        k_max: float = 1e2,
        n_k: int = 2048,
    ) -> jax.Array:
        """int f(k) w(k) dln k on a log grid in k.

        The spectrum and weight(k) may return a scalar or an array with
        leading axis n_k. Trailing batch axes broadcast against one another
        (right-aligned); the k axis is never treated as a batch axis.
        """
        lnk = jnp.linspace(jnp.log(k_min), jnp.log(k_max), n_k)
        k = jnp.exp(lnk)
        values = jnp.asarray(self(k, *args))
        if values.ndim == 0:
            values = jnp.broadcast_to(values, k.shape)
        elif values.shape[0] != n_k:
            raise ValueError("spectrum must be scalar or have leading axis n_k.")
        if weight is not None:
            w = jnp.asarray(weight(k))
            if w.ndim == 0:
                values = values * w
            else:
                if w.shape[0] != n_k:
                    raise ValueError("weight must be scalar or have leading axis n_k.")
                ndim = max(values.ndim, w.ndim)
                values = values.reshape(
                    (n_k,) + (1,) * (ndim - values.ndim) + values.shape[1:]
                )
                w = w.reshape((n_k,) + (1,) * (ndim - w.ndim) + w.shape[1:])
                values = values * w
        return jnp.trapezoid(values, lnk, axis=0)

    def plot(self, k: ArrayLike, *args, ax=None, **kwargs):
        """Log-log plot of f(k, *args); kwargs go to Axes.loglog."""
        from ..field.stats._plot import PaintResult, get_ax

        fig, ax = get_ax(ax)
        ax.loglog(k, self(jnp.asarray(k, dtype=float), *args), **kwargs)
        ax.set_xlabel("k")
        return PaintResult(fig=fig, ax=ax)
