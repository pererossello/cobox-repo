"""Projection kernels Delta_ell(k): the result of a radial projection."""

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

if TYPE_CHECKING:
    from ..field.stats._plot import PaintResult
    from ..spectrum import AngularSpectrum


class ProjectionKernel(eqx.Module):
    """Delta_ell(k) for 0 <= ell <= ell_max on a 1D k grid.

    values has shape (ell_max + 1, n_k); the first axis is the multipole, as
    in spherical_jn. k is in inverse units of the radial coordinate used to
    compute it. Kernels on the same grid add and subtract, and scale by
    scalars, so contributions compose as density + velocity. Values for one
    set of parameters; differentiate by recomputing inside the traced function.
    k and values are finite and real, using JAX's configured default float.
    Zero or unordered k is valid here; power integration requires positive,
    strictly increasing k.
    """

    k: jax.Array
    values: jax.Array

    def __init__(self, k: ArrayLike, values: ArrayLike):
        self.k = jnp.asarray(k)
        self.values = jnp.asarray(values)
        self._validate_inputs()
        self.k = eqx.error_if(
            self.k.astype(float),
            jnp.any(~jnp.isfinite(self.k) | (self.k < 0)),
            "k must be finite and nonnegative.",
        )
        self.values = eqx.error_if(
            self.values.astype(float),
            jnp.any(~jnp.isfinite(self.values)),
            "kernel values must be finite.",
        )

    @property
    def ell_max(self) -> int:
        return self.values.shape[0] - 1

    @property
    def ell(self) -> jax.Array:
        return jnp.arange(self.values.shape[0])

    # ------------------------
    # --- BASIC ARITHMETIC ---
    # ------------------------

    def __neg__(self):
        return replace(self, values=-self.values)

    def __add__(self, other):
        if not isinstance(other, ProjectionKernel):
            return NotImplemented
        return replace(self, values=self._matching(other) + other.values)

    def __sub__(self, other):
        if not isinstance(other, ProjectionKernel):
            return NotImplemented
        return replace(self, values=self._matching(other) - other.values)

    def __mul__(self, other):
        return replace(self, values=self.values * self._scalar(other))

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return replace(self, values=self.values / self._scalar(other))

    # ---------------------
    # --- ANGULAR POWER ---
    # ---------------------

    def angular_power(
        self,
        power_fn: Callable[[jax.Array], jax.Array],
        other: "ProjectionKernel | None" = None,
    ) -> "AngularSpectrum":
        """C_ell = 4 pi integral dln k Delta^2(k) Delta_ell(k) Delta'_ell(k).

        power_fn(k) is the auto-power P0 of the common source field, in the
        units of k; other=None gives the auto-spectrum. Same calculation as
        PowerSpectrum.angular_power; see cosphere.projection._angular.
        """
        from ._angular import angular_power

        return angular_power(power_fn, self, other)

    # ----------------
    # --- PLOTTING ---
    # ----------------

    def plot(
        self, ell: ArrayLike, ax=None, *, figsize=(6.5, 4.4), **kwargs
    ) -> "PaintResult":
        """Plot Delta_ell(k) for integer multipoles in [0, ell_max]; return a PaintResult."""
        from ..field.stats._plot import PaintResult, get_ax

        ell = np.atleast_1d(ell)
        if ell.ndim != 1 or ell.size == 0:
            raise ValueError("ell must be an integer or a nonempty 1D array of integers.")
        if not np.issubdtype(ell.dtype, np.integer):
            raise TypeError("ell must contain integers.")
        if np.any((ell < 0) | (ell > self.ell_max)):
            raise ValueError(f"ell must be in [0, {self.ell_max}].")
        fig, ax = get_ax(ax, figsize)
        k = np.asarray(self.k)
        for ell_i in ell:
            ax.plot(
                k, np.asarray(self.values[ell_i]), label=rf"$\ell={ell_i}$", **kwargs
            )
        ax.set_xscale("log")
        ax.set_xlabel(r"$k$")
        ax.set_ylabel(r"$\Delta_\ell(k)$")
        ax.legend()
        return PaintResult(fig=fig, ax=ax)

    # ------------------
    # --- VALIDATION ---
    # ------------------

    def _matching(self, other: "ProjectionKernel") -> jax.Array:
        """self.values, checked against other's shape (static) and k (runtime)."""
        if self.values.shape != other.values.shape:
            raise ValueError(
                f"kernel shapes differ: {self.values.shape} and {other.values.shape}."
            )
        return eqx.error_if(
            self.values, jnp.any(self.k != other.k), "kernels must share the k grid."
        )

    @staticmethod
    def _scalar(value: ArrayLike) -> jax.Array:
        value = jnp.asarray(value)
        if value.ndim != 0:
            raise ValueError("kernels scale by scalars only.")
        return value

    def _validate_inputs(self) -> None:
        if jnp.iscomplexobj(self.k) or jnp.iscomplexobj(self.values):
            raise TypeError("k and kernel values must be real.")
        if self.k.ndim != 1 or self.k.size == 0:
            raise ValueError(
                f"k must be a nonempty 1D array; got shape {self.k.shape}."
            )
        if (
            self.values.ndim != 2
            or self.values.shape[0] == 0
            or self.values.shape[1] != self.k.size
        ):
            raise ValueError(
                f"values must have shape (ell_max + 1, {self.k.size}); "
                f"got {self.values.shape}."
            )
