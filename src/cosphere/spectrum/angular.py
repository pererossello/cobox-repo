"""Angular power spectra C_ell, predicted or measured: data containers, not models."""

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

if TYPE_CHECKING:
    from ..field.stats._plot import PaintResult


class AngularSpectrum(eqx.Module):
    """C_ell for 0 <= ell <= ell_max, including the monopole.

    values has shape (ell_max + 1, ...): the first axis is the multipole. One
    auto- or cross-spectrum is (ell_max + 1,); spectra between n fields are
    (ell_max + 1, n, n). The same container holds predictions and full-sky
    measurements: both are values at every integer ell, without binning.
    Values for one set of parameters or one realization, as a JAX PyTree.
    """

    values: jax.Array

    def __init__(self, values: ArrayLike):
        self.values = jnp.asarray(values)
        self._validate_inputs()
        self.values = self.values.astype(float)

    @property
    def ell_max(self) -> int:
        return self.values.shape[0] - 1

    @property
    def ell(self) -> jax.Array:
        return jnp.arange(self.values.shape[0])

    @property
    def counts(self) -> jax.Array:
        """Full-sky modes per multipole, 2 ell + 1."""
        return 2 * self.ell + 1

    @property
    def scaled(self) -> jax.Array:
        """ell (ell + 1) C_ell / (2 pi), a plotting convention."""
        return self._along_ell(self.ell * (self.ell + 1)) * self.values / (2 * jnp.pi)

    @property
    def correlation(self) -> jax.Array:
        """C_ij / sqrt(C_ii C_jj) for (ell_max + 1, n, n) values; NaN where an
        auto-spectrum is not positive. No clipping."""
        if self.values.ndim != 3 or self.values.shape[1] != self.values.shape[2]:
            raise ValueError(
                "correlation requires values of shape (ell_max + 1, n, n)."
            )
        auto = jnp.diagonal(self.values, axis1=1, axis2=2)
        valid = auto > 0
        root = jnp.sqrt(jnp.where(valid, auto, 1))
        both = valid[:, :, None] & valid[:, None, :]
        return jnp.where(
            both, self.values / (root[:, :, None] * root[:, None, :]), jnp.nan
        )

    def ratio(self, reference: "AngularSpectrum | ArrayLike") -> jax.Array:
        """Divide by another spectrum of the same shape, an array or a scalar.

        Returns NaN wherever the reference is zero (including 0/0).
        """
        if isinstance(reference, AngularSpectrum):
            reference = reference.values
        reference = jnp.asarray(reference)
        if reference.shape not in ((), self.values.shape) or jnp.iscomplexobj(
            reference
        ):
            raise ValueError(
                f"reference must be real and scalar or of shape {self.values.shape}."
            )
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
    ) -> "PaintResult":
        """Plot raw C_ell, scaled C_ell, or a ratio; return a PaintResult.

        (ell_max + 1, n, n) values draw one labelled line per entry i <= j.
        Linear axes by default preserve negative cross-power and the monopole.
        For logarithmic plots, nonpositive points are masked rather than
        plotted as absolute values. ratio_to takes precedence over scaled.
        """
        import numpy as np

        from ..field.stats._plot import PaintResult, get_ax

        if ratio_to is not None:
            y, ylabel = self.ratio(ratio_to), "Ratio to reference"
        elif scaled:
            y, ylabel = self.scaled, r"$\ell(\ell+1)C_\ell/(2\pi)$"
        else:
            y, ylabel = self.values, r"$C_\ell$"
        x, y = np.asarray(self.ell), np.asarray(y)
        if y.ndim == 1:
            lines = [(y, {})]
        elif y.ndim == 3 and y.shape[1] == y.shape[2]:
            n = y.shape[1]
            lines = [
                (y[:, i, j], {"label": f"{i}{j}"})
                for i in range(n)
                for j in range(i, n)
            ]
        else:
            raise ValueError("plot supports values of shape (L,) or (L, n, n).")
        fig, ax = get_ax(ax, figsize)
        for line, label in lines:
            if loglog:
                line = np.where((x > 0) & (line > 0), line, np.nan)
            ax.plot(x, line, **{**label, **kwargs})
        if loglog:
            ax.set_xscale("log")
            ax.set_yscale("log")
        if y.ndim == 3:
            ax.legend()
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(ylabel)
        return PaintResult(fig=fig, ax=ax)

    # ------------------
    # --- VALIDATION ---
    # ------------------

    def _along_ell(self, per_ell: jax.Array) -> jax.Array:
        return per_ell.reshape((-1,) + (1,) * (self.values.ndim - 1))

    def _validate_inputs(self) -> None:
        if self.values.ndim == 0 or self.values.shape[0] == 0:
            raise ValueError(
                f"values must have shape (ell_max + 1, ...); got {self.values.shape}."
            )
        if jnp.iscomplexobj(self.values):
            raise TypeError("angular spectra must be real.")
