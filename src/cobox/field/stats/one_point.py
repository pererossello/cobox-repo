from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import jax
import jax.numpy as jnp

from ._plot import PaintResult, get_ax

if TYPE_CHECKING:
    from ..scalar import ScalarField


# ---------------
# --- RESULTS ---
# ---------------


@dataclass(frozen=True)
class Moments:
    mean: jax.Array
    std: jax.Array
    skewness: jax.Array  # <(d-mean)^3> / std^3
    kurtosis: jax.Array  # <(d-mean)^4> / std^4 - 3  (excess / connected)

    @property
    def var(self) -> jax.Array:
        return self.std**2

    @property
    def S3(self) -> jax.Array:
        """Reduced skewness <d^3>_c / sigma^4 = skewness / std."""
        return self.skewness / self.std

    @property
    def S4(self) -> jax.Array:
        """Reduced kurtosis <d^4>_c / sigma^6 = kurtosis / var."""
        return self.kurtosis / self.var

    def __str__(self) -> str:
        rows = ("mean", "std", "var", "skewness", "kurtosis", "S3", "S4")
        return "Moments\n" + "\n".join(
            f"  {name:<8} = {float(getattr(self, name)):+.4g}" for name in rows
        )


@dataclass(frozen=True)
class PDF:
    values: jax.Array  # (n_bins,)
    edges: jax.Array  # (n_bins + 1,)
    density: bool

    @property
    def centers(self) -> jax.Array:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    def plot(self, ax=None, *, figsize: tuple = (6.5, 4.4), **kwargs) -> PaintResult:
        fig, ax = get_ax(ax, figsize)
        ax.stairs(self.values, self.edges, baseline=None, **kwargs)
        ax.set_ylabel("PDF" if self.density else "counts")
        ax.grid(True, alpha=0.3)
        return PaintResult(fig=fig, ax=ax)


# ------------------
# --- ESTIMATORS ---
# ------------------


def moments(field: "ScalarField") -> Moments:
    """Real-space one-point moments. For a smoothing scale, pass
    ``field.convolve(window)``."""
    d = field.ifft().data
    mean = d.mean()
    dev = d - mean
    var = (dev**2).mean()
    safe = jnp.where(var > 0, var, 1.0)
    skew = jnp.where(var > 0, (dev**3).mean() / safe**1.5, 0.0)
    kurt = jnp.where(var > 0, (dev**4).mean() / safe**2 - 3.0, 0.0)
    return Moments(mean, jnp.sqrt(var), skew, kurt)


def pdf(
    field: "ScalarField",
    n_bins: int = 100,
    range: Optional[tuple[float, float]] = None,
    density: bool = False,
) -> PDF:
    """Histogram of real-space values (counts-in-cells after ``convolve``)."""
    values, edges = jnp.histogram(
        field.ifft().data, bins=n_bins, range=range, density=density
    )
    return PDF(values=values, edges=edges, density=density)
