"""Redshift histograms of the full selected catalogue."""

import numpy as np

from ._common import finish
from ._selection import select


def plot_redshift(
    catalog,
    *,
    ra_limits=None,
    dec_limits=None,
    z_limits=None,
    weights=None,
    bins=50,
    density=False,
    ax=None,
    **kwargs,
):
    """Plot observed-redshift counts, weighted counts or a probability density.

    Uses ALL selected rows, without display sampling. weights must be a finite
    nonnegative array aligned with the original catalogue. density=True uses
    NumPy's histogram-density convention (unit integral over the bin range).
    Returns CatalogPlotResult with the StepPatch in mappable; style kwargs go
    to ax.stairs. Bin edges should be finite and strictly increasing.
    """
    import matplotlib.pyplot as plt

    indices, _, _, _ = select(catalog, ra_limits, dec_limits, z_limits)
    w = None
    if weights is not None:
        w = np.asarray(weights)
        if w.shape != (len(catalog),) or not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("weights must be finite nonnegative per-object values.")
        w = w[indices]
    z = np.asarray(catalog.z)[indices]
    values, edges = np.histogram(z, bins=bins, weights=w, density=False)
    if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("Histogram bin edges must be finite and strictly increasing.")
    if density:
        if values.sum() <= 0:
            raise ValueError(
                "A density histogram needs positive total weight inside the bins."
            )
        values = values / (values.sum() * np.diff(edges))
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    artist = ax.stairs(values, edges, **kwargs)
    ax.set(
        xlabel="Observed redshift",
        ylabel="Probability density"
        if density
        else ("Counts" if weights is None else "Weighted counts"),
    )
    return finish(ax, artist, indices, len(indices))
