"""Shared Matplotlib result, colours and radial-coordinate preparation."""

from dataclasses import dataclass

import numpy as np

from cosphere.field._paint import PaintResult

from ._selection import sample, select


@dataclass(frozen=True)
class CatalogPlotResult(PaintResult):
    """PaintResult plus original displayed row indices and pre-sampling count.

    For histograms, indices contains every selected row. mappable is the
    scatter collection or the histogram's StepPatch (not a scalar mappable).
    """

    indices: np.ndarray | None = None
    n_selected: int = 0


def prepare(catalog, ra_limits, dec_limits, z_limits, max_points, seed):
    indices, ra, dec, span = select(catalog, ra_limits, dec_limits, z_limits)
    count = len(indices)
    return sample(indices, max_points, seed), count, ra, dec, span


def colors(catalog, c, indices):
    if c is None or isinstance(c, str):
        return np.asarray(catalog.z)[indices] if c == "z" else c
    values = np.asarray(c)
    if values.shape != (len(catalog),) or not np.all(np.isfinite(values)):
        raise ValueError(
            'c must be a colour string, "z", or finite per-object scalar values.'
        )
    return values[indices]


def radial(catalog, indices, radius, background):
    if radius not in ("redshift", "comoving"):
        raise ValueError('radius must be "redshift" or "comoving".')
    if radius == "redshift":
        if background is not None:
            raise ValueError('background is only used with radius="comoving".')
        values = np.asarray(catalog.z)[indices]
        label = "Redshift z"
    else:
        if background is None:
            raise ValueError('radius="comoving" requires background.')
        from cozcat import RedshiftCatalog

        subset = RedshiftCatalog(
            catalog.theta[indices], catalog.phi[indices], catalog.z[indices]
        )
        values = np.asarray(subset.comoving_distance(background))
        label = "Comoving radial distance [Mpc/h]"
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Radial plots require finite, nonnegative radii.")
    return values, label


def finish(ax, artist, indices, n_selected):
    return CatalogPlotResult(ax.figure, ax, artist, indices, n_selected)
