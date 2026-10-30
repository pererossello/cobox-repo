"""Optional-to-import catalogue plots; no figure is created on import.

All plotting is host-side. Angular limits are in degrees, redshift limits in z.
All selected rows are shown by default; scatter views accept an optional display limit.
"""

from ._common import CatalogPlotResult
from .histogram import plot_redshift
from .sky import plot_sky
from .spatial import plot_3d
from .wedge import plot_wedge

__all__ = ["CatalogPlotResult", "plot_3d", "plot_redshift", "plot_sky", "plot_wedge"]
