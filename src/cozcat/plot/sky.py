"""Catalogue sky scatter, in rectangular coordinates or Mollweide projection."""

import numpy as np

from ._common import colors, finish, prepare


def plot_sky(
    catalog,
    *,
    ra_limits=None,
    dec_limits=None,
    z_limits=None,
    projection="rectangular",
    ax=None,
    c=None,
    max_points=None,
    seed=0,
    **kwargs,
):
    """Scatter sky directions; cuts in degrees/z, with periodic RA selection.

    Rectangular RA increases rightwards, unwrapped across the selected RA seam.
    Mollweide uses RA in [-180, 180) degrees and is centred on RA=0. Positions
    are plotted, not surface-density estimates; rectangular areas are unequal.
    c, sampling and result follow plot_wedge. Extra kwargs go to ax.scatter.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    if projection not in ("rectangular", "mollweide"):
        raise ValueError('projection must be "rectangular" or "mollweide".')
    indices, count, ra, dec, _ = prepare(
        catalog, ra_limits, dec_limits, z_limits, max_points, seed
    )
    color = colors(catalog, c, indices)
    if ax is None:
        _, ax = plt.subplots(
            figsize=(9, 4.5),
            subplot_kw={}
            if projection == "rectangular"
            else {"projection": projection},
        )
    expected = "rectilinear" if projection == "rectangular" else "mollweide"
    if ax.name != expected:
        raise ValueError(f"Expected {expected} axes for this projection.")
    x, y = ra[indices], dec[indices]
    if projection == "mollweide":
        x, y = np.deg2rad((x + 180) % 360 - 180), np.deg2rad(y)
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x % 360:g}"))
    kwargs.setdefault("s", 1)
    kwargs.setdefault("alpha", 0.6)
    kwargs.setdefault("rasterized", True)
    artist = ax.scatter(x, y, c=color, **kwargs)
    ax.set(xlabel="RA [degrees]", ylabel="Dec [degrees]")
    ax.grid(alpha=0.2)
    return finish(ax, artist, indices, count)
