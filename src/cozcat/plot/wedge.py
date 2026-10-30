"""Polar redshift wedges."""

import numpy as np

from ._common import colors, finish, prepare, radial


def plot_wedge(
    catalog,
    *,
    ra_limits=None,
    dec_limits=None,
    z_limits=None,
    radius="redshift",
    background=None,
    ax=None,
    c=None,
    max_points=None,
    seed=0,
    **kwargs,
):
    """RA versus radial coordinate in a polar sector, collapsing declination.

    Angular cuts are in degrees, redshift cuts in observed z; all are half-open.
    RA (350, 10) crosses zero. Supply a narrow dec_limits for a survey wedge.
    radius='comoving' uses background and the existing comoving_distance method.
    c accepts a colour, 'z', or scalar values aligned with the original catalogue.
    Scatter kwargs (s, alpha, cmap, ...) pass to Matplotlib. Display sampling is
    reproducible and recorded in the result; max_points=None disables it.
    Returns CatalogPlotResult; never calls show(). Requires polar axes if given.
    The geometric origin stays at radius zero. z_limits also sets the visible
    radial interval (converted to distance for comoving plots), preserving an
    annular sector rather than collapsing its inner boundary to a point.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    indices, count, ra, _, span = prepare(
        catalog, ra_limits, dec_limits, z_limits, max_points, seed
    )
    r, label = radial(catalog, indices, radius, background)
    color = colors(catalog, c, indices)
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 6))
    if ax.name != "polar":
        raise ValueError("plot_wedge requires polar axes.")
    kwargs.setdefault("s", 1)
    kwargs.setdefault("alpha", 0.6)
    kwargs.setdefault("rasterized", True)
    artist = ax.scatter(np.deg2rad(ra[indices]), r, c=color, **kwargs)
    ax.set_thetamin(span[0])
    ax.set_thetamax(span[1])
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{np.rad2deg(x) % 360:g}°")
    )
    ax.set_xlabel("RA [degrees]")
    ax.set_ylabel(label)
    ax.set_rorigin(0)
    if z_limits is not None:
        from cozcat import RedshiftCatalog

        endpoints = RedshiftCatalog(np.zeros(2), np.zeros(2), z_limits)
        bounds, _ = radial(endpoints, np.arange(2), radius, background)
        ax.set_ylim(*bounds)
    else:
        ax.set_ylim(bottom=0)
    return finish(ax, artist, indices, count)
