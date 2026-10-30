"""Three-dimensional catalogue views."""

import numpy as np

from ._common import colors, finish, prepare, radial


def plot_3d(
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
    """Plot r times the catalogue unit directions on equal-scale Cartesian axes.

    With redshift radius this is a visual embedding, not physical distance.
    With comoving radius r=chi(z) in Mpc/h: Euclidean radial embedding, not a
    faithful spatial geometry for curved cosmologies. No RSD correction is made.
    Reuse ax to overlay shells. Cuts, colours and sampling follow plot_wedge.
    A new plot uses Matplotlib 3D axes; all scatter kwargs are passed through.
    """
    import matplotlib.pyplot as plt

    from cozcat import RedshiftCatalog

    indices, count, _, _, _ = prepare(
        catalog, ra_limits, dec_limits, z_limits, max_points, seed
    )
    r, _label = radial(catalog, indices, radius, background)
    color = colors(catalog, c, indices)
    subset = RedshiftCatalog(
        catalog.theta[indices], catalog.phi[indices], catalog.z[indices]
    )
    xyz = r * np.asarray(subset.unit_vectors)
    if ax is None:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(projection="3d")
    if ax.name != "3d":
        raise ValueError("plot_3d requires 3D axes.")
    kwargs.setdefault("s", 1)
    kwargs.setdefault("alpha", 0.4)
    kwargs.setdefault("rasterized", True)
    artist = ax.scatter(*xyz, c=color, **kwargs)
    unit = "redshift radius" if radius == "redshift" else "Mpc/h"
    ax.set(xlabel=f"x [{unit}]", ylabel=f"y [{unit}]", zlabel=f"z [{unit}]")
    # Match display-box aspect to data ranges, including previous overlays.
    ranges = np.array(
        [np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())]
    )
    ax.set_box_aspect(ranges)
    return finish(ax, artist, indices, count)
