"""Paint a slab of a ProtohaloCatalog: centres (scatter) or Lagrangian spheres
(circles). Returns a PaintResult."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from cobox.field._paint import PaintResult

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from cobox.box import Box

    from ..cosmology.background import BackgroundCosmo
    from .catalog import ProtohaloCatalog

StyleLiteral = Literal["scatter", "circles"]
ColorKey = str | np.ndarray | None


def _resolve_color(c: ColorKey, catalog: ProtohaloCatalog, axis: int):
    """Returns (color_value, is_scalar); is_scalar gates whether a colorbar applies."""
    if c is None:
        return None, False
    if isinstance(c, str):
        if c == "depth":
            return np.asarray(catalog.q[axis]), True
        if c == "M":
            return np.log10(np.asarray(catalog.M)), True
        if c == "a":
            return np.asarray(catalog.a), True
    if hasattr(c, "shape"):
        return np.asarray(c), True
    return c, False


def _slab_distance(x: np.ndarray, lo: float, hi: float, L: float) -> np.ndarray:
    """Periodic distance from coordinates x to the interval [lo, hi]; 0 inside."""
    inside = (x >= lo) & (x <= hi)
    return np.where(inside, 0.0, np.minimum((lo - x) % L, (x - hi) % L))


def paint(
    catalog: ProtohaloCatalog,
    box: Box,
    background: BackgroundCosmo | None = None,
    ax: Axes | None = None,
    *,
    style: StyleLiteral = "scatter",
    axis: int = 2,
    slab: tuple | None = None,
    units: bool = False,
    c: ColorKey = None,
    s: float = 3.0,
    alpha: float = 0.5,
    cmap: str = "viridis",
    colorbar: bool = False,
    cbar_label: str | None = None,
    title: str | None = None,
    show_ticks: bool = True,
    figsize: tuple = (5.5, 4.5),
    **kwargs,
) -> PaintResult:
    """Paint the protohalos of a slab along ``axis`` on the two remaining axes.

    ``slab`` is ``(lo, hi)`` in ``[0, box.N]`` node units by default (pass
    ``units=True`` for Mpc/h); ``None`` keeps the whole box. ``style``:
    ``"scatter"`` marks the centres inside the slab; ``"circles"`` draws each
    Lagrangian sphere that reaches the slab, cut where it is widest, and needs
    ``background`` (for R(M)). A plane, ``lo == hi``, is a valid slab.
    ``c`` accepts ``None``, ``"depth"`` (slab-axis coordinate), ``"M"``
    (log10 mass), ``"a"``, a per-protohalo array, or any matplotlib colour.
    Extra kwargs go to ``ax.scatter`` or the circles' ``PatchCollection``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle

    if box.D != 3 or catalog.q.shape[0] != 3:
        raise ValueError("paint requires a 3D catalog and box.")
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2; got {axis}.")
    if style not in ("scatter", "circles"):
        raise ValueError(f"style must be 'scatter' or 'circles'; got {style!r}.")
    if style == "circles" and background is None:
        raise ValueError("style='circles' needs background, for R(M).")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    fig = ax.figure

    L = box.L
    if slab is None:
        lo, hi = 0.0, L
    else:
        lo, hi = float(slab[0]), float(slab[1])
        if lo > hi:
            raise ValueError(f"slab must satisfy lo <= hi; got ({lo}, {hi}).")
        if not units:
            lo, hi = lo * box.R, hi * box.R

    q = np.asarray(catalog.q) % L
    rem = [a for a in range(3) if a != axis]
    c_arr, has_scalar_c = _resolve_color(c, catalog, axis)

    if style == "scatter":
        mask = (q[axis] >= lo) & (q[axis] <= hi)
        if isinstance(c_arr, np.ndarray):
            c_arr = c_arr[mask]
        mappable = ax.scatter(
            q[rem[0]][mask],
            q[rem[1]][mask],
            s=s,
            alpha=alpha,
            c=c_arr,
            cmap=cmap if has_scalar_c else None,
            **kwargs,
        )
    else:
        R = np.asarray(catalog.R_lag(background))
        d = _slab_distance(q[axis], lo, hi, L)
        mask = d < R
        radius = np.sqrt(R[mask] ** 2 - d[mask] ** 2)
        circles = [
            Circle((x, y), r)
            for x, y, r in zip(q[rem[0]][mask], q[rem[1]][mask], radius)
        ]
        mappable = PatchCollection(circles, facecolor="none", alpha=alpha, **kwargs)
        if has_scalar_c:
            mappable.set_array(c_arr[mask])
            mappable.set_cmap(cmap)
        else:
            mappable.set_edgecolor("k" if c_arr is None else c_arr)
        ax.add_collection(mappable)

    ax.set_xlim(0.0, L)
    ax.set_ylim(0.0, L)
    ax.set_xlabel(f"q{rem[0]}")
    ax.set_ylabel(f"q{rem[1]}")
    ax.set_aspect("equal")

    if colorbar and has_scalar_c:
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
        if cbar_label is not None:
            cbar.set_label(cbar_label)
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        ax.set_title(title)

    return PaintResult(fig=fig, ax=ax, mappable=mappable)
