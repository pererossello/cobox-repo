"""Scatter-paint periodic particle positions. Returns a PaintResult."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    import matplotlib.axes
    import matplotlib.cm
    import matplotlib.figure
    from .particles import Particles


@dataclass(frozen=True)
class PaintResult:
    fig: "matplotlib.figure.Figure | matplotlib.figure.SubFigure"
    ax: "matplotlib.axes.Axes"
    mappable: Optional["matplotlib.cm.ScalarMappable"] = None


CoordsLiteral = Literal["x", "q"]
Style1DLiteral = Literal["rug", "axvlines"]
ColorKey = Union[str, np.ndarray, None]


def _positions(elements: "Particles", coords: CoordsLiteral) -> np.ndarray:
    """(D, n_total) array of periodic Eulerian or Lagrangian positions."""
    if coords == "x":
        return np.asarray(elements.wrapped_position)
    if coords == "q":
        return np.asarray(elements.q)
    raise ValueError(f"coords must be 'x' or 'q'; got {coords!r}.")


def _resolve_color(c: ColorKey, elements: "Particles", axis: Optional[int]):
    """Returns (color_value, is_scalar); is_scalar gates whether a colorbar applies."""
    if c is None:
        return None, False
    if isinstance(c, str) and c == "depth":
        if axis is None:
            raise ValueError("c='depth' is only meaningful for D=3 (a slab axis).")
        return np.asarray(elements.wrapped_position[axis]), True
    if hasattr(c, "shape"):
        return np.asarray(c), True
    return c, False


def paint(
    elements: "Particles",
    ax: "Optional[Axes]" = None,
    *,
    coords: CoordsLiteral = "x",
    style: Style1DLiteral = "rug",
    axis: int = 2,
    slab: Optional[tuple] = None,
    units: bool = False,
    c: ColorKey = None,
    s: float = 3.0,
    alpha: float = 0.5,
    cmap: str = "viridis",
    colorbar: bool = False,
    cbar_label: Optional[str] = None,
    title: Optional[str] = None,
    show_ticks: bool = True,
    figsize: tuple = (5.5, 4.5),
    **scatter_kwargs,
) -> PaintResult:
    """Scatter-paint ``elements``, dispatched by ``box.D``.

    ``coords`` picks wrapped Eulerian (``"x"``) or Lagrangian (``"q"``)
    positions.
    1D: ``style`` picks a rug (``marker="|"``) or ``axvlines``. 2D: scatter
    of (x0, x1). 3D: ``axis``/``slab`` filter particles to a slab (``slab``
    in ``[0, box.N]`` adimensional units by default; pass ``units=True``
    for world coords), then scatter the two remaining axes. ``c`` accepts
    ``None``, ``"depth"`` (3D only -- colour by the slab-axis coordinate),
    a per-particle array, or any matplotlib colour.
    """
    import matplotlib.pyplot as plt

    box = elements.box
    D = box.D
    if D not in (1, 2, 3):
        raise ValueError(f"unsupported D={D}.")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    fig = ax.figure

    pos = _positions(elements, coords)  # (D, n_total)

    if D == 1:
        c_arr, has_scalar_c = _resolve_color(c, elements, None)
        if style == "rug":
            mappable = ax.scatter(
                pos[0],
                np.zeros_like(pos[0]),
                marker="|",
                s=s * 12.0,
                alpha=alpha,
                c=c_arr,
                cmap=cmap if has_scalar_c else None,
                **scatter_kwargs,
            )
        elif style == "axvlines":
            color = c_arr if isinstance(c_arr, str) else None
            ax.vlines(
                pos[0],
                0.0,
                1.0,
                transform=ax.get_xaxis_transform(),
                alpha=alpha,
                color=color,
                **scatter_kwargs,
            )
            mappable, has_scalar_c = None, False
        else:
            raise ValueError(f"unknown 1D style: {style!r}.")
        ax.set_yticks([])
        ax.set_xlim(0.0, box.L)
        ax.set_xlabel("x")

    elif D == 2:
        c_arr, has_scalar_c = _resolve_color(c, elements, None)
        mappable = ax.scatter(
            pos[0],
            pos[1],
            s=s,
            alpha=alpha,
            c=c_arr,
            cmap=cmap if has_scalar_c else None,
            **scatter_kwargs,
        )
        ax.set_xlim(0.0, box.L)
        ax.set_ylim(0.0, box.L)
        ax.set_xlabel("x0")
        ax.set_ylabel("x1")
        ax.set_aspect("equal")

    else:  # D == 3
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1, or 2; got {axis}.")
        if slab is None:
            lo, hi = 0.0, box.L
        else:
            lo, hi = float(slab[0]), float(slab[1])
            if lo >= hi:
                raise ValueError(f"slab must satisfy lo < hi; got ({lo}, {hi}).")
            if not units:
                lo, hi = lo * box.R, hi * box.R
        mask = (pos[axis] >= lo) & (pos[axis] <= hi)
        rem = [a for a in range(3) if a != axis]
        c_arr, has_scalar_c = _resolve_color(c, elements, axis)
        if isinstance(c_arr, np.ndarray):
            c_arr = c_arr[mask]
        mappable = ax.scatter(
            pos[rem[0]][mask],
            pos[rem[1]][mask],
            s=s,
            alpha=alpha,
            c=c_arr,
            cmap=cmap if has_scalar_c else None,
            **scatter_kwargs,
        )
        ax.set_xlim(0.0, box.L)
        ax.set_ylim(0.0, box.L)
        ax.set_xlabel(f"x{rem[0]}")
        ax.set_ylabel(f"x{rem[1]}")
        ax.set_aspect("equal")

    if colorbar and has_scalar_c and mappable is not None:
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
        if cbar_label is not None:
            cbar.set_label(cbar_label)
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        ax.set_title(title)

    return PaintResult(fig=fig, ax=ax, mappable=mappable)
