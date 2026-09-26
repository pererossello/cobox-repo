"""Paint a ScalarField (1D line / 2D imshow / 3D slice). Returns a PaintResult."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    import matplotlib.axes
    import matplotlib.cm
    import matplotlib.figure
    from .scalar import ScalarField


@dataclass(frozen=True)
class PaintResult:
    fig: "matplotlib.figure.Figure | matplotlib.figure.SubFigure"
    ax: "matplotlib.axes.Axes"
    mappable: Optional["matplotlib.cm.ScalarMappable"] = None


def paint(
    field: "ScalarField",
    ax: "Optional[Axes]" = None,
    *,
    axis: int = -1,
    slab: Optional[int] = None,
    width: int = 1,
    figsize: tuple = (5.5, 4.5),
    cmap: str = "viridis",
    colorbar: bool = True,
    title: Optional[str] = None,
    show_ticks: bool = True,
    show_tick_labels: bool = True,
    log_data: bool = False,
    label: Optional[str] = None,
    **kwargs,
) -> PaintResult:
    """Paint ``field`` on ``ax`` (created if ``None``). Painted in real space
    (``ifft`` is applied first); ``log_data`` plots ``log10(data)``. For 3D,
    ``axis``/``slab`` pick the slice (``slab`` defaults to ``box.N // 2``);
    ``width`` averages over that many cells centered on ``slab``, wrapping
    periodically (``width=1`` a single slice, ``width=box.N`` the full-axis
    mean). Extra kwargs go to ``ax.plot`` (1D) or ``ax.imshow`` (2D/3D)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    fig = ax.figure

    field = field.ifft()
    D = field.box.D
    data = np.asarray(field.data)

    def _display(a):
        # log is a display transform: apply AFTER the 3D width mean, not before.
        return np.log10(np.maximum(a, 1e-30)) if log_data else a

    mappable = None
    if D == 1:
        ax.plot(np.asarray(field.box.x_1D), _display(data), label=label, **kwargs)
        if show_tick_labels:
            ax.set_xlabel("$x$")
    elif D == 2:
        L = field.box.L
        mappable = ax.imshow(
            _display(data).T, origin="lower", extent=(0, L, 0, L), cmap=cmap, **kwargs
        )
        if show_tick_labels:
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
    elif D == 3:
        N = field.box.N
        slab = N // 2 if slab is None else slab
        if not 1 <= width <= N:
            raise ValueError(f"width must be in 1..{N}; got {width}.")
        ax_eff = axis % 3
        # mean over `width` cells centered on `slab`, wrapping periodically.
        sel = (slab - width // 2 + np.arange(width)) % N
        slice_2d = np.take(data, sel, axis=ax_eff).mean(axis=ax_eff)
        L = field.box.L
        mappable = ax.imshow(
            _display(slice_2d).T,
            origin="lower",
            extent=(0, L, 0, L),
            cmap=cmap,
            **kwargs,
        )
        if show_tick_labels:
            labels = ["$x$", "$y$", "$z$"]
            rem = [a for a in range(3) if a != ax_eff]
            ax.set_xlabel(labels[rem[0]])
            ax.set_ylabel(labels[rem[1]])
    else:
        raise ValueError(f"unsupported D={D}.")

    if colorbar and mappable is not None:
        fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        ax.set_title(title)

    return PaintResult(fig=fig, ax=ax, mappable=mappable)
