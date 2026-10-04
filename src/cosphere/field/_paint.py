"""Paint a ShellField as a Mollweide image on plain axes. Returns a PaintResult."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.cm
    import matplotlib.figure
    from matplotlib.axes import Axes

    from .shell_field import ShellField

FlipLiteral = Literal["geo", "astro"]

X_MAX, Y_MAX = 2 * sqrt(2), sqrt(2)  # Mollweide ellipse half-axes


@dataclass(frozen=True)
class PaintResult:
    fig: "matplotlib.figure.Figure | matplotlib.figure.SubFigure"
    ax: "matplotlib.axes.Axes"
    mappable: Optional["matplotlib.cm.ScalarMappable"] = None


def mollweide_inverse(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) in the Mollweide plane -> (lat, lon) in rad; NaN outside the ellipse."""
    inside = (x / X_MAX) ** 2 + (y / Y_MAX) ** 2 <= 1.0
    aux = np.arcsin(np.clip(y / Y_MAX, -1, 1))
    lat = np.arcsin(np.clip((2 * aux + np.sin(2 * aux)) / pi, -1, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        lon = pi * x / (X_MAX * np.cos(aux))
    inside &= np.abs(lon) <= pi
    return np.where(inside, lat, np.nan), np.where(inside, lon, np.nan)


def mollweide_forward(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(lat, lon) in rad -> (x, y) in the Mollweide plane."""
    target = pi * np.sin(lat)
    aux = np.asarray(lat, dtype=float).copy()
    for _ in range(50):  # Newton on 2 aux + sin(2 aux) = pi sin(lat)
        denom = 2 + 2 * np.cos(2 * aux)
        step = np.where(denom > 1e-12, (2 * aux + np.sin(2 * aux) - target) / np.where(denom > 1e-12, denom, 1.0), 0.0)
        aux = aux - step
    return X_MAX / pi * lon * np.cos(aux), Y_MAX * np.sin(aux)


def paint(
    field: "ShellField",
    ax: "Optional[Axes]" = None,
    *,
    flip: FlipLiteral = "geo",
    resolution: int = 800,
    figsize: tuple = (8.0, 4.8),
    cmap: str = "viridis",
    colorbar: bool = True,
    title: Optional[str] = None,
    graticule: bool = False,
    log_data: bool = False,
    **kwargs,
) -> PaintResult:
    """Paint ``field`` on ``ax`` (created if ``None``) in Mollweide projection,
    centred on phi = 0. Painted in pixel space (``isht`` is applied first);
    ``log_data`` plots ``log10(data)``. ``flip="geo"`` draws phi increasing to
    the right, ``"astro"`` to the left (the sky seen from inside).
    ``resolution`` is the image width in samples. Extra kwargs go to
    ``ax.imshow``."""
    import matplotlib.pyplot as plt

    if flip not in ("geo", "astro"):
        raise ValueError(f"flip must be 'geo' or 'astro'; got {flip!r}.")
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    fig = ax.figure

    field = field.isht()
    data = np.asarray(field.data)

    sign = 1.0 if flip == "geo" else -1.0
    x = np.linspace(-X_MAX, X_MAX, resolution)
    y = np.linspace(-Y_MAX, Y_MAX, resolution // 2)
    xx, yy = np.meshgrid(x, y)
    lat, lon = mollweide_inverse(xx, yy)
    inside = np.isfinite(lat)

    theta = np.where(inside, 0.5 * pi - lat, 0.0)
    phi = np.where(inside, np.mod(sign * lon, 2 * pi), 0.0)
    pix = np.asarray(field.shell.ang2pix(theta, phi))
    values = data[pix]
    if log_data:
        values = np.log10(np.maximum(values, 1e-30))
    image = np.where(inside, values, np.nan)

    mappable = ax.imshow(
        image,
        origin="lower",
        extent=(-X_MAX, X_MAX, -Y_MAX, Y_MAX),
        cmap=cmap,
        interpolation="nearest",
        **kwargs,
    )
    if graticule:
        _draw_graticule(ax)
    ax.set_aspect("equal")
    ax.set_axis_off()

    if colorbar:
        fig.colorbar(
            mappable, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03, shrink=0.6
        )
    if title:
        ax.set_title(title)

    return PaintResult(fig=fig, ax=ax, mappable=mappable)


def _draw_graticule(ax, step_deg: float = 30.0) -> None:
    style = dict(color="0.5", lw=0.5, alpha=0.7)
    t = np.linspace(-0.5 * pi, 0.5 * pi, 181)
    for lon_deg in np.arange(-180.0, 180.0 + 1e-9, step_deg):
        ax.plot(*mollweide_forward(t, np.full_like(t, np.deg2rad(lon_deg))), **style)
    s = np.linspace(-pi, pi, 361)
    for lat_deg in np.arange(-90.0 + step_deg, 90.0, step_deg):
        ax.plot(*mollweide_forward(np.full_like(s, np.deg2rad(lat_deg)), s), **style)
