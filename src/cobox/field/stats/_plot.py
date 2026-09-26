from __future__ import annotations

from .._paint import PaintResult

__all__ = ["PaintResult", "get_ax"]


def get_ax(ax=None, figsize: tuple = (6.5, 4.4)):
    """Return (fig, ax), creating them if ``ax`` is None."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax.figure, ax
