"""Plotting helpers shared by spherical statistic results."""

from .._paint import PaintResult

__all__ = ["PaintResult", "get_ax"]


def get_ax(ax=None, figsize=(6.5, 4.4)):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax.figure, ax
