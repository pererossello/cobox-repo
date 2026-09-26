"""|k| binning on the rfft half grid, shared by every Fourier-space estimator.

Each stored rfft mode stands for itself and its conjugate partner, except on
the k_last = 0 and k_last = N/2 planes; ``rfft_mode_weights`` restores
full-grid counting. (Box geometry: could live in ``box/_fourier.py``.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from ...box import Box


@dataclass(frozen=True, eq=False)
class KBins:
    """Static |k| bin edges. Bins are [e_i, e_{i+1}), the last one closed.

    Build once and pass to every estimator, so results are comparable and
    stackable (``same_as`` checks this).
    """

    edges: jax.Array  # (n_bins + 1,)
    is_log: bool = False

    # --- constructors ---

    @classmethod
    def log(
        cls,
        box: "Box",
        n_bins: int,
        k_min: Optional[float] = None,
        k_max: Optional[float] = None,
    ) -> "KBins":
        """Log-spaced; defaults k_min = K_RES / 2, k_max = K_NYQ."""
        k_min = 0.5 * box.K_RES if k_min is None else float(k_min)
        k_max = box.K_NYQ if k_max is None else float(k_max)
        _check_range(k_min, k_max, positive=True)
        edges = jnp.logspace(jnp.log10(k_min), jnp.log10(k_max), n_bins + 1)
        return cls(edges=edges, is_log=True)

    @classmethod
    def linear(
        cls,
        box: "Box",
        n_bins: int,
        k_min: Optional[float] = None,
        k_max: Optional[float] = None,
    ) -> "KBins":
        """Linearly spaced; defaults k_min = 0, k_max = K_NYQ."""
        k_min = 0.0 if k_min is None else float(k_min)
        k_max = box.K_NYQ if k_max is None else float(k_max)
        _check_range(k_min, k_max, positive=False)
        return cls(edges=jnp.linspace(k_min, k_max, n_bins + 1), is_log=False)

    @classmethod
    def fundamental(
        cls, box: "Box", width: float = 1.0, k_max: Optional[float] = None
    ) -> "KBins":
        """Linear bins of width ``width * K_RES`` centred on its multiples,
        i.e. edges (i + 1/2) * width * K_RES. The usual bispectrum choice."""
        dk = width * box.K_RES
        k_max = box.K_NYQ if k_max is None else float(k_max)
        n_bins = int(k_max / dk - 0.5)
        if n_bins < 1:
            raise ValueError(f"width={width} leaves no bin below k_max={k_max}.")
        return cls(edges=(jnp.arange(n_bins + 1) + 0.5) * dk, is_log=False)

    # --- properties ---

    @property
    def n_bins(self) -> int:
        return self.edges.shape[0] - 1

    @property
    def centers(self) -> jax.Array:
        lo, hi = self.edges[:-1], self.edges[1:]
        return jnp.sqrt(lo * hi) if self.is_log else 0.5 * (lo + hi)

    def same_as(self, other: "KBins") -> bool:
        return self.edges.shape == other.edges.shape and bool(
            jnp.array_equal(self.edges, other.edges)
        )


def _check_range(k_min: float, k_max: float, positive: bool) -> None:
    if positive and k_min <= 0.0:
        raise ValueError(f"log bins require k_min > 0; got {k_min}.")
    if not k_min < k_max:
        raise ValueError(f"require k_min < k_max; got ({k_min}, {k_max}).")


# -------------------
# --- REDUCTION -----
# -------------------


def rfft_mode_weights(box: "Box") -> jax.Array:
    """(KSHAPE) full-grid multiplicity of each stored rfft mode."""
    k_last = box.k_idx_axes[-1]
    self_paired = (k_last == 0) | (k_last == box.N // 2)
    return jnp.broadcast_to(jnp.where(self_paired, 1.0, 2.0), box.KSHAPE)


def bin_index(bins: KBins, box: "Box") -> jax.Array:
    """(KSHAPE) int bin of each mode; -1 outside the bins and at k = 0."""
    k = box.k
    idx = jnp.digitize(k, bins.edges) - 1
    idx = jnp.where(k == bins.edges[-1], bins.n_bins - 1, idx)  # close last bin
    inside = (idx >= 0) & (idx < bins.n_bins) & (k > 0)
    return jnp.where(inside, idx, -1)


def bin_average(
    values: jax.Array, bins: KBins, box: "Box"
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Mode-weighted per-bin mean of a real (KSHAPE) array.

    The one reduction behind every binned Fourier statistic: auto/cross P,
    multipoles (pass values * (2l+1) L_l(mu)), wedges (values * mask), ...
    Returns (mean, counts, k_eff); counts are full-grid modes, k_eff the
    mode-averaged |k|. Empty bins give NaN mean and k_eff.
    """
    idx = bin_index(bins, box).ravel()
    keep = idx >= 0
    idx = jnp.where(keep, idx, 0)
    w = jnp.where(keep, rfft_mode_weights(box).ravel(), 0.0)

    def total(x: jax.Array) -> jax.Array:
        return jnp.bincount(idx, weights=w * x.ravel(), length=bins.n_bins)

    counts = jnp.bincount(idx, weights=w, length=bins.n_bins)
    safe = jnp.where(counts > 0, counts, 1.0)
    mean = jnp.where(counts > 0, total(values) / safe, jnp.nan)
    k_eff = jnp.where(counts > 0, total(box.k) / safe, jnp.nan)
    return mean, counts, k_eff


# ---------------------
# --- SHELL FILTERS ---
# ---------------------
# Building block beyond two-point (bispectrum, integrated bispectrum,
# position-dependent P): products of shell-filtered real-space fields.


def shell_mask(bins: KBins, box: "Box", i: int) -> jax.Array:
    """(KSHAPE) bool: modes in bin ``i``."""
    return bin_index(bins, box) == i


def shell_filtered(hat: jax.Array, bins: KBins, box: "Box", i: int) -> jax.Array:
    """(*SHAPE) real-space field keeping only the modes of bin ``i``.

    With ``hat = ones`` this is the shell indicator I_i(x) that normalizes
    triangle counts in the bispectrum estimator.
    """
    return box.ifft(jnp.where(shell_mask(bins, box, i), hat, 0.0))
