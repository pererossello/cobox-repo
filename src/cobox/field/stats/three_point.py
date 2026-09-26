"""Three-point statistics. Placeholder: result type and signature are fixed,
the estimator is not written.

Same shape as two-point: ``bispectrum(a)`` is the auto case, ``(a, b, c)``
the cross case, bins are a shared KBins (``KBins.fundamental`` usually),
and derived quantities (Q) are properties computed from stored spectra.

Planned estimator (FFT shell filtering; Scoccimarro 2015, Sefusatti+ 2016):

    for each bin i:   a_i(x) = shell_filtered(a_hat, bins, box, i)   (b, c alike)
                      I_i(x) = shell_filtered(ones,  bins, box, i)
    for each closed triplet (i <= j <= l):
        N_ijl = sum_x I_i I_j I_l                       (triangle count)
        B_ijl = norm * sum_x a_i b_j c_l / N_ijl        (derive norm; test on 2LPT)

Memory: n_bins real-space shells of N^D each; loop or chunk for many bins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Union

import jax
import jax.numpy as jnp

from ._binning import KBins
from ._results import Binned

if TYPE_CHECKING:
    from ..scalar import ScalarField
    from .two_point import PowerSpectrum

TriangleConfigLiteral = Literal["all", "equilateral", "isosceles", "squeezed"]


@dataclass(frozen=True, eq=False, kw_only=True)
class Bispectrum(Binned):
    """Binned bispectrum over closed bin triplets (i <= j <= l).

    values, counts: (n_tri,); k_eff: (n_tri, 3).
    """

    triplets: jax.Array  # (n_tri, 3) int bin indices
    config: TriangleConfigLiteral = "all"

    @property
    def k(self) -> jax.Array:
        """(n_tri, 3) bin centers of each triplet."""
        return self.bins.centers[self.triplets]

    def reduced(self, ps: "PowerSpectrum") -> jax.Array:
        """Q = B / (P1 P2 + P2 P3 + P3 P1), with ``ps`` on the same bins."""
        if not ps.bins.same_as(self.bins):
            raise ValueError("reduced() requires a PowerSpectrum on the same bins.")
        p1, p2, p3 = (ps.values[self.triplets[:, a]] for a in range(3))
        return self.values / (p1 * p2 + p2 * p3 + p3 * p1)

    def plot(self, ax=None, **kwargs):
        """Reserved. By config: equilateral -> B(k) stairs; isosceles or
        squeezed -> B vs angle or k3; all -> B (or Q) vs triplet index."""
        raise NotImplementedError


def bispectrum(
    a: "ScalarField",
    b: Optional["ScalarField"] = None,
    c: Optional["ScalarField"] = None,
    *,
    bins: Union[KBins, int] = 16,
    triangles: TriangleConfigLiteral = "all",
    shot_noise: Optional[float] = None,
) -> Bispectrum:
    """Binned B_abc(k1, k2, k3); auto when b and c are None. An int ``bins``
    means ``KBins.fundamental`` with that many K_RES per bin. Poisson shot
    noise needs P as well, so it may end up taking a PowerSpectrum."""
    raise NotImplementedError


def _closed_triplets(bins: KBins, triangles: TriangleConfigLiteral) -> jax.Array:
    """Reserved: (n_tri, 3) bin triplets i <= j <= l whose edges allow a
    closed triangle, filtered by ``triangles``."""
    raise NotImplementedError
