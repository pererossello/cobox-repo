"""Shared result base and realization stacking.

Results store measured data only; every derived quantity (Delta^2, r, T, Q,
errors) is a property. ``stack`` averages the stored data and derived
quantities follow, e.g. r is computed from mean spectra, never averaged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable, Optional, Sequence, TypeVar, Union

import jax
import jax.numpy as jnp

from ._binning import KBins

if TYPE_CHECKING:
    from ...box import Box

    Reference = Union[jax.Array, "Binned", Callable[[jax.Array], jax.Array]]

R = TypeVar("R")


@dataclass(frozen=True, eq=False, kw_only=True)
class Binned:
    """A statistic measured in |k| bins (or bin tuples, for the bispectrum)."""

    bins: KBins
    values: jax.Array  # (n,) measured statistic; NaN where empty
    counts: jax.Array  # (n,) full-grid modes (or triangles) per entry
    k_eff: jax.Array  # (n,) or (n, 3): mode-averaged |k|
    box: "Box"
    scatter: Optional[jax.Array] = None  # std over realizations (set by stack)
    n_real: int = 1

    @property
    def edges(self) -> jax.Array:
        return self.bins.edges

    @property
    def centers(self) -> jax.Array:
        return self.bins.centers

    def ratio(self, reference: "Reference") -> jax.Array:
        """values / reference: per-entry array, a compatible Binned, or a
        callable of k_eff (e.g. a theory P(k))."""
        if isinstance(reference, Binned):
            self.check_compatible(reference)
            reference = reference.values
        elif callable(reference):
            reference = reference(self.k_eff)
        return self.values / reference

    def check_compatible(self, other: "Binned") -> None:
        if not self.bins.same_as(other.bins):
            raise ValueError("results use different bins.")
        if self.box != other.box:
            raise ValueError("results live on different boxes.")

    @classmethod
    def _stack(cls, results: Sequence["Binned"]) -> "Binned":
        first = results[0]
        for r in results[1:]:
            first.check_compatible(r)
        values = jnp.stack([r.values for r in results])
        n_real = sum(r.n_real for r in results)
        # Weight by n_real so stacking stacks (of stacks) stays exact for the mean.
        weights = jnp.array([r.n_real for r in results], dtype=values.dtype)
        mean = jnp.tensordot(weights, values, axes=1) / weights.sum()
        scatter = values.std(axis=0) if len(results) > 1 else None
        return replace(first, values=mean, scatter=scatter, n_real=n_real)


def stack(results: Sequence[R]) -> R:
    """Mean over realizations, same type as the inputs.

    Binned results gain ``scatter`` (std across the inputs). Bundles such as
    CrossSpectrum stack their components, so derived quantities are
    computed from the stacked data.
    """
    if not results:
        raise ValueError("stack needs at least one result.")
    kind = type(results[0])
    if any(type(r) is not kind for r in results):
        raise TypeError("stack needs results of a single type.")
    return kind._stack(results)
