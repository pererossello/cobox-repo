"""Physical contributions to scalar observables on the sky."""

from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from cosphere.projection import RadialWindow

from ..cosmology import BackgroundCosmo


class ProjectionTerm(eqx.Module):
    """One contribution weight * integral dchi W(chi) X(chi n, a(chi)).

    transfer(k, a, background) maps primordial curvature to X, following
    MatterTransfer's convention: k in h/Mpc, a the scale factor. None means
    unity (project primordial curvature itself). window uses chi in Mpc/h.
    weight is a finite real scalar, e.g. constant bias or a signed coefficient.
    All terms share the same primordial source. Only scalar j_ell projection
    is supported; velocity/RSD require distinct operators, not scalar transfers.
    """

    transfer: Callable[[jax.Array, jax.Array, BackgroundCosmo], jax.Array] | None
    window: RadialWindow
    weight: jax.Array

    def __init__(
        self,
        transfer: Callable | None,
        window: RadialWindow,
        weight: ArrayLike = 1.0,
    ):
        if transfer is not None and not callable(transfer):
            raise TypeError("transfer must be callable or None (unity).")
        if not isinstance(window, RadialWindow):
            raise TypeError("window must be a RadialWindow in comoving Mpc/h.")
        weight = jnp.asarray(weight)
        if weight.ndim != 0 or jnp.iscomplexobj(weight):
            raise ValueError("weight must be a real scalar.")
        self.weight = eqx.error_if(
            weight.astype(jnp.result_type(weight, jnp.float32)),
            ~jnp.isfinite(weight),
            "weight must be finite.",
        )
        self.transfer = transfer
        self.window = window


class Observable(eqx.Module):
    """A sum of projection terms; cross-terms are included in its power.

    Observable(ProjectionTerm(...)) describes a single contribution.
    Observable((term1, term2)) describes their sum, with signed weights kept.
    Terms contain physical models and windows, not quadrature settings or
    precomputed kernels. Cosmology is supplied to AngularPower when evaluating.
    """

    terms: tuple[ProjectionTerm, ...]

    def __init__(self, terms: ProjectionTerm | Sequence[ProjectionTerm]):
        if isinstance(terms, ProjectionTerm):
            terms = (terms,)
        self.terms = tuple(terms)
        if not self.terms or not all(isinstance(t, ProjectionTerm) for t in self.terms):
            raise TypeError("terms must contain at least one ProjectionTerm.")
