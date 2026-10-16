"""Radial projection of a 3D source onto the sphere, with no cosmological assumptions."""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ._bessel import _static_int, spherical_jn
from ._quadrature import gauss_legendre
from .kernel import ProjectionKernel

Amplitude = Callable[[jax.Array, jax.Array], jax.Array]
Radius = Callable[[jax.Array], jax.Array]


class RadialProjector(eqx.Module):
    """Projection kernels Delta_ell(k) = integral dx A(k, x) j_ell^(d)(k r(x)).

    Holds the evaluation grid: k, a finite nonnegative 1D array, and the static
    ell_max. The physics enters through two callables per call:

    - amplitude(k, x): every physical factor, normalization and Jacobian of
      the integration variable x. Called with k of shape (n, 1) and x of
      shape (1, n_q); returns real values broadcastable to (n, n_q).
    - radius(x): the radial coordinate r(x), in inverse units of k. Called
      with x of shape (n_q,); returns the same shape.

    x can be any coordinate (redshift, distance, scale factor). Parameters
    carried by the callables remain differentiable. batch_size is the number
    of k values per Bessel evaluation; intermediate Bessel arrays have shape
    (ell_max + derivative + 1, batch_size, n_q).

    derivative=0, 1 or 2 selects j_ell, j_ell' or j_ell'', differentiated
    with respect to the Bessel argument k r(x), not the coordinate x.
    Derivatives are evaluated directly; no derivatives of the amplitude or
    radius callables, nor integration-by-parts boundary terms, are required.
    """

    k: jax.Array
    ell_max: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)

    def __init__(self, k: ArrayLike, ell_max: int, *, batch_size: int = 64):
        k = jnp.asarray(k)
        if jnp.iscomplexobj(k) or k.ndim != 1 or k.size == 0:
            raise ValueError("k must be a real, nonempty 1D array.")
        k = k.astype(float)
        self.k = eqx.error_if(
            k, jnp.any(~jnp.isfinite(k) | (k < 0)), "k must be finite and nonnegative."
        )
        self.ell_max = _static_int(ell_max, "ell_max")
        self.batch_size = _static_int(batch_size, "batch_size")
        self._validate_inputs()

    def integrate(
        self,
        amplitude: Amplitude,
        radius: Radius,
        bounds: tuple[ArrayLike, ArrayLike],
        n_quad: int = 128,
        *,
        derivative: int = 0,
    ) -> ProjectionKernel:
        """Gauss-Legendre quadrature with n_quad nodes in x over bounds.

        bounds may be traced. Increase n_quad and check convergence for
        oscillatory integrands (large k r over the interval).
        derivative=0, 1 or 2 selects the Bessel argument derivative order.
        """
        nodes, weights = gauss_legendre(*bounds, _static_int(n_quad, "n_quad"))
        return self.integrate_nodes(
            amplitude, radius, nodes, weights, derivative=derivative
        )

    def integrate_nodes(
        self,
        amplitude: Amplitude,
        radius: Radius,
        nodes: ArrayLike,
        weights: ArrayLike,
        *,
        derivative: int = 0,
    ) -> ProjectionKernel:
        """sum_q w_q A(k, x_q) j_ell^(d)(k r(x_q)) on explicit 1D nodes and weights.

        A thin shell at x0 is nodes = [x0], weights = [1], which gives
        A(k, x0) j_ell^(d)(k r(x0)) exactly.
        """
        nodes, weights = jnp.asarray(nodes), jnp.asarray(weights)
        if jnp.iscomplexobj(nodes) or jnp.iscomplexobj(weights):
            raise TypeError("nodes and weights must be real.")
        if nodes.ndim != 1 or weights.shape != nodes.shape or nodes.size == 0:
            raise ValueError(
                "nodes and weights must be nonempty 1D arrays of equal length."
            )
        nodes, weights = nodes.astype(float), weights.astype(float)
        nodes = eqx.error_if(
            nodes,
            jnp.any(~jnp.isfinite(nodes)) | jnp.any(~jnp.isfinite(weights)),
            "nodes and weights must be finite.",
        )
        r = jnp.asarray(radius(nodes))
        if jnp.iscomplexobj(r):
            raise TypeError("radius must be real.")
        r = r.astype(float)
        if r.shape != nodes.shape:
            raise ValueError(
                f"radius(x) must return shape {nodes.shape}; got {r.shape}."
            )
        r = eqx.error_if(
            r,
            jnp.any(~jnp.isfinite(r) | (r < 0)),
            "radius must be finite and nonnegative.",
        )

        # Recompute each block's Bessel array in reverse mode instead of storing
        # it, so gradients keep the forward memory bound.
        @jax.checkpoint
        def block(k):
            kr = k[:, None] * r
            a = jnp.asarray(amplitude(k[:, None], nodes[None, :]))
            if jnp.iscomplexobj(a):
                raise TypeError("amplitude must be real.")
            a = jnp.broadcast_to(a.astype(float), kr.shape)
            j = spherical_jn(kr, self.ell_max, derivative)
            return jnp.einsum(
                "lbq,bq,q->lb", j, a, weights, precision=jax.lax.Precision.HIGHEST
            )

        # Pad k to whole blocks; each block is one Bessel evaluation, so its
        # branches are chosen per block rather than per element.
        n_k = self.k.size
        size = min(self.batch_size, n_k)
        padded = jnp.pad(self.k, (0, (-n_k) % size), mode="edge")
        values = jax.lax.map(block, padded.reshape(-1, size))
        values = jnp.moveaxis(values, 0, 1).reshape(self.ell_max + 1, -1)[:, :n_k]
        return ProjectionKernel(self.k, values)

    # ------------------
    # --- VALIDATION ---
    # ------------------

    def _validate_inputs(self) -> None:
        if self.ell_max < 0:
            raise ValueError(f"ell_max must be nonnegative; got {self.ell_max}.")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive; got {self.batch_size}.")
