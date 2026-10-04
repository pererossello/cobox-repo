"""Scalar radial projection kernels, with no cosmological assumptions."""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ._bessel import spherical_jn
from .windows import RadialWindow, _real_array


def radial_kernel(
    k: ArrayLike,
    window: RadialWindow,
    ell_max: int,
    *,
    transfer: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    n_r: int = 128,
) -> jax.Array:
    """K_ell(k) = integral dr W(r) t(k, r) j_ell(k r), for 0 <= ell <= ell_max.

    k is a finite nonnegative scalar or array, in inverse units of the radial
    coordinate. Output shape is (ell_max + 1, *k.shape). The callable transfer
    defaults to one. For every window it receives k[..., None] and a 1D radial
    grid and must return a scalar or values broadcastable to their common
    shape. A thin shell supplies a radial grid of length one.

    The window supplies its nodes and weights via quadrature(n_r). Weights
    already include the radial profile and normalization. The actual number
    of nodes is chosen by the window; a thin shell has one exact node.
    ell_max and n_r are static; window parameters and transfers may be traced.

    Increase n_r and check convergence for oscillatory kernels or structured
    transfers; no fixed resolution guarantees accuracy for arbitrary k/r.
    The intermediate Bessel array has shape (ell_max + 1, *k.shape, n_nodes);
    callers should split large k grids to bound memory. Only scalar j_ell
    projections are implemented here, not velocity/RSD derivative operators.
    """
    if not isinstance(window, RadialWindow):
        raise TypeError("window must be a RadialWindow instance.")
    k = _real_array(k)
    k = eqx.error_if(
        k, jnp.any(~jnp.isfinite(k) | (k < 0)), "k must be finite and nonnegative."
    )
    r, measure = window.quadrature(n_r)
    return _radial_kernel(k, r, measure, ell_max, transfer)


def _radial_kernel(k, r, measure, ell_max, transfer):
    """Evaluate on prepared quadrature nodes; shared by kernels and spectra."""
    kr = k[..., None] * r
    t = 1.0 if transfer is None else transfer(k[..., None], r)
    t = jnp.broadcast_to(jnp.asarray(t), kr.shape)
    return jnp.sum(spherical_jn(kr, ell_max) * (t * measure), axis=-1)
