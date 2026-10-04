"""Non-Limber angular spectra of projections of a common isotropic source."""

from collections.abc import Callable, Sequence
from operator import index

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ._kernels import radial_kernel
from .windows import RadialWindow, _real_array


def angular_cl(
    k: ArrayLike,
    pk_fn: Callable[[jax.Array], jax.Array],
    windows: Sequence[RadialWindow],
    ell_max: int,
    *,
    transfers: Sequence[Callable[[jax.Array, jax.Array], jax.Array] | None]
    | None = None,
    n_r: int = 128,
    k_chunk_size: int = 64,
) -> jax.Array:
    """C_ell[i,j] = (2/pi) integral dk k^2 P0(k) K_i(k) conj(K_j(k)).

    Returns shape (ell_max + 1, n_windows, n_windows), including the monopole.
    Diagonal entries are auto-spectra; off-diagonals are cross-spectra. The
    windows and transfers refer to projections of ONE common homogeneous,
    isotropic source with auto-power P0. This does not model independent noise
    or general unequal-time stochastic covariance.

    k is an explicit, strictly increasing positive 1D grid with at least two
    points. pk_fn(k) supplies dimensional, real, nonnegative P0(k), not Delta^2;
    scalar outputs broadcast to the grid. k and radial coordinates must have
    inverse units. transfers optionally supplies one callable (or None) per
    window, following radial_kernel's broadcast contract; None means unity.
    Signed windows/transfers preserve negative cross-spectra. Complex transfers
    give a Hermitian covariance; real inputs give a real symmetric matrix.

    Integration is trapezoidal in dk on the supplied grid, with no extrapolated
    tails. 'Non-Limber' describes the projection formula, not exact numerical
    quadrature. Check k bounds, k spacing and n_r for convergence, especially
    for narrow/disjoint windows. A logarithmic grid can undersample large-k
    oscillations even if it spans a wide range.

    k_chunk_size bounds the number of k nodes in each forward kernel evaluation.
    Kernels are reused across all pairs; a scan accumulates covariance matrices
    without storing the full (ell, k, r) grid. Differentiation can require more
    memory than forward evaluation. ell_max, n_r and k_chunk_size are static;
    physical inputs and grid values remain differentiable.
    """
    if not callable(pk_fn):
        raise TypeError("pk_fn must be callable.")
    windows = tuple(windows)
    if not windows or not all(isinstance(w, RadialWindow) for w in windows):
        raise TypeError(
            "windows must be a nonempty sequence of RadialWindow instances."
        )
    if transfers is None:
        transfers = (None,) * len(windows)
    else:
        transfers = tuple(transfers)
        if len(transfers) != len(windows):
            raise ValueError("transfers must have one entry per window.")
        if not all(t is None or callable(t) for t in transfers):
            raise TypeError("Each transfer must be callable or None.")
    if isinstance(k_chunk_size, bool):
        raise TypeError("k_chunk_size must be a positive integer.")
    k_chunk_size = index(k_chunk_size)
    if k_chunk_size < 1:
        raise ValueError("k_chunk_size must be positive.")
    k = _real_array(k)
    if k.ndim != 1 or k.size < 2:
        raise ValueError("k must be a 1D grid with at least two points.")
    k = eqx.error_if(
        k,
        jnp.any(~jnp.isfinite(k)) | jnp.any(k <= 0) | jnp.any(jnp.diff(k) <= 0),
        "k must be finite, positive and strictly increasing.",
    )
    power = jnp.asarray(pk_fn(k))
    if jnp.iscomplexobj(power):
        raise TypeError("pk_fn must return real source auto-power P0(k).")
    power = jnp.broadcast_to(power, k.shape)
    power = eqx.error_if(
        power,
        jnp.any(~jnp.isfinite(power)) | jnp.any(power < 0),
        "Source auto-power P0(k) must be finite and nonnegative.",
    )

    # Positive nodal trapezoid weights give a covariance Gram matrix and keep
    # it positive semidefinite up to floating-point roundoff.
    half_steps = jnp.diff(k) / 2
    measure = jnp.zeros_like(k).at[:-1].add(half_steps).at[1:].add(half_steps)
    measure = (2 / jnp.pi) * measure * k**2 * power
    chunk = min(k_chunk_size, k.size)
    padding = (-k.size) % chunk
    k_blocks = jnp.pad(k, (0, padding), mode="edge").reshape(-1, chunk)
    w_blocks = jnp.pad(measure, (0, padding)).reshape(-1, chunk)

    def integrate_block(k_block, w_block):
        kernels = jnp.stack(
            [
                radial_kernel(k_block, w, ell_max, transfer=t, n_r=n_r)
                for w, t in zip(windows, transfers, strict=True)
            ]
        )
        kernels = eqx.error_if(
            kernels, jnp.any(~jnp.isfinite(kernels)), "Radial kernels must be finite."
        )
        return jnp.einsum(
            "ilk,jlk,k->lij",
            kernels,
            jnp.conj(kernels),
            w_block,
            precision=jax.lax.Precision.HIGHEST,
        )

    # Evaluate the first block to establish the result's dtype (possibly complex).
    initial = integrate_block(k_blocks[0], w_blocks[0])

    def accumulate(total, block):
        return total + integrate_block(*block), None

    result, _ = jax.lax.scan(accumulate, initial, (k_blocks[1:], w_blocks[1:]))
    return result
