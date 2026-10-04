"""Validated covariance square roots for correlated real scalar skies."""

import equinox as eqx
import jax
import jax.numpy as jnp


@jax.custom_jvp
def symmetric_sqrt(matrix):
    """Principal square root of an already validated PSD matrix (batched).

    The derivative solves S dS + dS S = dC. This avoids differentiating
    eigenvectors, which is singular at repeated eigenvalues even for C > 0.
    At a singular covariance the square root is not generally differentiable;
    null/null components use zero, a convention rather than a true derivative.
    """
    values, vectors = jnp.linalg.eigh(matrix)
    roots = jnp.sqrt(jnp.maximum(values, 0))
    return (vectors * roots[..., None, :]) @ jnp.swapaxes(vectors, -1, -2)


@symmetric_sqrt.defjvp
def _sqrt_jvp(primals, tangents):
    (matrix,), (tangent,) = primals, tangents
    root = symmetric_sqrt(matrix)
    values, vectors = jnp.linalg.eigh(matrix)
    roots = jnp.sqrt(jnp.maximum(values, 0))
    transpose = jnp.swapaxes(vectors, -1, -2)
    local = transpose @ tangent @ vectors
    denominator = roots[..., :, None] + roots[..., None, :]
    local = jnp.where(
        denominator > 0, local / jnp.where(denominator > 0, denominator, 1), 0
    )
    return root, vectors @ local @ transpose


def covariance_factor(shell, mask, cl_fn, n_fields):
    """Evaluate and validate C_ell, returning its real symmetric square root.

    Only retained multipoles are validated; excluded multipoles may contain
    undefined spectra. Complex spectra are rejected: real scalar isotropic
    fields require real symmetric C_ell, including signed cross-spectra.
    """
    covariance = jnp.asarray(cl_fn(shell.ell_1D))
    expected = (shell.L, n_fields, n_fields)
    if covariance.shape != expected:
        raise ValueError(f"cl_fn must return shape {expected}; got {covariance.shape}.")
    if jnp.iscomplexobj(covariance):
        raise TypeError("Joint real scalar fields require real covariance spectra.")
    covariance = covariance.astype(jnp.result_type(covariance, jnp.float32))
    keep = jnp.any(mask, axis=-1)
    covariance = jnp.where(keep[:, None, None], covariance, 0)
    covariance = eqx.error_if(
        covariance,
        jnp.any(~jnp.isfinite(covariance)),
        "Covariance must be finite at retained multipoles.",
    )
    # Normalize each ell separately: tolerance is relative to its matrix scale,
    # never an absolute power floor. A zero matrix stays exactly zero.
    scale = jnp.max(jnp.abs(covariance), axis=(-2, -1), keepdims=True)
    safe_scale = jnp.where(scale > 0, scale, 1)
    normalized = covariance / safe_scale
    tolerance = 32 * n_fields * jnp.finfo(covariance.dtype).eps
    transpose = jnp.swapaxes(normalized, -1, -2)
    normalized = eqx.error_if(
        normalized,
        jnp.any(jnp.abs(normalized - transpose) > tolerance),
        "Covariance must be symmetric at retained multipoles.",
    )
    normalized = (normalized + jnp.swapaxes(normalized, -1, -2)) / 2
    eigenvalues = jnp.linalg.eigvalsh(normalized)
    normalized = eqx.error_if(
        normalized,
        jnp.any(eigenvalues < -tolerance),
        "Covariance must be positive semidefinite at retained multipoles.",
    )
    # Only eigenvalues consistent with roundoff are clipped, inside the root.
    # No jitter/noise is added, so exact zero-power and singular cases survive.
    return symmetric_sqrt(normalized) * jnp.sqrt(safe_scale)
