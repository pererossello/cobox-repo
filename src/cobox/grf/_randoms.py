from typing import Callable, TYPE_CHECKING

import jax
import jax.numpy as jnp

from ..box import _fourier

if TYPE_CHECKING:
    from ..box import Box
    from ..field.scalar import ScalarField

# ------------------------------
# --- REAL-SPACE WHITE NOISE ---
# ------------------------------


def n_dof_randoms_real(box: "Box"):
    return {"u": box.N**box.D}


def get_randoms_real(key: jax.Array, box: "Box") -> dict[str, jax.Array]:
    """iid N(0, 1) of shape (N^D,)"""
    n = n_dof_randoms_real(box)["u"]
    return {"u": jax.random.normal(key, shape=(n,))}


def sample_real(
    randoms: dict[str, jax.Array],
    box: "Box",
    pk_fn: Callable[[jax.Array], jax.Array],
) -> "ScalarField":
    """Periodic-fft real-space-white-noise GRF sampler."""
    from ..field.scalar import ScalarField

    randoms_flat = randoms["u"]
    N, D = box.N, box.D
    n_total = N**D
    if randoms_flat.shape != (n_total,):
        raise ValueError(
            f"expected randoms['u'] shape ({n_total},); got {randoms_flat.shape}."
        )
    randoms_grid = randoms_flat.reshape((N,) * D)
    F = box.fft(randoms_grid)
    k_mag = box.k
    pk_at_k = jnp.where(k_mag > 0, pk_fn(k_mag), 0.0)
    multiplier = jnp.sqrt(pk_at_k * box.INV_CELL_V)
    grf = F * multiplier
    return ScalarField(data=grf, box=box, has_hat=True)


def randoms_from_grf_real(
    field: "ScalarField",
    box: "Box",
    pk_fn: Callable[[jax.Array], jax.Array],
) -> dict:
    """Inverse of ``sample_real``. Returns ``{"u": (N^D,)}`` with mean 0"""
    field_hat = field.fft().data
    k_mag = box.k
    pk_at_k = jnp.where(k_mag > 0, pk_fn(k_mag), 0.0)
    multiplier = jnp.sqrt(pk_at_k * box.INV_CELL_V)
    safe_multiplier = jnp.where(multiplier > 0, multiplier, 1.0)
    field_hat_unscaled = jnp.where(multiplier > 0, field_hat / safe_multiplier, 0.0)
    field_unscaled = box.ifft(field_hat_unscaled)
    return {"u": field_unscaled.ravel()}


# ---------------------------------
# --- FOURIER-SPACE WHITE NOISE ---
# ---------------------------------


def n_dof_randoms_complex_cartesian(box: "Box", mask: jax.Array):
    n = int(_fourier.n_dof(mask, box))
    return {"u": n}


def get_randoms_complex_cartesian(
    key: jax.Array, box: "Box", mask: jax.Array
) -> dict[str, jax.Array]:
    """iid N(0, 1) of shape (n_dof(mask, box),).
    ``mask`` is expected to already exclude k=0 (see ``GRFSampler._get_mask``);
    this function has no k=0-specific behavior of its own.
    """
    n = n_dof_randoms_complex_cartesian(box, mask)["u"]
    return {"u": jax.random.normal(key, shape=(n,))}


def sample_complex_cartesian(
    randoms: dict[str, jax.Array],
    box: "Box",
    mask: jax.Array,
    pk_fn: Callable[[jax.Array], jax.Array],
) -> "ScalarField":
    """Fourier-space white-noise GRF sampler (Cartesian complex base).

    Each free (mask-kept, canonical) mode is drawn directly as a complex
    Gaussian; Hermitian symmetry is enforced by construction via ``pack``.
    """
    from ..field.scalar import ScalarField

    u = randoms["u"]
    n_dof = int(_fourier.n_dof(mask, box))
    if u.shape != (n_dof,):
        raise ValueError(f"expected randoms['u'] shape ({n_dof},); got {u.shape}.")

    raw = _fourier.pack(u, mask, box)

    k_mag = box.k
    pk_at_k = jnp.where(k_mag > 0, pk_fn(k_mag), 0.0)
    weight = _fourier.dof_weight(box)
    weight = jnp.maximum(weight, _fourier.conj_reverse(weight))
    multiplier = jnp.sqrt(pk_at_k * box.V / weight)

    grf = raw * multiplier
    return ScalarField(data=grf, box=box, has_hat=True)


def randoms_from_grf_complex_cartesian(
    field: "ScalarField",
    box: "Box",
    mask: jax.Array,
    pk_fn: Callable[[jax.Array], jax.Array],
) -> dict:
    """Inverse of ``sample_complex_cartesian``. Returns ``{"u": (n_dof(mask, box),)}``."""
    field_hat = field.fft().data
    k_mag = box.k
    pk_at_k = jnp.where(k_mag > 0, pk_fn(k_mag), 0.0)
    weight = _fourier.dof_weight(box)
    weight = jnp.maximum(weight, _fourier.conj_reverse(weight))
    multiplier = jnp.sqrt(pk_at_k * box.V / weight)
    safe_multiplier = jnp.where(multiplier > 0, multiplier, 1.0)
    field_hat_unscaled = jnp.where(multiplier > 0, field_hat / safe_multiplier, 0.0)
    return {"u": _fourier.unpack(field_hat_unscaled, mask, box)}
