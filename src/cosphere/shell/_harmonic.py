"""Spherical harmonic transforms on HEALPix, backed by s2fft (method "jax").

Our layout is the (L, L) half-plane [ell, m >= 0] of a real field; s2fft
uses the full (L, 2 L - 1) layout with m = -(L - 1) .. L - 1 at column
L - 1 + m. Real fields satisfy f_{l,-m} = (-1)^m conj(f_{lm}).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _import_s2fft():
    try:
        import s2fft
    except ImportError as e:
        raise ImportError(
            "Shell.sht / Shell.isht need s2fft; install it with `pip install s2fft`."
        ) from e
    return s2fft


def half_to_full(flm: jax.Array, L: int) -> jax.Array:
    """(L, L) half-plane -> (L, 2 L - 1), filling m < 0 by conjugate symmetry."""
    sign = (-1.0) ** jnp.arange(1, L)  # m = 1 .. L - 1
    negative = (sign * jnp.conj(flm[:, 1:]))[:, ::-1]  # m = -(L - 1) .. -1
    return jnp.concatenate([negative, flm], axis=1)


def full_to_half(flm: jax.Array, L: int) -> jax.Array:
    return flm[:, L - 1 :]


def sht(f: jax.Array, nside: int, L: int, iter: int) -> jax.Array:
    """Real HEALPix map (NPIX,) -> half-plane (L, L), f_lm = int f Y*_lm dOmega."""
    s2fft = _import_s2fft()
    flm = s2fft.forward(
        f,  # type: ignore
        L=L,
        nside=nside,
        sampling="healpix",
        method="jax",
        reality=True,
        iter=iter,
    )
    return full_to_half(flm, L)  # type: ignore


def isht(flm: jax.Array, nside: int, L: int) -> jax.Array:
    """Half-plane (L, L) -> real HEALPix map (NPIX,)."""
    s2fft = _import_s2fft()
    f = s2fft.inverse(
        half_to_full(flm, L),  # type: ignore
        L=L,
        nside=nside,
        sampling="healpix",
        method="jax",
        reality=True,
    )
    return jnp.real(f)


# -----------------------
# --- REAL DOF PACKING ---
# -----------------------
#
# A kept entry with m = 0 is one real dof, one with m > 0 two (Re, Im). The
# latent vector is [m = 0 entries | Re(m > 0) | Im(m > 0)], each block in
# row-major [ell, m] order. `mask` is static (a concrete bool array).


def _indices(mask) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    mask = np.asarray(mask)
    ell, m = np.nonzero(mask)
    real = m == 0
    return (ell[real], m[real]), (ell[~real], m[~real])


def n_dof(mask) -> int:
    (e0, _), (e1, _) = _indices(mask)
    return int(e0.size + 2 * e1.size)


def pack(u: jax.Array, mask) -> jax.Array:
    """Real dofs (n_dof,) -> half-plane (L, L), zero outside the mask."""
    idx0, idx1 = _indices(mask)
    n0, n1 = idx0[0].size, idx1[0].size
    flm = jnp.zeros(np.shape(mask), dtype=jnp.result_type(u.dtype, jnp.complex64))
    flm = flm.at[idx0].set(u[:n0])
    return flm.at[idx1].set(u[n0 : n0 + n1] + 1j * u[n0 + n1 :])


def unpack(flm: jax.Array, mask) -> jax.Array:
    """Half-plane (L, L) -> real dofs (n_dof,); inverse of ``pack`` on the mask."""
    idx0, idx1 = _indices(mask)
    return jnp.concatenate(
        [jnp.real(flm[idx0]), jnp.real(flm[idx1]), jnp.imag(flm[idx1])]
    )
