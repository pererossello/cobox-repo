from __future__ import annotations

import jax
import jax.numpy as jnp

from ...box import _fourier


def _import_jax_finufft():
    try:
        from jax_finufft import nufft2
    except ImportError as e:
        raise ImportError(
            "jax-finufft required; install it with `pip install jax-finufft`."
        ) from e
    return nufft2


def evaluate_spectral(
    values: jax.Array, scaled_pos: jax.Array, *, eps: float = 1e-8
) -> jax.Array:
    """Evaluate the band-limited real interpolant of ``values`` (real-space
    samples on a periodic ``(N,)*D`` grid) at ``scaled_pos`` (query
    positions in index units, shape ``(D, ...)``), via a type-2 NUFFT.
    """
    nufft2 = _import_jax_finufft()
    D = values.ndim
    N = values.shape[0]

    # rfft half-spectrum -> Nyquist split -> full (N+2)^D -> centred ordering
    hat = _fourier.pad_rfft(jnp.fft.rfftn(values), N + 2, D)
    spectrum = jnp.fft.fftshift(_fourier.rfft_to_full(hat)).astype(jnp.complex128)

    u = [(2.0 * jnp.pi * scaled_pos[a] / N).astype(jnp.float64) for a in range(D)]
    coeffs = nufft2(spectrum, *u, iflag=1, eps=eps)
    return jnp.real(coeffs) / N**D
