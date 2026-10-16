"""Quadrature nodes and weights for radial projection integrals."""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike


@lru_cache(maxsize=32)
def _legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    # Cache only host constants; never put traced physical arrays in a cache.
    return np.polynomial.legendre.leggauss(n)


def gauss_legendre(
    lower: ArrayLike, upper: ArrayLike, n: int
) -> tuple[jax.Array, jax.Array]:
    """n Gauss-Legendre nodes and weights for integral_lower^upper f(x) dx.

    n is static; the bounds may be traced and differentiated.
    """
    lower, upper = jnp.asarray(lower, dtype=float), jnp.asarray(upper, dtype=float)
    nodes, weights = (jnp.asarray(a, dtype=float) for a in _legendre(n))
    half_width = (upper - lower) / 2
    return lower + half_width * (nodes + 1), half_width * weights
