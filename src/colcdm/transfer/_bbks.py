"""BBKS matter transfer: Bardeen, Bond, Kaiser & Szalay (1986), ApJ 304, 15, Eq. (G3),
with the baryon-corrected shape parameter of Sugiyama (1995), ApJS 100, 281.

Input k in h/Mpc. Normalized to T -> 1 as k -> 0.
"""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..cosmology._constants import T_CMB


def bbks(k: ArrayLike, Om, Ob, h) -> jax.Array:
    gamma = Om * h * jnp.exp(-Ob * (1.0 + jnp.sqrt(2.0 * h) / Om))  # Sugiyama shape
    q = jnp.asarray(k) * (T_CMB / 2.7) ** 2 / gamma
    return (
        jnp.log(1.0 + 2.34 * q)
        / (2.34 * q)
        * (1.0 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4)
        ** -0.25
    )
