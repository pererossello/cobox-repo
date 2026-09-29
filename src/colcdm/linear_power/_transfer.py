from typing import Callable, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..background.cosmology import Cosmology
from ._symbolic_pofk import plin_new_emulated

TransferKindLiteral = Literal["symbolic_pofk"]
TRANSFER_KINDS = ("symbolic_pofk",)


def transfer_symbolic_pofk(cosmology: Cosmology, k: ArrayLike) -> jax.Array:

    c = cosmology
    k = jnp.asarray(k)
    k = eqx.error_if(
        k,
        ~c.is_flat,
        "The symbolic_pofk backend supports only flat cosmologies (Omega_k = 0).",
    )

    k_safe = jnp.maximum(k, 1e-30)
    pk = jnp.asarray(
        plin_new_emulated(
            k_safe,
            c.As1e9,
            c.Omega_m,
            c.Omega_b,
            c.h,
            c.n_s,
            w0=c.w0,
            wa=c.wa,
        )
    )
    return jnp.where(k > 0, pk, 0.0)  # type: ignore


TRANSFER_DISPATCH: dict[
    TransferKindLiteral, Callable[[Cosmology, ArrayLike], jax.Array]
] = {
    "symbolic_pofk": transfer_symbolic_pofk,
}
