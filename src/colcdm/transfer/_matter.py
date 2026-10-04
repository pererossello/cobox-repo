"""Matter transfer kinds: present-day T_0(k) for k > 0 [h/Mpc], and their dispatch.

Each kind is a function (background, k, growth) -> T_0(k). Physical fits use
T_0 = (2/5) (k/H0)^2 / Omega_m T(k) g(1), with g(1) the unnormalized growth of
the injected model (g -> a at early times); calibrated emulators may use their own.
"""

from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..cosmology._constants import C_LIGHT, H0
from ..cosmology.background import BackgroundCosmo
from ..growth import Growth
from ..growth._linear import growth_symbolic_pofk
from ._bbks import bbks
from ._eisenstein_hu import eisenstein_hu, eisenstein_hu_nw
from ._symbolic_pofk import eisenstein_hu_nw_transfer, lcdm_logF_fiducial, log10_S

MatterTransferKindLiteral = Literal[
    "symbolic_pofk", "eisenstein_hu", "eisenstein_hu_nw", "bbks"
]
MATTER_TRANSFER_KINDS = ("symbolic_pofk", "eisenstein_hu", "eisenstein_hu_nw", "bbks")


def _poisson(background: BackgroundCosmo, k: jax.Array) -> jax.Array:
    """(2/5) (k/H0)^2 / Omega_m, with k in h/Mpc."""
    return 0.4 * (C_LIGHT / H0 * k) ** 2 / background.Omega_m


def present_symbolic_pofk(
    background: BackgroundCosmo, k: ArrayLike, growth: Growth
) -> jax.Array:
    """
    symbolic_pofk emulator: T_0(k) = g(1) (2/5) (k/H0)^2 / Omega_m T_EH(k) sqrt(F(k) S(k)).

    g(1) is the emulator's own unnormalized symbolic growth at a = 1, which its
    P(k) shape is calibrated against, whatever the injected growth.
    H0 = 1/2998 h/Mpc, as in the emulator.
    """
    c = background
    k = jnp.asarray(k)
    k = eqx.error_if(
        k,
        ~c.is_flat,
        "The symbolic_pofk backend supports only flat cosmologies (Omega_k = 0).",
    )
    tk_eh = eisenstein_hu_nw_transfer(k, c.Omega_m, c.Omega_b, c.h)
    log_F = lcdm_logF_fiducial(k, c.Omega_m, c.Omega_b, c.h)
    log10_S_value = log10_S(k, c.Omega_m, c.Omega_b, c.h, 0.0, c.w0, c.wa)
    poisson = 0.4 * 2998.0**2 * k**2 / c.Omega_m
    g1 = growth_symbolic_pofk(c, 1.0)
    return g1 * poisson * tk_eh * jnp.exp(0.5 * (log_F + jnp.log(10.0) * log10_S_value))


def _physical(transfer: Callable) -> Callable:
    """Present-day kind from a shape T(k, Om, Ob, h) with T -> 1 as k -> 0."""

    def present(background: BackgroundCosmo, k: ArrayLike, growth: Growth) -> jax.Array:
        c = background
        k = jnp.asarray(k)
        shape = transfer(k, c.Omega_m, c.Omega_b, c.h)
        return _poisson(c, k) * shape * growth.unnormalized(1.0, c)

    return present


MATTER_TRANSFER_DISPATCH: dict[
    MatterTransferKindLiteral,
    Callable[[BackgroundCosmo, ArrayLike, Growth], jax.Array],
] = {
    "symbolic_pofk": present_symbolic_pofk,
    "eisenstein_hu": _physical(eisenstein_hu),
    "eisenstein_hu_nw": _physical(eisenstein_hu_nw),
    "bbks": _physical(bbks),
}
