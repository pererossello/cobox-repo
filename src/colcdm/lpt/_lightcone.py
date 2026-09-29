"""Past lightcone crossing, evaluated at each Lagrangian grid point.

Solve chi(a) = chi_of_varrho(|q + psi(q, a) - observer|).
Both solvers use a fixed number of iterations and support JIT and gradients.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp

from ._observer import get_distance_and_n_los

if TYPE_CHECKING:
    from ..background.cosmology import Cosmology
    from .lpt import LPTBasis

LightconeMethodLiteral = Literal["newton", "iterative"]


def crossing_a(
    lpt: LPTBasis,
    cosmology: Cosmology,
    *,
    observer: tuple[float, ...],
    method: LightconeMethodLiteral = "newton",
    n_iter: int = 3,
) -> jax.Array:
    """Crossing scale factor, one per grid point of the basis box."""
    if method not in ("newton", "iterative"):
        raise ValueError(f"Unknown light-cone method: {method!r}.")
    if isinstance(n_iter, bool) or not isinstance(n_iter, int) or n_iter < 1:
        raise ValueError("n_iter must be an integer >= 1.")

    lpt = lpt.ifft()  # Time-independent shapes: transform once.
    r0 = lpt.box.vec_from_point(observer)

    def iterative(_i, a):
        psi = lpt.get_psi(a, cosmology)
        varrho, _ = get_distance_and_n_los(r0 + psi.data)
        return cosmology.a_of_chi(cosmology.chi_of_varrho(varrho))

    def newton(_i, a):
        psi, dpsi_dlna = lpt.get_psi_and_dpsi_dlna(a, cosmology)
        varrho, n_los = get_distance_and_n_los(r0 + psi.data)
        dvarrho_dlna = jnp.sum(dpsi_dlna.data * n_los, axis=0)

        f = cosmology.chi_of_a(a) - cosmology.chi_of_varrho(varrho)
        df_dln_a = (
            a * cosmology.dchi_da(a) - cosmology.dchi_dvarrho(varrho) * dvarrho_dlna
        )
        return jnp.minimum(a * jnp.exp(-f / df_dln_a), 1.0)

    # Start from the undisplaced grid.
    varrho0, _ = get_distance_and_n_los(r0)
    a0 = cosmology.a_of_chi(cosmology.chi_of_varrho(varrho0))
    step = newton if method == "newton" else iterative
    return jax.lax.fori_loop(0, n_iter, step, a0)
