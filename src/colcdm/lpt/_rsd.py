from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp

from cobox.field import ScalarField, VectorField
from ._observer import get_distance_and_n_los


def get_dpsi_r_dlna_and_n_los(
    psi: VectorField,
    dpsi_dlna: VectorField,
    observer: Tuple[float, ...],
) -> tuple[ScalarField, VectorField]:
    """Return (dPsi/dln a) · n_los and the unit line of sight."""

    psi = psi.ifft()
    dpsi_dlna = dpsi_dlna.ifft()

    # Separation from the observer to the displaced position.
    r_vec = psi.box.vec_from_point(observer) + psi.data
    _, n_los = get_distance_and_n_los(r_vec)

    # Project dPsi/dln a onto the line of sight.
    dpsi_r_dlna = jnp.sum(dpsi_dlna.data * n_los, axis=0)

    dpsi_r_dlna = ScalarField(dpsi_r_dlna, box=psi.box)
    n_los = VectorField(n_los, box=psi.box)

    return dpsi_r_dlna, n_los
