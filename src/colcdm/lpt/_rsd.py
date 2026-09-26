from typing import Tuple, Callable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from cobox.field import VectorField


def radial_rsd(
    psi: VectorField,
    vel: VectorField,
    observer: Tuple[float, ...],
) -> VectorField:

    box = psi.box

    if len(observer) != box.D:
        raise ValueError(f"observer must have length {box.D}.")
    for coord in observer:
        if not (0.0 <= coord <= 1.0):
            raise ValueError("observer coords must be between 0.0 and 1.0.")

    observer_pos = jnp.asarray(observer) * box.L

    psi = psi.ifft()
    vel = vel.ifft()

    observer_pos = jnp.asarray(observer, dtype=psi.data.dtype) * box.L
    observer_pos = observer_pos.reshape((box.D,) + (1,) * box.D)

    # Separation from the observer to the displaced position.
    q = jnp.stack(box.x_grid, axis=0)
    r = q + psi.data - observer_pos

    # Unit sightline; zero at the observer itself.
    distance_sq = jnp.sum(r * r, axis=0)
    distance = jnp.sqrt(jnp.where(distance_sq > 0, distance_sq, 1.0))
    n = r / distance[None, ...]

    # Add radial component of v / (a H).
    vel_radial = jnp.sum(vel.data * n, axis=0)
    return VectorField(
        psi.data + vel_radial[None, ...] * n,
        box=box,
    )
