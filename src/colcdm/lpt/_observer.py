import jax
import jax.numpy as jnp


def get_distance_and_n_los(r_vec: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Distance |r| and unit sightline r / |r|; the sightline is zero at r = 0."""
    distance_sq = jnp.sum(r_vec * r_vec, axis=0)
    safe = jnp.sqrt(jnp.where(distance_sq > 0, distance_sq, 1.0))
    distance = jnp.where(distance_sq > 0, safe, 0.0)
    return distance, r_vec / safe[None, ...]
