from jax.typing import ArrayLike


def a_of_z(z: ArrayLike) -> ArrayLike:
    return 1.0 / (1.0 + z)


def z_of_a(a: ArrayLike) -> ArrayLike:
    return 1.0 / a - 1.0
