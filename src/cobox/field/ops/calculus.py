from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp

GradientKernelLiteral = Literal["spectral", "fd2", "fd4", "fd6", "fd8"]
LaplacianKernelLiteral = Literal["spectral", "fd2", "fd4", "fd6", "fd8"]

# ----------------
# --- GRADIENT ---
# ----------------


def grad_kernel(
    k_1D: jax.Array,
    R: float,
    kernel: GradientKernelLiteral = "spectral",
) -> jax.Array:
    """Complex 1D Fourier kernel for partial derivative along one axis."""
    if kernel == "spectral":
        return grad_kernel_spectral(k_1D)
    if kernel == "fd2":
        return grad_kernel_fd2(k_1D, R)
    if kernel == "fd4":
        return grad_kernel_fd4(k_1D, R)
    if kernel == "fd6":
        return grad_kernel_fd6(k_1D, R)
    if kernel == "fd8":
        return grad_kernel_fd8(k_1D, R)


def grad_kernel_spectral(k_1D: jax.Array) -> jax.Array:
    # Spectral gradient: i*k, with the Nyquist mode zeroed.
    abs_k = jnp.abs(k_1D)
    return jnp.where(abs_k == jnp.max(abs_k), 0.0, 1j * k_1D)


def grad_kernel_fd2(k_1D: jax.Array, R: float) -> jax.Array:
    # 3-point centered stencil: (1/2R) * [-1, 0, +1]
    return 1j * jnp.sin(k_1D * R) / R


def grad_kernel_fd4(k_1D: jax.Array, R: float) -> jax.Array:
    # 5-point centered stencil: (1/12R) * [+1, -8, 0, +8, -1]
    return 1j * (8 * jnp.sin(k_1D * R) - jnp.sin(2 * k_1D * R)) / (6 * R)


def grad_kernel_fd6(k_1D: jax.Array, R: float) -> jax.Array:
    # 7-point centered stencil: (1/60R) * [-1, +9, -45, 0, +45, -9, +1]
    return (
        1j
        * (45 * jnp.sin(k_1D * R) - 9 * jnp.sin(2 * k_1D * R) + jnp.sin(3 * k_1D * R))
        / (30 * R)
    )


def grad_kernel_fd8(k_1D: jax.Array, R: float) -> jax.Array:
    # 9-point centered stencil:
    # (1/840R) * [+3, -32, +168, -672, 0, +672, -168, +32, -3]
    return (
        1j
        * (
            672 * jnp.sin(k_1D * R)
            - 168 * jnp.sin(2 * k_1D * R)
            + 32 * jnp.sin(3 * k_1D * R)
            - 3 * jnp.sin(4 * k_1D * R)
        )
        / (420 * R)
    )


# -----------------
# --- LAPLACIAN ---
# -----------------


def laplacian_kernel(
    k_1D: jax.Array,
    R: float,
    kernel: LaplacianKernelLiteral = "spectral",
) -> jax.Array:
    """Real 1D Fourier kernel for ∂²/∂x² along one axis.

    The Laplacian in Fourier space sums this over axes: ``sum_a L_a(k_a) * F̂``.
    """
    if kernel == "spectral":
        return laplacian_kernel_spectral(k_1D)
    if kernel == "fd2":
        return laplacian_kernel_fd2(k_1D, R)
    if kernel == "fd4":
        return laplacian_kernel_fd4(k_1D, R)
    if kernel == "fd6":
        return laplacian_kernel_fd6(k_1D, R)
    if kernel == "fd8":
        return laplacian_kernel_fd8(k_1D, R)


def laplacian_kernel_spectral(k_1D: jax.Array) -> jax.Array:
    # Spectral second derivative along one axis: -k^2
    return -(k_1D**2)


def laplacian_kernel_fd2(k_1D: jax.Array, R: float) -> jax.Array:
    # 3-point stencil: (1/R^2) * [1, -2, 1]
    return 2 * (jnp.cos(k_1D * R) - 1) / (R**2)


def laplacian_kernel_fd4(k_1D: jax.Array, R: float) -> jax.Array:
    # 5-point stencil: (1/12R^2) * [-1, 16, -30, 16, -1]
    return (32 * jnp.cos(k_1D * R) - 2 * jnp.cos(2 * k_1D * R) - 30) / (12 * R**2)


def laplacian_kernel_fd6(k_1D: jax.Array, R: float) -> jax.Array:
    # 7-point stencil: (1/180R^2) * [2, -27, 270, -490, 270, -27, 2]
    return (
        540 * jnp.cos(k_1D * R)
        - 54 * jnp.cos(2 * k_1D * R)
        + 4 * jnp.cos(3 * k_1D * R)
        - 490
    ) / (180 * R**2)


def laplacian_kernel_fd8(k_1D: jax.Array, R: float) -> jax.Array:
    # 9-point stencil:
    # (1/5040R^2) * [-9, 128, -1008, 8064, -14350, 8064, -1008, 128, -9]
    return (
        16128 * jnp.cos(k_1D * R)
        - 2016 * jnp.cos(2 * k_1D * R)
        + 256 * jnp.cos(3 * k_1D * R)
        - 18 * jnp.cos(4 * k_1D * R)
        - 14350
    ) / (5040 * R**2)
