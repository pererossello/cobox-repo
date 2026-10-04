"""Eisenstein & Hu (1998) matter transfer functions, ApJ 496, 605.

Equation numbers refer to that paper. Inputs are Omega_m, Omega_b, h and k in
h/Mpc; the paper works in Mpc^-1, so k is converted once on entry.
Transfers are normalized to T -> 1 as k -> 0.
"""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..cosmology._constants import T_CMB

_THETA = T_CMB / 2.7  # Theta_2.7


def eisenstein_hu(k: ArrayLike, Om, Ob, h) -> jax.Array:
    """Full CDM + baryon transfer with acoustic oscillations (Section 3)."""
    k = jnp.asarray(k) * h  # Mpc^-1
    w_m, w_b = Om * h**2, Ob * h**2
    f_b = Ob / Om
    f_c = 1.0 - f_b

    z_eq = 2.50e4 * w_m * _THETA**-4  # (2)
    k_eq = 7.46e-2 * w_m * _THETA**-2  # (3), Mpc^-1
    b1 = 0.313 * w_m**-0.419 * (1.0 + 0.607 * w_m**0.674)  # (4)
    b2 = 0.238 * w_m**0.223
    z_d = 1291.0 * w_m**0.251 / (1.0 + 0.659 * w_m**0.828) * (1.0 + b1 * w_b**b2)

    def R(z):  # (5)
        return 31.5 * w_b * _THETA**-4 * (z / 1e3) ** -1

    R_d, R_eq = R(z_d), R(z_eq)
    s = (  # (6), sound horizon at the drag epoch, Mpc
        2.0
        / (3.0 * k_eq)
        * jnp.sqrt(6.0 / R_eq)
        * jnp.log((jnp.sqrt(1.0 + R_d) + jnp.sqrt(R_d + R_eq)) / (1.0 + jnp.sqrt(R_eq)))
    )
    k_silk = 1.6 * w_b**0.52 * w_m**0.73 * (1.0 + (10.4 * w_m) ** -0.95)  # (7)

    q = k / (13.41 * k_eq)  # (10)

    def T_tilde(alpha, beta):  # (19), (20)
        L = jnp.log(jnp.e + 1.8 * beta * q)
        C = 14.2 / alpha + 386.0 / (1.0 + 69.9 * q**1.08)
        return L / (L + C * q**2)

    # CDM: (11), (12), (17), (18)
    a1 = (46.9 * w_m) ** 0.670 * (1.0 + (32.1 * w_m) ** -0.532)
    a2 = (12.0 * w_m) ** 0.424 * (1.0 + (45.0 * w_m) ** -0.582)
    alpha_c = a1**-f_b * a2 ** -(f_b**3)
    bc1 = 0.944 / (1.0 + (458.0 * w_m) ** -0.708)
    bc2 = (0.395 * w_m) ** -0.0266
    beta_c = 1.0 / (1.0 + bc1 * (f_c**bc2 - 1.0))
    f = 1.0 / (1.0 + (k * s / 5.4) ** 4)
    T_c = f * T_tilde(1.0, beta_c) + (1.0 - f) * T_tilde(alpha_c, beta_c)

    # Baryons: (14), (15), (21), (22), (23), (24)
    y = (1.0 + z_eq) / (1.0 + z_d)
    sqrt_1py = jnp.sqrt(1.0 + y)
    G = y * (
        -6.0 * sqrt_1py + (2.0 + 3.0 * y) * jnp.log((sqrt_1py + 1.0) / (sqrt_1py - 1.0))
    )
    alpha_b = 2.07 * k_eq * s * (1.0 + R_d) ** -0.75 * G
    beta_b = 0.5 + f_b + (3.0 - 2.0 * f_b) * jnp.sqrt((17.2 * w_m) ** 2 + 1.0)
    beta_node = 8.41 * w_m**0.435
    s_tilde = s / (1.0 + (beta_node / (k * s)) ** 3) ** (1.0 / 3.0)
    j0 = jnp.sinc(k * s_tilde / jnp.pi)  # sin(x) / x
    T_b = (
        T_tilde(1.0, 1.0) / (1.0 + (k * s / 5.2) ** 2)
        + alpha_b / (1.0 + (beta_b / (k * s)) ** 3) * jnp.exp(-((k / k_silk) ** 1.4))
    ) * j0

    return f_b * T_b + f_c * T_c  # (16)


def eisenstein_hu_nw(k: ArrayLike, Om, Ob, h) -> jax.Array:
    """No-wiggle transfer: the zero-baryon form with a baryon-suppressed shape (Section 4.2)."""
    k = jnp.asarray(k) * h  # Mpc^-1
    w_m = Om * h**2
    w_b = Ob * h**2
    f_b = Ob / Om
    s = 44.5 * jnp.log(9.83 / w_m) / jnp.sqrt(1.0 + 10.0 * w_b**0.75)  # (26), Mpc
    alpha_gamma = (  # (31)
        1.0 - 0.328 * jnp.log(431.0 * w_m) * f_b + 0.38 * jnp.log(22.3 * w_m) * f_b**2
    )
    gamma_eff = (
        Om * h * (alpha_gamma + (1.0 - alpha_gamma) / (1.0 + (0.43 * k * s) ** 4))
    )  # (30)
    q = k / h * _THETA**2 / gamma_eff  # (28)
    L0 = jnp.log(2.0 * jnp.e + 1.8 * q)  # (29)
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    return L0 / (L0 + C0 * q**2)
