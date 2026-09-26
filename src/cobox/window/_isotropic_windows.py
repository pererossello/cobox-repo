from math import factorial
from typing import Literal, TypeGuard, get_args

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

IsotropicKindLiteral = Literal["radial_tophat", "gaussian"]
ISOTROPIC_KINDS: tuple[str, ...] = get_args(IsotropicKindLiteral)


def is_isotropic_kind(kind: str) -> TypeGuard[IsotropicKindLiteral]:
    return kind in ISOTROPIC_KINDS


# ---------------------
# --- RADIAL TOPHAT ---
# ---------------------


def _even_polynomial(x: jax.Array, coefficients: tuple[float, ...]) -> jax.Array:
    x2 = x * x
    value = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        value = value * x2 + coefficient
    return value


_TOPHAT_2D_SERIES = tuple(
    (-1.0) ** n / (4**n * factorial(n) * factorial(n + 1)) for n in range(15)
)
_TOPHAT_3D_SERIES = tuple(
    (-1.0) ** n * 6 * (n + 1) / factorial(2 * n + 3) for n in range(9)
)


# Chebyshev coefficients of 2 J1(x) / x on [4, 25], generated with
# numpy.polynomial.Chebyshev.interpolate(lambda x: 2*scipy.special.j1(x)/x,
# 32, domain=[4, 25]). A fixed polynomial avoids the unstable float32
# derivative of JAX's Bessel recurrence.
_TOPHAT_2D_MIDDLE = (
    -0.018576135983767013,
    0.029146241966371524,
    -0.02999792586706046,
    0.0142330867381853,
    -0.007233911778668221,
    -0.012217032657800964,
    0.022912732040461486,
    -0.024944729937659473,
    0.023929140739896222,
    0.003657930903447699,
    -0.016303305188535684,
    0.0024961004816347263,
    0.004378827258698429,
    -0.0010561691310998686,
    -0.0007072433471166063,
    0.00020493398118602525,
    7.918826953840889e-05,
    -2.573203570590339e-05,
    -6.618788815660631e-06,
    2.3483151723641556e-06,
    4.32510744869296e-07,
    -1.6555056471670116e-07,
    -2.2813592236367016e-08,
    9.370408873804019e-09,
    9.944884574020467e-10,
    -4.375383199509307e-10,
    -3.648320946330744e-11,
    1.7200347705145238e-11,
    1.142666085116794e-12,
    -5.784127122502652e-13,
    -3.090977443551558e-14,
    1.6834458111192568e-14,
    7.712484143020803e-16,
)


def _tophat_2d_middle_x(x: jax.Array) -> jax.Array:
    # Clenshaw evaluation on the interval [4, 25].
    t = (2.0 * x - 29.0) / 21.0
    b1, b2 = 0.0, 0.0
    for coefficient in reversed(_TOPHAT_2D_MIDDLE[1:]):
        b0 = 2.0 * t * b1 - b2 + coefficient
        b1, b2 = b0, b1
    return t * b1 - b2 + _TOPHAT_2D_MIDDLE[0]


def _tophat_2d_large_x(x: jax.Array) -> jax.Array:
    """Asymptotic J1 expansion, evaluated only for x >= 25."""
    coefficients = [1.0]
    for n in range(1, 12):
        coefficients.append(coefficients[-1] * (4 - (2 * n - 1) ** 2) / (8 * n))

    inverse_x = 1.0 / x
    inverse_x2 = inverse_x * inverse_x
    p, q = 0.0, 0.0
    for n in range(5, -1, -1):
        p = p * inverse_x2 + (-1) ** n * coefficients[2 * n]
        q = q * inverse_x2 + (-1) ** n * coefficients[2 * n + 1]
    q *= inverse_x

    sine, cosine = jnp.sin(x), jnp.cos(x)
    j1 = jnp.sqrt(1.0 / (jnp.pi * x)) * ((sine - cosine) * p + (sine + cosine) * q)
    return 2.0 * j1 * inverse_x


def _radial_tophat_hat(x: jax.Array, D: int) -> jax.Array:
    """Fourier profile of a D-ball top-hat (x = |k| R), W(0) = 1."""
    if D == 1:
        x_safe = jnp.where(x > 1e-3, x, 1.0)
        w = jnp.sin(x_safe) / x_safe
        return jnp.where(x > 1e-3, w, 1.0 - x**2 / 6.0)
    elif D == 2:
        small = _even_polynomial(jnp.where(x <= 4.0, x, 4.0), _TOPHAT_2D_SERIES)
        middle = _tophat_2d_middle_x(jnp.clip(x, 4.0, 25.0))
        large = _tophat_2d_large_x(jnp.where(x >= 25.0, x, 25.0))
        return jnp.where(x <= 4.0, small, jnp.where(x < 25.0, middle, large))
    elif D == 3:
        small = _even_polynomial(jnp.where(x <= 1.0, x, 1.0), _TOPHAT_3D_SERIES)
        x_safe = jnp.where(x > 1.0, x, 1.0)
        w = 3.0 * (jnp.sin(x_safe) - x_safe * jnp.cos(x_safe)) / x_safe**3
        return jnp.where(x <= 1.0, small, w)
    else:
        raise ValueError(f"radial top-hat supports D in (1, 2, 3); got D={D}.")


def _radial_tophat_norm(R: jax.Array, D: int) -> jax.Array:
    if D == 1:
        return 2.0 * R
    if D == 2:
        return jnp.pi * R**2
    if D == 3:
        return (4.0 / 3.0) * jnp.pi * R**3
    raise ValueError(f"radial top-hat supports D in (1, 2, 3); got D={D}.")


def _tophat_hat(scale: ArrayLike, k: jax.Array, D: int, normalized: bool) -> jax.Array:
    R = scale * 0.5  # scale is full width (diameter); radius is half
    w = _radial_tophat_hat(k * R, D)
    return w if normalized else w * _radial_tophat_norm(jnp.asarray(R), D)


def _tophat_real(scale: ArrayLike, r: jax.Array, D: int, normalized: bool) -> jax.Array:
    R = scale * 0.5
    w = jnp.where(r <= R, 1.0, 0.0)  # 1 inside the D-ball, 0 outside
    return w / _radial_tophat_norm(jnp.asarray(R), D) if normalized else w


# ----------------
# --- GAUSSIAN ---
# ----------------


def _gaussian_hat(
    scale: ArrayLike, k: jax.Array, D: int, normalized: bool
) -> jax.Array:
    sigma = scale * 0.5  # scale is full width; sigma is half
    w = jnp.exp(-0.5 * (k * sigma) ** 2)
    return w if normalized else w * (jnp.sqrt(2 * jnp.pi) * sigma) ** D


def _gaussian_real(
    scale: ArrayLike, r: jax.Array, D: int, normalized: bool
) -> jax.Array:
    sigma = scale * 0.5
    w = jnp.exp(-0.5 * (r / sigma) ** 2)
    return w / (jnp.sqrt(2 * jnp.pi) * sigma) ** D if normalized else w


# ---------------------
# --- WIN ISO / HAT ---
# ---------------------

# Each kind registers a real-space and a Fourier-space profile, both with the
# uniform (scale, coord, D, normalized) -> W signature.
_ISO_REAL = {
    "gaussian": _gaussian_real,
    "radial_tophat": _tophat_real,
}
_ISO_HAT = {
    "gaussian": _gaussian_hat,
    "radial_tophat": _tophat_hat,
}


def win_iso(
    kind: IsotropicKindLiteral, scale: ArrayLike, r: jax.Array, D: int, normalized: bool
) -> jax.Array:
    """Isotropic real-space window W(|x|) in D dims, dispatched by kind."""
    try:
        window = _ISO_REAL[kind]
    except KeyError:
        raise ValueError(f"unknown isotropic window kind: {kind!r}")
    return window(scale, jnp.abs(jnp.asarray(r)), D, normalized)  # radial: |x|


def win_hat_iso(
    kind: IsotropicKindLiteral, scale: ArrayLike, k: jax.Array, D: int, normalized: bool
) -> jax.Array:
    """Isotropic Fourier-space window W(|k|) in D dims, dispatched by kind."""
    try:
        window = _ISO_HAT[kind]
    except KeyError:
        raise ValueError(f"unknown isotropic window kind: {kind!r}")
    return window(scale, jnp.abs(jnp.asarray(k)), D, normalized)  # radial: |k|
