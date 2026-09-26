from typing import Literal, TypeGuard, get_args

import jax
import jax.numpy as jnp

BSplineKindLiteral = Literal["ngp", "cic", "tsc", "pcs"]
BSPLINE_KINDS: tuple[str, ...] = get_args(BSplineKindLiteral)

SeparableKindLiteral = Literal[BSplineKindLiteral, "gaussian"]
SEPARABLE_KINDS: tuple[str, ...] = get_args(SeparableKindLiteral)


def is_separable_kind(kind: str) -> TypeGuard[SeparableKindLiteral]:
    return kind in SEPARABLE_KINDS


# ----------------
# --- BSPLINES ---
# ----------------

BSPLINE_POWER_DICT: dict[BSplineKindLiteral, int] = {
    "ngp": 1,
    "cic": 2,
    "tsc": 3,
    "pcs": 4,
}


def _cardinal_bspline(u: jax.Array, p: int) -> jax.Array:
    """Centered, unit-integral B-spline formed from p unit-width top-hats."""
    u = jnp.asarray(u, dtype=float)
    if p == 1:
        # Preserve the original half-open top-hat convention: (-0.5, 0.5].
        return jnp.where((u > -0.5) & (u <= 0.5), 1.0, 0.0)
    if p not in (2, 3, 4):
        raise ValueError(f"B-spline power must be 1, 2, 3, or 4; got {p!r}.")

    # Bound every polynomial argument, including inactive jnp.where branches,
    # to avoid overflow and nonfinite gradients far outside the support.
    r = jnp.minimum(jnp.abs(u), 0.5 * p)
    if p == 2:
        return 1.0 - r
    if p == 3:
        return jnp.where(r < 0.5, 0.75 - r**2, 0.5 * (1.5 - r) ** 2)
    return jnp.where(
        r < 1.0,
        2.0 / 3.0 - r**2 + 0.5 * r**3,
        (2.0 - r) ** 3 / 6.0,
    )


# --------------
# --- WIN 1D ---
# --------------


def norm_factor_1D(kind: SeparableKindLiteral, scale: float) -> jax.Array:
    if kind == "gaussian":
        sigma = scale * 0.5
        return jnp.sqrt(2 * jnp.pi) * sigma
    elif kind in BSPLINE_KINDS:
        p = BSPLINE_POWER_DICT[kind]
        return jnp.asarray(scale**p)  # scale is full width
    else:
        raise ValueError(f"unknown window kind: {kind!r}")


def win_1D(
    kind: SeparableKindLiteral, scale: float, x: jax.Array, normalized: bool
) -> jax.Array:
    x = jnp.asarray(x)
    if kind == "gaussian":
        sigma = scale * 0.5
        base = jnp.exp(-0.5 * (x / sigma) ** 2) / (jnp.sqrt(2 * jnp.pi) * sigma)
    elif kind in BSPLINE_KINDS:
        p = BSPLINE_POWER_DICT[kind]
        base = _cardinal_bspline(x / scale, p) / scale
    else:
        raise ValueError(f"unknown window kind: {kind!r}")
    return base if normalized else base * norm_factor_1D(kind, scale)


def win_hat_1D(
    kind: SeparableKindLiteral, scale: float, k: jax.Array, normalized: bool
) -> jax.Array:
    k = jnp.asarray(k)
    if kind == "gaussian":
        sigma = scale * 0.5
        base = jnp.exp(-0.5 * (k * sigma) ** 2)
    elif kind in BSPLINE_KINDS:
        p = BSPLINE_POWER_DICT[kind]
        base = jnp.sinc(0.5 * k * scale / jnp.pi) ** p
    else:
        raise ValueError(f"unknown window kind: {kind!r}")
    return base if normalized else base * norm_factor_1D(kind, scale)
