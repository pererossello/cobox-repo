from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..background.cosmology import Cosmology
from ..linear_power.linear_growth import (
    LinearGrowthKindLiteral,
    LINEAR_GROWTH_DISPATCH,
    growth_factor,
)

LPTGrowthKindLiteral = Literal["eds", "fit", "ode"]

SHAPE_KEYS = ("1", "2", "3a", "3b", "3c")

# EdS coefficients: D_n -> EDS_COEFF[n] * D_1**ORDER[n]
EDS_COEFF = {
    "1": 1.0,
    "2": -3.0 / 7.0,
    "3a": 1.0 / 3.0,
    "3b": -10.0 / 21.0,
    "3c": 1.0 / 7.0,
}
ORDER = {"1": 1, "2": 2, "3a": 3, "3b": 3, "3c": 3}

# D_n / D_n^EdS ~ Omega_m(a)**(-p).
# Fitted against the ODE over 0 < z < 3 in flat LCDM (Omega_m=0.31)
FIT_EXPONENT = {"1": 0.0, "2": 0.007048, "3a": 0.014701, "3b": 0.015204, "3c": 0.014140}


# -------------------
# --- COEFFICIENTS ---
# -------------------


def coeffs(
    a: ArrayLike,
    cosmology: Cosmology,
    linear_growth_kind: LinearGrowthKindLiteral = "symbolic_pofk",
    lpt_growth_kind: LPTGrowthKindLiteral = "ode",
    *,
    order: Literal[1, 2, 3] = 3,
) -> dict:
    """Growth coefficients c_n(a), pointwise in a."""
    a = _validate_a(a)
    c1 = growth_factor(a, cosmology, linear_growth_kind)  # D_1(a) / D_1(1)
    if order == 1:
        return {"1": c1}
    r = _ratios(cosmology, a, lpt_growth_kind)
    return {
        k: EDS_COEFF[k] * c1 ** ORDER[k] * r[k] for k in SHAPE_KEYS if ORDER[k] <= order
    }


def rates(
    a: ArrayLike,
    cosmology: Cosmology,
    linear_growth_kind: LinearGrowthKindLiteral = "symbolic_pofk",
    lpt_growth_kind: LPTGrowthKindLiteral = "ode",
    *,
    order: Literal[1, 2, 3] = 3,
) -> dict:
    """Growth rates f_n = d ln|c_n| / d ln a, pointwise in a."""
    a = _validate_a(a)
    growth = LINEAR_GROWTH_DISPATCH[linear_growth_kind]
    f1 = _log_slope(lambda x: growth(cosmology, x), a)
    if order == 1:
        return {"1": f1}
    if lpt_growth_kind == "ode":
        _, dr = _ode_solution(cosmology, a)
    else:
        dr = _log_slope(lambda x: _ratios(cosmology, x, lpt_growth_kind), a)
    return {k: ORDER[k] * f1 + dr[k] for k in SHAPE_KEYS if ORDER[k] <= order}


def _validate_a(a: ArrayLike) -> jax.Array:
    a = jnp.asarray(a, dtype=float)
    return eqx.error_if(
        a,
        jnp.any(~jnp.isfinite(a) | (a <= 0)),
        "a must be finite and positive.",
    )


def _ratios(c: Cosmology, a: ArrayLike, lpt_growth_kind: LPTGrowthKindLiteral) -> dict:
    """D_n / D_n^EdS at fixed D_1."""
    a = jnp.asarray(a, dtype=float)
    if lpt_growth_kind == "eds":
        return {k: jnp.ones_like(a) for k in SHAPE_KEYS}
    if lpt_growth_kind == "fit":
        a = eqx.error_if(
            a,
            ~(c.is_flat & c.is_de_Lambda),
            "growth kind 'fit' is calibrated for flat LCDM only",
        )
        Om_a = c.Omega_m_of_a(a)
        return {k: Om_a ** (-FIT_EXPONENT[k]) for k in SHAPE_KEYS}
    if lpt_growth_kind == "ode":
        ratios, _ = _ode_solution(c, a)
        return ratios
    raise ValueError(f"Unknown LPT growth kind: {lpt_growth_kind!r}.")


def _log_slope(fn, a: ArrayLike):
    """d ln fn / d ln a for a fn that is pointwise in a (exact, forward mode)."""
    ln_a = jnp.log(jnp.asarray(a, dtype=float))
    _, slope = jax.jvp(
        lambda x: jax.tree.map(jnp.log, fn(jnp.exp(x))),
        (ln_a,),
        (jnp.ones_like(ln_a),),
    )
    return slope


# -----------
# --- ODE ---
# -----------


def _dlnH_dlna(c: "Cosmology", a):
    return 0.5 * jax.grad(lambda ln_a: jnp.log(c.E_sq(jnp.exp(ln_a))))(jnp.log(a))


def _rhs(c: "Cosmology", ln_a, y):
    """State y = [D1,F1, D2,F2, D3a,F3a, D3b,F3b, D3c],  F = dD/dln a."""
    D1, F1, D2, F2, D3a, F3a, D3b, F3b, _D3c = y
    a = jnp.exp(ln_a)
    Om = c.Omega_m_of_a(a)
    drag = 2.0 + _dlnH_dlna(c, a)
    return jnp.array(
        [
            F1,
            1.5 * Om * D1 - drag * F1,
            F2,
            1.5 * Om * D2 - 1.5 * Om * D1**2 - drag * F2,
            F3a,
            1.5 * Om * D3a + 3.0 * Om * D1**3 - drag * F3a,
            F3b,
            1.5 * Om * D3b + 3.0 * Om * (D1 * D2 - D1**3) - drag * F3b,
            F1 * D2 - D1 * F2,  # D3c is first order: dD3c = D1' D2 - D1 D2'
        ]
    )


def _ode_solution(c: "Cosmology", a, a_init: float = 1e-5, n_steps: int = 512):
    """Ratios and their log slopes on a fixed table, for a_init <= a <= 1.

    Cubic Hermite interpolation of log ratios uses the integrated derivatives
    at the nodes. Returned slopes differentiate that same interpolant.
    """
    a = _validate_a(a)
    a = eqx.error_if(
        a,
        jnp.any((a < a_init) | (a > 1.0)),
        f"ODE growth requires {a_init} <= a <= 1.",
    )
    ln_a0, ln_a1 = jnp.log(a_init), 0.0
    ai = a_init
    y0 = jnp.array(
        [
            ai,
            ai,
            -3 / 7 * ai**2,
            -6 / 7 * ai**2,
            1 / 3 * ai**3,
            ai**3,
            -10 / 21 * ai**3,
            -10 / 7 * ai**3,
            1 / 7 * ai**3,
        ]
    )
    grid = jnp.linspace(ln_a0, ln_a1, n_steps + 1)
    h = grid[1] - grid[0]

    def step(y, t):
        k1 = _rhs(c, t, y)
        k2 = _rhs(c, t + h / 2, y + h / 2 * k1)
        k3 = _rhs(c, t + h / 2, y + h / 2 * k2)
        k4 = _rhs(c, t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return y, y

    _, ys = jax.lax.scan(step, y0, grid[:-1])
    ys = jnp.concatenate([y0[None, :], ys], axis=0)
    D1, F1, D2, F2, D3a, F3a, D3b, F3b, D3c = ys.T
    D = {"1": D1, "2": D2, "3a": D3a, "3b": D3b, "3c": D3c}
    F = {"1": F1, "2": F2, "3a": F3a, "3b": F3b, "3c": F1 * D2 - D1 * F2}

    ln_a = jnp.log(a)
    index = jnp.clip(jnp.searchsorted(grid, ln_a, side="right") - 1, 0, n_steps - 1)
    width = grid[index + 1] - grid[index]
    offset = ln_a - grid[index]

    def at_a(values, slopes):
        y0, y1 = values[index], values[index + 1]
        m0, m1 = slopes[index], slopes[index + 1]
        secant = (y1 - y0) / width
        quadratic = (3 * secant - 2 * m0 - m1) / width
        cubic = (m0 + m1 - 2 * secant) / width**2
        value = y0 + offset * (m0 + offset * (quadratic + offset * cubic))
        slope = m0 + offset * (2 * quadratic + 3 * offset * cubic)
        return jnp.exp(value), slope

    ratios, ratio_rates = {}, {}
    for k in SHAPE_KEYS:
        log_ratio = jnp.log(D[k] / (EDS_COEFF[k] * D1 ** ORDER[k]))
        log_slope = F[k] / D[k] - ORDER[k] * F1 / D1
        ratios[k], ratio_rates[k] = at_a(log_ratio, log_slope)
    return ratios, ratio_rates
