from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..cosmology.background import BackgroundCosmo
from ..growth import Growth
from ..growth._ode import (
    A_INIT,
    N_STEPS,
    check_a_range,
    hermite_interp,
    linear_rhs,
    rk4_table,
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
    background: BackgroundCosmo,
    growth: Growth = Growth(),
    lpt_growth_kind: LPTGrowthKindLiteral = "ode",
    *,
    order: Literal[1, 2, 3] = 3,
) -> dict:
    """Growth coefficients c_n(a), pointwise in a."""
    a = _validate_a(a)
    c1 = growth.D(a, background)  # D_1(a) / D_1(1)
    if order == 1:
        return {"1": c1}
    r = _ratios(background, a, lpt_growth_kind)
    return {
        k: EDS_COEFF[k] * c1 ** ORDER[k] * r[k] for k in SHAPE_KEYS if ORDER[k] <= order
    }


def rates(
    a: ArrayLike,
    background: BackgroundCosmo,
    growth: Growth = Growth(),
    lpt_growth_kind: LPTGrowthKindLiteral = "ode",
    *,
    order: Literal[1, 2, 3] = 3,
) -> dict:
    """Growth rates f_n = d ln|c_n| / d ln a, pointwise in a."""
    a = _validate_a(a)
    f1 = growth.f(a, background)
    if order == 1:
        return {"1": f1}
    if lpt_growth_kind == "ode":
        _, dr = _ode_solution(background, a)
    else:
        dr = _log_slope(lambda x: _ratios(background, x, lpt_growth_kind), a)
    return {k: ORDER[k] * f1 + dr[k] for k in SHAPE_KEYS if ORDER[k] <= order}


def _validate_a(a: ArrayLike) -> jax.Array:
    a = jnp.asarray(a, dtype=float)
    return eqx.error_if(
        a,
        jnp.any(~jnp.isfinite(a) | (a <= 0)),
        "a must be finite and positive.",
    )


def _ratios(
    c: BackgroundCosmo, a: ArrayLike, lpt_growth_kind: LPTGrowthKindLiteral
) -> dict:
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


def _rhs(c: BackgroundCosmo, ln_a, y):
    """State y = [D1,F1, D2,F2, D3a,F3a, D3b,F3b, D3c],  F = dD/dln a.

    Orders 1, 2, 3a, 3b share the linear growth operator, plus sources.
    """
    D, F = y[0:8:2], y[1:8:2]
    dD, dF = linear_rhs(c, ln_a, (D, F))
    D1, D2 = D[0], D[1]
    source = (
        1.5
        * c.Omega_m_of_a(jnp.exp(ln_a))
        * jnp.array([0.0, -(D1**2), 2.0 * D1**3, 2.0 * (D1 * D2 - D1**3)])
    )
    dD3c = F[0] * D2 - D1 * F[1]  # D3c is first order: dD3c = D1' D2 - D1 D2'
    return jnp.concatenate([jnp.stack([dD, dF + source], axis=1).ravel(), dD3c[None]])


def _ode_solution(
    c: BackgroundCosmo, a, a_init: float = A_INIT, n_steps: int = N_STEPS
):
    """Ratios and their log slopes on a fixed table, for a_init <= a <= 1.

    Cubic Hermite interpolation of log ratios uses the integrated derivatives
    at the nodes. Returned slopes differentiate that same interpolant.
    """
    a = check_a_range(_validate_a(a), a_init)
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
    grid, ys = rk4_table(lambda t, y: _rhs(c, t, y), y0, a_init=a_init, n_steps=n_steps)
    D1, F1, D2, F2, D3a, F3a, D3b, F3b, D3c = ys.T
    D = {"1": D1, "2": D2, "3a": D3a, "3b": D3b, "3c": D3c}
    F = {"1": F1, "2": F2, "3a": F3a, "3b": F3b, "3c": F1 * D2 - D1 * F2}

    ln_a = jnp.log(a)
    ratios, ratio_rates = {}, {}
    for k in SHAPE_KEYS:
        log_ratio = jnp.log(D[k] / (EDS_COEFF[k] * D1 ** ORDER[k]))
        log_slope = F[k] / D[k] - ORDER[k] * F1 / D1
        value, ratio_rates[k] = hermite_interp(grid, log_ratio, log_slope, ln_a)
        ratios[k] = jnp.exp(value)
    return ratios, ratio_rates
