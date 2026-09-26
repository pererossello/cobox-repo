from typing import Literal

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..background.cosmology import Cosmology
from ..linear_power.linear_growth import LinearGrowthKindLiteral, LINEAR_GROWTH_DISPATCH

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
) -> dict:

    c = cosmology
    growth = LINEAR_GROWTH_DISPATCH[linear_growth_kind]
    c1 = growth(c, a) / growth(c, 1.0)  # D_1(a) / D_1(1)

    if lpt_growth_kind == "eds":
        r = {k: 1.0 for k in SHAPE_KEYS}
    elif lpt_growth_kind == "fit":
        if not (c.is_flat and c.is_de_Lambda):
            raise ValueError("growth kind 'fit' is calibrated for flat LCDM only")
        Om_a = c.Omega_m_of_a(a)
        r = {k: Om_a ** (-FIT_EXPONENT[k]) for k in SHAPE_KEYS}
    elif lpt_growth_kind == "ode":
        r = _ode_ratios(c, a)

    return {k: EDS_COEFF[k] * c1 ** ORDER[k] * r[k] for k in SHAPE_KEYS}


def rates(
    a: ArrayLike,
    cosmology: Cosmology,
    linear_growth_kind: LinearGrowthKindLiteral = "symbolic_pofk",
    lpt_growth_kind: LPTGrowthKindLiteral = "ode",
) -> dict:

    def log_coefficients(ln_a):
        values = coeffs(jnp.exp(ln_a), cosmology, linear_growth_kind, lpt_growth_kind)
        return {key: jnp.log(jnp.abs(values[key])) for key in SHAPE_KEYS}

    ln_a = jnp.log(jnp.asarray(a, dtype=float))
    if ln_a.ndim != 0:
        raise ValueError("rates expects a scalar scale factor.")
    return jax.jacfwd(log_coefficients)(ln_a)


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


def _ode_ratios(c: "Cosmology", a, a_init: float = 1e-5, n_steps: int = 512) -> dict:
    """Integrate from deep matter domination and return D_n / D_n^EdS."""
    a = jnp.asarray(a, dtype=float)
    ln_a0, ln_a1 = jnp.log(a_init), jnp.log(jnp.max(a))
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
    ln_a = jnp.log(a)

    def pick(col):
        return jnp.interp(ln_a, grid[1:], ys[:, col])

    D1, D2, D3a, D3b, D3c = pick(0), pick(2), pick(4), pick(6), pick(8)
    return {
        "1": jnp.ones_like(D1),
        "2": D2 / (-3 / 7 * D1**2),
        "3a": D3a / (1 / 3 * D1**3),
        "3b": D3b / (-10 / 21 * D1**3),
        "3c": D3c / (1 / 7 * D1**3),
    }
