from math import isfinite
from numbers import Integral
from typing import TYPE_CHECKING, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ._constants import H0, C_LIGHT

if TYPE_CHECKING:
    from .cosmology import Cosmology


def get_distance_table(
    cosmology: "Cosmology",
    *,
    n_quad: int = 4096,
    a_min: float = 1e-5,
) -> tuple[jax.Array, jax.Array]:
    """Return ascending a_grid and descending chi_grid, ending at (1, 0)."""
    if isinstance(n_quad, bool) or not isinstance(n_quad, Integral) or n_quad < 2:
        raise ValueError("n_quad must be an integer >= 2.")
    if not isfinite(a_min) or not 0 < a_min < 1:
        raise ValueError("a_min must be finite and between 0 and 1 (exclusive).")

    a_grid = jnp.geomspace(a_min, 1.0, n_quad)
    # Set endpoints explicitly so roundoff cannot reject a == a_min or 1.
    a_grid = a_grid.at[0].set(a_min).at[-1].set(1.0)
    integrand = _chi_integrand(cosmology, a_grid)
    integrand = eqx.error_if(
        integrand,
        jnp.any(~jnp.isfinite(integrand) | (integrand <= 0)),
        "The distance table requires a finite, positive expansion rate.",
    )
    da = jnp.diff(a_grid)
    increments = 0.5 * (integrand[:-1] + integrand[1:]) * da
    chi_backwards = jnp.concatenate(
        [jnp.zeros(1, dtype=integrand.dtype), jnp.cumsum(increments[::-1])]
    )
    chi_grid = (C_LIGHT / H0) * chi_backwards[::-1]
    return a_grid, chi_grid


def _interp_checked(
    values: ArrayLike,
    xp: jax.Array,
    fp: jax.Array,
    message: str,
) -> jax.Array:
    array = jnp.asarray(values, dtype=xp.dtype)
    array = cast(
        jax.Array,
        eqx.error_if(
            array,
            jnp.any(~jnp.isfinite(array) | (array < xp[0]) | (array > xp[-1])),
            message,
        ),
    )
    return jnp.interp(array.ravel(), xp, fp).reshape(array.shape)


def _chi_integrand(cosmology: "Cosmology", a: jax.Array) -> jax.Array:
    """-dchi/da in units of c/100: 1 / (a^2 E(a))."""
    return 1.0 / (a ** 2 * cosmology.E(a))


def dchi_da(cosmology: "Cosmology", a: ArrayLike) -> jax.Array:
    """Analytic derivative of chi(a), in Mpc/h."""
    a = jnp.asarray(a, dtype=float)
    return -(C_LIGHT / H0) * _chi_integrand(cosmology, a)


def chi_of_a(a: ArrayLike, a_grid: jax.Array, chi_grid: jax.Array) -> jax.Array:
    """Interpolate chi(a) [Mpc/h] on an existing distance table."""
    return _interp_checked(
        a, a_grid, chi_grid, "a must be finite and within [a_min, 1]."
    )


def a_of_chi(chi: ArrayLike, a_grid: jax.Array, chi_grid: jax.Array) -> jax.Array:
    """Interpolate a(chi), rejecting distances outside the table's range."""
    return _interp_checked(
        chi,
        chi_grid[::-1],
        a_grid[::-1],
        "chi must be finite and within [0, chi(a_min)] in Mpc/h.",
    )


_SERIES_LIMIT = 1e-4


def _nonnegative_distance(value: ArrayLike, name: str) -> jax.Array:
    array = jnp.asarray(value, dtype=float)
    return cast(
        jax.Array,
        eqx.error_if(
            array,
            jnp.any(~jnp.isfinite(array) | (array < 0)),
            f"{name} must be finite and nonnegative, in Mpc/h.",
        ),
    )


def chi_of_varrho(cosmology: "Cosmology", varrho: ArrayLike) -> jax.Array:
    """Isotropic coordinate radius -> radial comoving distance, in Mpc/h.

    R0 = (c/100) / sqrt(|Omega_k|).
    Closed: 2 R0 atan(varrho / (2 R0)); open: 2 R0 atanh(varrho / (2 R0)).
    Open coordinates require varrho < 2 R0. The flat limit is the identity.
    """
    varrho, s = _varrho_curvature(cosmology, varrho)
    series = 1 + s * (1 / 3 + s * (1 / 5 + s * (1 / 7 + s / 9)))
    open_root = jnp.sqrt(jnp.where(s > _SERIES_LIMIT, s, _SERIES_LIMIT))
    closed_root = jnp.sqrt(jnp.where(s < -_SERIES_LIMIT, -s, _SERIES_LIMIT))
    exact = jnp.where(
        s > 0,
        jnp.arctanh(open_root) / open_root,
        jnp.arctan(closed_root) / closed_root,
    )
    return varrho * jnp.where(jnp.abs(s) <= _SERIES_LIMIT, series, exact)


def dchi_dvarrho(cosmology: "Cosmology", varrho: ArrayLike) -> jax.Array:
    """Analytic derivative of chi_of_varrho: 1 / (1 - Omega_k (H0 varrho / 2c)^2)."""
    _, s = _varrho_curvature(cosmology, varrho)
    return 1.0 / (1.0 - s)


def _varrho_curvature(
    cosmology: "Cosmology", varrho: ArrayLike
) -> tuple[jax.Array, jax.Array]:
    """Validated varrho and s = Omega_k (H0 varrho / 2c)^2 = +-(varrho / 2 R0)^2."""
    varrho = _nonnegative_distance(varrho, "varrho")
    s = cosmology.Omega_k * (H0 * varrho / (2 * C_LIGHT)) ** 2
    s = cast(
        jax.Array,
        eqx.error_if(s, jnp.any(s >= 1), "Open geometry requires varrho < 2 R0."),
    )
    return varrho, s


def varrho_of_chi(cosmology: "Cosmology", chi: ArrayLike) -> jax.Array:
    """Radial distance -> isotropic coordinate radius, in Mpc/h.

    Closed: 2 R0 tan(chi / (2 R0)); open: 2 R0 tanh(chi / (2 R0)).
    Closed coordinates require chi < pi R0 (before the antipode).
    """
    chi = _nonnegative_distance(chi, "chi")
    s = cosmology.Omega_k * (H0 * chi / (2 * C_LIGHT)) ** 2
    s = cast(
        jax.Array,
        eqx.error_if(
            s,
            jnp.any(s <= -((jnp.pi / 2) ** 2)),
            "Closed geometry requires chi < pi R0 (before the antipode).",
        ),
    )
    series = 1 + s * (-1 / 3 + s * (2 / 15 + s * (-17 / 315 + s * 62 / 2835)))
    open_root = jnp.sqrt(jnp.where(s > _SERIES_LIMIT, s, _SERIES_LIMIT))
    closed_root = jnp.sqrt(jnp.where(s < -_SERIES_LIMIT, -s, _SERIES_LIMIT))
    exact = jnp.where(
        s > 0,
        jnp.tanh(open_root) / open_root,
        jnp.tan(closed_root) / closed_root,
    )
    return chi * jnp.where(jnp.abs(s) <= _SERIES_LIMIT, series, exact)


def S_k(cosmology: "Cosmology", chi: ArrayLike) -> jax.Array:
    """Transverse comoving distance S_k(chi), in Mpc/h.

    Closed: R0 sin(chi/R0); flat: chi; open: R0 sinh(chi/R0).
    The closed case is restricted to chi < pi R0, before the antipode.
    """
    chi = _nonnegative_distance(chi, "chi")
    t = cosmology.Omega_k * (H0 * chi / C_LIGHT) ** 2
    t = cast(
        jax.Array,
        eqx.error_if(
            t,
            jnp.any(t <= -(jnp.pi ** 2)),
            "Closed geometry requires chi < pi R0 (before the antipode).",
        ),
    )
    series = 1 + t * (1 / 6 + t * (1 / 120 + t * (1 / 5040 + t / 362880)))
    open_root = jnp.sqrt(jnp.where(t > _SERIES_LIMIT, t, _SERIES_LIMIT))
    closed_root = jnp.sqrt(jnp.where(t < -_SERIES_LIMIT, -t, _SERIES_LIMIT))
    exact = jnp.where(
        t > 0,
        jnp.sinh(open_root) / open_root,
        jnp.sin(closed_root) / closed_root,
    )
    return chi * jnp.where(jnp.abs(t) <= _SERIES_LIMIT, series, exact)
