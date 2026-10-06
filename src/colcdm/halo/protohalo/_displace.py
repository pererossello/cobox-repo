"""Protohalos -> halos: LPT displacement of each Lagrangian sphere."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from cobox.field import VectorField
from cobox.field.ops.interpolation import InterpolationMethodLiteral
from cobox.window import Window

from ...lpt._observer import get_distance_and_n_los
from ...lpt.lpt import LPTBasis
from ..catalog import HaloCatalog

if TYPE_CHECKING:
    from ...cosmology.background import BackgroundCosmo
    from .catalog import ProtohaloCatalog

# Relative slack on the R_grid ends, for round-off in R_lag(M_of_R_lag(R)).
_R_TOL = 1e-6


def displace(
    catalog: ProtohaloCatalog,
    basis: LPTBasis,
    background: BackgroundCosmo,
    *,
    observer: tuple[float, ...] | None = None,
    R_grid: Sequence[float] | None = None,
    interpolation: InterpolationMethodLiteral = "linear",
    n_iter: int = 3,
) -> HaloCatalog:
    """Halos at q + Psi_h(a), Psi_h(a) = sum_n c_n(a) <s_n>_h, with <s_n>_h
    the LPT shape s_n averaged over the protohalo's sphere (or read at its
    centre). See ProtohaloCatalog.displace."""
    if not isinstance(basis, LPTBasis):
        raise TypeError("basis must be an LPTBasis.")
    if basis.box.D != catalog.q.shape[0]:
        raise ValueError(
            f"basis is {basis.box.D}D but the catalogue is {catalog.q.shape[0]}D."
        )

    # Time-independent shapes of each sphere, one (D, n) per LPT term.
    keys = [k for k in basis.lpt.coeffs(1.0, background)]
    keys = [k for k in keys if getattr(basis.shapes, f"s_{k}") is not None]
    shapes = [getattr(basis.shapes, f"s_{k}") for k in keys]
    if R_grid is None:
        means = [s.interpolate_at(catalog.q, method=interpolation) for s in shapes]
    else:
        means = _sphere_means(catalog, shapes, background, R_grid, interpolation)

    def psi_and_dpsi_dlna(a):
        c = basis.lpt.coeffs(a, background)
        f = basis.lpt.rates(a, background)
        psi = sum(c[k] * m for k, m in zip(keys, means))
        dpsi = sum(c[k] * f[k] * m for k, m in zip(keys, means))
        return psi, dpsi

    a = catalog.a
    observer_x = None
    if observer is not None:
        a = _crossing(catalog.q, basis, background, observer, psi_and_dpsi_dlna, n_iter)
        observer_x = tuple(float(o) * basis.box.L for o in observer)
    psi, dpsi = psi_and_dpsi_dlna(a)
    v = a * background.H(a) / background.h * dpsi  # a dx/dt in km/s
    return HaloCatalog(x=catalog.q + psi, M=catalog.M, a=a, v=v, observer=observer_x)


def _crossing(q, basis, background, observer, psi_and_dpsi_dlna, n_iter):
    """Per-halo light-cone crossing, chi(a) = chi_of_varrho(|q + Psi_h(a) - obs|).

    Newton in ln a from the undisplaced crossing, as LPTBasis.get_a_lc on the
    grid; observer is a fraction of the basis box, as there.
    """
    if len(observer) != q.shape[0] or not all(0.0 <= o <= 1.0 for o in observer):
        raise ValueError(f"observer must be {q.shape[0]} box fractions in [0, 1].")
    if isinstance(n_iter, bool) or not isinstance(n_iter, int) or n_iter < 1:
        raise ValueError("n_iter must be an integer >= 1.")
    r0 = q - jnp.asarray(observer)[:, None] * basis.box.L

    varrho0, _ = get_distance_and_n_los(r0)
    a = background.a_of_chi(background.chi_of_varrho(varrho0))
    for _ in range(n_iter):
        psi, dpsi = psi_and_dpsi_dlna(a)
        varrho, n_los = get_distance_and_n_los(r0 + psi)
        dvarrho_dlna = jnp.sum(dpsi * n_los, axis=0)
        f = background.chi_of_a(a) - background.chi_of_varrho(varrho)
        df_dlna = (
            a * background.dchi_da(a) - background.dchi_dvarrho(varrho) * dvarrho_dlna
        )
        a = jnp.minimum(a * jnp.exp(-f / df_dlna), 1.0)
    return a


def _sphere_means(
    catalog: ProtohaloCatalog,
    fields: list[VectorField],
    background: BackgroundCosmo,
    R_grid: Sequence[float],
    interpolation: InterpolationMethodLiteral,
) -> list[jax.Array]:
    """Each field averaged over the sphere of radius R_lag around each centre.

    Spheres are evaluated on R_grid (any order); each protohalo blends the two
    grid radii around its own linearly in ln R, exactly one when it lies on
    the grid. Every radius is read at all centres, so array shapes do not
    change.
    """
    R_grid = tuple(sorted(float(R) for R in R_grid))
    if not R_grid or R_grid[0] <= 0 or len(set(R_grid)) != len(R_grid):
        raise ValueError("R_grid must hold distinct positive radii.")
    R_j = jnp.asarray(R_grid)

    R_h = catalog.R_lag(background)
    lo, hi = R_grid[0], R_grid[-1]
    if jnp.any(R_h < lo * (1 - _R_TOL)) or jnp.any(R_h > hi * (1 + _R_TOL)):
        raise ValueError(
            f"R_lag spans [{float(R_h.min())}, {float(R_h.max())}] Mpc/h, "
            f"outside R_grid [{lo}, {hi}]."
        )

    # (m, n) hat functions of ln R: column h sums to one.
    lnR_h = jnp.log(jnp.clip(R_h, lo, hi))
    weights = jnp.stack([jnp.interp(lnR_h, jnp.log(R_j), e) for e in jnp.eye(R_j.size)])

    fields = [f.fft() for f in fields]  # no-op for basis shapes, already Fourier
    out = [jnp.zeros_like(catalog.q) for _ in fields]
    for R, w in zip(R_grid, weights):
        if not jnp.any(w > 0):
            continue
        window = Window("radial_tophat", 2.0 * R, normalized=True)  # diameter
        for i, f in enumerate(fields):
            mean = f.convolve(window).interpolate_at(catalog.q, method=interpolation)
            out[i] = out[i] + w * mean
    return out
