"""Convert a redshift integration measure into a comoving radial measure."""

import equinox as eqx
import jax
import jax.numpy as jnp

from cosphere.projection import RadialWindow

from ..cosmology import BackgroundCosmo
from ..cosmology._distances import chi_of_a, get_distance_table


class _RedshiftMeasure(RadialWindow):
    """Private coordinate adapter; the selection owns its integration rule."""

    selection: RadialWindow
    a_grid: jax.Array
    chi_grid: jax.Array

    def _chi(self, z):
        return chi_of_a(1 / (1 + jnp.asarray(z)), self.a_grid, self.chi_grid)

    @property
    def support(self):
        bounds = self._chi(jnp.stack(self.selection.support))
        return bounds[0], bounds[1]

    def quadrature(self, n_r=128):
        z, measure = self.selection.quadrature(n_r)
        return self._chi(z), measure


def radial_window_from_redshift(
    selection: RadialWindow,
    background: BackgroundCosmo,
    *,
    n_distance: int = 4096,
    a_min: float = 1e-5,
) -> RadialWindow:
    """Map a selection in dz to W(chi) dchi, with chi in Mpc/h.

    Supply an existing window with its coordinate interpreted as redshift:
    GaussianWindow(center_z, sigma_z), TabulatedWindow(z, values), a top hat,
    or a thin shell. Normalization, signed amplitudes and truncation belong to
    that selection. values denotes a weight per redshift, not a volume density.

    The returned measure maps quadrature nodes from z to chi and leaves their
    weights unchanged: integral dchi W(chi) f(chi) = integral dz n(z) f(chi(z)).
    The Jacobian is therefore accounted for by the change of variables, without
    interpolating the profile again in chi. Smooth analytic selections use the
    requested n_r; piecewise-linear tables retain their per-segment quadrature.
    The result implements support/quadrature, not a pointwise profile or table.

    Rebuild inside the prediction when varying background at fixed selection.
    Selection parameters and distances remain differentiable. Conversion also
    supports curved backgrounds, though AngularPower requires flat space.
    Redshifts are cosmological; no Doppler/RSD correction is applied.
    """
    if not isinstance(selection, RadialWindow):
        raise TypeError("selection must be a RadialWindow with redshift coordinates.")
    a_grid, chi_grid = get_distance_table(background, n_quad=n_distance, a_min=a_min)
    # Validate endpoints too: Gaussian/Legendre nodes lie inside the support.
    bounds = jnp.stack(selection.support)
    a_grid = eqx.error_if(
        a_grid,
        jnp.any(~jnp.isfinite(bounds))
        | jnp.any(bounds < 0)
        | jnp.any(1 / (1 + bounds) < a_grid[0]),
        "Redshift support must be finite and within [0, 1/a_min - 1].",
    )
    return _RedshiftMeasure(selection, a_grid, chi_grid)
