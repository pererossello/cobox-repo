"""Redshift selection conversion; output uses the existing radial window API."""

from jax.typing import ArrayLike

from cosphere.projection import TabulatedWindow

from ..cosmology import BackgroundCosmo
from ..cosmology._constants import C_LIGHT, H0


def radial_window_from_redshift(
    z: ArrayLike,
    values: ArrayLike,
    background: BackgroundCosmo,
    *,
    normalize: bool = True,
    n_distance: int = 4096,
    a_min: float = 1e-5,
) -> TabulatedWindow:
    """Convert a tabulated weight in dz into W(chi) dchi, with chi in Mpc/h.

    W(chi(z)) = values(z) * dz/dchi, where dz/dchi = 100 E(z) / c.
    z is finite, nonnegative and strictly increasing. values may be signed;
    compensated selections require normalize=False. No volume factor is used:
    values denotes a weight per redshift, not a 3D number density.

    Returns a piecewise-linear approximation in chi at the transformed input
    knots. Refine the z grid for convergence. normalize=True gives the returned
    radial profile unit integral; normalize=False preserves the transformed
    amplitudes (its integral has interpolation error). Conversion is valid for
    curved backgrounds too, though AngularPower currently requires flat space.

    Arrays and background remain differentiable. The result is a radial window
    for THIS background: rebuild it inside the prediction when varying cosmology
    at fixed redshift selection. Redshifts are treated as cosmological, with no
    Doppler/RSD correction. No new window subclass is needed.
    """
    selection = TabulatedWindow(z, values, normalize=normalize)
    a = background.a_of_z(selection.r)
    chi = background.chi_of_a(a, n_quad=n_distance, a_min=a_min)
    jacobian = H0 * background.E(a) / C_LIGHT
    return TabulatedWindow(chi, selection.values * jacobian, normalize=normalize)
