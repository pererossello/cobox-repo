"""Redshift catalogue: sky positions and observed redshifts."""

from typing import TYPE_CHECKING, get_args

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from cosphere import AngularPositions
from jax.typing import ArrayLike

from .tomography import TomographicBin
from .window import RedshiftWindow

if TYPE_CHECKING:
    from cobox.box import Box
    from cobox.field import ScalarField
    from cobox.fluid.particles import MeshConventionLiteral
    from colcdm import BackgroundCosmo
    from cosphere import Shell, ShellField, ShellMask


class RedshiftCatalog(eqx.Module):
    """Objects with sky directions (AngularPositions) and observed redshifts z.

    Cosmology enters only through the BackgroundCosmo passed to the distance
    and box methods.
    """

    angular: AngularPositions
    z: jax.Array

    def __init__(self, theta: ArrayLike, phi: ArrayLike, z: ArrayLike):
        self.angular = AngularPositions(theta, phi)
        z = jnp.asarray(z, dtype=float)
        if z.shape != (len(self.angular),):
            raise ValueError(
                f"z must have shape ({len(self.angular)},); got {z.shape}."
            )
        self.z = eqx.error_if(
            z, jnp.any(~jnp.isfinite(z) | (z <= -1)), "z must be finite and > -1."
        )

    @classmethod
    def from_radec(
        cls, ra: ArrayLike, dec: ArrayLike, z: ArrayLike, *, degrees: bool = True
    ) -> "RedshiftCatalog":
        """From right ascension and declination: theta = pi/2 - dec, phi = ra."""
        a = AngularPositions.from_radec(ra, dec, degrees=degrees)
        return cls(a.theta, a.phi, z)

    @classmethod
    def from_vectors(cls, vectors: ArrayLike, z: ArrayLike) -> "RedshiftCatalog":
        """From directions, shape (3, N); need not be normalized."""
        a = AngularPositions.from_vectors(vectors)
        return cls(a.theta, a.phi, z)

    @property
    def theta(self) -> jax.Array:
        return self.angular.theta

    @property
    def phi(self) -> jax.Array:
        return self.angular.phi

    # -----------------
    # --- SELECTION ---
    # -----------------

    def select(self, mask: ArrayLike) -> "RedshiftCatalog":
        """Objects where mask is True. Changes the size, so it runs on the host,
        not under jit; inside jitted code, use zero weights instead."""
        a = self.angular.select(mask)
        return RedshiftCatalog(a.theta, a.phi, self.z[np.asarray(mask)])

    def z_slice(self, z_min: float, z_max: float) -> "RedshiftCatalog":
        """Objects with z_min <= z < z_max (half-open, so adjacent slices do not
        share objects). Host-side, like select."""
        if not z_min < z_max:
            raise ValueError(f"need z_min < z_max; got {z_min} and {z_max}.")
        z = np.asarray(self.z)
        return self.select((z >= z_min) & (z < z_max))

    # -----------
    # --- SKY ---
    # -----------

    @property
    def unit_vectors(self) -> jax.Array:
        """Direction of each object, shape (3, N), as Shell.unit_vectors."""
        return self.angular.unit_vectors

    def pixels(self, shell: "Shell") -> jax.Array:
        """Index of the shell pixel containing each object."""
        return shell.ang2pix(self.theta, self.phi)

    def values_at(self, field: "ShellField") -> jax.Array:
        """Value of a map in the pixel containing each object, shape (N,).

        Nearest-pixel lookup, no interpolation; a field given in harmonic space is
        first transformed to pixels.
        """
        field = field.isht()
        return field.data[self.pixels(field.shell)]

    # -----------------
    # --- DISTANCES ---
    # -----------------

    def comoving_distance(self, background: "BackgroundCosmo") -> jax.Array:
        """Radial comoving distance chi(z) in Mpc/h, shape (N,).

        The observed z is treated as purely cosmological: no peculiar-velocity
        (Doppler/RSD) correction. Requires z >= 0. Valid for any curvature.
        """
        z = eqx.error_if(
            self.z, jnp.any(self.z < 0), "comoving distances require z >= 0."
        )
        return background.chi_of_a(background.a_of_z(z))

    def comoving_positions(self, background: "BackgroundCosmo") -> jax.Array:
        """3D comoving positions in Mpc/h, shape (3, N), observer at the origin.

        Radius is the isotropic coordinate varrho(chi), the box coordinate used by
        LPT and the light cone; it equals chi in flat geometry.
        """
        varrho = background.varrho_of_chi(self.comoving_distance(background))
        return varrho * self.unit_vectors

    # --------------
    # --- COUNTS ---
    # --------------

    def shell_counts(
        self, shell: "Shell", weights: ArrayLike | None = None
    ) -> "ShellField":
        """Objects per shell pixel, or the sum of per-object weights (N,).

        E.g. weights = W(z) gives weighted counts, weights = z * W(z) the weighted
        redshift sum. Nearest pixel; see AngularPositions.to_shell.
        """
        return self.angular.to_shell(shell, weights)

    def box_counts(
        self,
        background: "BackgroundCosmo",
        box: "Box",
        *,
        observer: tuple[float, float, float] = (0.5, 0.5, 0.5),
        mesh_convention: "MeshConventionLiteral" = "node",
        weights: ArrayLike | None = None,
    ) -> "ScalarField":
        """Objects per cell of a 3D box, nearest grid point; counts sum to len(self),
        or to sum(weights) for optional per-object weights, shape (N,).

        observer is the observer position in box fractions, as in LPT. The box is
        not periodic: every object must fall inside it ("cell": [0, L); "node":
        [-R/2, L - R/2), cells centred on j R), otherwise an error is raised.
        """
        from cobox.field import ScalarField
        from cobox.fluid.mass_assign._native import deposit_native

        idx_pos = self._box_cell_positions(background, box, observer, mesh_convention)
        if weights is not None:
            weights = self._check_weights(weights)
        counts = deposit_native(idx_pos, weights, "ngp", box.SHAPE, mesh_convention)
        return ScalarField(counts, box=box)

    # --------------------------
    # --- ANGULAR STATISTICS ---
    # --------------------------

    def tomographic_bin(
        self,
        shell: "Shell",
        window: RedshiftWindow | None = None,
        weights: ArrayLike | None = None,
        *,
        randoms: "RedshiftCatalog | None" = None,
        random_weights: ArrayLike | None = None,
        coverage: "ShellField | None" = None,
        mask: "ShellMask | None" = None,
    ) -> TomographicBin:
        """These objects with weights w W(z), projected on shell; see TomographicBin."""
        return TomographicBin(
            self, shell, window=window, weights=weights, randoms=randoms,
            random_weights=random_weights, coverage=coverage, mask=mask,
        )

    # ------------
    # --- PLOT ---
    # ------------

    def plot_wedge(self, ax=None, **kwargs):
        """Polar RA/radius wedge; see cozcat.plot.plot_wedge for options.

        Angular limits are in degrees. Defaults to redshift radius; use
        radius="comoving", background=... for radial distance in Mpc/h.
        Returns CatalogPlotResult without calling show().
        """
        from .plot.wedge import plot_wedge

        return plot_wedge(self, ax=ax, **kwargs)

    def plot_sky(self, ax=None, **kwargs):
        """Sky scatter; see cozcat.plot.plot_sky for options.

        Supports rectangular and Mollweide projections, degree-valued angular
        cuts and reproducible display sampling. Returns CatalogPlotResult.
        """
        from .plot.sky import plot_sky

        return plot_sky(self, ax=ax, **kwargs)

    def plot_3d(self, ax=None, **kwargs):
        """3D radial view; see cozcat.plot.plot_3d for options.

        Reuse ax to overlay shells. Radius defaults to redshift, a visualization
        coordinate rather than physical distance. Returns CatalogPlotResult.
        """
        from .plot.spatial import plot_3d

        return plot_3d(self, ax=ax, **kwargs)

    def plot_redshift(self, ax=None, **kwargs):
        """Redshift histogram; see cozcat.plot.plot_redshift for options.

        Uses all selected objects, with optional per-object weights and bins.
        Returns CatalogPlotResult without calling show().
        """
        from .plot.histogram import plot_redshift

        return plot_redshift(self, ax=ax, **kwargs)

    # ---------------------
    # --- ARRAY SURFACE ---
    # ---------------------

    def __len__(self) -> int:
        return self.z.shape[0]

    # -------------
    # --- UTILS ---
    # -------------

    def _box_cell_positions(
        self,
        background: "BackgroundCosmo",
        box: "Box",
        observer: tuple[float, float, float],
        mesh_convention: "MeshConventionLiteral",
    ) -> jax.Array:
        """Positions in cell units of box, shape (3, N), checked to lie inside it."""
        from cobox.fluid.particles import MeshConventionLiteral

        if mesh_convention not in get_args(MeshConventionLiteral):
            raise ValueError(
                f"mesh_convention must be one of {get_args(MeshConventionLiteral)}; "
                f"got {mesh_convention!r}."
            )
        if box.D != 3:
            raise ValueError(f"box must be 3D; got D={box.D}.")
        if len(observer) != 3 or not all(0.0 <= o <= 1.0 for o in observer):
            raise ValueError("observer must be 3 box fractions in [0, 1].")

        origin = jnp.asarray(observer)[:, None] * box.L
        idx_pos = (self.comoving_positions(background) + origin) / box.R
        lo = 0.0 if mesh_convention == "cell" else -0.5
        return eqx.error_if(
            idx_pos,
            jnp.any((idx_pos < lo) | (idx_pos >= box.N + lo)),
            "objects fall outside the box; enlarge L or move the observer.",
        )

    # ------------------
    # --- VALIDATION ---
    # ------------------

    def _check_weights(self, weights: ArrayLike) -> jax.Array:
        """Per-object weights: shape (N,), finite; any sign is allowed."""
        weights = jnp.asarray(weights, dtype=float)
        if weights.shape != (len(self),):
            raise ValueError(
                f"weights must have shape ({len(self)},); got {weights.shape}."
            )
        return eqx.error_if(
            weights, jnp.any(~jnp.isfinite(weights)), "weights must be finite."
        )
