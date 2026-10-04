"""Redshift catalogue: sky positions and observed redshifts."""

from math import pi
from typing import TYPE_CHECKING, get_args

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

if TYPE_CHECKING:
    from cobox.box import Box
    from cobox.field import ScalarField
    from cobox.fluid.particles import MeshConventionLiteral
    from colcdm import BackgroundCosmo
    from cosphere import Shell, ShellField


class RedshiftCatalog(eqx.Module):
    """Objects with sky angles and observed redshifts.

    theta is the colatitude in [0, pi] and phi the longitude, in radians, in the
    same frame as the maps the catalogue is combined with (Shell convention).
    z is the observed redshift. Cosmology enters only through the BackgroundCosmo
    passed to the distance and box methods.
    """

    theta: jax.Array
    phi: jax.Array
    z: jax.Array

    def __init__(self, theta: ArrayLike, phi: ArrayLike, z: ArrayLike):
        self.theta = jnp.asarray(theta, dtype=float)
        self.phi = jnp.asarray(phi, dtype=float)
        self.z = jnp.asarray(z, dtype=float)
        self._validate()

    @classmethod
    def from_radec(
        cls, ra: ArrayLike, dec: ArrayLike, z: ArrayLike, *, degrees: bool = True
    ) -> "RedshiftCatalog":
        """From right ascension and declination: theta = pi/2 - dec, phi = ra."""
        ra, dec = jnp.asarray(ra, dtype=float), jnp.asarray(dec, dtype=float)
        if degrees:
            ra, dec = jnp.deg2rad(ra), jnp.deg2rad(dec)
        return cls(0.5 * pi - dec, ra, z)

    @classmethod
    def from_vectors(cls, vectors: ArrayLike, z: ArrayLike) -> "RedshiftCatalog":
        """From directions, shape (3, N), need not be normalized (e.g. positions
        relative to the observer); the inverse of unit_vectors."""
        vectors = jnp.asarray(vectors, dtype=float)
        if vectors.ndim != 2 or vectors.shape[0] != 3:
            raise ValueError(f"vectors must have shape (3, N); got {vectors.shape}.")
        x, y, w = vectors
        r = jnp.sqrt(x * x + y * y + w * w)
        r = eqx.error_if(r, jnp.any(r == 0), "vectors must be nonzero.")
        theta = jnp.arccos(jnp.clip(w / r, -1.0, 1.0))
        phi = jnp.mod(jnp.arctan2(y, x), 2 * pi)
        return cls(theta, phi, z)

    # -----------------
    # --- SELECTION ---
    # -----------------

    def select(self, mask: ArrayLike) -> "RedshiftCatalog":
        """Objects where mask is True. Changes the size, so it runs on the host,
        not under jit; inside jitted code, use zero weights instead."""
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != (len(self),):
            raise ValueError(f"mask must be a boolean array of shape ({len(self)},).")
        return RedshiftCatalog(self.theta[mask], self.phi[mask], self.z[mask])

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
        sin_theta = jnp.sin(self.theta)
        return jnp.stack(
            [
                sin_theta * jnp.cos(self.phi),
                sin_theta * jnp.sin(self.phi),
                jnp.cos(self.theta),
            ]
        )

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
        """Objects per shell pixel, or the sum of per-object weights, shape (N,).

        Counts sum to len(self), or to sum(weights). E.g. weights = W(z) gives
        weighted counts, weights = z * W(z) the weighted redshift sum. Angles only.
        """
        from cosphere import ShellField

        weights = 1.0 if weights is None else self._check_weights(weights)
        counts = jnp.zeros(shell.SHAPE).at[self.pixels(shell)].add(weights)
        return ShellField(counts, shell=shell)

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

    def _validate(self) -> None:
        """Shapes are checked eagerly; values with eqx.error_if, so also under jit."""
        shapes = (self.theta.shape, self.phi.shape, self.z.shape)
        if self.theta.ndim != 1 or len(set(shapes)) != 1:
            raise ValueError(
                f"theta, phi and z must be 1D arrays of equal length; got shapes {shapes}."
            )
        self.theta = eqx.error_if(
            self.theta,
            jnp.any(~jnp.isfinite(self.theta) | (self.theta < 0) | (self.theta > pi)),
            "theta must be finite and in [0, pi].",
        )
        self.phi = eqx.error_if(
            self.phi, jnp.any(~jnp.isfinite(self.phi)), "phi must be finite."
        )
        self.z = eqx.error_if(
            self.z,
            jnp.any(~jnp.isfinite(self.z) | (self.z <= -1)),
            "z must be finite and > -1.",
        )
