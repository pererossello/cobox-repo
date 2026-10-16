"""Spatial points with one active Cartesian or spherical representation."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from . import _coordinates

if TYPE_CHECKING:
    from cosphere import Shell, ShellField


class Positions(eqx.Module):
    """Euclidean 3D points, with component-first data of shape (3, N).

    data contains (x, y, z), or (radius, theta, phi) when has_spherical=True.
    Radii use arbitrary consistent units; angles are colatitude and longitude
    in radians. No cosmology, periodicity or physical units are implied.

    Conversions return Positions and return self when already in the requested
    representation. Only the active representation is stored. Directions are
    undefined at zero radius; angular derivatives are singular at the poles.
    """

    data: jax.Array
    has_spherical: bool = eqx.field(static=True, default=False)

    def __init__(self, data: ArrayLike, has_spherical: bool = False):
        self.data = _coordinates.real_array(data, "data")
        self.has_spherical = has_spherical
        if not isinstance(has_spherical, bool):
            raise TypeError("has_spherical must be a bool.")
        if self.data.ndim != 2 or self.data.shape[0] != 3:
            raise ValueError(f"data must have shape (3, N); got {self.data.shape}.")
        if has_spherical:
            self.data = _coordinates.spherical_data(*self.data)

    # -------------------
    # --- COORDINATES ---
    # -------------------

    @classmethod
    def from_spherical(
        cls, radius: ArrayLike, theta: ArrayLike, phi: ArrayLike
    ) -> Positions:
        """Store spherical coordinates without converting to Cartesian.

        Scalars and 1D arrays broadcast to a common length. Radius must be
        nonnegative and theta in [0, pi]. Longitude is wrapped to [0, 2*pi).
        A scalar triple produces one point, with shape (3, 1).
        """
        return cls(_coordinates.spherical_data(radius, theta, phi), has_spherical=True)

    def to_cartesian(self) -> Positions:
        """Return Cartesian Positions, or self when already Cartesian."""
        if not self.has_spherical:
            return self
        return replace(
            self, data=_coordinates.cartesian(*self.data), has_spherical=False
        )

    def to_spherical(self) -> Positions:
        """Return spherical Positions, or self when already spherical.

        Cartesian points at the origin have no angular coordinates and raise
        an error. Longitude at either Cartesian pole is conventionally zero.
        """
        if self.has_spherical:
            return self
        r = _coordinates.nonzero_radius(self.data)
        direction = self.data / r
        data = jnp.stack(
            (r, _coordinates.colatitude(direction), _coordinates.longitude(direction))
        )
        return replace(self, data=data, has_spherical=True)

    # ----------------
    # --- GEOMETRY ---
    # ----------------

    @property
    def radius(self) -> jax.Array:
        """Distance from the current origin, shape (N,)."""
        return self.data[0] if self.has_spherical else _coordinates.radius(self.data)

    def _nonzero_spherical(self) -> jax.Array:
        return eqx.error_if(
            self.data,
            jnp.any(self.data[0] == 0),
            "Direction is undefined at the origin.",
        )

    @property
    def unit_vectors(self) -> jax.Array:
        """Cartesian unit directions, shape (3, N); requires nonzero radius."""
        if self.has_spherical:
            _, theta, phi = self._nonzero_spherical()
            return _coordinates.cartesian(1.0, theta, phi)
        return _coordinates.unit_vectors(self.data)

    def directions_from(self, origin: ArrayLike = (0.0, 0.0, 0.0)) -> jax.Array:  # type: ignore
        """Unit vectors from origin to each position, shape (3, N).

        origin is a Cartesian coordinate with shape (3,), in the same units
        as these positions. Coincident points have no direction.
        """
        return self.relative_to(origin).unit_vectors

    @property
    def theta(self) -> jax.Array:
        """Colatitude in [0, pi], shape (N,); requires nonzero radius."""
        if self.has_spherical:
            return self._nonzero_spherical()[1]
        return _coordinates.colatitude(self.unit_vectors)

    @property
    def phi(self) -> jax.Array:
        """Longitude in [0, 2*pi), shape (N,); requires nonzero radius."""
        if self.has_spherical:
            return self._nonzero_spherical()[2]
        return _coordinates.longitude(self.unit_vectors)

    def relative_to(self, origin: ArrayLike) -> Positions:
        """Return Cartesian positions relative to origin, without rotating axes.

        origin has shape (3,) in coordinate units, not box fractions.
        """
        origin = _coordinates.real_array(origin, "origin")
        if origin.shape != (3,):  # type: ignore
            raise ValueError(f"origin must have shape (3,); got {origin.shape}.")  # type: ignore
        cartesian = self.to_cartesian()
        return replace(cartesian, data=cartesian.data - origin[:, None])  # type: ignore

    def radial_component(self, vectors: ArrayLike) -> jax.Array:
        """Project Cartesian vectors (3, N) onto the directions of these points.

        Returns (N,) in the vectors' units. No observer velocity is subtracted.
        """
        vectors = _coordinates.real_array(vectors, "vectors")
        if vectors.shape != self.data.shape:  # type: ignore
            raise ValueError(
                f"vectors must have shape {self.data.shape}; got {vectors.shape}."  # type: ignore
            )
        return jnp.sum(vectors * self.unit_vectors, axis=0)

    # -----------------
    # --- SELECTION ---
    # -----------------

    def radial_mask(self, r_min: ArrayLike, r_max: ArrayLike) -> jax.Array:
        """Mask for r_min <= radius < r_max, relative to the current origin.

        Bounds are finite scalars with 0 <= r_min < r_max, in radius units.
        JIT-compatible; the hard boundary is not a smooth selection function.
        """
        lo = _coordinates.real_array(r_min, "r_min")
        hi = _coordinates.real_array(r_max, "r_max")
        if lo.ndim != 0 or hi.ndim != 0:
            raise ValueError("radial bounds must be scalars.")
        lo = eqx.error_if(lo, (lo < 0) | (lo >= hi), "require 0 <= r_min < r_max.")
        r = self.radius
        return (r >= lo) & (r < hi)

    def select(self, mask: ArrayLike) -> Positions:
        """Select points, preserving representation and order.

        Host-side: the result changes length. Under JIT, keep all points and
        mask weights instead. Apply the same mask to external per-point data.
        """
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != (len(self),):
            raise ValueError(f"mask must be a boolean array of shape ({len(self)},).")
        return replace(self, data=self.data[:, mask])

    def radial_slice(self, r_min: ArrayLike, r_max: ArrayLike) -> Positions:
        """Host-side selection of r_min <= radius < r_max; see radial_mask."""
        return self.select(self.radial_mask(r_min, r_max))

    # ------------------------
    # --- SHELL ASSIGNMENT ---
    # ------------------------

    def shell_assign(
        self,
        shell: Shell,
        weights: ArrayLike | None = None,
        *,
        method: Literal["ngp", "bilinear"] = "ngp",
    ) -> ShellField:
        """Assign weights to shell pixels using directions from the current origin.

        Returns sums per pixel, not density per steradian. weights has shape
        (N,), with unit weights by default. Bilinear is piecewise smooth;
        NGP is discontinuous at pixel boundaries. No radial selection is added.
        """
        from cosphere.shell_assign import shell_assign

        spherical = self.to_spherical()
        _, theta, phi = spherical._nonzero_spherical()
        return shell_assign(shell, theta, phi, weights=weights, method=method)

    def __len__(self) -> int:
        return self.data.shape[1]
