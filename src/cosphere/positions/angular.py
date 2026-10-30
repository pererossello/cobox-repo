"""Directions on the sphere: colatitude theta in [0, pi], longitude phi, in radians."""

from math import pi

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ..field import ShellField
from ..shell import Shell
from ..shell._stencil import StencilMethod
from ._deposit import Reduce, deposit, deposit_gaussian


class AngularPositions(eqx.Module):
    """Directions on the sphere, in the same frame as the Shell they are used with."""

    theta: jax.Array
    phi: jax.Array

    def __init__(self, theta: ArrayLike, phi: ArrayLike):
        t = jnp.asarray(theta, dtype=float)
        p = jnp.asarray(phi, dtype=float)
        if t.ndim != 1 or t.shape != p.shape:
            raise ValueError(
                f"theta and phi must be 1D arrays of equal length; got {t.shape}, {p.shape}."
            )
        # ~(t >= 0) also catches NaN
        self.theta = eqx.error_if(
            t, jnp.any(~(t >= 0) | ~(t <= pi)), "theta must be in [0, pi]."
        )
        self.phi = eqx.error_if(
            jnp.mod(p, 2 * pi), jnp.any(~jnp.isfinite(p)), "phi must be finite."
        )

    @classmethod
    def from_radec(
        cls, ra: ArrayLike, dec: ArrayLike, *, degrees: bool = True
    ) -> "AngularPositions":
        """theta = pi/2 - dec, phi = ra."""
        ra, dec = jnp.asarray(ra, dtype=float), jnp.asarray(dec, dtype=float)
        if degrees:
            ra, dec = jnp.deg2rad(ra), jnp.deg2rad(dec)
        return cls(0.5 * pi - dec, ra)

    @classmethod
    def from_vectors(cls, vectors: ArrayLike) -> "AngularPositions":
        """Directions of nonzero vectors, shape (3, N); need not be normalized."""
        v = jnp.asarray(vectors, dtype=float)
        if v.ndim != 2 or v.shape[0] != 3:
            raise ValueError(f"vectors must have shape (3, N); got {v.shape}.")
        x, y, z = v
        rho = jnp.hypot(x, y)
        rho = eqx.error_if(
            rho, jnp.any((rho == 0) & (z == 0)), "vectors must be nonzero."
        )
        return cls(jnp.arctan2(rho, z), jnp.arctan2(y, x))

    @property
    def unit_vectors(self) -> jax.Array:
        """Shape (3, N), as Shell.unit_vectors."""
        s = jnp.sin(self.theta)
        return jnp.stack(
            [s * jnp.cos(self.phi), s * jnp.sin(self.phi), jnp.cos(self.theta)]
        )

    def select(self, mask: ArrayLike) -> "AngularPositions":
        """Directions where mask is True. The length changes, so not under jit."""
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != (len(self),):
            raise ValueError(f"mask must be a boolean array of shape ({len(self)},).")
        return AngularPositions(self.theta[mask], self.phi[mask])

    def to_shell(
        self,
        shell: Shell,
        values: ArrayLike | None = None,
        *,
        method: StencilMethod = "ngp",
        reduce: Reduce = "sum",
    ) -> ShellField:
        """Values at these points onto shell pixels: their sum per pixel, or their mean.

        values defaults to 1, so the sum counts points. ngp uses the containing pixel;
        bilinear spreads each point over 4 pixels, as interpolate_at reads them.
        Pixels with no points are NaN for "mean".
        """
        return deposit(self, shell, values, method, reduce)

    def to_shell_gaussian(
        self,
        shell: Shell,
        sigma: ArrayLike,
        values: ArrayLike | None = None,
        *,
        truncate: float = 4.0,
        reduce: Reduce = "sum",
    ) -> ShellField:
        """Like to_shell, but each point spreads over nearby pixels with Gaussian weights.

        sigma (rad) is one value or one per point. Each point's weights are
        exp(-d^2 / 2 sigma^2) over pixels within truncate * sigma, normalised to
        sum to 1; the containing pixel always counts, so sigma -> 0 gives ngp.
        Host-side, not under jit.
        """
        return deposit_gaussian(self, shell, sigma, values, truncate, reduce)

    def __len__(self) -> int:
        return self.theta.shape[0]
