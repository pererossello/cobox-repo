"""Normalized axisymmetric windows on the sphere."""

from math import pi
from numbers import Integral
from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from . import _harmonic, _profiles

if TYPE_CHECKING:
    from ..shell import Shell

WindowKindLiteral = Literal["gaussian", "disc", "annulus", "aperture"]


def _real(value: ArrayLike, name: str) -> jax.Array:
    value = jnp.asarray(value)
    if jnp.iscomplexobj(value):
        raise ValueError(f"{name} must be real.")
    value = value.astype(float)
    return eqx.error_if(value, jnp.any(~jnp.isfinite(value)), f"{name} must be finite.")


class Window(eqx.Module):
    """Axisymmetric window on the sphere; window(ell) gives its harmonic response b_ell.

    Angles in radians. gaussian: scale = sigma, b_ell = exp(-ell(ell+1) sigma^2 / 2).
    disc: radius scale. annulus: from scale to outer_radius. aperture: disc minus
    annulus. Disc and annulus integrate to 1 (they average); aperture to 0.
    """

    kind: WindowKindLiteral = eqx.field(static=True)
    scale: jax.Array
    outer_radius: jax.Array | None

    def __init__(
        self,
        kind: WindowKindLiteral,
        scale: ArrayLike,
        *,
        outer_radius: ArrayLike | None = None,
    ):
        if kind not in ("gaussian", "disc", "annulus", "aperture"):
            raise ValueError(f"Unknown angular window kind: {kind!r}.")
        self.kind = kind
        scale = _real(scale, "scale")
        if scale.ndim != 0:
            raise ValueError("scale must be a scalar.")
        invalid = scale <= 0
        if kind != "gaussian":
            invalid = invalid | (scale > pi)
        self.scale = eqx.error_if(
            scale, invalid, "scale must be positive; angular radii cannot exceed pi."
        )
        if kind in ("annulus", "aperture"):
            if outer_radius is None:
                raise ValueError("annulus and aperture require outer_radius.")
            outer = _real(outer_radius, "outer_radius")
            if outer.ndim != 0:
                raise ValueError("outer_radius must be a scalar.")
            self.outer_radius = eqx.error_if(
                outer,
                (outer <= self.scale) | (outer > pi),
                "require scale < outer_radius <= pi.",
            )
        else:
            if outer_radius is not None:
                raise ValueError("outer_radius is only used by annulus and aperture.")
            self.outer_radius = None

    def win_profile_hat(self, ell: ArrayLike) -> jax.Array:
        """Harmonic response b_ell at integer ell >= 0, same shape as ell."""
        ell = jnp.asarray(ell)
        if not jnp.issubdtype(ell.dtype, jnp.integer):
            raise ValueError("ell must contain integers.")
        ell = eqx.error_if(ell, jnp.any(ell < 0), "ell must be nonnegative.")
        return _harmonic.response(self.kind, self.scale, self.outer_radius, ell)

    def __call__(self, ell: ArrayLike) -> jax.Array:
        return self.win_profile_hat(ell)

    def win_hat_on_shell(self, shell: "Shell") -> jax.Array:
        """b_ell for ell = 0 .. shell.L - 1."""
        return self.win_profile_hat(shell.ell_1D)

    def win_profile(self, theta: ArrayLike, *, ell_max: int | None = None) -> jax.Array:
        """Profile per steradian at angle theta (radians) from the centre.

        gaussian only: summed from its harmonic series up to ell_max, so raise
        ell_max until the profile stops changing.
        """
        theta = _real(theta, "theta")
        theta = eqx.error_if(
            theta, jnp.any((theta < 0) | (theta > pi)), "theta must be in [0, pi]."
        )
        if self.kind == "gaussian":
            if (
                isinstance(ell_max, bool)
                or not isinstance(ell_max, Integral)
                or ell_max < 0
            ):
                raise ValueError(
                    "Gaussian profile requires a static nonnegative integer ell_max."
                )
        elif ell_max is not None:
            raise ValueError("ell_max is only used by the Gaussian profile.")
        return _profiles.profile(
            self.kind, self.scale, self.outer_radius, theta, ell_max
        )
