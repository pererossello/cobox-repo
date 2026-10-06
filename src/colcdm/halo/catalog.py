"""Halo catalogues, and the base they share with protohalo catalogues."""

from __future__ import annotations

import abc
from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..cosmology._constants import C_LIGHT
from ..cosmology.background import BackgroundCosmo

if TYPE_CHECKING:
    from cobox.box import Box
    from cozcat import RedshiftCatalog


class _AbstractCatalog(eqx.Module):
    """What protohalo and halo catalogues share: positions (D, n), masses
    M [M_sun/h] (n,) and an epoch a. Subclasses name their positions and
    implement select."""

    M: eqx.AbstractVar[jax.Array]
    a: eqx.AbstractVar[jax.Array]
    COORD: ClassVar[str]  # positions symbol, for axis labels

    @property
    @abc.abstractmethod
    def positions(self) -> jax.Array: ...

    def __len__(self) -> int:
        return self.M.shape[0]

    def R_lag(self, background: BackgroundCosmo) -> jax.Array:
        """Lagrangian radii in Mpc/h."""
        return background.R_lag_of_M(self.M)

    # -----------------
    # --- SELECTION ---
    # -----------------

    @abc.abstractmethod
    def select(self, mask: ArrayLike) -> Self:
        """Objects where mask is True. Changes the size, so it runs on the
        host, not under jit."""

    def M_slice(self, M_min: float, M_max: float) -> Self:
        """Objects with M_min <= M < M_max [M_sun/h] (half-open, so adjacent
        slices do not share objects). Host-side, like select."""
        if not M_min < M_max:
            raise ValueError(f"need M_min < M_max; got {M_min} and {M_max}.")
        return self.select((self.M >= M_min) & (self.M < M_max))

    def R_lag_slice(
        self, R_min: float, R_max: float, background: BackgroundCosmo
    ) -> Self:
        """Objects with Lagrangian radius R_min <= R_lag < R_max [Mpc/h]."""
        if not R_min < R_max:
            raise ValueError(f"need R_min < R_max; got {R_min} and {R_max}.")
        R = self.R_lag(background)
        return self.select((R >= R_min) & (R < R_max))

    def _mask(self, mask: ArrayLike) -> jax.Array:
        mask = jnp.asarray(mask)
        if mask.dtype != bool or mask.shape != (len(self),):
            raise ValueError(f"mask must be a boolean array of shape ({len(self)},).")
        return mask

    # ------------
    # --- PLOT ---
    # ------------

    def paint(self, box: Box, background=None, ax=None, **kwargs):
        """Slab of centres or Lagrangian spheres; see halo._paint.paint."""
        from ._paint import paint

        return paint(self, box, background, ax=ax, **kwargs)


class HaloCatalog(_AbstractCatalog):
    """Halos at positions x in Mpc/h, (D, n), unwrapped.

    M: masses in M_sun/h, (n,). a: the epoch the positions refer to, a scalar
    (snapshot) or (n,) (light cone). v: peculiar velocities a dx/dt in km/s,
    (D, n), or None when unknown. observer: position in Mpc/h of the observer
    whose light cone the halos lie on, or None for a snapshot.
    """

    x: jax.Array
    M: jax.Array
    a: jax.Array
    v: jax.Array | None = None
    observer: tuple[float, ...] | None = eqx.field(static=True, default=None)

    COORD: ClassVar[str] = "x"

    def __check_init__(self):
        n = self.x.shape[-1]
        if (
            self.x.ndim != 2
            or self.M.shape != (n,)
            or jnp.shape(self.a) not in ((), (n,))
        ):
            raise ValueError(
                f"expected x (D, n), M (n,) and a () or (n,); got {self.x.shape}, "
                f"{self.M.shape} and {jnp.shape(self.a)}."
            )
        if self.v is not None and self.v.shape != self.x.shape:
            raise ValueError(
                f"v must have the shape of x {self.x.shape}; got {self.v.shape}."
            )
        if self.observer is not None and len(self.observer) != self.x.shape[0]:
            raise ValueError(
                f"observer must have {self.x.shape[0]} coordinates; "
                f"got {self.observer!r}."
            )

    @property
    def positions(self) -> jax.Array:
        return self.x

    def select(self, mask: ArrayLike) -> HaloCatalog:
        mask = self._mask(mask)
        return replace(
            self,
            x=self.x[:, mask],
            M=self.M[mask],
            a=self.a if jnp.ndim(self.a) == 0 else self.a[mask],
            v=None if self.v is None else self.v[:, mask],
        )

    # ----------------
    # --- REDSHIFT ---
    # ----------------

    def to_redshift_catalog(self, rsd: bool = True) -> RedshiftCatalog:
        """Sky angles and redshifts as seen by the light cone's observer.

        Angles from x - observer; z = 1 / a - 1, and with rsd the Doppler
        shift of the radial peculiar velocity, 1 + z_obs = (1 + z)(1 + v_r / c).
        Masses are not carried over: select before converting.
        """
        from cozcat import RedshiftCatalog

        if self.observer is None:
            raise ValueError("not a light-cone catalogue: observer is None.")
        if rsd and self.v is None:
            raise ValueError("rsd=True needs velocities; v is None.")
        r = self.x - jnp.asarray(self.observer)[:, None]
        z = 1.0 / self.a - 1.0
        if rsd:
            v_r = jnp.sum(self.v * r, axis=0) / jnp.linalg.norm(r, axis=0)
            z = (1.0 + z) * (1.0 + v_r / C_LIGHT) - 1.0
        return RedshiftCatalog.from_vectors(r, jnp.broadcast_to(z, (len(self),)))
