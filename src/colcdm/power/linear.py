"""Linear power spectra from a primordial spectrum and one or two transfers."""

from dataclasses import replace

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from cobox.spectrum import PowerSpectrum, Spectrum

from ..cosmology import Cosmology, PrimordialCosmo
from ..transfer import transfer_from_dict
from .primordial import PrimordialSpectrum

_LINEAR_POWER_FIELDS = ("primordial", "transfer_x", "transfer_y")


class LinearPower(PowerSpectrum):
    """Linear auto- or cross-spectrum P_XY(k; a, a') = P_R(k) T_X(k, a) T_Y(k, a').

    k in h/Mpc and P in (Mpc/h)^3; the primordial spectrum is evaluated at the
    physical k_phys = h k. transfer_y=None uses transfer_x; a_prime=None uses a.
    P = 0 for k <= 0. Configuration only: the cosmology is passed on every call,
    including to the inherited PowerSpectrum methods, e.g. sigma8(a, cosmology).
    """

    primordial: PrimordialSpectrum
    transfer_x: Spectrum
    transfer_y: Spectrum | None = None

    def __check_init__(self):
        if not isinstance(self.primordial, PrimordialSpectrum):
            raise TypeError("primordial must be a PrimordialSpectrum instance.")
        for name in ("transfer_x", "transfer_y"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Spectrum):
                raise TypeError(f"{name} must be a transfer, e.g. MatterTransfer.")

    def __call__(
        self,
        k: ArrayLike,
        a: ArrayLike,
        cosmology: Cosmology,
        a_prime: ArrayLike | None = None,
    ) -> jax.Array:
        k = jnp.asarray(k, dtype=float)
        k_safe = jnp.where(k > 0, k, 1.0)
        background = cosmology.background
        transfer_y = self.transfer_x if self.transfer_y is None else self.transfer_y
        a_prime = a if a_prime is None else a_prime
        P_R = self.primordial(background.h * k_safe, cosmology.primordial)
        P = (
            background.h**3
            * P_R
            * self.transfer_x(k_safe, a, background)
            * transfer_y(k_safe, a_prime, background)
        )
        return jnp.where(k > 0, P, 0.0)

    def primordial_from_sigma8(
        self,
        target: ArrayLike,
        cosmology: Cosmology,
        *,
        a: ArrayLike = 1.0,
        **kw,
    ) -> PrimordialCosmo:
        """Primordial parameters with As1e9 rescaled so that sigma8(a) = target.

        Exact because P is linear in As; kw are the quadrature settings of sigma8.
        """
        current = self.sigma8(a, cosmology, **kw)
        As1e9 = cosmology.primordial.As1e9 * (target / current) ** 2
        return replace(cosmology.primordial, As1e9=As1e9)

    # ---------------------
    # --- SERIALIZATION ---
    # ---------------------

    def to_dict(self) -> dict:
        return {
            "primordial": self.primordial.to_dict(),
            "transfer_x": self.transfer_x.to_dict(),
            "transfer_y": (
                None if self.transfer_y is None else self.transfer_y.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, config: dict) -> "LinearPower":
        unknown = set(config) - set(_LINEAR_POWER_FIELDS)
        if unknown:
            raise ValueError(f"unknown keys in LinearPower config: {sorted(unknown)}.")
        missing = {"primordial", "transfer_x"} - set(config)
        if missing:
            raise ValueError(f"missing keys in LinearPower config: {sorted(missing)}.")
        transfer_y = config.get("transfer_y")
        return cls(
            PrimordialSpectrum.from_dict(config["primordial"]),
            transfer_from_dict(config["transfer_x"]),
            None if transfer_y is None else transfer_from_dict(transfer_y),
        )

    def to_yaml(self) -> str:
        import yaml

        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_yaml(cls, s: str) -> "LinearPower":
        import yaml

        return cls.from_dict(yaml.safe_load(s))
