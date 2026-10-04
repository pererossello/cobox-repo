"""Linear matter density transfer; cosmological parameters are supplied per call."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from cobox.spectrum import Spectrum

from ..cosmology.background import BackgroundCosmo
from ..growth import Growth
from ._matter import (
    MATTER_TRANSFER_DISPATCH,
    MATTER_TRANSFER_KINDS,
    MatterTransferKindLiteral,
)


class MatterTransfer(Spectrum):
    """Matter density transfer T_delta(k, a) = T_0(k) D(a), scale-independent growth.

    kind: symbolic_pofk (emulator), eisenstein_hu (with BAO), eisenstein_hu_nw
    (no wiggles) or bbks. T_0 is the kind's present-day transfer; D(a) is the
    injected growth, normalized to D(1) = 1. symbolic_pofk fixes T_0 with its own
    calibrated growth amplitude; the physical fits use the injected growth's g(1).
    k in h/Mpc; T is dimensionless, with delta(k, a) = T(k, a) R(k).
    """

    growth: Growth = eqx.field(default_factory=Growth)
    kind: MatterTransferKindLiteral = eqx.field(static=True, default="symbolic_pofk")

    def __check_init__(self):
        if self.kind not in MATTER_TRANSFER_DISPATCH:
            raise ValueError(
                f"unknown matter transfer kind: {self.kind!r}; expected one of {MATTER_TRANSFER_KINDS}."
            )
        if not isinstance(self.growth, Growth):
            raise TypeError("growth must be a Growth instance.")

    def __call__(
        self, k: ArrayLike, a: ArrayLike, background: BackgroundCosmo
    ) -> jax.Array:
        """T(k, a); k and a broadcast. Zero for k <= 0."""
        k = jnp.asarray(k, dtype=float)
        k_safe = jnp.where(k > 0, k, 1.0)
        T0 = MATTER_TRANSFER_DISPATCH[self.kind](background, k_safe, self.growth)
        return jnp.where(k > 0, T0, 0.0) * self.growth.D(a, background)

    def to_dict(self) -> dict:
        return {"type": "matter", "kind": self.kind, "growth": self.growth.to_dict()}

    @classmethod
    def from_dict(cls, config: dict) -> "MatterTransfer":
        config = dict(config)
        if config.pop("type", "matter") != "matter":
            raise ValueError("MatterTransfer config must have type 'matter'.")
        unknown = set(config) - {"kind", "growth"}
        if unknown:
            raise ValueError(
                f"unknown keys in MatterTransfer config: {sorted(unknown)}."
            )
        if "growth" in config:
            config["growth"] = Growth.from_dict(config["growth"])
        return cls(**config)
