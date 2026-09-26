"""Homogeneous wCDM background (late-time: no radiation, no neutrinos).

Conventions
- Primary parameters: h, Omega_b, Omega_cdm, Omega_k, n_s, As1e9, w0, wa.
    - Capital Omega_X = rho_X / rho_crit today (dimensionless)
    - lowercase omega_X = Omega_X h^2 (physical density).
- Dark energy: CPL, w(a) = w0 + wa (1 - a). LCDM for w0 = -1, wa = 0.
- As1e9 = 1e9 A_s at k_pivot = 0.05 Mpc^-1
"""

from typing import Literal
import json

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ._constants import H0, G_NEWTON, MPC_TO_M, M_SUN_TO_KG, c_light
from ._default_cosmos import DEFAULT_COSMOS, DEFAULT_NAME, DefaultCosmosLiteral

CosmoParamLiteral = Literal[
    "Omega_b", "Omega_cdm", "Omega_k", "n_s", "As1e9", "w0", "wa"
]
COSMO_PARAMS = ["Omega_b", "Omega_cdm", "Omega_k", "h", "n_s", "As1e9", "w0", "wa"]


class Cosmology(eqx.Module):
    h: float
    Omega_b: float
    Omega_cdm: float
    Omega_k: float
    n_s: float
    As1e9: float
    w0: float = -1.0
    wa: float = 0.0

    @classmethod
    def from_preset(
        cls, name: DefaultCosmosLiteral = DEFAULT_NAME, **overrides
    ) -> "Cosmology":
        """Build a Cosmology from a named preset (see ``_default_cosmos``).

        ``overrides`` perturb individual parameters on top of the preset, e.g.
        ``Cosmology.from_preset("planck18", h=0.70)``.
        """
        if name not in DEFAULT_COSMOS:
            raise KeyError(
                f"unknown cosmology preset {name!r}; "
                f"available: {sorted(DEFAULT_COSMOS)}."
            )
        return cls(**{**DEFAULT_COSMOS[name], **overrides})

    @property
    def Omega_de(self) -> float:
        return 1.0 - self.Omega_m - self.Omega_k

    @property
    def Omega_m(self) -> float:
        return self.Omega_b + self.Omega_cdm

    @property
    def omega_m(self) -> float:
        return self.Omega_m * self.h**2

    @property
    def omega_b(self) -> float:
        return self.Omega_b * self.h**2

    @property
    def omega_cdm(self) -> float:
        return self.Omega_cdm * self.h**2

    @property
    def omega_de(self) -> float:
        return self.Omega_de * self.h**2

    @property
    def omega_k(self) -> float:
        return self.Omega_k * self.h**2

    @property
    def is_flat(self) -> bool:
        return abs(self.Omega_k) < 1e-5

    @property
    def is_de_Lambda(self) -> bool:
        return self.w0 == -1.0 and self.wa == 0.0

    # ------------
    # --- FLRW ---
    # ------------

    def E_sq(self, a: ArrayLike) -> jax.Array:
        """Dimensionless Hubble rate squared E^2(a) = H^2(a) / H_0^2."""
        a = jnp.asarray(a)
        return (
            self.Omega_m * a**-3.0
            + self.Omega_k * a**-2.0
            + self.Omega_de * self._de_evolution(a)
        )

    def E(self, a: ArrayLike) -> jax.Array:
        """Dimensionless Hubble rate E(a) = H(a) / H_0."""
        return jnp.sqrt(self.E_sq(a))

    def H(self, a: ArrayLike) -> jax.Array:
        """Hubble rate H(a) in km/s/Mpc (H_0 = 100 h km/s/Mpc)."""
        return H0 * self.h * self.E(a)

    def Omega_m_of_a(self, a: ArrayLike) -> jax.Array:
        a = jnp.asarray(a)
        return self.Omega_m * a**-3.0 / self.E_sq(a)

    def Omega_de_of_a(self, a: ArrayLike) -> jax.Array:
        a = jnp.asarray(a)
        return self.Omega_de * self._de_evolution(a) / self.E_sq(a)

    def Omega_k_of_a(self, a: ArrayLike) -> jax.Array:
        a = jnp.asarray(a)
        return self.Omega_k * a**-2.0 / self.E_sq(a)

    def _de_evolution(self, a: ArrayLike) -> jax.Array:
        """CPL dark-energy density relative to today: rho_de(a) / rho_de(1)"""
        a = jnp.asarray(a)
        return a ** (-3.0 * (1.0 + self.w0 + self.wa)) * jnp.exp(
            -3.0 * self.wa * (1.0 - a)
        )

    def w_de(self, a: ArrayLike) -> jax.Array:
        """CPL dark-energy equation of state w(a) = w0 + wa (1 - a)."""
        a = jnp.asarray(a)
        return self.w0 + self.wa * (1.0 - a)

    def a_dot(self, a: ArrayLike) -> jax.Array:
        """Cosmic-time derivative da/dt = a H(a), in km/s/Mpc."""
        a = jnp.asarray(a)
        return a * self.H(a)

    def a_ddot(self, a: ArrayLike) -> jax.Array:
        """Second derivative d^2a/dt^2 in (km/s/Mpc)^2"""
        a = jnp.asarray(a)
        H0h_sq = (H0 * self.h) ** 2
        rho_m = self.Omega_m * a**-3.0  # matter: 1 + 3 w = 1
        rho_de = self.Omega_de * self._de_evolution(a) * (1.0 + 3.0 * self.w_de(a))
        return -0.5 * H0h_sq * a * (rho_m + rho_de)

    def q(self, a: ArrayLike) -> jax.Array:
        """Deceleration parameter q(a) = -a_ddot a / a_dot^2 (dimensionless).
        q > 0 decelerating, q < 0 accelerating."""
        a = jnp.asarray(a)
        return -self.a_ddot(a) * a / self.a_dot(a) ** 2

    # ---------------------
    # --- SERIALIZATION ---
    # ---------------------

    def to_dict(self) -> dict:
        dict_ = {name: float(getattr(self, name)) for name in COSMO_PARAMS}
        return dict_

    @classmethod
    def from_dict(cls, config: dict) -> "Cosmology":
        unknown = set(config) - set(COSMO_PARAMS)
        if unknown:
            raise ValueError(f"unknown keys in Cosmology config: {sorted(unknown)}.")
        return cls(**{name: float(value) for name, value in config.items()})

    def to_json(self, path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path) -> "Cosmology":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    # -----------------------
    # --- SPECIAL METHODS ---
    # -----------------------

    def __repr__(self) -> str:
        fields = ", ".join(
            f"{name}={getattr(self, name)!r}"
            for name in (
                "h",
                "Omega_b",
                "Omega_cdm",
                "Omega_k",
                "n_s",
                "As1e9",
                "w0",
                "wa",
            )
        )
        return f"Cosmology({fields})"
