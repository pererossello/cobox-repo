"""Background parameters and FLRW calculations (matter, curvature and CPL dark energy).

Capital Omega values are present-day density fractions; omega = Omega * h**2.
Distances use Mpc/h and H(a) uses km/s/Mpc. Radiation is not included.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ._constants import H0, RHO_CRIT_0
from ._utils import a_of_z, z_of_a

BACKGROUND_PARAMS = ("h", "Omega_b", "Omega_cdm", "Omega_k", "w0", "wa")


class BackgroundCosmo(eqx.Module):
    """Expansion and distance parameters, independently usable without primordial data."""

    h: float
    Omega_b: float
    Omega_cdm: float
    Omega_k: float
    w0: float = -1.0
    wa: float = 0.0

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
    def rho_m(self) -> float:
        """Comoving mean matter density Omega_m rho_crit,0, in (M_sun/h) / (Mpc/h)^3."""
        return self.Omega_m * RHO_CRIT_0

    @property
    def is_flat(self) -> jax.Array:
        """Boolean array, so it can be checked under jit (eqx.error_if)."""
        return jnp.abs(self.Omega_k) < 1e-5

    @property
    def is_de_Lambda(self) -> jax.Array:
        """Boolean array, so it can be checked under jit (eqx.error_if)."""
        return (jnp.asarray(self.w0) == -1.0) & (jnp.asarray(self.wa) == 0.0)

    # ---------------------------
    # --- REDSHIFT CONVERSION ---
    # ---------------------------

    @staticmethod
    def a_of_z(z: ArrayLike) -> ArrayLike:
        """Scale factor a = 1 / (1 + z); independent of the parameters."""
        return a_of_z(z)

    @staticmethod
    def z_of_a(a: ArrayLike) -> ArrayLike:
        """Redshift z = 1 / a - 1; independent of the parameters."""
        return z_of_a(a)

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

    def dlnH_dlna(self, a: ArrayLike) -> jax.Array:
        """Logarithmic slope d ln H / d ln a = -(3 Om(a) + 2 Ok(a) + 3 (1 + w) Ode(a)) / 2."""
        a = jnp.asarray(a)
        return -0.5 * (
            3.0 * self.Omega_m_of_a(a)
            + 2.0 * self.Omega_k_of_a(a)
            + 3.0 * (1.0 + self.w_de(a)) * self.Omega_de_of_a(a)
        )

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

    # -----------------
    # --- DISTANCES ---
    # -----------------

    def chi_of_a(
        self,
        a: ArrayLike,
        *,
        n_quad: int = 4096,
        a_min: float = 1e-5,
    ) -> jax.Array:
        """Radial comoving distance chi(a) in Mpc/h, observed today.
        chi(a) = (c / 100) integral_a^1 da' / (a'^2 E(a')).
        """
        from . import _distances

        table = _distances.get_distance_table(self, n_quad=n_quad, a_min=a_min)
        return _distances.chi_of_a(a, *table)

    def a_of_chi(
        self,
        chi: ArrayLike,
        *,
        n_quad: int = 4096,
        a_min: float = 1e-5,
    ) -> jax.Array:
        """Scale factor at radial comoving distance chi [Mpc/h]."""
        from . import _distances

        table = _distances.get_distance_table(self, n_quad=n_quad, a_min=a_min)
        return _distances.a_of_chi(chi, *table)

    def dchi_da(self, a: ArrayLike) -> jax.Array:
        """Analytic derivative of chi(a): -(c / 100) / (a^2 E(a)), in Mpc/h."""
        from . import _distances

        return _distances.dchi_da(self, a)

    def chi_of_varrho(self, varrho: ArrayLike) -> jax.Array:
        """Isotropic coordinate radius -> radial comoving distance [Mpc/h].
        Flat geometry is the identity."""
        from . import _distances

        return _distances.chi_of_varrho(self, varrho)

    def dchi_dvarrho(self, varrho: ArrayLike) -> jax.Array:
        """Analytic derivative of chi_of_varrho (1 in flat geometry)."""
        from . import _distances

        return _distances.dchi_dvarrho(self, varrho)

    def varrho_of_chi(self, chi: ArrayLike) -> jax.Array:
        """Radial comoving distance -> isotropic coordinate radius [Mpc/h]."""
        from . import _distances

        return _distances.varrho_of_chi(self, chi)

    def S_k(self, chi: ArrayLike) -> jax.Array:
        """Transverse comoving distance [Mpc/h] from radial distance chi.
        Flat geometry returns chi."""
        from . import _distances

        return _distances.S_k(self, chi)

    # -----------------------
    # --- LAGRANGIAN MASS ---
    # -----------------------

    def M_of_R_lag(self, R: ArrayLike) -> jax.Array:
        """Mass [M_sun/h] in a comoving sphere of Lagrangian radius R [Mpc/h]."""
        return 4.0 / 3.0 * jnp.pi * self.rho_m * jnp.asarray(R) ** 3

    def R_lag_of_M(self, M: ArrayLike) -> jax.Array:
        """Lagrangian radius [Mpc/h] of a comoving sphere of mass M [M_sun/h]."""
        return (3.0 * jnp.asarray(M) / (4.0 * jnp.pi * self.rho_m)) ** (1.0 / 3.0)

    def to_dict(self) -> dict:
        """Serialize scalar parameters on the host, outside JAX transformations."""
        return {name: float(getattr(self, name)) for name in BACKGROUND_PARAMS}

    @classmethod
    def from_dict(cls, config: dict) -> "BackgroundCosmo":
        unknown = set(config) - set(BACKGROUND_PARAMS)
        if unknown:
            raise ValueError(
                f"unknown keys in BackgroundCosmo config: {sorted(unknown)}."
            )
        return cls(**config)
