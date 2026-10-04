"""Primordial scalar parameters; spectrum evaluation belongs to a separate model."""

import equinox as eqx
import jax
import jax.numpy as jnp

PRIMORDIAL_PARAMS = ("As1e9", "n_s", "k_pivot")


class PrimordialCosmo(eqx.Module):
    """Scalar amplitude As1e9 = 10^9 As at k_pivot (physical Mpc^-1), and tilt n_s.

    Parameters remain dynamic PyTree leaves, including the pivot.
    """

    As1e9: float
    n_s: float
    k_pivot: float = 0.05

    @property
    def As(self) -> float:
        """Physical amplitude As = 10^-9 As1e9."""
        return self.As1e9 * 1e-9

    @property
    def ln1e10As(self) -> jax.Array:
        """Planck convention ln(10^10 As) = ln(10 As1e9)."""
        return jnp.log(10.0 * self.As1e9)

    def to_dict(self) -> dict:
        """Serialize scalar parameters on the host, outside JAX transformations."""
        return {name: float(getattr(self, name)) for name in PRIMORDIAL_PARAMS}

    @classmethod
    def from_dict(cls, config: dict) -> "PrimordialCosmo":
        unknown = set(config) - set(PRIMORDIAL_PARAMS)
        if unknown:
            raise ValueError(
                f"unknown keys in PrimordialCosmo config: {sorted(unknown)}."
            )
        return cls(**config)
