"""Composition of background and primordial parameters, presets and serialization."""

import json

import equinox as eqx
from jax.typing import ArrayLike

from ._presets import DEFAULT_COSMOS, DEFAULT_NAME, DefaultCosmosLiteral
from ._utils import a_of_z, z_of_a
from .background import BACKGROUND_PARAMS, BackgroundCosmo
from .primordial import PRIMORDIAL_PARAMS, PrimordialCosmo


class Cosmology(eqx.Module):
    """Complete parameters; access physics through .background and .primordial.

    Example: Cosmology.from_preset("planck18", h=0.70, As1e9=2.1).
    Background-only calculations receive the background block explicitly.
    """

    background: BackgroundCosmo
    primordial: PrimordialCosmo

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

    # ---------------
    # --- PRESETS ---
    # ---------------

    @classmethod
    def from_preset(
        cls, name: DefaultCosmosLiteral = DEFAULT_NAME, **overrides
    ) -> "Cosmology":
        """Build both blocks with flat, named parameter overrides.

        Construction accepts traced parameters; it performs no float coercion.
        """
        if name not in DEFAULT_COSMOS:
            raise KeyError(
                f"unknown cosmology preset {name!r}; available: {sorted(DEFAULT_COSMOS)}."
            )
        unknown = set(overrides) - set(BACKGROUND_PARAMS) - set(PRIMORDIAL_PARAMS)
        if unknown:
            raise ValueError(f"unknown cosmology parameters: {sorted(unknown)}.")
        preset = DEFAULT_COSMOS[name]
        background = {
            **preset["background"],
            **{k: v for k, v in overrides.items() if k in BACKGROUND_PARAMS},
        }
        primordial = {
            **preset["primordial"],
            **{k: v for k, v in overrides.items() if k in PRIMORDIAL_PARAMS},
        }
        return cls(BackgroundCosmo(**background), PrimordialCosmo(**primordial))

    # ---------------------
    # --- SERIALIZATION ---
    # ---------------------

    def to_dict(self) -> dict:
        """Write nested canonical parameters; host serialization, not a JIT operation."""
        return {
            "background": self.background.to_dict(),
            "primordial": self.primordial.to_dict(),
        }

    @classmethod
    def from_dict(cls, config: dict) -> "Cosmology":
        """Read nested parameters or a legacy flat configuration with As1e9.

        Flat/nested keys cannot be mixed. Missing required parameters are errors;
        reading a configuration does not silently fill it with a named preset.
        """
        if "background" in config or "primordial" in config:
            if set(config) != {"background", "primordial"}:
                raise ValueError(
                    "Nested Cosmology config requires exactly background and primordial."
                )
            return cls(
                BackgroundCosmo.from_dict(config["background"]),
                PrimordialCosmo.from_dict(config["primordial"]),
            )
        unknown = set(config) - set(BACKGROUND_PARAMS) - set(PRIMORDIAL_PARAMS)
        if unknown:
            raise ValueError(f"unknown keys in Cosmology config: {sorted(unknown)}.")
        return cls(
            BackgroundCosmo.from_dict(
                {k: v for k, v in config.items() if k in BACKGROUND_PARAMS}
            ),
            PrimordialCosmo.from_dict(
                {k: v for k, v in config.items() if k in PRIMORDIAL_PARAMS}
            ),
        )

    def to_json(self, path) -> None:
        with open(path, "w") as stream:
            json.dump(self.to_dict(), stream, indent=2)

    @classmethod
    def from_json(cls, path) -> "Cosmology":
        with open(path) as stream:
            return cls.from_dict(json.load(stream))
