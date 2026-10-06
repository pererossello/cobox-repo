"""Protohalo finder: excursion set of spheres on the linear density field."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from cobox.field import ScalarField

from ...growth import Growth
from ._exclusion import EXCLUSION_DISPATCH
from ._patches import measure
from ._radii import resolve_radii
from .catalog import ProtohaloCatalog
from .collapse import Collapse

if TYPE_CHECKING:
    from cobox.box import Box, ModeSupport

    from ...cosmology.background import BackgroundCosmo

SeedsLiteral = Literal["nodes"]
SEEDS_KINDS = ("nodes",)

ExclusionLiteral = Literal["full"]
EXCLUSION_KINDS = ("full",)

_FINDER_FIELDS = (
    "growth",
    "collapse",
    "seeds",
    "exclusion",
    "dlnR",
    "R_min_factor",
    "R_max",
)


class ProtohaloFinder(eqx.Module):
    """Protohalo options. No data: find(delta0, a, background) -> ProtohaloCatalog.

    Spheres are tested from the largest radius down: a seed hosts a
    protohalo of radius R when its sphere has collapsed by a and the
    exclusion rule admits it. M = background.M_of_R_lag(R).

    seeds: nodes (every node is a candidate centre).
    exclusion: full (accepted spheres never overlap).
    Radii are log spaced, step <= dlnR, from R_min = R_min_factor L / N_eff
    to R_max (None: L / 4), in Mpc/h.
    """

    growth: Growth = eqx.field(default_factory=Growth)
    collapse: Collapse = eqx.field(default_factory=Collapse)
    seeds: SeedsLiteral = eqx.field(static=True, default="nodes")
    exclusion: ExclusionLiteral = eqx.field(static=True, default="full")
    dlnR: float = eqx.field(static=True, default=0.1)
    R_min_factor: float = eqx.field(static=True, default=2.0)
    R_max: float | None = eqx.field(static=True, default=None)

    def __check_init__(self):
        self._validate()

    def find(
        self,
        delta0: ScalarField,
        a: ArrayLike,
        background: BackgroundCosmo,
    ) -> ProtohaloCatalog:
        """Protohalos of delta0, the linear field extrapolated to a = 1.

        a is a scalar (snapshot) or one value per node (light cone, e.g.
        LPTBasis.get_a_lc on the same box). delta0.support sets R_min.
        """
        box = delta0.box
        if box.D != 3:
            raise ValueError("ProtohaloFinder requires D == 3.")
        a = _validate_a(a, box)
        D = self.growth.D(a, background)
        delta0 = delta0.fft()  # one FFT, shared by every radius

        exclusion = EXCLUSION_DISPATCH[self.exclusion].start(box)
        for R in self.radii(box, support=delta0.support):
            patch = measure(delta0, R, self.collapse.needs)
            collapsed = self.collapse.collapsed(patch, a, D, background)
            exclusion = exclusion.update(patch, collapsed)
        idx, R = exclusion.finish()

        return ProtohaloCatalog(
            q=idx * box.R,
            M=background.M_of_R_lag(R),
            a=jnp.broadcast_to(a, box.SHAPE)[tuple(idx)],
        )

    def radii(
        self, box: Box, *, support: ModeSupport | None = None
    ) -> tuple[float, ...]:
        """Descending radii in Mpc/h tested on box for a field with support."""
        return resolve_radii(
            box,
            support,
            dlnR=self.dlnR,
            R_min_factor=self.R_min_factor,
            R_max=self.R_max,
        )

    # ---------------------
    # --- SERIALIZATION ---
    # ---------------------

    def to_dict(self) -> dict[str, Any]:
        nested = {"growth", "collapse"}
        return {
            name: (
                getattr(self, name).to_dict() if name in nested else getattr(self, name)
            )
            for name in _FINDER_FIELDS
        }

    @classmethod
    def from_dict(cls, config: dict) -> ProtohaloFinder:
        config = dict(config)
        unknown = set(config) - set(_FINDER_FIELDS)
        if unknown:
            raise ValueError(
                f"unknown keys in ProtohaloFinder config: {sorted(unknown)}."
            )
        if "growth" in config:
            config["growth"] = Growth.from_dict(config["growth"])
        if "collapse" in config:
            config["collapse"] = Collapse.from_dict(config["collapse"])
        return cls(**config)

    def to_yaml(self) -> str:
        import yaml

        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_yaml(cls, s: str) -> ProtohaloFinder:
        import yaml

        return cls.from_dict(yaml.safe_load(s))

    # ------------------
    # --- VALIDATION ---
    # ------------------

    def _validate(self) -> None:
        if not isinstance(self.growth, Growth):
            raise TypeError("growth must be a Growth instance.")
        if not isinstance(self.collapse, Collapse):
            raise TypeError("collapse must be a Collapse instance.")
        if self.seeds not in SEEDS_KINDS:
            raise ValueError(
                f"unknown seeds: {self.seeds!r}; expected one of {SEEDS_KINDS}."
            )
        if self.exclusion not in EXCLUSION_DISPATCH:
            raise ValueError(
                f"unknown exclusion: {self.exclusion!r}; expected one of {EXCLUSION_KINDS}."
            )
        if not self.dlnR > 0:
            raise ValueError(f"dlnR must be positive; got {self.dlnR}.")
        if not self.R_min_factor > 0:
            raise ValueError(f"R_min_factor must be positive; got {self.R_min_factor}.")
        if self.R_max is not None and not self.R_max > 0:
            raise ValueError(f"R_max must be positive or None; got {self.R_max}.")


def _validate_a(a: ArrayLike, box: Box) -> jax.Array:
    """Scalar or one scale factor per node."""
    a = jnp.asarray(a, dtype=float)
    if a.shape not in ((), box.SHAPE):
        raise ValueError(f"a must be scalar or have shape {box.SHAPE}; got {a.shape}.")
    return a
