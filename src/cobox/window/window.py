from __future__ import annotations

from numbers import Real
from typing import Literal, TypeGuard, get_args, TYPE_CHECKING
from ._separable_windows import (
    SeparableKindLiteral,
    SEPARABLE_KINDS,
    win_1D,
    win_hat_1D,
    is_separable_kind,
)
from ._isotropic_windows import (
    IsotropicKindLiteral,
    ISOTROPIC_KINDS,
    win_iso,
    win_hat_iso,
    is_isotropic_kind,
)

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

WindowKindLiteral = Literal[SeparableKindLiteral, IsotropicKindLiteral]
WINDOW_KINDS: tuple[str, ...] = get_args(WindowKindLiteral)


def is_window_kind(kind: str) -> TypeGuard[WindowKindLiteral]:
    return kind in WINDOW_KINDS


if TYPE_CHECKING:
    from ..box import Box


class Window(eqx.Module):
    kind: WindowKindLiteral = eqx.field(static=True)
    scale: float  # full width; a leaf, so it may be traced
    normalized: bool = eqx.field(static=True)  # whether integral over window is 1

    def __init__(
        self,
        kind: WindowKindLiteral,
        scale: float,
        normalized: bool,
    ):
        self.kind = kind
        self.scale = scale
        self.normalized = normalized

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if not is_window_kind(self.kind):
            raise ValueError(
                f"unknown window kind: {self.kind!r}; expected one of {WINDOW_KINDS}."
            )
        # A traced scale cannot be checked here.
        if isinstance(self.scale, Real) and self.scale <= 0:
            raise ValueError(f"scale must be positive; got {self.scale}.")

    @property
    def is_separable(self) -> bool:
        return self.kind in SEPARABLE_KINDS

    @property
    def is_isotropic(self) -> bool:
        return self.kind in ISOTROPIC_KINDS

    def win_hat_on_box(self, box: "Box") -> jax.Array:
        """Fourier-space window sampled on the box's k-grid (shape box.KSHAPE)."""
        if is_separable_kind(self.kind):
            win_hat = jnp.ones(box.KSHAPE)
            for k_ in box.k_grid:
                win_hat = win_hat * win_hat_1D(
                    self.kind, self.scale, k_, self.normalized
                )
            return win_hat
        if is_isotropic_kind(self.kind):
            return win_hat_iso(self.kind, self.scale, box.k, box.D, self.normalized)
        raise ValueError(f"no win_hat_on_box for kind {self.kind!r}")

    # ----------------
    # --- PROFILES ---
    # ----------------

    def win_profile(self, x: ArrayLike, D: int = 3) -> jax.Array:
        """Real-space window profile (radius for radial kinds)."""
        x = jnp.asarray(x)
        if is_separable_kind(self.kind):
            return win_1D(self.kind, self.scale, x, self.normalized)
        if is_isotropic_kind(self.kind):
            return win_iso(self.kind, self.scale, x, D, self.normalized)
        raise ValueError(f"no win_profile for kind {self.kind!r}")

    def win_profile_hat(self, k: ArrayLike, D: int = 3) -> jax.Array:
        """Fourier-space window profile (|k| for radial kinds)."""
        k = jnp.asarray(k)
        if is_separable_kind(self.kind):
            return win_hat_1D(self.kind, self.scale, k, self.normalized)
        if is_isotropic_kind(self.kind):
            return win_hat_iso(self.kind, self.scale, k, D, self.normalized)
        raise ValueError(f"no win_profile_hat for kind {self.kind!r}")

    # ---------------------
    # --- SERIALIZATION ---
    # ---------------------

    def to_dict(self) -> dict:
        from . import _serialize

        return _serialize.window_to_dict(self)

    @classmethod
    def from_dict(cls, config: dict) -> "Window":
        from . import _serialize

        return _serialize.window_from_dict(config)

    def to_yaml(self) -> str:
        from . import _serialize

        return _serialize.window_to_yaml(self)

    @classmethod
    def from_yaml(cls, s: str) -> "Window":
        from . import _serialize

        return _serialize.window_from_yaml(s, cls)
