from __future__ import annotations

from abc import abstractmethod
from math import isqrt
from typing import Literal, Optional, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from .support import ModeSupport

if TYPE_CHECKING:
    from .box import Box


IsoModeMaskLiteral = Literal["nyquist_open", "nyquist_closed", "custom"]


# ---------------------
# --- ISO MODE MASK ---
# ---------------------


class ModeMask(eqx.Module):
    """Fourier selection and an optional guaranteed per-axis support bound."""

    @abstractmethod
    def mask(self, box: "Box") -> jax.Array:
        """Boolean selection with shape box.KSHAPE."""

    def support(self, box: "Box") -> Optional[ModeSupport]:
        """None unless this restriction guarantees a tighter grid bound."""
        return None


class IsoModeMask(ModeMask):
    """Isotropic ``|k|`` cut.

    - ``nyquist_open``:   ``|k_idx|^2 <  (N/2)^2``  (drops the Nyquist corner)
    - ``nyquist_closed``: ``|k_idx|^2 <= (N/2)^2``  (keeps it)
    - ``custom``:         ``k_range[0] <= |k| < k_range[1]`` (``k_range[0]=None`` -> 0; ``k_range[1]=None`` -> inf``)
    """

    mode: IsoModeMaskLiteral = eqx.field(static=True)
    k_range: Optional[tuple[Optional[float], Optional[float]]] = eqx.field(
        static=True, default=None
    )

    def __init__(
        self,
        mode: IsoModeMaskLiteral,
        k_range: Optional[tuple[Optional[float], Optional[float]]] = None,
    ):
        self.mode = mode
        self.k_range = k_range
        self._validate()

    def mask(self, box: "Box") -> jax.Array:
        if self.mode == "nyquist_open":
            return box.k_idx_sq < (box.N // 2) ** 2
        elif self.mode == "nyquist_closed":
            return box.k_idx_sq <= (box.N // 2) ** 2
        else:
            assert self.k_range is not None

            k_lo = 0.0 if self.k_range[0] is None else float(self.k_range[0])
            k_hi = jnp.inf if self.k_range[1] is None else float(self.k_range[1])
            k = box.k
            return (k >= k_lo) & (k < k_hi)

    def support(self, box: "Box") -> Optional[ModeSupport]:
        """Tightest ModeSupport of the kept modes on box, or None.

        A kept mode has |k_idx|^2 <= n2_max, so each coordinate is at most
        isqrt(n2_max), which some kept mode on an axis attains. None when
        there is no upper cut or the cut does not fall below the box's Nyquist.
        """
        if self.mode == "nyquist_open":
            n2_max = (box.N // 2) ** 2 - 1
        elif self.mode == "nyquist_closed":
            n2_max = (box.N // 2) ** 2
        else:
            assert self.k_range is not None
            k_hi = self.k_range[1]
            if k_hi is None:
                return None
            # Inflated so float rounding at a shell edge stays conservative.
            n2_max = int((float(k_hi) / box.K_RES) ** 2 * (1 + 1e-6))
        return ModeSupport(isqrt(n2_max)).on(box)

    def _validate(self) -> None:
        if self.mode not in ("nyquist_open", "nyquist_closed", "custom"):
            raise ValueError(f"unknown iso mode: {self.mode!r}.")
        if self.mode == "custom":
            if self.k_range is None:
                raise ValueError("IsoModeMask(mode='custom') requires k_range.")
            k_lo, k_hi = self.k_range
            if (self.k_range[0] is None) and (self.k_range[1] is None):
                raise ValueError(
                    f"IsoModeMask(mode='custom') requires either or both k_range[0] and k_range[1] to be float. Got ({k_lo}, {k_hi})"
                )
            k_lo = float(k_lo) if k_lo is not None else None
            k_hi = float(k_hi) if k_hi is not None else None

            if k_hi is not None:
                if k_hi <= 0.0:
                    raise ValueError(f"k_range upper bound must be > 0; got {k_hi}.")
            if k_lo is not None:
                if k_lo < 0.0:
                    raise ValueError(f"k_range lower bound must be >= 0; got {k_lo}.")
            if k_lo is not None and k_hi is not None:
                if k_lo >= k_hi:
                    raise ValueError(
                        f"k_range requires k_lo < k_hi; got ({k_lo}, {k_hi})."
                    )
        elif self.k_range is not None:
            raise ValueError(
                "k_range is only meaningful with mode='custom'; "
                f"got mode={self.mode!r}, k_range={self.k_range!r}."
            )

    # ---------------------
    # --- SERIALIZATION ---
    # ---------------------

    def to_dict(self) -> dict:
        from . import _serialize

        return _serialize.iso_mask_to_dict(self)

    @classmethod
    def from_dict(cls, config: dict) -> "IsoModeMask":
        from . import _serialize

        return _serialize.iso_mask_from_dict(config, cls)

    def to_yaml(self) -> str:
        from . import _serialize

        return _serialize.iso_mask_to_yaml(self)

    @classmethod
    def from_yaml(cls, s: str) -> "IsoModeMask":
        from . import _serialize

        return _serialize.iso_mask_from_yaml(s, cls)


# ---------------------
# --- BOX MODE MASK ---
# ---------------------


class BoxModeMask(ModeMask):
    boxes: tuple[tuple[int, int], ...] = eqx.field(static=True)
    insides: tuple[bool, ...] = eqx.field(static=True)

    def __init__(self, boxes, insides=None):
        self.boxes = tuple((int(n), int(r)) for n, r in boxes)
        self.insides = (
            tuple([True] * len(self.boxes))
            if insides is None
            else tuple(bool(b) for b in insides)
        )
        self._validate_static()

    def mask(self, box: "Box") -> jax.Array:
        self._validate_against_box(box)
        m = jnp.ones(box.KSHAPE, dtype=bool)
        for (N_p, ratio), inside in zip(self.boxes, self.insides):
            sub = self._single_box_mask(box, N_p, ratio)
            m = m & (sub if inside else ~sub)
        return m

    def support(self, box: "Box") -> Optional[ModeSupport]:
        """Conservative bound from the intersection of interior selections.

        Multiples of ratio on the compressed axis include its upper boundary,
        giving ratio * floor(N_p/2). The other axes use strict boundaries and
        cannot exceed it. Exterior selections do not supply an upper bound;
        their intersection can only remove modes from an interior selection.
        """
        self._validate_against_box(box)
        bounds = [
            ratio * (N_p // 2)
            for (N_p, ratio), inside in zip(self.boxes, self.insides)
            if inside
        ]
        return ModeSupport(min(bounds)).on(box) if bounds else None

    @staticmethod
    def _single_box_mask(box: "Box", N_p: int, ratio: int) -> jax.Array:
        threshold = N_p * ratio / 2.0
        D = box.D
        m = jnp.ones(box.KSHAPE, dtype=bool)
        for a, axis in enumerate(box.k_idx_axes):
            in_range = (
                (axis <= threshold) if a == D - 1 else (jnp.abs(axis) < threshold)
            )
            m = m & in_range & (axis % ratio == 0)
        return m

    def _validate_static(self) -> None:
        """Checks that don't need a ``Box``."""
        if len(self.insides) != len(self.boxes):
            raise ValueError(
                f"insides has length {len(self.insides)} but boxes has length "
                f"{len(self.boxes)}."
            )
        if len(set(self.boxes)) != len(self.boxes):
            raise ValueError("duplicate (N', ratio) pairs in boxes.")
        for N_p, ratio in self.boxes:
            if N_p <= 0:
                raise ValueError(f"N' must be > 0; got {N_p}.")
            if ratio < 1:
                raise ValueError(f"ratio must be >= 1; got {ratio}.")

    def _validate_against_box(self, box: "Box") -> None:
        """Checks that need the target ``Box`` (sub-boxes must fit inside it)."""
        for N_p, ratio in self.boxes:
            if N_p * ratio > box.N:
                raise ValueError(
                    f"N' * ratio must be <= box.N={box.N}; got N'={N_p}, "
                    f"ratio={ratio} (product={N_p * ratio})."
                )
