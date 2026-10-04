from __future__ import annotations

from numbers import Integral
from typing import Optional, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from .shell import Shell


class EllModeMask(eqx.Module):
    """Multipole cut ``ell_range[0] <= ell < ell_range[1]`` on a Shell's half-plane.

    ``None`` ends are open (0 and L). Entries with m > ell are never kept.
    """

    ell_range: tuple[Optional[int], Optional[int]] = eqx.field(static=True)

    def __init__(self, ell_range: tuple[Optional[int], Optional[int]]):
        self.ell_range = tuple(ell_range)  # type: ignore
        self._validate()

    def mask(self, shell: "Shell") -> jax.Array:
        lo, hi = self.ell_range
        ell = shell.ell_axis
        keep = jnp.ones_like(ell, dtype=bool)
        if lo is not None:
            keep = keep & (ell >= lo)
        if hi is not None:
            keep = keep & (ell < hi)
        return keep & shell.is_mode

    def _validate(self) -> None:
        if len(self.ell_range) != 2:
            raise ValueError(f"ell_range must be a pair; got {self.ell_range!r}.")
        lo, hi = self.ell_range
        if lo is None and hi is None:
            raise ValueError("ell_range needs at least one bound; got (None, None).")
        for v in (lo, hi):
            if v is not None and (isinstance(v, bool) or not isinstance(v, Integral) or v < 0):
                raise ValueError(f"ell_range bounds must be non-negative ints; got {v!r}.")
        if lo is not None and hi is not None and lo >= hi:
            raise ValueError(f"ell_range requires lo < hi; got ({lo}, {hi}).")

    # ---------------------
    # --- SERIALIZATION ---
    # ---------------------

    def to_dict(self) -> dict:
        from . import _serialize

        return _serialize.ell_mask_to_dict(self)

    @classmethod
    def from_dict(cls, config: dict) -> "EllModeMask":
        from . import _serialize

        return _serialize.ell_mask_from_dict(config, cls)

    def to_yaml(self) -> str:
        import yaml

        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_yaml(cls, s: str) -> "EllModeMask":
        import yaml

        return cls.from_dict(yaml.safe_load(s))
