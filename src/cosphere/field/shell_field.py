from dataclasses import replace
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import numpy as np

from .ops import arithmetic

if TYPE_CHECKING:
    from ..shell import Shell


class ShellField(eqx.Module):
    """A real scalar field on a Shell: pixel values (NPIX,), or with has_hat the
    half-plane harmonic coefficients (L, L)."""

    data: jax.Array
    shell: "Shell" = eqx.field(static=True)
    has_hat: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        data: jax.Array,
        shell: "Shell",
        has_hat: bool = False,
    ):
        self.data = data
        self.shell = shell
        self.has_hat = has_hat
        self._validate_inputs()

    # ------------------
    # --- HARMONIC ---
    # ------------------

    def sht(self, iter: int = 3) -> "ShellField":
        if self.has_hat:
            return self
        data_sht = self.shell.sht(self.data, iter=iter)
        return replace(self, data=data_sht, has_hat=True)

    def isht(self) -> "ShellField":
        if not self.has_hat:
            return self
        data_isht = self.shell.isht(self.data)
        return replace(self, data=data_isht, has_hat=False)

    # ------------------------
    # --- BASIC ARITHMETIC ---
    # ------------------------

    def __neg__(self):
        return replace(self, data=-self.data)

    def __add__(self, other):
        return arithmetic.add_to_shell_field(self, other)

    def __radd__(self, other):
        return arithmetic.add_to_shell_field(self, other)

    def __sub__(self, other):
        return arithmetic.sub_to_shell_field(self, other)

    def __rsub__(self, other):
        return -arithmetic.sub_to_shell_field(self, other)

    def __mul__(self, other):
        return arithmetic.mul_to_shell_field(self, other)

    def __rmul__(self, other):
        return arithmetic.mul_to_shell_field(self, other)

    def __truediv__(self, other):
        return arithmetic.div_to_shell_field(self, other)

    def __rtruediv__(self, other):
        raise TypeError("c / field is not supported.")

    # ---------------------
    # --- ARRAY SURFACE ---
    # ---------------------

    __array_ufunc__ = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return self.data.ndim

    def __array__(self):
        return np.asarray(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __repr__(self):
        space = "harmonic space" if self.has_hat else "pixel space"
        return f"ShellField(shape={self.shape}, {space}, shell={self.shell!r})"

    # -----------
    # --- I/O ---
    # -----------

    def to_h5(self, path) -> None:
        """Write this field (data + static config) to an HDF5 file."""
        from ._io import save_shell_field

        save_shell_field(self, path)

    @classmethod
    def from_h5(cls, path) -> "ShellField":
        """Load a ShellField from HDF5 (inverse of ``to_h5``)."""
        from ._io import load_shell_field

        return load_shell_field(path)

    # ------------
    # --- PLOT ---
    # ------------

    def paint(self, ax=None, **kwargs):
        from ._paint import paint

        return paint(self, ax=ax, **kwargs)

    # ------------------
    # --- VALIDATION ---
    # ------------------

    def _validate_inputs(self) -> None:
        from . import _validate

        _validate.validate_shell_field(self)
