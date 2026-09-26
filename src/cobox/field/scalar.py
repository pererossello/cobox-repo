from typing import Optional, TYPE_CHECKING
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..window import Window
from .vector import VectorField
from .tensor import TensorField
from .ops import arithmetic
from .ops import field_product
from .ops.convolution import convolve_scalar, deconvolve_scalar
from .ops.calculus import (
    GradientKernelLiteral,
    LaplacianKernelLiteral,
    grad_kernel as _grad_kernel,
    laplacian_kernel as _laplacian_kernel,
)
from .ops.interpolation import (
    InterpolationMethodLiteral,
    scalar_interpolate_at,
)

if TYPE_CHECKING:
    from ..box import Box


class ScalarField(eqx.Module):
    data: jax.Array
    box: "Box" = eqx.field(static=True)
    has_hat: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        data: jax.Array,
        box: "Box",
        has_hat: bool = False,
    ):
        self.data = data
        self.box = box
        self.has_hat = has_hat
        self._validate_inputs()

    # ---------------
    # --- FOURIER ---
    # ---------------

    def fft(self):
        if self.has_hat:
            return self
        data_fft = self.box.fft(self.data)
        return replace(self, data=data_fft, has_hat=True)

    def ifft(self):
        if not self.has_hat:
            return self
        data_ifft = self.box.ifft(self.data)
        return replace(self, data=data_ifft, has_hat=False)

    def fft_full(self):
        if self.has_hat:
            raise ValueError("Field must be en real space")
        return self.box.fft_full(self.data)

    # --------------------
    # --- (DE)CONVOLVE ---
    # --------------------

    def convolve(self, window: Window) -> "ScalarField":
        return convolve_scalar(self, window)

    def deconvolve(
        self,
        window: Window,
        regularize: bool = False,
        eps: float = 1e-5,
    ) -> "ScalarField":
        return deconvolve_scalar(self, window=window, regularize=regularize, eps=eps)

    # ---------------------
    # --- INTERPOLATION ---
    # ---------------------

    def interpolate_at(
        self,
        positions: jax.Array,
        *,
        method: InterpolationMethodLiteral = "linear",
        eps: float = 1e-8,
    ) -> jax.Array:
        return scalar_interpolate_at(self, positions, method=method, eps=eps)

    # ----------------
    # --- CALCULUS ---
    # ----------------

    def partial(
        self,
        axis: int,
        grad_kernel: GradientKernelLiteral = "spectral",
    ) -> "ScalarField":
        if not (0 <= axis < self.box.D):
            raise ValueError(f"axis {axis} is out of range for a {self.box.D}D box.")
        f = self.fft()
        gk = _grad_kernel(self.box.k_axes[axis], self.box.R, kernel=grad_kernel)
        return replace(f, data=gk * f.data)

    def grad(
        self,
        grad_kernel: GradientKernelLiteral = "spectral",
    ) -> "VectorField":
        f = self.fft()
        data = jnp.stack(
            [
                _grad_kernel(self.box.k_axes[a], self.box.R, kernel=grad_kernel)
                * f.data
                for a in range(self.box.D)
            ],
            axis=0,
        )
        return VectorField(
            data=data,
            box=f.box,
            has_hat=True,
        )

    def hessian(
        self,
        grad_kernel: GradientKernelLiteral = "spectral",
    ) -> "TensorField":
        f = self.fft()
        D = self.box.D
        Gs = [
            _grad_kernel(self.box.k_axes[a], self.box.R, kernel=grad_kernel)
            for a in range(D)
        ]
        components = [Gs[k] * Gs[j] * f.data for k in range(D) for j in range(k, D)]
        data = jnp.stack(components, axis=0)
        return TensorField(
            data=data,
            box=f.box,
            has_hat=True,
            symmetry="symmetric",
        )

    def laplacian(
        self,
        laplacian_kernel: LaplacianKernelLiteral = "spectral",
    ) -> "ScalarField":
        f = self.fft()
        lap_op = sum(
            _laplacian_kernel(f.box.k_axes[a], f.box.R, kernel=laplacian_kernel)
            for a in range(f.box.D)
        )
        return replace(f, data=lap_op * f.data)

    def inv_laplacian(
        self,
        laplacian_kernel: LaplacianKernelLiteral = "spectral",
    ) -> "ScalarField":
        f = self.fft()
        lap_op = sum(
            _laplacian_kernel(f.box.k_axes[a], f.box.R, kernel=laplacian_kernel)
            for a in range(f.box.D)
        )
        inv_op = jnp.where(lap_op != 0, 1.0 / lap_op, 0.0)
        return replace(f, data=f.data * inv_op)

    def grad_inv_laplacian(
        self,
        grad_kernel: GradientKernelLiteral = "spectral",
        laplacian_kernel: LaplacianKernelLiteral = "spectral",
    ) -> "VectorField":
        return self.inv_laplacian(laplacian_kernel=laplacian_kernel).grad(
            grad_kernel=grad_kernel
        )

    # ----------------------------------------
    # --- FIELD & FIELD PRODUCT ARITHMETIC ---
    # ----------------------------------------

    def product(
        self,
        other: "ScalarField",
        *,
        dealias: bool = False,
        N_iso: tuple[int, int] | None = None,
        out_N: field_product.OutputN = None,
        return_hat: bool | None = None,
    ) -> "ScalarField":
        """Multiply two fields, optionally with dealiasing."""
        return field_product.product(
            self,
            other,
            dealias=dealias,
            N_iso=N_iso,
            out_N=out_N,
            return_hat=return_hat,
        )

    def tri_product(
        self,
        second: "ScalarField",
        third: "ScalarField",
        *,
        dealias: bool = False,
        N_iso: tuple[int, int, int] | None = None,
        out_N: field_product.OutputN = None,
        return_hat: bool | None = None,
    ) -> "ScalarField":
        """Multiply three fields without intermediate projection."""
        return field_product.tri_product(
            self,
            second,
            third,
            dealias=dealias,
            N_iso=N_iso,
            out_N=out_N,
            return_hat=return_hat,
        )

    def square(
        self,
        *,
        dealias: bool = False,
        N_iso: int | None = None,
        out_N: field_product.OutputN = None,
        return_hat: bool | None = None,
    ) -> "ScalarField":
        """Square a field, optionally with dealiasing."""
        return field_product.square(
            self,
            dealias=dealias,
            N_iso=N_iso,
            out_N=out_N,
            return_hat=return_hat,
        )

    def cube(
        self,
        *,
        dealias: bool = False,
        N_iso: int | None = None,
        out_N: field_product.OutputN = None,
        return_hat: bool | None = None,
    ) -> "ScalarField":
        """Cube a field without intermediate projection."""
        return field_product.cube(
            self,
            dealias=dealias,
            N_iso=N_iso,
            out_N=out_N,
            return_hat=return_hat,
        )

    # ------------------------
    # --- BASIC ARITHMETIC ---
    # ------------------------

    def __neg__(self):
        return replace(self, data=-self.data)

    def __add__(self, other):
        return arithmetic.add_to_scalar_field(self, other)

    def __radd__(self, other):
        return arithmetic.add_to_scalar_field(self, other)

    def __sub__(self, other):
        return arithmetic.sub_to_scalar_field(self, other)

    def __rsub__(self, other):
        return -arithmetic.sub_to_scalar_field(self, other)

    def __mul__(self, other):
        return arithmetic.mul_to_scalar_field(self, other)

    def __rmul__(self, other):
        return arithmetic.mul_to_scalar_field(self, other)

    def __truediv__(self, other):
        return arithmetic.div_to_scalar_field(self, other)

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
        space = "fourier space" if self.has_hat else "real space"
        return f"Field(shape={self.shape}, {space}, box={self.box!r})"

    # -----------
    # --- I/O ---
    # -----------

    def to_h5(self, path) -> None:
        """Write this field (data + static config) to an HDF5 file."""
        from ._io import save_scalar_field

        save_scalar_field(self, path)

    @classmethod
    def from_h5(cls, path) -> "ScalarField":
        """Load a ScalarField from HDF5 (inverse of ``to_h5``)."""
        from ._io import load_scalar_field

        return load_scalar_field(path)

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

        _validate.validate_scalar_field(self)
