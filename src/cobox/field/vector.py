from typing import Optional, TYPE_CHECKING
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..window import Window

from .ops import field_product
from .ops import arithmetic
from .ops.calculus import (
    GradientKernelLiteral,
    grad_kernel as _grad_kernel,
)
from .ops.convolution import convolve_vector, deconvolve_vector
from .ops.interpolation import (
    InterpolationMethodLiteral,
    vector_interpolate_at,
)

if TYPE_CHECKING:
    from ..box import Box
    from .scalar import ScalarField
    from .tensor import TensorField


class VectorField(eqx.Module):
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

    # -----------
    # --- FFT ---
    # -----------

    def fft(self):
        if self.has_hat:
            return self
        data_fft = self.box.fft_on_vec(self.data)
        return replace(self, data=data_fft, has_hat=True)

    def ifft(self):
        if not self.has_hat:
            return self
        data_ifft = self.box.ifft_on_vec(self.data)
        return replace(self, data=data_ifft, has_hat=False)

    # ------------------
    # --- COMPONENTS ---
    # ------------------

    def component(self, axis: int) -> "ScalarField":
        from .scalar import ScalarField

        if not (0 <= axis < self.box.D):
            raise ValueError(
                f"axis {axis} is out of range for a {self.box.D}D box "
                f"(valid: 0 to {self.box.D - 1})"
            )
        return ScalarField(
            data=self.data[axis],
            box=self.box,
            has_hat=self.has_hat,
        )

    # --------------------
    # --- (DE)CONVOLVE ---
    # --------------------

    def convolve(self, window: Window) -> "VectorField":
        return convolve_vector(self, window)

    def deconvolve(
        self, window: Window, regularize: bool = False, eps: float = 1e-5
    ) -> "VectorField":
        return deconvolve_vector(self, window=window, regularize=regularize, eps=eps)

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
        return vector_interpolate_at(self, positions, method=method, eps=eps)

    # ----------------
    # --- CALCULUS ---
    # ----------------

    def div(self, grad_kernel: GradientKernelLiteral = "spectral") -> "ScalarField":
        from .scalar import ScalarField

        v = self.fft()
        data = sum(
            (
                _grad_kernel(v.box.k_axes[a], v.box.R, kernel=grad_kernel) * v.data[a]
                for a in range(v.box.D)
            ),
            start=jnp.zeros_like(v.data[0]),
        )
        return ScalarField(
            data=data,
            box=v.box,
            has_hat=True,
        )

    def curl(
        self, grad_kernel: GradientKernelLiteral = "spectral"
    ) -> "ScalarField | VectorField":
        from .ops.vector_utils import curl as _curl

        return _curl(self, grad_kernel=grad_kernel)

    def inverse_curl(self) -> "VectorField":
        from .ops.vector_utils import inverse_curl as _inverse_curl

        return _inverse_curl(self)

    def jacobian(
        self,
        grad_kernel: GradientKernelLiteral = "spectral",
        *,
        claim_symmetric: bool = False,
    ) -> "TensorField":
        from .tensor import TensorField

        v = self.fft()
        D = v.box.D
        Gs = [
            _grad_kernel(v.box.k_axes[a], v.box.R, kernel=grad_kernel) for a in range(D)
        ]
        components = []
        if claim_symmetric:
            # Upper triangle only: (k, j) with k <= j.
            for k in range(D):
                for j in range(k, D):
                    components.append(Gs[j] * v.data[k])
        else:
            # Full D x D, row-major: (0,0), (0,1), ..., (D-1, D-1).
            for k in range(D):
                for j in range(D):
                    components.append(Gs[j] * v.data[k])
        data_hat = jnp.stack(components, axis=0)
        return TensorField(
            data=data_hat,
            box=v.box,
            has_hat=True,
            symmetry="symmetric" if claim_symmetric else None,
        )

    # ----------------------------------------
    # --- FIELD & FIELD PRODUCT ARITHMETIC ---
    # ----------------------------------------

    def dot_product(
        self,
        other: "VectorField",
        *,
        dealias: bool = False,
        N_iso: tuple[int, int] | None = None,
        out_N: field_product.OutputN = None,
        return_hat: bool | None = None,
    ) -> "ScalarField":
        """Vector dot product, optionally with dealiasing."""
        from .ops.vector_utils import dot_product

        return dot_product(
            self,
            other,
            dealias=dealias,
            N_iso=N_iso,
            out_N=out_N,
            return_hat=return_hat,
        )

    def cross_product(
        self,
        other: "VectorField",
        *,
        dealias: bool = False,
        N_iso: tuple[int, int] | None = None,
        out_N: field_product.OutputN = None,
        return_hat: bool | None = None,
    ) -> "ScalarField | VectorField":
        """Cross product: scalar in 2D, vector in 3D."""
        from .ops.vector_utils import cross_product

        return cross_product(
            self,
            other,
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
        return arithmetic.add_to_vector_field(self, other)

    def __radd__(self, other):
        return arithmetic.add_to_vector_field(self, other)

    def __sub__(self, other):
        return arithmetic.sub_to_vector_field(self, other)

    def __rsub__(self, other):
        return -arithmetic.sub_to_vector_field(self, other)

    def __mul__(self, other):
        return arithmetic.mul_to_vector_field(self, other)

    def __rmul__(self, other):
        return arithmetic.mul_to_vector_field(self, other)

    def __truediv__(self, other):
        return arithmetic.div_to_vector_field(self, other)

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
        return f"VecField(shape={self.shape}, {space}, " f"box={self.box!r})"

    # -----------
    # --- I/O ---
    # -----------

    def to_h5(self, path) -> None:
        """Write this field (data + static config) to an HDF5 file."""
        from ._io import save_vector_field

        save_vector_field(self, path)

    @classmethod
    def from_h5(cls, path) -> "VectorField":
        """Load a VectorField from HDF5 (inverse of ``to_h5``)."""
        from ._io import load_vector_field

        return load_vector_field(path)

    # ------------------
    # --- VALIDATION ---
    # ------------------

    def _validate_inputs(self) -> None:
        from . import _validate

        _validate.validate_vector_field(self)
