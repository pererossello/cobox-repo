from typing import Literal, Optional, TYPE_CHECKING
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp

from .ops import tensor_utils

if TYPE_CHECKING:
    from ..box.box import Box
    from .scalar import ScalarField

SymmetryLiteral = Literal["symmetric", "antisymmetric"]
SYMMETRIES: tuple[str, ...] = ("symmetric", "antisymmetric")


class TensorField(eqx.Module):
    data: jax.Array
    box: "Box" = eqx.field(static=True)
    has_hat: bool = eqx.field(static=True, default=False)
    symmetry: Optional[SymmetryLiteral] = eqx.field(static=True, default=None)

    def __init__(
        self,
        data: jax.Array,
        box: "Box",
        has_hat: bool = False,
        symmetry: Optional[SymmetryLiteral] = None,
    ):
        self.data = data
        self.box = box
        self.has_hat = has_hat
        self.symmetry = symmetry
        self._validate_inputs()

    @property
    def n_components(self) -> int:
        return tensor_utils._n_components(self.box.D, self.symmetry)

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return self.data.ndim

    # -----------
    # --- FFT ---
    # -----------

    def fft(self) -> "TensorField":
        if self.has_hat:
            return self
        # _fft_on_vec FFTs over the spatial axes for any leading-axis size.
        data_fft = self.box.fft_on_vec(self.data)
        return replace(self, data=data_fft, has_hat=True)

    def ifft(self) -> "TensorField":
        if not self.has_hat:
            return self
        data_ifft = self.box.ifft_on_vec(self.data)
        return replace(self, data=data_ifft, has_hat=False)

    # ------------------
    # --- LINEAR OPS ---
    # ------------------

    def trace(self) -> "ScalarField":
        """Trace (first invariant)."""
        return tensor_utils.trace(self)

    def transpose(self) -> "TensorField":
        """Transpose ``t^T``."""
        return tensor_utils.transpose(self)

    def symmetric_part(self) -> "TensorField":
        """Symmetric part ``(t + t^T) / 2``."""
        return tensor_utils.symmetric_part(self)

    def antisymmetric_part(self) -> "TensorField":
        """Antisymmetric part ``(t - t^T) / 2``."""
        return tensor_utils.antisymmetric_part(self)

    def traceless_part(self) -> "TensorField":
        """Traceless part ``T - (trace(T) / D) · I``."""
        return tensor_utils.traceless_part(self)

    # -----------------------------
    # --- POLYNOMIAL INVARIANTS ---
    # -----------------------------

    def trace_of_product(
        self,
        other: "TensorField",
        *,
        dealias: bool = False,
        N_iso: tuple[int, int] | None = None,
        out_N: int | Literal["full"] | None = None,
        return_hat: bool | None = None,
    ) -> "ScalarField":
        """Compute tr(self @ other)."""
        return tensor_utils.trace_of_product(
            self,
            other,
            dealias=dealias,
            N_iso=N_iso,
            out_N=out_N,
            return_hat=return_hat,
        )

    def second_invariant(
        self,
        other: Optional["TensorField"] = None,
        *,
        dealias: bool = False,
        N_iso: int | tuple[int, int] | None = None,
        out_N: int | Literal["full"] | None = None,
        return_hat: bool | None = None,
    ) -> "ScalarField":
        """Second invariant or its bilinear polarization."""
        return tensor_utils.second_invariant(
            self,
            other,
            dealias=dealias,
            N_iso=N_iso,
            out_N=out_N,
            return_hat=return_hat,
        )

    def det(
        self,
        *,
        dealias: bool = False,
        N_iso: int | None = None,
        out_N: int | Literal["full"] | None = None,
        return_hat: bool | None = None,
    ) -> "ScalarField":
        return tensor_utils.det(
            self,
            dealias=dealias,
            N_iso=N_iso,
            out_N=out_N,
            return_hat=return_hat,
        )

    def third_invariant(
        self,
        second: Optional["TensorField"] = None,
        third: Optional["TensorField"] = None,
        *,
        dealias: bool = False,
        N_iso: int | tuple[int, int] | tuple[int, int, int] | None = None,
        out_N: int | Literal["full"] | None = None,
        return_hat: bool | None = None,
    ) -> "ScalarField":
        """Determinant or its symmetric 3D polarization."""
        return tensor_utils.third_invariant(
            self,
            second,
            third,
            dealias=dealias,
            N_iso=N_iso,
            out_N=out_N,
            return_hat=return_hat,
        )

    # -------------------
    # --- EIGENVALUES ---
    # -------------------

    def eigenvalues(
        self,
        *,
        return_hat: bool = False,
    ) -> tuple["ScalarField", ...]:
        """Descending eigenvalues of a symmetric tensor."""
        return tensor_utils.eigenvalues(
            self,
            return_hat=return_hat,
        )

    # ------------------
    # --- VALIDATION ---
    # ------------------

    def _validate_inputs(self) -> None:
        from . import _validate

        _validate.validate_tensor_field(self)
