from typing import Literal, Optional, TYPE_CHECKING
from dataclasses import replace

import equinox as eqx
import jax

from ._validate import normalize_support
from .ops import tensor_utils

if TYPE_CHECKING:
    from ..box import Box, ModeMask, ModeSupport
    from .scalar import ScalarField

SymmetryLiteral = Literal["symmetric", "antisymmetric"]
SYMMETRIES: tuple[str, ...] = ("symmetric", "antisymmetric")


class TensorField(eqx.Module):
    data: jax.Array
    box: "Box" = eqx.field(static=True)
    has_hat: bool = eqx.field(static=True, default=False)
    symmetry: Optional[SymmetryLiteral] = eqx.field(static=True, default=None)
    support: Optional["ModeSupport"] = eqx.field(static=True, default=None)

    def __init__(
        self,
        data: jax.Array,
        box: "Box",
        has_hat: bool = False,
        symmetry: Optional[SymmetryLiteral] = None,
        support: int | Optional["ModeSupport"] = None,
    ):
        self.data = data
        self.box = box
        self.has_hat = has_hat
        self.symmetry = symmetry
        self.support = normalize_support(support, box)
        self._validate_inputs()

    # ---------------
    # --- SUPPORT ---
    # ---------------

    def with_support(self, support: int | Optional["ModeSupport"]) -> "TensorField":
        """Declare the band limit shared by all components (trusted, not checked)."""
        return replace(self, support=support)

    def resample(self, N: int) -> "TensorField":
        """Fourier crop or zero-pad onto an N grid; exact if the support fits."""
        from .ops.modes import resample

        return resample(self, N)

    def restrict(self, mask: "ModeMask") -> "TensorField":
        """Zero the modes outside mask (Fourier space); the support tightens."""
        from .ops.modes import restrict

        return restrict(self, mask)

    def drop_corner_modes(self) -> "TensorField":
        """Keep only |k_idx| < N/2: drops the Nyquist planes and the cube corners."""
        from ..box import IsoModeMask

        return self.restrict(IsoModeMask("nyquist_open"))

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
        out_N: int | Literal["full"] | None = None,
        return_hat: bool | None = None,
    ) -> "ScalarField":
        """Compute tr(self @ other)."""
        return tensor_utils.trace_of_product(
            self,
            other,
            dealias=dealias,
            out_N=out_N,
            return_hat=return_hat,
        )

    def trace_of_products(
        self,
        *others: "TensorField",
        dealias: bool = False,
        out_N: int | Literal["full"] | None = None,
        return_hat: bool | None = None,
    ) -> "ScalarField":
        """Compute tr(self @ others[0] @ ... @ others[-1])."""
        return tensor_utils.trace_of_products(
            (self,) + others,
            dealias=dealias,
            out_N=out_N,
            return_hat=return_hat,
        )

    def second_invariant(
        self,
        other: Optional["TensorField"] = None,
        *,
        dealias: bool = False,
        out_N: int | Literal["full"] | None = None,
        return_hat: bool | None = None,
    ) -> "ScalarField":
        """Second invariant or its bilinear polarization."""
        return tensor_utils.second_invariant(
            self,
            other,
            dealias=dealias,
            out_N=out_N,
            return_hat=return_hat,
        )

    def det(
        self,
        *,
        dealias: bool = False,
        out_N: int | Literal["full"] | None = None,
        return_hat: bool | None = None,
    ) -> "ScalarField":
        return tensor_utils.det(
            self,
            dealias=dealias,
            out_N=out_N,
            return_hat=return_hat,
        )

    def third_invariant(
        self,
        second: Optional["TensorField"] = None,
        third: Optional["TensorField"] = None,
        *,
        dealias: bool = False,
        out_N: int | Literal["full"] | None = None,
        return_hat: bool | None = None,
    ) -> "ScalarField":
        """Determinant or its symmetric 3D polarization."""
        return tensor_utils.third_invariant(
            self,
            second,
            third,
            dealias=dealias,
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

    # -----------------------
    # --- SPECIAL METHODS ---
    # -----------------------

    def __repr__(self):
        space = "fourier space" if self.has_hat else "real space"
        return (
            f"TensorField(shape={self.shape}, {space}, symmetry={self.symmetry!r}, "
            f"box={self.box!r}, support={self.support!r})"
        )

    # -----------
    # --- I/O ---
    # -----------

    def to_h5(self, path) -> None:
        """Write this field (data + static config) to an HDF5 file."""
        from ._io import save_field

        save_field(self, path)

    @classmethod
    def from_h5(cls, path) -> "TensorField":
        """Load a TensorField from HDF5 (inverse of ``to_h5``)."""
        from ._io import load_field

        return load_field(path, cls)

    # ------------------
    # --- VALIDATION ---
    # ------------------

    def _validate_inputs(self) -> None:
        from . import _validate

        _validate.validate_tensor_field(self)
