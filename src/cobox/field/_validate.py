from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from .scalar import ScalarField
    from .vector import VectorField
    from .tensor import TensorField


def validate_field_common(field: ScalarField | VectorField | TensorField) -> None:
    """Validates properties shared by all field types (Scalar, Vector, Tensor)."""
    if not field.has_hat and jnp.iscomplexobj(field.data):
        raise ValueError(
            f"complex data provided to {type(field).__name__} in real-space"
        )


def validate_scalar_field(field: "ScalarField") -> None:
    validate_field_common(field)

    b = field.box
    expected_shape = b.KSHAPE if field.has_hat else b.SHAPE
    if field.data.shape != expected_shape:
        raise ValueError(
            f"ScalarField data shape {tuple(field.data.shape)!r} does not match "
            f"box shape {tuple(expected_shape)!r}."
        )


def validate_vector_field(field: "VectorField") -> None:
    validate_field_common(field)

    b = field.box
    comp_shape = b.KSHAPE if field.has_hat else b.SHAPE
    expected_shape = (b.D,) + comp_shape
    if field.data.shape != expected_shape:
        raise ValueError(
            f"VectorField data shape {tuple(field.data.shape)!r} does not match "
            f"expected (D, *box_shape) = {expected_shape!r}."
        )


def validate_tensor_field(field: "TensorField") -> None:
    from .tensor import SYMMETRIES

    b = field.box
    if field.symmetry not in (None,) + SYMMETRIES:
        raise ValueError(
            f"symmetry must be one of {(None,) + SYMMETRIES}; got "
            f"{field.symmetry!r}."
        )
    if field.symmetry == "antisymmetric" and b.D == 1:
        raise ValueError(
            "antisymmetric rank-2 tensor is identically zero in D=1; use "
            "symmetry=None or symmetry='symmetric'."
        )

    component_shape = b.KSHAPE if field.has_hat else b.SHAPE
    expected = (field.n_components,) + component_shape
    if field.data.shape != expected:
        raise ValueError(
            f"data shape {tuple(field.data.shape)!r} does not match the "
            f"expected (n_components={field.n_components}, *box_shape) = "
            f"{expected!r}."
        )

    if not field.has_hat and jnp.iscomplexobj(field.data):
        raise ValueError("complex data provided in real-space")
