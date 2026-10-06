from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import jax.numpy as jnp

from ...box.support import ModeSupport

if TYPE_CHECKING:
    from ..scalar import ScalarField
    from ..vector import VectorField


# -------------
# --- UTILS ---
# -------------


def _is_number(x) -> bool:
    """True for a python/numpy/jax scalar (0-d)."""
    if isinstance(x, (bool, int, float, complex)):
        return True
    return getattr(x, "ndim", None) == 0


# --------------------------------------
# --- ADD / SUBTRACT ON SCALAR FIELD ---
# --------------------------------------


def add_to_scalar_field(this: "ScalarField", that) -> "ScalarField":
    """ScalarField + ScalarField (compatible), or ScalarField + real scalar (real space)."""
    return _additive_scalar_field(this, that, jnp.add)


def sub_to_scalar_field(this: "ScalarField", that) -> "ScalarField":
    """ScalarField - ScalarField (compatible), or ScalarField - real scalar (real space)."""
    return _additive_scalar_field(this, that, jnp.subtract)


def _additive_scalar_field(this: "ScalarField", that, op) -> "ScalarField":
    from ..scalar import ScalarField

    if isinstance(that, ScalarField):
        if this.box != that.box:
            raise ValueError("ScalarFields live on different boxes.")
        if this.has_hat != that.has_hat:
            raise ValueError("has_hat mismatch; align with .fft()/.ifft().")

        return replace(
            this,
            data=op(this.data, that.data),
            support=ModeSupport.of_sum(this.support, that.support),
        )

    if _is_number(that):
        if this.has_hat:
            raise ValueError(
                "adding/substracting a number to Fourier-space ScalarField is not supported."
            )
        if jnp.iscomplexobj(that):
            raise ValueError(
                "adding/substracting a complex number to a ScalarField is not supported."
            )
        return replace(this, data=op(this.data, that))
    else:
        raise TypeError(f"unsupported operand for Field +/-: {type(that).__name__}")


# ---------------------------------
# --- MUL / DIV ON SCALAR FIELD ---
# ---------------------------------


def mul_to_scalar_field(this: "ScalarField", that) -> "ScalarField":
    """Field * scalar: real scalar (any space), or complex scalar (Fourier only)."""
    return _multiplicative_scalar_field(this, that, jnp.multiply)


def div_to_scalar_field(this: "ScalarField", that) -> "ScalarField":
    """Field / scalar: real scalar (any space), or complex scalar (Fourier only)."""
    return _multiplicative_scalar_field(this, that, jnp.true_divide)


def _multiplicative_scalar_field(this: "ScalarField", that, op) -> "ScalarField":
    from ..scalar import ScalarField

    if isinstance(that, ScalarField):
        raise ValueError("ScalarField {*,/} ScalarField is not supported via * or /")

    if _is_number(that):
        if jnp.iscomplexobj(that):
            raise ValueError(
                "multiplication with complex number would break "
                "the realness of a real-space field."
            )
        return replace(this, data=op(this.data, that))

    raise TypeError(
        f"unsupported operand for ScalarField * or /: {type(that).__name__}"
    )


# --------------------------------------
# --- ADD / SUBTRACT ON VECTOR FIELD ---
# --------------------------------------


def add_to_vector_field(this: "VectorField", that) -> "VectorField":
    """VectorField + VectorField (compatible)."""
    return _additive_vector_field(this, that, jnp.add)


def sub_to_vector_field(this: "VectorField", that) -> "VectorField":
    """VectorField - VectorField (compatible)."""
    return _additive_vector_field(this, that, jnp.subtract)


def _additive_vector_field(this: "VectorField", that, op) -> "VectorField":
    from ..vector import VectorField

    if isinstance(that, VectorField):
        if this.box != that.box:
            raise ValueError("VectorFields live on different boxes.")
        if this.has_hat != that.has_hat:
            raise ValueError("has_hat mismatch; align with .fft()/.ifft().")
        return replace(
            this,
            data=op(this.data, that.data),
            support=ModeSupport.of_sum(this.support, that.support),
        )

    if _is_number(that):
        raise TypeError(
            "VectorField +/- number is not supported: a scalar has no direction."
        )

    raise TypeError(f"unsupported operand for VectorField +/-: {type(that).__name__}")


# ---------------------------------
# --- MUL / DIV ON VECTOR FIELD ---
# ---------------------------------


def mul_to_vector_field(this: "VectorField", that) -> "VectorField":
    """VectorField * real scalar (any space)."""
    return _multiplicative_vector_field(this, that, jnp.multiply)


def div_to_vector_field(this: "VectorField", that) -> "VectorField":
    """VectorField / real scalar (any space)."""
    return _multiplicative_vector_field(this, that, jnp.true_divide)


def _multiplicative_vector_field(this: "VectorField", that, op) -> "VectorField":
    from ..vector import VectorField

    if isinstance(that, VectorField):
        raise ValueError("VectorField {*,/} VectorField is not supported via * or /")

    if _is_number(that):
        if jnp.iscomplexobj(that):
            raise ValueError(
                "multiplying/dividing by a complex scalar is not supported."
            )
        return replace(this, data=op(this.data, that))

    raise TypeError(
        f"unsupported operand for VectorField * or /: {type(that).__name__}"
    )
