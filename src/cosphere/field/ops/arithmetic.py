from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from ..shell_field import ShellField


# -------------
# --- UTILS ---
# -------------


def _is_number(x) -> bool:
    """True for a python/numpy/jax scalar (0-d)."""
    if isinstance(x, (bool, int, float, complex)):
        return True
    return getattr(x, "ndim", None) == 0


# -------------------------------------
# --- ADD / SUBTRACT ON SHELL FIELD ---
# -------------------------------------


def add_to_shell_field(this: "ShellField", that) -> "ShellField":
    """ShellField + ShellField (compatible), or ShellField + real scalar (pixel space)."""
    return _additive_shell_field(this, that, jnp.add)


def sub_to_shell_field(this: "ShellField", that) -> "ShellField":
    """ShellField - ShellField (compatible), or ShellField - real scalar (pixel space)."""
    return _additive_shell_field(this, that, jnp.subtract)


def _additive_shell_field(this: "ShellField", that, op) -> "ShellField":
    from ..shell_field import ShellField

    if isinstance(that, ShellField):
        if this.shell != that.shell:
            raise ValueError("ShellFields live on different shells.")
        if this.has_hat != that.has_hat:
            raise ValueError("has_hat mismatch; align with .sht()/.isht().")

        return replace(this, data=op(this.data, that.data))

    if _is_number(that):
        if this.has_hat:
            raise ValueError(
                "adding/subtracting a number to a harmonic-space ShellField is not supported."
            )
        if jnp.iscomplexobj(that):
            raise ValueError(
                "adding/subtracting a complex number to a ShellField is not supported."
            )
        return replace(this, data=op(this.data, that))
    else:
        raise TypeError(f"unsupported operand for ShellField +/-: {type(that).__name__}")


# --------------------------------
# --- MUL / DIV ON SHELL FIELD ---
# --------------------------------


def mul_to_shell_field(this: "ShellField", that) -> "ShellField":
    """ShellField * real scalar (any space)."""
    return _multiplicative_shell_field(this, that, jnp.multiply)


def div_to_shell_field(this: "ShellField", that) -> "ShellField":
    """ShellField / real scalar (any space)."""
    return _multiplicative_shell_field(this, that, jnp.true_divide)


def _multiplicative_shell_field(this: "ShellField", that, op) -> "ShellField":
    from ..shell_field import ShellField

    if isinstance(that, ShellField):
        raise ValueError("ShellField {*,/} ShellField is not supported via * or /")

    if _is_number(that):
        if jnp.iscomplexobj(that):
            raise ValueError(
                "multiplication with complex number would break "
                "the realness of a pixel-space field."
            )
        return replace(this, data=op(this.data, that))

    raise TypeError(f"unsupported operand for ShellField * or /: {type(that).__name__}")
