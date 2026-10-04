from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from .shell_field import ShellField


def validate_shell_field(field: "ShellField") -> None:
    if not field.has_hat and jnp.iscomplexobj(field.data):
        raise ValueError("complex data provided to ShellField in pixel space")

    s = field.shell
    expected_shape = s.HSHAPE if field.has_hat else s.SHAPE
    if field.data.shape != expected_shape:
        raise ValueError(
            f"ShellField data shape {tuple(field.data.shape)!r} does not match "
            f"shell shape {tuple(expected_shape)!r}."
        )
