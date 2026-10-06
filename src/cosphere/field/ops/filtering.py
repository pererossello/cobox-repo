"""Isotropic harmonic filtering of real scalar shell fields."""

from dataclasses import replace
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

if TYPE_CHECKING:
    from ..shell_field import ShellField


def filter_ell(field: "ShellField", response, *, iter: int = 3) -> "ShellField":
    """Apply b_ell and return a harmonic field; see ShellField.filter_ell."""
    shell = field.shell
    response = response(shell.ell_1D) if callable(response) else response
    response = jnp.asarray(response)
    if response.shape not in ((), (shell.L,)) or jnp.iscomplexobj(response):
        raise ValueError("response must be real and scalar or have shape (shell.L,).")
    response = eqx.error_if(
        response, jnp.any(~jnp.isfinite(response)), "response must be finite."
    )
    response = jnp.broadcast_to(response, (shell.L,))
    field = field.sht(iter=iter)
    return replace(
        field, data=jnp.where(shell.is_mode, field.data * response[:, None], 0)
    )
