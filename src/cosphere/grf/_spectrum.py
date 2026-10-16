"""C_ell values for a Shell's multipoles, taken from an AngularSpectrum."""

from typing import TYPE_CHECKING

import jax

from ..spectrum import AngularSpectrum

if TYPE_CHECKING:
    from ..shell import Shell


def shell_values(cl: AngularSpectrum, shell: "Shell", trailing: tuple) -> jax.Array:
    """cl.values for 0 <= ell < shell.L; cl must reach at least shell.ELL_MAX."""
    if not isinstance(cl, AngularSpectrum):
        raise TypeError(f"cl must be an AngularSpectrum; got {type(cl).__name__}.")
    if cl.values.shape[1:] != trailing:
        raise ValueError(
            f"cl.values must have shape (ell_max + 1, *{trailing}); "
            f"got {cl.values.shape}."
        )
    if cl.ell_max < shell.ELL_MAX:
        raise ValueError(
            f"cl reaches ell_max = {cl.ell_max}; the shell needs {shell.ELL_MAX}."
        )
    return cl.values[: shell.L]
