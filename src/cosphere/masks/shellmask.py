from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..shell import Shell


class ShellMask(eqx.Module):
    """Included sky pixels on a HEALPix RING shell: True keeps, False excludes.

    Boolean or binary integer input, stored as Boolean data of shape (NPIX,).
    Logical operations require matching pixelizations (nside); harmonic band
    limits do not affect coverage. Results retain the left operand's shell.
    The mask describes coverage, not weights or an estimator's normalization.
    """

    data: jax.Array
    shell: Shell = eqx.field(static=True)

    def __init__(self, data: ArrayLike, shell: Shell):
        self.data = jnp.asarray(data)
        self.shell = shell
        self.data = self.validate().data.astype(bool)

    @classmethod
    def band(
        cls,
        shell: Shell,
        half_width: ArrayLike,
        pole: tuple[ArrayLike, ArrayLike] = (0.0, 0.0),
    ) -> "ShellMask":
        """Pixels whose centres lie within half_width of the great circle with this pole.

        Angles in radians; pole is (theta, phi) in the shell's frame. The default
        pole theta = 0 gives the band around the shell's equator. Use ~ to exclude it.
        """
        theta, phi = pole
        sin_theta = jnp.sin(theta)
        p = jnp.stack(
            [sin_theta * jnp.cos(phi), sin_theta * jnp.sin(phi), jnp.cos(theta)]
        )
        return cls(jnp.abs(p @ shell.unit_vectors) <= jnp.sin(half_width), shell)

    @property
    def fsky(self) -> jax.Array:
        """Fraction of included sky area; HEALPix pixels have equal area."""
        return jnp.mean(self.data)

    # ----------------------------
    # --- LOGICAL OPERATIONS ---
    # ----------------------------

    def __and__(self, other):
        if not isinstance(other, ShellMask):
            return NotImplemented
        return replace(self, data=self.data & self._matching(other))

    def __or__(self, other):
        if not isinstance(other, ShellMask):
            return NotImplemented
        return replace(self, data=self.data | self._matching(other))

    def __invert__(self):
        return replace(self, data=~self.data)

    def _matching(self, other: "ShellMask") -> jax.Array:
        if self.shell.nside != other.shell.nside:
            raise ValueError("masks must share the same pixelization (nside).")
        return other.data

    # ----------------
    # --- PLOTTING ---
    # ----------------

    def paint(
        self,
        ax=None,
        *,
        colors: tuple[str, str] = ("#3b6ea8", "0.85"),
        colorbar: bool = False,
        **kwargs,
    ):
        """Paint the mask in Mollweide projection, like ShellField.paint.

        colors are (excluded, included). Other kwargs go to ShellField.paint.
        """
        from matplotlib.colors import ListedColormap

        from ..field import ShellField
        from ..field._paint import paint

        field = ShellField(self.data.astype(float), self.shell)
        return paint(
            field,
            ax=ax,
            cmap=ListedColormap(colors),
            colorbar=colorbar,
            vmin=0,
            vmax=1,
            **kwargs,
        )

    # ------------------
    # --- VALIDATION ---
    # ------------------

    def validate(self) -> "ShellMask":
        """Check geometry and binary data; return the checked mask for use under JIT."""
        if not isinstance(self.shell, Shell):
            raise TypeError("shell must be a Shell.")
        if self.data.shape != self.shell.SHAPE:
            raise ValueError(
                f"mask data must have shape {self.shell.SHAPE}; got {self.data.shape}."
            )
        if jnp.issubdtype(self.data.dtype, jnp.integer):
            return eqx.error_if(
                self,
                jnp.any((self.data != 0) & (self.data != 1)),
                "integer mask values must be 0 or 1.",
            )
        if not jnp.issubdtype(self.data.dtype, jnp.bool_):
            raise TypeError("mask data must have Boolean or integer dtype.")
        return self
