from __future__ import annotations

from dataclasses import replace
from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax
import jax.numpy as jnp

from cobox.box import Box
from cobox.field import ScalarField
from cobox.fluid.mass_assign._native import deposit_native
from cobox.fluid.mass_assign._scaled import deposit_scaled, k_max_for

if TYPE_CHECKING:
    from cobox.field import VectorField
    from cobox.field.ops.interpolation import InterpolationMethodLiteral

MeshConventionLiteral = Literal["node", "cell"]
BSplineScaleLiteral = Literal["particles", "target_box"]


class Particles(eqx.Module):
    """Particles with unwrapped positions x on a periodic box.

    Supplied x must be unwrapped if displacement relative to q is needed.
    Use wrapped_position for coordinates inside the periodic box.
    """

    box: Box = eqx.field(static=True)
    shift: float | None = eqx.field(static=True, default=None)
    x: jax.Array | None = None

    def __init__(
        self,
        box: Box,
        shift: float | None = None,
        x: jax.Array | None = None,
    ):
        self.box = box
        self.shift = 0.0 if shift is None else float(shift)
        self.x = self.q if x is None else x

    @property
    def D(self) -> int:
        return self.box.D

    @property
    def N(self) -> int:
        return self.box.N

    @property
    def ND(self) -> int:
        return self.N**self.box.D

    @property
    def R(self) -> float:
        return self.box.L / self.N

    @property
    def q_axes(self) -> tuple:
        """Per-axis Lagrangian positions."""
        assert self.shift is not None
        return tuple(x + self.shift * self.R for x in self.box.x_axes)

    @property
    def q_grid(self) -> jax.Array:
        """Lagrangian positions on the full lattice grid, shape (D, *shape)."""
        assert self.shift is not None
        return jnp.stack(self.box.x_grid, axis=0) + self.shift * self.R

    @property
    def q(self) -> jax.Array:
        """Lagrangian positions flattened to (D, N**D)."""
        return self.q_grid.reshape(self.box.D, -1)

    @property
    def displacement(self) -> jax.Array:
        """Total displacement from the Lagrangian lattice, including crossings."""
        assert self.x is not None
        return self.x - self.q

    @property
    def wrapped_position(self) -> jax.Array:
        """Eulerian position reduced to the periodic box [0, L)."""
        assert self.x is not None
        return self.x % self.box.L

    # -------------
    # --- DRIFT ---
    # -------------

    def displace(self, displacement: jax.Array, wrap: bool = False):
        """Add a displacement while retaining boundary crossings in x.

        The legacy wrap keyword accepts False for compatibility. Wrapping x
        would discard information needed by the displacement property.
        """
        if wrap:
            raise ValueError("wrap=True discards displacement; use wrapped_position.")
        assert self.x is not None
        return replace(self, x=self.x + displacement)

    def displace_from_vector_field(
        self,
        vector_field: VectorField,
        interpolation: InterpolationMethodLiteral | None,
        wrap: bool = False,
    ):
        """Sample a vector field and retain the resulting unwrapped position.

        interpolation=None reads the field on its own grid nodes, i.e. at the
        Lagrangian lattice q. Otherwise the field is interpolated at the
        current wrapped position x. The two agree only while x == q.
        """
        assert self.x is not None
        if interpolation is None:
            if vector_field.box != self.box:
                raise ValueError(
                    "interpolation=None requires VectorField box == self.box; "
                    f"got {vector_field.box!r} and {self.box!r}."
                )
            if self.shift != 0.0:
                raise ValueError(
                    "interpolation=None requires shift == 0, since the lattice "
                    f"is then off the field grid; got shift={self.shift}."
                )
            disp = vector_field.ifft().data.reshape(vector_field.box.D, -1)
        else:
            disp = vector_field.interpolate_at(
                self.wrapped_position, method=interpolation
            )

        return self.displace(disp, wrap=wrap)

    # -------------------
    # --- MASS ASSIGN ---
    # -------------------

    def deposit(
        self,
        N: int,
        bspline_order: int,
        bspline_scale: float | BSplineScaleLiteral = "particles",
        mesh_convention: MeshConventionLiteral = "node",
        weights: jax.Array | None = None,
        return_contrast: bool = False,
    ) -> ScalarField:
        """Integrate separable particle clouds over periodic target cells.Literal["particles", "target_box"]

        Order 0 is a point mass (NGP). Orders 1, 2, and 3 use a top-hat,
        triangular, and quadratic B-spline cloud, respectively.

        The scale is the physical width of each constituent top-hat. "particles"
        uses self.box.R, "target_box" uses the target cell width, and a
        numeric value supplies a positive physical length.

        "node" centers cells on j * target_box.R; "cell" centers them on
        (j + 0.5) * target_box.R.
        """
        assert self.x is not None

        order_bspline_dict = {0: "ngp", 1: "cic", 2: "tsc", 3: "pcs"}
        if bspline_order not in order_bspline_dict:
            raise ValueError(
                f"bspline_order must be 0, 1, 2, or 3; got {bspline_order!r}."
            )

        target_box = Box(N=N, L=self.box.L, D=self.box.D)
        idx_pos = self.wrapped_position / target_box.R
        bspline_str = order_bspline_dict[bspline_order]

        if bspline_order == 0:
            data = deposit_native(
                idx_pos, weights, "ngp", target_box.SHAPE, mesh_convention
            )
        else:
            if isinstance(bspline_scale, str):
                if bspline_scale == "particles":
                    scale_in_cells = target_box.N / self.box.N
                elif bspline_scale == "target_box":
                    scale_in_cells = 1.0
            elif isinstance(bspline_scale, Real):
                cloud_scale = float(bspline_scale)
                if not isfinite(cloud_scale) or cloud_scale <= 0:
                    raise ValueError("bspline_scale must be positive and finite.")
                scale_in_cells = cloud_scale / target_box.R
            else:
                raise TypeError(
                    "bspline_scale must be 'particles', 'target_box', "
                    "or a real numeric physical length."
                )

            if bspline_order in (1, 2, 3) and scale_in_cells == 1.0:
                data = deposit_native(
                    idx_pos, weights, bspline_str, target_box.SHAPE, mesh_convention
                )
            else:
                K_max = k_max_for(bspline_str, scale_in_cells)
                data = deposit_scaled(
                    idx_pos,
                    weights,
                    bspline_str,
                    target_box.SHAPE,
                    mesh_convention,
                    scale_in_cells,
                    K_max,
                )

        if return_contrast:
            data = data / data.mean() - 1

        return ScalarField(data, box=target_box)

    # ------------
    # --- PLOT ---
    # ------------

    def paint(self, ax=None, **kwargs):
        from cobox.fluid._paint import paint

        return paint(self, ax=ax, **kwargs)
