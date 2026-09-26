from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING

import jax.numpy as jnp

from . import field_product
from .calculus import GradientKernelLiteral, grad_kernel as _grad_kernel


if TYPE_CHECKING:
    from ..scalar import ScalarField
    from ..vector import VectorField
    from ..tensor import TensorField

# ------------------
# --- ARITHMETIC ---
# ------------------


def dot_product(
    v1: "VectorField",
    v2: "VectorField",
    *,
    dealias: bool = False,
    N_iso: tuple[int, int] | None = None,
    out_N: field_product.OutputN = None,
    return_hat: bool | None = None,
) -> "ScalarField":
    """Compute sum_i v1_i * v2_i with optional dealiasing."""
    from ..scalar import ScalarField

    if v1.box != v2.box:
        raise ValueError("dot_product: operands live on different boxes.")

    terms = [
        field_product.product(
            v1.component(i),
            v2.component(i),
            dealias=dealias,
            N_iso=N_iso,
            out_N=out_N,
            return_hat=return_hat,
        )
        for i in range(v1.box.D)
    ]

    first = terms[0]
    total = first.data

    for term in terms[1:]:
        total = total + term.data

    return ScalarField(
        data=total,
        box=first.box,
        has_hat=first.has_hat,
    )


def cross_product(
    v1: "VectorField",
    v2: "VectorField",
    *,
    dealias: bool = False,
    N_iso: tuple[int, int] | None = None,
    out_N: field_product.OutputN = None,
    return_hat: bool | None = None,
) -> "ScalarField | VectorField":
    """Cross product: scalar in 2D, vector in 3D."""
    from ..scalar import ScalarField
    from ..vector import VectorField

    if v1.box != v2.box:
        raise ValueError("cross_product: operands live on different boxes.")

    D = v1.box.D
    if D not in (2, 3):
        raise ValueError("cross_product requires a 2D or 3D field.")

    def multiply(i: int, j: int) -> ScalarField:
        return field_product.product(
            v1.component(i),
            v2.component(j),
            dealias=dealias,
            N_iso=N_iso,
            out_N=out_N,
            return_hat=return_hat,
        )

    # Each tuple describes v1_i * v2_j - v1_k * v2_l.
    pairs = (
        ((0, 1, 1, 0),)
        if D == 2
        else (
            (1, 2, 2, 1),
            (2, 0, 0, 2),
            (0, 1, 1, 0),
        )
    )

    terms = [(multiply(i, j), multiply(k, l)) for i, j, k, l in pairs]

    first = terms[0][0]
    components = [positive.data - negative.data for positive, negative in terms]

    if D == 2:
        return ScalarField(
            data=components[0],
            box=first.box,
            has_hat=first.has_hat,
        )

    return VectorField(
        data=jnp.stack(components, axis=0),
        box=first.box,
        has_hat=first.has_hat,
    )


# ----------------
# --- CALCULUS ---
# ----------------


def curl(
    v: "VectorField", grad_kernel: GradientKernelLiteral = "spectral"
) -> "ScalarField | VectorField":
    """Curl: 3D -> VectorField, 2D -> Scalarfield. 1D undefined."""
    from ..scalar import ScalarField
    from ..vector import VectorField

    D = v.box.D
    if D == 1:
        raise ValueError("curl is undefined for a 1D vector field.")
    f = v.fft()
    G = [_grad_kernel(f.box.k_axes[a], f.box.R, kernel=grad_kernel) for a in range(D)]
    if D == 2:
        data = G[0] * f.data[1] - G[1] * f.data[0]
        return ScalarField(
            data=data,
            box=f.box,
            has_hat=True,
        )
    # D == 3
    cx = G[1] * f.data[2] - G[2] * f.data[1]
    cy = G[2] * f.data[0] - G[0] * f.data[2]
    cz = G[0] * f.data[1] - G[1] * f.data[0]
    return VectorField(
        data=jnp.stack([cx, cy, cz], axis=0),
        box=f.box,
        has_hat=True,
    )


def inverse_curl(omega: "ScalarField | VectorField") -> "VectorField":
    """Divergence-free pseudoinverse of curl with the spectral derivative.

    Each effective wavevector component kappa_i is k_i except at its axis's
    Nyquist frequency, where it is zero, matching the spectral curl/div.

    - 3D: ``v_hat = i (kappa x omega_hat) / kappa^2``. Curl of the result
      is the divergence-free projection of omega on representable modes.
    - 2D: ``v_hat = i (kappa_y, -kappa_x) omega_hat / kappa^2``.

    Modes with kappa^2 == 0 (DC and zero/Nyquist corners) return zero.
    This convention does not invert curls computed with finite differences.
    """
    from ..vector import VectorField
    from ..scalar import ScalarField

    box = omega.box
    D = box.D
    kappa = tuple(
        jnp.imag(_grad_kernel(axis, box.R, kernel="spectral")) for axis in box.k_axes
    )
    kappa_sq = sum((axis**2 for axis in kappa), jnp.array(0.0))
    nonzero = kappa_sq > 0
    inv = jnp.where(nonzero, 1.0 / jnp.where(nonzero, kappa_sq, 1.0), 0.0)
    f = omega.fft()

    if D == 3:
        if not isinstance(omega, VectorField):
            raise ValueError("3D vorticity must be a VectorField.")
        w = f.data
        cross = [
            kappa[1] * w[2] - kappa[2] * w[1],
            kappa[2] * w[0] - kappa[0] * w[2],
            kappa[0] * w[1] - kappa[1] * w[0],
        ]
        vhat = jnp.stack([1j * cross[a] * inv for a in range(3)], axis=0)
    elif D == 2:
        if not isinstance(omega, ScalarField):
            raise ValueError("2D vorticity must be a ScalarField.")
        w = f.data
        vhat = jnp.stack(
            [1j * kappa[1] * w * inv, -1j * kappa[0] * w * inv], axis=0
        )
    else:
        raise ValueError(
            "inverse_curl: vorticity is a 2D scalar or 3D vector; 1D has none."
        )
    return VectorField(
        data=vhat,
        box=box,
        has_hat=True,
    )
