"""Bare LPT shapes on a common working grid; all outputs are Fourier fields."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import jax.numpy as jnp

from cobox.field.ops.tensor_utils import component, det, second_invariant
from cobox.field.ops.vector_utils import cross_product, inverse_curl

if TYPE_CHECKING:
    from cobox.field import ScalarField, TensorField, VectorField


def psi_1(delta0: ScalarField) -> VectorField:
    """-grad lap^-1 delta0."""
    return -delta0.grad_inv_laplacian()


def psi_2(m1: TensorField) -> VectorField:
    """+grad lap^-1 I2(M1)."""
    source = second_invariant(m1, dealias=False, return_hat=False)
    return source.grad_inv_laplacian()


def psi_3a(m1: TensorField) -> VectorField:
    """-grad lap^-1 det(M1)."""
    if m1.box.D != 3:
        raise ValueError("psi_3a requires D == 3.")
    source = det(m1, dealias=False, return_hat=False)
    return -source.grad_inv_laplacian()


def psi_3b(m1: TensorField, m2: TensorField) -> VectorField:
    """-grad lap^-1 I2(M1, M2)."""
    if m1.box.D != 3:
        raise ValueError("psi_3b requires D == 3.")
    source = second_invariant(m1, m2, dealias=False, return_hat=False)
    return -source.grad_inv_laplacian()


def psi_3c(m1: TensorField, m2: TensorField) -> VectorField:
    """-curl^-1 sum_i [row_i(M1) x row_i(M2)]."""
    from cobox.field import VectorField

    if m1.box != m2.box:
        raise ValueError("psi_3c: operands live on different boxes.")
    if m1.box.D != 3:
        raise ValueError("psi_3c requires D == 3.")

    def row(t: TensorField, i: int) -> VectorField:
        return VectorField(
            data=jnp.stack([component(t, i, j).data for j in range(3)]),
            box=t.box,
            has_hat=t.has_hat,
        )

    terms = [
        cross_product(row(m1, i), row(m2, i), dealias=False, return_hat=False)
        for i in range(3)
    ]
    source = terms[0]
    data = source.data
    for term in terms[1:]:
        data = data + term.data
    return -inverse_curl(replace(source, data=data))
