"""Tensor storage, linear operations, polynomial invariants and eigenvalues."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace
from itertools import permutations, product
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp

from ...box.support import ModeSupport
from . import field_product

if TYPE_CHECKING:
    from ...box import Box
    from ..scalar import ScalarField
    from ..tensor import TensorField

Symmetry = Literal["symmetric", "antisymmetric"] | None
Monomial = tuple[float, tuple["ScalarField", ...]]


# --- Storage ---


def _n_components(D: int, symmetry: Symmetry) -> int:
    if symmetry is None:
        return D * D
    if symmetry == "symmetric":
        return D * (D + 1) // 2
    if symmetry == "antisymmetric":
        return D * (D - 1) // 2
    raise ValueError(f"Unknown tensor symmetry: {symmetry!r}.")


def _sym_upper_index(k: int, j: int, D: int) -> int:
    return k * D - k * (k - 1) // 2 + j - k


def _antisym_strict_upper_index(k: int, j: int, D: int) -> int:
    return k * (D - 1) - k * (k - 1) // 2 + j - k - 1


def _component_storage_index_signed(
    k: int, j: int, D: int, symmetry: Symmetry
) -> tuple[int | None, int]:
    """Return the storage index and sign; implicit zeros have index None."""
    if not (0 <= k < D and 0 <= j < D):
        raise ValueError(f"Component ({k}, {j}) is out of bounds for D={D}.")
    if symmetry is None:
        return k * D + j, 1
    if symmetry == "symmetric":
        return _sym_upper_index(min(k, j), max(k, j), D), 1
    if symmetry == "antisymmetric":
        if k == j:
            return None, 0
        return _antisym_strict_upper_index(min(k, j), max(k, j), D), (
            1 if k < j else -1
        )
    raise ValueError(f"Unknown tensor symmetry: {symmetry!r}.")


def component_index(t: TensorField, k: int, j: int) -> int:
    """Return a storage index, without applying an antisymmetric sign."""
    idx, _ = _component_storage_index_signed(k, j, t.box.D, t.symmetry)
    if idx is None:
        raise ValueError("An antisymmetric diagonal has no stored component.")
    return idx


def component(t: TensorField, k: int, j: int) -> ScalarField:
    """Extract a component, including implicit zeros and sign changes."""
    from ..scalar import ScalarField

    idx, sign = _component_storage_index_signed(k, j, t.box.D, t.symmetry)
    data = (
        jnp.zeros(t.data.shape[1:], dtype=t.data.dtype)
        if idx is None
        else sign * t.data[idx]
    )
    return ScalarField(data=data, box=t.box, has_hat=t.has_hat, support=t.support)


def to_full(t: TensorField) -> jax.Array:
    """Expand to (D, D, *spatial_shape)."""
    D = t.box.D
    if t.symmetry is None:
        return t.data.reshape((D, D) + t.data.shape[1:])
    return jnp.stack(
        [jnp.stack([component(t, i, j).data for j in range(D)]) for i in range(D)]
    )


def from_full(
    full: jax.Array,
    *,
    box: Box,
    symmetry: Symmetry,
    has_hat: bool,
    support: ModeSupport | None = None,
) -> TensorField:
    """Pack a dense tensor; the caller guarantees the declared symmetry."""
    from ..tensor import TensorField

    D = box.D
    expected = (D, D) + (box.KSHAPE if has_hat else box.SHAPE)
    if full.shape != expected:
        raise ValueError(f"Expected dense shape {expected}, got {full.shape}.")
    _n_components(D, symmetry)
    if symmetry is None:
        data = full.reshape((D * D,) + full.shape[2:])
    else:
        if symmetry == "antisymmetric" and D == 1:
            raise ValueError("Antisymmetric tensor storage requires D >= 2.")
        offset = int(symmetry == "antisymmetric")
        data = jnp.stack([full[i, j] for i in range(D) for j in range(i + offset, D)])
    return TensorField(
        data=data, box=box, has_hat=has_hat, symmetry=symmetry, support=support
    )


# --- Linear operations ---


def trace(t: TensorField) -> ScalarField:
    """Trace, preserving the input representation."""
    first = component(t, 0, 0)
    data = first.data
    for i in range(1, t.box.D):
        data = data + component(t, i, i).data
    return replace(first, data=data)


def transpose(t: TensorField) -> TensorField:
    if t.symmetry == "symmetric":
        return t
    if t.symmetry == "antisymmetric":
        return replace(t, data=-t.data)
    return replace(t, data=jnp.swapaxes(to_full(t), 0, 1).reshape(t.data.shape))


def symmetric_part(t: TensorField) -> TensorField:
    if t.symmetry == "symmetric":
        return t
    full = to_full(t)
    return from_full(
        0.5 * (full + jnp.swapaxes(full, 0, 1)),
        box=t.box,
        symmetry="symmetric",
        has_hat=t.has_hat,
        support=t.support,
    )


def antisymmetric_part(t: TensorField) -> TensorField:
    if t.box.D == 1:
        raise ValueError("Antisymmetric tensor storage requires D >= 2.")
    if t.symmetry == "antisymmetric":
        return t
    full = to_full(t)
    return from_full(
        0.5 * (full - jnp.swapaxes(full, 0, 1)),
        box=t.box,
        symmetry="antisymmetric",
        has_hat=t.has_hat,
        support=t.support,
    )


def traceless_part(t: TensorField) -> TensorField:
    if t.symmetry == "antisymmetric":
        return t
    mean = trace(t).data / t.box.D
    data = t.data
    for i in range(t.box.D):
        data = data.at[component_index(t, i, i)].add(-mean)
    return replace(t, data=data)


# --- Polynomial helpers ---


def _same_box(tensors: Sequence[TensorField]) -> None:
    if any(t.box != tensors[0].box for t in tensors[1:]):
        raise ValueError("Tensor operands live on different boxes.")


def _zero(t: TensorField) -> ScalarField:
    f = component(t, 0, 0)
    return replace(f, data=jnp.zeros_like(f.data))


def _sum_products(
    terms: Iterable[Monomial],
    *,
    dealias: bool,
    out_N: field_product.OutputN,
    return_hat: bool | None,
) -> ScalarField:
    def evaluate(term: Monomial) -> ScalarField:
        weight, operands = term
        result = field_product._dispatch_product(
            operands,
            dealias=dealias,
            out_N=out_N,
            return_hat=return_hat,
        )
        return replace(result, data=weight * result.data)

    results = [evaluate(term) for term in terms]
    data = results[0].data
    for result in results[1:]:
        data = data + result.data
    support = ModeSupport.of_sum(*(r.support for r in results))
    return replace(results[0], data=data, support=support)


def _trace_product_terms(a: TensorField, b: TensorField) -> list[Monomial]:
    D = a.box.D
    if a.symmetry == b.symmetry == "symmetric":
        return [
            (1.0 if i == j else 2.0, (component(a, i, j), component(b, i, j)))
            for i in range(D)
            for j in range(i, D)
        ]
    return [
        (1.0, (component(a, i, j), component(b, j, i)))
        for i in range(D)
        for j in range(D)
    ]


# --- Quadratic invariants ---


def trace_of_product(
    a: TensorField,
    b: TensorField,
    *,
    dealias: bool = False,
    out_N: field_product.OutputN = None,
    return_hat: bool | None = None,
) -> ScalarField:
    """tr(AB); dealiasing reads each tensor's support."""
    _same_box((a, b))
    return _sum_products(
        _trace_product_terms(a, b),
        dealias=dealias,
        out_N=out_N,
        return_hat=return_hat,
    )


def _stored(t: TensorField, s: int) -> ScalarField:
    """Stored component s as a ScalarField (no sign, no implicit zeros)."""
    from ..scalar import ScalarField

    return ScalarField(data=t.data[s], box=t.box, has_hat=t.has_hat, support=t.support)


def _trace_terms(tensors: Sequence[TensorField]) -> list[Monomial]:
    """Monomials of tr(T_1 T_2 ... T_n), merged up to the order of factors.

    tr(T_1 ... T_n) = sum over the closed index loop (i_1, ..., i_n) of
    (T_1)_{i_1 i_2} (T_2)_{i_2 i_3} ... (T_n)_{i_n i_1}: one stored component
    per tensor. Scalar factors commute, so summands that are the same multiset
    of (tensor, storage index) merge and their weights add; antisymmetric signs
    fold into the weight and implicit zeros drop out.
    """
    D, n = tensors[0].box.D, len(tensors)
    first: dict[int, int] = {}  # id(tensor) -> first position: repeats merge
    for p, t in enumerate(tensors):
        first.setdefault(id(t), p)

    merged: dict[tuple[tuple[int, int], ...], float] = {}
    for loop in product(range(D), repeat=n):
        weight = 1.0
        factors = []
        for p, t in enumerate(tensors):
            s, sign = _component_storage_index_signed(
                loop[p], loop[(p + 1) % n], D, t.symmetry
            )
            if s is None:  # antisymmetric diagonal: the summand vanishes
                break
            weight *= sign
            factors.append((first[id(t)], s))
        else:
            key = tuple(sorted(factors))
            merged[key] = merged.get(key, 0.0) + weight

    terms: list[Monomial] = [
        (w, tuple(_stored(tensors[q], s) for q, s in key))
        for key, w in merged.items()
        if w != 0.0
    ]
    return terms or [(1.0, tuple(_zero(t) for t in tensors))]


def trace_of_products(
    tensors: Sequence[TensorField],
    *,
    dealias: bool = False,
    out_N: field_product.OutputN = None,
    return_hat: bool | None = None,
) -> ScalarField:
    """tr(T_1 T_2 ... T_n) for n >= 2, as a sum of n-fold monomials.

    Each monomial is one n-fold product without intermediate projection, so
    dealias=True (which reads each tensor's support) is exact.
    """
    tensors = tuple(tensors)
    if len(tensors) < 2:
        raise ValueError("trace_of_products needs at least two tensors; use trace.")
    _same_box(tensors)
    return _sum_products(
        _trace_terms(tensors),
        dealias=dealias,
        out_N=out_N,
        return_hat=return_hat,
    )


def second_invariant(
    t: TensorField,
    other: TensorField | None = None,
    *,
    dealias: bool = False,
    out_N: field_product.OutputN = None,
    return_hat: bool | None = None,
) -> ScalarField:
    """I2(T), or the polarization (tr(A)tr(B) - tr(AB))/2."""
    if other is not None:
        _same_box((t, other))
        terms: list[Monomial] = [(0.5, (trace(t), trace(other)))]
        terms += [(-0.5 * w, fs) for w, fs in _trace_product_terms(t, other)]
    else:
        terms = []
        for i in range(t.box.D):
            for j in range(i + 1, t.box.D):
                off = component(t, i, j)
                if t.symmetry == "antisymmetric":
                    terms.append((1.0, (off, off)))
                else:
                    terms.append((1.0, (component(t, i, i), component(t, j, j))))
                    other_off = off if t.symmetry == "symmetric" else component(t, j, i)
                    terms.append((-1.0, (off, other_off)))
    if t.box.D == 1:
        terms = [(1.0, (_zero(t), _zero(t if other is None else other)))]
    return _sum_products(
        terms,
        dealias=dealias,
        out_N=out_N,
        return_hat=return_hat,
    )


# --- Determinants and polarization ---

_LEIBNIZ_3 = (
    ((0, 1, 2), 1.0),
    ((1, 2, 0), 1.0),
    ((2, 0, 1), 1.0),
    ((0, 2, 1), -1.0),
    ((1, 0, 2), -1.0),
    ((2, 1, 0), -1.0),
)


def det(
    t: TensorField,
    *,
    dealias: bool = False,
    out_N: field_product.OutputN = None,
    return_hat: bool | None = None,
) -> ScalarField:
    """Determinant, with each degree-D monomial evaluated in one step."""
    D = t.box.D
    if D == 2:
        return second_invariant(
            t,
            dealias=dealias,
            out_N=out_N,
            return_hat=return_hat,
        )
    if D == 1:
        terms: list[Monomial] = [(1.0, (component(t, 0, 0),))]
    elif D == 3:
        c = tuple(tuple(component(t, i, j) for j in range(D)) for i in range(D))
        if t.symmetry == "antisymmetric":
            z = _zero(t)
            terms = [(1.0, (z, z, z))]
        elif t.symmetry == "symmetric":
            terms = [
                (1.0, (c[0][0], c[1][1], c[2][2])),
                (2.0, (c[0][1], c[0][2], c[1][2])),
                (-1.0, (c[0][0], c[1][2], c[1][2])),
                (-1.0, (c[1][1], c[0][2], c[0][2])),
                (-1.0, (c[2][2], c[0][1], c[0][1])),
            ]
        else:
            terms = [
                (sign, tuple(c[i][p[i]] for i in range(3))) for p, sign in _LEIBNIZ_3
            ]
    else:
        raise ValueError("det requires D in (1, 2, 3).")
    return _sum_products(
        terms,
        dealias=dealias,
        out_N=out_N,
        return_hat=return_hat,
    )


def _mixed_det3(
    rows: tuple[TensorField, TensorField, TensorField],
    *,
    dealias: bool,
    out_N: field_product.OutputN,
    return_hat: bool | None,
) -> ScalarField:
    terms = [
        (sign, tuple(component(rows[i], i, p[i]) for i in range(3)))
        for p, sign in _LEIBNIZ_3
    ]
    return _sum_products(
        terms,
        dealias=dealias,
        out_N=out_N,
        return_hat=return_hat,
    )


def third_invariant(
    a: TensorField,
    b: TensorField | None = None,
    c: TensorField | None = None,
    *,
    dealias: bool = False,
    out_N: field_product.OutputN = None,
    return_hat: bool | None = None,
) -> ScalarField:
    """det(A), mu3(A,A,B), or mu3(A,B,C); mixed forms require 3D.

    Normalization: mu3(T,T,T) = det(T).
    """
    if b is None and c is None:
        return det(a, dealias=dealias, out_N=out_N, return_hat=return_hat)
    if b is None:
        raise ValueError("Pass (a), (a, b), or (a, b, c).")
    if a.box.D != 3:
        raise ValueError("Mixed third_invariant requires D == 3.")
    tensors = (a, a, b) if c is None else (a, b, c)
    _same_box(tensors)

    # Row assignments that coincide (repeated tensors) are evaluated once.
    assignments: dict[tuple[int, ...], tuple[int, tuple[int, ...]]] = {}
    for order in permutations(range(3)):
        key = tuple(id(tensors[i]) for i in order)
        weight, _ = assignments.get(key, (0, order))
        assignments[key] = (weight + 1, order)

    results = []
    for weight, order in assignments.values():
        result = _mixed_det3(
            (tensors[order[0]], tensors[order[1]], tensors[order[2]]),
            dealias=dealias,
            out_N=out_N,
            return_hat=return_hat,
        )
        results.append(replace(result, data=(weight / 6.0) * result.data))
    data = results[0].data
    for result in results[1:]:
        data = data + result.data
    support = ModeSupport.of_sum(*(r.support for r in results))
    return replace(results[0], data=data, support=support)


# --- Pointwise eigenvalues ---


def _safe_sqrt(x: jax.Array) -> jax.Array:
    positive = x > 0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, x, 1.0)), 0.0)


def eigenvalues(t: TensorField, *, return_hat: bool = False) -> tuple[ScalarField, ...]:
    """Descending symmetric eigenvalues, evaluated pointwise; not dealiased."""
    from ..scalar import ScalarField

    if t.symmetry != "symmetric":
        raise ValueError("eigenvalues requires symmetry='symmetric'.")
    if not isinstance(return_hat, bool):
        raise TypeError("return_hat must be a bool.")
    full = to_full(t.ifft())
    D = t.box.D
    if D == 1:
        values = (full[0, 0],)
    elif D == 2:
        mean = 0.5 * (full[0, 0] + full[1, 1])
        diff = 0.5 * (full[0, 0] - full[1, 1])
        root = _safe_sqrt(diff**2 + full[0, 1] ** 2)
        values = (mean + root, mean - root)
    elif D == 3:
        mean = (full[0, 0] + full[1, 1] + full[2, 2]) / 3.0
        d0, d1, d2 = (full[i, i] - mean for i in range(3))
        o01, o02, o12 = full[0, 1], full[0, 2], full[1, 2]
        p = _safe_sqrt((d0**2 + d1**2 + d2**2 + 2 * (o01**2 + o02**2 + o12**2)) / 6.0)
        p_safe = jnp.where(p > 0, p, 1.0)
        determinant = (
            d0 * d1 * d2 + 2 * o01 * o02 * o12 - d0 * o12**2 - d1 * o02**2 - d2 * o01**2
        )
        r = determinant / (2 * p_safe**3)
        endpoint = (r <= -1) | (r >= 1)
        phi = jnp.arccos(jnp.where(endpoint, 0.0, r)) / 3.0
        phi = jnp.where(r >= 1, 0.0, jnp.where(r <= -1, jnp.pi / 3.0, phi))
        largest = mean + 2 * p * jnp.cos(phi)
        smallest = mean + 2 * p * jnp.cos(phi + 2 * jnp.pi / 3.0)
        values = (largest, 3 * mean - largest - smallest, smallest)
    else:
        raise ValueError("eigenvalues requires D in (1, 2, 3).")
    fields = tuple(ScalarField(data=v, box=t.box, has_hat=False) for v in values)
    return tuple(f.fft() for f in fields) if return_hat else fields
