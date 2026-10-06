from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from numbers import Integral
from typing import TYPE_CHECKING, Any, Literal

import equinox as eqx
import jax.numpy as jnp
from jax.typing import ArrayLike

from cobox.field import ScalarField
from cobox.field.ops._grids import OutputN

from ._basis import _build_operators
from ._operators import specs_for

if TYPE_CHECKING:
    from cobox.box import Box

LagBiasOrderLiteral = Literal[1, 2, 3]

_LAG_BIAS_FIELDS = ("order", "dealias", "subtract_mean", "out_N")


class LagBias(eqx.Module):
    """Lagrangian bias options. No data: build operators for delta0 with .basis().

    order selects the complete Lagrangian basis up to that perturbative order
    (1, 2, 4 operators) plus lap tr(1). out_N has LPT's meaning: None keeps the
    input grid, 'full' keeps every generated mode, an int projects onto that
    grid; equal out_N gives the Lagrangian box of LPT.basis on the same delta0.

    These are bare polynomial operators of the supplied delta0. With
    subtract_mean=True, each output operator has its box mean removed by
    setting only its Fourier zero mode to zero. Nonzero modes are unchanged.
    This centering is not a full bias-renormalization prescription: no
    counterterms mixing operators of different orders are applied. Coefficients
    fitted to this basis refer to its operator normalization and the input's
    smoothing/cutoff; they are not automatically renormalized bias parameters.
    """

    order: LagBiasOrderLiteral = eqx.field(static=True)
    dealias: bool = eqx.field(static=True, default=True)
    subtract_mean: bool = eqx.field(static=True, default=True)
    out_N: OutputN = eqx.field(static=True, default=None)

    def __check_init__(self):
        self._validate()

    def keys(self) -> tuple[str, ...]:
        return tuple(spec.key for spec in specs_for(self.order))

    def basis(self, delta0: ScalarField) -> LagBiasBasis:
        """Build the operators for delta0, in Fourier space.

        Dealiasing sizes its working grid from delta0.support (the whole box
        when it has none); the support is trusted, not enforced.
        """
        operators = _build_operators(
            delta0,
            self.order,
            dealias=self.dealias,
            out_N=self.out_N,
            subtract_mean=self.subtract_mean,
        )
        return LagBiasBasis(self, operators)

    # ---------------------
    # --- SERIALIZATION ---
    # ---------------------

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in _LAG_BIAS_FIELDS}

    @classmethod
    def from_dict(cls, config: dict) -> LagBias:
        unknown = set(config) - set(_LAG_BIAS_FIELDS)
        if unknown:
            raise ValueError(f"unknown keys in LagBias config: {sorted(unknown)}.")
        return cls(**config)

    def to_yaml(self) -> str:
        import yaml

        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_yaml(cls, s: str) -> LagBias:
        import yaml

        return cls.from_dict(yaml.safe_load(s))

    # ------------------
    # --- VALIDATION ---
    # ------------------

    def _validate(self) -> None:
        if (
            isinstance(self.order, bool)
            or not isinstance(self.order, Integral)
            or self.order not in (1, 2, 3)
        ):
            raise ValueError("order must be 1, 2, or 3.")
        if not isinstance(self.dealias, bool) or not isinstance(
            self.subtract_mean, bool
        ):
            raise TypeError("dealias and subtract_mean must be bools.")
        if self.out_N is not None and self.out_N != "full":
            if (
                isinstance(self.out_N, bool)
                or not isinstance(self.out_N, Integral)
                or self.out_N < 1
            ):
                raise ValueError("out_N must be None, 'full', or a positive int.")


class LagBiasBasis(eqx.Module):
    """Lagrangian bias operators O(q) for one delta0, on one box, in one space.

    combine(b) returns sum_O b_O O(q); the coefficients b_O are free scalar
    weights in this basis's convention. In particular, tr(1) = -delta0;
    no factorials or conversions to renormalized bias conventions are applied.
    """

    bias: LagBias
    operators: dict[str, ScalarField]

    def __check_init__(self):
        fields = list(self.operators.values())
        if not fields:
            raise ValueError("LagBiasBasis needs at least one operator.")
        first = fields[0]
        if any(f.box != first.box or f.has_hat != first.has_hat for f in fields[1:]):
            raise ValueError("Bias operators must share a box and a space.")

    @property
    def box(self) -> Box:
        return next(iter(self.operators.values())).box

    def keys(self) -> tuple[str, ...]:
        return tuple(self.operators)

    def __getitem__(self, key: str) -> ScalarField:
        return self.operators[key]

    def fft(self) -> LagBiasBasis:
        return replace(self, operators={k: f.fft() for k, f in self.operators.items()})

    def ifft(self) -> LagBiasBasis:
        return replace(self, operators={k: f.ifft() for k, f in self.operators.items()})

    def combine(self, b: Mapping[str, ArrayLike]) -> ScalarField:
        """sum_O b_O O with scalar weights; missing keys count as zero.

        Uses the stored operators as built, including any mean subtraction.
        It does not renormalize or orthogonalize them.
        """
        unknown = set(b) - set(self.operators)
        if unknown:
            raise ValueError(f"unknown bias operators: {sorted(unknown)}.")
        terms = [self.operators[key] * b[key] for key in self.operators if key in b]
        if not terms:
            first = next(iter(self.operators.values()))
            return replace(first, data=jnp.zeros_like(first.data))
        return sum(terms[1:], start=terms[0])
