"""Cosmological angular power, composed from scalar observable contributions."""

from collections.abc import Sequence
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from cosphere.projection import angular_cl

from ..cosmology import Cosmology
from ..cosmology._distances import get_distance_table
from ..power import PrimordialSpectrum
from ._projection import LightConeTransfer, PrimordialPower, static_integer
from .observable import Observable


class AngularPower(eqx.Module):
    """Non-Limber scalar angular spectra from a shared primordial source.

    Configuration only; observables and cosmology are supplied on every call,
    as for LinearPower. Each term supplies its own curvature-to-field transfer;
    passing a LinearPower as the source would double-count those transfers.
    Does not inherit 3D PowerSpectrum (no sigma8 or k-space variance methods).

    k is an explicit grid in h/Mpc, windows use comoving Mpc/h. ell_max is a
    static upper bound: calls select integer multipoles in [0, ell_max], keeping
    their input shape. Integer-valued floats from Shell.ell_axis are accepted.
    All multipoles through ell_max are computed before selection, enabling JIT.

    Only flat backgrounds (Omega_k == 0) and scalar j_ell operators are supported.
    No Limber, noise, masks, beams, velocity/RSD operators or nonlinear kSZ model
    are supplied here. Physical arrays and parameters remain differentiable.
    Check k bounds/spacing, n_r and n_distance for convergence. There is no tail
    extrapolation or hidden cache. Distances use BackgroundCosmo's radiation-free
    model; high-redshift applications require a suitable background first.
    """

    primordial: PrimordialSpectrum
    k: jax.Array
    ell_max: int = eqx.field(static=True)
    n_r: int = eqx.field(static=True)
    k_chunk_size: int = eqx.field(static=True)
    n_distance: int = eqx.field(static=True)
    a_min: float = eqx.field(static=True)

    def __init__(
        self,
        primordial: PrimordialSpectrum,
        k: ArrayLike,
        ell_max: int,
        *,
        n_r: int = 128,
        k_chunk_size: int = 64,
        n_distance: int = 4096,
        a_min: float = 1e-5,
    ):
        if not isinstance(primordial, PrimordialSpectrum):
            raise TypeError("primordial must be a PrimordialSpectrum.")
        k = jnp.asarray(k)
        if jnp.iscomplexobj(k) or k.ndim != 1 or k.size < 2:
            raise ValueError("k must be a real 1D grid with at least two points.")
        k = k.astype(jnp.result_type(k, jnp.float32))
        self.k = eqx.error_if(
            k,
            jnp.any(~jnp.isfinite(k)) | jnp.any(k <= 0) | jnp.any(jnp.diff(k) <= 0),
            "k must be finite, positive and strictly increasing.",
        )
        self.primordial = primordial
        self.ell_max = static_integer(ell_max, "ell_max", 0)
        self.n_r = static_integer(n_r, "n_r", 2)
        self.k_chunk_size = static_integer(k_chunk_size, "k_chunk_size", 1)
        self.n_distance = static_integer(n_distance, "n_distance", 2)
        if not isfinite(a_min) or not 0 < a_min < 1:
            raise ValueError("a_min must be finite and between 0 and 1.")
        self.a_min = a_min

    def __call__(
        self,
        ell: ArrayLike,
        observable: Observable,
        cosmology: Cosmology,
        other: Observable | None = None,
    ) -> jax.Array:
        """Auto-power, or cross-power with other; output has shape ell.shape."""
        observables = (observable,) if other is None else (observable, other)
        return self.matrix(ell, observables, cosmology)[..., 0, -1]

    def matrix(
        self,
        ell: ArrayLike,
        observables: Sequence[Observable],
        cosmology: Cosmology,
    ) -> jax.Array:
        """All auto/cross spectra, shape ell.shape + (n_observables, n_observables).

        Includes interference between every term in each observable. Signed
        coefficients and complex transfers preserve the Hermitian covariance.
        The existing real-map sampler requires a real nonnegative auto-spectrum.
        One distance table is shared by all transfers for this evaluation.
        """
        observables = tuple(observables)
        if not observables or not all(isinstance(o, Observable) for o in observables):
            raise TypeError("observables must contain at least one Observable.")
        indices = self._ell_indices(ell)
        background = cosmology.background
        k = eqx.error_if(
            self.k,
            jnp.asarray(background.Omega_k) != 0,
            "AngularPower requires Omega_k == 0; curved projection is unsupported.",
        )
        a_grid, chi_grid = get_distance_table(
            background, n_quad=self.n_distance, a_min=self.a_min
        )
        terms = tuple(t for o in observables for t in o.terms)
        transfers = tuple(
            None
            if t.transfer is None
            else LightConeTransfer(t.transfer, background, a_grid, chi_grid)
            for t in terms
        )
        # Reject out-of-domain support even when the term's transfer is unity.
        bounds = jnp.stack([jnp.asarray(t.window.support) for t in terms])
        k = eqx.error_if(
            k,
            jnp.any(~jnp.isfinite(bounds))
            | jnp.any(bounds < 0)
            | jnp.any(bounds > chi_grid[0]),
            "Window support must be within [0, chi(a_min)] in Mpc/h.",
        )
        spectra = angular_cl(
            k,
            PrimordialPower(self.primordial, cosmology),
            tuple(t.window for t in terms),
            self.ell_max,
            transfers=transfers,
            n_r=self.n_r,
            k_chunk_size=self.k_chunk_size,
        )
        # Linear combinations act on both sides of the term covariance, so
        # auto-power of a sum includes its cross-terms, with the correct signs.
        dtype = jnp.result_type(*[t.weight for t in terms])
        mixing = jnp.zeros((len(observables), len(terms)), dtype=dtype)
        start = 0
        for i, observable in enumerate(observables):
            end = start + len(observable.terms)
            mixing = mixing.at[i, start:end].set(
                jnp.stack([t.weight for t in observable.terms])
            )
            start = end
        result = jnp.einsum(
            "it,ltu,ju->lij",
            mixing,
            spectra,
            mixing,
            precision=jax.lax.Precision.HIGHEST,
        )
        return result[indices]

    def bind(self, observable: Observable, *, cosmology: Cosmology) -> eqx.Partial:
        """Return a callable auto-spectrum cl_fn(ell), for ShellGRFSampler.

        Binding captures these inputs, not a precomputed spectrum. JAX can trace
        through the bound PyTree; each evaluation uses the captured parameters.
        """
        return eqx.Partial(self, observable=observable, cosmology=cosmology)

    def plot(self, ell, observable, cosmology, *, other=None, ax=None, **kwargs):
        """Plot raw C_ell (auto or signed cross-power); return the axes."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        values = self(ell, observable, cosmology, other=other)
        ax.plot(np.asarray(ell).ravel(), np.asarray(values).ravel(), **kwargs)
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$C_\ell$")
        return ax

    def _ell_indices(self, ell: ArrayLike) -> jax.Array:
        ell = jnp.asarray(ell)
        if jnp.iscomplexobj(ell) or jnp.issubdtype(ell.dtype, jnp.bool_):
            raise TypeError("ell must contain real integer multipoles.")
        ell = eqx.error_if(
            ell,
            jnp.any(~jnp.isfinite(ell))
            | jnp.any(ell < 0)
            | jnp.any(ell > self.ell_max)
            | jnp.any(ell != jnp.floor(ell)),
            "ell must contain integers in [0, ell_max].",
        )
        return ell.astype(jnp.int32)

    # ---------------------
    # --- SERIALIZATION ---
    # ---------------------

    def to_dict(self) -> dict:
        """Calculator configuration only; host operation, excluding call inputs."""
        return {
            "primordial": self.primordial.to_dict(),
            "k": np.asarray(self.k).tolist(),
            "ell_max": self.ell_max,
            "n_r": self.n_r,
            "k_chunk_size": self.k_chunk_size,
            "n_distance": self.n_distance,
            "a_min": self.a_min,
        }

    @classmethod
    def from_dict(cls, config: dict) -> "AngularPower":
        config = dict(config)
        allowed = {
            "primordial",
            "k",
            "ell_max",
            "n_r",
            "k_chunk_size",
            "n_distance",
            "a_min",
        }
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"unknown keys in AngularPower config: {sorted(unknown)}.")
        missing = {"primordial", "k", "ell_max"} - set(config)
        if missing:
            raise ValueError(f"missing keys in AngularPower config: {sorted(missing)}.")
        config["primordial"] = PrimordialSpectrum.from_dict(config["primordial"])
        return cls(**config)

    def to_yaml(self) -> str:
        import yaml

        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_yaml(cls, s: str) -> "AngularPower":
        import yaml

        return cls.from_dict(yaml.safe_load(s))
