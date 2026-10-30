"""A weighted redshift selection of a catalogue, projected on a Shell.

Notation and estimators follow docs/adf_arf_surveys/adf_arf_surveys.md.
"""

import warnings
from typing import TYPE_CHECKING, Literal, get_args

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.core import Tracer
from cosphere import Shell, ShellField, ShellMask
from jax.typing import ArrayLike

from .window import RedshiftWindow

if TYPE_CHECKING:
    from .catalog import RedshiftCatalog

ARFKind = Literal["global", "local"]
ZbarSource = Literal["bin", "randoms"]


class TomographicBin(eqx.Module):
    """A weighted redshift selection of a catalogue, projected on a Shell.

    q_i = w_i W(z_i) per galaxy, q_r = w_r W(z_r) per random. Cached at construction:
        S0 = sum_{i in p} q_i,   R0 = sum_{r in p} q_r   (or the coverage map, or 1).
    Footprint F = mask & (R0 > 0); alpha = sum_F S0 / sum_F R0; N = alpha R0 on F, the
    expected weighted count per pixel (not per steradian).

    window is a RedshiftWindow or None (W = 1). The caches are a snapshot of the inputs:
    to differentiate w.r.t. them, build the bin inside the transformed function. Keeps the
    randoms in memory; all inputs share one sky frame. See docs/adf_arf_surveys.
    """

    galaxies: "RedshiftCatalog"
    shell: Shell = eqx.field(static=True)
    window: RedshiftWindow | None
    weights: jax.Array | None
    randoms: "RedshiftCatalog | None"
    random_weights: jax.Array | None
    coverage: ShellField | None
    mask: ShellMask | None
    S0: ShellField
    R0: ShellField

    def __init__(
        self,
        galaxies: "RedshiftCatalog",
        shell: Shell,
        *,
        window: RedshiftWindow | None = None,
        weights: ArrayLike | None = None,
        randoms: "RedshiftCatalog | None" = None,
        random_weights: ArrayLike | None = None,
        coverage: ShellField | None = None,
        mask: ShellMask | None = None,
    ):
        if window is not None and not isinstance(window, RedshiftWindow):
            raise TypeError("window must be a RedshiftWindow instance or None.")
        if randoms is not None and coverage is not None:
            raise ValueError("pass randoms or coverage, not both.")
        if random_weights is not None and randoms is None:
            raise ValueError("random_weights needs randoms.")
        for name, m in (("coverage", coverage), ("mask", mask)):
            if m is not None and m.shell.nside != shell.nside:
                raise ValueError(f"{name} must share the shell's nside.")
        if coverage is not None and coverage.has_hat:
            raise ValueError("coverage must be a pixel-space map.")

        self.galaxies, self.shell, self.window = galaxies, shell, window
        self.weights = None if weights is None else galaxies._check_weights(weights)
        self.randoms, self.coverage, self.mask = randoms, coverage, mask
        self.random_weights = None
        if randoms is not None and random_weights is not None:
            self.random_weights = randoms._check_weights(random_weights)

        # Checks use eqx.error_if, so construction can run under jit and grad.
        if randoms is not None:
            q_r = _nonnegative(self.random_q, "random weights q_r = w_r W(z_r)")
            R0 = randoms.angular.to_shell(shell, q_r).data
        elif coverage is not None:
            c = jnp.asarray(coverage.data, dtype=float)
            R0 = eqx.error_if(
                c, jnp.any(~jnp.isfinite(c) | (c < 0)), "coverage must be finite and nonnegative."
            )
        else:
            R0 = jnp.ones(shell.SHAPE)
        q = _nonnegative(self.q, "galaxy weights q = w W(z)")
        S0 = galaxies.angular.to_shell(shell, q).data

        self.R0 = ShellField(R0, shell)
        self.S0 = ShellField(_footprint_checked(S0, S0, R0, self.footprint.data), shell)

        unsupported = self.weight_unsupported
        if not isinstance(unsupported, Tracer) and float(unsupported) > 0:
            warnings.warn(
                f"galaxy weight {float(unsupported):.4g} lies inside the mask where R0 = 0 and is "
                "excluded from every map and from zbar.",
                stacklevel=2,
            )

    # ---------------
    # --- WEIGHTS ---
    # ---------------

    @property
    def q(self) -> jax.Array:
        """q_i = w_i W(z_i), one per galaxy."""
        w = jnp.ones(len(self.galaxies)) if self.weights is None else self.weights
        return w if self.window is None else w * self.window(self.galaxies.z)

    @property
    def random_q(self) -> jax.Array:
        """q_r = w_r W(z_r), one per random."""
        randoms = self._require_randoms()
        w = jnp.ones(len(randoms)) if self.random_weights is None else self.random_weights
        return w if self.window is None else w * self.window(randoms.z)

    # -----------------
    # --- SELECTION ---
    # -----------------

    @property
    def footprint(self) -> ShellMask:
        """F = mask & (R0 > 0)."""
        inside = self.R0.data > 0
        if self.mask is not None:
            inside = inside & self.mask.data
        return ShellMask(inside, self.shell)

    @property
    def alpha(self) -> jax.Array:
        """alpha = sum_F S0 / sum_F R0."""
        F = self.footprint.data
        return jnp.sum(self.S0.data * F) / jnp.sum(self.R0.data * F)

    @property
    def N(self) -> ShellField:
        """N = alpha R0 on F, 0 outside: the weighted count per pixel expected with no structure."""
        return ShellField(jnp.where(self.footprint.data, self.alpha * self.R0.data, 0.0), self.shell)

    @property
    def weight_masked(self) -> jax.Array:
        """Galaxy weight in pixels excluded by the mask."""
        if self.mask is None:
            return jnp.zeros(())
        return jnp.sum(self.S0.data * ~self.mask.data)

    @property
    def weight_unsupported(self) -> jax.Array:
        """Galaxy weight inside the mask where R0 = 0 (no selection support)."""
        inside = self.R0.data <= 0
        if self.mask is not None:
            inside = inside & self.mask.data
        return jnp.sum(self.S0.data * inside)

    # ------------------
    # --- REDUCTIONS ---
    # ------------------

    def sum(self, values: ArrayLike = 1.0) -> ShellField:
        """Per-pixel sum_{i in p} q_i v_i (unnormalised). values: scalar or one per galaxy."""
        q = self._check_footprint(self.q)
        return self.galaxies.angular.to_shell(self.shell, q * self._per_galaxy(values, self.galaxies))

    def random_sum(self, values: ArrayLike = 1.0) -> ShellField:
        """Per-pixel sum_{r in p} q_r v_r over the randoms; recomputed (projects every random)."""
        randoms = self._require_randoms()
        q = self._check_footprint(self.random_q)
        return randoms.angular.to_shell(self.shell, q * self._per_galaxy(values, randoms))

    def map(self, values: ArrayLike) -> ShellField:
        """sum over galaxies in F of q v, divided by N; 0 outside F."""
        F = self.footprint.data
        sums = self._footprint_sum(values)
        return ShellField(jnp.where(F, sums / jnp.where(F, self.N.data, 1), 0.0), self.shell)

    def mean(self, values: ArrayLike) -> jax.Array:
        """One number: sum_i q_i v_i / sum_i q_i over the galaxies in F."""
        q, v = self._in_footprint(values)
        return jnp.sum(q * v) / jnp.sum(q)

    def pixel_mean(self, values: ArrayLike) -> ShellField:
        """sum(values) / S0 on populated pixels of F, NaN elsewhere."""
        populated = self.footprint.data & (self.S0.data > 0)
        sums = self._footprint_sum(values)
        return ShellField(jnp.where(populated, sums / jnp.where(populated, self.S0.data, 1), jnp.nan), self.shell)

    # ------------------------
    # --- REDSHIFT MOMENTS ---
    # ------------------------

    @property
    def S1(self) -> ShellField:
        """S1 = sum_{i in p} q_i z_i; recomputed."""
        return self.sum(self.galaxies.z)

    @property
    def R1(self) -> ShellField | None:
        """R1 = sum_{r in p} q_r z_r; None without randoms; recomputed (projects every random)."""
        return None if self.randoms is None else self.random_sum(self.randoms.z)

    @property
    def zbar(self) -> jax.Array:
        """The bin's mean redshift over F, mean(z)."""
        return self.mean(self.galaxies.z)

    @property
    def zbar_randoms(self) -> ShellField | None:
        """R1 / R0 on F, 0 outside: the randoms' mean redshift per pixel; None without randoms."""
        R1 = self.R1
        if R1 is None:
            return None
        F = self.footprint.data
        return ShellField(jnp.where(F, R1.data / jnp.where(F, self.R0.data, 1), 0.0), self.shell)

    # ------------------
    # --- ESTIMATORS ---
    # ------------------

    def adf(self) -> ShellField:
        """S0 / N - 1 on F, 0 outside."""
        F = self.footprint.data
        N = self.N.data
        return ShellField(jnp.where(F, self.S0.data / jnp.where(F, N, 1) - 1, 0.0), self.shell)

    def arf(
        self,
        kind: ARFKind = "global",
        zbar: ZbarSource | float | jax.Array | ShellField = "bin",
    ) -> ShellField:
        """Redshifts centred on a reference before deposition, divided by N (global, 0 outside F)
        or by S0 (local, NaN where S0 = 0 or outside F).

        zbar: "bin" (the bin's mean, the papers' ARF), "randoms" (zbar_randoms per pixel, doc
        Sec. 6.1), a scalar, or a ShellField (e.g. per-region references).
        """
        if kind not in get_args(ARFKind):
            raise ValueError(f"kind must be one of {get_args(ARFKind)}; got {kind!r}.")
        centred = self.galaxies.z - self._reference(zbar)
        return self.map(centred) if kind == "global" else self.pixel_mean(centred)

    # ---------------
    # --- HELPERS ---
    # ---------------

    def _check_footprint(self, value: jax.Array) -> jax.Array:
        """Attach the S0/R0 checks to a value a reduction consumes: under jit, a check whose
        output is unused is dropped, and reductions such as mean or sum skip the caches."""
        return _footprint_checked(value, self.S0.data, self.R0.data, self.footprint.data)

    def _reference(self, zbar: "ZbarSource | float | jax.Array | ShellField") -> jax.Array:
        """Reference redshift per galaxy. Map references are read at each galaxy's pixel, which
        equals S1 - ref S0 per pixel under nearest-pixel assignment."""
        if isinstance(zbar, str):
            if zbar not in get_args(ZbarSource):
                raise ValueError(f"zbar must be one of {get_args(ZbarSource)}, a number or a ShellField.")
            if zbar == "bin":
                return jnp.broadcast_to(self.zbar, (len(self.galaxies),))
            zbar_randoms = self.zbar_randoms
            if zbar_randoms is None:
                raise ValueError('zbar="randoms" needs randoms.')
            zbar = zbar_randoms
        if isinstance(zbar, ShellField):
            if zbar.has_hat or zbar.shell.nside != self.shell.nside:
                raise ValueError("a zbar map must be in pixel space and share the shell's nside.")
            return jnp.asarray(zbar.data)[self.galaxies.pixels(self.shell)]
        z = jnp.asarray(zbar, dtype=float)
        if z.ndim != 0:
            raise ValueError("a zbar value must be a number or a ShellField.")
        return jnp.broadcast_to(z, (len(self.galaxies),))

    def _in_footprint(self, values: ArrayLike) -> tuple[jax.Array, jax.Array]:
        """q and values with galaxies outside F set to 0, so their values (e.g. NaN) never
        enter; nonfinite values of galaxies in F are an error."""
        v = self._per_galaxy(values, self.galaxies)
        inside = self.footprint.data[self.galaxies.pixels(self.shell)]
        v = eqx.error_if(v, jnp.any(inside & ~jnp.isfinite(v)), "values must be finite for galaxies in the footprint.")
        q = self._check_footprint(self.q)
        return jnp.where(inside, q, 0.0), jnp.where(inside, v, 0.0)

    def _footprint_sum(self, values: ArrayLike) -> jax.Array:
        """Per-pixel sum of q v over the galaxies in F."""
        q, v = self._in_footprint(values)
        return self.galaxies.angular.to_shell(self.shell, q * v).data

    def _require_randoms(self) -> "RedshiftCatalog":
        if self.randoms is None:
            raise ValueError("this bin has no randoms.")
        return self.randoms

    @staticmethod
    def _per_galaxy(values: ArrayLike, catalog: "RedshiftCatalog") -> jax.Array:
        return jnp.broadcast_to(jnp.asarray(values, dtype=float), (len(catalog),))


def _nonnegative(q: jax.Array, name: str) -> jax.Array:
    return eqx.error_if(q, jnp.any(q < 0), f"{name} must be nonnegative.")


def _footprint_checked(value: jax.Array, S0: jax.Array, R0: jax.Array, F: jax.Array) -> jax.Array:
    """value, with an error if the footprint has no galaxy or no selection weight."""
    return eqx.error_if(
        value,
        ~(jnp.sum(S0 * F) > 0) | ~(jnp.sum(R0 * F) > 0),
        "the footprint has zero galaxy or zero selection weight.",
    )
