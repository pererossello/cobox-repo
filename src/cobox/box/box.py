from math import pi, sqrt
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp

from . import _fourier

DIMENSIONS: tuple[int, ...] = (1, 2, 3)


class Box(eqx.Module):

    N: int = eqx.field(static=True)
    L: float = eqx.field(static=True)
    D: int = eqx.field(static=True)

    def __init__(self, N: int, L: float, D: int):

        self.N = N
        self.L = float(L)
        self.D = D

        self._validate_inputs()

    @property
    def R(self) -> float:
        return self.L / self.N

    @property
    def INV_R(self) -> float:
        return self.N / self.L

    @property
    def V(self) -> float:
        return self.L**self.D

    @property
    def INV_V(self) -> float:
        return 1.0 / self.V

    @property
    def ND(self) -> int:
        return self.N**self.D

    @property
    def INV_ND(self) -> float:
        return 1.0 / self.ND

    @property
    def CELL_V(self) -> float:
        return self.R**self.D

    @property
    def INV_CELL_V(self) -> float:
        return 1.0 / self.CELL_V

    @property
    def SHAPE(self) -> tuple[int, ...]:
        return (self.N,) * self.D

    @property
    def KSHAPE(self) -> tuple[int, ...]:
        return (self.N,) * (self.D - 1) + (self.N // 2 + 1,)

    @property
    def K_RES(self) -> float:
        return 2 * pi / self.L

    @property
    def K_NYQ(self) -> float:
        return 0.5 * self.K_RES * self.N

    @property
    def K_NYQ_CORNER(self) -> float:
        return self.K_NYQ * sqrt(self.D)

    # -------------------------
    # --- WAVENUMBER ARRAYS ---
    # -------------------------

    @property
    def k_idx_1D(self) -> jax.Array:
        return jnp.fft.fftfreq(self.N, d=1 / self.N).astype(jnp.int32)

    @property
    def k_idx_1D_rfft(self) -> jax.Array:
        return jnp.fft.rfftfreq(self.N, d=1 / self.N).astype(jnp.int32)

    @property
    def k_idx_axes(self) -> tuple[jax.Array, ...]:
        full_axes = tuple(
            self.k_idx_1D.reshape((1,) * i + (-1,) + (1,) * (self.D - 1 - i))
            for i in range(self.D - 1)
        )
        half_axis = (self.k_idx_1D_rfft.reshape((1,) * (self.D - 1) + (-1,)),)
        return full_axes + half_axis

    @property
    def k_1D(self) -> jax.Array:
        return self.K_RES * self.k_idx_1D

    @property
    def k_1D_rfft(self) -> jax.Array:
        return self.K_RES * self.k_idx_1D_rfft

    @property
    def k_axes(self) -> tuple[jax.Array, ...]:
        return tuple(self.K_RES * axis for axis in self.k_idx_axes)

    @property
    def k_grid(self) -> tuple[jax.Array, ...]:
        return tuple(jnp.broadcast_arrays(*self.k_axes))

    @property
    def k_sq(self) -> jax.Array:
        return sum((k**2 for k in self.k_axes), jnp.array(0.0))

    @property
    def k(self) -> jax.Array:
        return jnp.sqrt(self.k_sq)

    @property
    def inv_k_sq(self) -> jax.Array:
        k_sq = self.k_sq
        return jnp.where(k_sq > 0, 1.0 / jnp.where(k_sq > 0, k_sq, 1.0), 0.0)

    @property
    def k_full(self) -> jax.Array:
        axes = tuple(
            self.k_idx_1D.reshape((1,) * i + (-1,) + (1,) * (self.D - 1 - i))
            for i in range(self.D)
        )
        k_sq = sum(((self.K_RES * a) ** 2 for a in axes), jnp.array(0.0))
        return jnp.sqrt(k_sq)

    @property
    def k_idx_grid(self) -> tuple[jax.Array, ...]:
        return tuple(jnp.broadcast_arrays(*self.k_idx_axes))

    @property
    def k_idx_sq(self) -> jax.Array:
        return sum((z**2 for z in self.k_idx_axes), jnp.array(0.0))

    @property
    def k_idx(self) -> jax.Array:
        return jnp.sqrt(self.k_idx_sq)

    @property
    def inv_k_idx_sq(self) -> jax.Array:
        k_idx_sq = self.k_idx_sq
        return jnp.where(
            k_idx_sq > 0, 1.0 / jnp.where(k_idx_sq > 0, k_idx_sq, 1.0), 0.0
        )

    # --------------------
    # --- SPACE ARRAYS ---
    # --------------------

    @property
    def x_1D(self) -> jax.Array:
        return self.R * jnp.arange(self.N)

    @property
    def x_axes(self) -> tuple[jax.Array, ...]:
        return tuple(
            self.x_1D.reshape((1,) * a + (-1,) + (1,) * (self.D - 1 - a))
            for a in range(self.D)
        )

    @property
    def x_grid(self) -> tuple[jax.Array, ...]:
        return tuple(jnp.broadcast_arrays(*self.x_axes))

    # ------------
    # --- FFTS ---
    # ------------

    def fft(self, field: jax.Array) -> jax.Array:
        return _fourier.fft(field, cell_v=self.CELL_V)

    def ifft(self, field_hat: jax.Array) -> jax.Array:
        return _fourier.ifft(field_hat, inv_cell_v=self.INV_CELL_V)

    def fft_on_vec(self, vec_field: jax.Array) -> jax.Array:
        return _fourier.fft_on_vec(vec_field, D=self.D, cell_v=self.CELL_V)

    def ifft_on_vec(self, vec_field_hat: jax.Array) -> jax.Array:
        return _fourier.ifft_on_vec(vec_field_hat, D=self.D, inv_cell_v=self.INV_CELL_V)

    def fft_full(self, field: jax.Array) -> jax.Array:
        return _fourier.fft_full(field, cell_v=self.CELL_V)

    def rfft_to_full(self, field_hat: jax.Array) -> jax.Array:
        return _fourier.rfft_to_full(field_hat)

    def pad_rfft(self, field_hat: jax.Array, NEW_N: int) -> jax.Array:
        return _fourier.pad_rfft(field_hat, NEW_N, D=self.D)

    def crop_rfft(self, field_hat: jax.Array, NEW_N: int) -> jax.Array:
        return _fourier.crop_rfft(field_hat, NEW_N, D=self.D)

    # -----------------
    # --- RESCALING ---
    # -----------------

    def rescale(self, refine: int = 1, under: int = 1) -> "Box":
        if refine < 1:
            raise ValueError(f"refine must be >= 1, got {refine}.")
        if under < 1:
            raise ValueError(f"under must be >= 1, got {under}.")
        if refine != 1 and under != 1:
            raise ValueError("set either refine or under, not both.")
        if self.N % under != 0:
            raise ValueError(f"N={self.N} not divisible by under={under}.")
        N_NEW = self.N * refine // under
        return replace(self, N=N_NEW)

    # ---------------------
    # --- SERIALIZATION ---
    # ---------------------

    def to_dict(self) -> dict:
        from . import _serialize

        return _serialize.box_to_dict(self)

    @classmethod
    def from_dict(cls, config: dict) -> "Box":
        from . import _serialize

        return _serialize.box_from_dict(config, cls)

    def to_yaml(self) -> str:
        from . import _serialize

        return _serialize.box_to_yaml(self)

    @classmethod
    def from_yaml(cls, s: str) -> "Box":
        from . import _serialize

        return _serialize.box_from_yaml(s, cls)

    # -----------------------
    # --- SPECIAL METHODS ---
    # -----------------------

    def __hash__(self):
        return hash((self.N, self.L, self.D))

    def __eq__(self, other):
        return isinstance(other, Box) and (
            (self.N, self.L, self.D) == (other.N, other.L, other.D)
        )

    def __repr__(self) -> str:
        return f"Box(N={self.N}, L={self.L}, D={self.D})"

    # ------------------
    # --- VALIDATION ---
    # ------------------

    def _validate_inputs(self) -> None:
        if self.N <= 0:
            raise ValueError(f"N must be strictly positive; got {self.N!r}.")
        if self.N % 2 != 0:
            raise ValueError(f"N must be even; got {self.N!r}.")
        if self.L <= 0.0:
            raise ValueError(f"L must be strictly positive; got {self.L!r}.")
        if self.D not in DIMENSIONS:
            raise ValueError(f"D must be one of {DIMENSIONS}; got {self.D!r}.")
