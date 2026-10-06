from math import pi, sqrt

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


class Shell(eqx.Module):
    """HEALPix pixelization of the unit sphere, RING ordering, band-limited to ell < L.

    L defaults to the conservative band limit 2 nside. HEALPix transforms
    are approximate; higher band limits require convergence checks (healpy's
    lmax = 3 nside - 1 is L = 3 nside). Harmonic coefficients of real fields
    are stored on the (L, L) half-plane m >= 0, indexed [ell, m]; entries
    with m > ell are not modes and stay zero.
    """

    nside: int = eqx.field(static=True)
    L: int = eqx.field(static=True)

    def __init__(self, nside: int, L: int | None = None):
        self.nside = nside
        self.L = 2 * nside if L is None else L

        self._validate_inputs()

    # -------------
    # --- SIZES ---
    # -------------

    @property
    def NPIX(self) -> int:
        return 12 * self.nside**2

    @property
    def SHAPE(self) -> tuple[int]:
        return (self.NPIX,)

    @property
    def PIX_AREA(self) -> float:
        """Solid angle of one pixel, in sr."""
        return 4 * pi / self.NPIX

    @property
    def PIX_RES(self) -> float:
        """Square root of the pixel area, in rad."""
        return sqrt(self.PIX_AREA)

    @property
    def NRING(self) -> int:
        return 4 * self.nside - 1

    @property
    def ELL_MAX(self) -> int:
        return self.L - 1

    @property
    def HSHAPE(self) -> tuple[int, int]:
        return (self.L, self.L)

    @property
    def N_MODES(self) -> int:
        """Number of (ell, m >= 0) entries that are modes."""
        return self.L * (self.L + 1) // 2

    # --------------------
    # --- PIXEL ARRAYS ---
    # --------------------

    @property
    def pix_1D(self) -> jax.Array:
        return jnp.arange(self.NPIX)

    @property
    def theta(self) -> jax.Array:
        """Pixel-centre colatitude in [0, pi], shape SHAPE."""
        return self.pix2ang(self.pix_1D)[0]

    @property
    def phi(self) -> jax.Array:
        """Pixel-centre longitude in [0, 2 pi), shape SHAPE."""
        return self.pix2ang(self.pix_1D)[1]

    @property
    def unit_vectors(self) -> jax.Array:
        """Pixel-centre unit vectors, shape (3, NPIX)."""
        return self.pix2vec(self.pix_1D)

    @property
    def quad_weights(self) -> jax.Array:
        """Solid angle per pixel for sum_p w_p f_p ~ int f dOmega; uniform, sums to 4 pi.

        Not exact even for smooth f: the relative error falls as 1 / nside^2
        (e.g. -2.3e-5 for cos^2(theta) at nside = 64).
        """
        return jnp.full(self.SHAPE, self.PIX_AREA)

    # -------------------
    # --- RING ARRAYS ---
    # -------------------

    @property
    def ring_theta(self) -> jax.Array:
        """Colatitude of each iso-latitude ring, north to south, shape (NRING,)."""
        from . import _pixels

        return _pixels.rings(self.nside)["theta"]

    @property
    def ring_npix(self) -> jax.Array:
        from . import _pixels

        return _pixels.rings(self.nside)["npix"]

    @property
    def ring_start(self) -> jax.Array:
        """Index of each ring's first pixel."""
        from . import _pixels

        return _pixels.rings(self.nside)["start"]

    @property
    def ring_phi0(self) -> jax.Array:
        """Longitude of each ring's first pixel (0 or half a pixel)."""
        from . import _pixels

        return _pixels.rings(self.nside)["phi0"]

    # ---------------------------
    # --- DIRECTIONS / PIXELS ---
    # ---------------------------

    def pix2ang(self, pix: ArrayLike) -> tuple[jax.Array, jax.Array]:
        """Pixel-centre (theta, phi) in radians, each shaped like pix."""
        from . import _pixels

        return _pixels.pix2ang(self.nside, jnp.asarray(pix))

    def pix2vec(self, pix: ArrayLike) -> jax.Array:
        """Pixel-centre unit vectors, shape (3, *pix.shape)."""
        theta, phi = self.pix2ang(pix)
        sin_theta = jnp.sin(theta)
        return jnp.stack(
            [sin_theta * jnp.cos(phi), sin_theta * jnp.sin(phi), jnp.cos(theta)]
        )

    def ang2pix(self, theta: ArrayLike, phi: ArrayLike) -> jax.Array:
        """Pixel containing each direction; theta in [0, pi], any phi (taken mod 2 pi)."""
        from . import _pixels

        theta, phi = jnp.broadcast_arrays(jnp.asarray(theta), jnp.asarray(phi))
        theta = eqx.error_if(
            theta, jnp.any((theta < 0) | (theta > pi)), "theta must be in [0, pi]."
        )
        return _pixels.zphi2pix(self.nside, jnp.cos(theta), jnp.sin(theta), phi)

    def vec2pix(self, vec: ArrayLike) -> jax.Array:
        """Pixel containing each direction vec, shape (3, ...); need not be normalized."""
        from . import _pixels

        vec = jnp.asarray(vec)
        if vec.shape[0] != 3:
            raise ValueError(f"vec must have shape (3, ...); got {tuple(vec.shape)}.")
        x, y, z = vec
        r_perp = jnp.sqrt(x * x + y * y)
        r = jnp.sqrt(r_perp * r_perp + z * z)
        r = eqx.error_if(r, jnp.any(r == 0), "vec must be nonzero.")
        return _pixels.zphi2pix(self.nside, z / r, r_perp / r, jnp.arctan2(y, x))

    # -------------------------
    # --- HARMONIC ARRAYS ---
    # -------------------------

    @property
    def ell_1D(self) -> jax.Array:
        return jnp.arange(self.L)

    @property
    def m_1D(self) -> jax.Array:
        return jnp.arange(self.L)

    @property
    def ell_axis(self) -> jax.Array:
        return self.ell_1D.reshape(-1, 1)

    @property
    def m_axis(self) -> jax.Array:
        return self.m_1D.reshape(1, -1)

    @property
    def ell_grid(self) -> jax.Array:
        return jnp.broadcast_to(self.ell_axis, self.HSHAPE)

    @property
    def m_grid(self) -> jax.Array:
        return jnp.broadcast_to(self.m_axis, self.HSHAPE)

    @property
    def is_mode(self) -> jax.Array:
        """True where m <= ell, shape HSHAPE."""
        return self.m_axis <= self.ell_axis

    @property
    def m_weights(self) -> jax.Array:
        """Modes each half-plane entry stands for: 1 at m = 0, 2 at m > 0, 0 off modes.

        Summed over m they give 2 ell + 1, so sum_m w |f_lm|^2 / (2 ell + 1) is C_ell.
        """
        w = jnp.where(self.m_axis == 0, 1.0, 2.0)
        return jnp.where(self.is_mode, w, 0.0)

    @property
    def ell_ell1(self) -> jax.Array:
        """ell (ell + 1): minus the eigenvalue of the Laplacian, shape (L, 1)."""
        ell = self.ell_axis.astype(float)
        return ell * (ell + 1.0)

    @property
    def inv_ell_ell1(self) -> jax.Array:
        """1 / (ell (ell + 1)), zero at ell = 0, shape (L, 1)."""
        e = self.ell_ell1
        return jnp.where(e > 0, 1.0 / jnp.where(e > 0, e, 1.0), 0.0)

    # ------------------
    # --- TRANSFORMS ---
    # ------------------

    def sht(self, f: ArrayLike, iter: int = 3) -> jax.Array:
        """Real maps (..., NPIX) -> half-plane (..., L, L), f_lm = int f Y*_lm dOmega.

        HEALPix has no sampling theorem: the forward transform is a quadrature,
        refined `iter` times against the exact inverse.
        """
        from . import _harmonic

        f = jnp.asarray(f)
        if jnp.iscomplexobj(f) or f.shape[-1:] != self.SHAPE:
            raise ValueError(f"f must be real with shape (..., {self.NPIX}); got {f.dtype}{tuple(f.shape)}.")
        one = lambda x: _harmonic.sht(x, self.nside, self.L, iter)
        flat = jax.vmap(one)(f.reshape(-1, self.NPIX))
        return flat.reshape(f.shape[:-1] + self.HSHAPE)

    def isht(self, flm: ArrayLike) -> jax.Array:
        """Half-plane (..., L, L) -> real maps (..., NPIX); entries with m > ell are ignored."""
        from . import _harmonic

        flm = jnp.asarray(flm)
        if flm.shape[-2:] != self.HSHAPE:
            raise ValueError(f"flm must have shape (..., {self.L}, {self.L}); got {tuple(flm.shape)}.")
        flm = jnp.where(self.is_mode, flm, 0.0)
        one = lambda x: _harmonic.isht(x, self.nside, self.L)
        flat = jax.vmap(one)(flm.reshape((-1,) + self.HSHAPE))
        return flat.reshape(flm.shape[:-2] + self.SHAPE)

    # ---------------------
    # --- SERIALIZATION ---
    # ---------------------

    def to_dict(self) -> dict:
        from . import _serialize

        return _serialize.shell_to_dict(self)

    @classmethod
    def from_dict(cls, config: dict) -> "Shell":
        from . import _serialize

        return _serialize.shell_from_dict(config, cls)

    def to_yaml(self) -> str:
        from . import _serialize

        return _serialize.shell_to_yaml(self)

    @classmethod
    def from_yaml(cls, s: str) -> "Shell":
        from . import _serialize

        return _serialize.shell_from_yaml(s, cls)

    # -----------------------
    # --- SPECIAL METHODS ---
    # -----------------------

    def __hash__(self):
        return hash((self.nside, self.L))

    def __eq__(self, other):
        return isinstance(other, Shell) and (
            (self.nside, self.L) == (other.nside, other.L)
        )

    def __repr__(self) -> str:
        return f"Shell(nside={self.nside}, L={self.L})"

    # ------------------
    # --- VALIDATION ---
    # ------------------

    def _validate_inputs(self) -> None:
        if (
            isinstance(self.nside, bool)
            or not isinstance(self.nside, int)
            or self.nside < 1
            or self.nside & (self.nside - 1)
        ):
            raise ValueError(
                f"nside must be a positive power of 2; got {self.nside!r}."
            )
        if isinstance(self.L, bool) or not isinstance(self.L, int) or self.L < 1:
            raise ValueError(f"L must be a positive int; got {self.L!r}.")
