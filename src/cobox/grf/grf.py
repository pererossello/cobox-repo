from typing import Callable, Literal, Sequence, Optional, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from . import _randoms

from ..box import _fourier
from ..box.modemask import IsoModeMask

if TYPE_CHECKING:
    from ..box import Box
    from ..field.scalar import ScalarField

KindLiteral = Literal["real", "complex"]
ComplexBaseLiteral = Literal["cartesian", "polar", "polar_fix_amp"]


class GRFSampler(eqx.Module):

    kind: KindLiteral = eqx.field(static=True)
    complex_base: Optional[ComplexBaseLiteral] = eqx.field(static=True, default=None)
    restrictions: tuple[IsoModeMask, ...] = eqx.field(static=True, default=())

    def __init__(
        self,
        kind: KindLiteral,
        complex_base: Optional[ComplexBaseLiteral] = None,
        restrictions: IsoModeMask | Sequence[IsoModeMask] = (),
    ):
        self.kind = kind
        self.complex_base = complex_base
        if isinstance(restrictions, IsoModeMask):
            restrictions = (restrictions,)
        self.restrictions = tuple(restrictions)
        self._validate_static()

    def get_randoms(self, seed: int, box: "Box") -> dict:
        key = jax.random.PRNGKey(seed)
        if self.kind == "real":
            return _randoms.get_randoms_real(key, box)
        # complex
        mask = self._get_mask(box)
        if self.complex_base == "cartesian":
            return _randoms.get_randoms_complex_cartesian(key, box, mask)
        else:
            raise NotImplementedError("...")

    def sample(
        self,
        randoms: dict,
        box: "Box",
        pk_fn: Callable[[jax.Array], jax.Array],
    ) -> "ScalarField":
        if self.kind == "real":
            return _randoms.sample_real(randoms, box, pk_fn)
        elif self.complex_base == "cartesian":
            mask = self._get_mask(box)
            return _randoms.sample_complex_cartesian(randoms, box, mask, pk_fn)
        else:
            raise NotImplementedError("...")

    def sample_from_seed(
        self,
        seed: int,
        box: "Box",
        pk_fn: Callable[[jax.Array], jax.Array],
    ) -> "ScalarField":
        return self.sample(self.get_randoms(seed, box), box, pk_fn)

    def get_n_dof(self, box: "Box") -> dict:
        if self.kind == "real":
            return _randoms.n_dof_randoms_real(box)
        else:  # complex
            mask = self._get_mask(box)
            if self.complex_base == "cartesian":
                return _randoms.n_dof_randoms_complex_cartesian(box, mask)
            else:
                raise NotImplementedError("")

    # ------------------
    # --- INFERENCE ----
    # ------------------

    def get_dist(self) -> dict:
        if self.kind == "real":
            return {"u": dist.Normal(0.0, 1.0)}
        else:  # complex
            if self.complex_base == "cartesian":
                return {"u": dist.Normal(0.0, 1.0)}
            else:
                raise NotImplementedError("")

    def get_randoms_numpyro(self, box: "Box", prefix: str = "") -> dict:

        return {
            key_name: numpyro.sample(
                f"{prefix}{key_name}",
                self.get_dist()[key_name].expand((n,)).to_event(1),
            )
            for key_name, n in self.get_n_dof(box).items()
        }

    # ------------------
    # --- UTILITIES ----
    # ------------------

    def N_iso(self, box: "Box") -> Optional[int]:
        """Declared support |k_idx| < N_iso / 2 of sampled fields, or None.

        Restrictions are intersected, so the smallest cut bounds the support.
        """
        if self.kind == "real":
            return None
        cuts = [n for r in self.restrictions if (n := r.N_iso(box)) is not None]
        return min(cuts, default=None)

    def _get_mask(self, box: "Box"):
        # Depends only on static data: evaluate at trace time so the sizes
        # derived from it (n_dof, pack/unpack) stay Python ints under jit.
        with jax.ensure_compile_time_eval():
            mask = jnp.ones(box.KSHAPE, dtype=bool)
            assert isinstance(self.restrictions, tuple)
            for r in self.restrictions:
                mask = mask & r.mask(box)
            # k=0 is never a free DOF: its power is always forced to zero.
            return mask.at[(0,) * box.D].set(False)

    # ---------------------
    # --- SERIALIZATION ---
    # ---------------------

    def to_dict(self) -> dict:
        from . import _serialize

        return _serialize.grf_to_dict(self)

    @classmethod
    def from_dict(cls, config: dict) -> "GRFSampler":
        from . import _serialize

        return _serialize.grf_from_dict(config, cls)

    def to_yaml(self) -> str:
        from . import _serialize

        return _serialize.grf_to_yaml(self)

    @classmethod
    def from_yaml(cls, s: str) -> "GRFSampler":
        from . import _serialize

        return _serialize.grf_from_yaml(s, cls)

    # ------------------
    # --- VALIDATION ---
    # ------------------

    def _validate_static(self) -> None:
        if self.kind not in ("real", "complex"):
            raise ValueError(f"unknown kind: {self.kind!r}.")
        if self.kind == "real":
            if self.complex_base is not None:
                raise ValueError("complex_base is only meaningful with kind='complex'.")
            if self.restrictions:
                raise ValueError(
                    "restrictions are only meaningful with kind='complex'."
                )
        else:  # complex
            if self.complex_base not in ("cartesian", "polar", "polar_fix_amp"):
                raise ValueError(
                    "kind='complex' requires complex_base in {'cartesian', 'polar', 'polar_fix_amp'}; "
                    f"got {self.complex_base!r}."
                )
