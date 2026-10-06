from typing import Callable, Literal, Sequence, Optional, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from . import _randoms

from ..box.modemask import ModeMask
from ..box.support import ModeSupport

if TYPE_CHECKING:
    from ..box import Box
    from ..field.scalar import ScalarField

KindLiteral = Literal["real", "complex"]
ComplexBaseLiteral = Literal["cartesian"]


class GRFSampler(eqx.Module):
    kind: KindLiteral = eqx.field(static=True)
    complex_base: Optional[ComplexBaseLiteral] = eqx.field(static=True, default=None)
    restrictions: tuple[ModeMask, ...] = eqx.field(static=True, default=())

    def __init__(
        self,
        kind: KindLiteral,
        complex_base: Optional[ComplexBaseLiteral] = None,
        restrictions: ModeMask | Sequence[ModeMask] = (),
    ):
        self.kind = kind
        self.complex_base = complex_base
        if isinstance(restrictions, ModeMask):
            restrictions = (restrictions,)
        self.restrictions = tuple(restrictions)
        self._validate_static()

    def get_randoms(self, seed: int, box: "Box") -> dict:
        key = jax.random.PRNGKey(seed)
        if self.kind == "real":
            return _randoms.get_randoms_real(key, box)
        # complex
        mask = self._get_mask(box)
        return _randoms.get_randoms_complex_cartesian(key, box, mask)

    def sample(
        self,
        randoms: dict,
        box: "Box",
        pk_fn: Callable[[jax.Array], jax.Array],
    ) -> "ScalarField":
        if self.kind == "real":
            return _randoms.sample_real(randoms, box, pk_fn)
        mask = self._get_mask(box)
        field = _randoms.sample_complex_cartesian(randoms, box, mask, pk_fn)
        return field.with_support(self.support(box))

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
        mask = self._get_mask(box)
        return _randoms.n_dof_randoms_complex_cartesian(box, mask)

    # ------------------
    # --- INFERENCE ----
    # ------------------

    def get_dist(self) -> dict:
        return {"u": dist.Normal(0.0, 1.0)}

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

    def support(self, box: "Box") -> Optional[ModeSupport]:
        """ModeSupport of sampled fields, or None (real-space white noise).

        Restrictions are intersected, so the tightest one bounds the support.
        """
        if self.kind == "real":
            return None
        return ModeSupport.of_intersection(
            None, *(r.support(box) for r in self.restrictions)
        )

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
            if self.complex_base != "cartesian":
                raise ValueError(
                    "kind='complex' only supports complex_base='cartesian'; "
                    f"got {self.complex_base!r}."
                )
