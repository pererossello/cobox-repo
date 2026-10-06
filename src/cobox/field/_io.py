"""HDF5 I/O shared by ScalarField, VectorField and TensorField."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, TypeVar

import numpy as np

if TYPE_CHECKING:
    from .scalar import ScalarField
    from .vector import VectorField
    from .tensor import TensorField

F = TypeVar("F", "ScalarField", "VectorField", "TensorField")

__all__ = ["save_field", "load_field"]


def save_field(field: ScalarField | VectorField | TensorField, path) -> None:
    """Write data and static configuration (box, space, support, symmetry)."""
    import h5py

    from ..box._serialize import box_to_dict
    from .tensor import TensorField

    config = {
        "type": type(field).__name__,
        "box": box_to_dict(field.box),
        "has_hat": bool(field.has_hat),
        "support": None if field.support is None else field.support.bound,
    }
    if isinstance(field, TensorField):
        config["symmetry"] = field.symmetry  # None | "symmetric" | "antisymmetric"
    with h5py.File(path, "w") as f:
        f.attrs["config"] = json.dumps(config)
        f.create_dataset("data", data=np.asarray(field.data))


def load_field(path, cls: type[F]) -> F:
    """Inverse of ``save_field``; the file must hold a ``cls``."""
    import h5py
    import jax.numpy as jnp

    from ..box import Box, ModeSupport
    from ..box._serialize import box_from_dict
    from .tensor import TensorField

    with h5py.File(path, "r") as f:
        config = json.loads(str(f.attrs["config"]))
        data = np.asarray(f["data"])

    if config["type"] != cls.__name__:
        raise TypeError(f"file holds a {config['type']}, not a {cls.__name__}.")
    bound = config["support"]
    kwargs = {"symmetry": config["symmetry"]} if cls is TensorField else {}
    return cls(
        data=jnp.asarray(data),
        box=box_from_dict(config["box"], Box),
        has_hat=bool(config["has_hat"]),
        support=None if bound is None else ModeSupport(bound),
        **kwargs,
    )
