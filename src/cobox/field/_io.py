from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .scalar import ScalarField
    from .vector import VectorField
    from .tensor import TensorField

__all__ = ["save_scalar_field", "load_scalar_field"]


def save_scalar_field(field: "ScalarField", path) -> None:
    import h5py

    from ..box._serialize import box_to_dict
    from ..window._serialize import windows_to_list

    config = {
        "box": box_to_dict(field.box),
        "has_hat": bool(field.has_hat),
    }
    with h5py.File(path, "w") as f:
        f.attrs["config"] = json.dumps(config)
        f.create_dataset("data", data=np.asarray(field.data))


def load_scalar_field(path) -> "ScalarField":
    import h5py
    import jax.numpy as jnp

    from .scalar import ScalarField
    from ..box import Box
    from ..box._serialize import box_from_dict
    from ..window._serialize import windows_from_list

    with h5py.File(path, "r") as f:
        config = json.loads(str(f.attrs["config"]))
        data = np.asarray(f["data"])

    return ScalarField(
        data=jnp.asarray(data),
        box=box_from_dict(config["box"], Box),
        has_hat=bool(config["has_hat"]),
    )


def save_vector_field(field: "VectorField", path) -> None:
    import h5py

    from ..box._serialize import box_to_dict

    config = {
        "box": box_to_dict(field.box),
        "has_hat": bool(field.has_hat),
    }
    with h5py.File(path, "w") as f:
        f.attrs["config"] = json.dumps(config)
        f.create_dataset("data", data=np.asarray(field.data))


def load_vector_field(path) -> "VectorField":
    import h5py
    import jax.numpy as jnp

    from .vector import VectorField
    from ..box import Box
    from ..box._serialize import box_from_dict

    with h5py.File(path, "r") as f:
        config = json.loads(str(f.attrs["config"]))
        data = np.asarray(f["data"])

    return VectorField(
        data=jnp.asarray(data),
        box=box_from_dict(config["box"], Box),
        has_hat=bool(config["has_hat"]),
    )


def save_tensor_field(field: "TensorField", path) -> None:
    import h5py

    from ..box._serialize import box_to_dict

    config = {
        "box": box_to_dict(field.box),
        "has_hat": bool(field.has_hat),
        "symmetry": field.symmetry,  # None | "symmetric" | "antisymmetric" — JSON-native
    }
    with h5py.File(path, "w") as f:
        f.attrs["config"] = json.dumps(config)
        f.create_dataset("data", data=np.asarray(field.data))


def load_tensor_field(path) -> "TensorField":
    import h5py
    import jax.numpy as jnp

    from .tensor import TensorField
    from ..box import Box
    from ..box._serialize import box_from_dict

    with h5py.File(path, "r") as f:
        config = json.loads(str(f.attrs["config"]))
        data = np.asarray(f["data"])

    return TensorField(
        data=jnp.asarray(data),
        box=box_from_dict(config["box"], Box),
        has_hat=bool(config["has_hat"]),
        symmetry=config["symmetry"],
    )
