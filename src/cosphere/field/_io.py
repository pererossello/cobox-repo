from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .shell_field import ShellField

__all__ = ["save_shell_field", "load_shell_field"]


def save_shell_field(field: "ShellField", path) -> None:
    import h5py

    from ..shell._serialize import shell_to_dict

    config = {
        "shell": shell_to_dict(field.shell),
        "has_hat": bool(field.has_hat),
    }
    with h5py.File(path, "w") as f:
        f.attrs["config"] = json.dumps(config)
        f.create_dataset("data", data=np.asarray(field.data))


def load_shell_field(path) -> "ShellField":
    import h5py
    import jax.numpy as jnp

    from .shell_field import ShellField
    from ..shell import Shell
    from ..shell._serialize import shell_from_dict

    with h5py.File(path, "r") as f:
        config = json.loads(str(f.attrs["config"]))
        data = np.asarray(f["data"])

    return ShellField(
        data=jnp.asarray(data),
        shell=shell_from_dict(config["shell"], Shell),
        has_hat=bool(config["has_hat"]),
    )
