from __future__ import annotations

from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from .shell import Shell


_SHELL_FIELDS = ("nside", "L")


def shell_to_dict(shell: "Shell") -> dict:
    return {"nside": int(shell.nside), "L": int(shell.L)}


def shell_from_dict(config: dict, cls: Type[Shell]) -> Shell:
    unknown = set(config) - set(_SHELL_FIELDS)
    if unknown:
        raise ValueError(f"unknown keys in Shell config: {sorted(unknown)}.")
    L = config.get("L")
    return cls(nside=int(config["nside"]), L=None if L is None else int(L))


def shell_to_yaml(shell: "Shell") -> str:
    import yaml

    return yaml.safe_dump(shell_to_dict(shell), sort_keys=False)


def shell_from_yaml(s: str, cls: Type[Shell]) -> Shell:
    import yaml

    return shell_from_dict(yaml.safe_load(s), cls)
