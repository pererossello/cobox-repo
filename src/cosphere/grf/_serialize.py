from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .grf import ShellGRFSampler


_GRF_FIELDS = ("restrictions",)


def grf_to_dict(sampler: ShellGRFSampler) -> dict:
    return {"restrictions": [r.to_dict() for r in sampler.restrictions]}


def grf_from_dict(config: dict, cls: type[ShellGRFSampler]) -> ShellGRFSampler:
    from ..shell.modemask import EllModeMask

    unknown = set(config) - set(_GRF_FIELDS)
    if unknown:
        raise ValueError(f"unknown keys in ShellGRFSampler config: {sorted(unknown)}.")
    restrictions = tuple(
        EllModeMask.from_dict(r) for r in config.get("restrictions", ())
    )
    return cls(restrictions=restrictions)
