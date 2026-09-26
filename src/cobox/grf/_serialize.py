from __future__ import annotations
from typing import Any, Type, TYPE_CHECKING

from ..box.modemask import IsoModeMask

if TYPE_CHECKING:
    from .grf import GRFSampler


_GRF_FIELDS = ("kind", "complex_base", "restrictions")


def grf_to_dict(sampler: "GRFSampler") -> dict[str, Any]:
    assert isinstance(sampler.restrictions, tuple)
    return {
        "kind": sampler.kind,
        "complex_base": sampler.complex_base,
        "restrictions": [r.to_dict() for r in sampler.restrictions],
    }


def grf_from_dict(config: dict, cls: Type[GRFSampler]) -> GRFSampler:
    unknown = set(config) - set(_GRF_FIELDS)
    if unknown:
        raise ValueError(f"unknown keys in GRFSampler config: {sorted(unknown)}.")
    return cls(
        kind=config["kind"],
        complex_base=config.get("complex_base"),
        restrictions=tuple(
            IsoModeMask.from_dict(r) for r in (config.get("restrictions") or ())
        ),
    )


def grf_to_yaml(sampler: "GRFSampler") -> str:
    import yaml

    return yaml.safe_dump(grf_to_dict(sampler), sort_keys=False)


def grf_from_yaml(s: str, cls: Type[GRFSampler]) -> GRFSampler:
    import yaml

    return grf_from_dict(yaml.safe_load(s), cls)
