from __future__ import annotations
from typing import Any, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .linear_power import LinearMatterPowSpec


_LINEAR_POWER_FIELDS = ("transfer_kind", "growth_kind")


def linear_power_to_dict(spec: "LinearMatterPowSpec") -> dict[str, Any]:
    return {
        "transfer_kind": spec.transfer_kind,
        "growth_kind": spec.growth_kind,
    }


def linear_power_from_dict(
    config: dict,
    cls: Type[LinearMatterPowSpec],
) -> LinearMatterPowSpec:
    unknown = set(config) - set(_LINEAR_POWER_FIELDS)
    if unknown:
        raise ValueError(
            f"unknown keys in LinearMatterPowSpec config: {sorted(unknown)}."
        )
    return cls(
        transfer_kind=config["transfer_kind"],
        growth_kind=config["growth_kind"],
    )


def linear_power_to_yaml(spec: "LinearMatterPowSpec") -> str:
    import yaml

    return yaml.safe_dump(linear_power_to_dict(spec), sort_keys=False)


def linear_power_from_yaml(
    s: str,
    cls: Type[LinearMatterPowSpec],
) -> LinearMatterPowSpec:
    import yaml

    return linear_power_from_dict(yaml.safe_load(s), cls)
