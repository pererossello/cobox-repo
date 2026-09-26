from __future__ import annotations
from typing import Optional, Any, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .box import Box
    from .modemask import IsoModeMask


# -----------
# --- BOX ---
# -----------

_BOX_FIELDS = ("N", "L", "D")


def box_to_dict(box: "Box") -> dict:
    return {
        "N": int(box.N),
        "L": float(box.L),
        "D": int(box.D),
    }


def box_from_dict(config: dict, cls: Type[Box]) -> Box:
    unknown = set(config) - set(_BOX_FIELDS)
    if unknown:
        raise ValueError(f"unknown keys in Box config: {sorted(unknown)}.")
    return cls(
        N=int(config["N"]),
        L=float(config["L"]),
        D=int(config["D"]),
    )


def box_to_yaml(box: "Box") -> str:
    import yaml

    return yaml.safe_dump(box_to_dict(box), sort_keys=False)


def box_from_yaml(s: str, cls: Type[Box]) -> Box:
    import yaml

    return box_from_dict(yaml.safe_load(s), cls)


# ----------------
# --- ISO MASK ---
# ----------------


_ISO_FIELDS = ("mode", "k_range")


def iso_mask_to_dict(m: "IsoModeMask") -> dict[str, Any]:
    d: dict[str, Any] = {"mode": m.mode}
    if m.k_range is not None:
        d["k_range"] = [None if v is None else float(v) for v in m.k_range]
    return d


def iso_mask_from_dict(config: dict, cls: Type[IsoModeMask]) -> IsoModeMask:
    unknown = set(config) - set(_ISO_FIELDS)
    if unknown:
        raise ValueError(f"unknown keys in IsoModeMask config: {sorted(unknown)}.")

    k_range: Optional[tuple[Optional[float], Optional[float]]] = None
    raw = config.get("k_range")
    if raw is not None:
        k_lo, k_hi = raw
        k_range = (
            None if k_lo is None else float(k_lo),
            None if k_hi is None else float(k_hi),
        )
    return cls(mode=config["mode"], k_range=k_range)


def iso_mask_to_yaml(m: "IsoModeMask") -> str:
    import yaml

    return yaml.safe_dump(iso_mask_to_dict(m), sort_keys=False)


def iso_mask_from_yaml(s: str, cls: Type[IsoModeMask]) -> IsoModeMask:
    import yaml

    return iso_mask_from_dict(yaml.safe_load(s), cls)
