"""Configuration serialization for multipole masks."""

_ELL_MASK_FIELDS = ("ell_range",)


def ell_mask_to_dict(m) -> dict:
    return {"ell_range": [None if v is None else int(v) for v in m.ell_range]}


def ell_mask_from_dict(config: dict, cls):
    unknown = set(config) - set(_ELL_MASK_FIELDS)
    if unknown:
        raise ValueError(f"unknown keys in EllModeMask config: {sorted(unknown)}.")
    lo, hi = config["ell_range"]
    return cls(ell_range=(None if lo is None else int(lo), None if hi is None else int(hi)))
