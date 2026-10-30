"""Host-side plotting cuts and reproducible display sampling."""

from numbers import Integral

import numpy as np


def limits(value, name):
    value = np.asarray(value)
    if value.shape != (2,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain two finite endpoints.")
    return tuple(float(x) for x in value)


def select(catalog, ra_limits, dec_limits, z_limits):
    """Return original row indices and unwrapped RA; cuts are half-open."""
    ra = np.mod(np.rad2deg(np.asarray(catalog.phi)), 360)
    dec = 90 - np.rad2deg(np.asarray(catalog.theta))
    z = np.asarray(catalog.z)
    keep = np.ones(len(catalog), dtype=bool)
    span = (0.0, 360.0)
    if ra_limits is not None:
        lo, hi = limits(ra_limits, "ra_limits")
        width = hi - lo
        if width == 0 or abs(width) > 360:
            raise ValueError(
                "RA range must have a nonzero width of at most 360 degrees."
            )
        width = 360.0 if abs(width) == 360 else width % 360
        ra = lo + (ra - lo) % 360
        keep &= ra < lo + width
        span = (lo, lo + width)
    if dec_limits is not None:
        lo, hi = limits(dec_limits, "dec_limits")
        if not -90 <= lo < hi <= 90:
            raise ValueError("Require -90 <= dec_min < dec_max <= 90 degrees.")
        keep &= (dec >= lo) & (dec < hi)
    if z_limits is not None:
        lo, hi = limits(z_limits, "z_limits")
        if lo >= hi:
            raise ValueError("Require z_min < z_max.")
        keep &= (z >= lo) & (z < hi)
    return np.flatnonzero(keep), ra, dec, span


def sample(indices, max_points, seed):
    if max_points is None:
        return indices
    if (
        isinstance(max_points, bool)
        or not isinstance(max_points, Integral)
        or max_points <= 0
    ):
        raise ValueError("max_points must be a positive integer or None.")
    if len(indices) <= max_points:
        return indices
    return np.sort(
        np.random.default_rng(seed).choice(indices, max_points, replace=False)
    )
