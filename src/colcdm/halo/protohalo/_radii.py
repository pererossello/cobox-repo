"""Descending radius grid of the excursion-set sweep, in Mpc/h."""

from math import ceil, exp, log

from cobox.box import Box, ModeSupport


def resolve_radii(
    box: Box,
    support: ModeSupport | None,
    *,
    dlnR: float,
    R_min_factor: float,
    R_max: float | None,
) -> tuple[float, ...]:
    """Log-spaced radii from R_max down to R_min = R_min_factor * L / N_eff.

    N_eff is the smallest grid holding the field's declared support
    (support.N_min), or box.N without one. Below ~L / N_eff the sphere average
    no longer depends on R, so the field, not the grid, sets R_min.
    R_max=None uses L / 4, also the upper bound: larger spheres overlap their
    own periodic images. Both ends are included and the step in ln R is at
    most dlnR.
    """
    N_eff = box.N if support is None else support.N_min
    R_min = R_min_factor * box.L / N_eff
    R_max = 0.25 * box.L if R_max is None else float(R_max)
    if R_max > 0.25 * box.L:
        raise ValueError(f"R_max must be <= L / 4 = {0.25 * box.L}; got {R_max}.")
    if not R_min < R_max:
        raise ValueError(f"require R_min < R_max; got ({R_min}, {R_max}).")

    n = ceil(log(R_max / R_min) / dlnR) + 1
    step = log(R_max / R_min) / (n - 1)
    return tuple(R_max * exp(-step * i) for i in range(n))
