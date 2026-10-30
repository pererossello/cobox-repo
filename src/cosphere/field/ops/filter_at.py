"""Local window averages of a pixel-space shell field at given directions.

Each disc or annulus sum is a difference of prefix sums of the map along RING
order (shell._discs gives the pixel runs); see ShellField.filter_at.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

import jax
import jax.numpy as jnp
import numpy as np
from ...masks import ShellMask
from ...positions import AngularPositions
from ...shell._discs import disc_runs, run_sums
from ...shell._healpix import UNSEEN
from ...window import Window
from ...window._profiles import compact_regions

if TYPE_CHECKING:
    from ..shell_field import ShellField


def filter_at(
    field: "ShellField",
    points: AngularPositions,
    window: "Window | Sequence[Window]",
    *,
    mask: Optional["ShellMask"] = None,
    chunk_size: int = 10_000,
) -> jax.Array:
    """Evaluate compact windows locally; see ShellField.filter_at."""
    windows = (window,) if isinstance(window, Window) else tuple(window)
    _validate(field, windows, mask, chunk_size)

    # each window is a sum of coefficient * (mean over a disc or annulus)
    regions = [
        tuple(
            (None if inner is None else float(inner), float(outer), coefficient)
            for inner, outer, coefficient in compact_regions(
                w.kind, w.scale, w.outer_radius
            )
        )
        for w in windows
    ]
    radii = sorted(
        {r for rs in regions for region in rs for r in region[:2] if r is not None}
    )
    nside = field.shell.nside
    mean, prefix, count_prefix, bad_prefix = _prefix_sums(
        field, None if mask is None else mask.data
    )

    theta, phi = points.theta, points.phi
    results = []
    for lo in range(0, len(theta), chunk_size):
        t, p = theta[lo : lo + chunk_size], phi[lo : lo + chunk_size]

        # sums, included-pixel counts and invalid-pixel counts over each disc
        sums, counts, bad = {}, {}, {}
        for r in radii:
            runs = disc_runs(nside, t, p, r)
            sums[r] = run_sums(prefix, *runs)
            counts[r] = (
                runs[3].sum(axis=1)
                if count_prefix is None
                else run_sums(count_prefix, *runs)
            )
            if bad_prefix is not None:
                bad[r] = run_sums(bad_prefix, *runs)

        # annulus = outer disc minus inner disc; each region normalized by its count,
        # NaN where it has no included pixels
        columns = []
        for rs in regions:
            value = 0.0
            for inner, outer, coefficient in rs:
                total, count = sums[outer], counts[outer]
                if inner is not None:
                    total, count = total - sums[inner], count - counts[inner]
                if bad_prefix is not None:
                    n_bad = bad[outer] - (0 if inner is None else bad[inner])
                    if jnp.any(n_bad > 0):
                        i = lo + int(jnp.argmax(n_bad > 0))
                        raise ValueError(
                            f"Invalid map values in filter region at position {i}."
                        )
                empty = count == 0
                region = jnp.where(empty, jnp.nan, total / jnp.where(empty, 1, count))
                value = value + coefficient * (region + mean)
            columns.append(value)
        results.append(jnp.stack(columns, axis=1))
    if not results:
        result = jnp.zeros((0, len(windows)))
    else:
        result = jnp.concatenate(results)
    return result[:, 0] if isinstance(window, Window) else result


def _validate(
    field: "ShellField",
    windows: tuple[Window, ...],
    mask: Optional[ShellMask],
    chunk_size: int,
) -> None:
    """Reject inputs that would otherwise give silently wrong results or obscure errors."""
    if field.has_hat:
        raise ValueError("filter_at requires a pixel field; call .isht() explicitly.")
    if np.ma.isMaskedArray(field.data):
        raise ValueError("NumPy masked maps are not supported; pass a ShellMask.")
    if not windows:
        raise ValueError("window must be a Window or a nonempty sequence of Windows.")
    if any(w.kind == "gaussian" for w in windows):
        raise ValueError(
            "filter_at needs compact windows (disc, annulus, aperture); a gaussian has "
            "no edge. Use field.filter_ell(window).interpolate_at(theta, phi) instead."
        )
    if mask is not None:
        # a 0/1 ShellField would pass silently and ~ on integers flips bits
        if not isinstance(mask, ShellMask):
            raise TypeError("mask must be a ShellMask or None.")
        if mask.shell.nside != field.shell.nside:
            raise ValueError("mask must share the field's pixelization (nside).")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive; got {chunk_size!r}.")
    if not jax.config.jax_enable_x64:
        raise ValueError(
            "filter_at requires jax_enable_x64: in float32 the disc pixel selection "
            "is wrong near the poles and the map prefix sums lose precision."
        )


def _prefix_sums(
    field: "ShellField", keep: Optional[jax.Array]
) -> tuple[jax.Array, jax.Array, Optional[jax.Array], Optional[jax.Array]]:
    """Mean of the usable pixels and prefix sums of values, included and invalid pixels.

    Subtracting the mean keeps the value prefix small. Excluded pixels contribute
    neither signal nor count; invalid included pixels are tracked separately and
    rejected only inside a region. The count prefix is None without a mask, and
    the invalid prefix is None when no included pixel is invalid. Full-map
    float64 temporaries are released as soon as possible (nside 8192: 6.4 GB each).
    """
    values = jnp.asarray(field.data, dtype=float)
    invalid = ~jnp.isfinite(values) | jnp.isclose(values, UNSEEN)
    if keep is not None:
        invalid = invalid & keep
    any_invalid = bool(jnp.any(invalid))
    usable = ~invalid if keep is None else keep & ~invalid
    mean = jnp.mean(values, where=usable)
    values = values - mean
    if any_invalid or keep is not None:
        values = jnp.where(usable, values, 0.0)
    prefix = jnp.cumsum(values)
    count_prefix = jnp.cumsum(keep) if keep is not None else None
    bad_prefix = jnp.cumsum(invalid) if any_invalid else None
    return mean, prefix, count_prefix, bad_prefix
