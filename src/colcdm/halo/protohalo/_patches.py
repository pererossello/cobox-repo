"""Statistics of the spheres of radius R centred on every node."""

import equinox as eqx
import jax

from cobox.field import ScalarField
from cobox.window import Window

PATCH_STATISTICS = ("delta",)


class Patch(eqx.Module):
    """Spheres of radius R [Mpc/h], one per node; linear statistics at a = 1."""

    R: float = eqx.field(static=True)
    delta: jax.Array  # (*SHAPE) mean linear overdensity in the sphere


def measure(delta0: ScalarField, R: float, needs: tuple[str, ...]) -> Patch:
    """Sphere averages of delta0 around each node.

    The Fourier top-hat gives the exact sphere average of the band-limited
    field, sampled at the nodes. Pass delta0 in Fourier space to share its
    FFT across radii. needs lists the statistics a collapse criterion reads.
    """
    unknown = set(needs) - set(PATCH_STATISTICS)
    if unknown:
        raise ValueError(f"unknown patch statistics: {sorted(unknown)}.")
    window = Window("radial_tophat", 2.0 * R, normalized=True)  # scale is a diameter
    return Patch(R=R, delta=delta0.convolve(window).ifft().data)
