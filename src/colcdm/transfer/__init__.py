"""Transfers T_X(k, a) from the primordial curvature perturbation to fields.

X(k, a) = T_X(k, a) R(k), with k in h/Mpc; each transfer is a Spectrum.
"""

from ._serialize import transfer_from_dict
from .matter import MatterTransfer

__all__ = ["MatterTransfer", "transfer_from_dict"]
