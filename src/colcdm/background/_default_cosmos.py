"""
``Omega_de`` fixed by flatness closure (Omega_de = 1 - Omega_m - Omega_k)
"""

from typing import Literal

# Planck 2018 (TT,TE,EE+lowE+lensing, base LambdaCDM; Planck 2018 VI, Table 2;
# matches astropy Planck18 background: H0=67.66, Om0=0.30966, Ob0=0.04897).
# omega_b = 0.02242, omega_cdm = 0.11933, n_s = 0.9665, A_s = 2.105e-9.
PLANCK18 = {
    "h": 0.6766,
    "Omega_b": 0.04897,
    "Omega_cdm": 0.26069,
    "Omega_k": 0.0,
    "n_s": 0.9665,
    "As1e9": 2.105,
}

# Round-number flat LambdaCDM for tests / quick experiments.
#   Omega_m = 0.30, Omega_de = 0.70 (derived).
FIDUCIAL = {
    "h": 0.7,
    "Omega_b": 0.05,
    "Omega_cdm": 0.25,
    "Omega_k": 0.0,
    "n_s": 0.96,
    "As1e9": 2.1,
}

DEFAULT_COSMOS = {
    "planck18": PLANCK18,
    "fiducial": FIDUCIAL,
}

DefaultCosmosLiteral = Literal["planck18", "fiducial"]

DEFAULT_NAME = "planck18"
