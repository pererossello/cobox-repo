"""Complete parameter presets; closure sets Omega_de = 1 - Omega_m - Omega_k."""

from typing import Literal

# Planck 2018 TT,TE,EE+lowE+lensing, base LCDM (VI, Table 2).
# Background matches astropy Planck18: H0=67.66, Om0=0.30966, Ob0=0.04897.
PLANCK18 = {
    "background": {
        "h": 0.6766,
        "Omega_b": 0.04897,
        "Omega_cdm": 0.26069,
        "Omega_k": 0.0,
        "w0": -1.0,
        "wa": 0.0,
    },
    "primordial": {"As1e9": 2.105, "n_s": 0.9665, "k_pivot": 0.05},
}

# Round-number flat LCDM for tests and experiments.
FIDUCIAL = {
    "background": {
        "h": 0.7,
        "Omega_b": 0.05,
        "Omega_cdm": 0.25,
        "Omega_k": 0.0,
        "w0": -1.0,
        "wa": 0.0,
    },
    "primordial": {"As1e9": 2.1, "n_s": 0.96, "k_pivot": 0.05},
}

DEFAULT_COSMOS = {"planck18": PLANCK18, "fiducial": FIDUCIAL}
DefaultCosmosLiteral = Literal["planck18", "fiducial"]
DEFAULT_NAME = "planck18"
