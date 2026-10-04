"""JAX port of symbolic_pofk linear power spectrum emulators.

Ported from https://github.com/DeaglanBartlett/symbolic_pofk (linear_new.py, linear.py).
Original code Copyright (c) 2023 Deaglan Bartlett. MIT License.

Pruned to the present-day matter transfer: the no-wiggle Eisenstein & Hu transfer
without its primordial prefactor, and the logF and S corrections. Unused
arguments are removed; equations and coefficients are unchanged.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import jax.numpy as np


def eisenstein_hu_nw_transfer(k, Om, Ob, h):
    """
    No-wiggles Eisenstein & Hu transfer function T_EH(k), without the
    primordial and Poisson prefactors of the original P(k).

    Args:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc

    Returns:
        :tk_eh (np.ndarray): No-wiggle transfer function at corresponding k values
    """

    ombom0 = Ob / Om
    om0h2 = Om * h**2
    ombh2 = Ob * h**2
    theta2p7 = 2.7255 / 2.7  # Assuming Tcmb0 = 2.7255 Kelvin

    # Compute scale factor s, alphaGamma, and effective shape Gamma
    s = 44.5 * np.log(9.83 / om0h2) / np.sqrt(1.0 + 10.0 * ombh2**0.75)
    alphaGamma = (
        1.0
        - 0.328 * np.log(431.0 * om0h2) * ombom0
        + 0.38 * np.log(22.3 * om0h2) * ombom0**2
    )
    Gamma = Om * h * (alphaGamma + (1.0 - alphaGamma) / (1.0 + (0.43 * k * h * s) ** 4))

    # Compute q, C0, L0, and tk_eh
    q = k * theta2p7**2 / Gamma
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    L0 = np.log(2.0 * np.exp(1.0) + 1.8 * q)
    tk_eh = L0 / (L0 + C0 * q**2)

    return tk_eh


def lcdm_logF_fiducial(k, Om, Ob, h):
    """
    Compute the emulated logarithm of the ratio between the true linear
    power spectrum and the Eisenstein & Hu 1998 fit. Here we use the fiducial exprssion
    given in Bartlett et al. 2023, extrapolated to all k.

    Args:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc

    Returns:
        :logF (np.ndarray): The logarithm of the ratio between the linear P(k) and the
            Eisenstein & Hu 1998 zero-baryon fit
    """

    b = [
        0.05448654,
        0.00379,
        0.0396711937097927,
        0.127733431568858,
        1.35,
        4.053543862744234,
        0.0008084539054750851,
        1.8852431049189666,
        0.11418372931475675,
        3.798,
        14.909,
        5.56,
        15.8274343004709,
        0.0230755621512691,
        0.86531976,
        0.8425442636372944,
        4.553956000000005,
        5.116999999999995,
        70.0234239999998,
        0.01107,
        5.35,
        6.421,
        134.309,
        5.324,
        21.532,
        4.741999999999985,
        16.68722499999999,
        3.078,
        16.987,
        0.05881491,
        0.0006864690561825617,
        195.498,
        0.0038454457516892,
        0.276696018851544,
        7.385,
        12.3960625361899,
        0.0134114370723638,
    ]

    line1 = b[0] * h - b[1]

    line2 = ((Ob * b[2]) / np.sqrt(h**2 + b[3])) ** (b[4] * Om) * (
        (b[5] * k - Ob)
        / np.sqrt(b[6] + (Ob - b[7] * k) ** 2)
        * b[8]
        * (b[9] * k) ** (-b[10] * k)
        * np.cos(Om * b[11] - (b[12] * k) / np.sqrt(b[13] + Ob**2))
        - b[14]
        * ((b[15] * k) / np.sqrt(1 + b[16] * k**2) - Om)
        * np.cos(b[17] * h / np.sqrt(1 + b[18] * k**2))
    )

    line3 = (
        b[19]
        * (b[20] * Om + b[21] * h - np.log(b[22] * k) + (b[23] * k) ** (-b[24] * k))
        * np.cos(b[25] / np.sqrt(1 + b[26] * k**2))
    )

    line4 = (
        (b[27] * k) ** (-b[28] * k)
        * (
            b[29] * k
            - (b[30] * np.log(b[31] * k)) / np.sqrt(b[32] + (Om - b[33] * h) ** 2)
        )
        * np.cos(Om * b[34] - (b[35] * k) / np.sqrt(Ob**2 + b[36]))
    )

    logF = line1 + line2 + line3 + line4

    return logF


def log10_S(k, Om, Ob, h, mnu, w0, wa):
    """
    Corrections to the present-day linear power spectrum

    Args:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :mnu (float): Sum of neutrino masses [eV / c^2]
        :w0 (float): Time independent part of the dark energy EoS
        :wa (float): Time dependent part of the dark energy EoS

    Returns:
        :result (np.ndarray): Corrections to the present-day linear power spectrum
    """

    e = np.array(
        [
            0.2841,
            0.1679,
            0.0534,
            0.0024,
            0.1183,
            0.3971,
            0.0985,
            0.0009,
            0.1258,
            0.2476,
            0.1841,
            0.0316,
            0.1385,
            0.2825,
            0.8098,
            0.019,
            0.1376,
            0.3733,
        ]
    )

    part1 = -e[0] * h
    part2 = -e[1] * w0
    part3 = -e[2] * mnu / np.sqrt(e[3] + k**2)

    part4 = -(e[4] * h) / (e[5] * h + mnu)

    part5 = e[6] * mnu / (h * np.sqrt(e[7] + (Om * e[8] + k) ** 2))

    numerator_inner = (
        e[9] * Ob - e[10] * w0 - e[11] * wa + (e[12] * w0 + e[13]) / (e[14] * wa + w0)
    )
    denominator_inner = np.sqrt(e[15] + (Om + e[16] * np.log(-e[17] * w0)) ** 2)

    part6 = numerator_inner / denominator_inner

    # Sum all parts to get the final result
    result = part1 + part2 + part3 + part4 + part5 + part6

    return result / 10
