"""JAX port of symbolic_pofk linear power spectrum emulators.

Ported from https://github.com/DeaglanBartlett/symbolic_pofk (linear_new.py, linear.py).
Original code Copyright (c) 2023 Deaglan Bartlett. MIT License.

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


def growth_correction_R(As, Om, Ob, h, ns, mnu, w0, wa, a):
    """
    Correction to the growth factor

    Args:
        :As (float): 10^9 times the amplitude of the primordial P(k)
        :Om (float): The z=0 total matter density parameter, Om
        :Ob (float): The z=0 baryonic density parameter, Ob
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :mnu (float): Sum of neutrino masses [eV / c^2]
        :w0 (float): Time independent part of the dark energy EoS
        :wa (float): Time dependent part of the dark energy EoS
        :a (float): The scale factor to evaluate P(k) at

    Returns:
        :result (float): correction to the growth factor
    """

    d = np.array(
        [
            0.8545,
            0.394,
            0.7294,
            0.5347,
            0.4662,
            4.6669,
            0.4136,
            1.4769,
            0.5959,
            0.4553,
            0.0799,
            5.8311,
            5.8014,
            6.7085,
            0.3445,
            1.2498,
            0.3756,
            0.2136,
        ]
    )

    part1 = d[0]

    denominator_inner1 = (
        a * d[1] + d[2] + (Om * d[3] - a * d[4]) * np.log(-d[5] * w0 - d[6] * wa)
    )
    part2 = -1 / denominator_inner1

    numerator_inner2 = Om * d[7] - a * d[8] + np.log(-d[9] * w0 - d[10] * wa)
    denominator_inner2 = (
        -a * d[11]
        + d[12]
        + d[13] * (Om * d[14] + a * d[15] - 1) * (d[16] * w0 + d[17] * wa + 1)
    )
    part3 = -numerator_inner2 / denominator_inner2

    result = 1 + (1 - a) * (part1 + part2 + part3)

    return result


def get_approximate_D(k, As, Om, Ob, h, ns, mnu, w0, wa, a):
    """
    Approximation to the growth factor using the results of
    Bond et al. 1980, Lahav et al. 1992, Carrol et al. 1992
    and Eisenstein & Hu 1997 (D_cbnu).

    There are two differences between our method and theirs.
    First, in Eisenstein & Hu 1997 D is chosen to be (1 + zeq) a at
    early times, whereas we instead choose D -> a at early times.
    Second, the formulae reported there assume that w=-1, whereas we
    change the Omega_Lambda terms to include a w0-wa parameterisation.

    Args:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]
        :As (float): 10^9 times the amplitude of the primordial P(k)
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :mnu (float): Sum of neutrino masses [eV / c^2]
        :w0 (float): Time independent part of the dark energy EoS
        :wa (float): Time dependent part of the dark energy EoS
        :a (float): Scale factor to consider

    Returns:
        :D (np.ndarray): Approximate linear growth factor at corresponding k values
    """

    # avoid singularities
    mnu = mnu + 1e-10

    #  Get fitting formula without free-streaming
    z = 1 / a - 1
    theta2p7 = 2.7255 / 2.7  # Assuming Tcmb0 = 2.7255 Kelvin
    zeq = 2.5e4 * Om * h**2 / theta2p7**4

    Omega = Om * a ** (-3)
    OL = (1 - Om) * a ** (-3 * (1 + w0 + wa)) * np.exp(-3 * wa * (1 - a))
    g = np.sqrt(Omega + OL)
    Omega /= g**2
    OL /= g**2

    D1 = (
        (1 + zeq)
        / (1 + z)
        * 5
        * Omega
        / 2
        / (Omega ** (4 / 7) - OL + (1 + Omega / 2) * (1 + OL / 70))
    )

    # Split Omega_m into CDM, Baryons and Neutrinos
    Onu = mnu / 93.14 / h**2
    Oc = Om - Ob - Onu
    fc = Oc / Om
    fb = Ob / Om
    fnu = Onu / Om
    fcb = fc + fb

    # Add Bond et al. 1980 suppression
    pcb = 1 / 4 * (5 - np.sqrt(1 + 24 * fcb))
    Nnu = np.where(mnu != 0.0, 3, 0)
    q = k * h * theta2p7**2 / (Om * h**2)
    yfs = 17.2 * fnu * (1 + 0.488 / fnu ** (7 / 6)) * (Nnu * q / fnu) ** 2
    Dcbnu = (fcb ** (0.7 / pcb) + (D1 / (1 + yfs)) ** 0.7) ** (pcb / 0.7) * D1 ** (
        1 - pcb
    )

    # Remove 1+zeq normalisation given in Eisenstein & Hu 1997
    D = Dcbnu / (1 + zeq)

    return D
