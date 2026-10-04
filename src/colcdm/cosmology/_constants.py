from math import pi

H0 = 100  # km/s/(Mpc/h)
C_LIGHT = 299792.458  # km/s
T_CMB = 2.7255  # K, CMB temperature today

# SI constants used for explicit physical-density conversions.
G_NEWTON = 6.67430e-11  # m^3 kg^-1 s^-2
MPC_TO_M = 3.0856775814913673e22  # m / Mpc
M_SUN_TO_KG = 1.98847e30  # kg / M_sun

# Critical density today, 3 H0^2 / (8 pi G), in (M_sun/h) / (Mpc/h)^3: 2.775e11.
_H0_SI = H0 * 1e3 / MPC_TO_M  # 1/s, for h = 1
RHO_CRIT_0 = 3.0 * _H0_SI**2 / (8.0 * pi * G_NEWTON) * MPC_TO_M**3 / M_SUN_TO_KG
