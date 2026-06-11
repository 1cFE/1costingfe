"""Layer 2c: 0D Axisymmetric Mirror Plasma Model.

Confinement, end losses, and plasma state for axisymmetric magnetic
mirrors, following the tokamak 0D pattern. Pure JAX; runtime math is
float32-safe, constants pre-folded in float64.
Physics per docs/superpowers/specs/2026-06-11-mirror-0d-sizing-design.md:
classical (Bing & Roberts 1961), Pastukhov electrostatic plugging
(Pastukhov 1974, Cohen et al. 1978), gas-dynamic (Mirnov & Ryutov 1979).
"""

import math

import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Constants (CODATA 2018 values)
# ---------------------------------------------------------------------------
_EV = 1.602176634e-19  # J per eV (exact by 2019 SI redefinition)
_KEV_TO_J = _EV * 1e3  # 1 keV -> Joules
_M_P = 1.67262192369e-27  # Proton mass [kg]

_LN_LAMBDA = 17.0  # Coulomb logarithm, fusion-relevant plasmas
_M_P_OVER_M_E = 1836.15267  # Proton-to-electron mass ratio

# Ion-ion collision time prefactor [s] for T in keV, n in m^-3.
# Derived from NRL formula: tau_ii = 2.09e13 * A^0.5 * T_eV^1.5 / (n * lnL)
# (T_eV = T_keV * 1e3) -> multiply by (1e3)^1.5 to accept T in keV:
_TAU_II_PREFACTOR = 2.09e13 * (1.0e3) ** 1.5  # 6.609e17 s * m^3 / (keV^1.5)

# Thermal ion speed prefactor [m/s per sqrt(keV/amu)].
# v_thi = sqrt(2 * T_keV * KEV_TO_J / (A * m_p)) = _V_THI_PREFACTOR * sqrt(T_keV / A)
_V_THI_PREFACTOR = math.sqrt(2.0 * _KEV_TO_J / _M_P)  # 4.3769e5

# Ion gyroradius prefactor [m per sqrt(amu * keV) / T].
# rho_i = sqrt(2 * A * m_p * T_keV * KEV_TO_J) / (e * B)
#       = _RHO_I_PREFACTOR * sqrt(A * T_keV) / B
_RHO_I_PREFACTOR = math.sqrt(2.0 * _M_P * _KEV_TO_J) / _EV  # 4.5694e-3


# ---------------------------------------------------------------------------
# Core confinement functions (all pure, JAX-differentiable)
# ---------------------------------------------------------------------------


def compute_tau_ii(n_i, T_i, A):
    """Ion-ion collision time [s]. tau_ii ~ T^1.5 sqrt(A) / n.

    n_i in m^-3, T_i in [keV], A is ion mass number.
    NRL formulary (Huba), Z = 1.
    """
    return _TAU_II_PREFACTOR * T_i**1.5 * jnp.sqrt(A) / (n_i * _LN_LAMBDA)


def compute_ambipolar_potential(T_e, A):
    """Ambipolar potential e*phi [keV] for a simple mirror.

    Boltzmann relation: e*phi = T_e * ln(sqrt(m_i / (2 pi m_e))).
    T_e in [keV].
    """
    return T_e * jnp.log(jnp.sqrt(A * _M_P_OVER_M_E / (2.0 * jnp.pi)))


def compute_tau_classical(tau_ii, R_m):
    """Classical mirror confinement time [s] (Bing & Roberts 1961).

    tau_classical = 2.6 * ln(R_m) * tau_ii
    """
    return 2.6 * jnp.log(R_m) * tau_ii


def compute_tau_pastukhov(tau_ii, R_m, phi_keV, T_i):
    """Pastukhov electrostatically plugged confinement time [s].

    Pastukhov 1974, Cohen et al. 1978. phi_keV = e*phi [keV],
    T_i [keV]. Note: tau_ii must be evaluated at the same (T_i, A).
    """
    x = phi_keV / T_i
    return (
        tau_ii
        * (jnp.sqrt(jnp.pi) / 2.0)
        * ((R_m + 1.0) / R_m)
        * jnp.log(2.0 * R_m + 2.0)
        * x
        * jnp.exp(x)
    )


def compute_tau_gas_dynamic(R_m, L, T_i, A):
    """Gas-dynamic confinement time [s] (Mirnov & Ryutov 1979).

    tau_GD = R_m * L / v_thi, where v_thi = sqrt(2 T_i / m_i).
    T_i in keV, L in m, A is ion mass number.
    """
    v_thi = _V_THI_PREFACTOR * jnp.sqrt(T_i / A)
    return R_m * L / v_thi


def compute_tau_radial(tau_ii, a, T_i, A, B_min):
    """Classical cross-field diffusion time [s].

    tau_radial = (a / rho_i)^2 * tau_ii, where rho_i is the midplane
    ion gyroradius at B = B_min. T_i [keV], a [m], B_min [T].
    Note: tau_ii must be evaluated at the same (T_i, A).
    """
    rho_i = _RHO_I_PREFACTOR * jnp.sqrt(A * T_i) / B_min
    return (a / rho_i) ** 2 * tau_ii
