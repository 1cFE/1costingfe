"""Layer 2c: 0D Axisymmetric Mirror Plasma Model.

Confinement, end losses, and plasma state for axisymmetric magnetic
mirrors, following the tokamak 0D pattern. Pure JAX; runtime math is
float32-safe, constants pre-folded in float64.
Physics per docs/superpowers/specs/2026-06-11-mirror-0d-sizing-design.md:
classical (Bing & Roberts 1961), Pastukhov electrostatic plugging
(Pastukhov 1974, Cohen et al. 1978), gas-dynamic (Mirnov & Ryutov 1979).
"""

import math
from dataclasses import dataclass

import jax.numpy as jnp

from costingfe.layers.physics import ash_neutron_split, compute_p_rad
from costingfe.layers.reactivity import (
    fusion_power,
    n_i_over_n_e,
)
from costingfe.types import Fuel

# ---------------------------------------------------------------------------
# Constants (CODATA 2018 values)
# ---------------------------------------------------------------------------
_EV = 1.602176634e-19  # J per eV (exact by 2019 SI redefinition)
_KEV_TO_J = _EV * 1e3  # 1 keV -> Joules
_M_P = 1.67262192369e-27  # Proton mass [kg]
_MU_0 = 1.25663706127e-06  # Vacuum permeability [T*m/A]

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
# MirrorPlasmaState
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MirrorPlasmaState:
    """Complete 0D plasma state for an axisymmetric magnetic mirror."""

    n_e: float  # Operating density [m^-3]
    T_i: float  # Ion temperature [keV]
    T_e: float  # Electron temperature [keV]
    beta: float  # Midplane beta [-]
    tau_p: float  # Particle confinement time [s]
    tau_E: float  # Energy confinement time [s]
    tau_classical: float  # Classical mirror confinement [s]
    tau_Pastukhov: float  # Pastukhov confinement [s]
    tau_GD: float  # Gas-dynamic confinement [s]
    phi: float  # Ambipolar potential e*phi [keV]
    p_fus: float  # Fusion power [MW]
    p_alpha: float  # Alpha (charged-particle) heating [MW]
    p_end: float  # End-loss power, axial channel [MW]
    p_radial: float  # Radial transport power [MW]
    p_rad: float  # Radiation power [MW]
    f_axial_derived: float  # Diagnostic: axial loss fraction (NOT used as f_dec)
    V_plasma: float  # Plasma volume [m^3]
    fw_area: float  # First-wall area [m^2]
    wall_loading: float  # Neutron wall loading [MW/m^2]
    R_m: float  # Mirror ratio [-]
    collisionality: float  # L / ion mean free path diagnostic [-]
    dhe3_dd_frac_eff: float = 0.0  # Effective D-D side-channel fraction (D-He3 only)


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


# ---------------------------------------------------------------------------
# Forward mode
# ---------------------------------------------------------------------------
def mirror_0d_forward(
    L,
    a,
    B_min,
    R_m,
    T_i,
    T_e,
    n_e,
    p_input,
    fuel=Fuel.DT,
    M_ion=2.5,
    Z_eff=1.5,
    R_w=0.4,
    *,
    dd_f_T: float,
    dd_f_He3: float,
    dhe3_dd_frac_pin: float | None,
    dhe3_f_T: float,
    dhe3_f_He3: float,
    pb11_f_alpha_n: float,
    pb11_f_p_n: float,
    dhe3_fuel_ratio: float,
    pb11_fuel_ratio: float,
    f_rad_fus: float | None = None,
):
    """Forward 0D mirror model: machine params -> MirrorPlasmaState.

    Given geometry (L, a, B_min, R_m), temperatures (T_i, T_e), density
    (n_e), and auxiliary heating (p_input), computes the complete plasma
    state including fusion power, confinement times, beta, end losses,
    and wall loading.

    p_input: accepted for signature parity with tokamak_0d_forward and the
    model.py dispatch; intentionally unused here (confinement does not scale
    with heating power in the mirror 0D model).
    R_w: wall reflectivity for synchrotron radiation (default 0.4, reduced
    from 0.6 for tokamaks to account for radiation escaping open ends).
    f_rad_fus: when set, p_rad = f_rad_fus * p_fus (advanced fuel proxy).
    Returns MirrorPlasmaState.
    """
    # 1. Geometry
    V_plasma = jnp.pi * a**2 * L
    fw_area = 2.0 * jnp.pi * a * L

    # 2. Quasineutrality: n_i/n_e (fuel-aware dilution)
    n_i_frac = n_i_over_n_e(fuel, dhe3_fuel_ratio, pb11_fuel_ratio)

    # 3. Fusion power (density enters via the 1e-10 barrier inside fusion_power)
    p_fus, dhe3_dd_frac_eff = fusion_power(
        fuel,
        n_e,
        T_i,
        V_plasma,
        dhe3_fuel_ratio=dhe3_fuel_ratio,
        pb11_fuel_ratio=pb11_fuel_ratio,
        dhe3_dd_frac_pin=dhe3_dd_frac_pin,
        dd_f_T=dd_f_T,
        dd_f_He3=dd_f_He3,
        dhe3_f_T=dhe3_f_T,
        dhe3_f_He3=dhe3_f_He3,
        pb11_f_alpha_n=pb11_f_alpha_n,
        pb11_f_p_n=pb11_f_p_n,
    )

    # 4. Ash/neutron split (uses effective fraction for D-He3 energy partition)
    p_alpha, p_neutron = ash_neutron_split(
        p_fus,
        fuel,
        dd_f_T=dd_f_T,
        dd_f_He3=dd_f_He3,
        dhe3_dd_frac=dhe3_dd_frac_eff,
        dhe3_f_T=dhe3_f_T,
        dhe3_f_He3=dhe3_f_He3,
        pb11_f_alpha_n=pb11_f_alpha_n,
        pb11_f_p_n=pb11_f_p_n,
    )

    # 5. Radiation: per-fuel resolution matching the tokamak path.
    # DT/DD: full compute_p_rad (f_rad_fus=None -> None path).
    # DHE3/PB11: f_rad_fus proxy when provided; caller passes cc.f_rad_fus(fuel).
    # Synchrotron geometry: R_eff = L / (2*pi) maps the cylinder to an
    # equivalent torus for the Albajar formula; kappa=1.0 for a cylinder.
    if f_rad_fus is not None:
        p_rad = f_rad_fus * p_fus
    else:
        R_eff = L / (2.0 * jnp.pi)
        p_rad = compute_p_rad(
            n_e,
            T_e,
            Z_eff,
            V_plasma,
            B_min,
            R=R_eff,
            a=a,
            kappa=1.0,
            R_w=R_w,
        )
    p_rad = jnp.minimum(p_rad, p_alpha)

    # 6. Ion-ion collision time; n_i = n_i_frac * n_e is the fuel-dilution mix.
    n_i = n_i_frac * n_e
    tau_ii = compute_tau_ii(n_i, T_i, M_ion)

    # 7. Ambipolar potential
    phi = compute_ambipolar_potential(T_e, M_ion)

    # 8. Confinement time chain
    # tau_classical is diagnostic only; confinement chain uses Pastukhov + gas-dynamic.
    tau_classical = compute_tau_classical(tau_ii, R_m)
    tau_Pastukhov = compute_tau_pastukhov(tau_ii, R_m, phi, T_i)
    tau_GD = compute_tau_gas_dynamic(R_m, L, T_i, M_ion)
    tau_radial = compute_tau_radial(tau_ii, a, T_i, M_ion, B_min)

    # Combined axial: 1/tau_axial = 1/tau_Pastukhov + 1/tau_GD
    inv_tau_axial = 1.0 / tau_Pastukhov + 1.0 / tau_GD

    # Total particle: 1/tau_p = 1/tau_axial + 1/tau_radial
    inv_tau_p = inv_tau_axial + 1.0 / tau_radial
    tau_p = 1.0 / inv_tau_p

    # 9. Energy confinement time
    # tau_E = tau_p * 1.5*(T_i+T_e) / (2*phi + T_i + T_e)
    tau_E = tau_p * (1.5 * (T_i + T_e)) / (2.0 * phi + T_i + T_e)

    # 10. Stored thermal energy [MW when divided by tau_E in seconds]
    W_th_MW = 1.5 * n_e * (T_e + n_i_frac * T_i) * _KEV_TO_J * V_plasma * 1e-6

    # 11. End-loss power split between axial and radial channels
    # Total loss: P_total = W_th / tau_E
    # Axial fraction: (1/tau_axial) / (1/tau_p) = inv_tau_axial / inv_tau_p
    # p_end (axial) = P_total * axial_fraction
    # p_radial = P_total - p_end
    P_total = W_th_MW / tau_E
    axial_frac = inv_tau_axial / inv_tau_p
    p_end = P_total * axial_frac
    p_radial = P_total - p_end

    # 12. Diagnostic: f_axial_derived = p_end / (p_end + p_radial)
    # By construction this equals the tau-channel weight (axial_frac), but
    # expressed as the power ratio to make the diagnostic intent explicit.
    f_axial_derived = p_end / (p_end + p_radial)

    # 13. Beta (dilution-aware, midplane field B_min)
    # beta = 2 * mu_0 * n_e * (T_e + n_i_frac * T_i) * KEV_TO_J / B_min^2
    beta = 2.0 * _MU_0 * n_e * (T_e + n_i_frac * T_i) * _KEV_TO_J / B_min**2

    # 14. Wall loading
    wall_loading = p_neutron / fw_area

    # 15. Collisionality: L / ion mean free path
    # Mean free path = v_thi * tau_ii
    v_thi = _V_THI_PREFACTOR * jnp.sqrt(T_i / M_ion)
    collisionality = L / (v_thi * tau_ii)

    return MirrorPlasmaState(
        n_e=n_e,
        T_i=T_i,
        T_e=T_e,
        beta=beta,
        tau_p=tau_p,
        tau_E=tau_E,
        tau_classical=tau_classical,
        tau_Pastukhov=tau_Pastukhov,
        tau_GD=tau_GD,
        phi=phi,
        p_fus=p_fus,
        p_alpha=p_alpha,
        p_end=p_end,
        p_radial=p_radial,
        p_rad=p_rad,
        f_axial_derived=f_axial_derived,
        V_plasma=V_plasma,
        fw_area=fw_area,
        wall_loading=wall_loading,
        R_m=R_m,
        collisionality=collisionality,
        dhe3_dd_frac_eff=dhe3_dd_frac_eff,
    )
