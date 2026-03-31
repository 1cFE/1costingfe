"""Layer 2b: 0D Tokamak Plasma Model.

Derives fusion power, density, confinement, and radial build from machine
parameters (R, a, B, q95, etc.) using standard tokamak scaling laws.

All core functions are pure and JAX-differentiable.
"""

from dataclasses import dataclass

import numpy as jnp

from costingfe.layers.physics import (
    ash_neutron_split,
    compute_p_rad,
    mfe_forward_power_balance,
    mfe_inverse_power_balance,
)
from costingfe.types import Fuel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_EV = 1.602176634e-19  # J per eV
MU_0 = 1.25663706127e-06  # Vacuum permeability [T·m/A]
E_FUS_DT = 17.58  # DT fusion energy [MeV]
MEV_TO_J = _EV * 1e6  # 1 MeV -> Joules
KEV_TO_J = _EV * 1e3  # 1 keV -> Joules


# ---------------------------------------------------------------------------
# PlasmaState
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlasmaState:
    """Complete 0D plasma state for a tokamak."""

    I_p: float  # Plasma current [MA]
    n_GW: float  # Greenwald density limit [10^20 m^-3]
    n_e: float  # Operating density [m^-3]
    T_e: float  # Electron temperature [keV]
    beta_N: float  # Normalized beta [%·m·T/MA]
    tau_E: float  # Energy confinement time [s]
    p_fus: float  # Fusion power [MW]
    p_alpha: float  # Alpha heating [MW]
    p_rad: float  # Radiation power [MW]
    V_plasma: float  # Plasma volume [m^3]
    fw_area: float  # First wall surface area [m^2]
    q95: float  # Safety factor
    f_GW: float  # Greenwald fraction
    wall_loading: float  # Neutron wall loading [MW/m^2]
    div_heat_flux: float  # Divertor heat flux estimate [MW/m^2]
    H_factor: float  # tau_E_actual / tau_E_scaling
    disruption_rate: float = 0.0  # Disruptions per FPY


# ---------------------------------------------------------------------------
# Bosch-Hale DT reactivity
# ---------------------------------------------------------------------------
# Coefficients from Bosch & Hale, Nuclear Fusion 32 (1992) 611
# Valid for 0.2 keV < T < 100 keV
_BH_C1 = 1.17302e-9
_BH_C2 = 1.51361e-2
_BH_C3 = 7.51886e-2
_BH_C4 = 4.60643e-3
_BH_C5 = 1.35000e-2
_BH_C6 = -1.06750e-4
_BH_C7 = 1.36600e-5

_BH_BG = 34.3827  # Gamow constant [keV^(1/2)]
_BH_MRC2 = 1124656.0  # Reduced mass * c^2 [keV]


def sigma_v_dt(T_keV):
    """Bosch-Hale DT reactivity <sigma*v> [m^3/s].

    Smooth analytic parameterization valid 1-100 keV.
    Uses JAX operations for differentiability.
    """
    theta = T_keV / (
        1.0
        - T_keV
        * (_BH_C2 + T_keV * (_BH_C4 + T_keV * _BH_C6))
        / (1.0 + T_keV * (_BH_C3 + T_keV * (_BH_C5 + T_keV * _BH_C7)))
    )
    xi = (_BH_BG**2 / (4.0 * theta)) ** (1.0 / 3.0)
    sigma_v = _BH_C1 * theta * jnp.sqrt(xi / (_BH_MRC2 * T_keV**3)) * jnp.exp(-3.0 * xi)
    return sigma_v * 1e-6  # cm^3/s -> m^3/s


# ---------------------------------------------------------------------------
# Core physics functions (all pure, JAX-differentiable)
# ---------------------------------------------------------------------------
def compute_plasma_current(a, kappa, B, R, q95):
    """Plasma current from MHD equilibrium [MA].

    I_p = 2*pi * a^2 * kappa * B / (mu_0 * R * q95) / 1e6
    """
    return 2.0 * jnp.pi * a**2 * kappa * B / (MU_0 * R * q95) / 1e6


def compute_greenwald_density(I_p_MA, a):
    """Greenwald density limit [10^20 m^-3].

    n_GW = I_p [MA] / (pi * a^2 [m])
    """
    return I_p_MA / (jnp.pi * a**2)


def compute_fusion_power(n_e, T_i, V_plasma):
    """DT fusion power [MW].

    P_fus = (1/4) * n_e^2 * <sigma*v>(T_i) * E_fus * V / 1e6
    Factor 1/4 = n_D*n_T/n_e^2 for 50/50 DT mix.

    Multiplication order avoids float32 overflow (n_e^2 ~ 1e40).
    """
    sv = sigma_v_dt(T_i)
    E_fus_J = E_FUS_DT * MEV_TO_J
    # n_e * sv keeps intermediates in safe range (~1e-2)
    rate = n_e * sv
    return 0.25 * rate * n_e * E_fus_J * V_plasma * 1e-6  # W -> MW


def compute_beta_N(n_e, T_i, B, I_p_MA, a):
    """Normalized beta [%·m·T/MA].

    beta_t = 2 * mu_0 * n_e * T_i [J] / B^2  (dimensionless)
    beta_N = beta_t * (a * B / I_p_MA)  with beta_t in %, i.e. * 100
    """
    T_J = T_i * KEV_TO_J
    beta_t = 2.0 * MU_0 * n_e * T_J / B**2  # dimensionless fraction
    beta_N = beta_t * 100.0 * a * B / I_p_MA  # %·m·T/MA
    return beta_N


def compute_tau_E_ipb98y2(I_p_MA, B, n_e19, P_heat_MW, R, a, kappa, M):
    """IPB98(y,2) H-mode energy confinement time scaling [s].

    tau_E = 0.0562 * I_p^0.93 * B^0.15 * n_e19^0.41 * P^-0.69
              * R^1.97 * (a/R)^0.58 * kappa^0.78 * M^0.19

    n_e19 in units of 10^19 m^-3, P_heat in MW, I_p in MA, M in AMU.
    """
    epsilon = a / R
    return (
        0.0562
        * I_p_MA**0.93
        * B**0.15
        * n_e19**0.41
        * P_heat_MW ** (-0.69)
        * R**1.97
        * epsilon**0.58
        * kappa**0.78
        * M**0.19
    )


def compute_wall_loading(p_neutron_MW, fw_area):
    """Neutron wall loading [MW/m^2]."""
    return p_neutron_MW / fw_area


def compute_div_heat_flux(p_transport_MW, R, a, kappa, lambda_q=0.002):
    """Divertor heat flux estimate [MW/m^2].

    Simplified SOL model:
    P_div ~ p_transport / (2*pi*R * 2*lambda_q * f_expansion)
    lambda_q: SOL power width at midplane [m] (~1-3 mm)
    f_expansion: flux expansion factor to divertor (~5-10x)
    """
    f_expansion = 5.0
    wetted_area = 2.0 * jnp.pi * R * 2.0 * lambda_q * f_expansion
    return p_transport_MW / wetted_area


# ---------------------------------------------------------------------------
# Geometry helpers (JAX-compatible)
# ---------------------------------------------------------------------------
def _plasma_volume(R, a, kappa):
    """Plasma volume of an elongated torus [m^3]."""
    return 2.0 * jnp.pi**2 * R * a**2 * kappa


def _first_wall_area(R, a, kappa):
    """Approximate first wall surface area [m^2]."""
    return 4.0 * jnp.pi**2 * R * a * kappa


# ---------------------------------------------------------------------------
# Forward mode
# ---------------------------------------------------------------------------
def tokamak_0d_forward(
    R,
    a,
    kappa,
    B,
    q95,
    f_GW,
    T_e,
    p_input,
    fuel=Fuel.DT,
    M_ion=2.5,
    Z_eff=1.5,
    lambda_q=0.002,
    *,
    dd_f_T: float,
    dd_f_He3: float,
    dhe3_dd_frac: float,
    dhe3_f_T: float,
    pb11_f_alpha_n: float,
    pb11_f_p_n: float,
):
    """Forward 0D tokamak model: machine params -> PlasmaState.

    Given geometry (R, a, kappa), field (B), safety factor (q95),
    Greenwald fraction (f_GW), temperature (T_e), and auxiliary heating
    (p_input), computes all plasma parameters self-consistently.

    Returns PlasmaState with all derived quantities.
    """
    # 1. Plasma current from MHD equilibrium
    I_p = compute_plasma_current(a, kappa, B, R, q95)

    # 2. Density from Greenwald fraction
    n_GW = compute_greenwald_density(I_p, a)
    n_e = f_GW * n_GW * 1e20  # Convert to m^-3

    # 3. Geometry
    V_plasma = _plasma_volume(R, a, kappa)
    fw_area = _first_wall_area(R, a, kappa)

    # 4. Fusion power
    # TODO: separate T_i from T_e for non-DT fuels (p-B11: T_i >> T_e)
    T_i = T_e  # DT assumption: T_i ≈ T_e
    p_fus = compute_fusion_power(n_e, T_i, V_plasma)

    # 5. Alpha power and neutron split
    p_alpha, p_neutron = ash_neutron_split(
        p_fus,
        fuel,
        dd_f_T=dd_f_T,
        dd_f_He3=dd_f_He3,
        dhe3_dd_frac=dhe3_dd_frac,
        dhe3_f_T=dhe3_f_T,
        pb11_f_alpha_n=pb11_f_alpha_n,
        pb11_f_p_n=pb11_f_p_n,
    )

    # 6. Radiation
    p_rad = compute_p_rad(n_e, T_e, Z_eff, V_plasma, B, R=R, a=a, kappa=kappa)
    p_rad = jnp.minimum(p_rad, p_alpha)

    # 7. Heating power for confinement scaling
    p_heat = p_alpha + p_input

    # 8. Confinement time (IPB98y2)
    n_e19 = n_e / 1e19
    tau_E_scaling = compute_tau_E_ipb98y2(I_p, B, n_e19, p_heat, R, a, kappa, M_ion)

    # 9. Actual confinement: W_th = 1.5 * n_e * T * V (electrons + ions)
    W_th_J = 3.0 * n_e * T_e * KEV_TO_J * V_plasma  # factor 3 = 1.5*(n_e+n_i), n_i~n_e
    W_th_MW = W_th_J * 1e-6  # J -> MJ
    tau_E_actual = W_th_MW / p_heat  # s (W in MJ, P in MW -> s)
    H_factor = tau_E_actual / tau_E_scaling

    # 10. Beta
    beta_N = compute_beta_N(n_e, T_i, B, I_p, a)

    # 11. Wall loading
    wall_loading = compute_wall_loading(p_neutron, fw_area)

    # 12. Divertor heat flux
    p_transport = p_alpha - p_rad
    div_heat_flux = compute_div_heat_flux(p_transport, R, a, kappa, lambda_q)

    # 13. Disruption rate
    disruption_rate = compute_disruption_rate(f_GW, beta_N, q95)

    return PlasmaState(
        I_p=I_p,
        n_GW=n_GW,
        n_e=n_e,
        T_e=T_e,
        beta_N=beta_N,
        tau_E=tau_E_actual,
        p_fus=p_fus,
        p_alpha=p_alpha,
        p_rad=p_rad,
        V_plasma=V_plasma,
        fw_area=fw_area,
        q95=q95,
        f_GW=f_GW,
        wall_loading=wall_loading,
        div_heat_flux=div_heat_flux,
        H_factor=H_factor,
        disruption_rate=disruption_rate,
    )


# ---------------------------------------------------------------------------
# Inverse mode: find T_e that produces target p_fus
# ---------------------------------------------------------------------------
def _find_T_for_pfus(target_pfus, n_e, V_plasma, T_lo=1.0, T_hi=100.0, n_iter=60):
    """Bisection to find T_i [keV] that yields target fusion power.

    Uses jnp.where for JAX differentiability (no Python control flow).
    Returns T_i; caller sets T_e = T_i (DT assumption).
    """

    def body(i, state):
        lo, hi = state
        mid = 0.5 * (lo + hi)
        p_mid = compute_fusion_power(n_e, mid, V_plasma)  # mid = T_i
        lo = jnp.where(p_mid < target_pfus, mid, lo)
        hi = jnp.where(p_mid >= target_pfus, mid, hi)
        return (lo, hi)

    state = (T_lo, T_hi)
    for i in range(n_iter):
        state = body(i, state)
    lo, hi = state
    return 0.5 * (lo + hi)


def tokamak_0d_inverse(
    p_net_target,
    R,
    a,
    kappa,
    B,
    q95,
    f_GW,
    fuel=Fuel.DT,
    M_ion=2.5,
    Z_eff=1.5,
    lambda_q=0.002,
    # Power balance params (passed through to mfe_inverse/forward)
    p_input=50.0,
    mn=1.1,
    eta_th=0.46,
    eta_p=0.5,
    eta_pin=0.5,
    eta_de=0.85,
    f_sub=0.03,
    f_dec=0.0,
    p_coils=2.0,
    p_cool=13.7,
    p_pump=1.0,
    p_trit=10.0,
    p_house=4.0,
    p_cryo=0.5,
    n_mod=1,
    *,
    dd_f_T: float,
    dd_f_He3: float,
    dhe3_dd_frac: float,
    dhe3_f_T: float,
    pb11_f_alpha_n: float,
    pb11_f_p_n: float,
    # Impurity / synchrotron params for power balance
    wall_material=None,
    T_edge: float = 0.05,
    tau_ratio: float = 3.0,
    fw_area: float = 0.0,
    R_w: float = 0.6,
):
    """Inverse 0D tokamak: p_net target -> PlasmaState + PowerTable.

    1. Use existing mfe_inverse_power_balance to get required p_fus
    2. Compute I_p, n_e from machine geometry
    3. Bisect on T_e to match target p_fus
    4. Return (PlasmaState, PowerTable)
    """
    p_net_per_mod = p_net_target / n_mod

    # Step 1: Required p_fus from energy balance
    I_p = compute_plasma_current(a, kappa, B, R, q95)
    n_GW = compute_greenwald_density(I_p, a)
    n_e = f_GW * n_GW * 1e20
    V_plasma = _plasma_volume(R, a, kappa)

    # Get required p_fus from the existing inverse power balance
    p_fus_required = mfe_inverse_power_balance(
        p_net_target=p_net_per_mod,
        fuel=fuel,
        p_input=p_input,
        mn=mn,
        eta_th=eta_th,
        eta_p=eta_p,
        eta_pin=eta_pin,
        eta_de=eta_de,
        f_sub=f_sub,
        f_dec=f_dec,
        p_coils=p_coils,
        p_cool=p_cool,
        p_pump=p_pump,
        p_trit=p_trit,
        p_house=p_house,
        p_cryo=p_cryo,
        n_e=n_e,
        T_e=15.0,
        Z_eff=Z_eff,
        plasma_volume=V_plasma,
        B=B,
        R_major=R,
        a_minor=a,
        kappa=kappa,
        R_w=R_w,
        wall_material=wall_material,
        T_edge=T_edge,
        tau_ratio=tau_ratio,
        fw_area=fw_area,
        dd_f_T=dd_f_T,
        dd_f_He3=dd_f_He3,
        dhe3_dd_frac=dhe3_dd_frac,
        dhe3_f_T=dhe3_f_T,
        pb11_f_alpha_n=pb11_f_alpha_n,
        pb11_f_p_n=pb11_f_p_n,
    )

    # Step 2: Find T_i that produces this p_fus (T_e = T_i for DT)
    # TODO: separate T_i from T_e for non-DT fuels (p-B11: T_i >> T_e)
    T_e = _find_T_for_pfus(p_fus_required, n_e, V_plasma)

    # Step 3: Build full plasma state at found T_e
    plasma_state = tokamak_0d_forward(
        R=R,
        a=a,
        kappa=kappa,
        B=B,
        q95=q95,
        f_GW=f_GW,
        T_e=T_e,
        p_input=p_input,
        fuel=fuel,
        M_ion=M_ion,
        Z_eff=Z_eff,
        lambda_q=lambda_q,
        dd_f_T=dd_f_T,
        dd_f_He3=dd_f_He3,
        dhe3_dd_frac=dhe3_dd_frac,
        dhe3_f_T=dhe3_f_T,
        pb11_f_alpha_n=pb11_f_alpha_n,
        pb11_f_p_n=pb11_f_p_n,
    )

    # Step 4: Build power table using actual p_fus
    pt = mfe_forward_power_balance(
        p_fus=plasma_state.p_fus,
        fuel=fuel,
        p_input=p_input,
        mn=mn,
        eta_th=eta_th,
        eta_p=eta_p,
        eta_pin=eta_pin,
        eta_de=eta_de,
        f_sub=f_sub,
        f_dec=f_dec,
        p_coils=p_coils,
        p_cool=p_cool,
        p_pump=p_pump,
        p_trit=p_trit,
        p_house=p_house,
        p_cryo=p_cryo,
        n_e=n_e,
        T_e=T_e,
        Z_eff=Z_eff,
        plasma_volume=V_plasma,
        B=B,
        R_major=R,
        a_minor=a,
        kappa=kappa,
        R_w=R_w,
        wall_material=wall_material,
        T_edge=T_edge,
        tau_ratio=tau_ratio,
        fw_area=fw_area,
        dd_f_T=dd_f_T,
        dd_f_He3=dd_f_He3,
        dhe3_dd_frac=dhe3_dd_frac,
        dhe3_f_T=dhe3_f_T,
        pb11_f_alpha_n=pb11_f_alpha_n,
        pb11_f_p_n=pb11_f_p_n,
    )

    return plasma_state, pt


# ---------------------------------------------------------------------------
# Physics limits (runs on concrete values, not JAX-traced)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlasmaLimits:
    """Configurable plasma physics limits."""

    beta_N_max: float = 3.5  # Troyon limit [%·m·T/MA]
    q95_min: float = 2.0  # Kink stability
    wall_loading_max: float = 5.0  # [MW/m^2]
    div_heat_flux_max: float = 10.0  # [MW/m^2]


def check_plasma_limits(state: PlasmaState, limits: PlasmaLimits = None):
    """Check plasma state against physics limits.

    Returns list of (severity, message) tuples.
    severity: "error" or "warning"
    """
    if limits is None:
        limits = PlasmaLimits()

    issues = []

    # Greenwald limit
    f_GW = float(state.f_GW)
    if f_GW > 1.0:
        issues.append(("error", f"Greenwald fraction f_GW = {f_GW:.2f} > 1.0"))

    # Troyon limit
    beta_N = float(state.beta_N)
    if beta_N > limits.beta_N_max:
        issues.append(
            (
                "error",
                f"Normalized beta beta_N = {beta_N:.2f} > {limits.beta_N_max} %·m·T/MA",
            )
        )

    # Kink stability
    q95 = float(state.q95)
    if q95 < limits.q95_min:
        issues.append(("error", f"Safety factor q95 = {q95:.2f} < {limits.q95_min}"))

    # Wall loading (design feedback)
    wl = float(state.wall_loading)
    if wl > limits.wall_loading_max:
        issues.append(
            (
                "warning",
                f"Neutron wall loading = {wl:.2f} MW/m^2"
                f" > {limits.wall_loading_max} MW/m^2",
            )
        )

    # Divertor heat flux (design feedback)
    dhf = float(state.div_heat_flux)
    if dhf > limits.div_heat_flux_max:
        issues.append(
            (
                "warning",
                f"Divertor heat flux = {dhf:.2f} MW/m^2"
                f" > {limits.div_heat_flux_max} MW/m^2",
            )
        )

    return issues


# ---------------------------------------------------------------------------
# Disruption penalty model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DisruptionModel:
    """Parameters for the disruption rate model.

    Converts proximity to stability limits (Greenwald, Troyon, kink)
    into a disruption frequency, then applies penalties to component
    lifetime and plant availability.
    """

    rate_base: float = 0.1  # Baseline disruptions/FPY far from limits
    steepness: float = 15.0  # Exponential steepness parameter
    damage_per_disruption: float = 0.02  # Fraction of component life per disruption
    downtime_per_disruption: float = 72.0  # Hours of downtime per disruption
    beta_N_max: float = 3.5  # Troyon limit
    q95_min: float = 2.0  # Kink limit


def compute_disruption_rate(f_GW, beta_N, q95, model=None):
    """Disruption frequency [disruptions/FPY] from stability margins.

    Each stability boundary contributes a partial rate that increases
    exponentially as the operating point approaches the limit.
    Channels are independent — any one can trigger a disruption.

    JAX-differentiable (uses jnp only).
    """
    if model is None:
        model = DisruptionModel()

    margin_GW = 1.0 - f_GW
    margin_beta = 1.0 - beta_N / model.beta_N_max
    margin_kink = 1.0 - model.q95_min / q95

    rate_GW = model.rate_base * jnp.exp(-model.steepness * margin_GW)
    rate_beta = model.rate_base * jnp.exp(-model.steepness * margin_beta)
    rate_kink = model.rate_base * jnp.exp(-model.steepness * margin_kink)

    return rate_GW + rate_beta + rate_kink


def apply_disruption_penalty(core_lifetime, availability, disruption_rate, model=None):
    """Apply disruption penalties to core lifetime and availability.

    Returns (effective_core_lifetime, effective_availability).
    JAX-differentiable.
    """
    if model is None:
        model = DisruptionModel()

    # Cumulative damage shortens component life
    effective_core_lifetime = core_lifetime / (
        1.0 + disruption_rate * model.damage_per_disruption * core_lifetime
    )

    # Downtime reduces availability
    disruption_downtime_fraction = (
        disruption_rate * model.downtime_per_disruption / 8760.0
    )
    effective_availability = availability * (1.0 - disruption_downtime_fraction)

    return effective_core_lifetime, effective_availability


# ---------------------------------------------------------------------------
# Radial build derivation
# ---------------------------------------------------------------------------
_RADIAL_BUILD_DEFAULTS = {
    Fuel.DT: {
        "blanket_t": 1.0,
        "ht_shield_t": 0.5,
        "structure_t": 0.20,
        "vessel_t": 0.20,
    },
    Fuel.DD: {
        "blanket_t": 0.5,
        "ht_shield_t": 0.3,
        "structure_t": 0.18,
        "vessel_t": 0.15,
    },
    Fuel.DHE3: {
        "blanket_t": 0.0,
        "ht_shield_t": 0.1,
        "structure_t": 0.15,
        "vessel_t": 0.10,
    },
    Fuel.PB11: {
        "blanket_t": 0.0,
        "ht_shield_t": 0.02,
        "structure_t": 0.15,
        "vessel_t": 0.10,
    },
}


def derive_radial_build(fuel, **overrides):
    """Physics-based radial build thickness defaults by fuel type.

    Returns dict of thickness values suitable for RadialBuild construction.
    User overrides take precedence.
    """
    defaults = dict(_RADIAL_BUILD_DEFAULTS[fuel])
    for k, v in overrides.items():
        if v is not None and k in defaults:
            defaults[k] = v
    return defaults
