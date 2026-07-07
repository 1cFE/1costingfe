"""Layer 2b: 0D Tokamak Plasma Model.

Derives fusion power, density, confinement, and radial build from machine
parameters (R, a, B, q95, etc.) using standard tokamak scaling laws.

All core functions are pure and JAX-differentiable.
"""

from dataclasses import dataclass

from costingfe._backend import Tracer
from costingfe._backend import fori_loop as jax_fori_loop
from costingfe._backend import xp as jnp
from costingfe.layers.physics import (
    OperatingPointInfeasible as OperatingPointInfeasible,  # noqa: F401 — re-exported for existing importers
)
from costingfe.layers.physics import (
    SizingInfeasible as SizingInfeasible,  # noqa: F401 — re-exported for existing importers
)
from costingfe.layers.physics import (
    ash_neutron_split,
    compute_p_rad,
    mfe_forward_power_balance,
    mfe_inverse_power_balance,
)
from costingfe.layers.reactivity import (
    fusion_power,
    n_i_over_n_e,
    sigv_dt,
)
from costingfe.types import Fuel, WallMaterial

# ---------------------------------------------------------------------------
# Constants (CODATA 2018 values)
# ---------------------------------------------------------------------------
_EV = 1.602176634e-19  # J per eV (exact by 2019 SI redefinition)
MU_0 = 1.25663706127e-06  # Vacuum permeability [T·m/A]
E_FUS_DT = 17.58  # DT fusion energy [MeV]
MEV_TO_J = _EV * 1e6  # 1 MeV -> Joules
KEV_TO_J = _EV * 1e3  # 1 keV -> Joules

# Sizing golden-section search tuning
_GSS_ITERS = 40  # Golden-section iterations to localize the optimum T_e
_BETA_PENALTY = 1.0e6  # Penalty slope for beta-limit violations [MW per %·m·T/MA]
_R0_BISECT_ITERS = 60  # Bisection iterations to locate the target-power radius


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
    dhe3_dd_frac_eff: float = 0.0  # effective D-D side-channel fraction (D-He3)


# ---------------------------------------------------------------------------
# Bosch-Hale DT reactivity
# ---------------------------------------------------------------------------
# Bosch-Hale DT reactivity now lives in costingfe.layers.reactivity;
# re-exported because tests and downstream code import it from here.
sigma_v_dt = sigv_dt


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


def compute_beta_N(n_e, T_e, T_i, n_i_frac, B, I_p_MA, a):
    """Normalized beta [%·m·T/MA] from electron + fuel-ion pressure.

    Toroidal beta is plasma pressure over magnetic pressure B^2/(2*mu_0):
    beta_t = 2 * mu_0 * p / B^2 with total pressure
    p = n_e*T_e + n_i*T_i = n_e * (T_e + n_i_frac * T_i).
    """
    p_J = (T_e + n_i_frac * T_i) * KEV_TO_J
    beta_t = 2.0 * MU_0 * n_e * p_J / B**2
    return beta_t * 100.0 * a * B / I_p_MA


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


def resistive_recirc_power(recirc_power_factor, B0, R0, a, kappa):
    """Continuous power drawn by resistive coils [MW]. Zero for superconductors
    (recirc_power_factor = 0). Scales with stored-field energy density (B0^2)
    times plasma volume."""
    return recirc_power_factor * B0**2 * _plasma_volume(R0, a, kappa)


def b0_from_radial_build(R0, a, b_max, blanket_t, ht_shield_t, structure_t, vessel_t):
    """On-axis toroidal field [T] from the peak-field ceiling and the inboard
    radial build.

    The toroidal field falls as 1/R from the inboard TF leg to the axis:
        B0 = B_max * R_coil_inner / R0
    where R_coil_inner = R0 - a - (blanket + ht_shield + structure + vessel) is
    the major radius of the inboard coil leg. Fixed-meter inboard layers penalize
    small machines on field.
    """
    inboard = blanket_t + ht_shield_t + structure_t + vessel_t
    r_coil_inner = R0 - a - inboard
    # Floor at a small positive field: when the fixed-meter inboard build exceeds
    # the available radius (very small R0), r_coil_inner goes negative, which is
    # geometrically infeasible. Returning a tiny positive B0 keeps the 0D physics
    # real-valued (no sqrt of a negative field) and makes such points read as
    # essentially zero net power, so the R0 bisection moves away from them.
    r_coil_inner = jnp.maximum(r_coil_inner, 1e-3)
    return b_max * r_coil_inner / R0


# ---------------------------------------------------------------------------
# Auxiliary heating from confinement requirement (sizing mode)
# ---------------------------------------------------------------------------
def aux_heating_from_confinement(
    H_factor,
    I_p,
    B0,
    n_e,
    T_e,
    V_plasma,
    p_alpha,
    R0,
    a,
    kappa,
    M_ion,
    *,
    T_i_over_T_e,
    n_i_frac,
):
    """Auxiliary heating power [MW] required to sustain the operating point at a
    given confinement quality H_factor.

    From tau_E = H_factor * tau_E_scaling (IPB98y2, P_heat^-0.69) and the model
    convention tau_E = W_th / P_heat, the heating power is closed-form:
        P_heat = (W_th / (H_factor * K))^(1/0.31),  K = IPB98 coeff at P_heat=1
    The auxiliary part is P_heat - p_alpha, floored at 0 (0 = ignited).
    """
    n_e19 = n_e / 1e19
    K = compute_tau_E_ipb98y2(I_p, B0, n_e19, 1.0, R0, a, kappa, M_ion)
    T_i = T_i_over_T_e * T_e
    W_th_MW = 1.5 * n_e * (T_e + n_i_frac * T_i) * KEV_TO_J * V_plasma * 1e-6
    P_heat = (W_th_MW / (H_factor * K)) ** (1.0 / 0.31)
    return jnp.maximum(0.0, P_heat - p_alpha)


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
    dhe3_dd_frac_pin: float | None,
    dhe3_f_T: float,
    dhe3_f_He3: float,
    pb11_f_alpha_n: float,
    pb11_f_p_n: float,
    T_i_over_T_e: float,
    dhe3_fuel_ratio: float,
    pb11_fuel_ratio: float,
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

    # 4. Fusion power (T_i from the hot-ion ratio; fuel-aware reactivity)
    T_i = T_i_over_T_e * T_e
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

    # 5. Alpha power and neutron split
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

    # 6. Radiation
    p_rad = compute_p_rad(n_e, T_e, Z_eff, V_plasma, B, R=R, a=a, kappa=kappa)
    p_rad = jnp.minimum(p_rad, p_alpha)

    # 7. Heating power for confinement scaling
    p_heat = p_alpha + p_input

    # 8. Confinement time (IPB98y2)
    n_e19 = n_e / 1e19
    tau_E_scaling = compute_tau_E_ipb98y2(I_p, B, n_e19, p_heat, R, a, kappa, M_ion)

    # 9. Actual confinement: W_th = 1.5 * (n_e T_e + n_i T_i) * V
    n_i_frac = n_i_over_n_e(fuel, dhe3_fuel_ratio, pb11_fuel_ratio)
    W_th_J = 1.5 * n_e * (T_e + n_i_frac * T_i) * KEV_TO_J * V_plasma
    W_th_MW = W_th_J * 1e-6  # J -> MJ
    tau_E_actual = W_th_MW / p_heat  # s (W in MJ, P in MW -> s)
    H_factor = tau_E_actual / tau_E_scaling

    # 10. Beta
    beta_N = compute_beta_N(n_e, T_e, T_i, n_i_frac, B, I_p, a)

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
        dhe3_dd_frac_eff=dhe3_dd_frac_eff,
    )


# ---------------------------------------------------------------------------
# Inverse mode: find T_e that produces target p_fus
# ---------------------------------------------------------------------------
def _find_T_for_pfus(
    target_pfus, n_e, V_plasma, fuel, T_i_over_T_e, fpd_kwargs, T_lo, T_hi, n_iter=60
):
    """Bisection for the T_e [keV] yielding the target fusion power.

    fpd_kwargs are exactly fusion_power's remaining keyword args (mix ratios,
    pin, burn fractions).
    """

    def body(i, state):
        lo, hi = state
        mid = 0.5 * (lo + hi)
        p_mid, _ = fusion_power(fuel, n_e, T_i_over_T_e * mid, V_plasma, **fpd_kwargs)
        lo = jnp.where(p_mid < target_pfus, mid, lo)
        hi = jnp.where(p_mid >= target_pfus, mid, hi)
        return (lo, hi)

    lo, hi = jax_fori_loop(0, n_iter, body, (T_lo, T_hi))
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
    dhe3_f_He3: float,
    pb11_f_alpha_n: float,
    pb11_f_p_n: float,
    T_i_over_T_e: float,
    dhe3_fuel_ratio: float,
    pb11_fuel_ratio: float,
    dhe3_dd_frac_pin: float | None,
    # Impurity / synchrotron params for power balance
    wall_material=None,
    T_edge: float = 0.05,
    tau_ratio: float = 3.0,
    fw_area: float = 0.0,
    R_w: float = 0.6,
    # Genuine optional override (keeps default), mirroring the power-balance
    # functions: when set, p_rad = f_rad_fus * p_fus replaces the computed
    # radiation model on BOTH sides of the solve, which also makes the
    # roundtrip self-consistent (the required-p_fus step otherwise evaluates
    # radiation at its fixed seed temperature).
    f_rad_fus: float | None = None,
    # Behavior flag: the solved operating point is the one the stated power
    # IMPLIES at this geometry; by default an implied point that violates an
    # error-severity plasma limit raises OperatingPointInfeasible rather than
    # being costed. False is the explicit escape hatch for exploration runs
    # that want to inspect the implied (unphysical) point anyway.
    enforce_plasma_limits: bool = True,
):
    """Inverse 0D tokamak: p_net target -> PlasmaState + PowerTable.

    1. Use existing mfe_inverse_power_balance to get required p_fus
    2. Compute I_p, n_e from machine geometry
    3. Bisect on T_e to match target p_fus
    4. Check the implied operating point against the plasma limits
    5. Return (PlasmaState, PowerTable)
    """
    p_net_per_mod = p_net_target / n_mod

    # Step 1: Required p_fus from energy balance
    I_p = compute_plasma_current(a, kappa, B, R, q95)
    n_GW = compute_greenwald_density(I_p, a)
    n_e = f_GW * n_GW * 1e20
    V_plasma = _plasma_volume(R, a, kappa)

    fpd_kwargs = dict(
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
    T_lo, T_hi = _T_BRACKET_DEFAULTS[fuel]

    # Two-pass fixed point: the required p_fus depends on the energy
    # partition, whose D-He3 side-channel fraction depends on the solved T.
    # One refinement suffices at costing fidelity. No residual check is
    # performed, and the final table's p_fus was solved against the pass-1
    # fraction, so p_net carries a small uncorrected residual for D-He3.
    frac = dhe3_dd_frac if dhe3_dd_frac_pin is None else dhe3_dd_frac_pin
    for _ in range(2):
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
            dhe3_dd_frac=frac,
            dhe3_f_T=dhe3_f_T,
            dhe3_f_He3=dhe3_f_He3,
            pb11_f_alpha_n=pb11_f_alpha_n,
            pb11_f_p_n=pb11_f_p_n,
            f_rad_fus=f_rad_fus,
        )
        T_e = _find_T_for_pfus(
            p_fus_required, n_e, V_plasma, fuel, T_i_over_T_e, fpd_kwargs, T_lo, T_hi
        )
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
            dhe3_dd_frac_pin=dhe3_dd_frac_pin,
            dhe3_f_T=dhe3_f_T,
            dhe3_f_He3=dhe3_f_He3,
            pb11_f_alpha_n=pb11_f_alpha_n,
            pb11_f_p_n=pb11_f_p_n,
            T_i_over_T_e=T_i_over_T_e,
            dhe3_fuel_ratio=dhe3_fuel_ratio,
            pb11_fuel_ratio=pb11_fuel_ratio,
        )
        # Only an unpinned D-He3 run can change frac between passes; everyone
        # else gets identical passes, so skip the redundant second one.
        if fuel != Fuel.DHE3 or dhe3_dd_frac_pin is not None:
            break
        frac = plasma_state.dhe3_dd_frac_eff

    # Step 4: The solved point is what the stated power implies at this
    # geometry. An implied point beyond an error-severity stability limit is
    # a verdict on the claim, not an operating point: refuse to cost it.
    # Skipped under JAX tracing (sensitivity), where values are abstract.
    if enforce_plasma_limits and not isinstance(plasma_state.beta_N, Tracer):
        errors = [
            msg for sev, msg in check_plasma_limits(plasma_state) if sev == "error"
        ]
        if errors:
            raise OperatingPointInfeasible(
                f"net target {float(p_net_target):.1f} MW at this geometry "
                f"implies an operating point beyond stability limits: "
                f"{'; '.join(errors)}. Pass enforce_plasma_limits=False to "
                "inspect the implied point anyway."
            )

    # Step 5: Build power table using actual p_fus
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
        dhe3_dd_frac=frac,
        dhe3_f_T=dhe3_f_T,
        dhe3_f_He3=dhe3_f_He3,
        pb11_f_alpha_n=pb11_f_alpha_n,
        pb11_f_p_n=pb11_f_p_n,
        f_rad_fus=f_rad_fus,
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

    rate_base: float = 1.0  # Baseline disruptions/FPY far from limits
    steepness: float = 12.0  # Exponential steepness parameter
    damage_per_disruption: float = 0.01  # Fraction of component life per disruption
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

    # Numerical ceiling: far past the limits the exponentials overflow
    # float32 to inf, which poisons the lifetime/replacement math into NaN.
    # 1e6 disruptions/FPY already zeroes availability and floors component
    # life; beyond it every point is equally "never runs".
    return jnp.minimum(rate_GW + rate_beta + rate_kink, 1.0e6)


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
    # Floor at zero: an operating point deep past the stability limits would
    # otherwise drive availability arbitrarily negative and cascade -inf/nan
    # through the cost stack; zero availability reads as an unambiguous
    # "this plant never runs" instead.
    effective_availability = jnp.maximum(
        0.0, availability * (1.0 - disruption_downtime_fraction)
    )

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


# Operating-temperature solve brackets [keV] by fuel. DT preserves the
# historical values (inverse bisection 1-100; sizing reads YAML T_min/T_max).
_T_BRACKET_DEFAULTS = {
    Fuel.DT: (1.0, 100.0),
    Fuel.DD: (5.0, 100.0),
    Fuel.DHE3: (20.0, 190.0),  # Bosch-Hale D-He3 fit validity tops out at 190
    Fuel.PB11: (50.0, 400.0),
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


# ---------------------------------------------------------------------------
# Sizing: net electric at a fixed R0 (inner operating point)
# ---------------------------------------------------------------------------
def _net_at_R0_T(R0, T_e, params, fuel):
    """Net electric power [MW] at fixed R0 and operating temperature T_e.

    Pins the operating point at the constraint boundary for this R0: density
    from Greenwald (via f_GW), field from the inboard radial build, and the
    given temperature. Wraps tokamak_0d_forward + mfe_forward_power_balance
    exactly as the forward branch of model._power_balance_0d does.

    Note: this function consumes whatever dhe3_f_He3 the caller places in params.
    _size_tokamak sets it to the effective bred-He3 value (_dhe3_f_He3_eff) before
    calling, so for D-He3 sizing the caller is responsible for providing the
    effective value (as the model's sizing path does).

    Returns (p_net [MW], beta_N).
    """
    A = params["aspect_ratio"]
    a = R0 / A
    kappa = params["elon"]
    b_max = params["b_max"]
    B0 = b0_from_radial_build(
        R0,
        a,
        b_max,
        params["blanket_t"],
        params["ht_shield_t"],
        params["structure_t"],
        params["vessel_t"],
    )

    base_frac_kw = dict(
        dd_f_T=params["dd_f_T"],
        dd_f_He3=params["dd_f_He3"],
        dhe3_f_T=params["dhe3_f_T"],
        dhe3_f_He3=params["dhe3_f_He3"],
        pb11_f_alpha_n=params["pb11_f_alpha_n"],
        pb11_f_p_n=params["pb11_f_p_n"],
    )

    M_ion = params["M_ion"]
    ps = tokamak_0d_forward(
        R=R0,
        a=a,
        kappa=kappa,
        B=B0,
        q95=params["q95"],
        f_GW=params["f_GW"],
        T_e=T_e,
        p_input=params["p_input"],
        fuel=fuel,
        M_ion=M_ion,
        Z_eff=params["Z_eff"],
        lambda_q=params["lambda_q"],
        dhe3_dd_frac_pin=params["dhe3_dd_frac_pin"],
        T_i_over_T_e=params["T_i_over_T_e"],
        dhe3_fuel_ratio=params["dhe3_fuel_ratio"],
        pb11_fuel_ratio=params["pb11_fuel_ratio"],
        **base_frac_kw,
    )

    # Solve the auxiliary heating from the confinement requirement so that
    # H_factor (not a fixed p_input) sets the recirculating power.
    p_aux = float(
        aux_heating_from_confinement(
            params["H_factor"],
            ps.I_p,
            B0,
            ps.n_e,
            ps.T_e,
            ps.V_plasma,
            ps.p_alpha,
            R0,
            a,
            kappa,
            M_ion,
            T_i_over_T_e=params["T_i_over_T_e"],
            n_i_frac=n_i_over_n_e(
                fuel, params["dhe3_fuel_ratio"], params["pb11_fuel_ratio"]
            ),
        )
    )

    p_coils = params["p_coils"] + resistive_recirc_power(
        params["recirc_power_factor"], B0, R0, a, kappa
    )

    wm_raw = params.get("wall_material")
    wall_mat = None
    if wm_raw is not None:
        wall_mat = WallMaterial(wm_raw) if isinstance(wm_raw, str) else wm_raw

    pb_frac = ps.dhe3_dd_frac_eff if fuel == Fuel.DHE3 else params["dhe3_dd_frac"]
    pb_frac_kw = dict(
        **base_frac_kw,
        dhe3_dd_frac=pb_frac,
        f_rad_fus=params.get("f_rad_fus"),
    )
    pt = mfe_forward_power_balance(
        p_fus=ps.p_fus,
        fuel=fuel,
        p_input=p_aux,
        mn=params["mn"],
        eta_th=params["eta_th"],
        eta_p=params["eta_p"],
        eta_pin=params["eta_pin"],
        eta_de=params["eta_de"],
        f_sub=params["f_sub"],
        f_dec=params["f_dec"],
        p_coils=p_coils,
        p_cool=params["p_cool"],
        p_pump=params["p_pump"],
        p_trit=params["p_trit"],
        p_house=params["p_house"],
        p_cryo=params["p_cryo"],
        n_e=ps.n_e,
        T_e=ps.T_e,
        Z_eff=params["Z_eff"],
        plasma_volume=ps.V_plasma,
        B=B0,
        R_major=R0,
        a_minor=a,
        kappa=kappa,
        R_w=params["R_w"],
        wall_material=wall_mat,
        seeded_impurities=params.get("seeded_impurities") or None,
        T_edge=params["T_edge"],
        tau_ratio=params["tau_ratio"],
        fw_area=ps.fw_area,
        **pb_frac_kw,
    )

    return float(pt.p_net), float(ps.beta_N)


def net_electric_at_R0(R0, params, fuel, return_state=False):
    """Net electric power [MW] at the constraint-boundary operating point for a
    fixed R0.

    Temperature maximizes net power within [T_min, T_max] subject to
    beta_N <= beta_N_max, via golden-section search. Runs in plain Python
    (not under JAX tracing), so jax scalars are wrapped with float().
    """
    T_lo, T_hi = params["T_min"], params["T_max"]
    beta_cap = params["beta_N_max"]
    invphi = (5**0.5 - 1) / 2  # 1/phi, the golden-section ratio
    invphi2 = (3 - 5**0.5) / 2  # 1/phi^2

    def feasible_net(T):
        pn, beta = _net_at_R0_T(R0, T, params, fuel)
        if beta > beta_cap:
            return -_BETA_PENALTY * (beta - beta_cap)
        return pn

    # Golden-section search for the MAXIMUM that recycles one probe per
    # iteration (about _GSS_ITERS evals rather than 2x that).
    a, b = T_lo, T_hi
    h = b - a
    c = a + invphi2 * h
    d = a + invphi * h
    fc = feasible_net(c)
    fd = feasible_net(d)
    for _ in range(_GSS_ITERS):
        if fc > fd:  # maximizing: max is in [a, d]
            b, d, fd = d, c, fc
            h = b - a
            c = a + invphi2 * h
            fc = feasible_net(c)
        else:  # max is in [c, b]
            a, c, fc = c, d, fd
            h = b - a
            d = a + invphi * h
            fd = feasible_net(d)
    T_star = 0.5 * (a + b)

    # The scalar return the outer solver bisects on must carry the beta
    # penalty so an all-infeasible point reads strongly negative rather than a
    # deceptive positive. return_state still exposes the TRUE state for
    # diagnostics.
    #
    # The beta-constrained optimum sits ON the cap, so T_star routinely lands a
    # sub-epsilon above beta_cap in float32. Penalizing that microscopic overshoot
    # would return a near-zero negative for a genuinely feasible point and corrupt
    # the outer R0 bisection (it would read the point as failing the target and
    # walk R0 too high). Apply a small tolerance: report true net power when beta
    # is within tol of the cap; penalize only a real violation beyond it.
    pn, beta = _net_at_R0_T(R0, T_star, params, fuel)
    if return_state:
        return pn, T_star, beta
    beta_tol = 1e-4 * beta_cap
    if beta > beta_cap + beta_tol:
        return -_BETA_PENALTY * (beta - beta_cap)
    return pn


def tokamak_max_net_electric(params, fuel):
    """Net electric [MW] a single tokamak delivers at R0_max (the unit ceiling).

    Used by the size-from-power n_mod fallback: if the plant target exceeds this,
    one device cannot reach it and the plant needs multiple identical units.
    """
    return net_electric_at_R0(params["R0_max"], params, fuel)


# ---------------------------------------------------------------------------
# Outer R0 bisection solver
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SizingResult:
    """Solved tokamak geometry and operating point from a power target."""

    R0: float
    a: float
    B0: float
    T_e: float


def tokamak_size_from_power(params, fuel):
    """Solve major radius R0 so net electric power equals the target.

    Net power is monotonic increasing in R0 at the boundary operating point, so
    bisection is well posed. Raises SizingInfeasible if the target exceeds what
    R0_max can deliver.
    """
    target = params["net_electric_mw"]
    lo, hi = params["R0_min"], params["R0_max"]

    pn_hi = net_electric_at_R0(hi, params, fuel)
    if pn_hi < target:
        raise SizingInfeasible(
            f"net power at R0_max={hi} m is {pn_hi:.1f} MW < target {target} MW; "
            "machine cannot reach the power with these physics inputs"
        )

    for _ in range(_R0_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        if net_electric_at_R0(mid, params, fuel) < target:
            lo = mid
        else:
            hi = mid
    R0 = 0.5 * (lo + hi)
    a = R0 / params["aspect_ratio"]
    _, T_star, _ = net_electric_at_R0(R0, params, fuel, return_state=True)
    B0 = b0_from_radial_build(
        R0,
        a,
        params["b_max"],
        params["blanket_t"],
        params["ht_shield_t"],
        params["structure_t"],
        params["vessel_t"],
    )
    return SizingResult(R0=R0, a=a, B0=B0, T_e=T_star)
