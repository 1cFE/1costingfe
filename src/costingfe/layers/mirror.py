"""Layer 2c: 0D Axisymmetric Mirror Plasma Model.

Confinement, end losses, and plasma state for axisymmetric magnetic
mirrors, following the tokamak 0D pattern. Pure JAX; runtime math is
float32-safe, constants pre-folded in float64.
Physics per docs/superpowers/specs/2026-06-11-mirror-0d-sizing-design.md:
classical (Bing & Roberts 1961), Pastukhov electrostatic plugging
(Pastukhov 1974, Cohen et al. 1978), gas-dynamic (Mirnov & Ryutov 1979).
"""

import math
import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from costingfe.layers.physics import (
    OperatingPointInfeasible,
    ash_neutron_split,
    compute_p_rad,
    mfe_forward_power_balance,
    mfe_inverse_power_balance,
)
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
_WALL_LOADING_MAX = 5.0  # MW/m^2, matches tokamak PlasmaLimits.wall_loading_max

# Ion-ion collision time prefactor [s] for T in keV, n in 1e20 m^-3 units.
# Derived from NRL formula: tau_ii = 2.09e13 * A^0.5 * T_eV^1.5 / (n * lnL)
# (T_eV = T_keV * 1e3) -> multiply by (1e3)^1.5 to accept T in keV; the 1e-20
# density rescale folds in so the constant is benign and every intermediate
# stays near unity in float32 (XLA constant-gathering hazard, see reactivity.py):
_TAU_II_PREFACTOR_20 = 2.09e13 * (1.0e3) ** 1.5 * 1e-20  # 6.609e-3

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
    NRL formulary (Huba), Z = 1. Internally n is rescaled to 1e20 m^-3
    units so all intermediates stay near unity in float32.
    """
    n20 = n_i * 1e-20
    return _TAU_II_PREFACTOR_20 * T_i**1.5 * jnp.sqrt(A) / (n20 * _LN_LAMBDA)


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


# ---------------------------------------------------------------------------
# Inverse mode: bisect T_i to match required fusion power
# ---------------------------------------------------------------------------

# Fuel-keyed T_i brackets [keV] for the mirror inverse bisection.
# Ranges reflect mirror operating regimes and fit validity.
_T_BRACKET_MIRROR = {
    Fuel.DT: (2.0, 80.0),
    Fuel.DD: (5.0, 100.0),
    Fuel.DHE3: (20.0, 100.0),
    Fuel.PB11: (50.0, 300.0),
}

# Number of bisection iterations (same as tokamak _find_T_for_pfus)
_T_BISECT_ITERS = 60


def _find_T_i_for_pfus(
    target_pfus,
    n_e,
    V_plasma,
    fuel,
    fpd_kwargs,
    T_lo,
    T_hi,
    n_iter=_T_BISECT_ITERS,
):
    """Bisect on T_i [keV] to match the target fusion power.

    T_e is held fixed; T_i is the only free variable. fusion_power is
    monotonically increasing in T_i across all fuels at mirror-relevant
    temperatures, so bisection is well-posed.
    """

    def body(i, state):
        lo, hi = state
        mid = 0.5 * (lo + hi)
        p_mid, _ = fusion_power(fuel, n_e, mid, V_plasma, **fpd_kwargs)
        lo = jnp.where(p_mid < target_pfus, mid, lo)
        hi = jnp.where(p_mid >= target_pfus, mid, hi)
        return (lo, hi)

    lo, hi = jax.lax.fori_loop(0, n_iter, body, (T_lo, T_hi))
    return 0.5 * (lo + hi)


def mirror_0d_inverse(
    p_net_target,
    L,
    a,
    B_min,
    R_m,
    n_e,
    T_e,
    fuel=Fuel.DT,
    M_ion=2.5,
    Z_eff=1.5,
    R_w=0.4,
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
    dhe3_fuel_ratio: float,
    pb11_fuel_ratio: float,
    dhe3_dd_frac_pin: float | None,
    # Impurity / synchrotron params for power balance
    wall_material=None,
    T_edge: float = 0.05,
    tau_ratio: float = 3.0,
    R_w_pb: float | None = None,
    # Advanced fuel radiation proxy (when set, p_rad = f_rad_fus * p_fus)
    f_rad_fus: float | None = None,
    # Required: beta feasibility limit. No default; comes from YAML.
    beta_max: float,
    # Behavior flag: False is the escape hatch for exploration runs.
    enforce_plasma_limits: bool = True,
):
    """Inverse 0D mirror: p_net target -> (MirrorPlasmaState, PowerTable).

    Single-pass (f_dec is a fixed input; no outer self-consistency loop):
    1. Call mfe_inverse_power_balance with the YAML f_dec to get required p_fus.
    2. Bisect on T_i to match that p_fus (T_e is held fixed).
    3. Run mirror_0d_forward at the solved T_i to get the full plasma state.
    4. Apply beta feasibility gate (error-severity; wall loading is warning-only).
    5. Call mfe_forward_power_balance with the solved p_fus to build the PowerTable.
    6. Return (MirrorPlasmaState, PowerTable).
    """
    p_net_per_mod = p_net_target / n_mod

    # Step 1: Geometry (fixed inputs; L,a,B_min are not solved here).
    V_plasma = jnp.pi * a**2 * L
    fw_area = 2.0 * jnp.pi * a * L
    # Synchrotron geometry: R_eff = L / (2*pi) maps cylinder to equivalent torus.
    R_eff = L / (2.0 * jnp.pi)

    # Step 2: Required p_fus from MFE energy balance.
    # T_e used as the seed temperature for radiation at this stage.
    # Use dhe3_dd_frac_pin when pinned; otherwise use the YAML-sourced seed
    # (single-pass, so no refinement loop for D-He3).
    frac = dhe3_dd_frac if dhe3_dd_frac_pin is None else dhe3_dd_frac_pin
    R_w_inv = R_w_pb if R_w_pb is not None else R_w
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
        T_e=T_e,
        Z_eff=Z_eff,
        plasma_volume=V_plasma,
        B=B_min,
        R_major=R_eff,
        a_minor=a,
        kappa=1.0,
        R_w=R_w_inv,
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

    # Step 3: Bisect T_i to match the required fusion power.
    T_lo, T_hi = _T_BRACKET_MIRROR[fuel]
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
    T_i_solved = _find_T_i_for_pfus(
        p_fus_required, n_e, V_plasma, fuel, fpd_kwargs, T_lo, T_hi
    )

    # Step 4: Full plasma state at the solved T_i.
    plasma_state = mirror_0d_forward(
        L=L,
        a=a,
        B_min=B_min,
        R_m=R_m,
        T_i=T_i_solved,
        T_e=T_e,
        n_e=n_e,
        p_input=p_input,
        fuel=fuel,
        M_ion=M_ion,
        Z_eff=Z_eff,
        R_w=R_w,
        dd_f_T=dd_f_T,
        dd_f_He3=dd_f_He3,
        dhe3_dd_frac_pin=dhe3_dd_frac_pin,
        dhe3_f_T=dhe3_f_T,
        dhe3_f_He3=dhe3_f_He3,
        pb11_f_alpha_n=pb11_f_alpha_n,
        pb11_f_p_n=pb11_f_p_n,
        dhe3_fuel_ratio=dhe3_fuel_ratio,
        pb11_fuel_ratio=pb11_fuel_ratio,
        f_rad_fus=f_rad_fus,
    )

    # Step 5: Feasibility gate — tracer-skip guard matches the tokamak pattern.
    # Under JAX tracing (jit/grad), beta is an abstract tracer; skip the gate.
    if enforce_plasma_limits and not isinstance(plasma_state.beta, jax.core.Tracer):
        beta_val = float(plasma_state.beta)
        if beta_val > beta_max:
            raise OperatingPointInfeasible(
                f"net target {float(p_net_target):.1f} MW at this mirror geometry "
                f"implies beta = {beta_val:.4f} > beta_max = {beta_max:.4f}. "
                f"Pass enforce_plasma_limits=False to inspect the implied point."
            )
        # Wall loading: warning-severity only (matches tokamak pattern of not raising)
        wl = float(plasma_state.wall_loading)
        if wl > _WALL_LOADING_MAX:
            warnings.warn(
                f"Neutron wall loading = {wl:.2f} MW/m^2 > {_WALL_LOADING_MAX} MW/m^2",
                stacklevel=2,
            )

    # Step 6: Power table at the actual p_fus.
    # Single-pass: p_fus was solved against the seed dhe3_dd_frac, so unpinned
    # D-He3 p_net carries a small uncorrected residual (same caveat as tokamak).
    frac_eff = plasma_state.dhe3_dd_frac_eff
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
        B=B_min,
        R_major=R_eff,
        a_minor=a,
        kappa=1.0,
        R_w=R_w_inv,
        wall_material=wall_material,
        T_edge=T_edge,
        tau_ratio=tau_ratio,
        fw_area=fw_area,
        dd_f_T=dd_f_T,
        dd_f_He3=dd_f_He3,
        dhe3_dd_frac=frac_eff,
        dhe3_f_T=dhe3_f_T,
        dhe3_f_He3=dhe3_f_He3,
        pb11_f_alpha_n=pb11_f_alpha_n,
        pb11_f_p_n=pb11_f_p_n,
        f_rad_fus=f_rad_fus,
    )

    return plasma_state, pt
