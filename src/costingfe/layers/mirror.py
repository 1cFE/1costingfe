"""Layer 2c: 0D Axisymmetric Mirror Plasma Model.

Confinement, end losses, and plasma state for axisymmetric magnetic
mirrors, following the tokamak 0D pattern. Backend-agnostic; runtime math is
float32-safe, constants pre-folded in float64.
Physics per docs/superpowers/specs/2026-06-11-mirror-0d-sizing-design.md:
classical (Bing & Roberts 1961), Pastukhov electrostatic plugging
(Pastukhov 1974, Cohen et al. 1978), gas-dynamic (Mirnov & Ryutov 1979).
"""

import dataclasses
import math
import warnings
from dataclasses import dataclass

from costingfe._backend import Tracer, fori_loop
from costingfe._backend import xp as jnp
from costingfe.layers.physics import (
    OperatingPointInfeasible,
    SizingInfeasible,  # noqa: F401 — re-exported so tests can import from mirror
    ash_neutron_split,
    compute_p_rad,
    event_energies,
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
_M_E_G = 9.1093837015e-28  # electron mass [g]
_AMU_G = 1.66053906660e-24  # atomic mass unit [g]

# Ion-ion collision time prefactor [s] for T in keV, n in 1e20 m^-3 units.
# Derived from NRL formula: tau_ii = 2.09e13 * A^0.5 * T_eV^1.5 / (n * lnL)
# (T_eV = T_keV * 1e3) -> multiply by (1e3)^1.5 to accept T in keV; the 1e-20
# density rescale folds in so the constant is benign and every intermediate
# stays near unity in float32 (XLA constant-gathering hazard, see reactivity.py):
_TAU_II_PREFACTOR_20 = 2.09e13 * (1.0e3) ** 1.5 * 1e-20  # 6.609e-3

# Ion-electron energy equilibration prefactor [s^-1 per (n_i_20 * Z^2 * sqrt(A) /
# (m_e_g * T_i_keV + m_i_g * T_e_keV)^1.5)].  Absorbs: the 1.8e-19 NRL constant,
# sqrt(m_e_g * m_amu_g), lnL, the n unit conversion (n_i_20 -> cm^-3: ×1e14),
# and the T unit factor (T_keV -> T_eV: (1e3)^1.5 factored out of denominator).
# Pre-folded in float64 so JAX float32 only sees O(1) intermediates; avoids the
# product m_e_g * m_i_g ~ 1e-51 which underflows float32 (~1.18e-38 minimum).
_K_IE_NU_PREFACTOR = (
    1.8e-19
    * math.sqrt(_M_E_G * _AMU_G)  # g -- computed in Python float64
    * _LN_LAMBDA
    * 1e14  # n_i_20 [1e20 m^-3] -> n_i_cm3 [cm^-3]
    / 1000.0**1.5  # (T_keV -> T_eV)^1.5 factored out of denominator
)  # ~3.76e-34; representable in float32

# Thermal ion speed prefactor [m/s per sqrt(keV/amu)].
# v_thi = sqrt(2 * T_keV * KEV_TO_J / (A * m_p)) = _V_THI_PREFACTOR * sqrt(T_keV / A)
_V_THI_PREFACTOR = math.sqrt(2.0 * _KEV_TO_J / _M_P)  # 4.3769e5

# Ion gyroradius prefactor [m per sqrt(amu * keV) / T].
# rho_i = sqrt(2 * A * m_p * T_keV * KEV_TO_J) / (e * B)
#       = _RHO_I_PREFACTOR * sqrt(A * T_keV) / B
_RHO_I_PREFACTOR = math.sqrt(2.0 * _M_P * _KEV_TO_J) / _EV  # 4.5694e-3

# Confinement-regime gate width [decades of collisionality].
# Logistic smoothing width of the gas-dynamic-vs-Pastukhov bridge, centered on
# the sourced boundary collisionality = 1/R_m (Rognlien and Cutler 1980: mean
# free path reduced by the mirror ratio equals the system length). The boundary
# is sourced; this width is a constructed smoothing parameter chosen for a smooth
# differentiable crossover. The width is narrow enough that the suppressed
# gas-dynamic rate falls well below the Pastukhov rate at the collisionless
# anchors, which matters because tau_GD can be up to 1e4 shorter than
# tau_Pastukhov so even a small residual gate leaks meaningfully. See
# docs/account_justification/mirror_confinement_regimes.md.
_REGIME_GATE_WIDTH_DECADES = 0.13
_LN10 = math.log(10.0)  # natural log of 10, for log10 in JAX


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
    # Tandem plug confining potential e*phi [keV]
    # (feeds tau; see compute_plug_potential)
    phi: float
    p_fus: float  # Fusion power [MW]
    p_alpha: float  # Alpha (charged-particle) heating [MW]
    p_end: float  # End-loss power, axial channel [MW]
    p_radial: float  # Radial transport power [MW]
    p_rad: float  # Radiation power [MW]
    f_axial_derived: float  # Diagnostic: axial loss fraction (NOT used as f_dec)
    V_plasma: float  # Plasma volume [m^3]
    fw_area: float  # First-wall area [m^2]
    wall_loading: float  # Neutron wall loading [MW/m^2]
    q_surface: float  # Surface heat-flux (photons + radial transport) on FW [MW/m^2]
    R_m: float  # Mirror ratio [-]
    collisionality: float  # L / ion mean free path diagnostic [-]
    # Pastukhov-Maxwellian validity flag (bool-as-float, 1.0 valid / 0.0 invalid).
    # 1.0 when collisionality >= collisionality_min (the bare Pastukhov-Maxwellian
    # core assumption holds); 0.0 when deeply collisionless, where the formula
    # over-credits confinement. Informational: a tandem legitimately runs
    # collisionless and plugged, so this flags where the assumption is stretched,
    # not a hard error. See docs/account_justification/mirror_confinement_regimes.md.
    pastukhov_valid: float
    # DCLC microstability diagnostic: number of ion gyroradii across the plasma
    # radius, a / rho_i (Post 1987 loss-cone criterion). The drift-cyclotron-loss-
    # cone mode grows more readily as this grows; a warm-plasma stream stabilises
    # DCLC when its fraction exceeds about rho_i/a (~ 1/dclc_parameter). Diagnostic
    # only in Task 3 (not a constraint). See mirror_confinement_regimes.md.
    dclc_parameter: float
    dhe3_dd_frac_eff: float = 0.0  # Effective D-D side-channel fraction (D-He3 only)
    # Audit-mode sustainment-consistency diagnostic: stated p_input divided by
    # the confinement-required auxiliary power (mirror analog of the tokamak's
    # implied H_factor). 1.0 means the stated heating exactly sustains the
    # operating point; >1 over-drives it, <1 under-sustains. Populated only on
    # the inverse/audit path (where a stated p_input exists); 0.0 otherwise.
    sustainment_ratio: float = 0.0


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


def compute_K_ie(n_e, n_i, T_i, T_e, Z_i, A, volume):
    """Ion-electron collisional energy transfer power to electrons [MW].

    NRL formulary energy-equilibration (Huba). Positive when T_i > T_e.
    n_e, n_i [m^-3]; T_i, T_e [keV]; Z_i ion charge; A ion mass number;
    volume [m^3]. Float32-safe: _K_IE_NU_PREFACTOR absorbs mass constants and
    unit conversions; the denominator uses T in keV so no intermediate
    falls below float32's normalized minimum (~1.18e-38). The full
    (m_e T_i + m_i T_e)^1.5 denominator is kept (no m_e << m_i
    approximation) so the sign and small-difference limit are exact.
    """
    n_i_20 = n_i * 1e-20  # rescale to 1e20 m^-3 (value near unity)
    # Energy exchange rate [s^-1]: T in keV, masses in grams (see _K_IE_NU_PREFACTOR).
    nu_eps = (
        _K_IE_NU_PREFACTOR
        * jnp.sqrt(A)
        * Z_i**2
        * n_i_20
        / (_M_E_G * T_i + A * _AMU_G * T_e) ** 1.5
    )
    # Power density to electrons [W/m^3] = 1.5 n_e nu_eps (T_i - T_e)keV * keV->J
    p_density_W = 1.5 * n_e * nu_eps * (T_i - T_e) * _KEV_TO_J
    return p_density_W * volume * 1e-6  # -> MW


def compute_ambipolar_potential(T_e, A):
    """Ambipolar potential e*phi [keV] for a SIMPLE mirror (diagnostic only).

    Boltzmann relation: e*phi = T_e * ln(sqrt(m_i / (2 pi m_e))). T_e in [keV].

    This is the unbounded simple-mirror value (about 3-9 T_e, growing with T_e
    without limit). It is retained for reference and used by the regime-bridge
    unit tests; not computed in the production forward path. It is
    NOT used in the tandem confinement chain: the cost model commits to a
    tandem (n_plug_coils = 4, Hammir class), so the operative confining potential
    is the bounded tandem value from compute_plug_potential. See
    docs/account_justification/mirror_confinement_regimes.md.
    """
    return T_e * jnp.log(jnp.sqrt(A * _M_P_OVER_M_E / (2.0 * jnp.pi)))


def compute_plug_potential(T_e_plug, plug_density_ratio):
    """Tandem plug confining potential e*phi [keV] (central-cell confinement).

    In a tandem mirror the central-cell ions are confined by the electrostatic
    potential drop to the end plugs (Fowler & Logan 1977; Frank et al. 2024,
    arXiv:2411.06644 eq. 18), set by the plug HOT-ELECTRON temperature and the
    plug-to-central density ratio:

        e*phi = T_e_plug * ln(n_p / n_c).

    The potential is built by the HOT-ELECTRON PLUG, not the central cell. Real
    advanced-fuel tandems run a separate hot-electron plug (ECH-heated, possibly a
    different ion species) DISTINCT from a coolable central cell: the plug holds
    the confining potential while the central cell runs the working fuel and keeps
    its electrons cool to limit bremsstrahlung. This function therefore takes the
    PLUG electron temperature T_e_plug (hot, set by plug ECH independent of the
    central fuel), NOT the central-cell electron temperature. The central-cell T_e
    sets central-cell radiation only; it does not enter the plug potential.

    This is the REAL Fowler-Logan form: e*phi is set by the FIXED plug hardware
    (the end-plug coils and plug heating the cost model commits to), not by the
    central-cell ion temperature. It is therefore independent of T_i, so during a
    T_i scan at fixed plug electron temperature T_e_plug the ratio e*phi / T_i
    = T_e_plug * ln(n_p/n_c) / T_i FALLS as T_i rises and the Pastukhov
    enhancement exp(e*phi/T_i) weakens at high T_i: heating the central cell costs
    confinement rather than buying it for free. This is what stops the free ride
    to ignition and settles the optimum at a cooler, genuinely DRIVEN point.

    Calibrated to the Realta Hammir Q > 5 design point: at T_e_plug = 125 keV and
    n_p/n_c = 1.818 (= 1/0.55, the published n_c/n_p = 0.55), e*phi = 125 *
    ln(1.818) = 74.7 keV, reproducing the published central-cell tau_c ~ 5 s and
    Q ~ 5.2.

    T_e_plug in [keV], plug_density_ratio = n_p/n_c dimensionless (> 1). Pure JAX,
    differentiable. See docs/account_justification/mirror_confinement_regimes.md.
    """
    return T_e_plug * jnp.log(plug_density_ratio)


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


def compute_tau_axial(tau_ii, R_m, L, T_i, A, phi_keV, n_i):
    """Axial confinement time [s], collisionality-gated regime bridge.

    Combines the gas-dynamic (collisional) and Pastukhov (collisionless) axial
    loss channels so the gas-dynamic channel governs only when the plasma is
    collisional and Pastukhov governs when it is collisionless. The transition
    follows Rognlien and Cutler 1980 (Nucl. Fusion 20, 1003): the Pastukhov
    formula breaks down when the ion mean free path reduced by the mirror ratio
    is of order the system length, i.e. collisionality (L/mfp) of order 1/R_m.

    The gas-dynamic loss RATE is gated by a logistic function of collisionality
    centered on collisionality = 1/R_m, then combined with the Pastukhov rate as
    a loss-rate (harmonic) sum:

        inv_tau_axial = 1/tau_Pastukhov + g(collisionality) * (1/tau_GD)

    so g -> 1 (gas-dynamic present, dominant) when collisional and g -> 0
    (gas-dynamic suppressed) when collisionless. Smooth and differentiable;
    the logistic argument is clamped to [-30, 30] so the gate carries a finite
    gradient at every collisionality (no NaN from a saturated logistic under
    jax.grad). float32-safe (constants folded float64).

    See docs/account_justification/mirror_confinement_regimes.md.

    tau_ii [s], R_m [-], L [m], T_i [keV], A ion mass number, phi_keV = e*phi
    [keV], n_i [m^-3] (accepted for signature parity / callers; collisionality
    is computed from L, v_thi(T_i, A), and tau_ii).
    """
    tau_Pastukhov = compute_tau_pastukhov(tau_ii, R_m, phi_keV, T_i)
    tau_GD = compute_tau_gas_dynamic(R_m, L, T_i, A)

    # Collisionality L/mfp, identical to the mirror_0d_forward diagnostic.
    v_thi = _V_THI_PREFACTOR * jnp.sqrt(T_i / A)
    collisionality = L / (v_thi * tau_ii)

    # Logistic gate in log10(collisionality), centered on the sourced boundary
    # collisionality_crit = 1/R_m. log10(x) = ln(x)/ln(10).
    log10_coll = jnp.log(collisionality) / _LN10
    log10_crit = jnp.log(1.0 / R_m) / _LN10
    # Clamp the logistic argument before exp so the saturated tails carry a finite
    # reverse-mode gradient (an unclamped saturated logistic yields inf*0 = NaN in
    # jax.grad). At physical collisionalities |arg| is well under 30, so the clamp
    # leaves the gate value unchanged; it only guards the extreme-collisionless and
    # extreme-collisional tails traversed by golden-section sizing and grad sweeps.
    arg = jnp.clip((log10_coll - log10_crit) / _REGIME_GATE_WIDTH_DECADES, -30.0, 30.0)
    gate = 1.0 / (1.0 + jnp.exp(-arg))

    inv_tau_axial = 1.0 / tau_Pastukhov + gate / tau_GD
    return 1.0 / inv_tau_axial


def compute_tau_radial(tau_ii, a, T_i, A, B_min):
    """Classical cross-field diffusion time [s].

    tau_radial = (a / rho_i)^2 * tau_ii, where rho_i is the midplane
    ion gyroradius at B = B_min. T_i [keV], a [m], B_min [T].
    Note: tau_ii must be evaluated at the same (T_i, A).
    """
    rho_i = _RHO_I_PREFACTOR * jnp.sqrt(A * T_i) / B_min
    return (a / rho_i) ** 2 * tau_ii


def compute_dclc_parameter(a, T_i, A, B_min):
    """DCLC microstability diagnostic: ion gyroradii across the plasma [-].

    Returns a / rho_i, the number of midplane ion gyroradii spanning the plasma
    radius, where rho_i is the gyroradius at B = B_min (same kernel as
    compute_tau_radial). The drift-cyclotron-loss-cone (DCLC) mode is loss-cone
    driven and grows more readily as the plasma spans more gyroradii (Post 1987,
    "The magnetic mirror approach to fusion"). A warm-plasma stream filling the
    loss cone stabilises DCLC when its fraction exceeds about rho_i/a, i.e.
    roughly 1/(a/rho_i); both GDT and WHAM rely on warm-plasma DCLC stabilisation.
    Diagnostic only (Task 3), not a constraint.

    a [m], T_i [keV], A ion mass number, B_min [T]. Pure JAX, differentiable.
    See docs/account_justification/mirror_confinement_regimes.md.
    """
    rho_i = _RHO_I_PREFACTOR * jnp.sqrt(A * T_i) / B_min
    return a / rho_i


def _confinement_and_losses(T_e, *, n_e, T_i, n_i_frac, M_ion, R_m, L, a, B_min, phi):
    """Confinement times and the axial/radial loss split at a given central-cell T_e.

    Pure function of T_e and the fixed operating point, so the electron-balance
    solver can evaluate p_end(T_e). Extracted verbatim from mirror_0d_forward; the
    physics is unchanged. tau_classical/tau_Pastukhov/tau_GD are diagnostic only.
    """
    n_i = n_i_frac * n_e
    tau_ii = compute_tau_ii(n_i, T_i, M_ion)
    tau_classical = compute_tau_classical(tau_ii, R_m)
    tau_Pastukhov = compute_tau_pastukhov(tau_ii, R_m, phi, T_i)
    tau_GD = compute_tau_gas_dynamic(R_m, L, T_i, M_ion)
    tau_radial = compute_tau_radial(tau_ii, a, T_i, M_ion, B_min)
    tau_axial = compute_tau_axial(tau_ii, R_m, L, T_i, M_ion, phi, n_i)
    inv_tau_axial = 1.0 / tau_axial
    inv_tau_p = inv_tau_axial + 1.0 / tau_radial
    tau_p = 1.0 / inv_tau_p
    tau_E = tau_p * (1.5 * (T_i + T_e)) / (2.0 * phi + T_i + T_e)
    V_plasma = jnp.pi * a**2 * L
    W_th_MW = 1.5 * n_e * (T_e + n_i_frac * T_i) * _KEV_TO_J * V_plasma * 1e-6
    P_total = W_th_MW / tau_E
    axial_frac = inv_tau_axial / inv_tau_p
    p_end = P_total * axial_frac
    p_radial = P_total - p_end
    v_thi = _V_THI_PREFACTOR * jnp.sqrt(T_i / M_ion)
    collisionality = L / (v_thi * tau_ii)
    return {
        "tau_E": tau_E,
        "W_th_MW": W_th_MW,
        "p_end": p_end,
        "p_radial": p_radial,
        "axial_frac": axial_frac,
        "tau_p": tau_p,
        "collisionality": collisionality,
        "tau_classical": tau_classical,
        "tau_Pastukhov": tau_Pastukhov,
        "tau_GD": tau_GD,
    }


def mirror_aux_heating(state, p_aux_floor, f_alpha_heat):
    """Auxiliary heating power [MW] required to sustain a mirror operating point.

    Mirror analog of the tokamak's aux_heating_from_confinement. The steady-state
    plasma energy balance credits only the fraction of the fusion-alpha power that
    deposits before scattering into the loss cone, f_alpha_heat * P_alpha, so

        P_alpha_heat + P_aux = P_end + P_radial + P_rad,
        P_alpha_heat = f_alpha_heat * P_alpha,
        P_aux = max(P_aux_floor, P_end + P_radial + P_rad - f_alpha_heat * P_alpha)

    where P_end, P_radial, P_rad, P_alpha are the corrected forward powers on the
    MirrorPlasmaState and P_aux_floor is a small control/startup floor so an
    ignited point still pays some control power. f_alpha_heat (about 0.80,
    Santarius & Callen 1983) is the alpha loss-cone heating fraction: a mirror
    loses about 50 percent of fusion alphas by count but under 25 percent by
    energy out the loss cone, so about 75-85 percent of the alpha power deposits.

    Feeding this P_aux as p_input to mfe_forward_power_balance makes its
    p_transport = p_ash + P_aux - p_rad collapse to
    P_end + P_radial + (1 - f_alpha_heat) * P_alpha (the real transport loss plus
    the directed loss-cone alpha exhaust, since p_ash ~ p_alpha), so the lost
    alpha fraction is conserved in the axial end-loss / DEC channel and the
    existing recirculating (P_aux/eta_pin) and DEC (f_dec*eta_de*p_transport)
    terms net correctly. Pure JAX, differentiable. See
    docs/account_justification/mirror_confinement_regimes.md.
    """
    return jnp.maximum(
        p_aux_floor,
        state.p_end + state.p_radial + state.p_rad - f_alpha_heat * state.p_alpha,
    )


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
    vacuum_t: float,
    plug_density_ratio: float,
    collisionality_min: float,
    T_e_plug: float,
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
    vacuum_t: plasma-to-first-wall vacuum gap [m]; first-wall area is
    2 pi (a + vacuum_t) L (matches the geometry layer's firstwall_area).
    f_rad_fus: when set, p_rad = f_rad_fus * p_fus (advanced fuel proxy).
    T_e_plug: HOT plug-electron temperature [keV] that sets the Fowler-Logan
    confining potential e*phi = T_e_plug*ln(n_p/n_c). Decoupled from the
    central-cell T_e (which sets central-cell bremsstrahlung and is coolable for
    advanced fuels), per the hot-electron-plug / cool-central-cell tandem split.
    Returns MirrorPlasmaState.
    """
    # 1. Geometry
    V_plasma = jnp.pi * a**2 * L
    # First-wall area uses (a + vacuum_t) to match the geometry layer:
    # firstwall_area = 2 pi (a + vacuum_t) L (first-wall basis, not bare plasma).
    fw_area = 2.0 * jnp.pi * (a + vacuum_t) * L

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

    # 6. Tandem plug confining potential (BOUNDED, FIXED by the plug hardware).
    # The cost model commits to a tandem (n_plug_coils = 4, Hammir class), so the
    # central-cell ions are confined by the end-plug electrostatic potential
    # e*phi = T_e_plug*ln(n_p/n_c) (Fowler & Logan 1977; Frank et al. 2024 eq. 18),
    # set by the FIXED plug-to-central density ratio and the HOT-ELECTRON PLUG
    # temperature T_e_plug. The plug is decoupled from the central cell: T_e_plug
    # is hot (ECH-heated) to build the potential, while the central-cell T_e (used
    # for radiation, beta, and W_th below) is coolable so advanced fuels limit
    # bremsstrahlung. Because e*phi does not scale with T_i, the Pastukhov
    # enhancement exp(e*phi/T_i) WEAKENS as T_i rises (at fixed T_e_plug): heating
    # the central cell costs confinement rather than buying it, so the optimum
    # settles at a cooler driven point instead of igniting. The unbounded
    # simple-mirror Boltzmann value is kept only as a diagnostic.
    phi = compute_plug_potential(T_e_plug, plug_density_ratio)

    # 8-11. Confinement times and the axial/radial loss split (extracted helper).
    cl = _confinement_and_losses(
        T_e,
        n_e=n_e,
        T_i=T_i,
        n_i_frac=n_i_frac,
        M_ion=M_ion,
        R_m=R_m,
        L=L,
        a=a,
        B_min=B_min,
        phi=phi,
    )
    tau_classical = cl["tau_classical"]
    tau_Pastukhov = cl["tau_Pastukhov"]
    tau_GD = cl["tau_GD"]
    tau_p = cl["tau_p"]
    tau_E = cl["tau_E"]
    p_end = cl["p_end"]
    p_radial = cl["p_radial"]
    collisionality = cl["collisionality"]

    # 12. Diagnostic: f_axial_derived = p_end / (p_end + p_radial)
    # By construction this equals the tau-channel weight (axial_frac), but
    # expressed as the power ratio to make the diagnostic intent explicit.
    f_axial_derived = p_end / (p_end + p_radial)

    # 13. Beta (dilution-aware, midplane field B_min)
    # beta = 2 * mu_0 * n_e * (T_e + n_i_frac * T_i) * KEV_TO_J / B_min^2
    beta = 2.0 * _MU_0 * n_e * (T_e + n_i_frac * T_i) * _KEV_TO_J / B_min**2

    # 14. Wall loading
    wall_loading = p_neutron / fw_area

    # 14b. Surface heat-flux: photons + radial cross-field transport onto FW.
    # Axial end losses (p_end) exit through the throats to the expander/DEC
    # plates and never touch the lateral first wall.
    q_surface = (p_rad + p_radial) / fw_area

    # 15b. Stability / validity diagnostics (Task 3; informational, not constraints).
    # Pastukhov-Maxwellian validity flag: 1.0 when collisional enough for the bare
    # Pastukhov-Maxwellian core assumption (collisionality >= collisionality_min),
    # 0.0 when deeply collisionless (confinement over-credited). A tandem
    # legitimately runs collisionless and plugged, so this flags where the bare
    # assumption is stretched, not a hard error. Bool-as-float for JAX/state parity.
    pastukhov_valid = jnp.where(collisionality >= collisionality_min, 1.0, 0.0)
    # DCLC microstability proxy: ion gyroradii across the plasma, a / rho_i.
    dclc_parameter = compute_dclc_parameter(a, T_i, M_ion, B_min)

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
        q_surface=q_surface,
        R_m=R_m,
        collisionality=collisionality,
        pastukhov_valid=pastukhov_valid,
        dclc_parameter=dclc_parameter,
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

    lo, hi = fori_loop(0, n_iter, body, (T_lo, T_hi))
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
    # Plasma-to-first-wall vacuum gap [m]; required (no default; comes from YAML).
    vacuum_t: float,
    # Impurity / synchrotron params for power balance
    wall_material=None,
    T_edge: float = 0.05,
    tau_ratio: float = 3.0,
    R_w_pb: float | None = None,
    # Advanced fuel radiation proxy (when set, p_rad = f_rad_fus * p_fus)
    f_rad_fus: float | None = None,
    # Required: beta feasibility limit. No default; comes from YAML.
    beta_max: float,
    # Required: neutron wall-load cap [MW/m^2]. No default; comes from YAML.
    # In inverse (audit) mode the computed wall loading is compared against
    # this threshold and a warning is issued when it is exceeded.
    q_wall_max: float,
    # Required: surface heat-flux cap [MW/m^2]. No default; comes from YAML.
    # In inverse (audit) mode the computed q_surface is compared against
    # this threshold and a warning is issued when it is exceeded.
    q_surface_max: float,
    # Required: control/startup floor on auxiliary sustainment power [MW].
    # No default; comes from YAML. Used only for the audit-mode
    # sustainment-consistency diagnostic (the inverse keeps the stated p_input).
    p_aux_floor: float,
    # Required: alpha loss-cone heating fraction [-]. No default; comes from YAML.
    # Fraction of fusion-alpha power that deposits as self-heating before
    # scattering into the loss cone (Santarius & Callen 1983, about 0.80). Used in
    # the audit-mode sustainment-consistency diagnostic (mirror_aux_heating).
    f_alpha_heat: float,
    # Required: tandem plug-to-central density ratio n_p/n_c [-]. No default;
    # comes from YAML. Sets the bounded, T_i-independent central-cell confining
    # potential e*phi = T_e*ln(n_p/n_c) (compute_plug_potential), calibrated to
    # the Hammir Q>5 design point.
    plug_density_ratio: float,
    # Required: Pastukhov-Maxwellian validity floor on collisionality [-]. No
    # default; comes from YAML. Sets the pastukhov_valid diagnostic flag (1.0 when
    # collisionality >= this floor, 0.0 when deeply collisionless). Informational.
    collisionality_min: float,
    # Required: HOT plug-electron temperature [keV]. No default; comes from YAML.
    # Sets the Fowler-Logan confining potential e*phi = T_e_plug*ln(n_p/n_c),
    # decoupled from the central-cell T_e (which sets bremsstrahlung). Calibrated
    # to the Hammir hot-electron plug (about 125 keV).
    T_e_plug: float,
    # Required: plug sustainment power [MW]. No default; comes from YAML. The
    # ECH/NBI power holding the hot-electron plug, charged into the mirror
    # recirculating power (calibrated to Hammir's about 30 MW plug drive).
    p_plug: float,
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

    # Plug sustainment power into the mirror recirculating budget. The ECH/NBI
    # holding the hot-electron plug is a recirculating load distinct from the
    # central-cell heating: it does NOT heat the central cell (so it must not enter
    # the plasma energy balance via p_input), it is the power cost of the plug
    # that builds the confining potential. It is charged as an additive
    # recirculating term on the mirror side by folding it into the p_coils bucket
    # passed to the shared balance (the established mirror-side hook; the shared
    # function's recirculating sum is p_coils + ... + p_input_eff/eta_pin, so this
    # adds P_plug at unit recirculating cost without touching tokamak behavior).
    # Calibrated to Hammir's about 30 MW plug drive. See
    # docs/account_justification/mirror_confinement_regimes.md.
    p_coils_eff = p_coils + p_plug

    # Step 1: Geometry (fixed inputs; L,a,B_min are not solved here).
    V_plasma = jnp.pi * a**2 * L
    # First-wall area uses (a + vacuum_t) to match the geometry layer.
    fw_area = 2.0 * jnp.pi * (a + vacuum_t) * L
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
        p_coils=p_coils_eff,
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
        vacuum_t=vacuum_t,
        plug_density_ratio=plug_density_ratio,
        collisionality_min=collisionality_min,
        T_e_plug=T_e_plug,
        f_rad_fus=f_rad_fus,
    )

    # Step 4b: Audit-mode sustainment-consistency diagnostic. The inverse keeps
    # the stated p_input (auditing a given machine, NOT closing the balance like
    # sizing mode); report the ratio of stated p_input to the confinement-
    # required auxiliary power (mirror analog of the tokamak's implied H_factor).
    sustainment_ratio = p_input / mirror_aux_heating(
        plasma_state, p_aux_floor, f_alpha_heat
    )
    plasma_state = dataclasses.replace(
        plasma_state, sustainment_ratio=sustainment_ratio
    )

    # Step 5: Feasibility gate — tracer-skip guard matches the tokamak pattern.
    # Under JAX tracing (jit/grad), beta is an abstract tracer; skip the gate.
    if enforce_plasma_limits and not isinstance(plasma_state.beta, Tracer):
        beta_val = float(plasma_state.beta)
        if beta_val > beta_max:
            raise OperatingPointInfeasible(
                f"net target {float(p_net_target):.1f} MW at this mirror geometry "
                f"implies beta = {beta_val:.4f} > beta_max = {beta_max:.4f}. "
                f"Pass enforce_plasma_limits=False to inspect the implied point."
            )
        # Wall loading: warning-severity only (matches tokamak pattern of not raising).
        # Threshold from q_wall_max (YAML-sourced required kwarg; no module constant).
        wl = float(plasma_state.wall_loading)
        # Relative tolerance so an operating point sitting exactly AT the cap
        # (which sizing produces) does not warn "5.00 > 5.00".
        if wl > q_wall_max * (1 + 1e-3):
            warnings.warn(
                f"Neutron wall loading = {wl:.2f} MW/m^2"
                f" > q_wall_max = {q_wall_max:.2f} MW/m^2",
                stacklevel=2,
            )
        # Surface heat-flux: warning-severity only (symmetric with wall loading).
        # Threshold from q_surface_max (YAML-sourced required kwarg).
        qs = float(plasma_state.q_surface)
        # Relative tolerance, symmetric with the wall-loading warning above.
        if qs > q_surface_max * (1 + 1e-3):
            warnings.warn(
                f"Surface heat flux = {qs:.2f} MW/m^2"
                f" > q_surface_max = {q_surface_max:.2f} MW/m^2"
                " (photons + radial cross-field transport on first wall)",
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
        p_coils=p_coils_eff,
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


# ---------------------------------------------------------------------------
# Sizing: net electric at a fixed L (inner operating point)
# ---------------------------------------------------------------------------

# GSS iterations — same convergence parameters as the tokamak sizing solve.
_GSS_ITERS = 40  # Golden-section iterations to locate the optimum T_i
_L_BISECT_ITERS = 60  # Bisection iterations to locate the target-power length


def _density_from_f_beta(
    T_i, T_e, f_beta, beta_max, B_min, fuel, dhe3_fuel_ratio, pb11_fuel_ratio
):
    """Density at the beta boundary [10^20 m^-3].

    n_e = f_beta * beta_max * B^2 / (2 * mu_0 * (T_e + n_i/n_e * T_i) * KEV_TO_J)

    Returns n20 in [10^20 m^-3]. Conversion to SI happens once at the
    _net_at_L_T boundary: n_e = n20 * 1e20.

    Internally computes n20 = ... / (T_sum) with the constant product folded in
    float64 so intermediates remain near unity in float32.
    """
    n_i_frac = n_i_over_n_e(fuel, dhe3_fuel_ratio, pb11_fuel_ratio)
    T_sum = T_e + n_i_frac * T_i
    # Pre-fold constants in float64; T_sum is near unity (tens of keV).
    n20 = f_beta * beta_max * B_min**2 / (2.0 * _MU_0 * _KEV_TO_J * 1e20) / T_sum
    return n20  # [10^20 m^-3]


def _density_from_wall_cap(T_i, T_e, q_wall_max, a, vacuum_t, fuel, params_mix):
    """Density at the neutron wall-load cap [10^20 m^-3].

    q_n = f_n * C_fus(T) * n_e^2 * V / A_fw with V/A_fw = a^2 / (2 (a + vacuum_t))
    (cylinder, L cancels), so

        n_e = sqrt( 2 * q_wall_max * (a + vacuum_t) / (f_n * C_fus(T) * a^2) )

    Work in n20 units throughout to avoid the 1e40 density-squared hazard
    (see reactivity.py docstring). Evaluate fusion_power at n20 = 1 (n_e = 1e20
    m^-3) with unit volume to get C_fus_20 [MW] — an order-unity number whose
    gradient intermediates stay well within float32 range. Then:

        n20 = sqrt(2 * q_wall_max * (a + vacuum_t) / (f_n * C_fus_20 * a^2))

    Returns n20 in [10^20 m^-3]. Conversion to SI happens once at the
    _net_at_L_T boundary: n_e = n20 * 1e20.

    f_n is from event_energies using the effective D-He3 fraction (pin-aware:
    resolved the same way mirror_0d_forward does). event_energies avoids
    the VJP underflow hazard of dividing p_neutron / p_fus (whose quotient-rule
    backward squares a tiny denominator in float32).
    """
    # Reference density: n20 = 1 (n_e = 1e20 m^-3). C_fus_20 is order unity
    # (DT at 20 keV: ~3 MW/m^3), so all gradient intermediates stay in float32.
    C_fus_20, frac_eff = fusion_power(
        fuel,
        1e20,  # n_e = 1e20 m^-3 (n20 = 1; order-unity C_fus_20)
        T_i,
        1.0,  # unit volume [m^3]
        dhe3_fuel_ratio=params_mix["dhe3_fuel_ratio"],
        pb11_fuel_ratio=params_mix["pb11_fuel_ratio"],
        dhe3_dd_frac_pin=params_mix["dhe3_dd_frac_pin"],
        dd_f_T=params_mix["dd_f_T"],
        dd_f_He3=params_mix["dd_f_He3"],
        dhe3_f_T=params_mix["dhe3_f_T"],
        dhe3_f_He3=params_mix["dhe3_f_He3"],
        pb11_f_alpha_n=params_mix["pb11_f_alpha_n"],
        pb11_f_p_n=params_mix["pb11_f_p_n"],
    )

    # f_n = E_neutron / E_total (energy partition from event_energies).
    # frac_eff is the effective D-He3 fraction resolved from the rate ratio
    # (pin-aware — matches mirror_0d_forward).
    E_total, E_neutron = event_energies(
        fuel,
        params_mix["dd_f_T"],
        params_mix["dd_f_He3"],
        frac_eff,  # effective D-He3 fraction (T_i-dependent for DHE3)
        params_mix["dhe3_f_T"],
        params_mix["dhe3_f_He3"],
        params_mix["pb11_f_alpha_n"],
        params_mix["pb11_f_p_n"],
    )
    f_n = E_neutron / E_total

    # n20 = sqrt(2 * q * (a+v) / (f_n * C_fus_20 * a^2))
    # C_fus_20 ~ 3 MW for DT at 20 keV -> denominator ~ 3*2.25 ~ 7 (order unity).
    # Guard: when f_n ~ 0 (aneutronic fuel, e.g. p-B11 with no side reactions),
    # the neutron wall cap does not apply; return a very large n20 so
    # jnp.minimum(n_beta_20, n_wall_20) = n_beta_20 (cap is non-binding).
    # Sentinel: n20 = 1e8 (finite in float32; far above any physical density).
    #
    # JAX jnp.where evaluates both branches unconditionally; the VJP of
    # sqrt(2q(a+v) / denominator) at denominator = 0 is -inf, and
    # 0 * (-inf) = nan in the backward pass even when the condition masks the
    # true branch. Fix: replace denominator with 1.0 (via an inner jnp.where)
    # before the sqrt so the true-branch VJP is finite everywhere; the outer
    # jnp.where then selects the sentinel 1e8 with gradient 0 for the
    # aneutronic case.
    denominator = f_n * C_fus_20 * a**2
    safe_denom = jnp.where(denominator > 0, denominator, jnp.ones_like(denominator))
    n20_if_active = jnp.sqrt(2.0 * q_wall_max * (a + vacuum_t) / safe_denom)
    n20 = jnp.where(denominator > 0, n20_if_active, jnp.array(1e8))
    return n20  # [10^20 m^-3]


def _density_from_surface_cap(
    T_i,
    T_e,
    q_surface_max,
    a,
    vacuum_t,
    fuel,
    params_mix,
    forward_kwargs,
    n_beta_20=None,
):
    """Density at the surface heat-flux cap [10^20 m^-3].

    Finds n20 such that q_surface(n20; T_i, T_e) == q_surface_max, where
    q_surface = (p_rad + p_radial) / fw_area (photons + radial cross-field
    transport onto the lateral first wall; axial end-loss exits through the
    throats and is excluded).

    Uses uniform monotone bisection on n20. Radiation (bremsstrahlung +
    synchrotron) and radial transport both increase monotonically with density
    at fixed T, so bisection is well-posed.

    n_beta_20: optional beta-limited density ceiling [10^20 m^-3]. When
    provided (the normal sizing path), the bisection bracket is clamped to
    [1e-3, n_beta_20] and the early-out fires when q_surface at n_beta_20
    is already below the cap -- at that point the surface cap cannot bind
    (n_surf > n_beta) so returning n_beta_20 causes min(n_beta, n_surf)
    = n_beta (the beta branch remains active). This eliminates the 40-
    iteration bisection for the common non-binding case. When n_beta_20 is
    None (standalone call for testing/diagnostic), the bracket is [1e-3, 1e2].

    Eager-only. The function calls float() on the probe q_surface value and
    branches on ``if q_mid < q_surface_max`` with a plain Python float, so it
    cannot be jit-traced. This is deliberate: the entire sizing path
    (_net_at_L_T -> GSS -> L bisection) is a Python eager loop, so no tracing
    is ever attempted. See the jit-compiled counterpart _density_from_wall_cap
    for the closed-form approach that is actually jit-safe.

    Returns n_surf_20 in [10^20 m^-3]. Conversion to SI happens once at the
    _net_at_L_T boundary: n_e = jnp.minimum(n_beta_20, jnp.minimum(n_wall_20,
    n_surf_20)) * 1e20.
    """

    def _q_surface_at_n20(n20_probe):
        # Evaluate q_surface at this density: run a mini-forward at L=1, unit volume.
        # L cancels: q_surface = (p_rad + p_radial) / (fw_area_per_L * L), and
        # all power terms scale with L, so we can use L=1 throughout.
        # Read q_surface directly from the probe state -- the forward already
        # computes it, so no local fw_area recomputation is needed.
        n_e_probe = n20_probe * 1e20
        ps_probe = mirror_0d_forward(
            L=1.0,
            a=a,
            B_min=forward_kwargs["B_min"],
            R_m=forward_kwargs["R_m"],
            T_i=T_i,
            T_e=T_e,
            n_e=n_e_probe,
            p_input=0.0,  # heating not needed for radiation/loss estimate
            fuel=fuel,
            M_ion=forward_kwargs.get("M_ion", 2.5),
            Z_eff=forward_kwargs.get("Z_eff", 1.2),
            R_w=forward_kwargs.get("R_w", 0.4),
            dd_f_T=params_mix["dd_f_T"],
            dd_f_He3=params_mix["dd_f_He3"],
            dhe3_dd_frac_pin=params_mix["dhe3_dd_frac_pin"],
            dhe3_f_T=params_mix["dhe3_f_T"],
            dhe3_f_He3=params_mix["dhe3_f_He3"],
            pb11_f_alpha_n=params_mix["pb11_f_alpha_n"],
            pb11_f_p_n=params_mix["pb11_f_p_n"],
            dhe3_fuel_ratio=params_mix["dhe3_fuel_ratio"],
            pb11_fuel_ratio=params_mix["pb11_fuel_ratio"],
            vacuum_t=vacuum_t,
            plug_density_ratio=forward_kwargs["plug_density_ratio"],
            collisionality_min=forward_kwargs["collisionality_min"],
            T_e_plug=forward_kwargs["T_e_plug"],
            f_rad_fus=forward_kwargs.get("f_rad_fus"),
        )
        return float(ps_probe.q_surface)

    # Bracket: [1e-3, n_beta_20] when the ceiling is known; [1e-3, 1e2] standalone.
    lo_n20 = 1e-3
    hi_n20 = float(n_beta_20) if n_beta_20 is not None else 1e2

    # Early-out: if q_surface at the bracket top (= n_beta_20 in the sizing path)
    # is still below the cap, the surface cap cannot bind at any density the
    # machine will actually operate at. Return hi_n20 as a sentinel so that
    # min(n_beta_20, n_surf_20) = n_beta_20 (the beta branch remains active).
    # This costs ONE forward evaluation instead of 40, which is the common case
    # for DT/DD at modest radiation fractions and typical q_surface_max = 1 MW/m^2.
    q_hi = _q_surface_at_n20(hi_n20)
    if q_hi <= q_surface_max:
        return hi_n20  # sentinel: cap not binding

    # Bisection: 40 iterations, Python floats (eager-only; see docstring).
    for _ in range(40):
        mid_n20 = 0.5 * (lo_n20 + hi_n20)
        q_mid = _q_surface_at_n20(mid_n20)
        if q_mid < q_surface_max:
            lo_n20 = mid_n20
        else:
            hi_n20 = mid_n20

    return 0.5 * (lo_n20 + hi_n20)  # [10^20 m^-3]


def _net_at_L_T(L, T_i, params, fuel, *, _surf_cap_cache=None, return_full=False):
    """Net electric power [MW] at fixed L and T_i.

    Derives density from the f_beta closed form, runs mirror_0d_forward, then
    mfe_forward_power_balance. Returns (p_net [MW], n_e [m^-3], T_i [keV]).
    With return_full=True, returns (p_net [MW], MirrorPlasmaState, PowerTable):
    the full forward state and power table already built internally, surfaced so
    callers (e.g. the sizing handoff) can use the GSS-optimum operating point
    directly instead of re-solving T_i through the inverse path.
    T_e is held fixed from params (the spec: GSS scans T_i with T_e from YAML).

    _surf_cap_cache: optional dict keyed by T_i (float). When provided, the
    surface cap density n_surf_20 is looked up rather than recomputed if T_i
    was already evaluated. Results are stored on miss. Since n_surf_20 is L-
    independent, the same value is valid for every L at the same T_i. Across
    the L bisection loop the GSS probes the same golden-section T_i values each
    step (the bracket is fixed), so a shared cache across all L steps reduces
    the total _density_from_surface_cap calls from O(L_iters x GSS_iters) to
    O(GSS_iters) -- typically ~42 misses instead of ~4800.
    """
    T_e = params["T_e"]
    # returns [10^20 m^-3], converted at the _net_at_L_T boundary
    n_beta_20 = _density_from_f_beta(
        T_i,
        T_e,
        params["f_beta"],
        params["beta_max"],
        params["B_min"],
        fuel,
        params["dhe3_fuel_ratio"],
        params["pb11_fuel_ratio"],
    )
    params_mix = dict(
        dd_f_T=params["dd_f_T"],
        dd_f_He3=params["dd_f_He3"],
        dhe3_dd_frac_pin=params.get("dhe3_dd_frac_pin"),
        dhe3_f_T=params["dhe3_f_T"],
        dhe3_f_He3=params["dhe3_f_He3"],
        pb11_f_alpha_n=params["pb11_f_alpha_n"],
        pb11_f_p_n=params["pb11_f_p_n"],
        dhe3_fuel_ratio=params["dhe3_fuel_ratio"],
        pb11_fuel_ratio=params["pb11_fuel_ratio"],
    )
    # returns [10^20 m^-3], converted at the _net_at_L_T boundary
    n_wall_20 = _density_from_wall_cap(
        T_i,
        T_e,
        params["q_wall_max"],
        params["a"],
        params["vacuum_t"],
        fuel,
        params_mix,
    )
    # Surface heat-flux branch: bisection on n20 so q_surface(n) == q_surface_max.
    # forward_kwargs is the subset of params that _density_from_surface_cap needs.
    forward_kwargs = dict(
        B_min=params["B_min"],
        R_m=params["R_m"],
        M_ion=params.get("M_ion", 2.5),
        Z_eff=params.get("Z_eff", 1.2),
        R_w=params.get("R_w", 0.4),
        plug_density_ratio=params["plug_density_ratio"],
        collisionality_min=params["collisionality_min"],
        T_e_plug=params["T_e_plug"],
        f_rad_fus=params.get("f_rad_fus"),
    )
    # Use the cross-L cache when available (T_i is the key; n_surf_20 is L-independent).
    cache_key = float(T_i)
    if _surf_cap_cache is not None and cache_key in _surf_cap_cache:
        n_surf_20 = _surf_cap_cache[cache_key]
    else:
        # returns [10^20 m^-3], converted at the _net_at_L_T boundary.
        # Pass n_beta_20 as the upper bracket ceiling so the early-out fires
        # in the common non-binding case with only ONE forward evaluation.
        n_surf_20 = _density_from_surface_cap(
            T_i,
            T_e,
            params["q_surface_max"],
            params["a"],
            params["vacuum_t"],
            fuel,
            params_mix,
            forward_kwargs,
            n_beta_20=float(n_beta_20),
        )
        if _surf_cap_cache is not None:
            _surf_cap_cache[cache_key] = n_surf_20
    # Three-branch density resolution: n_e = min(n_beta, n_wall, n_surf).
    # Chained jnp.minimum preserves JAX-compatibility.
    n_e = (
        jnp.minimum(n_beta_20, jnp.minimum(n_wall_20, n_surf_20)) * 1e20
    )  # Convert to m^-3
    a = params["a"]
    B_min = params["B_min"]
    R_m = params["R_m"]

    ps = mirror_0d_forward(
        L=L,
        a=a,
        B_min=B_min,
        R_m=R_m,
        T_i=T_i,
        T_e=T_e,
        n_e=n_e,
        p_input=params["p_input"],
        fuel=fuel,
        M_ion=params.get("M_ion", 2.5),
        Z_eff=params.get("Z_eff", 1.2),
        R_w=params.get("R_w", 0.4),
        dd_f_T=params["dd_f_T"],
        dd_f_He3=params["dd_f_He3"],
        dhe3_dd_frac_pin=params.get("dhe3_dd_frac_pin"),
        dhe3_f_T=params["dhe3_f_T"],
        dhe3_f_He3=params["dhe3_f_He3"],
        pb11_f_alpha_n=params["pb11_f_alpha_n"],
        pb11_f_p_n=params["pb11_f_p_n"],
        dhe3_fuel_ratio=params["dhe3_fuel_ratio"],
        pb11_fuel_ratio=params["pb11_fuel_ratio"],
        vacuum_t=params["vacuum_t"],
        plug_density_ratio=params["plug_density_ratio"],
        collisionality_min=params["collisionality_min"],
        T_e_plug=params["T_e_plug"],
        f_rad_fus=params.get("f_rad_fus"),
    )

    R_eff = L / (2.0 * math.pi)
    # First-wall basis moved to a + vacuum_t
    fw_area = 2.0 * math.pi * (a + params["vacuum_t"]) * L
    wm_raw = params.get("wall_material")
    if isinstance(wm_raw, str):
        from costingfe.types import WallMaterial

        wall_mat = WallMaterial(wm_raw)
    else:
        wall_mat = wm_raw

    frac_eff = (
        float(ps.dhe3_dd_frac_eff) if fuel == Fuel.DHE3 else params["dhe3_dd_frac"]
    )
    # Sizing-mode energy-balance closure: charge the confinement-derived
    # auxiliary power as p_input (the tokamak pattern), not the fixed YAML
    # p_input. This makes the GSS objective (net electric) reflect the true
    # recirculating cost of sustainment. Only f_alpha_heat * p_alpha deposits as
    # self-heating (alpha loss-cone reduction, Santarius & Callen 1983); the lost
    # fraction (1 - f_alpha_heat) * p_alpha is routed to the transport/DEC channel
    # below.
    f_alpha_heat = params["f_alpha_heat"]
    p_aux = float(mirror_aux_heating(ps, params["p_aux_floor"], f_alpha_heat))

    # DEC routing (corrected end-loss-only path, extended for the loss-cone alpha
    # exhaust). The mirror's direct converter sits at the end-plug expanders and
    # recovers the AXIAL channel that exits through the throats: the end-loss
    # P_end plus the directed loss-cone alpha exhaust (1 - f_alpha_heat) * P_alpha
    # (the alphas that scatter out the loss cone before thermalising are charged
    # particles that stream to the expander/DEC plates, not radiation or radial
    # transport). P_radial and P_rad strike the lateral first wall and go to the
    # thermal cycle. The shared balance instead routes DEC off its own
    # p_transport = p_ash + p_input_eff - p_rad. p_rad_override below pins the
    # shared p_rad to the mirror's own ps.p_rad, so p_transport_shared here is
    # exactly the value the shared function computes internally; with the alpha
    # loss-cone reduction it equals P_end + P_radial + (1 - f_alpha_heat)*P_alpha.
    # An effective f_dec then makes the shared term recover exactly
    # f_dec * eta_de * (P_end + (1 - f_alpha_heat)*P_alpha):
    #     f_dec_eff * p_transport = f_dec * P_axial_recoverable
    #       ->  f_dec_eff = f_dec * P_axial_recoverable / p_transport.
    # This charges the physically-correct end-plug recovery in code that already
    # exists (the shared function is not modified) and the credit tracks the
    # axial loss, leaving P_radial to the wall. Energy is conserved: the lost
    # alpha power appears in p_transport and is split DEC-recovered / wall exactly
    # like the end loss. See
    # docs/account_justification/mirror_confinement_regimes.md.
    p_ash_mir = float(ps.p_alpha)
    p_rad_mir = float(ps.p_rad)
    p_input_eff_mir = max(p_aux, p_rad_mir - p_ash_mir)
    p_transport_shared = p_ash_mir + p_input_eff_mir - p_rad_mir
    p_alpha_lost = (1.0 - f_alpha_heat) * float(ps.p_alpha)
    p_axial_recoverable = float(ps.p_end) + p_alpha_lost
    f_dec_raw = params["f_dec"]
    if p_transport_shared > 0.0:
        f_dec_eff = f_dec_raw * p_axial_recoverable / p_transport_shared
    else:
        f_dec_eff = f_dec_raw
    pt = mfe_forward_power_balance(
        p_fus=float(ps.p_fus),
        fuel=fuel,
        p_input=p_aux,
        mn=params["mn"],
        eta_th=params["eta_th"],
        eta_p=params["eta_p"],
        eta_pin=params["eta_pin"],
        eta_de=params["eta_de"],
        f_sub=params["f_sub"],
        f_dec=f_dec_eff,
        # Plug sustainment power into the recirculating budget (mirror-side hook;
        # the ECH/NBI holding the hot-electron plug, calibrated to Hammir's about
        # 30 MW). It is a recirculating load, NOT central-cell heating, so it is
        # added to the p_coils bucket rather than p_input. See
        # docs/account_justification/mirror_confinement_regimes.md.
        p_coils=params["p_coils"] + params["p_plug"],
        p_cool=params["p_cool"],
        p_pump=params["p_pump"],
        p_trit=params["p_trit"],
        p_house=params["p_house"],
        p_cryo=params["p_cryo"],
        n_e=float(ps.n_e),
        T_e=T_e,
        Z_eff=params.get("Z_eff", 1.2),
        plasma_volume=float(ps.V_plasma),
        B=B_min,
        R_major=R_eff,
        a_minor=a,
        kappa=1.0,
        R_w=params.get("R_w", 0.4),
        wall_material=wall_mat,
        T_edge=params.get("T_edge", 0.2),
        tau_ratio=params.get("tau_ratio", 3.0),
        fw_area=fw_area,
        # Use the mirror forward's own radiation power (p_rad, clamped to p_alpha
        # and on the open-ended synchrotron geometry) so the shared balance and
        # the confinement-derived aux see ONE consistent p_rad. Without this the
        # shared function recomputes a different (impurity-laden, unclamped) p_rad,
        # which breaks the p_transport = P_end + P_radial identity and the DEC
        # routing (the reviewer's dual-p_rad finding). p_rad_override makes
        # p_transport = p_ash + p_input_eff - ps.p_rad collapse to the real
        # transport loss at sub-ignition, so the plain YAML f_dec nets correctly.
        p_rad_override=float(ps.p_rad),
        dd_f_T=params["dd_f_T"],
        dd_f_He3=params["dd_f_He3"],
        dhe3_dd_frac=frac_eff,
        dhe3_f_T=params["dhe3_f_T"],
        dhe3_f_He3=params["dhe3_f_He3"],
        pb11_f_alpha_n=params["pb11_f_alpha_n"],
        pb11_f_p_n=params["pb11_f_p_n"],
        f_rad_fus=params.get("f_rad_fus"),
    )

    if return_full:
        return float(pt.p_net), ps, pt
    return float(pt.p_net), float(n_e), float(T_i)


def net_electric_at_L(
    L, params, fuel, return_state=False, _surf_cap_cache=None, return_full=False
):
    """Net electric power [MW] at the constraint-boundary operating point for a fixed L.

    Temperature T_i maximizes net power within the fuel-keyed bracket via
    golden-section search with T_e held fixed from params["T_e"]. Density is
    derived from the f_beta closed form at each T_i probe.

    Returns p_net [MW] by default.
    With return_state=True, returns (p_net, T_i_star, n_e_star).
    With return_full=True, returns (p_net, MirrorPlasmaState, PowerTable) at the
    GSS-optimum operating point -- the full forward state and power table the
    sizing handoff consumes directly (no inverse T re-solve).

    _surf_cap_cache: dict shared across L bisection steps. The surface heat-flux
    cap density n_surf_20 is L-independent and the GSS probes the same golden-
    section T_i values each step (fixed bracket), so hits accumulate quickly and
    reduce total _density_from_surface_cap evaluations from O(L x GSS) to O(GSS).
    Pass the same dict to every call to benefit from cross-L reuse.
    """
    T_lo, T_hi = _T_BRACKET_MIRROR[fuel]
    invphi = (5**0.5 - 1) / 2  # golden-section ratio
    invphi2 = (3 - 5**0.5) / 2

    def feasible_net(T_i):
        pn, _n, _T = _net_at_L_T(L, T_i, params, fuel, _surf_cap_cache=_surf_cap_cache)
        return pn

    a_gss, b_gss = T_lo, T_hi
    h = b_gss - a_gss
    c = a_gss + invphi2 * h
    d = a_gss + invphi * h
    fc = feasible_net(c)
    fd = feasible_net(d)
    for _ in range(_GSS_ITERS):
        if fc > fd:  # maximizing
            b_gss, d, fd = d, c, fc
            h = b_gss - a_gss
            c = a_gss + invphi2 * h
            fc = feasible_net(c)
        else:
            a_gss, c, fc = c, d, fd
            h = b_gss - a_gss
            d = a_gss + invphi * h
            fd = feasible_net(d)
    T_star = 0.5 * (a_gss + b_gss)

    if return_full:
        pn, ps, pt = _net_at_L_T(
            L, T_star, params, fuel, _surf_cap_cache=_surf_cap_cache, return_full=True
        )
        return pn, ps, pt

    pn, n_e_star, _ = _net_at_L_T(
        L, T_star, params, fuel, _surf_cap_cache=_surf_cap_cache
    )
    if return_state:
        return pn, T_star, n_e_star
    return pn


# ---------------------------------------------------------------------------
# Outer L bisection solver
# ---------------------------------------------------------------------------
def mirror_size_from_power(params, fuel):
    """Solve chamber length L so net electric power equals the target.

    Net power is monotonic increasing in L at the constraint-boundary operating
    point (P_fus scales with volume; tau_GD improves with L; Pastukhov
    confinement is L-independent), so bisection is well posed.

    Raises SizingInfeasible if the target exceeds what L_max can deliver.
    When the wall-load cap is binding at L_max, the error message names
    q_wall_max and the achieved p_net (decision b: no silent fallback).
    Returns the solved L [m].
    """
    n_mod = params.get("n_mod", 1)
    target = params["net_electric_mw"] / n_mod
    lo, hi = params["L_min"], params["L_max"]

    # Shared surface-cap cache: n_surf_20 is L-independent, so the same value
    # applies at every L for a given T_i. The GSS probes the same golden-section
    # T_i values across L bisection steps (fixed bracket), so the cache rapidly
    # fills on the first few L steps and then hits for all subsequent ones.
    # This reduces _density_from_surface_cap calls from O(L_iters x GSS_iters)
    # to O(GSS_iters) -- ~42 misses instead of ~4800 for a typical DT sizing run.
    _surf_cache: dict = {}

    pn_hi = net_electric_at_L(hi, params, fuel, _surf_cap_cache=_surf_cache)
    if pn_hi < target:
        # Detect whether the wall or surface cap was binding at L_max to give an
        # informative error (decision b). Check by comparing n_wall/n_surf vs n_beta
        # at the GSS T_star.
        _pn, T_star_hi, n_e_star_hi = net_electric_at_L(
            hi, params, fuel, return_state=True, _surf_cap_cache=_surf_cache
        )
        T_e = params["T_e"]
        # returns [10^20 m^-3]
        n_beta_hi_20 = _density_from_f_beta(
            T_star_hi,
            T_e,
            params["f_beta"],
            params["beta_max"],
            params["B_min"],
            fuel,
            params["dhe3_fuel_ratio"],
            params["pb11_fuel_ratio"],
        )
        # n_e_star_hi is m^-3 (from _net_at_L_T boundary); n_beta_hi_20 is [10^20 m^-3].
        # 0.9999: guards against float32 equality at the non-cap boundary;
        # a real cap-bound gap is far larger than the 1e-4 relative tolerance.
        cap_binding = float(n_e_star_hi) < float(n_beta_hi_20) * 1e20 * 0.9999
        if cap_binding:
            # Determine which cap was the binding one: compare n_wall vs n_surf.
            params_mix_hi = dict(
                dd_f_T=params["dd_f_T"],
                dd_f_He3=params["dd_f_He3"],
                dhe3_dd_frac_pin=params.get("dhe3_dd_frac_pin"),
                dhe3_f_T=params["dhe3_f_T"],
                dhe3_f_He3=params["dhe3_f_He3"],
                pb11_f_alpha_n=params["pb11_f_alpha_n"],
                pb11_f_p_n=params["pb11_f_p_n"],
                dhe3_fuel_ratio=params["dhe3_fuel_ratio"],
                pb11_fuel_ratio=params["pb11_fuel_ratio"],
            )
            n_wall_hi_20 = float(
                _density_from_wall_cap(
                    T_star_hi,
                    T_e,
                    params["q_wall_max"],
                    params["a"],
                    params["vacuum_t"],
                    fuel,
                    params_mix_hi,
                )
            )
            fwd_kwargs_hi = dict(
                B_min=params["B_min"],
                R_m=params["R_m"],
                M_ion=params.get("M_ion", 2.5),
                Z_eff=params.get("Z_eff", 1.2),
                R_w=params.get("R_w", 0.4),
                plug_density_ratio=params["plug_density_ratio"],
                collisionality_min=params["collisionality_min"],
                T_e_plug=params["T_e_plug"],
                f_rad_fus=params.get("f_rad_fus"),
            )
            n_surf_hi_20 = _density_from_surface_cap(
                T_star_hi,
                T_e,
                params["q_surface_max"],
                params["a"],
                params["vacuum_t"],
                fuel,
                params_mix_hi,
                fwd_kwargs_hi,
                n_beta_20=float(n_beta_hi_20),
            )
            # Surface cap is the binding one when n_surf < n_wall.
            if n_surf_hi_20 < n_wall_hi_20 * 0.9999:
                raise SizingInfeasible(
                    f"net power at L_max={hi} m is {pn_hi:.1f} MW"
                    f" < target {target:.1f} MW; "
                    "surface heat-flux cap"
                    f" q_surface_max={params['q_surface_max']:.2f} MW/m^2"
                    " is binding (n_surf < n_wall < n_beta at L_max):"
                    " the machine cannot reach the power target under this"
                    " surface heat-flux constraint."
                    " Raise q_surface_max or reduce the target."
                )
            raise SizingInfeasible(
                f"net power at L_max={hi} m is {pn_hi:.1f} MW"
                f" < target {target:.1f} MW; "
                f"wall-load cap q_wall_max={params['q_wall_max']:.2f} MW/m^2"
                " is binding (n_wall < n_beta at L_max): the machine cannot"
                " reach the power target under this neutron wall-load constraint."
                " Raise q_wall_max or reduce the target."
            )
        raise SizingInfeasible(
            f"net power at L_max={hi} m is {pn_hi:.1f} MW < target {target:.1f} MW; "
            "machine cannot reach the power with these physics inputs"
        )

    for _ in range(_L_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        if net_electric_at_L(mid, params, fuel, _surf_cap_cache=_surf_cache) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def alpha_electron_fraction(T_e, E_alpha_keV, e_crit_over_te):
    """Fraction of fast-alpha slowing-down energy delivered to electrons.

    Stix (1972) ion heating fraction f_i(u) = (1/u) int_0^u dx/(1+x^1.5),
    u = E_alpha / E_crit, E_crit = e_crit_over_te * T_e. Returns 1 - f_i.
    e_crit_over_te is about 33 for D-T alphas. Fixed 256-node trapezoid.
    """
    e_crit = e_crit_over_te * T_e
    u = E_alpha_keV / e_crit
    xs = jnp.linspace(0.0, u, 256)
    y = 1.0 / (1.0 + xs**1.5)
    # Manual trapezoid (jnp.trapezoid is not backend-safe on numpy < 2.0).
    integral = jnp.sum(0.5 * (y[1:] + y[:-1]) * (xs[1:] - xs[:-1]))
    f_ion = integral / u
    return 1.0 - f_ion
