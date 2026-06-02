"""CAS22: Reactor Plant Equipment sub-accounts.

Hybrid volume + thermal-intensity costing for geometry-dependent items:
  cost = unit_cost * volume * (p_th / p_th_ref)^alpha

Volume captures reactor size (geometry dimensions). Thermal intensity
captures the fact that components handling more power need better
cooling, thicker walls, and higher-grade materials per unit volume.

Power-scaled for remaining items (coils, heating, power supplies, divertor).
Fuel-dependent config for blanket, isotope sep, fuel handling.

All costs in M$. Source: pyFECONs costing/calculations/cas22/
"""

import math

import jax.numpy as jnp

from costingfe.defaults import CostingConstants
from costingfe.types import (
    BlanketFill,
    BlanketForm,
    CoilMaterial,
    ConfinementConcept,
    ConfinementFamily,
    Fuel,
    PulsedConversion,
)

# Concept-dependent coil defaults (from pyFECONs cas220103_coils.py)
# markup: manufacturing complexity multiplier over raw conductor cost
# path_factor: extra coil path length for 3D geometries (stellarator)
# n_coils: number of discrete coils — only used for mirror (G = n_coils * 4*pi);
#         ignored by tokamak/stellarator branches whose G is empirical total-system
# None → no confinement magnets (IFE drivers, magnet-free pulsed concepts)
_COIL_DEFAULTS = {
    # MFE / electrostatic — full confinement magnets
    ConfinementConcept.TOKAMAK: {"markup": 8.0, "path_factor": 1.0, "n_coils": 0},
    ConfinementConcept.STELLARATOR: {"markup": 12.0, "path_factor": 2.0, "n_coils": 0},
    # Mirror n_coils calibrated to Realta HAMMIR-class tandem mirror:
    # 4 end-plug HTS coils (2 per end, Hammer evolution) + ~6 LTS central-cell
    # solenoid coils discretizing the 50 m central cell. Simple-mirror devices
    # (WHAM/BEAM/Anvil) would use n_coils ≈ 4.
    ConfinementConcept.MIRROR: {"markup": 2.5, "path_factor": 1.0, "n_coils": 10},
    # DIPOLE n_coils is the STATIONARY levitation/support coil set only (>=2):
    # the external SC coils that produce the field holding the floating coil up.
    # The levitated coil itself is NOT in this count — it is costed as a separate
    # additive term in C220103 (lev_coil_markup + integral cryostat), because it
    # is a distinct piece of tech (cooled mid-plasma, persistent current while
    # disconnected, unshielded). See cas22 DIPOLE branch below.
    ConfinementConcept.DIPOLE: {"markup": 3.0, "path_factor": 1.0, "n_coils": 2},
    # Steady-state FRC (beam-driven, e.g. TAE; RMF-driven PFRC) is a linear,
    # open-field-line device like a mirror: external formation + mirror/end-plug
    # coils discretized as independent solenoids (G = n_coils * 4*pi). The
    # self-organized internal FRC field carries no coil cost. n_coils counts only
    # the external coil set; resistive-magnet markup is lower than HTS mirrors.
    ConfinementConcept.STEADY_FRC: {"markup": 2.0, "path_factor": 1.0, "n_coils": 4},
    ConfinementConcept.PULSED_FRC: {"markup": 1.5, "path_factor": 1.0, "n_coils": 0},
    ConfinementConcept.THETA_PINCH: {"markup": 1.5, "path_factor": 1.0, "n_coils": 0},
    ConfinementConcept.ORBITRON: {"markup": 1.5, "path_factor": 1.0, "n_coils": 0},
    ConfinementConcept.POLYWELL: {"markup": 2.0, "path_factor": 1.0, "n_coils": 0},
    # MIF — guide-field solenoids (simpler, smaller than full confinement)
    ConfinementConcept.MAG_TARGET: {"markup": 1.5, "path_factor": 1.0, "n_coils": 0},
    ConfinementConcept.PLASMA_JET: {"markup": 1.5, "path_factor": 1.0, "n_coils": 0},
    ConfinementConcept.MAGLIF: {"markup": 2.0, "path_factor": 1.0, "n_coils": 0},
    # IFE / magnet-free pulsed — no confinement magnets
    ConfinementConcept.LASER_IFE: None,
    ConfinementConcept.ZPINCH: None,
    ConfinementConcept.HEAVY_ION: None,
    ConfinementConcept.DENSE_PLASMA_FOCUS: None,
    ConfinementConcept.STAGED_ZPINCH: None,
}

_MU0 = 4 * math.pi * 1e-7  # Vacuum permeability (T·m/A)
_RHO_CU = 8960.0  # Copper density (kg/m³), for resistive-coil mass build-up


def _compute_geometry_factor(
    concept: ConfinementConcept,
    path_factor: float,
    n_coils: int,
) -> float:
    """Geometry factor G for conductor quantity scaling.

    total_kAm = G * B_center * R^2 / (mu_0 * 1000)

    Derived from Biot-Savart for a thin circular loop:
      B_center = mu_0 * N*I / (2 R)  ->  N*I = 2 * B_center * R / mu_0
      kA*m     = N*I * (2 pi R) / 1000 = 4 pi * B_center * R^2 / (mu_0 * 1000)
    so a single circular loop has G = 4 pi. Multi-coil and 3D-path systems
    multiply by per-concept factors below. The B input is the field at the
    center of the loop (axis), NOT the peak field on the conductor.

    Tokamak: G = 4pi^2 — empirical total-system (TF+CS+PF) scaling.
    Mirror:  G = n_coils * 4*pi — sum over independent solenoid coils.
    Steady FRC: G = n_coils * 4*pi — linear device, same as mirror.
    Dipole:  G = n_coils * 4*pi — sum over the stationary levitation coils
             (the levitated coil is costed separately, not via this factor).
    Stellarator: G = 4*pi^2 * path_factor — 3D coil paths ~2x longer.
    """
    if concept in (
        ConfinementConcept.MIRROR,
        ConfinementConcept.STEADY_FRC,
        ConfinementConcept.DIPOLE,
    ):
        return n_coils * 4 * math.pi
    elif concept == ConfinementConcept.STELLARATOR:
        return 4 * math.pi**2 * path_factor
    else:  # tokamak (default)
        return 4 * math.pi**2


def cas22_reactor_plant_equipment(
    cc: CostingConstants,
    p_net: float,
    p_th: float,
    p_et: float,
    p_fus: float,
    p_cryo: float,
    n_mod: int,
    fuel: Fuel,
    noak: bool,
    blanket_vol: float,
    shield_vol: float,
    structure_vol: float,
    vessel_vol: float,
    family: ConfinementFamily,
    concept: ConfinementConcept,
    b_center: float,
    r_bore: float,
    coil_material: CoilMaterial,
    blanket_form: BlanketForm,
    p_nbi: float,
    p_icrf: float,
    p_ecrh: float,
    p_lhcd: float,
    p_driver: float,
    f_dec: float,
    p_dee: float,
    burn_fraction: float,
    vac_op_pressure_pa: float,
    e_driver_mj: float = 0.0,  # Per-pulse driver energy [MJ]; $/J laser/accel basis
    e_preheat_mj: float = 0.0,  # Per-pulse preheat laser energy [MJ]; 0 = no preheat
    # Pulsed DEC parameters
    pulsed_conversion=None,
    e_stored_mj: float = 0.0,
    q_sci: float = 0.0,
    f_ch: float = 0.0,
    eta_dec: float = 0.0,
    n_coils: int | None = None,
    lev_coil_markup: float | None = None,
    lev_coil_cryostat_cost: float | None = None,
    blanket_fill: BlanketFill | None = None,
    peak_to_center_ratio: float = 3.0,
) -> dict[str, float]:
    """Compute all CAS22 sub-accounts. Returns dict of account_code -> M$.

    Volume-based accounts use hybrid formula:
      cost = unit_cost * volume * (power / power_ref)^alpha
    This captures both reactor size (volume) and thermal intensity (power).

    Sub-line convention: where one account aggregates components with distinct
    cost bases, informational sub-lines keyed `<CODE>_<component>` are emitted
    alongside the canonical total (e.g. C220106_vessel, C220106_pump). They are
    excluded from all aggregation and exist for visibility / sensitivity only.
    """
    # Reference power levels at calibration geometry (1 GWe DT tokamak)
    P_TH_REF = 2500.0  # MW thermal
    P_ET_REF = 1100.0  # MW gross electric

    # -----------------------------------------------------------------------
    # 220101: First Wall + Blanket + Neutron Multiplier
    # DT: breeding blanket (TBR>1.05) + neutron multiplier (RAFM steel +
    #   PbLi/Li breeder + Be multiplier + W FW armor). Complex assembly.
    # DD: energy-capture blanket (no breeding). Simpler RAFM steel + coolant.
    # DHe3/pB11: minimal (X-ray shielding only)
    # See docs/account_justification/CAS22_reactor_components.md
    # -----------------------------------------------------------------------
    blanket_unit = {
        Fuel.DT: cc.blanket_unit_cost_dt,
        Fuel.DD: cc.blanket_unit_cost_dd,
        Fuel.DHE3: cc.blanket_unit_cost_dhe3,
        Fuel.PB11: cc.blanket_unit_cost_pb11,
    }
    # Chemistry override: the fuel-keyed table above is calibrated against PbLi
    # liquid-metal blankets (density ~9.4 t/m^3, flow loop + MHD ducts + online
    # tritium extraction). Solid Li2O (~2 t/m^3, no flow loop) is ~3-5x cheaper
    # per m^3. Branch on blanket_fill so DIPOLE-class machines using Li2O do not
    # pay the PbLi premium volumetrically.
    if blanket_fill == BlanketFill.LI2O:
        unit = cc.blanket_unit_cost_li2o
    else:
        unit = blanket_unit[fuel]
    # TODO: incorporate wall_material cost multiplier into C220101
    # (W tiles vs flowing Li systems vs SiC composites have very different
    # fabrication costs — requires dedicated research)
    c220101 = (
        unit
        * blanket_form.structure_factor
        * blanket_vol
        * (p_th / P_TH_REF) ** 0.6
    )

    # -----------------------------------------------------------------------
    # 220102: Shield (HT + LT + Bioshield)
    # Full shield for DT (14.1 MeV), reduced for lower-neutron fuels.
    # See docs/account_justification/CAS22_reactor_components.md
    # -----------------------------------------------------------------------
    shield_scale = {
        Fuel.DT: 1.0,  # Heavy shield (14.1 MeV neutrons)
        Fuel.DD: 0.7,  # Mixed (2.45 MeV neutrons)
        Fuel.DHE3: 0.3,  # Light (~5% neutron fraction)
        Fuel.PB11: 0.1,  # Minimal (aneutronic)
    }
    c220102 = (
        cc.shield_unit_cost * shield_vol * shield_scale[fuel] * (p_th / P_TH_REF) ** 0.6
    )

    # -----------------------------------------------------------------------
    # 220103: Coils. Two cost regimes on the same ampere-meter quantity
    # (total_kAm = G * b_center * r_bore^2 / mu0). b_center is the field at the
    # geometric center of the loop (axis), NOT the peak field on the conductor;
    # peak-on-conductor is a factor peak_to_center_ratio higher (typically 2-4
    # for high-field SC). r_bore is the effective winding-bore radius, NOT a
    # cross-section / cable dimension. See _compute_geometry_factor docstring
    # and docs/account_justification/CAS22_reactor_components.md.
    #   - Superconducting (HTS/LTS): cost = total_kAm * $/kAm * markup. The
    #     expensive conductor dominates; REBCO $50/kAm (NOAK), markup captures
    #     winding, quench protection, cryostat, testing.
    #   - Resistive (copper): the conductor is cheap per kAm, so cost is set by
    #     the bulk copper MASS plus structure and fabrication, not by $/kAm.
    #     Re-price the same ampere-meters by mass: m_Cu = (rho_Cu / J) *
    #     ampere_meters, add steel support, apply fabrication markups. This is
    #     the right basis for low-field FRC/linear copper coils, where the
    #     $/kAm path collapses to a near-zero, unphysical number.
    # -----------------------------------------------------------------------
    # peak_to_center_ratio is currently informational (documents the relation
    # B_max = peak_to_center_ratio * b_center for the conductor) and not used
    # in the kA*m calculation, which is intrinsically a b_center quantity.
    _ = peak_to_center_ratio
    defaults = _COIL_DEFAULTS.get(concept)
    if defaults is None:
        # No confinement magnets (IFE drivers, magnet-free pulsed)
        c220103 = 0.0
    else:
        coil_markup = defaults["markup"]
        path_factor = defaults["path_factor"]
        # Honor per-call override; fall back to concept default
        n_coils_eff = n_coils if n_coils is not None else defaults["n_coils"]
        if n_coils_eff < 0:
            raise ValueError(f"n_coils must be >= 0, got {n_coils_eff}")
        G = _compute_geometry_factor(concept, path_factor, n_coils_eff)
        total_kAm = G * b_center * r_bore**2 / (_MU0 * 1000)
        if coil_material == CoilMaterial.COPPER:
            ampere_meters = total_kAm * 1000.0
            j_a_m2 = cc.coil_cu_current_density_a_mm2 * 1e6
            m_cu = (_RHO_CU / j_a_m2) * ampere_meters
            m_steel = cc.coil_struct_fraction * m_cu
            c220103 = (
                m_cu * cc.coil_cu_price_per_kg * cc.coil_cu_fab_markup
                + m_steel * cc.coil_steel_price_per_kg * cc.coil_steel_fab_markup
            ) / 1e6
        else:
            conductor_cost = total_kAm * coil_material.default_cost_per_kAm / 1e6
            c220103 = conductor_cost * coil_markup

    # Levitated dipole: the floating field coil is a distinct cost object, not
    # another entry in the stationary-coil sum. It is a single HTS ring (G=4*pi)
    # carrying persistent current with an integral onboard cryostat (cooled
    # mid-plasma, electrically disconnected). lev_coil_markup captures the
    # float/energize/no-access engineering; lev_coil_cryostat_cost is the
    # integral cryoplant + levitation control, added on top of the stationary
    # levitation coils already in c220103 above.
    if concept == ConfinementConcept.DIPOLE:
        lev_kAm = 4 * math.pi * b_center * r_bore**2 / (_MU0 * 1000)
        lev_conductor = lev_kAm * coil_material.default_cost_per_kAm / 1e6
        c220103 = c220103 + lev_conductor * lev_coil_markup + lev_coil_cryostat_cost

    # -----------------------------------------------------------------------
    # 220104: Supplementary Heating (MFE) or Primary Driver (pulsed)
    # MFE: per-MW linear costs calibrated to ITER procurement (FOAK→NOAK)
    # Pulsed: concept-specific driver capital (laser, accelerator, mechanical)
    # Concepts whose driver is purely electrical use C220107 instead.
    # See docs/account_justification/CAS22_reactor_components.md
    # -----------------------------------------------------------------------
    if family == ConfinementFamily.STEADY_STATE:
        c220104 = (
            cc.heating_nbi_per_mw * p_nbi
            + cc.heating_icrf_per_mw * p_icrf
            + cc.heating_ecrh_per_mw * p_ecrh
            + cc.heating_lhcd_per_mw * p_lhcd
        )
    else:
        # Lasers, accelerators, and electromagnetic guns: capital is set by
        # per-pulse energy (aperture/diode count, ring charge, coaxial-gun size and
        # peak current), not rep rate, so cost on $/J of E_driver. STAGED_ZPINCH is
        # the sheared-flow coaxial gun + gas injection; its cap bank is in C220107.
        _DRIVER_COST_PER_MJ = {
            ConfinementConcept.LASER_IFE: cc.driver_laser_per_mj,
            ConfinementConcept.HEAVY_ION: cc.driver_heavy_ion_per_mj,
            ConfinementConcept.PLASMA_JET: cc.driver_plasma_jet_per_mj,
            ConfinementConcept.STAGED_ZPINCH: cc.driver_staged_zpinch_per_mj,
        }
        # Pneumatic/mechanical injectors accelerate mass each shot, so average
        # power (throughput) is the defensible basis.
        _DRIVER_COST_PER_MW = {
            ConfinementConcept.MAG_TARGET: cc.driver_mag_target_per_mw,
        }
        # MAGLIF is in neither map: its main driver is electrical (C220107). Only
        # laser preheat lands here, costed per joule of preheat pulse energy, so a
        # magnetized-compression concept with e_preheat_mj=0 pays nothing.
        c220104 = (
            _DRIVER_COST_PER_MJ.get(concept, 0.0) * e_driver_mj
            + _DRIVER_COST_PER_MW.get(concept, 0.0) * p_driver
            + cc.laser_preheat_per_mj * e_preheat_mj
        )

    # -----------------------------------------------------------------------
    # 220105: Primary Structure — gravity supports, thermal shields,
    # inter-coil structure, machine base.
    # See docs/account_justification/CAS22_reactor_components.md
    # -----------------------------------------------------------------------
    c220105 = cc.structure_unit_cost * structure_vol * (p_et / P_ET_REF) ** 0.5

    # -----------------------------------------------------------------------
    # 220106: Vacuum System = vessel shell (volume-based) + gas-load pumping.
    # Vessel: double-walled SS chamber, port extensions, gauges, leak detection.
    # Pumping: installed speed S_req = Q_gas / P_op set by the gas throughput,
    # NOT vessel volume. Q_gas = NBI neutral-gas load (beam particle rate / E_b)
    # + fueling/exhaust throughput (fusion rate / burn_fraction). Outgassing
    # (surface-area term) is negligible for a baked UHV system and is omitted.
    # Applied uniformly to every concept; P_op (the plenum pressure at the pump
    # throat) is the concept-specific knob that makes low-pressure devices
    # (FRC, mirror) pump far more than a high-pressure tokamak divertor.
    # See docs/account_justification/CAS220106_vacuum_pumping.md
    # -----------------------------------------------------------------------
    c220106_vessel = cc.vessel_unit_cost * vessel_vol * (p_et / P_ET_REF) ** 0.6

    _KB = 1.380649e-23  # Boltzmann constant [J/K]
    _QE = 1.602176634e-19  # elementary charge [J/eV]
    kt_gas = _KB * cc.pump_gas_temp_k  # Pa·m³ per pumped particle (Q = N*kT)
    e_fus_mev = {
        Fuel.DT: cc.e_fus_mev_dt,
        Fuel.DD: cc.e_fus_mev_dd,
        Fuel.DHE3: cc.e_fus_mev_dhe3,
        Fuel.PB11: cc.e_fus_mev_pb11,
    }[fuel]
    # NBI neutral-gas throughput [Pa·m³/s]: beam particle rate = p_nbi / E_b.
    e_b_j = cc.nbi_beam_energy_kev * 1e3 * _QE
    q_nbi = cc.pump_nbi_gas_amplification * (p_nbi * 1e6 / e_b_j) * kt_gas
    # Fueling/exhaust throughput [Pa·m³/s]: fuel feed = reaction_rate / burn_fraction;
    # the unburned fraction circulates through the pump each pass.
    e_fus_j = e_fus_mev * 1e6 * _QE
    reaction_rate = p_fus * 1e6 / e_fus_j  # reactions/s
    q_fuel = (
        cc.pump_ion_per_reaction
        * (1.0 - burn_fraction)
        / burn_fraction
        * reaction_rate
        * kt_gas
    )
    s_req = (q_nbi + q_fuel) / vac_op_pressure_pa  # required pumping speed [m³/s]
    c220106_pump = cc.pump_unit_cost * s_req
    c220106 = c220106_vessel + c220106_pump

    # -----------------------------------------------------------------------
    # 220107: Power Supplies — vendor-purchased (ABB, GE, Siemens)
    # Steady-state: high-current DC for superconducting magnets, switchgear.
    # Pulsed: cap bank + switches + charging + buswork on $/J_stored basis.
    # See docs/account_justification/CAS22_plant_systems.md
    # -----------------------------------------------------------------------
    if family == ConfinementFamily.PULSED:
        # $/J_stored basis: pulsed driver (cap bank, laser, accelerator)
        c220107 = cc.c_cap_allin_per_joule * e_stored_mj  # $/J * MJ = M$
    else:
        c220107 = cc.power_supplies_base * (p_et / 1000.0) ** 0.7

    # -----------------------------------------------------------------------
    # 220108: Divertor (MFE) or Target Factory (IFE/MIF)
    # MFE: W monoblock cassettes on CuCrZr heat sinks
    # IFE/MIF: high-rep-rate target manufacturing infrastructure
    # See docs/account_justification/CAS22_plant_systems.md
    # -----------------------------------------------------------------------
    if family == ConfinementFamily.STEADY_STATE:
        c220108 = cc.divertor_base * (p_th / 1000.0) ** 0.5
    elif concept in (
        ConfinementConcept.PULSED_FRC,
        ConfinementConcept.THETA_PINCH,
        ConfinementConcept.DENSE_PLASMA_FOCUS,
        ConfinementConcept.STAGED_ZPINCH,
    ):
        c220108 = 0.0  # No manufactured targets
    else:  # IFE or MIF — target factory
        c220108 = cc.target_factory_base * (p_et / 1000.0) ** 0.7

    # -----------------------------------------------------------------------
    # 220109: Direct Energy Converter
    # Inductive DEC: circuit-derived markups on pulsed driver cost.
    # Electrostatic DEC: for mirrors/FRCs with directed axial exhaust.
    # See docs/account_justification/CAS220109_direct_energy_converter.md
    # -----------------------------------------------------------------------
    if pulsed_conversion == PulsedConversion.INDUCTIVE_DEC:
        # Inductive DEC: circuit-derived markups on pulsed driver cost
        markup_cap = eta_dec * (1.0 + q_sci * f_ch) - 1.0
        delta_cap = c220107 * jnp.maximum(markup_cap, 0.0)
        delta_switch = c220107 * cc.markup_switch_bidir
        delta_inv = cc.c_inv_per_kw_net * p_net / 1e3  # $/kW * MW -> M$
        delta_ctrl = c220107 * cc.markup_controls
        c220109 = delta_cap + delta_switch + delta_inv + delta_ctrl
    else:
        # Electrostatic DEC for mirrors (existing logic) — JAX-safe
        P_DEE_REF = 400.0
        p_dee_safe = jnp.where(p_dee > 0, p_dee, 1.0)
        c220109 = jnp.where(
            p_dee > 0,
            cc.dec_base * (p_dee_safe / P_DEE_REF) ** 0.7,
            0.0,
        )

    # -----------------------------------------------------------------------
    # 220110: Remote Handling & Maintenance Equipment
    # Fuel-dependent (rad-hardening tier) x concept-dependent (vessel geometry).
    # Base costs calibrated to toroidal geometry (tokamak/stellarator).
    # Linear concepts (mirror) have simpler end-access → lower cost.
    # See docs/account_justification/CAS220110_remote_handling.md
    # -----------------------------------------------------------------------
    rh_base = {
        Fuel.DT: cc.remote_handling_dt_base,
        Fuel.DD: cc.remote_handling_dd_base,
        Fuel.DHE3: cc.remote_handling_dhe3_base,
        Fuel.PB11: cc.remote_handling_pb11_base,
    }
    # Toroidal vessels (narrow ports) vs linear (end-access)
    rh_concept_scale = {
        ConfinementConcept.TOKAMAK: 1.0,
        ConfinementConcept.STELLARATOR: 1.0,
        ConfinementConcept.MIRROR: 0.55,
    }
    concept_scale = rh_concept_scale.get(concept, 0.5)
    c220110 = rh_base[fuel] * concept_scale * (p_et / 1000.0) ** 0.5

    # -----------------------------------------------------------------------
    # 220111: Installation Labor — 14% of reactor subtotal
    # Industry norm: 10-20% (nuclear 15-25%, conventional 10-15%)
    # See docs/account_justification/CAS22_plant_systems.md
    # -----------------------------------------------------------------------
    reactor_subtotal = (
        c220101
        + c220102
        + c220103
        + c220104
        + c220105
        + c220106
        + c220107
        + c220108
        + c220109
        + c220110
    )
    c220111 = cc.installation_frac * reactor_subtotal

    # -----------------------------------------------------------------------
    # 220112: Isotope Separation Plant — zeroed
    # No on-site separation plant. All isotope procurement is modeled as
    # market purchase in CAS80 (enriched $/kg prices). The separation
    # plant capital is embedded in the market price.
    # See: docs/account_justification/CAS220112_isotope_separation.md
    # -----------------------------------------------------------------------
    c220112 = 0.0

    # -----------------------------------------------------------------------
    # 220200: Main & Secondary Coolant
    # Primary loops + intermediate HX + secondary to steam generators
    # See docs/account_justification/CAS22_plant_systems.md
    # -----------------------------------------------------------------------
    # Plant-wide accounts use total plant power (n_mod * per-module)
    p_th_total = n_mod * p_th
    p_net_total = n_mod * p_net

    c220201 = 166.0 * (p_net_total / 1000.0)  # Primary coolant
    c220202 = 40.6 * (p_th_total / 3500.0) ** 0.55  # Intermediate coolant
    c220200 = c220201 + c220202

    # -----------------------------------------------------------------------
    # 220300: Auxiliary Cooling + Cryoplant
    # Cryoplant calibrated to ITER: EUR 148M for 75kW @ 4.5K (Air Liquide)
    # See docs/account_justification/CAS22_plant_systems.md
    # -----------------------------------------------------------------------
    c220301 = 1.10e-3 * p_th_total  # Aux coolant
    c220302 = 200.0 * (p_cryo / 30.0) ** 0.7  # Cryoplant (ref: $200M @ 30MW)
    c220300 = c220301 + c220302

    # -----------------------------------------------------------------------
    # 220400: Radioactive Waste Management
    # Low-level activated waste (no fission products)
    # See docs/account_justification/CAS22_plant_systems.md
    # -----------------------------------------------------------------------
    c220400 = 1.96 * (p_th_total / 1000.0)

    # -----------------------------------------------------------------------
    # 220500: Fuel Handling & Storage — fuel-dependent
    # DT: full tritium processing + containment ($120M @ 1 GWe)
    # DD: small-scale tritium + deuterium ($60M)
    # DHe3: He-3 recovery/recycling ($40M)
    # pB11: boron powder injection ($15M)
    # See docs/account_justification/CAS22_plant_systems.md
    # -----------------------------------------------------------------------
    fuel_handling_base = {
        Fuel.DT: cc.fuel_handling_dt_base,
        Fuel.DD: cc.fuel_handling_dd_base,
        Fuel.DHE3: cc.fuel_handling_dhe3_base,
        Fuel.PB11: cc.fuel_handling_pb11_base,
    }
    c220500 = fuel_handling_base[fuel] * (p_net_total / 1000.0) ** 0.7

    # -----------------------------------------------------------------------
    # 220600: Other Reactor Plant Equipment — catch-all
    # See docs/account_justification/CAS22_plant_systems.md
    # -----------------------------------------------------------------------
    c220600 = 11.5 * (p_net_total / 1000.0) ** 0.8

    # -----------------------------------------------------------------------
    # 220700: Instrumentation & Control — plasma control, diagnostics,
    # safety interlocks, data acquisition, plant computer
    # See docs/account_justification/CAS22_plant_systems.md
    # -----------------------------------------------------------------------
    c220700 = 85.0 * (p_th_total / 3500.0) ** 0.65

    # -----------------------------------------------------------------------
    # Total CAS22 (per module, then multiply)
    # -----------------------------------------------------------------------
    per_module_equipment = (
        c220101
        + c220102
        + c220103
        + c220104
        + c220105
        + c220106
        + c220107
        + c220108
        + c220109
        + c220110
        + c220112
    )
    # Multi-unit labor: unit 1 at full c220111, each subsequent unit
    # discounted by cc.multi_unit_labor_factor (default 0.92, fission norm).
    # Equipment is not discounted — each module is still a manufactured copy.
    total_labor = c220111 * (1.0 + (n_mod - 1) * cc.multi_unit_labor_factor)
    plant_wide = c220200 + c220300 + c220400 + c220500 + c220600 + c220700
    c220000 = per_module_equipment * n_mod + total_labor + plant_wide

    return {
        "C220101": c220101,
        "C220102": c220102,
        "C220103": c220103,
        "C220104": c220104,
        "C220105": c220105,
        "C220106": c220106,
        # Informational split of C220106 (vessel shell vs gas-load pumping).
        # Not aggregated: the canonical C220106 already carries the sum. These
        # are for visibility and sensitivity tracing only.
        "C220106_vessel": c220106_vessel,
        "C220106_pump": c220106_pump,
        "C220107": c220107,
        "C220108": c220108,
        "C220109": c220109,
        "C220110": c220110,
        "C220111": c220111,
        "C220112": c220112,
        "C220200": c220200,
        "C220300": c220300,
        "C220400": c220400,
        "C220500": c220500,
        "C220600": c220600,
        "C220700": c220700,
        "C220000": c220000,
    }
