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

from costingfe.defaults import CostingConstants
from costingfe.types import CoilMaterial, ConfinementConcept, ConfinementFamily, Fuel

# Concept-dependent coil defaults (from pyFECONs cas220103_coils.py)
# markup: manufacturing complexity multiplier over raw conductor cost
# path_factor: extra coil path length for 3D geometries (stellarator)
_COIL_DEFAULTS = {
    ConfinementConcept.TOKAMAK: {"markup": 8.0, "path_factor": 1.0},
    ConfinementConcept.STELLARATOR: {"markup": 12.0, "path_factor": 2.0},
    ConfinementConcept.MIRROR: {"markup": 2.5, "path_factor": 1.0},
}

_MU0 = 4 * math.pi * 1e-7  # Vacuum permeability (T·m/A)


def _compute_geometry_factor(
    concept: ConfinementConcept,
    path_factor: float,
) -> float:
    """Geometry factor G for conductor quantity scaling.

    total_kAm = G * B * R^2 / (mu_0 * 1000)

    Tokamak: G = 4pi^2 — empirical total-system (TF+CS+PF) scaling.
    Mirror:  G = 4 * 4*pi — 4 independent solenoid coils.
    Stellarator: G = 4*pi^2 * path_factor — 3D coil paths ~2x longer.
    """
    if concept == ConfinementConcept.MIRROR:
        return 4 * 4 * math.pi  # n_coils=4 for mirror
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
    b_max: float,
    r_coil: float,
    coil_material: CoilMaterial,
    p_nbi: float,
    p_icrf: float,
    p_ecrh: float,
    p_lhcd: float,
) -> dict[str, float]:
    """Compute all CAS22 sub-accounts. Returns dict of account_code -> M$.

    Volume-based accounts use hybrid formula:
      cost = unit_cost * volume * (power / power_ref)^alpha
    This captures both reactor size (volume) and thermal intensity (power).
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
    # TODO: incorporate wall_material cost multiplier into C220101
    # (W tiles vs flowing Li systems vs SiC composites have very different
    # fabrication costs — requires dedicated research)
    c220101 = blanket_unit[fuel] * blanket_vol * (p_th / P_TH_REF) ** 0.6

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
    # 220103: Coils — conductor scaling law
    # cost = total_kAm * $/kAm * markup / 1e6
    # REBCO HTS default: $50/kAm (NOAK target; current market $150-300/kAm)
    # Markup captures winding, insulation, quench protection, cryostat, testing
    # See docs/account_justification/CAS22_reactor_components.md
    # -----------------------------------------------------------------------
    defaults = _COIL_DEFAULTS.get(concept, _COIL_DEFAULTS[ConfinementConcept.TOKAMAK])
    coil_markup = defaults["markup"]
    path_factor = defaults["path_factor"]
    G = _compute_geometry_factor(concept, path_factor)
    total_kAm = G * b_max * r_coil**2 / (_MU0 * 1000)
    conductor_cost = total_kAm * coil_material.default_cost_per_kAm / 1e6
    c220103 = conductor_cost * coil_markup

    # -----------------------------------------------------------------------
    # 220104: Supplementary Heating — vendor-purchased turnkey systems
    # Per-MW linear costs calibrated to ITER procurement (FOAK-to-NOAK adjusted)
    # See docs/account_justification/CAS22_reactor_components.md
    # -----------------------------------------------------------------------
    c220104 = (
        cc.heating_nbi_per_mw * p_nbi
        + cc.heating_icrf_per_mw * p_icrf
        + cc.heating_ecrh_per_mw * p_ecrh
        + cc.heating_lhcd_per_mw * p_lhcd
    )

    # -----------------------------------------------------------------------
    # 220105: Primary Structure — gravity supports, thermal shields,
    # inter-coil structure, machine base.
    # See docs/account_justification/CAS22_reactor_components.md
    # -----------------------------------------------------------------------
    c220105 = cc.structure_unit_cost * structure_vol * (p_et / P_ET_REF) ** 0.5

    # -----------------------------------------------------------------------
    # 220106: Vacuum System — vessel (double-walled SS), port extensions,
    # cryopumps, vacuum gauges, leak detection.
    # See docs/account_justification/CAS22_reactor_components.md
    # -----------------------------------------------------------------------
    c220106 = cc.vessel_unit_cost * vessel_vol * (p_et / P_ET_REF) ** 0.6

    # -----------------------------------------------------------------------
    # 220107: Power Supplies — vendor-purchased (ABB, GE, Siemens)
    # High-current DC for magnets, pulsed power, switchgear.
    # See docs/account_justification/CAS22_plant_systems.md
    # -----------------------------------------------------------------------
    c220107 = cc.power_supplies_base * (p_et / 1000.0) ** 0.7

    # -----------------------------------------------------------------------
    # 220108: Divertor (MFE) or Target Factory (IFE/MIF)
    # MFE: W monoblock cassettes on CuCrZr heat sinks
    # IFE/MIF: high-rep-rate target manufacturing infrastructure
    # See docs/account_justification/CAS22_plant_systems.md
    # -----------------------------------------------------------------------
    if family == ConfinementFamily.MFE:
        c220108 = cc.divertor_base * (p_th / 1000.0) ** 0.5
    else:  # IFE or MIF — target factory
        c220108 = cc.target_factory_base * (p_et / 1000.0) ** 0.7

    # -----------------------------------------------------------------------
    # 220109: Direct Energy Converter — excluded for tokamak/stellarator.
    # Relevant only for mirror/FRC concepts with charged-particle exhaust.
    # Override via cost_overrides["C220109"] for custom configurations.
    # See docs/account_justification/CAS22_plant_systems.md
    # -----------------------------------------------------------------------
    c220109 = 0.0

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
    per_module = (
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
        + c220111
        + c220112
    )
    plant_wide = c220200 + c220300 + c220400 + c220500 + c220600 + c220700
    c220000 = per_module * n_mod + plant_wide

    return {
        "C220101": c220101,
        "C220102": c220102,
        "C220103": c220103,
        "C220104": c220104,
        "C220105": c220105,
        "C220106": c220106,
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
