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

from costingfe.defaults import CostingConstants
from costingfe.types import Fuel


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
    blanket_vol: float = 0.0,
    shield_vol: float = 0.0,
    structure_vol: float = 0.0,
    vessel_vol: float = 0.0,
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
    # DT: breeding blanket (TBR>1.05) + neutron multiplier
    # DD: energy-capture blanket (no breeding)
    # DHe3/pB11: minimal (X-ray shielding only)
    # Source: pyFECONs cas220101_reactor_equipment.py
    # -----------------------------------------------------------------------
    blanket_unit = {
        Fuel.DT: cc.blanket_unit_cost_dt,
        Fuel.DD: cc.blanket_unit_cost_dd,
        Fuel.DHE3: cc.blanket_unit_cost_dhe3,
        Fuel.PB11: cc.blanket_unit_cost_pb11,
    }
    c220101 = blanket_unit[fuel] * blanket_vol * (p_th / P_TH_REF) ** 0.6

    # -----------------------------------------------------------------------
    # 220102: Shield (HT + LT + Bioshield)
    # Source: pyFECONs cas220102_shield.py
    # -----------------------------------------------------------------------
    shield_scale = {
        Fuel.DT: 1.0,      # Heavy shield (14.1 MeV neutrons)
        Fuel.DD: 0.7,       # Mixed (2.45 MeV neutrons)
        Fuel.DHE3: 0.3,     # Light (~5% neutron fraction)
        Fuel.PB11: 0.1,     # Minimal (aneutronic)
    }
    c220102 = cc.shield_unit_cost * shield_vol * shield_scale[fuel] * (p_th / P_TH_REF) ** 0.6

    # -----------------------------------------------------------------------
    # 220103: Coils (MFE tokamak: TF + CS + PF + shim + structure + cooling)
    # Power-scaled (coil cost depends on B-field and current, not simple volume)
    # Source: pyFECONs cas220103_coils.py
    # -----------------------------------------------------------------------
    c220103 = cc.coils_base * (p_et / 1000.0) ** 0.7

    # -----------------------------------------------------------------------
    # 220104: Supplementary Heating (NBI + ICRF + ECRH + LHCD)
    # Source: pyFECONs cas220104_supplementary_heating.py
    # -----------------------------------------------------------------------
    c220104 = cc.heating_base * (p_et / 1000.0) ** 0.8

    # -----------------------------------------------------------------------
    # 220105: Primary Structure
    # Source: pyFECONs cas220105_primary_structure.py
    # -----------------------------------------------------------------------
    c220105 = cc.structure_unit_cost * structure_vol * (p_et / P_ET_REF) ** 0.5

    # -----------------------------------------------------------------------
    # 220106: Vacuum System (vessel + cryo cooling + pumps)
    # Source: pyFECONs cas220106_vacuum_system.py
    # -----------------------------------------------------------------------
    c220106 = cc.vessel_unit_cost * vessel_vol * (p_et / P_ET_REF) ** 0.6

    # -----------------------------------------------------------------------
    # 220107: Power Supplies
    # Source: pyFECONs cas220107_power_supplies.py
    # -----------------------------------------------------------------------
    c220107 = cc.power_supplies_base * (p_et / 1000.0) ** 0.7

    # -----------------------------------------------------------------------
    # 220108: Divertor (MFE tokamak)
    # Source: pyFECONs cas220108_divertor.py
    # -----------------------------------------------------------------------
    c220108 = cc.divertor_base * (p_th / 1000.0) ** 0.5

    # -----------------------------------------------------------------------
    # 220109: Direct Energy Converter (optional, zero by default)
    # Source: pyFECONs cas220109_direct_energy_converter.py
    # -----------------------------------------------------------------------
    c220109 = 0.0  # DEC placeholder (f_dec=0 for default tokamak)

    # -----------------------------------------------------------------------
    # 220111: Installation Labor
    # Source: pyFECONs cas220111_installation.py
    # -----------------------------------------------------------------------
    reactor_subtotal = (
        c220101 + c220102 + c220103 + c220104 + c220105
        + c220106 + c220107 + c220108 + c220109
    )
    c220111 = cc.installation_frac * reactor_subtotal

    # -----------------------------------------------------------------------
    # 220112: Isotope Separation Plant
    # Fuel-dependent: D extraction, Li-6 enrichment, He-3, B-11
    # Source: pyFECONs cas220112_isotope_separation.py
    # -----------------------------------------------------------------------
    p_gwe = p_net / 1000.0
    scale_06 = p_gwe ** 0.6

    if fuel == Fuel.DT:
        c220112 = (cc.deuterium_extraction_base + cc.li6_enrichment_base) * scale_06
    elif fuel == Fuel.DD:
        c220112 = cc.deuterium_extraction_base * scale_06
    elif fuel == Fuel.DHE3:
        c220112 = (cc.deuterium_extraction_base + cc.he3_extraction_base) * scale_06
    elif fuel == Fuel.PB11:
        c220112 = (cc.protium_purification_base + cc.b11_enrichment_base) * scale_06
    else:
        c220112 = 0.0

    # -----------------------------------------------------------------------
    # 220119: Scheduled Replacement (fraction of reactor subtotal)
    # Source: pyFECONs cas220119_replacement.py
    # -----------------------------------------------------------------------
    replacement_frac = {
        Fuel.DT: cc.replacement_frac_dt,
        Fuel.DD: cc.replacement_frac_dd,
        Fuel.DHE3: cc.replacement_frac_dhe3,
        Fuel.PB11: cc.replacement_frac_pb11,
    }
    c220119 = replacement_frac[fuel] * reactor_subtotal

    # -----------------------------------------------------------------------
    # 220200: Main & Secondary Coolant
    # Source: pyFECONs cas220200_coolant.py
    # -----------------------------------------------------------------------
    c220201 = 166.0 * (n_mod * p_net / 1000.0)  # Primary coolant
    c220202 = 40.6 * (p_th / 3500.0) ** 0.55     # Intermediate coolant
    c220200 = c220201 + c220202

    # -----------------------------------------------------------------------
    # 220300: Auxiliary Cooling + Cryoplant
    # Source: pyFECONs cas220300_auxilary_cooling.py
    # -----------------------------------------------------------------------
    c220301 = 1.10e-3 * n_mod * p_th              # Aux coolant
    c220302 = 200.0 * (p_cryo / 30.0) ** 0.7      # Cryoplant (ref: $200M @ 30MW)
    c220300 = c220301 + c220302

    # -----------------------------------------------------------------------
    # 220400: Radioactive Waste Management
    # Source: pyFECONs cas220400_rad_waste.py
    # -----------------------------------------------------------------------
    c220400 = 1.96 * (p_th / 1000.0)

    # -----------------------------------------------------------------------
    # 220500: Fuel Handling & Storage
    # DT: full tritium processing + containment
    # DD: small-scale tritium + deuterium handling
    # DHe3: He-3 handling
    # pB11: boron powder injection (cheapest)
    # Source: pyFECONs cas220500_fuel_handling_and_storage.py
    # -----------------------------------------------------------------------
    fuel_handling_base = {
        Fuel.DT: cc.fuel_handling_dt_base,
        Fuel.DD: cc.fuel_handling_dd_base,
        Fuel.DHE3: cc.fuel_handling_dhe3_base,
        Fuel.PB11: cc.fuel_handling_pb11_base,
    }
    c220500 = fuel_handling_base[fuel] * (p_net / 1000.0) ** 0.7

    # -----------------------------------------------------------------------
    # 220600: Other Reactor Plant Equipment
    # Source: pyFECONs cas220600_other_plant_equipment.py
    # -----------------------------------------------------------------------
    c220600 = 11.5 * (p_net / 1000.0) ** 0.8

    # -----------------------------------------------------------------------
    # 220700: Instrumentation & Control
    # Source: pyFECONs cas220700_instrumentation_and_control.py
    # -----------------------------------------------------------------------
    c220700 = 85.0 * (p_th / 3500.0) ** 0.65

    # -----------------------------------------------------------------------
    # Total CAS22 (per module, then multiply)
    # -----------------------------------------------------------------------
    per_module = (
        c220101 + c220102 + c220103 + c220104 + c220105
        + c220106 + c220107 + c220108 + c220109 + c220111
        + c220112 + c220119
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
        "C220111": c220111,
        "C220112": c220112,
        "C220119": c220119,
        "C220200": c220200,
        "C220300": c220300,
        "C220400": c220400,
        "C220500": c220500,
        "C220600": c220600,
        "C220700": c220700,
        "C220000": c220000,
    }
