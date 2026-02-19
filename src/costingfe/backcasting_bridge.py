"""Bridge between 1costingfe and fusion-backcasting.

Runs the physics-based forward model and maps CAS22 sub-account costs
to the fusion-backcasting Subsystem format. This replaces the static
default_subsystems.json with physics-derived costs.

Usage:
    from costingfe.backcasting_bridge import generate_subsystems

    subsystems, financial = generate_subsystems(
        concept="tokamak", fuel="dt",
        net_electric_mw=1000.0, availability=0.85, lifetime_yr=30,
    )
    # subsystems: list of dicts matching fusion-backcasting Subsystem schema
    # financial: dict matching fusion-backcasting FinancialParams schema
"""

from costingfe.model import CostModel
from costingfe.types import ConfinementConcept, Fuel, CONCEPT_TO_FAMILY, ConfinementFamily

# Map 1costingfe concepts to fusion-backcasting confinement types
_FAMILY_TO_CONFINEMENT = {
    ConfinementFamily.MFE: "MCF",
    ConfinementFamily.IFE: "ICF",
    ConfinementFamily.MIF: "MCF",  # MIF uses magnetic guide fields
}

# Map 1costingfe fuels to fusion-backcasting fuel types
_FUEL_MAP = {
    Fuel.DT: "D-T",
    Fuel.DD: "D-T",     # fusion-backcasting has no DD, closest is D-T
    Fuel.DHE3: "D-He3",
    Fuel.PB11: "p-B11",
}


def generate_subsystems(
    concept: str,
    fuel: str,
    net_electric_mw: float = 1000.0,
    availability: float = 0.85,
    lifetime_yr: float = 30,
    n_mod: int = 1,
    construction_time_yr: float = 6.0,
    interest_rate: float = 0.07,
    **overrides,
) -> tuple[list[dict], dict]:
    """Run 1costingfe forward model and return fusion-backcasting-compatible data.

    Returns:
        (subsystems, financial_params) where:
        - subsystems: list of dicts matching fusion-backcasting Subsystem schema
        - financial_params: dict matching fusion-backcasting FinancialParams schema
    """
    concept_enum = ConfinementConcept(concept)
    fuel_enum = Fuel(fuel)
    family = CONCEPT_TO_FAMILY[concept_enum]

    model = CostModel(concept=concept_enum, fuel=fuel_enum)
    result = model.forward(
        net_electric_mw=net_electric_mw,
        availability=availability,
        lifetime_yr=lifetime_yr,
        n_mod=n_mod,
        construction_time_yr=construction_time_yr,
        interest_rate=interest_rate,
        **overrides,
    )

    c = result.costs
    pt = result.power_table

    # Get CAS22 sub-account detail from a fresh call
    from costingfe.layers.cas22 import cas22_reactor_plant_equipment
    from costingfe.layers.geometry import RadialBuild, compute_geometry
    from dataclasses import fields as dc_fields

    params = result.params
    rb_field_names = {f.name for f in dc_fields(RadialBuild)}
    rb_params = {k: params[k] for k in rb_field_names if k in params}
    rb = RadialBuild(**rb_params)
    geo = compute_geometry(rb, concept_enum)

    cas22 = cas22_reactor_plant_equipment(
        model.cc, pt.p_net, pt.p_th, pt.p_et, pt.p_fus,
        params["p_cryo"], n_mod, fuel_enum, True,
        blanket_vol=geo.firstwall_vol + geo.blanket_vol + geo.reflector_vol,
        shield_vol=geo.ht_shield_vol + geo.lt_shield_vol,
        structure_vol=geo.structure_vol,
        vessel_vol=geo.vessel_vol,
    )

    # O&M split: distribute CAS70 proportional to capital cost
    total_cas22 = float(cas22["C220000"])
    om_annual = float(c.cas70)
    cas22_om_share = om_annual * (total_cas22 / float(c.total_capital)) if c.total_capital > 0 else 0
    bop_om_share = om_annual - cas22_om_share

    def om_frac(sub_cost):
        """Proportional O&M share for a sub-account."""
        if total_cas22 <= 0:
            return 0
        return cas22_om_share * (float(sub_cost) / total_cas22)

    # Build subsystem list matching fusion-backcasting schema
    subsystems = [
        {
            "account": "22.1.1",
            "name": "First Wall/Blanket",
            "absolute_capital_cost": float(cas22["C220101"]),
            "absolute_fixed_om": om_frac(cas22["C220101"]),
            "variable_om": 0,
            "trl": 4,
            "idiot_index": 8.5,
            "description": f"Physics-based: {concept} {fuel} blanket ({float(cas22['C220101']):.0f} M$)",
        },
        {
            "account": "22.1.2",
            "name": "Neutron Shielding",
            "absolute_capital_cost": float(cas22["C220102"]),
            "absolute_fixed_om": om_frac(cas22["C220102"]),
            "variable_om": 0,
            "trl": 5,
            "idiot_index": 3.0,
            "description": f"Shield cost scaled by fuel neutron fraction",
        },
        {
            "account": "22.1.3",
            "name": "Magnets",
            "absolute_capital_cost": float(cas22["C220103"]),
            "absolute_fixed_om": om_frac(cas22["C220103"]),
            "variable_om": 0,
            "trl": 6,
            "idiot_index": 12.0,
            "description": "Superconducting magnets (TF + CS + PF)",
        },
        {
            "account": "22.1.5",
            "name": "Structural Support",
            "absolute_capital_cost": float(cas22["C220105"]),
            "absolute_fixed_om": om_frac(cas22["C220105"]),
            "variable_om": 0,
            "trl": 7,
            "idiot_index": 2.5,
            "description": "Primary structure (volume-based)",
        },
        {
            "account": "22.1.6",
            "name": "Vacuum Systems",
            "absolute_capital_cost": float(cas22["C220106"]),
            "absolute_fixed_om": om_frac(cas22["C220106"]),
            "variable_om": 0,
            "trl": 7,
            "idiot_index": 4.0,
            "description": "Vacuum vessel (volume-based)",
        },
        {
            "account": "22.1.7",
            "name": "Power Supplies",
            "absolute_capital_cost": float(cas22["C220107"]),
            "absolute_fixed_om": om_frac(cas22["C220107"]),
            "variable_om": 0,
            "trl": 7,
            "idiot_index": 3.5,
            "description": "Magnet power supplies and heating systems",
        },
        {
            "account": "22.1.8",
            "name": "Laser/Driver",
            "absolute_capital_cost": float(cas22["C220104"]),
            "absolute_fixed_om": om_frac(cas22["C220104"]),
            "variable_om": 0,
            "trl": 5,
            "idiot_index": 18.0,
            "description": "Supplementary heating / driver systems",
        },
        {
            "account": "22.1.9",
            "name": "Direct Energy Conversion",
            "absolute_capital_cost": float(cas22["C220109"]),
            "absolute_fixed_om": om_frac(cas22["C220109"]),
            "variable_om": 0,
            "trl": 3,
            "idiot_index": 15.0,
            "description": "Direct energy converter (if used)",
        },
        {
            "account": "22.5",
            "name": "Fuel Handling",
            "absolute_capital_cost": float(cas22["C220500"]) + float(cas22["C220112"]),
            "absolute_fixed_om": om_frac(cas22["C220500"]) + om_frac(cas22["C220112"]),
            "variable_om": 0,
            "trl": 5,
            "idiot_index": 10.0,
            "description": f"Fuel handling + isotope separation ({fuel})",
        },
        {
            "account": "23",
            "name": "Turbine Plant",
            "absolute_capital_cost": float(c.cas23),
            "absolute_fixed_om": bop_om_share * 0.4,
            "variable_om": 0.5,
            "trl": 9,
            "idiot_index": 2.0,
            "description": "Steam turbine and generator",
        },
        {
            "account": "24-26",
            "name": "Balance of Plant",
            "absolute_capital_cost": float(c.cas24) + float(c.cas25) + float(c.cas26),
            "absolute_fixed_om": bop_om_share * 0.6,
            "variable_om": 0.3,
            "trl": 9,
            "idiot_index": 1.5,
            "description": "Electrical, misc, heat rejection",
        },
    ]

    # Round all costs
    for s in subsystems:
        s["absolute_capital_cost"] = round(s["absolute_capital_cost"], 1)
        s["absolute_fixed_om"] = round(s["absolute_fixed_om"], 1)

    financial_params = {
        "wacc": interest_rate,
        "lifetime": int(lifetime_yr),
        "capacity_factor": availability,
        "capacity_mw": net_electric_mw,
        "construction_time": int(construction_time_yr),
    }

    return subsystems, financial_params


def generate_subsystems_json(
    concept: str = "tokamak",
    fuel: str = "dt",
    **kwargs,
) -> dict:
    """Return fusion-backcasting-compatible JSON structure.

    Can be used to replace default_subsystems.json or passed to the API.
    """
    subsystems, financial = generate_subsystems(concept, fuel, **kwargs)
    family = CONCEPT_TO_FAMILY[ConfinementConcept(concept)]

    return {
        "subsystems": subsystems,
        "financial_params": financial,
        "fuel_type": _FUEL_MAP[Fuel(fuel)],
        "confinement_type": _FAMILY_TO_CONFINEMENT[family],
        "reference_capacity_mw": kwargs.get("net_electric_mw", 1000.0),
        "source": "1costingfe",
        "concept": concept,
        "fuel": fuel,
    }
