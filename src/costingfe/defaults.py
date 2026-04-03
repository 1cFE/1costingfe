"""Load and manage default parameters from YAML files."""

from dataclasses import dataclass, fields, replace
from pathlib import Path

import yaml

from costingfe.types import PowerCycle

_DATA_DIR = Path(__file__).parent / "data" / "defaults"


@dataclass(frozen=True)
class CostingConstants:
    """All costing coefficients. Immutable — use .replace() for overrides."""

    # CAS10
    site_permits: float = 3.0
    plant_studies_foak: float = 20.0
    plant_studies_noak: float = 4.0
    plant_permits: float = 2.0
    plant_reports: float = 1.0
    other_precon: float = 1.0
    land_intensity: float = 0.25  # acres/MWe (CFS ARC: 100 acres / 400 MWe)
    land_cost: float = 10000.0  # $/acre (industrial-zoned US average)
    licensing_cost_dt: float = 5.0
    licensing_cost_dd: float = 3.0
    licensing_cost_dhe3: float = 1.0
    licensing_cost_pb11: float = 0.1
    # Licensing times per DI-015/016 regulatory framework research
    licensing_time_dt: float = 2.0  # Part 30, 1-2yr range
    licensing_time_dd: float = 1.5  # Reduced tritium, 6-18mo range
    licensing_time_dhe3: float = 0.75  # Minimal radioactivity, 6-12mo
    licensing_time_pb11: float = 0.0  # No NRC jurisdiction

    # CAS22 — Reactor Plant Equipment
    # 220101: First Wall + Blanket — volume-based unit costs (M$/m³)
    # Calibrated against pyFECONs at reference tokamak geometry
    # (R=6.2m, a=2.0m, κ=1.7, blanket_t=0.7m → ~1018 m³ assembly)
    blanket_unit_cost_dt: float = 0.60  # Full breeding blanket (TBR>1.05)
    blanket_unit_cost_dd: float = 0.30  # Energy capture, no breeding
    blanket_unit_cost_dhe3: float = 0.08  # Minimal X-ray + ~5% neutron
    blanket_unit_cost_pb11: float = 0.05  # Minimal X-ray only

    # 220102: Shield — volume-based unit cost (M$/m³)
    # Calibrated at reference shield volume ~516 m³
    shield_unit_cost: float = 0.74  # M$/m³, DT reference

    # 220103-220108: Reactor components
    # 220104: Supplementary Heating — per-MW linear costs (M$/MW, 2023$)
    # Source: pyFECONs cas220104 / ARIES + ITER average costs
    heating_nbi_per_mw: float = 7.0642  # Neutral Beam Injection
    heating_icrf_per_mw: float = 4.1494  # Ion Cyclotron Resonance Frequency
    heating_ecrh_per_mw: float = 5.0  # Electron Cyclotron Resonance Heating (gyrotrons)
    heating_lhcd_per_mw: float = 4.0  # Lower Hybrid Current Drive (klystrons)
    # 220105: Primary Structure — volume-based (M$/m³)
    structure_unit_cost: float = 0.15  # Calibrated at ~208 m³
    # 220106: Vacuum System — volume-based (M$/m³)
    vessel_unit_cost: float = 0.72  # Calibrated at ~148 m³
    power_supplies_base: float = 80.0
    divertor_base: float = 60.0
    # IFE/MIF target factory capital (M$ at 1 GWe reference)
    target_factory_base: float = 244.0

    # C220109: DEC add-on for linear devices
    # Source: docs/account_justification/CAS220109_direct_energy_converter.md
    # Subsystem build-up: grids + power conditioning + incremental vacuum/tank
    dec_base: float = 100.0  # M$ at 400 MWe DEC electric output (P_DEE_REF)
    dec_grid_cost: float = 12.0  # M$ replaceable grid/collector modules at P_DEE_REF

    # DEC grid lifetime (FPY) — HIGH UNCERTAINTY, no reactor-scale data.
    # Conservative estimates. Primary degradation: sputtering + He blistering
    # from charged particle exhaust. Neutron damage additive for DT/DD.
    # Sensitivity range: 0.5x to 3x these values.
    dec_grid_lifetime_dt: float = 2.0  # Sputtering + 14.1 MeV neutron damage
    dec_grid_lifetime_dd: float = 3.0  # Sputtering + 2.45 MeV neutron damage
    dec_grid_lifetime_dhe3: float = 4.0  # 14.7 MeV proton sputtering + He blistering
    dec_grid_lifetime_pb11: float = (
        3.0  # 2.9 MeV alpha sputtering + severe He blistering
    )

    # 220110: Remote Handling & Maintenance Equipment (M$ at 1 GWe, tokamak ref)
    # See docs/account_justification/CAS220110_remote_handling.md
    remote_handling_dt_base: float = 150.0
    remote_handling_dd_base: float = 100.0
    remote_handling_dhe3_base: float = 30.0
    remote_handling_pb11_base: float = 20.0

    # 220111: Installation labor (fraction of reactor subtotal)
    installation_frac: float = 0.14

    # Core component lifetime (FPY — full power years between replacements)
    # Source: 20260208-fusion-reactor-subsystems-by-fuel-type.md
    core_lifetime_dt: float = 5.0  # 5-10 FPY, ~20 dpa/yr
    core_lifetime_dd: float = 10.0  # 10-15 FPY, ~7 dpa/yr
    core_lifetime_dhe3: float = 30.0  # 30+ FPY, ~1 dpa/yr
    core_lifetime_pb11: float = 50.0  # 50+ FPY, ~0.1 dpa/yr

    # CAS22 sub-accounts that need periodic replacement (neutron/thermal damage)
    # Default: blanket/FW + divertor. Extend to include "C220103" (coils) for
    # designs with insufficient HTS shielding.
    replaceable_accounts: tuple = ("C220101", "C220108")

    # 220500: Fuel Handling (M$ at 1 GWe reference)
    fuel_handling_dt_base: float = 120.0  # Full tritium processing
    fuel_handling_dd_base: float = 60.0  # Small-scale tritium + deuterium
    fuel_handling_dhe3_base: float = 40.0  # He-3 handling
    fuel_handling_pb11_base: float = 15.0  # Boron powder injection

    # CAS21 — per-building, per-fuel costs (M$ at 1 GWe reference)
    # Each entry has fuel keys (dt/dd/dhe3/pb11 or 'all') + 'scales' key
    building_costs: dict[str, dict] = None  # loaded from YAML

    # CAS27 — Special materials: initial reactor material inventory (M$ at 1 GWe)
    # Default assumes PbLi blanket concept for DT (~4,500 tonnes PbLi @ $3/kg
    # + enriched Li top-up). HCPB concepts with Be multiplier override to ~$200M.
    # See docs/account_justification/CAS27_special_materials.md
    special_materials_dt: float = 15.0  # PbLi fill + enriched Li for breeding
    special_materials_dd: float = 2.0  # Conventional coolant fills only
    special_materials_dhe3: float = 1.0  # Minimal special materials
    special_materials_pb11: float = 0.0  # No special materials (conventional)

    # CAS23-26 — BOP equipment (M$/MW gross electric, 2024$)
    # Source: ARIES/NETL calibration (2019$ × 1.22 CPI inflation)
    # See docs/account_justification/CAS23_26_balance_of_plant.md
    turbine_per_mw: float = 0.19764  # Steam TG, condenser, feedwater
    electric_per_mw: float = 0.08418  # Switchyard, transformers, cabling
    misc_per_mw: float = 0.05124  # Fire protection, compressed air, HVAC
    heat_rej_per_mw: float = 0.03416  # Cooling towers, circ water

    # CAS28 — Digital twin (M$, fixed)
    # Source: NtTau Digital LTD estimate (pyFECONS)
    digital_twin: float = 5.0

    # CAS29 — Contingency on direct costs (Gen-IV EMWG convention)
    contingency_rate_foak: float = 0.10
    contingency_rate_noak: float = 0.0

    # CAS30
    indirect_fraction: float = 0.20
    reference_construction_time: float = 6.0

    # CAS40 — Capitalized owner's costs (M$ at 1 GWe reference, 2023$)
    # Source: CAS40_capitalized_owners_costs.md — INL methodology on CAS71-73 staffing
    owner_cost_dt: float = 39.0  # 117 staff, full neutron + tritium pre-op training
    owner_cost_dd: float = 31.0  # 94 staff, reduced tritium scope
    owner_cost_dhe3: float = 23.0  # 69 staff, light HP program
    owner_cost_pb11: float = 20.0  # 59 staff, near-industrial, RSO-only

    # CAS50 — Capitalized supplementary costs
    # Source: CAS50_supplementary_costs.md — sub-account model
    shipping_frac: float = 0.015  # fraction of CAS20 (WNA ~2%, discounted for fusion)
    spare_parts_frac_dt: float = (
        0.03  # fraction of CAS22-28, activated component spares
    )
    spare_parts_frac_dd: float = 0.025
    spare_parts_frac_dhe3: float = 0.015
    spare_parts_frac_pb11: float = 0.01  # conventional industrial spares only
    tax_frac: float = 0.01  # fraction of CAS20, after typical energy project exemptions
    construction_insurance_frac: float = (
        0.015  # fraction of (CAS20+CAS30), builder's risk
    )
    startup_fuel_dt: float = 40.0  # M$ at 1 GWe — ~1.3 kg tritium at $30k/g
    startup_fuel_dd: float = 0.1  # M$ at 1 GWe — deuterium, commodity
    startup_fuel_dhe3: float = 10.0  # M$ at 1 GWe — He3, supply-constrained
    startup_fuel_pb11: float = 0.1  # M$ at 1 GWe — H + B11, industrial commodities
    decom_provision_dt: float = 127.0  # M$ at 1 GWe — PV of $410M over 40yr at 3%
    decom_provision_dd: float = 93.0  # M$ at 1 GWe — PV of $300M
    decom_provision_dhe3: float = 65.0  # M$ at 1 GWe — PV of $210M
    decom_provision_pb11: float = 53.0  # M$ at 1 GWe — PV of $170M

    # CAS70 — Annual O&M cost (M$/yr at 1 GWe reference, 2023$)
    # Source: CAS70_staffing_and_om_costs.md — staffing-based build-up by fuel type
    # Power-law scaling: annual_om = om_cost(fuel) * (P_net / 1 GWe)^0.5
    om_cost_dt: float = 52.0  # Full neutron + tritium operational overhead
    om_cost_dd: float = 39.0  # ~1/3 DT neutron flux, smaller tritium inventory
    om_cost_dhe3: float = 26.0  # ~5% neutron fraction, minimal tritium
    om_cost_pb11: float = 24.0  # Aneutronic, no tritium, RSO-only

    # CAS80 — fuel isotope unit costs ($/kg)
    # STARFIRE (1980) inflation-adjusted via GDP IPD. Range: $1,500-3,500/kg.
    u_deuterium: float = 2175.0  # $/kg
    u_li6: float = 1000.0  # $/kg, enriched Li-6 (90%) for breeding blanket
    u_he3: float = 2_000_000.0  # $/kg, He-3 ($2,000/g — optimistic self-production)
    u_protium: float = 5.0  # $/kg, commodity H2
    u_b11: float = 10_000.0  # $/kg, FOAK enriched B-11 (no industrial supply)
    u_b11_noak: float = (
        75.0  # $/kg, NOAK B-11 (industrial chemical exchange distillation)
    )

    # CAS80 — fuel utilization
    burn_fraction: float = (
        0.05  # Fraction of injected fuel that undergoes fusion per pass
    )
    fuel_recovery: float = 0.95  # Fraction of unburned fuel recovered and recycled

    def replace(self, **kwargs):
        return replace(self, **kwargs)

    def owner_cost(self, fuel):
        """Pre-operational owner's cost (M$ at 1 GWe ref) for a given fuel type."""
        from costingfe.types import Fuel

        return {
            Fuel.DT: self.owner_cost_dt,
            Fuel.DD: self.owner_cost_dd,
            Fuel.DHE3: self.owner_cost_dhe3,
            Fuel.PB11: self.owner_cost_pb11,
        }.get(fuel, self.owner_cost_dt)

    def om_cost(self, fuel):
        """Annual O&M cost (M$/yr at 1 GWe reference) for a given fuel type."""
        from costingfe.types import Fuel

        return {
            Fuel.DT: self.om_cost_dt,
            Fuel.DD: self.om_cost_dd,
            Fuel.DHE3: self.om_cost_dhe3,
            Fuel.PB11: self.om_cost_pb11,
        }.get(fuel, self.om_cost_dt)

    def licensing_cost(self, fuel):
        from costingfe.types import Fuel

        return {
            Fuel.DT: self.licensing_cost_dt,
            Fuel.DD: self.licensing_cost_dd,
            Fuel.DHE3: self.licensing_cost_dhe3,
            Fuel.PB11: self.licensing_cost_pb11,
        }.get(fuel, self.licensing_cost_dt)

    def licensing_time(self, fuel):
        from costingfe.types import Fuel

        return {
            Fuel.DT: self.licensing_time_dt,
            Fuel.DD: self.licensing_time_dd,
            Fuel.DHE3: self.licensing_time_dhe3,
            Fuel.PB11: self.licensing_time_pb11,
        }.get(fuel, self.licensing_time_dt)

    def core_lifetime(self, fuel):
        """Core component lifetime in FPY for a given fuel type."""
        from costingfe.types import Fuel

        return {
            Fuel.DT: self.core_lifetime_dt,
            Fuel.DD: self.core_lifetime_dd,
            Fuel.DHE3: self.core_lifetime_dhe3,
            Fuel.PB11: self.core_lifetime_pb11,
        }.get(fuel, self.core_lifetime_dt)

    def dec_grid_lifetime(self, fuel):
        """DEC grid replacement interval in FPY for a given fuel type."""
        from costingfe.types import Fuel

        return {
            Fuel.DT: self.dec_grid_lifetime_dt,
            Fuel.DD: self.dec_grid_lifetime_dd,
            Fuel.DHE3: self.dec_grid_lifetime_dhe3,
            Fuel.PB11: self.dec_grid_lifetime_pb11,
        }.get(fuel, self.dec_grid_lifetime_dt)

    def spare_parts_frac(self, fuel):
        """Initial spare parts fraction of CAS22-28 for a given fuel type."""
        from costingfe.types import Fuel

        return {
            Fuel.DT: self.spare_parts_frac_dt,
            Fuel.DD: self.spare_parts_frac_dd,
            Fuel.DHE3: self.spare_parts_frac_dhe3,
            Fuel.PB11: self.spare_parts_frac_pb11,
        }.get(fuel, self.spare_parts_frac_dt)

    def startup_fuel(self, fuel):
        """Startup fuel/inventory cost (M$ at 1 GWe ref) for a given fuel type."""
        from costingfe.types import Fuel

        return {
            Fuel.DT: self.startup_fuel_dt,
            Fuel.DD: self.startup_fuel_dd,
            Fuel.DHE3: self.startup_fuel_dhe3,
            Fuel.PB11: self.startup_fuel_pb11,
        }.get(fuel, self.startup_fuel_dt)

    def decom_provision(self, fuel):
        """Decommissioning provision (M$ at 1 GWe ref) for a given fuel type."""
        from costingfe.types import Fuel

        return {
            Fuel.DT: self.decom_provision_dt,
            Fuel.DD: self.decom_provision_dd,
            Fuel.DHE3: self.decom_provision_dhe3,
            Fuel.PB11: self.decom_provision_pb11,
        }.get(fuel, self.decom_provision_dt)

    def contingency_rate(self, noak):
        return self.contingency_rate_noak if noak else self.contingency_rate_foak


def cc_float_fields() -> list[str]:
    """Return names of all float fields on CostingConstants."""
    return [
        f.name for f in fields(CostingConstants) if f.type == "float" or f.type is float
    ]


def load_costing_constants(path: Path = None) -> CostingConstants:
    """Load costing constants from YAML, falling back to dataclass defaults."""
    if path is None:
        path = _DATA_DIR / "costing_constants.yaml"
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f)
        valid_fields = {f.name for f in fields(CostingConstants)}
        return CostingConstants(**{k: v for k, v in data.items() if k in valid_fields})
    return CostingConstants()


def load_engineering_defaults(concept_fuel: str) -> dict:
    """Load engineering defaults for a concept (e.g., 'mfe_tokamak')."""
    path = _DATA_DIR / f"{concept_fuel}.yaml"
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


POWER_CYCLE_DEFAULTS: dict[PowerCycle, dict[str, float]] = {
    PowerCycle.RANKINE: {
        "eta_th": 0.40,
        "turbine_per_mw": 0.19764,
        "heat_rej_per_mw": 0.03416,
    },
    PowerCycle.BRAYTON_SCO2: {
        "eta_th": 0.47,
        "turbine_per_mw": 0.155,
        "heat_rej_per_mw": 0.022,
    },
    PowerCycle.COMBINED: {
        "eta_th": 0.53,
        "turbine_per_mw": 0.235,
        "heat_rej_per_mw": 0.018,
    },
}
