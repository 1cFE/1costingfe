"""Example: D-He3 Pulsed Colliding FRC — Helion-like concept.

Models a pulsed colliding Field-Reversed Configuration power plant operating
on D-He3 fuel with direct electromagnetic energy recovery.

Architecture:
  - Modular array of factory-built pulsed FRC generators (~50 MWe each)
  - Direct inductive energy recovery (no steam cycle)
  - Normal-conducting copper coils (no superconductors, no cryogenics)
  - D-He3 fuel with ~5% neutron fraction from DD side reactions
"""

from costingfe import CostModel, Fuel
from costingfe.defaults import load_costing_constants
from costingfe.types import ConfinementConcept, PulsedConversion

cc = load_costing_constants().replace(
    burn_fraction=0.10,
    fuel_recovery=0.95,
)
model = CostModel(
    concept=ConfinementConcept.MAG_TARGET,
    fuel=Fuel.DHE3,
    pulsed_conversion=PulsedConversion.INDUCTIVE_DEC,
    costing_constants=cc,
)

N_MODULES = 20
NET_ELECTRIC_MW = 1000.0

result = model.forward(
    net_electric_mw=NET_ELECTRIC_MW,
    availability=0.85,
    lifetime_yr=30,
    n_mod=N_MODULES,
    construction_time_yr=4.0,
    interest_rate=0.07,
    inflation_rate=0.02,
    noak=True,
    # Pulsed parameters
    e_driver_mj=12.0,
    f_rep=1.0,
    eta_pin=0.95,
    eta_dec=0.85,
    f_pdv=0.80,
    eta_th=0.0,  # no thermal BOP (pure DEC, D-He3)
    mn=1.0,  # no breeding blanket
    p_cryo=0.0,  # copper coils
    p_target=0.0,  # in-situ FRC formation
    p_coils=0.5,
    # Geometry (cylindrical, per module)
    R0=0.0,
    plasma_t=0.5,
    blanket_t=0.05,
    ht_shield_t=0.05,
    structure_t=0.10,
    vessel_t=0.10,
)

# Results
c = result.costs
pt = result.power_table

print(
    f"D-He3 Pulsed Colliding FRC — {N_MODULES} modules x "
    f"{NET_ELECTRIC_MW / N_MODULES:.0f} MWe"
)
print(f"  {NET_ELECTRIC_MW:.0f} MWe net, 85% availability, 30 yr lifetime")
print()
lcoe_ckwh = float(c.lcoe) / 10
print(
    f"LCOE: {c.lcoe:.1f} $/MWh ({lcoe_ckwh:.2f} c/kWh)"
    f" | Overnight: {c.overnight_cost:.0f} $/kW"
)
print(f"Fusion: {pt.p_fus:.0f} MW | Net: {pt.p_net:.0f} MW | Q_eng: {pt.q_eng:.1f}")
print(f"Recirculating fraction: {pt.rec_frac:.1%}")
print(f"Scientific Q (P_fus/P_driver): {pt.q_sci:.1f}")
print(f"E_stored: {pt.e_stored_mj:.1f} MJ/pulse | f_ch: {pt.f_ch:.2f}")
print()

# Cost breakdown
cas = [
    ("CAS10", "Preconstruction", c.cas10),
    ("CAS21", "Buildings", c.cas21),
    ("CAS22", "Reactor Plant Equipment", c.cas22),
    ("CAS23", "Turbine Plant", c.cas23),
    ("CAS24", "Electrical Plant", c.cas24),
    ("CAS25", "Miscellaneous", c.cas25),
    ("CAS26", "Heat Rejection", c.cas26),
    ("CAS28", "Digital Twin", c.cas28),
    ("CAS29", "Contingency", c.cas29),
    ("CAS30", "Indirect Costs", c.cas30),
    ("CAS40", "Owner's Costs", c.cas40),
    ("CAS50", "Supplementary", c.cas50),
    ("CAS60", "IDC", c.cas60),
    ("CAS70", "O&M (annualized)", c.cas70),
    ("CAS80", "Fuel (annualized)", c.cas80),
    ("CAS90", "Financial", c.cas90),
]

print(f"{'Code':<8} {'Account':<28} {'M$':>10}")
print("-" * 48)
for code, name, val in cas:
    print(f"{code:<8} {name:<28} {float(val):>10.1f}")
print("-" * 48)
print(f"{'':8} {'Total Capital':<28} {float(c.total_capital):>10.1f}")

# CAS22 sub-account detail
print(f"\nCAS22 Reactor Plant Equipment (per-module x {N_MODULES} + plant-wide):")
print("-" * 56)
per_mod_keys = [
    ("C220101", "First Wall / Blanket"),
    ("C220102", "Shield"),
    ("C220103", "Coils (Cu pulsed)"),
    ("C220104", "Supplementary Heating"),
    ("C220105", "Primary Structure"),
    ("C220106", "Vacuum System"),
    ("C220107", "Power Supplies (cap bank)"),
    ("C220108", "Target Factory"),
    ("C220109", "Direct Energy Converter"),
    ("C220110", "Remote Handling"),
    ("C220111", "Installation"),
]
plant_keys = [
    ("C220200", "Coolant Systems"),
    ("C220300", "Aux Cooling + Cryo"),
    ("C220400", "Rad Waste"),
    ("C220500", "Fuel Handling (He-3)"),
    ("C220600", "Other Equipment"),
    ("C220700", "I&C"),
]

per_mod_total = 0.0
for key, label in per_mod_keys:
    v = float(result.cas22_detail.get(key, 0.0))
    per_mod_total += v
    if v > 0.01:
        tot = v * N_MODULES
        print(f"  {key} {label:<26} {v:>8.1f} M$/mod  x{N_MODULES} = {tot:>8.1f}")
print(
    f"  {'':7} {'Per-module subtotal':<26}"
    f" {per_mod_total:>8.1f} M$/mod  x{N_MODULES} = {per_mod_total * N_MODULES:>8.1f}"
)
print()

plant_total = 0.0
for key, label in plant_keys:
    v = float(result.cas22_detail.get(key, 0.0))
    plant_total += v
    if v > 0.01:
        print(f"  {key} {label:<26} {v:>8.1f} M$ (plant-wide)")
print(f"  {'':7} {'Plant-wide subtotal':<26} {plant_total:>8.1f} M$")
print(f"\n  {'':7} {'CAS22 Total':<26} {float(c.cas22):>8.1f} M$")

print(f"\nOverridden: {', '.join(result.overridden) or 'None'}")
