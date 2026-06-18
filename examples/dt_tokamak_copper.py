"""Example: DT Tokamak with copper coils.

Demonstrates that `coil_material` and related coil/cryo parameters are
editable via `model.forward(**overrides)`. The default tokamak YAML
(steady_state_tokamak.yaml) sets `coil_material: rebco_hts` implicitly;
here we override it to copper and adjust the coupled parameters that a
resistive-magnet machine implies:

- b_max drops (copper-limited, no HTS critical-current envelope)
- p_coils rises sharply (ohmic dissipation in resistive windings)
- p_cryo drops to ~0 (no superconductor to keep at <77 K)
"""

from costingfe import ConfinementConcept, CostModel, Fuel

model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
result = model.forward(
    net_electric_mw=1000.0,
    availability=0.85,
    lifetime_yr=30,
    n_mod=1,
    construction_time_yr=6.0,
    interest_rate=0.07,
    inflation_rate=0.0245,
    noak=True,
    # Geometry (CATF spherical-tokamak reference)
    R0=3.0,
    elon=3.0,
    plasma_t=1.1,
    blanket_t=0.8,
    ht_shield_t=0.2,
    structure_t=0.2,
    vessel_t=0.2,
    # Coil configuration -- THIS IS THE EDIT
    coil_material="copper",  # "rebco_hts" | "nb3sn" | "nbti" | "copper"
    b_center=7.0,  # Field at coil center [T], resistive copper limit
    r_bore=1.85,  # Effective winding bore radius [m]
    # Power balance adjusted for resistive coils
    p_input=50.0,
    mn=1.1,
    eta_th=0.46,
    eta_p=0.5,
    # eta_pin is derived from eta_source x eta_couple for a heated concept;
    # override eta_couple, not eta_pin, to change injector wall-plug efficiency.
    eta_couple=0.8333,
    eta_de=0.85,
    f_sub=0.03,
    f_dec=0.0,
    p_coils=120.0,  # MW -- resistive dissipation, was 2 MW for HTS
    p_cool=13.7,
    p_pump=1.0,
    p_trit=10.0,
    p_house=4.0,
    p_cryo=0.0,  # No SC cryogenics
)

c = result.costs
pt = result.power_table

print("DT Tokamak (copper coils) -- 1 GWe, 85% availability, 30 yr")
print(f"LCOE: {c.lcoe:.1f} $/MWh | Overnight: {c.capital_per_kw:.0f} $/kW")
print(f"Fusion: {pt.p_fus:.0f} MW | Net: {pt.p_net:.0f} MW | Q_eng: {pt.q_eng:.2f}")
print()

cas = [
    ("CAS22", "Reactor Plant Equipment", c.cas22),
    ("CAS22.01.01", "Magnets (in CAS22)", None),  # see breakdown below
]
print(f"{'Code':<14} {'Account':<28} {'M$':>10}")
print("-" * 54)
for code, name, val in cas:
    if val is not None:
        print(f"{code:<14} {name:<28} {float(val):>10.1f}")
print(f"{'Total capital':<14} {'':<28} {float(c.total_capital):>10.1f}")

# Magnet sub-account breakdown (shows the copper cost factor at work)
detail = result.cas22_detail
print("\nCAS22 magnet/coil sub-accounts (M$):")
for k in sorted(detail):
    if k.startswith("C2201"):
        print(f"  {k:<10} {float(detail[k]):>8.2f}")
