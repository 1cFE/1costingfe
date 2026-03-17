"""Example: D-He3 Pulsed Colliding FRC — Helion-like concept.

Models a pulsed colliding Field-Reversed Configuration power plant operating
on D-He3 fuel with direct electromagnetic energy recovery, as described in
Pulsed_FRC_LCOE_Writeup.md.

Architecture:
  - Modular array of factory-built pulsed FRC generators (~50 MWe each)
  - Direct EM energy recovery (no steam cycle) at >90% round-trip efficiency
  - Normal-conducting copper coils (no superconductors, no cryogenics)
  - D-He3 fuel with ~5% neutron fraction from DD side reactions
  - Sub-ignition operation: high circuit efficiency replaces high Q requirement

Modeling approach:
  Uses the MIF (magneto-inertial fusion) power balance with eta_th=0.90 to
  represent the direct electromagnetic conversion efficiency. f_dec is left
  at 0 (MIF has no DEC pathway); instead, the 90% "thermal" efficiency
  mimics the round-trip circuit efficiency of the expanding-FRC energy
  recovery mechanism. This is an approximation: in reality, the ~5% neutron
  power deposits in the walls and would NOT be recovered at 90% by the EM
  pathway, but since the neutron fraction is small the error is minor.

  The MIF cost defaults assume IFE-like architecture (HTS coils, NBI heating,
  target factory). Extensive cost_overrides are required to represent the
  FRC's copper pulsed coils, capacitor-bank-based driver, in-situ plasmoid
  formation (no target factory), and elimination of the steam cycle.

Key deviations from 1costingfe MIF defaults:
  Fuel utilization:
    - burn_fraction = 0.10 (vs 0.05 default) — FRC compression burn
    - fuel_recovery = 0.95 — efficient exhaust gas recycling

  Power balance:
    - eta_th = 0.90 (vs 0.40) — direct EM recovery, not thermal cycle
    - eta_pin = 0.95 (vs 0.30) — modern solid-state pulsed power
    - mn = 1.0 (vs 1.1) — no breeding blanket to multiply neutrons
    - p_cryo = 0.0 — copper coils, no superconductors
    - p_target = 0.0 — FRC plasmoids formed in-situ, not manufactured targets
    - p_trit = 0.5 MW — tritium monitoring (DD side reactions), not DT processing

  Cost overrides (per-module CAS22 sub-accounts):
    - C220103 = $5M — copper pulsed coils (vs $516M HTS w/ tokamak markup)
    - C220104 = $10M — capacitor bank + solid-state switches (vs $353M NBI)
    - C220107 = $3M — aux power supplies only (main driver is in C220104)
    - C220108 = $0 — no target factory (in-situ FRC formation from gas)
    - C220111 = $4M — installation labor (14% of adjusted reactor subtotal)

  Cost overrides (plant-wide):
    - CAS21 = $400M — no turbine hall, no cryogenics bldg, reduced hot cell,
      added capacitor bank storage + power electronics buildings
    - CAS23 = $0 — no steam turbine plant
    - CAS26 = $7M — reduced heat rejection (only ~10% conversion losses)
    - C220200 = $30M — minimal coolant (no steam loop; only coil cooling,
      neutron wall heating, and EM circuit losses)
"""

from costingfe import ConfinementConcept, CostModel, Fuel
from costingfe.defaults import load_costing_constants

cc = load_costing_constants().replace(
    burn_fraction=0.10,  # ~10% of injected fuel fuses per pulse
    fuel_recovery=0.95,  # 95% of unburned fuel recovered from exhaust
)
model = CostModel(
    concept=ConfinementConcept.MAG_TARGET, fuel=Fuel.DHE3, costing_constants=cc
)

# ── Plant configuration ──────────────────────────────────────────────
# 20 modules x 50 MWe = 1 GWe total net electric output.
# Each module is a bilateral colliding FRC generator with its own
# formation sections, acceleration coils, compression chamber,
# capacitor bank, and power electronics.

N_MODULES = 20
NET_ELECTRIC_MW = 1000.0  # 20 x 50 MWe

result = model.forward(
    net_electric_mw=NET_ELECTRIC_MW,
    availability=0.85,
    lifetime_yr=30,
    n_mod=N_MODULES,
    construction_time_yr=4.0,  # Factory-built modular assembly
    interest_rate=0.07,
    inflation_rate=0.02,
    noak=True,
    # ── Cost overrides ────────────────────────────────────────────
    #
    # The MIF CAS22 defaults assume HTS superconducting coils ($50/kAm
    # with 8x tokamak manufacturing markup), NBI heating systems ($7M/MW),
    # and an IFE-style target/liner factory ($244M base). None of these
    # apply to a pulsed colliding FRC with copper coils, capacitor-bank
    # driver, and in-situ plasmoid formation.
    #
    # Per-module CAS22 sub-accounts (multiplied by n_mod internally):
    #   C220103: Copper pulsed coils — formation, acceleration, and
    #            compression sections. Simple solenoidal geometry, Cu
    #            conductor at $1/kAm (vs HTS $50/kAm), low manufacturing
    #            markup (pulsed solenoids, not complex tokamak coils).
    #            Key uncertainty: fatigue life under repetitive pulsed
    #            loading at 20-40T compression field.
    #   C220104: Pulsed power driver — capacitor banks (energy storage),
    #            solid-state IGBT/GTO switches, bus work, and pulse
    #            transformers. This IS the energy investment and recovery
    #            system — the economic core of the concept.
    #   C220107: Auxiliary power supplies only — diagnostics, control,
    #            vacuum. The main pulsed driver is costed in C220104.
    #   C220108: No target factory — FRC plasmoids are formed in-situ
    #            from deuterium gas by field-reversed theta-pinch, not
    #            manufactured as physical targets.
    #   C220111: Installation labor at 14% of adjusted reactor subtotal.
    #
    # Plant-wide overrides:
    #   CAS21: Buildings — remove turbine hall ($54/kW), heat exchanger
    #          ($12/kW), cryogenics bldg ($15/kW). Reduce hot cell from
    #          $93/kW to ~$15/kW (DHe3 linear geometry, light activation).
    #          Add capacitor bank storage ($35/kW), power electronics hall
    #          ($25/kW). Net: ~$400/kW at ~1 GW gross.
    #   CAS23: No steam turbine plant — direct EM conversion.
    #   CAS26: Reduced heat rejection — only ~10% of thermal power is
    #          waste heat (vs ~55% for Rankine cycle).
    #   C220200: Minimal coolant system — no steam loop. Cooling needed
    #            only for copper coil resistive losses, neutron heating
    #            in first wall (~5% of p_fus), and EM circuit losses.
    #
    cost_overrides={
        # Per-module reactor equipment
        "C220103": 5.0,  # Copper pulsed coils [M$/module]
        "C220104": 10.0,  # Capacitor bank + switches [M$/module]
        "C220107": 3.0,  # Aux power supplies [M$/module]
        "C220108": 0.0,  # No target factory
        "C220111": 4.0,  # Installation (14% of ~$27M subtotal)
        # Plant-wide
        "C220200": 30.0,  # Minimal coolant (no steam loop) [M$]
        "CAS21": 400.0,  # Adjusted buildings [M$]
        "CAS23": 0.0,  # No turbine plant
        "CAS26": 7.0,  # Reduced heat rejection [M$]
    },
    # ── MIF power balance ─────────────────────────────────────────
    # p_driver: average pulsed power delivered to form, accelerate,
    # and compress FRC plasmoids. At ~1 Hz rep rate with ~12 MJ per
    # pulse, the time-averaged driver power is ~12 MW per module.
    p_driver=12.0,  # Pulsed driver output [MW average]
    mn=1.0,  # No neutron multiplier (no breeding blanket)
    eta_th=0.90,  # Direct EM recovery efficiency (not thermal)
    eta_p=0.5,  # Pumping efficiency
    eta_pin=0.95,  # Pulsed power wall-plug efficiency (solid-state)
    f_sub=0.03,  # Subsystem power fraction
    p_pump=0.5,  # Vacuum pumping [MW]
    p_trit=0.5,  # Tritium monitoring from DD side reactions [MW]
    p_house=2.0,  # Housekeeping [MW]
    p_cryo=0.0,  # No cryogenics (copper coils)
    p_target=0.0,  # No manufactured targets (in-situ FRC formation)
    p_coils=0.5,  # Formation/acceleration coils average power [MW]
    # ── Geometry (cylindrical, per module) ────────────────────────
    # Each module is a linear machine ~10m long with ~0.5m plasma radius.
    # Thin first wall (no breeding blanket), minimal shielding.
    R0=0.0,  # Linear geometry (no major radius)
    plasma_t=0.5,  # Plasma / chamber radius [m]
    blanket_t=0.05,  # Thin first wall only — no breeding blanket [m]
    ht_shield_t=0.05,  # Minimal shielding (~5% neutron fraction) [m]
    structure_t=0.10,  # Primary structure [m]
    vessel_t=0.10,  # Vacuum vessel [m]
)

# ── Results ───────────────────────────────────────────────────────────
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
print()

# ── Cost breakdown ────────────────────────────────────────────────────
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

# ── CAS22 sub-account detail ─────────────────────────────────────────
print(f"\nCAS22 Reactor Plant Equipment (per-module x {N_MODULES} + plant-wide):")
print("-" * 56)
per_mod_keys = [
    ("C220101", "First Wall / Blanket"),
    ("C220102", "Shield"),
    ("C220103", "Coils (Cu pulsed)"),
    ("C220104", "Pulsed Driver (cap bank)"),
    ("C220105", "Primary Structure"),
    ("C220106", "Vacuum System"),
    ("C220107", "Aux Power Supplies"),
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

# ── Overridden accounts ──────────────────────────────────────────────
print(f"\nOverridden: {', '.join(result.overridden)}")

# ── Key assumptions summary ──────────────────────────────────────────
print("\nKey Assumptions:")
print("-" * 48)
print(f"  Direct EM conversion eff:    {0.90:.0%} (eta_th proxy)")
print(f"  Pulsed power wall-plug eff:  {0.95:.0%}")
print(f"  Neutron multiplier:          {1.0} (no breeding blanket)")
print(f"  Modules:                     {N_MODULES}")
print("  Construction time:           4.0 yr (factory-built)")
print(f"  Burn fraction:               {cc.burn_fraction:.0%}")
print(f"  Fuel recovery:               {cc.fuel_recovery:.0%}")
print("  Cryogenics:                  None (copper coils)")
print("  Steam cycle:                 None (CAS23 = 0)")
print()
print("Modeling notes:")
print("  - eta_th=0.90 approximates direct EM recovery; the ~5% neutron")
print("    power fraction is also converted at 90% instead of ~40%")
print("    thermal, introducing a small upward bias in net electric.")
print("  - He-3 fuel cost uses market price ($2M/kg). The writeup notes")
print("    this does not reflect the amortized cost of the DD breeding")
print("    fleet required to produce He-3 at scale.")
print("  - Coil cost ($5M/mod) is a rough estimate for copper pulsed")
print("    coils. Key uncertainty: fatigue life under millions of")
print("    20-40T pulsed loading cycles per year.")
print("  - Capacitor bank cost ($10M/mod) assumes commercial-scale")
print("    solid-state pulsed power at ~12 MJ/pulse, 1 Hz rep rate.")

# ── Sensitivity Analysis ─────────────────────────────────────────────
sens = model.sensitivity(result.params)

print("\nSensitivity (elasticity = %LCOE / %param)")
print("-" * 48)

print("\nEngineering levers:")
for k, v in sorted(sens["engineering"].items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {k:<28} {v:+.4f}")

print("\nFinancial:")
for k, v in sorted(sens["financial"].items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {k:<28} {v:+.4f}")
