"""Example: DT Modular HTS Stellarator — cost breakdown and sensitivity analysis.

Models a compact, quasi-axisymmetric stellarator power plant using
high-temperature superconducting (REBCO) non-planar modular coils, as
described in HTS_Stellarator_LCOE_Writeup.md.

Architecture:
  - Quasi-axisymmetric optimized magnetic geometry (W7-X heritage)
  - Non-planar HTS (REBCO) modular coils at 15-20 T peak field
  - Island divertor for steady-state plasma exhaust
  - Inherent steady-state operation — no plasma current drive needed
  - No disruptions, no ELMs — simplified first wall design
  - Modular blanket cassettes for remote handling

Key stellarator differences from tokamak (CAS22 coil model):
  - markup=12x (vs 8x tokamak): non-planar 3D winding of REBCO tape
    with tight bend-radius management, compound curvatures, and strict
    field-quality tolerances to preserve quasi-symmetry
  - path_factor=2x: non-planar coil winding path is ~2x longer per
    coil than a comparable planar TF coil at the same major radius
  - b_max=18T (vs 12T tokamak): HTS REBCO enables higher peak field
  - Heating costed as ECRH at $5/MW (vs NBI at $7.06/MW for tokamak)
  - Zero current drive power: p_input is pure auxiliary heating for
    startup and profile control, not sustaining a plasma current

Modeling notes:
  - R0=5.5 m, a=1.8 m targets a mid-range compact HTS stellarator
    (compare W7-X at R0=5.5 m, a=0.53 m with LTS coils at ~3T;
    HTS at 15-20 T enables comparable fusion power in similar R0
    with much larger plasma cross-section)
  - elon=1.6: effective average elongation approximating the
    toroidally-varying bean/triangular/elliptical cross-sections
    typical of a quasi-axisymmetric stellarator (the model uses a
    single elongation parameter; real stellarator cross-sections
    vary from bean-shaped to triangular around each field period)
  - p_input=30 MW: ECRH-dominated heating for startup and profile
    control only — no current drive requirement (costed via p_ecrh
    in the stellarator YAML defaults)
  - availability=0.90: steady-state operation + modular maintenance
    enables higher availability than pulsed tokamak concepts
  - construction_time_yr=8: complex 3D coil fabrication and assembly
    with stringent field-quality verification adds schedule (from YAML)

  Most power balance and geometry parameters use the stellarator YAML
  defaults (mfe_stellarator.yaml). Only availability and elon are
  overridden here — elon because the default (1.0) is too low for
  a realistic quasi-axisymmetric design.
"""

from costingfe import ConfinementConcept, CostModel, Fuel

model = CostModel(concept=ConfinementConcept.STELLARATOR, fuel=Fuel.DT)
result = model.forward(
    net_electric_mw=1000.0,
    availability=0.90,  # Steady-state + modular maintenance advantage
    lifetime_yr=30,
    n_mod=1,
    interest_rate=0.07,
    inflation_rate=0.02,
    noak=True,
    # Override: effective avg elongation for bean-shaped cross-sections.
    # Default is 1.0 which understates the toroidally-varying stellarator
    # cross-section. All other params use mfe_stellarator.yaml defaults.
    elon=1.6,
)

# ── Results ───────────────────────────────────────────────────────────
c = result.costs
pt = result.power_table

print("DT Modular HTS Stellarator — 1 GWe, 90% availability, 30 yr lifetime")
lcoe_ckwh = float(c.lcoe) / 10
print(
    f"LCOE: {c.lcoe:.1f} $/MWh ({lcoe_ckwh:.2f} c/kWh)"
    f" | Overnight: {c.overnight_cost:.0f} $/kW"
)
print(f"Fusion: {pt.p_fus:.0f} MW | Net: {pt.p_net:.0f} MW | Q_eng: {pt.q_eng:.1f}")
print(f"Recirculating fraction: {pt.rec_frac:.1%}")
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
    ("CAS27", "Special Materials", c.cas27),
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
print("\nCAS22 Reactor Plant Equipment Breakdown:")
print("-" * 48)
for k, v in sorted(result.cas22_detail.items()):
    if float(v) > 0:
        print(f"  {k:<28} {float(v):>10.1f} M$")

# ── Stellarator vs Tokamak comparison ─────────────────────────────────
print("\nStellarator Cost Drivers (vs Tokamak):")
print("-" * 48)
print("  Coil markup:           12x (vs 8x tokamak)")
print("  Coil path factor:      2.0 (non-planar 3D winding)")
print("  Peak coil field:       18 T (vs 12 T tokamak, HTS REBCO)")
print("  Coil cost multiplier:  4.5x vs tokamak (12/8 * 2.0 * 18/12)")
print("  Heating:               ECRH @ $5/MW (vs NBI @ $7.06/MW)")
print("  Cross-section:         bean/triangular (elon~1.6 effective avg)")
print("  Availability:          90% (steady-state, no disruptions)")
print("  Current drive power:   0 MW (inherent steady-state)")
print("  Construction time:     8 yr (3D coil assembly complexity)")
print("  Auxiliary heating:     30 MW (ECRH only, profile control)")

# ── Sensitivity Analysis ─────────────────────────────────────────────
sens = model.sensitivity(result.params)

print("\nSensitivity (elasticity = %LCOE / %param)")
print("-" * 48)

print("\nEngineering levers:")
for k, v in sorted(sens["engineering"].items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {k:<36} {v:+.4f}")

print("\nFinancial:")
for k, v in sorted(sens["financial"].items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {k:<36} {v:+.4f}")

print("\nCosting constants (top 15):")
costing = sorted(sens["costing"].items(), key=lambda x: abs(x[1]), reverse=True)
for k, v in costing[:15]:
    print(f"  {k:<36} {v:+.4f}")
