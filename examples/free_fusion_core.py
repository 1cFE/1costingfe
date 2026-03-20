"""Example: What if the fusion core were free?

Zeros out all CAS22 reactor plant equipment (magnets, blanket, heating,
divertor, vacuum, power supplies, remote handling, etc.) to find the
LCOE floor from balance-of-plant alone: buildings, turbines, electrical,
heat rejection, indirect costs, owner's costs, O&M, and financing.

This answers: "How cheap can fusion electricity get, even with a
magically free heat source?"
"""

from costingfe import ConfinementConcept, CostModel, Fuel

# ── Configuration ──────────────────────────────────────────────────
CONCEPTS = [
    ("pB11 Mirror", ConfinementConcept.MIRROR, Fuel.PB11),
    ("DT Tokamak", ConfinementConcept.TOKAMAK, Fuel.DT),
    ("DT Mirror", ConfinementConcept.MIRROR, Fuel.DT),
]
NET_MW = 1000.0
AVAIL = 0.85
LIFETIME = 30
INFLATION = 0.0245

# ── Run baseline vs free-core for each concept ────────────────────
print("Free Fusion Core — What's the LCOE floor?")
print("=" * 72)
print(
    f"{'Concept':<16} {'Baseline':>10} {'Free Core':>10}"
    f" {'Floor':>8} {'CAS22':>8} {'BOP':>8}"
)
print(f"{'':16} {'$/MWh':>10} {'$/MWh':>10} {'$/kW':>8} {'M$':>8} {'M$':>8}")
print("-" * 72)

for name, concept, fuel in CONCEPTS:
    m = CostModel(concept=concept, fuel=fuel)
    kw = dict(
        net_electric_mw=NET_MW,
        availability=AVAIL,
        lifetime_yr=LIFETIME,
        inflation_rate=INFLATION,
    )

    base = m.forward(**kw)

    # Zero the entire fusion core — CAS22 and special materials
    free = m.forward(
        **kw,
        cost_overrides={"CAS22": 0.0, "CAS27": 0.0},
    )

    bop = free.costs.cas23 + free.costs.cas24 + free.costs.cas25 + free.costs.cas26
    print(
        f"{name:<16} {base.costs.lcoe:>10.1f} {free.costs.lcoe:>10.1f}"
        f" {free.costs.overnight_cost:>8.0f}"
        f" {base.costs.cas22:>8.0f} {bop:>8.0f}"
    )

# ── Detailed breakdown for pB11 mirror (cheapest) ─────────────────
print("\n" + "=" * 72)
print("Detailed: pB11 Mirror with free fusion core")
print("=" * 72)

m = CostModel(concept=ConfinementConcept.MIRROR, fuel=Fuel.PB11)
base = m.forward(
    net_electric_mw=NET_MW,
    availability=AVAIL,
    lifetime_yr=LIFETIME,
    inflation_rate=INFLATION,
)
free = m.forward(
    net_electric_mw=NET_MW,
    availability=AVAIL,
    lifetime_yr=LIFETIME,
    inflation_rate=INFLATION,
    cost_overrides={"CAS22": 0.0, "CAS27": 0.0},
)

bc = base.costs
fc = free.costs

print(f"\n{'Account':<24} {'Baseline':>10} {'Free Core':>10} {'Delta':>10}")
print(f"{'':24} {'M$':>10} {'M$':>10} {'M$':>10}")
print("-" * 56)
rows = [
    ("CAS10 Pre-construction", bc.cas10, fc.cas10),
    ("CAS21 Buildings", bc.cas21, fc.cas21),
    ("CAS22 Reactor equip", bc.cas22, fc.cas22),
    ("CAS23 Turbine", bc.cas23, fc.cas23),
    ("CAS24 Electrical", bc.cas24, fc.cas24),
    ("CAS25 Miscellaneous", bc.cas25, fc.cas25),
    ("CAS26 Heat rejection", bc.cas26, fc.cas26),
    ("CAS27 Special materials", bc.cas27, fc.cas27),
    ("CAS28 Digital twin", bc.cas28, fc.cas28),
    ("CAS29 Contingency", bc.cas29, fc.cas29),
    ("CAS30 Indirect", bc.cas30, fc.cas30),
    ("CAS40 Owner's costs", bc.cas40, fc.cas40),
    ("CAS50 Supplementary", bc.cas50, fc.cas50),
]
for label, bv, fv in rows:
    print(f"{label:<24} {bv:>10.1f} {fv:>10.1f} {fv - bv:>+10.1f}")

print("-" * 56)
print(
    f"{'Overnight cost':<24} {bc.overnight_cost * NET_MW / 1000:>10.0f}"
    f" {fc.overnight_cost * NET_MW / 1000:>10.0f}"
    f" {(fc.overnight_cost - bc.overnight_cost) * NET_MW / 1000:>+10.0f}"
)

print(f"\n{'Metric':<24} {'Baseline':>10} {'Free Core':>10}")
print("-" * 46)
print(f"{'Overnight ($/kW)':<24} {bc.overnight_cost:>10.0f} {fc.overnight_cost:>10.0f}")
print(f"{'LCOE ($/MWh)':<24} {bc.lcoe:>10.1f} {fc.lcoe:>10.1f}")
print(f"{'LCOE (¢/kWh)':<24} {bc.lcoe / 10:>10.2f} {fc.lcoe / 10:>10.2f}")

# ── What the floor means ──────────────────────────────────────────
bop_total = fc.cas23 + fc.cas24 + fc.cas25 + fc.cas26
print(f"""
Interpretation:
  Even with a COMPLETELY FREE fusion heat source (CAS22 = $0),
  the LCOE floor is ${fc.lcoe:.1f}/MWh ({fc.lcoe / 10:.2f} ¢/kWh)
  at {AVAIL:.0%} availability, {LIFETIME}-yr life, default WACC.

  This floor comes from:
  - Buildings ({fc.cas21:.0f}M$) — turbine hall, control room,
    switchyard, site infrastructure
  - BOP ({bop_total:.0f}M$) — steam turbines, generators,
    electrical plant, cooling towers
  - O&M ({fc.cas71:.1f}M$/yr levelized) — staff to run the plant
  - Financing — IDC and capital recovery on the above

  The fusion core (CAS22 = {bc.cas22:.0f}M$) accounts for
  {(bc.lcoe - fc.lcoe) / bc.lcoe * 100:.0f}% of baseline LCOE.
""")

# ══════════════════════════════════════════════════════════════════════
# PART 3: PATHS TO <1 ¢/kWh WITH A FREE FUSION CORE
# ══════════════════════════════════════════════════════════════════════
print("=" * 72)
print("PART 3: Paths to <1 ¢/kWh ($10/MWh) with free fusion core")
print("=" * 72)

TARGET = 10.0
FREE_CORE = {"CAS22": 0.0, "CAS27": 0.0}

# ── Path A: Scale ─────────────────────────────────────────────────
print("\n── Path A: Plant scale (spread fixed costs over more MWh) ──")
print(f"  {'P_net':>8} {'LCOE':>8} {'¢/kWh':>8} {'O/N $/kW':>10} {'Gap':>8}")
for p in [1000, 2000, 3000, 5000]:
    r = m.forward(
        net_electric_mw=float(p),
        availability=AVAIL,
        lifetime_yr=LIFETIME,
        inflation_rate=INFLATION,
        cost_overrides=FREE_CORE,
    )
    gap = r.costs.lcoe - TARGET
    marker = " ***" if gap <= 0 else ""
    print(
        f"  {p:>7d} {r.costs.lcoe:>8.1f} {r.costs.lcoe / 10:>8.2f}"
        f" {r.costs.overnight_cost:>10.0f} {gap:>+8.1f}{marker}"
    )

# ── Path B: Availability ──────────────────────────────────────────
print("\n── Path B: Higher availability (more MWh per $ of capital) ──")
print(f"  {'Avail':>8} {'LCOE':>8} {'¢/kWh':>8} {'O/N $/kW':>10} {'Gap':>8}")
for av in [0.85, 0.90, 0.95, 0.98]:
    r = m.forward(
        net_electric_mw=NET_MW,
        availability=av,
        lifetime_yr=LIFETIME,
        inflation_rate=INFLATION,
        cost_overrides=FREE_CORE,
    )
    gap = r.costs.lcoe - TARGET
    marker = " ***" if gap <= 0 else ""
    print(
        f"  {av:>7.0%} {r.costs.lcoe:>8.1f} {r.costs.lcoe / 10:>8.2f}"
        f" {r.costs.overnight_cost:>10.0f} {gap:>+8.1f}{marker}"
    )

# ── Path C: Cheaper financing ─────────────────────────────────────
print("\n── Path C: Lower WACC (cheaper capital) ──")
print(f"  {'WACC':>8} {'LCOE':>8} {'¢/kWh':>8} {'O/N $/kW':>10} {'Gap':>8}")
for wacc in [0.07, 0.05, 0.04, 0.03, 0.02]:
    r = m.forward(
        net_electric_mw=NET_MW,
        availability=AVAIL,
        lifetime_yr=LIFETIME,
        inflation_rate=INFLATION,
        interest_rate=wacc,
        cost_overrides=FREE_CORE,
    )
    gap = r.costs.lcoe - TARGET
    marker = " ***" if gap <= 0 else ""
    print(
        f"  {wacc:>7.1%} {r.costs.lcoe:>8.1f} {r.costs.lcoe / 10:>8.2f}"
        f" {r.costs.overnight_cost:>10.0f} {gap:>+8.1f}{marker}"
    )

# ── Path D: Longer plant life ─────────────────────────────────────
print("\n── Path D: Longer plant lifetime (more years to amortize) ──")
print(f"  {'Life':>8} {'LCOE':>8} {'¢/kWh':>8} {'O/N $/kW':>10} {'Gap':>8}")
for life in [30, 40, 50, 60]:
    r = m.forward(
        net_electric_mw=NET_MW,
        availability=AVAIL,
        lifetime_yr=life,
        inflation_rate=INFLATION,
        cost_overrides=FREE_CORE,
    )
    gap = r.costs.lcoe - TARGET
    marker = " ***" if gap <= 0 else ""
    print(
        f"  {life:>6d}yr {r.costs.lcoe:>8.1f} {r.costs.lcoe / 10:>8.2f}"
        f" {r.costs.overnight_cost:>10.0f} {gap:>+8.1f}{marker}"
    )

# ── Path E: Faster construction ───────────────────────────────────
print("\n── Path E: Faster construction (less IDC) ──")
print(f"  {'Build':>8} {'LCOE':>8} {'¢/kWh':>8} {'O/N $/kW':>10} {'Gap':>8}")
for ct in [6, 4, 3, 2]:
    r = m.forward(
        net_electric_mw=NET_MW,
        availability=AVAIL,
        lifetime_yr=LIFETIME,
        inflation_rate=INFLATION,
        construction_time_yr=float(ct),
        cost_overrides=FREE_CORE,
    )
    gap = r.costs.lcoe - TARGET
    marker = " ***" if gap <= 0 else ""
    print(
        f"  {ct:>6d}yr {r.costs.lcoe:>8.1f} {r.costs.lcoe / 10:>8.2f}"
        f" {r.costs.overnight_cost:>10.0f} {gap:>+8.1f}{marker}"
    )

# ── Path F: Halve buildings ───────────────────────────────────────
print("\n── Path F: Compact siting / cheaper buildings ──")
print(f"  {'CAS21':>8} {'LCOE':>8} {'¢/kWh':>8} {'O/N $/kW':>10} {'Gap':>8}")
for frac in [1.0, 0.75, 0.50, 0.25]:
    ovr = dict(FREE_CORE)
    ovr["CAS21"] = fc.cas21 * frac
    r = m.forward(
        net_electric_mw=NET_MW,
        availability=AVAIL,
        lifetime_yr=LIFETIME,
        inflation_rate=INFLATION,
        cost_overrides=ovr,
    )
    gap = r.costs.lcoe - TARGET
    marker = " ***" if gap <= 0 else ""
    label = f"{frac:.0%} base"
    print(
        f"  {label:>8} {r.costs.lcoe:>8.1f} {r.costs.lcoe / 10:>8.2f}"
        f" {r.costs.overnight_cost:>10.0f} {gap:>+8.1f}{marker}"
    )


# ── Path G: Direct Energy Conversion ──────────────────────────────
# DEC converts charged particles directly to electricity at 85%+
# efficiency, bypassing the steam cycle. For pB11 (~99.8% charged
# particles), DEC eliminates most of the turbine island.
# The BOP costs (CAS23 turbine, CAS26 heat rejection) should scale
# with THERMAL electric, not total electric. With f_dec=0.9,
# only 10% of charged power goes through steam → CAS23/26 shrink.
print("\n── Path G: Direct Energy Conversion (skip steam cycle) ──")
print(
    f"  {'f_dec':>8} {'eta_de':>8} {'LCOE':>8} {'¢/kWh':>8}"
    f" {'O/N $/kW':>10} {'Gap':>8} {'Notes':>20}"
)

for f_dec, eta_de, notes in [
    (0.0, 0.0, "Steam only"),
    (0.5, 0.80, "Hybrid"),
    (0.8, 0.85, "Mostly DEC"),
    (0.9, 0.85, "DEC + small steam"),
    (0.95, 0.90, "Aggressive DEC"),
]:
    # Run with DEC parameters
    r_dec = m.forward(
        net_electric_mw=NET_MW,
        availability=AVAIL,
        lifetime_yr=LIFETIME,
        inflation_rate=INFLATION,
        f_dec=f_dec,
        eta_de=eta_de,
        cost_overrides=FREE_CORE,
    )
    # The model scales CAS23/26 with p_et (total gross electric),
    # but with DEC most of p_et is direct electric, not thermal.
    # Correct: turbine/heat rejection scale with p_the only.
    pt = r_dec.power_table
    # Fraction of gross electric that is thermal
    thermal_frac = (
        float(pt.p_et - f_dec * eta_de * pt.p_fus * 0.8) / float(pt.p_et)
        if f_dec > 0
        else 1.0
    )
    # Clamp to [0.05, 1.0] — always need some thermal handling
    thermal_frac = max(0.05, min(1.0, thermal_frac))

    # Re-run with corrected BOP costs
    ovr_dec = dict(FREE_CORE)
    ovr_dec["CAS23"] = r_dec.costs.cas23 * thermal_frac
    ovr_dec["CAS26"] = r_dec.costs.cas26 * thermal_frac
    # DEC building replaces part of turbine building —
    # but DEC hardware is simpler, net CAS21 reduction ~20%
    ovr_dec["CAS21"] = r_dec.costs.cas21 * (0.8 + 0.2 * thermal_frac)

    r_corr = m.forward(
        net_electric_mw=NET_MW,
        availability=AVAIL,
        lifetime_yr=LIFETIME,
        inflation_rate=INFLATION,
        f_dec=f_dec,
        eta_de=eta_de,
        cost_overrides=ovr_dec,
    )
    gap = r_corr.costs.lcoe - TARGET
    marker = " ***" if gap <= 0 else ""
    print(
        f"  {f_dec:>8.0%} {eta_de:>8.0%}"
        f" {r_corr.costs.lcoe:>8.1f} {r_corr.costs.lcoe / 10:>8.2f}"
        f" {r_corr.costs.overnight_cost:>10.0f}"
        f" {gap:>+8.1f} {notes:>20}{marker}"
    )


# ══════════════════════════════════════════════════════════════════════
# PART 4: COMBINED SCENARIOS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART 4: Combined scenarios — stacking levers")
print("=" * 72)

scenarios = []


def run_scenario(label, **kw):
    ovr = dict(kw.pop("cost_overrides", FREE_CORE))
    r = m.forward(cost_overrides=ovr, inflation_rate=INFLATION, **kw)
    scenarios.append((label, r))
    return r


print(f"\n  {'Scenario':<40} {'LCOE':>7} {'¢/kWh':>7} {'O/N $/kW':>9} {'Gap':>7}")
print("-" * 74)

# Start from baseline free-core
run_scenario(
    "Free core (baseline)",
    net_electric_mw=NET_MW,
    availability=AVAIL,
    lifetime_yr=LIFETIME,
)

# Path 1: Scale + availability
run_scenario(
    "+ 2 GWe",
    net_electric_mw=2000.0,
    availability=AVAIL,
    lifetime_yr=LIFETIME,
)
run_scenario(
    "+ 95% availability",
    net_electric_mw=2000.0,
    availability=0.95,
    lifetime_yr=LIFETIME,
)

# Path 2: Finance
run_scenario(
    "+ 4% WACC",
    net_electric_mw=2000.0,
    availability=0.95,
    lifetime_yr=LIFETIME,
    interest_rate=0.04,
)
run_scenario(
    "+ 3% WACC",
    net_electric_mw=2000.0,
    availability=0.95,
    lifetime_yr=LIFETIME,
    interest_rate=0.03,
)

# Path 3: Construction + lifetime
run_scenario(
    "+ 3-yr construction",
    net_electric_mw=2000.0,
    availability=0.95,
    lifetime_yr=LIFETIME,
    interest_rate=0.03,
    construction_time_yr=3.0,
)
run_scenario(
    "+ 50-yr lifetime",
    net_electric_mw=2000.0,
    availability=0.95,
    lifetime_yr=50,
    interest_rate=0.03,
    construction_time_yr=3.0,
)

# Path 4: Compact buildings
ovr_compact = dict(FREE_CORE)
ovr_compact["CAS21"] = fc.cas21 * 0.5
run_scenario(
    "+ CAS21 halved (compact)",
    net_electric_mw=2000.0,
    availability=0.95,
    lifetime_yr=50,
    interest_rate=0.03,
    construction_time_yr=3.0,
    cost_overrides=ovr_compact,
)

# Path 5: mega-scale
run_scenario(
    "5 GWe, 95%, 3yr, 2% WACC, 50yr",
    net_electric_mw=5000.0,
    availability=0.95,
    lifetime_yr=50,
    interest_rate=0.02,
    construction_time_yr=3.0,
)


# ── DEC paths: eliminate steam cycle ──────────────────────────────
# Helper: compute corrected BOP overrides for DEC
def dec_overrides(base_result, f_dec, eta_de):
    """Scale CAS23/26 to thermal fraction when DEC handles most power."""
    ovr = dict(FREE_CORE)
    pt = base_result.power_table
    if f_dec > 0 and float(pt.p_et) > 0:
        thermal_frac = max(
            0.05,
            float(pt.p_et - f_dec * eta_de * pt.p_fus * 0.8) / float(pt.p_et),
        )
    else:
        thermal_frac = 1.0
    ovr["CAS23"] = base_result.costs.cas23 * thermal_frac
    ovr["CAS26"] = base_result.costs.cas26 * thermal_frac
    ovr["CAS21"] = base_result.costs.cas21 * (0.8 + 0.2 * thermal_frac)
    return ovr


# DEC at baseline conditions
r_dec_base = m.forward(
    net_electric_mw=NET_MW,
    availability=AVAIL,
    lifetime_yr=LIFETIME,
    inflation_rate=INFLATION,
    f_dec=0.9,
    eta_de=0.85,
    cost_overrides=FREE_CORE,
)
ovr_dec = dec_overrides(r_dec_base, 0.9, 0.85)
run_scenario(
    "Free core + DEC (90%, 85% eff)",
    net_electric_mw=NET_MW,
    availability=AVAIL,
    lifetime_yr=LIFETIME,
    f_dec=0.9,
    eta_de=0.85,
    cost_overrides=ovr_dec,
)

# DEC + moderate scale + finance
r_dec_mod = m.forward(
    net_electric_mw=2000.0,
    availability=0.95,
    lifetime_yr=50,
    inflation_rate=INFLATION,
    interest_rate=0.03,
    construction_time_yr=3.0,
    f_dec=0.9,
    eta_de=0.85,
    cost_overrides=FREE_CORE,
)
ovr_dec_mod = dec_overrides(r_dec_mod, 0.9, 0.85)
run_scenario(
    "DEC + 2GWe, 95%, 3yr, 3% WACC, 50yr",
    net_electric_mw=2000.0,
    availability=0.95,
    lifetime_yr=50,
    interest_rate=0.03,
    construction_time_yr=3.0,
    f_dec=0.9,
    eta_de=0.85,
    cost_overrides=ovr_dec_mod,
)

# DEC + mega-scale
r_dec_mega = m.forward(
    net_electric_mw=5000.0,
    availability=0.95,
    lifetime_yr=50,
    inflation_rate=INFLATION,
    interest_rate=0.02,
    construction_time_yr=3.0,
    f_dec=0.9,
    eta_de=0.85,
    cost_overrides=FREE_CORE,
)
ovr_dec_mega = dec_overrides(r_dec_mega, 0.9, 0.85)
run_scenario(
    "DEC + 5GWe, 95%, 3yr, 2% WACC, 50yr",
    net_electric_mw=5000.0,
    availability=0.95,
    lifetime_yr=50,
    interest_rate=0.02,
    construction_time_yr=3.0,
    f_dec=0.9,
    eta_de=0.85,
    cost_overrides=ovr_dec_mega,
)

for label, r in scenarios:
    gap = r.costs.lcoe - TARGET
    marker = " ***" if gap <= 0 else ""
    print(
        f"  {label:<40} {r.costs.lcoe:>7.1f} {r.costs.lcoe / 10:>7.2f}"
        f" {r.costs.overnight_cost:>9.0f} {gap:>+7.1f}{marker}"
    )


# ══════════════════════════════════════════════════════════════════════
# PART 5: WHAT THIS MEANS
# ══════════════════════════════════════════════════════════════════════
print(f"""
{"=" * 72}
PART 5: What this means
{"=" * 72}

FINDING: Even with a free fusion core, reaching $10/MWh requires
stacking multiple favorable conditions. DEC is the single most
impactful lever — it eliminates the steam cycle, the largest
remaining cost after the fusion core itself.

PATH 1 — STEAM: SCALE + FINANCE (no DEC, no cost breakthrough):
  2 GWe, 95% availability, 3% WACC, 3-yr build, 50-yr life
  → ~${scenarios[6][1].costs.lcoe:.0f}/MWh
  Still needs low cost of capital and long plant life.

PATH 2 — STEAM: MEGA-SCALE:
  5 GWe, 95% availability, 2% WACC, 3-yr build, 50-yr life
  → ~${scenarios[8][1].costs.lcoe:.0f}/MWh
  Government-scale projects with sovereign financing.

PATH 3 — STEAM: COMPACT + MODERATE SCALE:
  2 GWe, 95% availability, 3% WACC, 3-yr build, 50-yr life,
  buildings halved (compact siting)
  → ~${scenarios[7][1].costs.lcoe:.0f}/MWh
  Needs both cheap capital AND smaller physical footprint.

PATH 4 — DEC + MODERATE SCALE + FINANCE:
  2 GWe, 95% availability, 3% WACC, 3-yr build, 50-yr life,
  90% DEC at 85% efficiency
  → ~${scenarios[10][1].costs.lcoe:.0f}/MWh
  DEC replaces most of the steam island. For pB11 where
  ~99.8% of energy is in charged particles, this eliminates
  the dominant BOP cost. Achievable with moderate scale.

PATH 5 — DEC + MEGA-SCALE:
  5 GWe, 95% availability, 2% WACC, 3-yr build, 50-yr life,
  90% DEC at 85% efficiency
  → ~${scenarios[11][1].costs.lcoe:.0f}/MWh
  The deepest floor — what remains is mostly O&M and land.

THE IRREDUCIBLE COSTS (even with free core):
  - Staff: ~$24M/yr base (pB11, scales with P^0.5)
  - Buildings: ~$586M at 1 GWe (scales with gross electric)
  - BOP: ~$421M at 1 GWe (turbines, electrical, cooling)
  - Indirects: ~20% of whatever direct costs remain
  - Financing: CRF × capital charges, IDC during construction

IMPLICATION FOR FUSION CORE COST TARGETS:
  If the floor with a FREE core is ~${fc.lcoe:.0f}/MWh at default
  conditions, then the fusion core must not only be cheap — the
  entire plant must be built at scale with favorable financing.
  No amount of reactor cost reduction alone reaches 1 ¢/kWh.
  The path requires simultaneous advances in scale, availability,
  construction speed, financing, and plant lifetime.
""")

# ── Sensitivity: what dominates the floor? ────────────────────────
print("=" * 72)
print("Sensitivity of the free-core LCOE floor")
print("=" * 72)

sens = m.sensitivity(
    free.params,
    cost_overrides=FREE_CORE,
)

print("\nAll parameters ranked by |elasticity| (top 20):")
all_sens = {}
for cat, items in sens.items():
    for k, v in items.items():
        all_sens[k] = (v, cat)
for k, (v, cat) in sorted(all_sens.items(), key=lambda x: abs(x[1][0]), reverse=True)[
    :20
]:
    print(f"  {k:<36} {v:+.4f}  ({cat})")
