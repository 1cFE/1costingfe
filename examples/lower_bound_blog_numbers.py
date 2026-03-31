"""Generate all numbers used in the blog post:
'The Lower Bound for Fusion Energy Cost'

Reproduces every table and inline figure in the post using the
1costingfe model with a supercritical CO2 Brayton cycle (lowest floor).
"""

from costingfe import ConfinementConcept, CostModel, Fuel
from costingfe.types import PowerCycle

FREE_CORE = {"CAS22": 0.0, "CAS27": 0.0}
INFLATION = 0.0245

m_pb11 = CostModel(
    concept=ConfinementConcept.MIRROR,
    fuel=Fuel.PB11,
    power_cycle=PowerCycle.BRAYTON_SCO2,
)
m_dt = CostModel(
    concept=ConfinementConcept.TOKAMAK,
    fuel=Fuel.DT,
    power_cycle=PowerCycle.BRAYTON_SCO2,
)

# ══════════════════════════════════════════════════════════════════════
# TABLE 1: BOP component breakdown (1 GWe pB11, free core)
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TABLE 1: BOP component breakdown (1 GWe pB11, sCO2, free core)")
print("=" * 70)

r = m_pb11.forward(
    net_electric_mw=1000.0,
    availability=0.85,
    lifetime_yr=30,
    inflation_rate=INFLATION,
    cost_overrides=FREE_CORE,
)
c = r.costs
bop = c.cas23 + c.cas24 + c.cas25 + c.cas26

print(f"  Buildings:              ${c.cas21:,.0f}M")
print(f"  Turbine & generator:    ${c.cas23:,.0f}M")
print(f"  Electrical plant:       ${c.cas24:,.0f}M")
print(f"  Miscellaneous:          ${c.cas25:,.0f}M")
print(f"  Heat rejection:         ${c.cas26:,.0f}M")
print(f"  BOP subtotal:           ${bop:,.0f}M")
print(f"  Total (buildings+BOP):  ${c.cas21 + bop:,.0f}M")

# ══════════════════════════════════════════════════════════════════════
# SECTION: The Price of the Balance of Plant
# ══════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("SECTION: The Price of the Balance of Plant")
print("=" * 70)

# Baseline free-core floor
free_base = m_pb11.forward(
    net_electric_mw=1000.0,
    availability=0.85,
    lifetime_yr=30,
    inflation_rate=INFLATION,
    cost_overrides=FREE_CORE,
)
# Fully costed baseline
full_base = m_pb11.forward(
    net_electric_mw=1000.0,
    availability=0.85,
    lifetime_yr=30,
    inflation_rate=INFLATION,
)

print(f"  Free-core LCOE floor:   ${free_base.costs.lcoe:.1f}/MWh")
print(f"  Free-core overnight:    ${free_base.costs.overnight_cost:,.0f}/kW")
print(f"  Fully costed LCOE:      ${full_base.costs.lcoe:.1f}/MWh")
core_share = (full_base.costs.lcoe - free_base.costs.lcoe) / full_base.costs.lcoe
print(f"  Core share of LCOE:     {core_share * 100:.0f}%")
print(f"  O&M (free core):        ${free_base.costs.cas70:.1f}M/yr")

# ══════════════════════════════════════════════════════════════════════
# TABLE 2: Floor at different conditions
# ══════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("TABLE 2: Floor at different conditions (sCO2, pB11, free core)")
print("=" * 70)

TARGET = 10.0
scenarios = [
    (
        "Baseline: 1 GWe, 85%, 7%, 30yr, 6yr",
        dict(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30),
    ),
    (
        "2 GWe, 85%, 7%, 30yr, 6yr",
        dict(net_electric_mw=2000.0, availability=0.85, lifetime_yr=30),
    ),
    (
        "2 GWe, 95%, 3%, 50yr, 3yr",
        dict(
            net_electric_mw=2000.0,
            availability=0.95,
            lifetime_yr=50,
            interest_rate=0.03,
            construction_time_yr=3.0,
        ),
    ),
    (
        "3 GWe, 95%, 3%, 50yr, 3yr",
        dict(
            net_electric_mw=3000.0,
            availability=0.95,
            lifetime_yr=50,
            interest_rate=0.03,
            construction_time_yr=3.0,
        ),
    ),
    (
        "5 GWe, 95%, 2%, 50yr, 3yr",
        dict(
            net_electric_mw=5000.0,
            availability=0.95,
            lifetime_yr=50,
            interest_rate=0.02,
            construction_time_yr=3.0,
        ),
    ),
]

print(f"  {'Scenario':<42} {'Floor':>6} {'O/N':>7} {'Budget':>8}")
print("-" * 70)
for label, kw in scenarios:
    r = m_pb11.forward(**kw, inflation_rate=INFLATION, cost_overrides=FREE_CORE)
    budget = TARGET - r.costs.lcoe
    print(
        f"  {label:<42} {r.costs.lcoe:>5.1f} {r.costs.overnight_cost:>7.0f}"
        f" {budget:>+7.1f}"
    )

# ══════════════════════════════════════════════════════════════════════
# SECTION: Core budget at aggressive conditions
# ══════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("SECTION: Core budget at aggressive conditions")
print("=" * 70)

agg_kw = dict(
    net_electric_mw=2000.0,
    availability=0.95,
    lifetime_yr=50,
    inflation_rate=INFLATION,
    interest_rate=0.03,
    construction_time_yr=3.0,
)
free_agg = m_pb11.forward(**agg_kw, cost_overrides=FREE_CORE)
full_agg = m_pb11.forward(**agg_kw)
core_budget_kw = full_agg.costs.overnight_cost - free_agg.costs.overnight_cost

print(f"  Free-core floor:        ${free_agg.costs.lcoe:.1f}/MWh")
print(f"  Budget for core:        ${TARGET - free_agg.costs.lcoe:.1f}/MWh")
print(f"  Core budget ($/kW):     ${core_budget_kw:,.0f}/kW")
print(f"  Full-core LCOE:         ${full_agg.costs.lcoe:.1f}/MWh")

# ══════════════════════════════════════════════════════════════════════
# SECTION: O&M and automation
# ══════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("SECTION: O&M and automation")
print("=" * 70)

energy_mwh = 8760 * 2000 * 0.95
om_per_mwh = free_agg.costs.cas70 * 1e6 / energy_mwh

print(f"  O&M at aggressive:      ${free_agg.costs.cas70:.1f}M/yr")
print(f"  O&M as $/MWh:           ${om_per_mwh:.1f}/MWh")
print(f"  Floor:                  ${free_agg.costs.lcoe:.1f}/MWh")
print(f"  O&M share of floor:     {om_per_mwh / free_agg.costs.lcoe * 100:.0f}%")

# Automation: halve O&M (30 staff -> 15 staff in free-core scenario)
halved_om_mwh = om_per_mwh / 2
automated_floor = free_agg.costs.lcoe - halved_om_mwh
print(f"  Automated floor:        ${automated_floor:.1f}/MWh (15 FTE)")
print(f"  Automated core budget:  ${TARGET - automated_floor:.1f}/MWh")

# ══════════════════════════════════════════════════════════════════════
# TABLE 3: DT vs pB11 comparison
# ══════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("TABLE 3: DT vs pB11 comparison (1 GWe, sCO2)")
print("=" * 70)

dt_full = m_dt.forward(
    net_electric_mw=1000.0,
    availability=0.85,
    lifetime_yr=30,
    inflation_rate=INFLATION,
)
dt_free = m_dt.forward(
    net_electric_mw=1000.0,
    availability=0.85,
    lifetime_yr=30,
    inflation_rate=INFLATION,
    cost_overrides=FREE_CORE,
)
pb_full = full_base
pb_free = free_base

hdr = f"  {'Metric':<20} {'pB11':>10} {'DT':>10}"
print(hdr)
print("-" * len(hdr))
pb_b, dt_b = pb_full.costs.cas21, dt_full.costs.cas21
print(f"  {'Buildings (1 GWe)':<20} ${pb_b:>7.0f}M ${dt_b:>7.0f}M")
pb_f, dt_f = pb_free.costs.lcoe, dt_free.costs.lcoe
print(f"  {'Free-core floor':<20} ${pb_f:>5.1f}/MWh ${dt_f:>5.1f}/MWh")
pb_c, dt_c = pb_full.costs.lcoe, dt_full.costs.lcoe
print(f"  {'Fully costed LCOE':<20} ${pb_c:>5.1f}/MWh ${dt_c:>5.1f}/MWh")

# ══════════════════════════════════════════════════════════════════════
# CROSS-CHECK: Power cycle comparison
# ══════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("CROSS-CHECK: Power cycle comparison (1 GWe pB11, free core)")
print("=" * 70)

print(f"  {'Cycle':<15} {'eta_th':>6} {'Floor':>8} {'O/N':>8}")
print("-" * 45)
for cycle in [PowerCycle.RANKINE, PowerCycle.BRAYTON_SCO2, PowerCycle.COMBINED]:
    m = CostModel(
        concept=ConfinementConcept.MIRROR,
        fuel=Fuel.PB11,
        power_cycle=cycle,
    )
    r = m.forward(
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
        inflation_rate=INFLATION,
        cost_overrides=FREE_CORE,
    )
    print(
        f"  {cycle.value:<15} {r.params['eta_th']:>6.2f}"
        f" {r.costs.lcoe:>7.1f} {r.costs.overnight_cost:>8.0f}"
    )
