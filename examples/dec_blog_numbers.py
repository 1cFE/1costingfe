"""Generate all numbers used in the blog post:
'Direct Energy Conversion and the Cost Floor'

Reproduces every table and inline figure in the post using the
1costingfe model with a p-B11 mirror and various DEC configurations.
"""

from costingfe import ConfinementConcept, CostModel, Fuel
from costingfe.types import PowerCycle

FREE_CORE = {"CAS22": 0.0, "CAS27": 0.0}
INFLATION = 0.0245
TARGET = 10.0

m = CostModel(
    concept=ConfinementConcept.MIRROR,
    fuel=Fuel.PB11,
    power_cycle=PowerCycle.BRAYTON_SCO2,
)

m_dhe3 = CostModel(
    concept=ConfinementConcept.MIRROR,
    fuel=Fuel.DHE3,
    power_cycle=PowerCycle.BRAYTON_SCO2,
)


def free_core_with_dec(model, f_dec, eta_de, **kw):
    """Run free-core analysis but keep C220109 (DEC hardware) cost."""
    kw.setdefault("inflation_rate", INFLATION)
    if f_dec > 0:
        kw["f_dec"] = f_dec
        kw["eta_de"] = eta_de

    # First pass: get DEC hardware cost from full model
    r_full = model.forward(**kw)
    c220109 = float(r_full.cas22_detail.get("C220109", 0.0))

    # Second pass: zero core except DEC hardware
    kw["cost_overrides"] = {"CAS22": c220109, "CAS27": 0.0}
    r_free = model.forward(**kw)
    return r_free, c220109


# ══════════════════════════════════════════════════════════════════════
# TABLE 1: Venetian blind DEC at 1 GWe baseline (blog table 1)
# ══════════════════════════════════════════════════════════════════════
print("=" * 75)
print("TABLE 1: Venetian blind DEC — 1 GWe pB11 baseline, free core")
print("=" * 75)

BASE_KW = dict(
    net_electric_mw=1000.0,
    availability=0.85,
    lifetime_yr=30,
    interest_rate=0.07,
    construction_time_yr=6.0,
)

configs_vb = [
    ("Thermal only (sCO2 Brayton, 47%)", 0.0, 0.0, {}),
    ("VB DEC at 48% + thermal (hybrid)", 0.9, 0.48, {}),
    ("VB DEC at 60% + thermal (hybrid)", 0.9, 0.60, {}),
]

print(f"  {'Configuration':<45} {'Floor':>6} {'O/N':>7}")
print("-" * 62)
for label, f_dec, eta_de, extra in configs_vb:
    r, dec_cost = free_core_with_dec(m, f_dec, eta_de, **BASE_KW)
    if extra:
        # Re-run with extra overrides
        kw2 = dict(BASE_KW, inflation_rate=INFLATION)
        if f_dec > 0:
            kw2["f_dec"] = f_dec
            kw2["eta_de"] = eta_de
        ovr = {"CAS22": dec_cost, "CAS27": 0.0}
        ovr.update(extra)
        kw2["cost_overrides"] = ovr
        r = m.forward(**kw2)
    c = r.costs
    print(f"  {label:<45} {c.lcoe:>5.0f} {c.overnight_cost:>7.0f}")


# TABLE 2 (pulsed p-B11 no-turbine) removed: wasting 87% brem heat
# requires enormous fusion power, making no-turbine unviable for p-B11.
# Pulsed inductive DEC is modeled for D-He3 in TABLE 5 below.

# Thermal reference (needed for supplementary table)
r_th = m.forward(**BASE_KW, inflation_rate=INFLATION, cost_overrides=FREE_CORE)


# ══════════════════════════════════════════════════════════════════════
# TABLE 3: All approaches at aggressive conditions (blog table 3)
# ══════════════════════════════════════════════════════════════════════
print()
print("=" * 75)
print("TABLE 3: All DEC approaches — 2 GWe pB11 aggressive, free core")
print("=" * 75)

AGG_KW = dict(
    net_electric_mw=2000.0,
    availability=0.95,
    lifetime_yr=50,
    interest_rate=0.03,
    construction_time_yr=3.0,
)

print(f"  {'Approach':<45} {'Floor':>6} {'O/N':>7} {'Budget':>8}")
print("-" * 70)

# Thermal
r_th_agg = m.forward(**AGG_KW, inflation_rate=INFLATION, cost_overrides=FREE_CORE)
budget = TARGET - r_th_agg.costs.lcoe
print(
    f"  {'Thermal only (sCO2 Brayton, 47%)':<45}"
    f" {r_th_agg.costs.lcoe:>5.1f} {r_th_agg.costs.overnight_cost:>7.0f}"
    f" {budget:>+7.1f}"
)

# VB DEC 60% hybrid
r_vb_agg, dec_agg = free_core_with_dec(m, 0.9, 0.60, **AGG_KW)
budget = TARGET - r_vb_agg.costs.lcoe
print(
    f"  {'VB DEC 60% + thermal (hybrid)':<45}"
    f" {r_vb_agg.costs.lcoe:>5.1f} {r_vb_agg.costs.overnight_cost:>7.0f}"
    f" {budget:>+7.1f}"
)

# No-turbine cases omitted for p-B11: wasting 87% brem heat requires
# enormous fusion power (16-30 GW for 1-2 GWe net), making the fully
# costed LCOE far worse than thermal.


# ══════════════════════════════════════════════════════════════════════
# TABLE 4: Fully costed D-He3 — DEC vs thermal (blog "Where DEC Helps")
# ══════════════════════════════════════════════════════════════════════
print()
print("=" * 75)
print("TABLE 4: Fully costed D-He3 mirror — DEC vs thermal (1 GWe baseline)")
print("=" * 75)

energy_base = 8760 * 1000.0 * 0.85

r_dhe3_th = m_dhe3.forward(**BASE_KW, inflation_rate=INFLATION)
r_dhe3_vb = m_dhe3.forward(**BASE_KW, inflation_rate=INFLATION, f_dec=0.9, eta_de=0.60)
r_dhe3_pi = m_dhe3.forward(**BASE_KW, inflation_rate=INFLATION, f_dec=0.95, eta_de=0.85)

print(f"  {'Configuration':<35} {'LCOE':>7} {'Core':>8} {'Fuel':>7}")
print("-" * 62)
for label, r in [
    ("Thermal only (47%)", r_dhe3_th),
    ("VB DEC 60% (hybrid)", r_dhe3_vb),
    ("Pulsed inductive (85%)", r_dhe3_pi),
]:
    fuel_mwh = r.costs.cas80 * 1e6 / energy_base
    print(f"  {label:<35} {r.costs.lcoe:>6.0f} {r.costs.cas22:>7.0f} {fuel_mwh:>6.0f}")

# Also p-B11 for comparison
print()
print("  p-B11 comparison:")
r_pb_th = m.forward(**BASE_KW, inflation_rate=INFLATION)
r_pb_vb = m.forward(**BASE_KW, inflation_rate=INFLATION, f_dec=0.9, eta_de=0.60)
pb_th_lcoe = f"{r_pb_th.costs.lcoe:>6.0f}"
pb_th_core = f"{r_pb_th.costs.cas22:>7.0f}"
print(f"  {'Thermal only (47%)':<35} {pb_th_lcoe} {pb_th_core}")
pb_vb_lcoe = f"{r_pb_vb.costs.lcoe:>6.0f}"
pb_vb_core = f"{r_pb_vb.costs.cas22:>7.0f}"
print(f"  {'VB DEC 60% (hybrid)':<35} {pb_vb_lcoe} {pb_vb_core}")


# ══════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY: Cost breakdown details
# ══════════════════════════════════════════════════════════════════════
print()
print("=" * 75)
print("SUPPLEMENTARY: Detailed cost breakdown at 1 GWe baseline")
print("=" * 75)

print(f"\n  {'Account':<20} {'Thermal':>10} {'VB 60%':>10}")
print(f"  {'':20} {'M$':>10} {'M$':>10}")
print("-" * 45)

# Get VB 60% free-core result
r_vb, _ = free_core_with_dec(m, 0.9, 0.60, **BASE_KW)

for label, attr in [
    ("CAS21 Buildings", "cas21"),
    ("CAS22 (DEC only)", "cas22"),
    ("CAS23 Turbine", "cas23"),
    ("CAS24 Electrical", "cas24"),
    ("CAS25 Miscellaneous", "cas25"),
    ("CAS26 Heat rejection", "cas26"),
    ("CAS30 Indirect", "cas30"),
    ("CAS70 O&M/yr", "cas70"),
]:
    vt = float(getattr(r_th.costs, attr))
    vv = float(getattr(r_vb.costs, attr))
    print(f"  {label:<20} {vt:>10.0f} {vv:>10.0f}")

print("-" * 45)
print(f"  {'LCOE ($/MWh)':<20} {r_th.costs.lcoe:>10.1f} {r_vb.costs.lcoe:>10.1f}")
print(
    f"  {'Overnight ($/kW)':<20}"
    f" {r_th.costs.overnight_cost:>10.0f}"
    f" {r_vb.costs.overnight_cost:>10.0f}"
)


# ══════════════════════════════════════════════════════════════════════
# TABLE 5: D-He3 BOP floors (blog D-He3 section)
# ══════════════════════════════════════════════════════════════════════
print()
print("=" * 75)
print("TABLE 5: D-He3 BOP floors — free core, fuel separated")
print("=" * 75)

FREE_CORE_DHE3 = {"CAS22": 0.0, "CAS27": 0.0}

configs_dhe3 = [
    ("Thermal only (sCO2 Brayton, 47%)", 0.0, 0.0),
    ("VB DEC 60% + thermal (hybrid)", 0.9, 0.60),
]

energy_1gw = 8760 * 1000.0 * 0.85

print(f"  {'Configuration':<45} {'BOP':>6} {'Fuel':>6} {'Total':>6}")
print("-" * 68)
for label, f_dec, eta_de in configs_dhe3:
    kw = dict(BASE_KW, inflation_rate=INFLATION, cost_overrides=FREE_CORE_DHE3)
    if f_dec > 0:
        kw["f_dec"] = f_dec
        kw["eta_de"] = eta_de
    r = m_dhe3.forward(**kw)
    fuel_mwh = r.costs.cas80 * 1e6 / energy_1gw
    bop_floor = r.costs.lcoe - fuel_mwh
    print(f"  {label:<45} {bop_floor:>5.0f} {fuel_mwh:>5.0f} {r.costs.lcoe:>5.0f}")

# Pulsed inductive D-He3
r_th_dhe3 = m_dhe3.forward(
    **BASE_KW, inflation_rate=INFLATION, cost_overrides=FREE_CORE_DHE3
)
inv_dhe3 = 150.0
cas21_p_dhe3 = float(r_th_dhe3.costs.cas21) * 0.75
cas26_p_dhe3 = float(r_th_dhe3.costs.cas26) * 0.15
r_p_dhe3 = m_dhe3.forward(
    **BASE_KW,
    inflation_rate=INFLATION,
    f_dec=0.95,
    eta_de=0.85,
    cost_overrides={
        "CAS22": inv_dhe3,
        "CAS27": 0.0,
        "CAS23": 0.0,
        "CAS26": cas26_p_dhe3,
        "CAS21": cas21_p_dhe3,
    },
)
fuel_p = r_p_dhe3.costs.cas80 * 1e6 / energy_1gw
bop_p = r_p_dhe3.costs.lcoe - fuel_p
print(
    f"  {'Pulsed inductive (85%, no turbine)':<45}"
    f" {bop_p:>5.0f} {fuel_p:>5.0f} {r_p_dhe3.costs.lcoe:>5.0f}"
)
print()
print("  He-3 price assumption: $2M/kg (optimistic).")
