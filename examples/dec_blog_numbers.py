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
    ("VB DEC at 60%, no turbine (waste brem)", 0.9, 0.60, {"CAS23": 0.0}),
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


# ══════════════════════════════════════════════════════════════════════
# TABLE 2: Pulsed inductive DEC at 1 GWe baseline (blog table 2)
# ══════════════════════════════════════════════════════════════════════
print()
print("=" * 75)
print("TABLE 2: Pulsed inductive DEC — 1 GWe pB11 baseline, free core")
print("=" * 75)

# Thermal reference
r_th = m.forward(**BASE_KW, inflation_rate=INFLATION, cost_overrides=FREE_CORE)

# Pulsed: no turbine, inverter only, reduced buildings and heat rejection
inverter_1gw = 150.0  # M$, $150/kW_net from NREL benchmarks
cas21_pulsed = float(r_th.costs.cas21) * 0.75  # 25% building reduction
cas26_pulsed = float(r_th.costs.cas26) * 0.15  # only brem waste heat

r_pulsed = m.forward(
    **BASE_KW,
    inflation_rate=INFLATION,
    f_dec=0.95,
    eta_de=0.85,
    cost_overrides={
        "CAS22": inverter_1gw,
        "CAS27": 0.0,
        "CAS23": 0.0,
        "CAS26": cas26_pulsed,
        "CAS21": cas21_pulsed,
    },
)

th_lcoe = float(r_th.costs.lcoe)
th_oc = float(r_th.costs.overnight_cost)
p_lcoe = float(r_pulsed.costs.lcoe)
p_oc = float(r_pulsed.costs.overnight_cost)
print(f"  Thermal floor:          ${th_lcoe:.0f}/MWh  (${th_oc:.0f}/kW)")
print(f"  Pulsed inductive floor: ${p_lcoe:.0f}/MWh  (${p_oc:.0f}/kW)")
print(f"    Inverter: ${inverter_1gw:.0f}M, CAS23=$0")
print(f"    Buildings: ${cas21_pulsed:.0f}M (vs ${r_th.costs.cas21:.0f}M thermal)")


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

# VB DEC 60% no turbine
kw_vb_nt = dict(AGG_KW, inflation_rate=INFLATION, f_dec=0.9, eta_de=0.60)
kw_vb_nt["cost_overrides"] = {"CAS22": dec_agg, "CAS27": 0.0, "CAS23": 0.0}
r_vb_nt = m.forward(**kw_vb_nt)
budget = TARGET - r_vb_nt.costs.lcoe
print(
    f"  {'VB DEC 60%, no turbine':<45}"
    f" {r_vb_nt.costs.lcoe:>5.1f} {r_vb_nt.costs.overnight_cost:>7.0f}"
    f" {budget:>+7.1f}"
)

# Pulsed inductive
inverter_2gw = 300.0
cas21_p_agg = float(r_th_agg.costs.cas21) * 0.75
cas26_p_agg = float(r_th_agg.costs.cas26) * 0.15
r_p_agg = m.forward(
    **AGG_KW,
    inflation_rate=INFLATION,
    f_dec=0.95,
    eta_de=0.85,
    cost_overrides={
        "CAS22": inverter_2gw,
        "CAS27": 0.0,
        "CAS23": 0.0,
        "CAS26": cas26_p_agg,
        "CAS21": cas21_p_agg,
    },
)
budget = TARGET - r_p_agg.costs.lcoe
print(
    f"  {'Pulsed inductive (85%, no turbine)':<45}"
    f" {r_p_agg.costs.lcoe:>5.1f} {r_p_agg.costs.overnight_cost:>7.0f}"
    f" {budget:>+7.1f}"
)


# ══════════════════════════════════════════════════════════════════════
# TABLE 4: Fully costed comparison (blog inline)
# ══════════════════════════════════════════════════════════════════════
print()
print("=" * 75)
print("TABLE 4: Fully costed pB11 mirror — DEC vs thermal (1 GWe baseline)")
print("=" * 75)

r_full_th = m.forward(**BASE_KW, inflation_rate=INFLATION)
r_full_vb = m.forward(**BASE_KW, inflation_rate=INFLATION, f_dec=0.9, eta_de=0.60)

print(f"  {'Configuration':<35} {'LCOE':>7} {'CAS22':>8}")
print("-" * 55)
print(
    f"  {'Thermal only (47%)':<35}"
    f" {r_full_th.costs.lcoe:>6.0f} {r_full_th.costs.cas22:>7.0f}"
)
print(
    f"  {'VB DEC 60% (hybrid)':<35}"
    f" {r_full_vb.costs.lcoe:>6.0f} {r_full_vb.costs.cas22:>7.0f}"
)


# ══════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY: Cost breakdown details
# ══════════════════════════════════════════════════════════════════════
print()
print("=" * 75)
print("SUPPLEMENTARY: Detailed cost breakdown at 1 GWe baseline")
print("=" * 75)

print(f"\n  {'Account':<20} {'Thermal':>10} {'VB 60%':>10} {'Pulsed':>10}")
print(f"  {'':20} {'M$':>10} {'M$':>10} {'M$':>10}")
print("-" * 55)

# Get VB 60% free-core result
r_vb, _ = free_core_with_dec(m, 0.9, 0.60, **BASE_KW)
r_p = r_pulsed  # from earlier

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
    vp = float(getattr(r_p.costs, attr))
    print(f"  {label:<20} {vt:>10.0f} {vv:>10.0f} {vp:>10.0f}")

print("-" * 55)
print(
    f"  {'LCOE ($/MWh)':<20}"
    f" {r_th.costs.lcoe:>10.1f} {r_vb.costs.lcoe:>10.1f}"
    f" {r_p.costs.lcoe:>10.1f}"
)
print(
    f"  {'Overnight ($/kW)':<20}"
    f" {r_th.costs.overnight_cost:>10.0f}"
    f" {r_vb.costs.overnight_cost:>10.0f}"
    f" {r_p.costs.overnight_cost:>10.0f}"
)
