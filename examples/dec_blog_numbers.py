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
    # p-B11 and D-3He are aneutronic primary; no breeding blanket needed.
    blanket_form="none",
    blanket_fill="none",
    mn=1.0,
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
# TABLE 1b: VB break-even efficiency at fixed hardware cost (blog addition)
# For each fixed VB CAS22 cost, find the eta_de where hybrid floor
# matches the thermal-only floor. f_dec=0.9.
# ══════════════════════════════════════════════════════════════════════
print()
print("=" * 75)
print("TABLE 1b: VB break-even eta at fixed hardware cost — 1 GWe pB11 free core")
print("=" * 75)

r_th_ref = m.forward(
    **BASE_KW, inflation_rate=INFLATION, cost_overrides={"CAS22": 0.0, "CAS27": 0.0}
)
lcoe_th = float(r_th_ref.costs.lcoe)
print(f"  Thermal-only reference: LCOE=${lcoe_th:.2f}/MWh")
print()
print(f"  {'Fixed VB cost':>14} {'Break-even eta':>16} {'Note':<30}")
print("-" * 64)


def find_breakeven_eta(fixed_cas22):
    prev_eta, prev_lcoe = None, None
    for eta_pct in range(30, 101):
        eta = eta_pct / 100.0
        r = m.forward(
            **BASE_KW,
            inflation_rate=INFLATION,
            f_dec=0.9,
            eta_de=eta,
            cost_overrides={"CAS22": float(fixed_cas22), "CAS27": 0.0},
        )
        lcoe = float(r.costs.lcoe)
        if prev_eta is not None and (prev_lcoe - lcoe_th) * (lcoe - lcoe_th) <= 0:
            frac = (prev_lcoe - lcoe_th) / (prev_lcoe - lcoe)
            return prev_eta + frac * (eta - prev_eta)
        prev_eta, prev_lcoe = eta, lcoe
    return None


for c in [20.0, 30.0, 45.0, 58.0, 80.0, 100.0]:
    eta_be = find_breakeven_eta(c)
    if eta_be is None:
        note = "no crossover in 30-100% range"
        cell = "  —  "
    elif eta_be > 0.70:
        note = "above 70% theoretical ceiling"
        cell = f"{eta_be * 100:.0f}%"
    else:
        note = "inside 48-70% physical window"
        cell = f"{eta_be * 100:.0f}%"
    print(f"  {'$' + f'{c:.0f}M':>14} {cell:>16} {note:<30}")


# ══════════════════════════════════════════════════════════════════════
# TABLE 1c: Heating-power sensitivity (blog addition)
# Sweep p_input and compare thermal-only vs VB 60% hybrid.
# VB hybrid preserves the model-scaled VB hardware cost (CAS22 = C220109);
# thermal-only has CAS22 = 0. This matches the apples-to-apples accounting
# used in Table 1 above.
# ══════════════════════════════════════════════════════════════════════
print()
print("=" * 75)
print("TABLE 1c: Heating-power sensitivity — 1 GWe pB11 free core")
print("=" * 75)
print(
    f"  {'p_in':>5} {'p_fus':>7} {'Q_eng':>6}"
    f"  {'th LCOE':>8} {'vb LCOE':>8} {'vb CAS22':>9} {'th-vb':>7}"
)
print("-" * 68)
for p_in in [5, 40, 80, 150, 250, 400, 600]:
    # Thermal-only: no DEC hardware
    r_th_sweep = m.forward(
        **BASE_KW,
        inflation_rate=INFLATION,
        p_input=float(p_in),
        cost_overrides={"CAS22": 0.0, "CAS27": 0.0},
    )
    # VB hybrid: first pass to read the model-scaled VB hardware cost,
    # second pass to free the core but preserve it.
    r_full_sweep = m.forward(
        **BASE_KW,
        inflation_rate=INFLATION,
        f_dec=0.9,
        eta_de=0.60,
        p_input=float(p_in),
    )
    vb_cas22 = float(r_full_sweep.cas22_detail.get("C220109", 0.0))
    r_vb_sweep = m.forward(
        **BASE_KW,
        inflation_rate=INFLATION,
        f_dec=0.9,
        eta_de=0.60,
        p_input=float(p_in),
        cost_overrides={"CAS22": vb_cas22, "CAS27": 0.0},
    )
    delta = float(r_th_sweep.costs.lcoe) - float(r_vb_sweep.costs.lcoe)
    print(
        f"  {p_in:>5} {float(r_th_sweep.power_table.p_fus):>7.0f}"
        f" {float(r_th_sweep.power_table.q_eng):>6.2f}"
        f"  {float(r_th_sweep.costs.lcoe):>8.2f}"
        f" {float(r_vb_sweep.costs.lcoe):>8.2f}"
        f" {vb_cas22:>9.1f}"
        f" {delta:>+7.2f}"
    )


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
    # p-B11 and D-3He are aneutronic primary; no breeding blanket needed.
    blanket_form="none",
    blanket_fill="none",
    mn=1.0,
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

# Pulsed inductive (fully costed): swap the model's electrostatic-DEC
# hardware cost (which is the VB formula it applies by default for mirrors)
# for $593M of pulsed DEC hardware, zero the turbine, and reduce buildings,
# heat rejection, and electrical plant (no synchronous generator / GSU).
# Helion-likely operating point:
#   - Magneto-inertial timescale: tritium not confined long enough to burn,
#     so f_T = 0; tritium is exhausted (or held until decay).
#   - 99% He-3 recovery between shots; D-T cycle (and its 14 MeV neutron)
#     deliberately not closed, keeping the design neutron-averse.
#   - D-rich mix (n_D/n_He3 ≈ 3) at T ≈ 100 keV minimizes bremsstrahlung;
#     f_DD = 0.314, f_brem = 0.163 from Bosch-Hale + relativistic brem
#     (examples/dhe3_mix_optimization.py).
#   - eta_th = 0: no thermal bottoming cycle. Helion has no turbine; wall
#     heat from bremsstrahlung and DEC waste is dumped to cooling.
PI_BURN = dict(
    dhe3_f_T=0.0,
    dhe3_dd_frac=0.314,
    f_rad_fus=0.163,  # bypass cc.f_rad_fus_dhe3 (which doesn't propagate)
    eta_th=0.0,
)
r_pi_raw = m_dhe3.forward(
    **BASE_KW, inflation_rate=INFLATION, f_dec=0.95, eta_de=0.85, **PI_BURN
)
vb_dec_cost = float(r_pi_raw.cas22_detail.get("C220109", 0.0))
pi_cas22 = float(r_pi_raw.costs.cas22) - vb_dec_cost + 593.0
r_dhe3_pi = m_dhe3.forward(
    **BASE_KW,
    inflation_rate=INFLATION,
    f_dec=0.95,
    eta_de=0.85,
    **PI_BURN,
    cost_overrides={
        "CAS22": pi_cas22,
        "CAS23": 0.0,
        "CAS21": float(r_pi_raw.costs.cas21) * 0.75,
        "CAS26": float(r_pi_raw.costs.cas26) * 0.15,
        "CAS24": float(r_pi_raw.costs.cas24) - 18.0,
    },
)

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
# Pulsed DEC hardware (C220109), dual-cap topology, 1 GWe D-He3:
#   Recovery cap bank  = $0.50/J x 702.6 MJ stored                 = $351M
#   DC-DC links        = $75/kW x 1034 MW gross                    = $78M
#   Grid inverter      = $150/kW x 1000 MW net                     = $150M
#   Controls / FPGA    = 4% of compression bank                    = $14M
#                                                                  -------
#                                                                   $593M
# Compression bank itself is in C220107 (driver, free in this run).
inv_dhe3 = 593.0
cas21_p_dhe3 = float(r_th_dhe3.costs.cas21) * 0.75
cas26_p_dhe3 = float(r_th_dhe3.costs.cas26) * 0.15
# Deduct synchronous-generator GSU (~$15M) and sync/protection gear (~$3M)
# from CAS24: pulsed inductive has no synchronous generator.
cas24_p_dhe3 = float(r_th_dhe3.costs.cas24) - 18.0
r_p_dhe3 = m_dhe3.forward(
    **BASE_KW,
    inflation_rate=INFLATION,
    f_dec=0.95,
    eta_de=0.85,
    **PI_BURN,
    cost_overrides={
        "CAS22": inv_dhe3,
        "CAS27": 0.0,
        "CAS23": 0.0,
        "CAS24": cas24_p_dhe3,
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
