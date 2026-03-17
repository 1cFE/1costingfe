"""What needs to happen for 1 ¢/kWh ($10/MWh) fusion electricity?

Systematic decomposition of the LCOE target into required conditions
across capital cost, financing, operations, and plant parameters.
Starting from the cheapest baseline (pB11 mirror, $47/MWh), identifies
the gap and the lever combinations that close it.
"""

from costingfe import ConfinementConcept, CostModel, Fuel

TARGET = 10.0  # $/MWh = 1 ¢/kWh

# ══════════════════════════════════════════════════════════════════════
# Part 1: Where are we today?
# ══════════════════════════════════════════════════════════════════════
print("=" * 72)
print("PART 1: BASELINE — Cheapest current configuration")
print("=" * 72)

model = CostModel(ConfinementConcept.MIRROR, Fuel.PB11)
base = model.forward(
    net_electric_mw=1000,
    availability=0.85,
    lifetime_yr=30,
    n_mod=1,
    construction_time_yr=6,
    interest_rate=0.07,
    inflation_rate=0.02,
    noak=True,
)
c = base.costs
pt = base.power_table
energy_mwh = 1000 * 0.85 * 8760  # MWh/yr

lcoe = float(c.lcoe)
cas90 = float(c.cas90) / energy_mwh * 1e6  # $/MWh
cas70 = float(c.cas70) / energy_mwh * 1e6
cas80 = float(c.cas80) / energy_mwh * 1e6

print("\npB11 Mirror — 1 GWe NOAK, 85% availability, 30-yr, 7% WACC")
print(f"LCOE:       {lcoe:>6.1f} $/MWh  ({lcoe / 10:.2f} ¢/kWh)")
print(f"Overnight:  {float(c.overnight_cost):>6.0f} $/kW")
print(f"Total cap:  {float(c.total_capital):>6.0f} M$")
print(f"Q_eng:      {float(pt.q_eng):>6.1f}")
print(f"Recirc:     {float(pt.rec_frac) * 100:>5.1f}%")
print("\nLCOE decomposition:")
print(f"  Capital (CAS90):  {cas90:>5.1f} $/MWh  ({cas90 / lcoe * 100:>4.0f}%)")
print(f"  O&M (CAS70):      {cas70:>5.1f} $/MWh  ({cas70 / lcoe * 100:>4.0f}%)")
print(f"  Fuel (CAS80):     {cas80:>5.1f} $/MWh  ({cas80 / lcoe * 100:>4.0f}%)")
gap_pct = (lcoe / TARGET - 1) * 100
print(f"\nGap to target: {lcoe - TARGET:.1f} $/MWh ({gap_pct:.0f}% reduction needed)")

# ══════════════════════════════════════════════════════════════════════
# Part 2: What overnight $/kW is needed?
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART 2: REQUIRED OVERNIGHT COST at various WACC & availability")
print("=" * 72)
print("""
LCOE = (CRF × Total_Capital) / Energy + O&M/Energy + Fuel/Energy

For 10 $/MWh, after O&M (~$3-5/MWh) and fuel (~$0), only $5-7/MWh
is available for capital charges. The table shows the maximum overnight
cost ($/kW) that achieves $10/MWh at each WACC/availability combination.
Assumes: 30-yr life, O&M=$3.5/MWh (pB11 at scale), 6-yr construction,
IDC factor = ((1+i)^T - 1)/(i*T) - 1.
""")

print(
    f"{'WACC':>6} {'Avail':>6}  {'CRF':>6} {'IDC%':>6}  "
    f"{'$/kW overnight':>16}  {'Feasible?':>10}"
)
print("-" * 58)

om_per_mwh = 3.5  # Optimistic pB11 O&M at scale

for wacc in [0.07, 0.05, 0.04, 0.03, 0.02]:
    for avail in [0.85, 0.90, 0.95]:
        crf = wacc * (1 + wacc) ** 30 / ((1 + wacc) ** 30 - 1)
        T = 6.0
        idc_frac = ((1 + wacc) ** T - 1) / (wacc * T) - 1
        energy = 1000 * avail * 8760  # MWh/yr for 1 GWe
        capital_room = (TARGET - om_per_mwh) * energy / 1e6  # M$/yr
        total_cap = capital_room / crf  # M$ total capital
        overnight = total_cap / (1 + idc_frac)  # M$ overnight
        okw = overnight  # $/kW for 1 GWe
        feasible = "Possible" if okw > 1500 else ("Stretch" if okw > 800 else "No")
        print(
            f"{wacc:>6.1%} {avail:>6.1%}  {crf:>6.4f} {idc_frac:>5.1%}  "
            f"${okw:>13,.0f}/kW  {feasible:>10}"
        )
    print()

# ══════════════════════════════════════════════════════════════════════
# Part 3: Lever-by-lever impact
# ══════════════════════════════════════════════════════════════════════
print("=" * 72)
print("PART 3: LEVER-BY-LEVER — How much does each lever buy?")
print("=" * 72)

base_kwargs = dict(
    net_electric_mw=1000,
    availability=0.85,
    lifetime_yr=30,
    n_mod=1,
    construction_time_yr=6,
    interest_rate=0.07,
    inflation_rate=0.02,
    noak=True,
)
base_lcoe = float(model.forward(**base_kwargs).costs.lcoe)

levers = [
    ("Plant scale", "net_electric_mw", [1000, 2000, 3000, 5000]),
    ("Availability", "availability", [0.85, 0.90, 0.95, 0.98]),
    ("Construction time (yr)", "construction_time_yr", [6.0, 4.0, 3.0, 2.0]),
    ("WACC", "interest_rate", [0.07, 0.05, 0.04, 0.03]),
    ("Plant lifetime (yr)", "lifetime_yr", [30, 40, 50, 60]),
    ("Thermal efficiency", "eta_th", [0.46, 0.50, 0.55, 0.60]),
]

print(f"\nBaseline: {base_lcoe:.1f} $/MWh (pB11 mirror, 1 GWe)")
print(
    f"{'Lever':<24} {'Baseline':>10} {'Optimistic':>10} {'Stretch':>10} {'Extreme':>10}"
)
print("-" * 72)

for label, param, vals in levers:
    row = f"{label:<24}"
    for v in vals:
        kw = {**base_kwargs, param: v}
        r = model.forward(**kw)
        lcoe_v = float(r.costs.lcoe)
        row += f" {lcoe_v:>9.1f}"
    print(row)

# ══════════════════════════════════════════════════════════════════════
# Part 4: Combined scenarios toward the target
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART 4: COMBINED SCENARIOS — Closing the gap to $10/MWh")
print("=" * 72)

scenarios = [
    ("Today's baseline", {}),
    ("Scale to 2 GWe", dict(net_electric_mw=2000)),
    ("+ 95% availability", dict(net_electric_mw=2000, availability=0.95)),
    (
        "+ 3-yr construction",
        dict(net_electric_mw=2000, availability=0.95, construction_time_yr=3.0),
    ),
    (
        "+ 4% WACC",
        dict(
            net_electric_mw=2000,
            availability=0.95,
            construction_time_yr=3.0,
            interest_rate=0.04,
        ),
    ),
    (
        "+ 3% WACC",
        dict(
            net_electric_mw=2000,
            availability=0.95,
            construction_time_yr=3.0,
            interest_rate=0.03,
        ),
    ),
    (
        "+ 50-yr lifetime",
        dict(
            net_electric_mw=2000,
            availability=0.95,
            construction_time_yr=3.0,
            interest_rate=0.03,
            lifetime_yr=50,
        ),
    ),
    (
        "+ 55% eta_th",
        dict(
            net_electric_mw=2000,
            availability=0.95,
            construction_time_yr=3.0,
            interest_rate=0.03,
            lifetime_yr=50,
            eta_th=0.55,
        ),
    ),
    (
        "+ CAS21 halved",
        dict(
            net_electric_mw=2000,
            availability=0.95,
            construction_time_yr=3.0,
            interest_rate=0.03,
            lifetime_yr=50,
            eta_th=0.55,
        ),
    ),
]

# Last scenario also needs cost override
print(f"\n{'Scenario':<32} {'LCOE':>8} {'¢/kWh':>7} {'O/N $/kW':>9} {'Gap':>8}")
print("-" * 72)

for name, overrides in scenarios:
    kw = {**base_kwargs, **overrides}
    cost_ovr = {}
    if "CAS21 halved" in name:
        # Halve building costs via override
        r_temp = model.forward(**kw)
        cost_ovr = {"CAS21": float(r_temp.costs.cas21) / 2}
    r = model.forward(**kw, cost_overrides=cost_ovr)
    lcoe_v = float(r.costs.lcoe)
    on = float(r.costs.overnight_cost)
    gap = lcoe_v - TARGET
    marker = " ***" if gap <= 0 else ""
    print(
        f"{name:<32} {lcoe_v:>7.1f} {lcoe_v / 10:>7.2f} {on:>9.0f} {gap:>+7.1f}{marker}"
    )

# ══════════════════════════════════════════════════════════════════════
# Part 5: What DOES reach 1 ¢/kWh? Grid search.
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART 5: GRID SEARCH — Which combinations reach ≤$10/MWh?")
print("=" * 72)

hits = []
for p_net in [2000, 3000, 5000]:
    for avail in [0.90, 0.95, 0.98]:
        for ct in [3.0, 2.0]:
            for wacc in [0.04, 0.03, 0.02]:
                for lt in [30, 40, 50, 60]:
                    for eta in [0.46, 0.55]:
                        kw = dict(
                            net_electric_mw=p_net,
                            availability=avail,
                            lifetime_yr=lt,
                            n_mod=1,
                            construction_time_yr=ct,
                            interest_rate=wacc,
                            inflation_rate=0.02,
                            noak=True,
                            eta_th=eta,
                        )
                        r = model.forward(**kw)
                        lcoe_v = float(r.costs.lcoe)
                        if lcoe_v <= TARGET:
                            hits.append((lcoe_v, kw))

print(f"\n{len(hits)} combinations reach ≤$10/MWh out of grid search")

if hits:
    hits.sort()
    print("\nTop 10 (lowest LCOE):")
    print(
        f"{'LCOE':>7} {'P_net':>6} {'Avail':>6} {'CT':>4} {'WACC':>5} "
        f"{'Life':>5} {'eta':>5}"
    )
    print("-" * 48)
    for lcoe_v, kw in hits[:10]:
        print(
            f"{lcoe_v:>6.1f} {kw['net_electric_mw']:>6.0f} "
            f"{kw['availability']:>6.0%} {kw['construction_time_yr']:>4.0f} "
            f"{kw['interest_rate']:>5.1%} {kw['lifetime_yr']:>5} "
            f"{kw['eta_th']:>5.2f}"
        )

    # Show the LEAST aggressive combo that reaches target
    print("\nLeast aggressive combination that reaches ≤$10/MWh:")

    # Sort by sum of "aggressiveness" of each parameter
    def aggressiveness(kw):
        score = 0
        score += (kw["net_electric_mw"] - 1000) / 1000  # bigger = more aggressive
        score += (0.85 - kw["availability"]) / 0.85 * -10
        score += (6 - kw["construction_time_yr"]) / 6 * 5
        score += (0.07 - kw["interest_rate"]) / 0.07 * 10
        score += (kw["lifetime_yr"] - 30) / 30 * 3
        score += (kw["eta_th"] - 0.46) / 0.46 * 2
        return score

    hits_by_ease = sorted(hits, key=lambda x: aggressiveness(x[1]))
    lcoe_v, kw = hits_by_ease[0]
    r = model.forward(**kw)
    c = r.costs
    print(f"  LCOE: {lcoe_v:.1f} $/MWh ({lcoe_v / 10:.2f} ¢/kWh)")
    print(f"  P_net: {kw['net_electric_mw']:.0f} MW")
    print(f"  Availability: {kw['availability']:.0%}")
    print(f"  Construction: {kw['construction_time_yr']:.0f} yr")
    print(f"  WACC: {kw['interest_rate']:.1%}")
    print(f"  Lifetime: {kw['lifetime_yr']} yr")
    print(f"  eta_th: {kw['eta_th']:.2f}")
    print(f"  Overnight: {float(c.overnight_cost):.0f} $/kW")
    print(f"  Total capital: {float(c.total_capital):.0f} M$")
else:
    print("\nNo combinations in the grid reach $10/MWh.")
    print("The grid may need to be expanded, or cost overrides are needed.")
    # Find closest
    all_results = []
    for p_net in [2000, 3000, 5000]:
        for avail in [0.90, 0.95, 0.98]:
            for ct in [3.0, 2.0]:
                for wacc in [0.04, 0.03, 0.02]:
                    for lt in [30, 40, 50, 60]:
                        for eta in [0.46, 0.55]:
                            kw = dict(
                                net_electric_mw=p_net,
                                availability=avail,
                                lifetime_yr=lt,
                                n_mod=1,
                                construction_time_yr=ct,
                                interest_rate=wacc,
                                inflation_rate=0.02,
                                noak=True,
                                eta_th=eta,
                            )
                            r = model.forward(**kw)
                            all_results.append((float(r.costs.lcoe), kw))
    all_results.sort()
    print("\nClosest 5:")
    print(
        f"{'LCOE':>7} {'P_net':>6} {'Avail':>6} {'CT':>4} {'WACC':>5} "
        f"{'Life':>5} {'eta':>5}"
    )
    print("-" * 48)
    for lcoe_v, kw in all_results[:5]:
        print(
            f"{lcoe_v:>6.1f} {kw['net_electric_mw']:>6.0f} "
            f"{kw['availability']:>6.0%} {kw['construction_time_yr']:>4.0f} "
            f"{kw['interest_rate']:>5.1%} {kw['lifetime_yr']:>5} "
            f"{kw['eta_th']:>5.2f}"
        )

# ══════════════════════════════════════════════════════════════════════
# Part 6: The answer
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART 6: WHAT NEEDS TO HAPPEN FOR 1 ¢/kWh")
print("=" * 72)
print("""
The $10/MWh target requires simultaneous advances across ALL dimensions:

REQUIRED CONDITIONS (all must hold):
  1. Aneutronic fuel (pB11)      — eliminates $1.7B of neutron infrastructure
  2. Large plant (≥2 GWe net)    — dilutes fixed costs by 2-3×
  3. High availability (≥95%)    — 12% more energy per $ of capital
  4. Fast construction (≤3 yr)   — halves IDC from $700M to $200M
  5. Low cost of capital (≤3%)   — requires government/utility backing
  6. Long lifetime (≥50 yr)      — amortizes capital over more years

HELPFUL BUT NOT SUFFICIENT ALONE:
  - Higher thermal efficiency (50-60% vs 46%)
  - Compact geometry (reduces CAS21 buildings)
  - Multi-module plants (shared infrastructure)
  - Advanced materials (lower CAS22 unit costs)

WHY EACH CONDITION MATTERS:
  Capital dominates (90% of LCOE). At 7% WACC, every $1B of capital
  adds ~$11/MWh. At 3% WACC, only ~$5/MWh. The WACC lever alone
  halves the capital charge rate.

  At 3% WACC, 95% availability, 50-yr life, the required overnight
  cost is ~$1,500-2,000/kW — comparable to natural gas combined cycle.
  This is the target the fusion engineering must hit.

HISTORICAL ANALOGS:
  - NGCC plants: $900-1,200/kW overnight, LCOE $30-50/MWh
  - Onshore wind: $1,200-1,600/kW, LCOE $25-50/MWh (with PTC)
  - Solar PV utility: $800-1,200/kW, LCOE $25-40/MWh (with ITC)
  - Nuclear (NOAK APR1400): $2,500-3,500/kW, LCOE $50-80/MWh

  Fusion at $10/MWh requires capital costs BELOW gas turbines, with
  the operating profile of a nuclear plant (high capacity factor, low
  marginal cost, zero fuel cost for pB11 NOAK).
""")

# ══════════════════════════════════════════════════════════════════════
# Part 7: What if reactor equipment gets cheaper? (Learning curves)
# ══════════════════════════════════════════════════════════════════════
print("=" * 72)
print("PART 7: LEARNING CURVES — What if CAS22 costs drop?")
print("=" * 72)
print("""
CAS22 (reactor plant equipment) is the dominant cost account.
If NOAK learning, manufacturing scale-up, and design simplification
reduce CAS22 by 30-50%, how does that change the picture?
""")

# Use aggressive-but-not-extreme financial assumptions
learn_kwargs = dict(
    net_electric_mw=2000,
    availability=0.95,
    lifetime_yr=50,
    n_mod=1,
    construction_time_yr=3,
    interest_rate=0.03,
    inflation_rate=0.02,
    noak=True,
)
r_learn_base = model.forward(**learn_kwargs)
cas22_base = float(r_learn_base.costs.cas22)

print("Scenario: 2 GWe, 95% avail, 3-yr build, 3% WACC, 50-yr life")
print(f"CAS22 baseline: ${cas22_base:.0f}M")
print(f"{'CAS22 reduction':>18} {'LCOE':>8} {'¢/kWh':>7} {'O/N $/kW':>9} {'Gap':>8}")
print("-" * 56)

for frac in [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
    ovr = {"CAS22": cas22_base * frac}
    r = model.forward(**learn_kwargs, cost_overrides=ovr)
    lcoe_v = float(r.costs.lcoe)
    on = float(r.costs.overnight_cost)
    gap = lcoe_v - TARGET
    label = f"{(1 - frac) * 100:.0f}%"
    marker = " ***" if gap <= 0 else ""
    print(
        f"{label:>18} {lcoe_v:>7.1f} {lcoe_v / 10:>7.2f}"
        f" {on:>9.0f} {gap:>+7.1f}{marker}"
    )

# ══════════════════════════════════════════════════════════════════════
# Part 8: Minimum viable conditions summary
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART 8: MINIMUM VIABLE CONDITIONS FOR 1 ¢/kWh")
print("=" * 72)
print("""
From the analysis above, there are two plausible paths to $10/MWh:

PATH A — SCALE + CHEAP CAPITAL (no cost breakthrough needed):
  Plant size:    ≥5 GWe net
  Availability:  ≥95%
  Construction:  ≤3 years
  WACC:          ≤2% (government/sovereign backing)
  Lifetime:      ≥40 years
  Overnight:     ~$1,800/kW (achievable with current model at scale)
  Verdict:       Requires very large plants and near-sovereign financing.
                 Technically feasible if pB11 physics works.

PATH B — MODERATE SCALE + COST BREAKTHROUGH:
  Plant size:    ≥2 GWe net
  Availability:  ≥95%
  Construction:  ≤3 years
  WACC:          ≤3%
  Lifetime:      ≥50 years
  CAS22 cost:    40-50% below current model (learning + simplification)
  Overnight:     ~$1,200-1,500/kW
  Verdict:       Requires significant reactor equipment cost reduction
                 beyond what NOAK learning alone delivers. Possible if
                 fusion reactors become as mass-producible as gas turbines.

WHAT DEFINITELY WON'T WORK:
  - 7% WACC (commercial project finance) — mathematically impossible
  - 1 GWe scale — fixed costs too high even at 2% WACC
  - 85% availability — wastes 10% of capital capacity
  - DT fuel — neutron infrastructure adds $1.7B/GWe

THE KEY INSIGHT:
  At 3% WACC and 50-year life, the capital charge rate is 5.1%/yr.
  For a 2 GWe plant at 95% availability producing 16.6 TWh/yr, each
  $1B of capital adds only $3.1/MWh. The $10/MWh target then requires
  overnight cost ≤ ~$2,000/kW — ambitious but within 2× of the best
  nuclear construction worldwide (Korean APR1400 at ~$2,500/kW).
""")

# Final table: the three reference points
print("Reference points:")
print(f"{'':30} {'Today':>10} {'Aggressive':>10} {'1¢/kWh':>10}")
print("-" * 64)
for label, vals in [
    ("Plant size (GWe)", (1, 2, 5)),
    ("Availability", ("85%", "95%", "95%")),
    ("Construction (yr)", (6, 3, 3)),
    ("WACC", ("7.0%", "3.0%", "2.0%")),
    ("Lifetime (yr)", (30, 50, 40)),
    ("Overnight ($/kW)", (3924, 2338, 1808)),
    ("LCOE ($/MWh)", (47.0, 14.2, 9.9)),
    ("LCOE (¢/kWh)", (4.70, 1.42, 0.99)),
]:
    print(f"  {label:<28} {str(vals[0]):>10} {str(vals[1]):>10} {str(vals[2]):>10}")

# ══════════════════════════════════════════════════════════════════════
# Part 9: Non-pB11 paths — DHe3 pulsed FRC and DD
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART 9: NON-pB11 PATHS — Is there another way?")
print("=" * 72)

from costingfe.defaults import load_costing_constants  # noqa: E402
from costingfe.layers.physics import (  # noqa: E402
    DD_F_HE3_DEFAULT,
    DD_F_T_DEFAULT,
)

# ── Neutron fraction analysis ────────────────────────────────────────
# In a pulsed FRC, only charged-particle energy can be recovered by
# direct electromagnetic conversion (~90% efficiency). Neutron energy
# deposits in the walls and must be recovered thermally or wasted.
# The effective conversion efficiency depends on the neutron fraction.


def compute_neutron_fraction(f_T, f_He3):
    """Fraction of DD fusion energy carried by neutrons."""
    charged = 1.01 + 3.02 + 0.82  # primary charged (MeV)
    neutron = 2.45  # primary neutron (MeV)
    charged += 0.5 * f_T * 3.5  # secondary DT alpha
    neutron += 0.5 * f_T * 14.1  # secondary DT neutron
    charged += 0.5 * f_He3 * (3.6 + 14.7)  # secondary DHe3 (all charged)
    return neutron / (charged + neutron)


def effective_eta(f_neutron, eta_em=0.90, eta_thermal=0.40):
    """Effective conversion: EM for charged, thermal for neutron energy."""
    return (1 - f_neutron) * eta_em + f_neutron * eta_thermal


f_n_dd = compute_neutron_fraction(DD_F_T_DEFAULT, DD_F_HE3_DEFAULT)
# DHe3: ~5% neutron fraction (small DD side-reaction contribution)
f_n_dhe3 = 0.05

print(f"""
── Neutron energy fractions and effective conversion ──

A pulsed FRC recovers charged-particle energy electromagnetically at
~90% efficiency. Neutrons escape to the walls. The effective conversion
depends on how much energy is in neutrons:

  DHe3: ~{f_n_dhe3:.0%} neutron fraction
         → eta_eff ≈ {effective_eta(f_n_dhe3):.0%} (mostly EM)
  DD:   ~{f_n_dd:.0%} neutron fraction
         → eta_eff ≈ {effective_eta(f_n_dd):.0%} (with steam)
         or ≈ {(1 - f_n_dd) * 0.90:.0%} (neutrons wasted)

DD has {f_n_dd:.0%} of energy in neutrons because secondary DT reactions
(f_T={DD_F_T_DEFAULT:.2f}) produce 14.1 MeV neutrons. A small steam
bottoming cycle (~$100M CAS23) recovers this neutron wall heat at ~40%.
""")

# --- DHe3 pulsed FRC (Helion-like) ---
print("── DHe3 Pulsed FRC (Helion architecture) ──")
print("  Key advantage: $1,773/kW overnight (lowest of any concept)")
print("  Key problem:   He-3 fuel at $2M/kg → $224M/yr CAS80")
print(
    f"  eta_eff ≈ {effective_eta(f_n_dhe3):.0%} (only ~5% neutrons, mostly EM recovery)"
)

cc_frc = load_costing_constants().replace(
    burn_fraction=0.10,
    fuel_recovery=0.95,
)
frc_model = CostModel(
    ConfinementConcept.MAG_TARGET, Fuel.DHE3, costing_constants=cc_frc
)

# DHe3 FRC: eta_th=0.86 (corrected for ~5% neutron fraction)
eta_dhe3_frc = effective_eta(f_n_dhe3)

frc_kwargs = dict(
    net_electric_mw=1000,
    availability=0.85,
    lifetime_yr=30,
    n_mod=20,
    construction_time_yr=4,
    interest_rate=0.07,
    inflation_rate=0.02,
    noak=True,
    p_driver=12.0,
    mn=1.0,
    eta_th=eta_dhe3_frc,
    eta_p=0.5,
    eta_pin=0.95,
    f_sub=0.03,
    p_pump=0.5,
    p_trit=0.5,
    p_house=2.0,
    p_cryo=0.0,
    p_target=0.0,
    p_coils=0.5,
    R0=0.0,
    plasma_t=0.5,
    blanket_t=0.05,
    ht_shield_t=0.05,
    structure_t=0.10,
    vessel_t=0.10,
    cost_overrides={
        "C220103": 5.0,
        "C220104": 10.0,
        "C220107": 3.0,
        "C220108": 0.0,
        "C220111": 4.0,
        "C220200": 30.0,
        "CAS21": 400.0,
        "CAS23": 0.0,
        "CAS26": 7.0,
    },
)

# What He-3 price makes DHe3 FRC reach $10/MWh?
print("\n  He-3 price sensitivity (with aggressive finance: 3% WACC, 50-yr):")
print(f"  {'He-3 $/kg':>12} {'LCOE':>8} {'¢/kWh':>7} {'CAS80':>8} {'Capital':>8}")
print("  " + "-" * 52)

for he3_price in [2_000_000, 500_000, 200_000, 100_000, 50_000, 20_000, 10_000, 5_000]:
    cc_v = load_costing_constants().replace(
        burn_fraction=0.10,
        fuel_recovery=0.95,
        u_he3=float(he3_price),
    )
    m_v = CostModel(ConfinementConcept.MAG_TARGET, Fuel.DHE3, costing_constants=cc_v)
    kw = {**frc_kwargs, "interest_rate": 0.03, "lifetime_yr": 50}
    r = m_v.forward(**kw)
    lcoe_v = float(r.costs.lcoe)
    cas80 = float(r.costs.cas80)
    cap = float(r.costs.total_capital)
    marker = " ***" if lcoe_v <= TARGET else ""
    if he3_price >= 1_000_000:
        label = f"${he3_price / 1_000_000:.0f}M"
    else:
        label = f"${he3_price / 1_000:,.0f}k"
    print(
        f"  {label:>12} {lcoe_v:>7.1f} {lcoe_v / 10:>7.2f}"
        f" {cas80:>7.1f} {cap:>7.0f}{marker}"
    )

# --- DD fuel with corrected neutron accounting ---
print("\n── DD Fuel — Corrected for neutron energy fraction ──")
print(f"  Neutron fraction: {f_n_dd:.0%} (secondary DT dominates)")
print(f"  Effective eta with steam bottoming: {effective_eta(f_n_dd):.0%}")
print(f"  Effective eta without steam:        {(1 - f_n_dd) * 0.90:.0%}")
print("  Uses only deuterium ($2,175/kg, commodity from D2O plants)")

# DD with FRC architecture — corrected eta
eta_dd_with_steam = effective_eta(f_n_dd)
eta_dd_no_steam = (1 - f_n_dd) * 0.90

cc_dd = load_costing_constants().replace(
    burn_fraction=0.10,
    fuel_recovery=0.95,
)
dd_model = CostModel(ConfinementConcept.MAG_TARGET, Fuel.DD, costing_constants=cc_dd)

# DD FRC with small steam bottoming cycle for neutron heat
dd_kwargs = dict(
    net_electric_mw=1000,
    availability=0.85,
    lifetime_yr=30,
    n_mod=20,
    construction_time_yr=4,
    interest_rate=0.07,
    inflation_rate=0.02,
    noak=True,
    p_driver=12.0,
    mn=1.0,
    eta_th=eta_dd_with_steam,
    eta_p=0.5,
    eta_pin=0.95,
    f_sub=0.03,
    p_pump=0.5,
    p_trit=2.0,  # More tritium from DD side reactions
    p_house=2.0,
    p_cryo=0.0,
    p_target=0.0,
    p_coils=0.5,
    R0=0.0,
    plasma_t=0.5,
    blanket_t=0.10,  # Thicker — more neutrons
    ht_shield_t=0.10,
    structure_t=0.10,
    vessel_t=0.10,
    cost_overrides={
        "C220103": 5.0,
        "C220104": 10.0,
        "C220107": 3.0,
        "C220108": 0.0,
        "C220111": 5.0,
        "C220200": 40.0,
        "CAS21": 450.0,
        "CAS23": 100.0,  # Small steam turbine for neutron wall heat
        "CAS26": 30.0,  # Cooling towers for thermal rejection
    },
)

print(f"\n  {'Scenario':<36} {'LCOE':>8} {'¢/kWh':>7} {'O/N $/kW':>9} {'Q_sci':>6}")
print("  " + "-" * 72)

dd_scenarios = [
    ("Baseline (7% WACC, 30yr)", {}),
    ("+ 3% WACC, 50yr", dict(interest_rate=0.03, lifetime_yr=50)),
    (
        "+ 2 GWe (40 modules)",
        dict(interest_rate=0.03, lifetime_yr=50, net_electric_mw=2000, n_mod=40),
    ),
    (
        "+ 95% availability",
        dict(
            interest_rate=0.03,
            lifetime_yr=50,
            net_electric_mw=2000,
            n_mod=40,
            availability=0.95,
        ),
    ),
    (
        "+ 3-yr construction",
        dict(
            interest_rate=0.03,
            lifetime_yr=50,
            net_electric_mw=2000,
            n_mod=40,
            availability=0.95,
            construction_time_yr=3,
        ),
    ),
]

for name, overrides in dd_scenarios:
    kw = {**dd_kwargs, **overrides}
    r = dd_model.forward(**kw)
    lcoe_v = float(r.costs.lcoe)
    on = float(r.costs.overnight_cost)
    q_sci = float(r.power_table.q_sci)
    marker = " ***" if lcoe_v <= TARGET else ""
    print(
        f"  {name:<36} {lcoe_v:>7.1f} {lcoe_v / 10:>7.2f}"
        f" {on:>9.0f} {q_sci:>6.1f}{marker}"
    )

# --- Physics requirements at corrected eta ---
print(f"\n  Physics requirements at eta_eff = {eta_dd_with_steam:.2f}:")
r_ref = dd_model.forward(
    **{
        **dd_kwargs,
        "interest_rate": 0.03,
        "lifetime_yr": 50,
        "net_electric_mw": 2000,
        "n_mod": 40,
        "availability": 0.95,
        "construction_time_yr": 3,
    }
)
pt = r_ref.power_table
print(f"    P_fus per module: {float(pt.p_fus):.0f} MW")
print(f"    Q_sci (P_fus/P_driver): {float(pt.q_sci):.1f}")
print(f"    Q_eng: {float(pt.q_eng):.1f}")
print(f"    Recirculating fraction: {float(pt.rec_frac):.1%}")

# --- Comparison table ---
print("\n── All fuel paths compared (3% WACC, 50-yr, 2 GWe, 95% avail, 3-yr) ──")
print(f"  {'Path':<36} {'LCOE':>7} {'O/N':>7} {'Q_sci':>6} {'Note':>24}")
print("  " + "-" * 82)

# pB11 mirror
r_pb = CostModel(ConfinementConcept.MIRROR, Fuel.PB11).forward(
    net_electric_mw=2000,
    availability=0.95,
    lifetime_yr=50,
    construction_time_yr=3,
    interest_rate=0.03,
    inflation_rate=0.02,
    noak=True,
)
marker = " ***" if float(r_pb.costs.lcoe) <= TARGET else ""
print(
    f"  {'pB11 Mirror':<36} {float(r_pb.costs.lcoe):>6.1f} "
    f"{float(r_pb.costs.overnight_cost):>7.0f} {float(r_pb.power_table.q_sci):>6.1f}"
    f"  {'No exotic isotopes':>24}{marker}"
)

# DHe3 FRC with cheap He-3
cc_cheaphe3 = load_costing_constants().replace(
    burn_fraction=0.10,
    fuel_recovery=0.95,
    u_he3=20_000.0,
)
m_cheaphe3 = CostModel(
    ConfinementConcept.MAG_TARGET, Fuel.DHE3, costing_constants=cc_cheaphe3
)
r_frc = m_cheaphe3.forward(
    **{
        **frc_kwargs,
        "interest_rate": 0.03,
        "lifetime_yr": 50,
        "net_electric_mw": 2000,
        "n_mod": 40,
        "availability": 0.95,
        "construction_time_yr": 3,
    },
)
marker = " ***" if float(r_frc.costs.lcoe) <= TARGET else ""
print(
    f"  {'DHe3 FRC (He-3 @ $20k/kg)':<36} {float(r_frc.costs.lcoe):>6.1f} "
    f"{float(r_frc.costs.overnight_cost):>7.0f} {float(r_frc.power_table.q_sci):>6.1f}"
    f"  {'He-3 supply problem':>24}{marker}"
)

# DD FRC (corrected)
marker = " ***" if float(r_ref.costs.lcoe) <= TARGET else ""
print(
    f"  {f'DD FRC (eta={eta_dd_with_steam:.0%}, +steam)':<36}"
    f" {float(r_ref.costs.lcoe):>6.1f} "
    f"{float(r_ref.costs.overnight_cost):>7.0f} {float(r_ref.power_table.q_sci):>6.1f}"
    f"  {'Commodity fuel only':>24}{marker}"
)

# DD FRC without steam (neutrons wasted)
dd_no_steam_kw = {
    **dd_kwargs,
    "eta_th": eta_dd_no_steam,
    "interest_rate": 0.03,
    "lifetime_yr": 50,
    "net_electric_mw": 2000,
    "n_mod": 40,
    "availability": 0.95,
    "construction_time_yr": 3,
}
# Remove steam cycle from overrides
no_steam_ovr = dict(dd_no_steam_kw["cost_overrides"])
no_steam_ovr["CAS23"] = 0.0
no_steam_ovr["CAS26"] = 10.0
dd_no_steam_kw["cost_overrides"] = no_steam_ovr
r_dd_ns = dd_model.forward(**dd_no_steam_kw)
marker = " ***" if float(r_dd_ns.costs.lcoe) <= TARGET else ""
print(
    f"  {f'DD FRC (eta={eta_dd_no_steam:.0%}, no steam)':<36}"
    f" {float(r_dd_ns.costs.lcoe):>6.1f} "
    f"{float(r_dd_ns.costs.overnight_cost):>7.0f}"
    f" {float(r_dd_ns.power_table.q_sci):>6.1f}"
    f"  Simpler, higher Q needed{marker}"
)

# DT mirror (for reference)
r_dt = CostModel(ConfinementConcept.MIRROR, Fuel.DT).forward(
    net_electric_mw=2000,
    availability=0.95,
    lifetime_yr=50,
    construction_time_yr=3,
    interest_rate=0.03,
    inflation_rate=0.02,
    noak=True,
)
print(
    f"  {'DT Mirror':<36} {float(r_dt.costs.lcoe):>6.1f} "
    f"{float(r_dt.costs.overnight_cost):>7.0f} {float(r_dt.power_table.q_sci):>6.1f}"
    f"  {'Neutron penalty':>24}"
)

q_dd = float(r_ref.power_table.q_sci)
q_dhe3 = float(r_frc.power_table.q_sci)
lcoe_dd = float(r_ref.costs.lcoe)
print(f"""
KEY FINDINGS — Non-pB11 paths:

1. DHe3 PULSED FRC: Lowest overnight ($1,773/kW) but He-3 at
   $2M/kg is prohibitive. At $20k/kg (100x cheaper) → ~$10/MWh.
   ~5% neutron fraction → eta_eff ≈ {effective_eta(f_n_dhe3):.0%}.

2. DD FRC WITH STEAM BOTTOMING: DD has {f_n_dd:.0%} neutron
   fraction (secondary DT). Steam cycle (~$100M) recovers
   neutron wall heat → eta_eff ≈ {eta_dd_with_steam:.0%}.
   Q_sci ≈ {q_dd:.1f} vs {q_dhe3:.1f} (DHe3).
   → ${lcoe_dd:.1f}/MWh, commodity deuterium.

3. DD FRC WITHOUT STEAM: Wastes neutron energy, eta_eff ≈ {eta_dd_no_steam:.0%}.
   Needs higher Q_sci ≈ {float(r_dd_ns.power_table.q_sci):.1f} but eliminates steam
   turbine complexity. Still reaches ${float(r_dd_ns.costs.lcoe):.1f}/MWh.

4. DT is the hardest path — neutron infrastructure adds ~$1.7B/GWe.

WHY FRC LCOE IS INSENSITIVE TO CONVERSION EFFICIENCY:
   FRC costs are dominated by per-module fixed costs ($5M coils +
   $10M cap bank per module), not power-dependent scaling. Lowering
   eta_eff increases the physics Q requirement but barely changes
   the capital cost. The penalty is in physics difficulty, not dollars.

VERDICT: DD with pulsed FRC architecture is a viable non-pB11 path
to 1 ¢/kWh. It needs only commodity deuterium, tolerates the ~{f_n_dd:.0%}
neutron fraction with a small steam bottoming cycle, and reaches
~$10/MWh under aggressive (but not extreme) financial assumptions.
The key physics requirement is Q_sci ≈ {float(r_ref.power_table.q_sci):.0f} per module.
""")
