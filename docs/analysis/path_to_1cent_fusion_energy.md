# Path to 1 ¢/kWh Fusion Energy

**Date:** 2026-03-20
**Source model:** 1costingfe (pB11 mirror baseline, NOAK)

---

## Abstract

We decompose the \$10/MWh (1 ¢/kWh) levelized cost of electricity target for fusion power into its constituent requirements, using the 1costingfe techno-economic model. Starting from a baseline of \$43/MWh for a 1 GWe NOAK pB11 mirror plant, we identify the lever stack required to close the 4.3x gap. We examine two bounding cases: (i) a free fusion core, which isolates the irreducible balance-of-plant floor, and (ii) a fully costed core, which represents the engineering challenge. In both cases, direct energy conversion (DEC) emerges as a transformative lever by eliminating the steam cycle that dominates non-core costs. No single lever reaches the target; the path requires simultaneous advances in plant scale, availability, construction speed, financing, and — for charged-particle-dominated fuels — DEC.

---

## 1. Baseline: Where Are We Today?

The cheapest configuration in the current model is a 1 GWe NOAK pB11 mirror plant at standard financial assumptions (7% WACC, 85% availability, 30-year life, 6-year construction):

| Metric | Value |
|--------|-------|
| LCOE | \$42.9/MWh |
| Overnight cost | \$3,548/kW |
| CAS22 (reactor equipment) | \$1,563M |
| CAS21 (buildings) | \$332M |
| CAS23-26 (BOP) | \$421M |
| CAS70 (annual O&M) | \$24M/yr |

Capital charges (CAS90) account for ~90% of LCOE, with O&M contributing ~10% and fuel <1%. The gap to the \$10/MWh target is \$33/MWh — a 4.3x reduction.

Because capital dominates, the path to \$10/MWh is overwhelmingly a capital cost problem. O&M and fuel are nearly irrelevant at this level.

Note: CAS21 buildings are priced at industrial/enhanced-industrial grade appropriate for each fuel type under 10 CFR Part 30 — not nuclear-grade (Part 50). pB11 buildings at \$308/kW reflect zero hot cell, zero tritium infrastructure, standard HVAC, and industrial construction throughout. See `CAS21_buildings.md` for the full derivation.

---

## 2. Free Fusion Core: The Balance-of-Plant Floor

To isolate the irreducible non-core costs, we zero out all reactor plant equipment (CAS22 = 0, CAS27 = 0) and ask: what is the LCOE floor from balance-of-plant alone?

Even with a completely free heat source, the LCOE floor is **\$18/MWh (1.8 ¢/kWh)** for pB11. This floor comprises buildings (\$332M), BOP equipment (\$421M), indirect costs (\$152M), O&M (\$24M/yr base), and financing.

The fusion core accounts for 58% of baseline LCOE. The remaining 42% exists for *any* thermal power plant that converts heat to grid electricity through a steam cycle.

### 2.1 Closing the Gap Without a Fusion Core

Even with CAS22 = 0, the \$18/MWh floor is still 1.8x the target. Stacking levers:

| Cumulative scenario | LCOE (\$/MWh) | O/N (\$/kW) |
|---------------------|-------------|-----------|
| Free core (baseline) | 18 | 1,231 |
| + 2 GWe, 95%, 3% WACC, 3yr, 50yr | **7.3** | 854 |

The target is reached by stacking five levers: scale (2 GWe), availability (95%), cheap capital (3% WACC), fast construction (3 yr), and long life (50 yr).

### 2.2 Effect of Direct Energy Conversion

For pB11 fuel, ~99.8% of fusion energy is carried by charged alpha particles. Direct energy conversion (DEC) recovers this energy electromagnetically at 80-90% efficiency, bypassing the steam cycle. DEC eliminates most of CAS23 (turbines) and CAS26 (heat rejection).

| Scenario | LCOE (\$/MWh) | O/N (\$/kW) |
|----------|-------------|-----------|
| Steam: 2 GWe, 95%, 3% WACC, 3yr, 50yr | 7.3 | 854 |
| DEC: 2 GWe, 95%, 3% WACC, 3yr, 50yr | **5.7** | 524 |
| DEC: 5 GWe, 95%, 2% WACC, 3yr, 50yr | **3.9** | 448 |

At mega-scale with sovereign financing, the DEC floor reaches \$3.9/MWh — dominated almost entirely by O&M staff costs.

---

## 3. Fully Costed Fusion Core

With a realistic fusion core (CAS22 computed from first principles), the baseline LCOE is \$43/MWh for pB11 mirror. The CAS22 reactor equipment accounts for \$1,563M, comprising magnets, heating systems, vacuum vessel, power supplies, divertor, and installation labor.

### 3.1 Combined Scenario Stacking

| Cumulative scenario | LCOE (\$/MWh) | O/N (\$/kW) |
|---------------------|-------------|-----------|
| Baseline (1 GWe, 7% WACC, 30yr) | 42.9 | 3,548 |
| + 2 GWe, 95%, 3yr, 3% WACC, 50yr | **12.5** | 1,982 |

With all levers stacked and a steam cycle, the fully costed pB11 mirror reaches \$12.5/MWh — close but still above target.

### 3.2 Effect of Direct Energy Conversion

| Scenario | f\_dec | LCOE (\$/MWh) | O/N (\$/kW) |
|----------|-------|-------------|-----------|
| Baseline (steam only) | 0% | 42.9 | 3,548 |
| DEC at baseline | 90% | 35.8 | 2,896 |
| DEC + 2 GWe, 95%, 3yr, 3%, 50yr | 90% | **10.4** | 1,526 |
| DEC + 5 GWe, 95%, 3yr, 2%, 50yr | 90% | **6.1** | 1,026 |

**Key finding:** With DEC, the fully costed pB11 mirror reaches \$10.4/MWh at moderate scale (2 GWe) and favorable-but-not-extreme financing (3% WACC, 50-yr life, 3-yr build) — essentially at target. At mega-scale (5 GWe, 2% WACC), DEC pushes the fully costed plant to \$6.1/MWh.

### 3.3 Automation

The CAS21 rework already prices pB11 buildings at industrial grade. The remaining lever specific to pB11 is **automated operations** — reducing the 59-person baseline staff to ~30 FTE via a data-center operating model (remote monitoring, AI predictive maintenance, robot-accessible plant with zero activation barriers).

| Scenario | LCOE (\$/MWh) | O/N (\$/kW) |
|----------|-------------|-----------|
| DEC + moderate (59 FTE) | 10.4 | 1,526 |
| DEC + moderate + automated (30 FTE) | **8.8** | 1,526 |

Automation buys \$1.6/MWh — the difference between "at target" and "below target with margin."

---

## 4. Summary of Paths

| Path | Description | LCOE | Core | Key requirement |
|------|-------------|------|------|----------------|
| 1 | Steam, free core, all levers | \$7.3 | Free | 2 GWe, 95%, 3% WACC, 3yr, 50yr |
| 2 | DEC, free core, moderate | \$5.7 | Free | 2 GWe, 95%, 3% WACC, 3yr, 50yr |
| 3 | DEC, free core, mega-scale | \$3.9 | Free | 5 GWe, 95%, 2% WACC, 3yr, 50yr |
| 4 | Steam, full core, all levers | \$12.5 | Full | 2 GWe, 95%, 3% WACC, 3yr, 50yr |
| 5 | DEC, full core, moderate | \$10.4 | Full | 2 GWe, 95%, 3% WACC, 3yr, 50yr |
| 6 | DEC, full core, mega-scale | \$6.1 | Full | 5 GWe, 95%, 2% WACC, 3yr, 50yr |
| **7** | **DEC + automated, full core** | **\$8.8** | **Full** | **2 GWe, 95%, 3% WACC, 3yr, 50yr, 30 FTE** |

### 4.1 Required Conditions

All paths to \$10/MWh require:

1. **Aneutronic fuel (pB11).** DT adds ~\$1.7B/GWe in neutron infrastructure (breeding blanket, shielding, remote handling, radwaste). Even with a free DT core, the floor is ~\$40/MWh vs \$18/MWh for pB11.
2. **Large plant (>=2 GWe net).** Fixed costs (buildings, O&M, indirects) must be spread across many MWh. At 1 GWe, no combination of levers reaches the target.
3. **High availability (>=95%).** 12% more energy per dollar of capital versus 85%.
4. **Low cost of capital (<=3% real WACC).** Requires government, utility, or sovereign backing. At 7% WACC, the target is mathematically impossible.
5. **Fast construction (<=3 yr).** Reduces IDC substantially.

### 4.2 The DEC Dividend

DEC is the discriminating lever between "close but not quite" and "at target." For pB11:

- **Without DEC:** \$10/MWh requires mega-scale (5 GWe) with sovereign financing. The steam-only path at moderate scale reaches \$12.5/MWh — 25% above target.
- **With DEC** at f\_dec = 0.9, eta\_de = 0.85: moderate scale reaches \$10.4/MWh — essentially at target. With automation (30 FTE), \$8.8/MWh with margin.

The reason DEC is so powerful for pB11 is that ~99.8% of fusion energy is in charged alphas. Routing 90% through DEC at 85% efficiency eliminates ~\$400M of turbines and cooling towers, and raises effective conversion efficiency from ~46% (steam) to ~80% (weighted average).

For DT or DD fuels, where a large fraction of energy is in neutrons, DEC provides much smaller benefit — neutrons must still go through a thermal cycle. DEC is specifically a pB11 (and to a lesser extent DHe3) advantage.

---

## 5. What Definitely Will Not Work

- **7% WACC** (commercial project finance): At this rate, \$10/MWh requires <\$500/kW overnight cost — below gas turbines, physically implausible for any power plant.
- **1 GWe scale**: Fixed costs are too high even at 2% WACC with every other lever at its extreme.
- **DT fuel**: Neutron infrastructure adds ~\$1.7B/GWe. The free-core floor for DT (~\$40/MWh) is close to the pB11 *fully costed* result (\$43/MWh with no levers).
- **Any single lever in isolation**: The deepest any one lever reaches is ~\$22/MWh (5 GWe scale). The target requires at least 4-5 levers stacked simultaneously.

---

## 6. Conclusions

The \$10/MWh (1 ¢/kWh) target for fusion electricity is achievable under specific conditions, but the requirements are stringent and mutually reinforcing.

1. **The fusion core is necessary but not sufficient.** Even a free core leaves an \$18/MWh floor from buildings, BOP, O&M, and financing. Core cost reduction alone cannot reach the target.

2. **DEC is the highest-value technology lever for pB11.** By eliminating the steam cycle, DEC provides ~\$2/MWh of headroom at moderate scale — the difference between \$12.5 (steam) and \$10.4 (DEC).

3. **Scale matters more than unit cost reduction.** Going from 1 to 2 GWe buys more LCOE reduction than any other single lever. The P^0.5 staffing economy of scale and the spreading of fixed costs over more MWh are the dominant mechanisms.

4. **Financial conditions are as important as engineering.** The WACC lever (elasticity +0.79) is second only to availability (-0.90) in LCOE sensitivity. Government-backed financing at 3% WACC is worth ~\$15/MWh versus commercial 7% WACC.

5. **The path is pB11 + DEC + scale + favorable finance.** Specifically: >=2 GWe net, >=95% availability, <=3% WACC, <=3-yr construction, >=50-yr lifetime, with 90% DEC at 85% efficiency. This produces \$10.4/MWh with a fully costed reactor (Path 5). Adding automation (30 FTE) reaches \$8.8/MWh (Path 7).

The key physics challenge is not reaching ignition — it is reaching ignition in a configuration that supports DEC, high availability, and multi-GWe deployment at low unit cost. The key economic insight is that pB11's radiological advantages (near-zero activation, no tritium) cascade beyond the reactor into buildings, construction grade, staffing, and automation — compounding advantages that DT fusion cannot access regardless of reactor cost.

---

## Sources

- `examples/free_fusion_core.py` — free-core LCOE floor analysis with DEC scenarios
- `examples/path_to_1cent.py` — fully costed lever stacking and grid search
- `docs/analysis/dt_vs_pb11_cost_comparison.md` — DT vs pB11 cost chain
- `docs/account_justification/CAS21_buildings.md` — building cost derivation (industrial-grade)
- `docs/account_justification/CAS40_capitalized_owners_costs.md` — owner's cost methodology
- `docs/account_justification/CAS70_staffing_and_om_costs.md` — staffing and O&M
- `docs/account_justification/CAS80_annualized_fuel_cost.md` — fuel cost methodology
