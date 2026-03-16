# CAS21: Buildings and Structures

**Date:** 2026-03-16
**Status:** Justified — values retained, fuel scaling validated

## Overview

CAS21 covers all buildings and structures on the fusion plant site.
The cost model sums per-kW costs ($/kW of gross electric) across 19
building types, with fuel-dependent scaling on four tritium-related
buildings.

**Adopted total:** ~760 $/kW (DT), ~511 $/kW (non-DT) of gross electric.

At a reference 1 GWe net DT tokamak (~1.165 GW gross), CAS21 = **$886M**
(NOAK, no contingency: $760M; FOAK +10% = $836M).  This represents
~14% of overnight capital cost — consistent with the NREL ATB 2024
nuclear cost structure (16.1% for "Structures") and the pyFECONS
framework (~$714/kW).

## Building List and Per-kW Costs

| # | Building | $/kW | % of Total | Fuel-dep? | Scope |
|---|----------|-----:|----------:|-----------:|-------|
| 1 | Site improvements | 268.0 | 35.3% | Yes | Land clearing, grading, roads, parking, fencing, utilities, drainage, tritium monitoring perimeter, emergency infrastructure |
| 2 | Fusion heat island | 126.0 | 16.6% | Yes | Reactor containment building, crane hall, primary confinement, shielding walls, tritium barrier systems |
| 3 | Turbine building | 54.0 | 7.1% | No | Turbine hall, condenser bay, feedwater equipment area |
| 4 | Heat exchanger building | 12.0 | 1.6% | No | Primary/secondary heat exchanger housing |
| 5 | Power supply storage | 17.0 | 2.2% | No | Magnet power supply building, energy storage |
| 6 | Reactor auxiliaries | 35.0 | 4.6% | No | Auxiliary systems building (vacuum, gas, diagnostics) |
| 7 | Hot cell | 93.4 | 12.3% | Yes | Activated component handling, decontamination, size reduction, packaging |
| 8 | Reactor services | 25.0 | 3.3% | No | Maintenance shops, laydown areas |
| 9 | Service water | 11.0 | 1.4% | No | Water treatment, storage, distribution |
| 10 | Fuel storage | 9.1 | 1.2% | Yes | Tritium storage, deuterium storage, gas handling |
| 11 | Control room | 17.0 | 2.2% | No | Main control room, emergency control, I&C rooms |
| 12 | Onsite AC power | 21.0 | 2.8% | No | Electrical switchgear building, diesel generator building |
| 13 | Administration | 10.0 | 1.3% | No | Office building, training center |
| 14 | Site services | 4.0 | 0.5% | No | Warehouses, fire station, medical |
| 15 | Cryogenics | 15.0 | 2.0% | No | Cryoplant building (LHe/LN2) |
| 16 | Security | 8.0 | 1.1% | No | Security building, guard posts, barriers |
| 17 | Ventilation stack | 9.2 | 1.2% | No | Stack, HVAC discharge, monitoring equipment |
| 18 | Assembly hall | 20.0 | 2.6% | No | Pre-assembly of reactor modules |
| 19 | Direct energy building | 5.0 | 0.7% | No | Direct energy conversion equipment housing |
| | **Total** | **759.7** | **100%** | | |

## Fuel-Dependent Scaling

Four buildings are classified as "tritium-related" and receive a **0.5×
scaling factor** for non-DT fuels:

    fuel_scale = 1.0 if fuel == DT else 0.5

| Building | DT cost | non-DT cost | Rationale for scaling |
|----------|--------:|------------:|----------------------|
| Site improvements | 268 | 134 | DT requires tritium monitoring perimeter, protected-area fencing, emergency infrastructure per 10 CFR Part 30. Non-DT eliminates nuclear-specific site infrastructure. |
| Fusion heat island | 126 | 63 | DT reactor building requires tritium primary confinement barriers, double-walled penetrations, and glove-box interfaces. Non-DT uses conventional industrial construction. |
| Hot cell | 93.4 | 46.7 | DT produces activated components requiring remote handling hot cell. pB11 needs no hot cell; DD/DHe3 need smaller ones. |
| Fuel storage | 9.1 | 4.6 | Tritium storage, processing, and accountability systems. Non-DT fuels use conventional gas storage. |
| **Subtotal** | **496.5** | **248.3** | |

This yields:
- **DT total:** 760 $/kW (496.5 tritium + 263.2 non-tritium)
- **Non-DT total:** 511 $/kW (248.3 scaled tritium + 263.2 non-tritium)
- **Reduction:** 33% for non-DT fuels

### Is binary DT/non-DT scaling appropriate?

The 0.5× factor is a simplification.  A more granular model would
differentiate DD (produces tritium via side reaction, needs some tritium
handling) from pB11 (no tritium, no significant activation).  However:

1. DD tritium production is small (side-reaction rates are ~1/50 of DT)
   and would require a much smaller tritium handling system — closer to
   0.5× than 1.0×.
2. DHe3 has ~5% neutron fraction — similar argument, small hot cell and
   minimal tritium handling.
3. The building cost uncertainty (~±30%) dominates the fuel-scaling
   uncertainty (~±0.1-0.2 on the multiplier).

The binary model is retained for simplicity.  If fuel-specific building
costs become a significant LCOE driver, a 4-level model (DT=1.0,
DD=0.8, DHe3=0.6, pB11=0.5) could be implemented.

## Source Analysis

### Primary source: ARIES / pyFECONS

The building list and per-kW costs derive from the ARIES cost-account
tradition (Waganer, UCSD-CER-13-01, 2013) as implemented in pyFECONS.
pyFECONS CAS21 references NETL Case B12A (supercritical coal plant
buildings) and ARIES-ACT facility layouts.

The pyFECONS building costs total ~$714/kW across 17 buildings.  The
1costingfe values total ~$760/kW across 19 buildings (adding assembly
hall and direct energy building).  Individual allocations differ between
the two implementations, but the totals are within 6%, reflecting
different calibration vintages or sub-account boundary choices.

### Cross-check: NREL ATB 2024 (fission nuclear)

The NREL Annual Technology Baseline (2024) reports structures at 16.1%
of total capital for a 1 GWe fission reactor at ~$7,000/kW total,
implying ~$1,130/kW for structures.  The fission number is ~50% higher
than our fusion value ($760/kW) due to:

- Nuclear-grade containment building (not required for fusion under
  Part 30)
- Seismic Category I structures (DT fusion buildings are industrial
  construction, not Seismic Cat I)
- Redundant safety-related structures (emergency diesel building,
  refueling water storage, etc.)

Our lower value is consistent with the ARIES assumption that NOAK fusion
plants use commercial/industrial construction standards rather than
nuclear safety-grade construction.

### Cross-check: ITER

ITER's 39 buildings and technical areas are part of a ~EUR 20B total
construction budget.  Buildings represent ~15-20% of ITER's total
cost (~EUR 3-4B).  However, ITER is a FOAK research facility with
bespoke one-off construction, international design review overhead,
and extreme seismic requirements (Cadarache, France).  ITER building
costs per kW are not meaningful because ITER does not generate electricity.

### Cross-check: Industrial construction benchmarks

For a ~250-acre site with 19 buildings totaling ~200,000 m² of floor
space:

| Building type | Industry cost range | Our implied cost |
|--------------|-------------------:|----------------:|
| Large industrial site development | $50-150M | $312M (site_improvements) |
| Heavy industrial building (crane-served) | $200-400/ft² | $250-350/ft² implied (fusion heat island) |
| Hot cell facility | $50-150M | $109M |
| Conventional turbine hall | $40-80M | $63M |
| Control room / admin | $10-30M | $17M + $10M = $27M |

Site improvements ($312M) appears high relative to conventional
industrial sites ($50-150M), but includes fusion-specific infrastructure:
tritium monitoring perimeter, protected-area infrastructure, emergency
systems, and utility distribution for a nuclear-regulated facility.

## Decision

**Retain current values.**  The total building cost (~$760/kW DT,
~$511/kW non-DT) is consistent with:
- pyFECONS framework (~$714/kW)
- NREL ATB fission structures (16% of total, discounted for non-nuclear
  construction)
- ITER building fraction (15-20% of total)
- Industry benchmarks for heavy industrial facilities

Individual building allocations carry ±30% uncertainty, but the
aggregate is more robust.  The fuel-dependent scaling (0.5× for non-DT)
appropriately captures the tritium handling infrastructure difference.

## References

- Waganer, L. M., "ARIES Cost Account Documentation," UCSD-CER-13-01,
  University of California San Diego, June 2013.
- DOE/NETL, "Cost and Performance Baseline for Fossil Energy Plants,"
  DOE/NETL-2022/3575, October 2022.  (Case B12A reference)
- NREL, "Annual Technology Baseline — Nuclear," 2024 edition.
  https://atb.nrel.gov/electricity/2024/nuclear
- ITER Organization, "Building ITER," https://www.iter.org/building-iter
- Woodruff, S., "A Costing Framework for Fusion Power Plants,"
  arXiv:2601.21724, January 2026.
