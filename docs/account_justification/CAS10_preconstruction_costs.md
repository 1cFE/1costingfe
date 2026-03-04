# CAS10: Pre-Construction Costs

**Date:** 2026-02-20
**Status:** Implemented (land intensity + licensing times updated)

## Account Structure

| CAS | Description | Key parameter |
|-----|-------------|---------------|
| C110000 | Land and Land Rights | `land_intensity`, `land_cost` |
| C120000 | Site Permits | `site_permits` |
| C130000 | Plant Licensing | `licensing_cost_{fuel}` |
| C140000 | Plant Permits | `plant_permits` |
| C150000 | Plant Studies | `plant_studies_foak` / `_noak` |
| C160000 | Plant Reports | `plant_reports` |
| C170000 | Other Pre-Construction | `other_precon` |
| C190000 | Contingency | `contingency_rate` * subtotal |

CAS10 is typically <1% of total capital. Individual sub-accounts are
not worth detailed parametric modeling.

## C110000 — Land and Land Rights

### Source: DI-018 / DI-019

Research document: `fusion-tea/knowledge/research/approved/20260206-land-and-land-rights-fusion-cost.md`

Key findings:
- Modern compact fusion plants need **20-200 acres** (not the legacy
  1,000-acre ARIES assumption)
- CFS ARC: **100 acres for 400 MWe** → **0.25 acres/MWe**
- No exclusion zone (Part 30 for DT, no NRC for pB11) enables compact siting
- US farmland averages $4,350/acre (USDA 2025); industrial-zoned 2-5x higher
- Land cost is <0.5% of total capital — negligible LCOE driver

### Updated values

| Parameter | New | Source |
|-----------|-----|--------|
| `land_intensity` | **0.25** acres/MWe | CFS ARC (100 acres / 400 MWe) |
| `land_cost` | **10,000** $/acre (unchanged) | Industrial-zoned US average |

At 1 GWe: 0.25 * 1000 * $10,000 / 1e6 = **$2.5M**.

## C130000 — Plant Licensing (times)

| Fuel | pyfecons | Research range |
|------|----------|---------------|
| DT | 2.5 yr | 1-2 yr |
| DD | 1.5 yr | 6-18 mo |
| DHe3 | 0.75 yr | 6-12 mo |
| PB11 | 0.0 yr | ~0 yr |

### Source: DI-015 / DI-016 / DI-017

Research document: `fusion-tea/knowledge/research/approved/20260203-100000_fusion-regulatory-framework-dt-pb11.md`

Key findings:
- **DI-015**: NRC decided (2023) to regulate DT fusion under Part 30
  (byproduct materials), not Part 50/52 (reactor licensing). Reduces
  timeline from 5-10yr (fission) to 1-2yr.
- **DI-016**: pB11 fusion likely falls outside NRC jurisdiction entirely
  (no radioactive materials). Licensing timeline ~0yr for NRC; only
  standard industrial permits required.
- **DI-017**: Regulatory burden affects LCOE through licensing timeline
  (IDC), safety system requirements (CAS20), compliance staff (CAS70),
  and risk premium (CRF).

### Updated values

| Parameter | New | Rationale |
|-----------|-----|-----------|
| `licensing_time_dt` | **2.0** yr | Part 30, upper end of 1-2yr range |
| `licensing_time_dd` | **1.5** yr | Reduced tritium, upper end of 6-18mo |
| `licensing_time_dhe3` | **0.75** yr | Minimal radioactivity, midpoint of 6-12mo |
| `licensing_time_pb11` | **0.0** yr | No NRC jurisdiction |

### Impact

Licensing times affect FOAK builds only (NOAK uses `construction_time`
alone). They flow into:
- `_total_project_time` → `levelized_annual_cost` for CAS71 (O&M) and
  CAS80 (fuel), shifting costs to operation-start dollars
- CAS10 licensing cost (fuel-dependent, unchanged)

Licensing times no longer affect CAS60 (IDC uses construction time only)
or CAS90 (plain CRF, no construction-time adjustment).

## C130000 — Licensing costs

Current values ($5M DT, $3M DD, $1M DHe3, $0.1M PB11) are within the
$0.5-10M range from research (DI-015/016). No change needed.

## References

- DI-018: "Fusion Plant Land Cost is Negligible LCOE Driver"
  (`20260206-land-and-land-rights-fusion-cost.md`)
- DI-019: "No Exclusion Zone Enables Compact Fusion Siting"
  (same document)
- DI-015: "D-T Fusion Regulated Under Part 30"
  (`20260203-100000_fusion-regulatory-framework-dt-pb11.md`)
- DI-016: "p-B11 Fusion Likely Outside NRC Jurisdiction"
  (same document)
- DI-017: "Regulatory Burden as LCOE Multiplier"
  (same document)
- USDA, "Land Values 2025 Summary," August 2025
- CFS ARC project filings (Chesterfield, VA)
