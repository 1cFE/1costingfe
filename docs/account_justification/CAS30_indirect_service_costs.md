# CAS30: Capitalized Indirect Service Costs

**Date:** 2026-02-20
**Status:** Research complete, implementation pending

## Account Structure

CAS30 covers costs of construction support services that are not direct material/labor for specific plant systems. Sub-accounts per GEN-IV EMWG (2007):

| CAS | Description | Old Account |
|-----|-------------|-------------|
| 31 | Field Indirect Costs | 93 |
| 32 | Construction Supervision | 91 |
| 33 | Commissioning and Start-up | — |
| 34 | Demonstration Test Run | — |
| 35 | Design Services (Offsite) | 92 |
| 36-39 | PM/CM, contingency on support | — |

Scope of each sub-account (from GEN-IV EMWG / pyfecons CAS300000.tex):

- **C31 (Field Indirect):** Construction equipment rental/purchase, temporary buildings, laydown areas, tools, heavy equipment (cranes, bulldozers, welders), transport vehicles, consumables, safety equipment, utilities, construction insurance, warehousing, site cleanup.
- **C32 (Construction Supervision):** Direct supervision of construction by contractors. Field engineers and superintendents. Non-manual supervisory staff. Does NOT include craft laborers (those are in CAS21-28).
- **C35 (Design Services Offsite):** A/E home office and equipment vendor home office work. Site-related engineering during construction. Project engineering. QA costs related to design. For NOAK only (FOAK certification is separate).

## Source Documents

### Primary sources (read directly)

1. **Schulte, S.C., Willke, T.L., Young, J.R.** "Fusion Reactor Design Studies — Standard Accounts for Cost Estimates," PNL-2648, Battelle Pacific Northwest Laboratory, May 1978. [OSTI](https://www.osti.gov/servlets/purl/6635206)
   - Paragraph 49, page 20: Account 91 = 15%, Account 92 = 15%, Account 93 = 5% of total direct cost.
   - Total indirect (91+92+93) = **35% of TDC**.
   - No LSA levels — single set of fractions.

2. **ARIES-RS Systems Code Output** (version 9.a, August 30, 1996). [qedfusion.org](https://qedfusion.org/DOCS/ARIES-RS/RS6/output.html)
   - Contains the complete LSA factor table with all four levels. Verified LSA=1 factors against ARIES-RS dollar values — exact match.

3. **pyfecons source code and LaTeX templates.** [GitHub](https://github.com/Woodruff-Scientific-Ltd/PyFECONS)
   - `lsa_levels.py`: LSA factor table
   - `cas30_capitalized_indirect_service.py`: Both LSA and bottom-up calculation
   - `CAS300000.tex`: Sub-account descriptions and bottom-up staffing derivation

### Secondary sources (cited in literature, PDFs not text-extractable)

4. **Miller, R.L.** "Economics and Costing," Chapter 3, STARLITE Final Report, UCSD, 1997. [PDF](https://qedfusion.org/LIB/REPORT/STARLITE/FINAL/chap3.pdf) — Origin of the LSA factor table.
5. **Miller, R.L.** "Economic goals and requirements for competitive fusion energy," Fusion Engineering and Design 41 (1998) 393-400.
6. **Waganer, L.M.** "ARIES Cost Account Documentation," UCSD-CER-13-01, June 2013. [PDF](https://qedfusion.org/LIB/REPORT/ARIES-ACT/UCSD-CER-13-01.pdf)
7. **Piet, S.J.** "Inherent/Passive Safety for Fusion," Fusion Technology 10(3P2B), November 1986. [OSTI](https://www.osti.gov/biblio/7005793) — Defined the 4 LSA levels.
8. **Logan, B.G.** "A rationale for fusion economies based on inherent safety," Journal of Fusion Energy 4(4), 245-267, 1985. — Economic rationale for LSA cost credits.
9. **Bourque, R.F. et al.** "Fusion reactor cost reductions by employing non-nuclear grade components," General Atomics, November 1987. [OSTI](https://www.osti.gov/biblio/5704646) — Found 23% savings for all-conventional vs nuclear-grade construction.
10. **Woodruff, S.** "A Costing Framework for Fusion Power Plants," arXiv:2601.21724, January 2026.

## Existing Methods

### Method A: LSA fraction of direct cost (Schulte/Miller/ARIES)

The ARIES tradition computes CAS30 as a fraction of CAS20 (total direct cost):

```
C31 = fac_93[LSA] * CAS20
C32 = fac_91[LSA] * CAS20
C35 = fac_92[LSA] * CAS20
```

LSA factor table (from ARIES-RS output, matches pyfecons `lsa_levels.py`):

| Account | LSA=1 | LSA=2 | LSA=3 | LSA=4 |
|---------|-------|-------|-------|-------|
| 91 (C32) | 0.113 | 0.120 | 0.128 | 0.151 |
| 92 (C35) | 0.052 | 0.052 | 0.052 | 0.052 |
| 93 (C31) | 0.052 | 0.060 | 0.064 | 0.087 |
| **Sum** | **0.217** | **0.232** | **0.244** | **0.290** |

**Provenance of these specific numbers:**

The LSA concept was defined by Piet (1986). The factor table first appeared in Miller's STARLITE Ch.3 (1997). The derivation methodology is in a scanned PDF that cannot be text-extracted; the specific values appear to be Miller's expert judgment.

**Key discrepancy with Schulte:** Miller's LSA=4 column (which should represent fission-like/nuclear-grade construction, closest to Schulte's baseline) does NOT match Schulte:

| Account | Schulte (1978) | Miller LSA=4 | Change |
|---------|---------------|-------------|--------|
| 91 | 15% | 15.1% | ~same |
| 92 | 15% | 5.2% | **3x reduction, LSA-independent** |
| 93 | 5% | 8.7% | 1.7x increase |
| Total | 35% | 29.0% | 17% reduction |

Account 92 was cut from 15% to 5.2% at all LSA levels with no documented justification. This is not an LSA effect — it's a rebasing. Combined 92+93 went from 20% to 13.9%, a 30% cut on engineering costs.

**Note:** The pyfecons LaTeX template (CAS300000.tex) cites "Schulte et al. (1978)" for its percentages but actually quotes Miller's LSA=2 values (12%, 5.2%, 6%), not the real Schulte values (15%, 15%, 5%). This is a misattribution.

### Method B: Bottom-up parametric (Woodruff/pyfecons)

pyfecons also computes CAS30 via a power-scaling formula:

```
C3x = (P_net / 150)^(-0.5) * P_net * coeff * T_construction
```

Coefficients derived from a staffing model for a 150 MWe reference plant:

| Account | Staff | Hours/yr | Rate | Annual cost | Coeff (M$/MW/yr) |
|---------|-------|----------|------|-------------|-----------------|
| C31 | 10 engineers | 2000 | $150/hr | $3.0M/yr | 0.02 |
| C32 | 25 managers | 2000 | $150/hr | $7.5M/yr | 0.05 |
| C35 | 15 engineers | 1000 | $150/hr | $2.25M/yr | 0.03 |

At 1 GWe, 6-year construction: **$232M** (5.8% of ~$4B direct cost).

**Problems with this derivation:**
1. C31 scope includes equipment, temp buildings, cranes, consumables, utilities — the staffing model only counts 10 people's labor
2. 129 implied indirect staff at 1 GWe is very lean for nuclear-grade construction
3. $150/hr has no overhead/burden markup; fully loaded rates are $200-300/hr
4. The economy-of-scale exponent (-0.5) is assumed, not derived
5. No external citation; this is Woodruff's own estimate

**pyfecons uses Method B for the total** (`C300000 = C310000 + C320000 + C350000`). The LSA values are computed but stored separately (`C3x0000LSA`) for reporting only.

### Comparison

For a 1 GWe plant with ~$4B direct cost, 6-year construction:

| Method | CAS30 | % of TDC | Source |
|--------|-------|----------|--------|
| Schulte (1978) | $1,400M | 35% | PNL-2648 |
| Miller LSA=1 | $868M | 21.7% | STARLITE/ARIES |
| Miller LSA=4 | $1,160M | 29.0% | STARLITE/ARIES |
| Woodruff bottom-up | $232M | 5.8% | pyfecons |
| 1costingfe (current, broken) | $7.2M | 0.2% | Bugs |

Real-world fission nuclear for context (not directly comparable):
- Korean APR1400 (standardized NOAK): indirect ~15-20% of direct
- Vogtle 3&4 (US FOAK, troubled): indirect ~35%+ of direct
- Chinese Hualong One (standardized): ~12-18% of direct

## Assessment

**No real fusion construction data exists to calibrate CAS30.** Private fusion companies (CFS, Helion, Type One, Realta, etc.) do not publish cost breakdowns at this granularity. ITER is a one-of-a-kind R&D facility. UK STEP and EU-DEMO cost studies use ARIES-lineage models — circular reference.

**The sub-account split (C31/C32/C35) is not empirically distinguishable** for fusion. Even fission nuclear data doesn't cleanly map to these categories.

**What IS defensible:**
1. Indirect costs are real and significant (somewhere in the 15-35% range of direct cost)
2. They scale with construction time (more years = more supervision, equipment rental, etc.)
3. They are larger for FOAK and for nuclear-grade (LSA=4) construction
4. For NOAK fusion at LSA=1-2, 10-25% of direct cost brackets the plausible range

**What is NOT defensible:**
- Pretending to know whether field engineering is 5.2% or 6% of TDC
- The pyfecons bottom-up staffing model (labor-only, ignores equipment/consumables)
- Using Schulte's 1978 fission-era 35% baseline unchanged for NOAK fusion
- Any sub-account precision within CAS30

## Recommendation for 1costingfe

Replace the current broken parametric model with:

```python
CAS30 = indirect_fraction * CAS20 * (construction_time / reference_time)
```

Where:
- `indirect_fraction`: single configurable parameter, default **0.20** (center of plausible range)
- `reference_time`: baseline construction duration, default **6 years**
- Construction time scaling captures the main driver that varies between scenarios
- No sub-accounts; no power-law scaling on a 150 MW reference plant

Default of 20% is chosen as:
- Below Schulte's 35% (1978 fission baseline, too high for NOAK fusion)
- Below Miller LSA=2 at 23.2% (ARIES tradition, likely somewhat high for NOAK)
- Above Woodruff's 5.8% (bottom-up, clearly too low — labor only)
- Consistent with efficient NOAK nuclear construction (Korean/Chinese 15-20%)
- Conservative but not unreasonable for a standardized fusion plant