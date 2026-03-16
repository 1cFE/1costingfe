# CAS29: Contingency on Direct Costs

**Date:** 2026-03-16
**Status:** Justified — values retained

## Overview

CAS29 is a percentage-based contingency allowance applied to the sum of all
direct cost accounts (CAS21-28).  It reflects uncertainty in cost estimates,
design maturity, and construction risk.

| Mode | Contingency rate | Rationale |
|------|-----------------|-----------|
| FOAK | 10% | First-of-a-kind design and construction uncertainty |
| NOAK | 0% | Mature design, established supply chain, no contingency |

## Methodology

    CAS29 = contingency_rate × (CAS21 + CAS22 + ... + CAS28)

The contingency is applied *only* to direct costs (CAS20), not to indirect
costs (CAS30), owner's costs (CAS40), or supplementary costs (CAS50).
This follows the Gen-IV EMWG cost estimating guidelines (Rev. 4.2, 2007),
which distinguish between project contingency on direct costs and program-
level contingency on the total project.

## FOAK: 10%

The 10% FOAK contingency is deliberately conservative (low) relative to
typical nuclear construction contingencies:

- **Gen-IV EMWG recommendation:** 15-20% for FOAK advanced reactor designs.
- **AACE International guidelines:** Class 3 estimates (feasibility study
  level) carry 10-15% contingency; Class 2 (detailed engineering) carry
  5-15%.
- **pyFECONS:** Uses 10% FOAK / 0% NOAK (identical to our values).

We adopt 10% rather than 15-20% because:
1. Fusion direct costs already include significant conservatism in individual
   account estimates (e.g., nuclear-grade fabrication markups in CAS22).
2. The FOAK contingency captures *cost estimation uncertainty*, not *schedule
   or regulatory risk* — those are addressed through CAS10 licensing time
   and CAS60 IDC.
3. A 10% contingency on direct costs, combined with 20% indirect costs
   (CAS30) and 10% CAS29 on those directs, effectively provides ~13%
   overall markup on the base direct cost.

## NOAK: 0%

NOAK (Nth-of-a-kind) plants benefit from:
- Standardised designs with proven constructability.
- Established supply chains and fabrication facilities.
- Experienced construction workforce and management.
- Resolved licensing and regulatory precedents.

The 0% NOAK contingency reflects the assumption that cost estimates for a
mature, standardised design are sufficiently accurate that no contingency
reserve is needed.  This is consistent with:
- The pyFECONS NOAK convention (0% contingency, suppressed).
- The Gen-IV EMWG NOAK methodology, which zeroes project contingency for
  equilibrium commercial designs.
- Standard practice in LCOE studies for mature technologies (e.g., EIA
  AEO assumptions for existing generation technologies).

## Note on CAS10 contingency

CAS10 (pre-construction costs) carries its own contingency at the same
rate (10% FOAK / 0% NOAK), applied within the `cas10_preconstruction()`
function.  CAS50 (supplementary costs) also carries a contingency on its
sub-accounts.  These are separate from CAS29.

## References

- GEN-IV EMWG, "Cost Estimating Guidelines for Generation IV Nuclear
  Energy Systems," Rev. 4.2, GIF/EMWG/2007/004, 2007.
- AACE International, "Cost Estimate Classification System,"
  Recommended Practice No. 18R-97 (Rev. 2020).
- pyFECONS source: `cas29_contingency.py` — 10% FOAK, 0% NOAK.
- Woodruff, S., "A Costing Framework for Fusion Power Plants,"
  arXiv:2601.21724, January 2026.
