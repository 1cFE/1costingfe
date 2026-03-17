# CAS50: Capitalized Supplementary Costs

**Date:** 2026-03-16
**Status:** Research complete, implementation pending

## Account Structure

CAS50 covers capitalized supplementary costs — items beyond direct construction, indirect services, and owner's costs that must be incurred before commercial operation. Sub-accounts per GEN-IV EMWG (2007) Table 1.2 and Woodruff (2026):

| CAS | Description |
|-----|-------------|
| 51 | Shipping and Transportation |
| 52 | Spare Parts |
| 53 | Taxes |
| 54 | Insurance (Construction Period) |
| 55 | Initial Fuel / Startup Inventory |
| 56 | Decommissioning Provisions |
| 58 | Other Supplementary Costs |
| 59 | Contingency on Supplementary Costs |

CAS50 completes the overnight cost: `C_overnight = C10 + C20 + C30 + C40 + C50`. Total capital investment cost: `TCIC = C_overnight + C60`.

**Scope boundaries:**
- CAS50 is distinct from CAS30 (indirect construction services) — CAS30 covers contractor-side construction support; CAS50 covers owner-side capital items not directly associated with construction labor or equipment installation.
- CAS50 insurance (C54) covers the construction period only. Operating insurance is in CAS71 (annual O&M).
- CAS50 decommissioning (C56) is the capitalized provision (sinking fund initial deposit or financial assurance); annual decommissioning fund contributions are in CAS71.
- CAS50 spare parts (C52) are the initial on-site inventory at COD; ongoing spare parts procurement is operating expense (CAS71).

## Source Documents

### Primary sources (read directly)

1. **Woodruff, S.** "A Costing Framework for Fusion Power Plants," arXiv:2601.21724v2, January 2026.
   - Table 2: CAS50 = $213M (10% of TCC) for a 637 MWe plasma-jet MIF example.
   - MARS comparison: CAS50 = $700M (7.7% of TCC).
   - Sub-accounts listed: transportation/insurance (C50), spare parts (C51), contingencies (C52), insurance (C53), decommissioning (C54).
   - Notes that "decommissioning costs are represented as a capitalized indirect cost, enabling clearer attribution, sensitivity studies, and discussion of the drivers of end-of-life obligations."

2. **GEN-IV EMWG (2007).** "Cost Estimating Guidelines for Generation IV Nuclear Energy Systems," GIF/EMWG/2007/004, Rev 4.2. [PDF](https://www.gen-4.org/gif/upload/docs/application/pdf/2013-09/emwg_guidelines.pdf)
   - Account 50 defined on p.31 as "Capitalized Supplementary Costs."
   - Overnight cost = Accounts 10 + 20 + 30 + 40 + 50 (p.29).
   - Sub-account descriptions on p.152.

3. **Shropshire, D. et al.** "Advanced Nuclear Reactor Cost Estimation: Sodium-Cooled Fast Reactor Case Study," INL, Sort_67398, 2024. [PDF](https://inldigitallibrary.inl.gov/sites/sti/sti/Sort_67398.pdf)
   - CAS50 estimated for a 1,243 MWe SFR. Total TCIC = $4,537/kWe (Accounts 10-60).
   - Provides bottom-up estimates for all CAS50 sub-accounts.

4. **World Nuclear Association.** "Economics of Nuclear Power," 2025. [Web](https://world-nuclear.org/information-library/economic-aspects/economics-of-nuclear-power)
   - Decommissioning: 9-15% of initial capital cost of a fission plant.
   - When discounted over plant life, contributes only 0.1-0.2 ¢/kWh.
   - Transportation: ~2% of overnight cost.
   - First fuel load (commissioning + fuel): ~5% of overnight cost; fuel portion ~3%.

5. **Waganer, L.M.** "ARIES Cost Account Documentation," UCSD-CER-13-01, June 2013. [PDF](https://qedfusion.org/LIB/REPORT/ARIES-ACT/UCSD-CER-13-01.pdf)
   - ARIES studies used a spare parts allowance of ~2% of equipment/subcontract costs.

6. **NRC.** "Backgrounder on Decommissioning Nuclear Power Plants." [Web](https://www.nrc.gov/reading-rm/doc-collections/fact-sheets/decommissioning)
   - Fission decommissioning generally costs $300-400M per reactor.
   - 10 CFR 50.75(c) provides minimum funding formulas for Part 50 reactors (1986$): PWR = min($75M + $0.0088M×MWt, $105M); BWR = min($104M + $0.009M×MWt, $135M). Escalated to ~2023$, these are roughly $500-700M.
   - Part 30 licensees (fusion) have separate, much lower requirements under 10 CFR 30.35.

7. **OECD-NEA (2016).** "Costs of Decommissioning Nuclear Power Plants." [PDF](https://www.oecd-nea.org/upload/docs/application/pdf/2019-12/7201-costs-decom-npp.pdf)
   - Cross-country survey: fission decommissioning costs range $750-1,250/kWe (2022$), median ~$1,000/kWe.
   - Dismantling and waste disposal each ~30% of total decommissioning cost.

8. **IAEA (2020).** "Decommissioning and Waste Management Considerations for Fusion," IAEA-TECDOC-2116. [PDF](https://www-pub.iaea.org/MTCD/publications/PDF/TE-2116web.pdf)
   - Fusion activated materials can be recycled or cleared within 50-100 years (vs thousands of years for fission actinides).
   - Low-activation materials (RAFM steels, vanadium alloys) dramatically reduce decommissioning waste volumes.
   - No spent fuel management — the dominant fission decommissioning cost driver — applies to fusion.
   - Tritium decontamination is a significant but bounded cost for DT plants.

9. **DOE/INL (2024).** "Nuclear Energy Cost Estimates for Net Zero World Initiative." [PDF](https://www.energy.gov/sites/default/files/2024-10/NZW09%20Nuclear%20Energy%20Cost%20Estimates%20for%20Net%20Zero%20World%20Initiative.pdf)
   - Median fission decommissioning cost: $1,000/kWe (2022$).

10. **pB11 regulatory risk analysis.** `docs/analysis/pb11_side_reactions_nrc_regulatory_risk.md`.
    - All fusion fuels under Part 30 (not Part 50) per Feb 2026 proposed rule.
    - pB11 has ~10,000× fewer neutrons than DT — minimal activation, near-industrial end-of-life.

### Secondary sources (cited in literature)

11. **Schulte, S.C. et al.** "Fusion Reactor Design Studies — Standard Accounts for Cost Estimates," PNL-2648, 1978. — Original COA for fusion, includes supplementary cost category.
12. **IAEA (1991).** "Staffing of Nuclear Power Plants and the Recruitment, Training and Authorization of Operating Personnel." — Basis for insurance and operating staffing estimates referenced in decommissioning.
13. **El-Guebaly, L.A. et al.** "Goals, Challenges, and Successes of Managing Fusion Activated Materials," Fusion Engineering and Design 83, 2008. — Low-activation material strategies for fusion waste minimization.

## Existing Methods

### Method A: Flat values (current 1costingfe / pyfecons-derived)

The current implementation uses hardcoded flat dollar values with no scaling or fuel dependence:

| Component | Current Default | Basis |
|-----------|----------------|-------|
| Shipping | $1.0M | None documented |
| Spare parts | 1% of CAS23-28 | Unattributed |
| Taxes | $0.5M | None documented |
| Insurance | $0.5M | None documented |
| Fuel load | (p_net/1000) × $10M | "rough scaling" (code comment) |
| Decommissioning | $5.0M | None documented |

**Critical flaws:**
1. No source or derivation for any value. The code contains `# REQUIRES CHECKING`.
2. Shipping at $1M is absurdly low — a single heavy reactor component can cost $1-5M to ship. Heavy-haul transport of a steam generator alone costs $2-3M.
3. Insurance at $0.5M is ~100× too low — builder's risk insurance for a multi-billion-dollar project runs 1-3% of construction cost.
4. Decommissioning at $5M is ~100× too low for a DT plant and ~10× too low even for a pB11 plant. Fission decommissioning is $300-700M.
5. No fuel-type dependence — DT (tritium startup inventory, activated waste) should cost far more than pB11 (industrial commodities, minimal activation).

### Method B: Fraction of total capital (Woodruff/ARIES)

The Woodruff paper reports CAS50 = 10% of TCC for a plasma-jet MIF example and 7.7% for MARS. ARIES studies typically used supplementary costs in the 5-12% range of total capital.

**Problems:**
1. These are total CAS50 values — no sub-account methodology is documented.
2. They embed fission-calibrated assumptions about fuel loading (uranium first core) and decommissioning (Part 50 requirements) that don't apply to fusion under Part 30.
3. No fuel-type differentiation.

### Method C: Bottom-up by sub-account (INL Sort_67398 for fission)

The INL SFR study provides the most detailed bottom-up CAS50 for a nuclear plant. While the specific values are fission-specific, the *methodology* — estimating each sub-account from its cost drivers — is the right approach for fusion.

## Derivation: Sub-Account Analysis for Fusion

### C51: Shipping and Transportation

**What it covers:** Transport of major plant components from manufacturer to site. Includes heavy-haul trucking, rail, barge/ship for large components (reactor vessel, steam generators/heat exchangers, turbine-generator, large magnets, vacuum vessel segments).

**Fission reference:** World Nuclear Association reports transportation at ~2% of overnight capital cost. For a $6B fission plant, that's ~$120M.

**Fusion considerations:**
- Fusion plants have large, specialized components (superconducting magnets, vacuum vessel sectors, blanket modules) that require heavy-haul transport.
- Unlike fission reactor pressure vessels (~500 tonnes), most fusion components can be shipped in segments and assembled on-site (ITER model).
- Magnet systems may be the heaviest individual components (~100-300 tonnes per coil for tokamaks).
- No spent fuel shipping (fission's ongoing transport cost is not capitalized here, but the initial logistics infrastructure is comparable).

**Scaling basis:** Shipping scales roughly with the mass and number of major components, which in turn scales with plant size. As a fraction of direct cost, 1-2% is appropriate for fusion — lower than fission's ~2% because fusion avoids the extreme weight of reactor pressure vessels and containment structures.

**Estimate:** 1.5% of CAS20 (direct costs) for all fuel types. Shipping is dominated by heavy equipment logistics, which is fuel-independent (the magnets, vacuum vessel, and turbine are the same regardless of fuel).

### C52: Spare Parts (Initial Inventory)

**What it covers:** Initial on-site inventory of critical spare parts and replacement components at COD. This is distinct from CAS72 (annualized scheduled replacement) — C52 is the one-time capital cost of stocking the warehouse.

**Fission reference:** ARIES studies used ~2% of equipment cost. General industrial practice for critical rotating equipment is 1-3% of installed equipment cost.

**Fusion considerations:**
- DT plants need spare blanket modules, first-wall tiles, and divertor cassettes in inventory because these are replaced on multi-year cycles due to neutron damage. The initial stock must cover the first replacement cycle.
- DD plants: similar but with longer replacement intervals (lower neutron flux).
- DHe3 plants: some activated components but much less frequent replacement.
- pB11 plants: near-conventional spare parts — pumps, valves, heat exchangers, electrical components. No neutron-damaged components to stock.

**Scaling basis:** Percentage of CAS22-28 (reactor plant equipment + BOP). Fuel-dependent because DT/DD plants need expensive radiation-hardened spare components.

| Fuel | Spare Parts % | Rationale |
|------|--------------|-----------|
| DT | 3% | Activated component spares (blanket, FW, divertor), remote handling consumables |
| DD | 2.5% | Similar but less frequent replacement cycles |
| DHe3 | 1.5% | Minor activation, mostly conventional spares |
| pB11 | 1% | Conventional industrial spare parts only |

### C53: Taxes

**What it covers:** Sales tax, use tax, and property tax during construction on purchased equipment and materials.

**Considerations:**
- State sales tax on construction materials and equipment typically ranges 0-8% across US states.
- Many states provide sales tax exemptions for manufacturing equipment, power generation equipment, or specific economic development incentives.
- Nuclear projects commonly negotiate tax abatement or PILOT (Payment In Lieu Of Taxes) agreements with local jurisdictions (NY RPTL §485 provides explicit nuclear exemptions).
- Property tax during the multi-year construction period can be significant but is highly site-specific.
- For a pre-conceptual estimate, 1-2% of materials cost is reasonable. Materials are roughly 50-60% of CAS20.

**Estimate:** 1% of CAS20 for all fuel types. Taxes are jurisdiction-dependent, not fuel-dependent. This assumes partial exemptions typical of large energy projects.

### C54: Insurance (Construction Period)

**What it covers:** Builder's risk insurance covering the project during construction against fire, weather damage, theft, and construction accidents. Also includes general liability and professional liability.

**General construction reference:** Builder's risk premiums are typically 1-3% of total construction cost, with 1-2% for standard projects and higher for complex or hazardous projects.

**Fusion considerations:**
- Fusion plants under Part 30 do not require Price-Anderson nuclear liability insurance ($450M primary + retrospective premium pool) that Part 50 fission reactors need.
- Construction insurance covers the physical plant during building — this is driven by project value and risk profile, not fuel type.
- A multi-billion-dollar construction project with specialized equipment and extended timeline (5-7 years) would be at the higher end of the premium range.
- Most insurers exclude nuclear-related damages, but Part 30 fusion may not trigger these exclusions (no fissile material, no criticality risk).

**Estimate:** 1.5% of (CAS20 + CAS30) for all fuel types. The insurance base is the insured construction value (direct + indirect costs). Fuel-independent because construction risk is driven by project complexity and value, not plasma fuel.

### C55: Initial Fuel / Startup Inventory

**What it covers:** The fuel and working materials needed to achieve first plasma and ramp to full power. This is the fusion analogue of the fission "first core" (which costs ~3% of overnight capital, ~$120M for a 1 GWe PWR).

**This is the most fuel-dependent CAS50 sub-account.**

| Fuel | Startup Inventory | Cost Estimate | Notes |
|------|-------------------|---------------|-------|
| **DT** | 1-2 kg tritium | $30-60M | Tritium at ~$30,000/g (CANDU production). Startup inventory needed before breeding blanket reaches equilibrium. Price highly uncertain — could increase with CANDU retirements. Also ~10 kg deuterium ($5-10k, negligible). |
| **DD** | 10-50 kg deuterium | <$0.1M | Deuterium at ~$500-1,000/kg. Abundantly available from heavy water production. Negligible cost. |
| **DHe3** | 1-10 kg He3 + deuterium | $5-20M | He3 at ~$2,000-15,000/g depending on source (tritium decay vs. natural gas extraction). Highly uncertain supply and price. Deuterium component negligible. |
| **pB11** | Hydrogen + boron-11 | <$0.1M | Industrial commodities. Hydrogen ~$1-3/kg, boron-11 (enriched) ~$50-200/kg but needed in small quantities. Negligible. |

**Why this differs from fission:** Fission's first core cost ($120M+) is driven by the enormous quantity of enriched uranium (~100 tonnes UO₂) and the complex fuel fabrication process. Fusion fuel inventories are measured in kilograms, not tonnes. Even DT's tritium startup cost ($30-60M) is well below fission's first core.

### C56: Decommissioning Provisions

**What it covers:** The capitalized financial provision (initial fund deposit or financial assurance instrument) to cover eventual plant decommissioning. The actual decommissioning occurs decades later; this is the up-front financial commitment.

**Fission reference:**
- NRC 10 CFR 50.75(c) minimum: ~\$500-700M in 2023\$ for a 1 GWe PWR (after escalation from 1986$ base).
- Actual costs: $300-400M per reactor (NRC), $750-1,250/kWe (OECD-NEA 2016 survey, median $1,000/kWe).
- Dominant cost drivers: spent fuel management, decontamination of fission products, long-lived actinide waste disposal, and licensed nuclear-grade demolition.

**Why fusion decommissioning is fundamentally different:**

1. **No spent fuel.** Fission spent fuel management (dry cask storage, geological disposal) is the single largest decommissioning cost. Fusion has no equivalent.
2. **Short-lived activation.** Fusion activated materials (RAFM steels, tungsten, vanadium alloys) decay to clearance levels within 50-100 years. Fission produces actinides that remain hazardous for >100,000 years.
3. **Recyclable materials.** With low-activation material choices, most fusion structural components can be recycled after a decay period, reducing waste disposal costs.
4. **No criticality hazard.** No fissile material inventory to secure during decommissioning.
5. **Part 30 vs Part 50.** Decommissioning under Part 30 has far lighter regulatory requirements than Part 50 reactor decommissioning. No Licensed Decommissioning Plan approval process.

**Fuel-dependent decommissioning cost drivers:**

| Driver | DT | DD | DHe3 | pB11 |
|--------|----|----|------|------|
| Neutron-activated structures | Heavy (14 MeV, full flux) | Moderate (~1/3 DT flux) | Light (~5% DT flux) | Minimal (~0.01% DT flux) |
| Tritium decontamination | Major (breeding blanket, processing systems, contaminated surfaces) | Minor (small in-situ production) | Minor (DD side reactions) | None |
| Remote dismantling needed | Yes (activated components) | Partial (some activated areas) | Minimal | No |
| Waste classification | Intermediate-level (decays to LLW in ~50yr) | Low-to-intermediate | Low-level | Near-conventional |
| Conventional demolition | Same for all fuels — buildings, turbine hall, electrical systems, cooling towers |

**Estimation approach:**

Start from the fission reference ($1,000/kWe median for Part 50 reactors) and systematically remove cost drivers that don't apply to fusion:

| Cost Driver | Fission ($/kWe) | DT Fusion | pB11 Fusion |
|-------------|-----------------|-----------|-------------|
| Spent fuel management | ~$300 | $0 | $0 |
| Long-lived actinide waste | ~$200 | $0 | $0 |
| Part 50 regulatory overhead | ~$100 | $0 (Part 30) | $0 (Part 30) |
| Activated structure removal | ~$150 | $150 (similar volume, shorter-lived) | $15 (minimal activation) |
| Tritium decontamination | $0 | $100 | $0 |
| Conventional demolition | ~$150 | $150 | $150 |
| Security during decommissioning | ~$100 | $10 (no SNM) | $5 (industrial) |
| **Total** | **~$1,000** | **~$410** | **~$170** |

Cross-check: For a 1 GWe plant, DT decommissioning ≈ $410M, pB11 ≈ $170M. These are full decommissioning costs over the ~10-20 year decommissioning period. The capitalized *provision* at COD is the present value of this future obligation, discounted over the ~40-year plant life. At 3% real discount rate over 40 years: PV factor ≈ 0.31. So the capitalized provision is roughly 31% of the total decommissioning cost.

| Fuel | Est. Decommissioning Cost | Capitalized Provision (PV) |
|------|---------------------------|---------------------------|
| DT | $410M ($410/kWe) | $127M ($127/kWe) |
| DD | $300M ($300/kWe) | $93M ($93/kWe) |
| DHe3 | $210M ($210/kWe) | $65M ($65/kWe) |
| pB11 | $170M ($170/kWe) | $53M ($53/kWe) |

**Validation:** DT at ~$410/kWe total decommissioning is ~40% of fission's $1,000/kWe median. This is plausible — DT fusion eliminates the dominant fission cost drivers (spent fuel, actinides, Part 50 overhead) but retains significant activated structure removal and adds tritium decontamination. pB11 at $170/kWe is ~17% of fission, consistent with a near-industrial plant with minimal radiological legacy.

### C59: Contingency on Supplementary Costs

**Estimate:** 15% of (C51 + C52 + C53 + C54 + C55 + C56). This is lower than the 20% contingency used for direct costs because supplementary costs are derived from more mature cost bases (insurance quotes, tax rates, commodity prices) than custom-engineered construction.

## Worked Example: 1 GWe Reference Plant

Assuming CAS20 = $4,000M, CAS30 = $800M (20% of CAS20), CAS22-28 = $3,000M (75% of CAS20):

| Sub-account | Formula | DT | DD | DHe3 | pB11 |
|-------------|---------|-----|-----|------|------|
| C51 Shipping | 1.5% × CAS20 | $60M | $60M | $60M | $60M |
| C52 Spare Parts | fuel% × CAS22-28 | $90M (3%) | $75M (2.5%) | $45M (1.5%) | $30M (1%) |
| C53 Taxes | 1% × CAS20 | $40M | $40M | $40M | $40M |
| C54 Insurance | 1.5% × (CAS20+CAS30) | $72M | $72M | $72M | $72M |
| C55 Fuel/Startup | See above | $40M | $0.1M | $10M | $0.1M |
| C56 Decommissioning | PV provision | $127M | $93M | $65M | $53M |
| Subtotal | | $429M | $340M | $292M | $255M |
| C59 Contingency | 15% of subtotal | $64M | $51M | $44M | $38M |
| **CAS50 Total** | | **$493M** | **$391M** | **$336M** | **$293M** |
| **$/kWe** | | **$493** | **$391** | **$336** | **$293** |
| **% of CAS20** | | **12.3%** | **9.8%** | **8.4%** | **7.3%** |

### Validation Against References

| Source | CAS50 | % of TCC | Notes |
|--------|-------|----------|-------|
| Woodruff MIF example | $213M | 10% | 637 MWe, includes decommissioning |
| MARS reference | $700M | 7.7% | Large plant |
| **This analysis (DT, 1 GWe)** | **$493M** | **~8%** | Includes decommissioning provision |
| **This analysis (pB11, 1 GWe)** | **$293M** | **~5%** | Near-industrial decommissioning |

The DT estimate at ~8% of approximate total capital is in the range of the Woodruff and MARS references. The pB11 estimate is lower, reflecting reduced spare parts, negligible fuel cost, and lighter decommissioning — exactly what we'd expect for a near-industrial plant.

## Assessment

**What we know with confidence:**

1. The current 1costingfe CAS50 values are placeholder-quality, with no source documentation and values that are 10-100× too low on several sub-accounts (especially insurance and decommissioning).
2. CAS50 is a significant cost account — typically 5-12% of total capital in nuclear/fusion literature.
3. The sub-account structure is well-established across EMWG, ARIES, and INL references.
4. Decommissioning is the largest and most fuel-dependent sub-account, driven by neutron activation and tritium contamination.
5. Fusion decommissioning is fundamentally cheaper than fission due to the absence of spent fuel, long-lived actinides, and Part 50 regulatory overhead.
6. Initial fuel cost is uniquely fuel-dependent: DT's tritium startup inventory ($30-60M) dwarfs the negligible cost of DD, DHe3, or pB11 fuels.

**What is NOT defensible:**

1. Sub-account precision beyond ±50%. These are pre-conceptual estimates for plants that don't exist. The sub-account split provides useful structure for sensitivity analysis, but the individual values are order-of-magnitude estimates.
2. The decommissioning cost reduction factors (DT at 41% of fission, pB11 at 17%) are based on engineering judgment about which fission cost drivers are eliminated, not on actual fusion decommissioning experience (none exists).
3. Tritium startup inventory cost ($30-60M) is highly uncertain — tritium supply is constrained to ~25 kg/year globally from CANDU reactors, and prices may rise as these reactors age.
4. Tax estimates are jurisdiction-specific and could vary 3× depending on site selection.

**Fuel-type differentiation summary:**

The fuel gradient in CAS50 is driven by three sub-accounts:
- **C52 (Spare Parts):** DT needs activated component spares; pB11 needs conventional industrial spares.
- **C55 (Fuel/Startup):** DT needs $30-60M of tritium; pB11 needs <$0.1M of hydrogen and boron.
- **C56 (Decommissioning):** DT has heavy activation + tritium contamination; pB11 has near-conventional end-of-life.

The DT/pB11 ratio of ~1.7:1 in total CAS50 reflects these three drivers. C51 (shipping), C53 (taxes), and C54 (insurance) are fuel-independent because they're driven by plant size and construction value, not plasma fuel.

## Recommendation for 1costingfe

Replace the current flat-value implementation with a sub-account model:

```python
def cas50_supplementary(cc, fuel, cas20, cas22_to_28, cas30, p_net, noak):
    """CAS50: Capitalized supplementary costs. Returns M$.

    Sub-account model with fuel-dependent spare parts, startup
    inventory, and decommissioning provisions. Shipping, taxes,
    and insurance scale with plant cost (fuel-independent).

    See docs/account_justification/CAS50_supplementary_costs.md
    """
    c51_shipping = cc.shipping_frac * cas20
    c52_spares = cc.spare_parts_frac(fuel) * cas22_to_28
    c53_taxes = cc.tax_frac * cas20
    c54_insurance = cc.construction_insurance_frac * (cas20 + cas30)
    c55_fuel = cc.startup_fuel_cost(fuel) * (p_net / 1000.0)
    c56_decom = cc.decom_provision(fuel) * (p_net / 1000.0)
    subtotal = c51_shipping + c52_spares + c53_taxes + c54_insurance + c55_fuel + c56_decom
    c59_contingency = cc.contingency_rate(noak) * subtotal
    return subtotal + c59_contingency
```

New `CostingConstants` fields:

| Parameter | Value | Unit | Derivation |
|-----------|-------|------|------------|
| `shipping_frac` | 0.015 | fraction of CAS20 | WNA ~2%, discounted for fusion (lighter components, no RPV) |
| `spare_parts_frac_dt` | 0.03 | fraction of CAS22-28 | Activated component spares, remote handling consumables |
| `spare_parts_frac_dd` | 0.025 | fraction of CAS22-28 | Similar to DT, longer replacement intervals |
| `spare_parts_frac_dhe3` | 0.015 | fraction of CAS22-28 | Mostly conventional, minor activation |
| `spare_parts_frac_pb11` | 0.01 | fraction of CAS22-28 | Conventional industrial spare parts |
| `tax_frac` | 0.01 | fraction of CAS20 | ~1% after typical energy project exemptions |
| `construction_insurance_frac` | 0.015 | fraction of (CAS20+CAS30) | Builder's risk 1-3%, midpoint for large energy project |
| `startup_fuel_dt` | 40.0 | M$ at 1 GWe | ~1.3 kg tritium at $30k/g |
| `startup_fuel_dd` | 0.1 | M$ at 1 GWe | Deuterium, commodity pricing |
| `startup_fuel_dhe3` | 10.0 | M$ at 1 GWe | He3 supply-constrained, uncertain |
| `startup_fuel_pb11` | 0.1 | M$ at 1 GWe | H + B11, industrial commodities |
| `decom_provision_dt` | 127.0 | M$ at 1 GWe | PV of $410M over 40yr at 3% real |
| `decom_provision_dd` | 93.0 | M$ at 1 GWe | PV of $300M over 40yr at 3% real |
| `decom_provision_dhe3` | 65.0 | M$ at 1 GWe | PV of $210M over 40yr at 3% real |
| `decom_provision_pb11` | 53.0 | M$ at 1 GWe | PV of $170M over 40yr at 3% real |

**Power scaling:** `startup_fuel` and `decom_provision` scale linearly with `p_net / 1000.0`. Spare parts, shipping, taxes, and insurance scale via their base accounts (CAS20, CAS22-28, CAS30), which already embed power scaling.

**Why this approach:**

1. **Sub-account structure** enables transparent sensitivity analysis — users can see which components drive CAS50 and vary them independently.
2. **Fuel dependence** is concentrated in the three sub-accounts where it matters (spare parts, fuel, decommissioning) while keeping fuel-independent accounts simple.
3. **Cost-driver scaling** — each sub-account scales with its actual cost driver (equipment cost for spares, construction value for insurance, plant size for decommissioning) rather than a single opaque fraction.
4. **Defensible against fission references** — the decommissioning analysis starts from well-documented fission costs and systematically removes inapplicable drivers.
5. **Corrects order-of-magnitude errors** in the current implementation — especially insurance ($0.5M → $72M) and decommissioning ($5M → $53-127M).

## References

1. Woodruff, S. "A Costing Framework for Fusion Power Plants," arXiv:2601.21724v2, January 2026.
2. GEN-IV EMWG. "Cost Estimating Guidelines for Generation IV Nuclear Energy Systems," GIF/EMWG/2007/004, Rev 4.2, 2007.
3. Shropshire, D. et al. "Advanced Nuclear Reactor Cost Estimation: Sodium-Cooled Fast Reactor Case Study," INL Sort_67398, 2024.
4. World Nuclear Association. "Economics of Nuclear Power," 2025.
5. Waganer, L.M. "ARIES Cost Account Documentation," UCSD-CER-13-01, June 2013.
6. NRC. "Backgrounder on Decommissioning Nuclear Power Plants," 2024.
7. OECD-NEA. "Costs of Decommissioning Nuclear Power Plants," 2016.
8. IAEA. "Decommissioning and Waste Management Considerations for Fusion," IAEA-TECDOC-2116, 2020.
9. DOE/INL. "Nuclear Energy Cost Estimates for Net Zero World Initiative," 2024.
10. El-Guebaly, L.A. et al. "Goals, Challenges, and Successes of Managing Fusion Activated Materials," Fusion Engineering and Design 83, 2008.
11. 10 CFR 50.75. "Reporting and recordkeeping for decommissioning planning."
12. 10 CFR 30.35. "Financial assurance and recordkeeping for decommissioning."
13. NY RPTL §485. "Nuclear Powered Electric Generating Facilities" (tax exemption).
