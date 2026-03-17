# Economic Advantage of pB11 over DT: Cost Chain Analysis

**Date:** 2026-03-16 (updated from 2026-03-07)
**Status:** Directional analysis (anchoring the 1c/kWh framing)
**Related:**
- `docs/analysis/pb11_side_reactions_nrc_regulatory_risk.md`
- `fusion-tea/knowledge/research/approved/20260208-fusion-reactor-subsystems-by-fuel-type.md`
- `fusion-tea/knowledge/research/approved/20260203-100000_fusion-regulatory-framework-dt-pb11.md`

---

## Purpose

This document traces the specific cost chain where aneutronic pB11 operation reduces capex and opex relative to DT, for a production-scale (1 GWe) plant. All numbers are from the 1costingfe model (NOAK, 1 GWe tokamak, 85% availability, 30-year lifetime, 7% WACC) unless otherwise noted. The goal is to anchor the cost advantage in concrete subsystem-level differences, not aspirational claims.

---

## 1. No Tritium Breeding Blanket

**What DT requires:** A ~1m thick structure surrounding the plasma, containing lithium (enriched Li-6), a neutron multiplier (beryllium or lead), structural steel, and coolant channels. This is the single most technically unproven subsystem in DT fusion. It must achieve TBR > 1.05 (tritium breeding ratio) while surviving 14.1 MeV neutron bombardment at ~20 dpa/yr.

**What pB11 requires:** A thin first-wall/vacuum vessel for X-ray absorption. No breeding, no lithium, no neutron multiplier.

| Parameter | DT | pB11 | Source |
|-----------|-----|------|--------|
| Blanket unit cost | 0.60 M$/m³ | 0.05 M$/m³ | `costing_constants.yaml` |
| C220101 (first wall + blanket) | $389M | $32M | Model output |
| Fuel handling (C220500) | $120M | $15M | Model output |

**Combined blanket + fuel handling saving: ~$462M**

The tritium processing plant for DT includes cryogenic distillation columns, isotope separation, tritium containment barriers (typically triple-walled), leak detection systems, and a full accountability program. pB11 fuel handling is boron powder injection — commodity industrial equipment.

---

## 2. No Neutron Shielding Beyond Plasma Vessel

**What DT requires:** 1-2 meters of heavy shielding (borated concrete, steel, water) to attenuate 14.1 MeV neutrons before they reach the magnets, buildings, and personnel. The shield volume is comparable to the blanket volume.

**What pB11 requires:** Minimal shielding. The 11B(alpha,n)14N side reaction produces ~2 MeV neutrons at <0.2% of total fusion power — roughly 10,000x lower neutron flux than DT. Standard structural walls provide adequate attenuation.

| Parameter | DT | pB11 | Source |
|-----------|-----|------|--------|
| Shield fuel scale | 1.0 | 0.1 | `cas22.py` |
| C220102 (shield) | $261M | $26M | Model output |

**Shielding saving: ~$235M**

The DT shield must protect superconducting magnets (which are intolerant of neutron heating and radiation damage) and provide biological shielding for personnel access during shutdown. pB11 magnets see negligible neutron flux, simplifying both the shield and the magnet design.

---

## 3. No Structural Activation: Longer Lifetimes, No LLW

**What DT requires:** 14.1 MeV neutrons create ~20 dpa/yr in first-wall and blanket materials, causing embrittlement, swelling, and transmutation. Core components must be replaced every 5 full-power-years (FPY). Each replacement requires remote handling in a hot cell, generating low-level radioactive waste (LLW) that must be characterized, packaged, shipped, and disposed of at a licensed facility.

**What pB11 requires:** Essentially no neutron damage (~0.1 dpa/yr). Core components last 50+ FPY — the full plant lifetime.

| Parameter | DT | pB11 | Source |
|-----------|-----|------|--------|
| Core component lifetime | 5 FPY | 50 FPY | `costing_constants.yaml` |
| dpa rate | ~20 dpa/yr | ~0.1 dpa/yr | `20260208-fusion-reactor-subsystems` |
| Cost per replacement (blanket + divertor) | ~$485M | N/A | C220101 + C220108 |
| CAS72 (annualized replacement) | $69M/yr | $0/yr | Model output |
| CAS72 over 30 years (undiscounted) | ~$2.1B | $0 | |

**30-year annualized replacement saving: $69M/yr (~$2.1B undiscounted)**

This is the largest single cost advantage. Each DT replacement cycle also involves:
- 3-6 months of downtime (lost revenue at ~$50M/month for 1 GWe)
- Remote handling operations in a hot cell ($93M capital for the building alone)
- LLW disposal costs ($5-20M per cycle)
- Surge maintenance staffing

The revenue loss from downtime alone could exceed $150-300M over the plant lifetime.

---

## 4. Remote Handling & Maintenance Equipment

**What DT requires:** A full remote handling suite — articulated robotic arms for in-vessel work, divertor cassette handlers, shielded cask transfer systems, hot cell robotic disassembly, in-pipe welding/cutting robotics, and rad-hardened actuators and sensors throughout. All of this exists because 14.1 MeV neutrons make the vacuum vessel interior inaccessible to humans (~10 Sv/hr contact dose at the first wall shortly after shutdown). UKAEA's RACE programme at Culham is developing many of these systems specifically for fusion; ITER's neutral beam RH system alone was a EUR 70M contract.

**What pB11 requires:** No rad-hardened robotics, no shielded casks, no hot cell robotic systems. However, pB11 still needs non-trivial maintenance equipment — vessel access tooling, heavy-lift capability, vessel opening/closing systems, and confined-space equipment. The cost is concept-dependent: a tokamak with narrow ports needs more sophisticated (but conventional) in-vessel tooling than an FRC with simple cylindrical access.

| Parameter | DT | pB11 | Source |
|-----------|-----|------|--------|
| Base cost (tokamak, 1 GWe) | $162M | $22M | Model output (C220110) |
| Base cost (FRC/linear, 1 GWe) | $60-80M | $10-15M | `CAS220110_remote_handling.md` |

**Capital saving for pB11: ~$140M** (tokamak geometry)

The key insight is that the cost difference is driven by two compounding factors: (1) the equipment itself must be rad-hardened for DT, adding ~3-5x cost premium over conventional robotics, and (2) DT needs the equipment far more frequently (replacement every 5-10 FPY vs rarely for pB11), driving both the capital scope and the operational wear budget.

See `docs/account_justification/CAS220110_remote_handling.md` for the full bottom-up cost build-up.

---

## 5. Special Materials (CAS27)

**What DT requires:** Initial inventory of breeding blanket fill material — PbLi eutectic (~4,000 tonnes for the default PbLi concept) plus enriched lithium. For HCPB concepts, 300-490 tonnes of beryllium pebbles ($180-300M).

**What pB11 requires:** No breeding material, no neutron multiplier, conventional coolants. CAS27 = $0.

| Parameter | DT | pB11 | Source |
|-----------|-----|------|--------|
| CAS27 (special materials) | $15M | $0 | Model output (PbLi default) |
| CAS27 if HCPB with Be | $180-300M | $0 | `CAS27_special_materials.md` |

**Saving: $15M** (PbLi default) or **$180-300M** (HCPB with beryllium)

For PbLi blankets this is a small account, but for HCPB concepts the beryllium cost is a major discriminator — global Be production is only ~300 tonnes/yr.

---

## 6. Compact Siting: Plasma-Facing Components Closer to Plasma

**What DT requires:** The blanket (~1m) + shield (~1m) create a 2m standoff between the plasma and the magnets. This sets the minimum machine size: for a tokamak with plasma major radius R, the magnet bore must be at least R + 2m. Larger magnets cost more, require more superconductor, and increase the building volume.

**What pB11 requires:** A thin first wall (10-20 cm) with no blanket or heavy shield. The magnet bore can be R + 0.2m.

| Effect | Impact | Direction |
|--------|--------|-----------|
| Magnet bore reduced by ~1.8m | Superconductor volume scales as bore² | Savings in coils (CAS220103) |
| Building volume reduced | Reactor building + hot cell smaller | Savings in CAS21 |
| Higher power density per unit volume | Smaller machine for same output | Capital intensity reduction |

**Estimated saving: $50-150M** (highly geometry-dependent)

This advantage is real but hard to quantify precisely because it depends on the confinement concept. For a tokamak, the effect is large (magnet cost dominates). For an FRC or mirror, the effect is smaller but still meaningful through building volume reduction.

---

## 7. Reduced Regulatory Burden

**What DT requires:** NRC Part 30 licensing (1-2 years), tritium accountability program, radiation protection program (ALARA, dosimetry, area monitoring), emergency preparedness plan, decommissioning financial assurance, and ongoing compliance reporting.

**What pB11 requires:** Per the Feb 2026 proposed rule, pB11 is explicitly named as in-scope for fusion machine regulation. The most likely outcome (50-60% probability) is Part 30 with graded/reduced requirements — not full exemption, but significantly streamlined relative to DT. See `pb11_side_reactions_nrc_regulatory_risk.md` for the full assessment.

| Parameter | DT | pB11 | Source |
|-----------|-----|------|--------|
| Licensing cost | $5M | $0.1-1.0M | `costing_constants.yaml`, regulatory analysis |
| Licensing time | 2.0 yr | 0.0-0.5 yr | `costing_constants.yaml`, regulatory analysis |
| Compliance staff | 5-10 FTE | 1-2 FTE | `20260203-regulatory-framework` |
| Compliance staff cost delta | $0.5-1.0M/yr | — | At ~$100k loaded |

**Licensing capital saving: ~$4-5M**
**Licensing time saving: 1.5-2.0 years** (flows into IDC — see below)
**Annual compliance saving: ~$0.5-1.0M/yr**

The licensing cost itself is small. The real impact is indirect: licensing time adds to the total project timeline, which increases interest during construction (CAS60).

---

## 8. Lower Decommissioning Liability

**What DT requires:** After 30 years of 14.1 MeV neutron bombardment, most reactor internals are LLW. Decommissioning requires:
- Characterization of activated materials (radionuclide inventory)
- Remote disassembly of activated components
- LLW packaging and shipment to licensed disposal facility
- Site remediation and final radiological survey
- NRC license termination

**What pB11 requires:** Negligible activation. Decommissioning is conventional industrial demolition plus recycling of high-value materials (superconductor, structural steel). No radiological characterization, no LLW disposal, no NRC license termination process.

| Parameter | DT | pB11 | Source |
|-----------|-----|------|--------|
| Decommissioning provision (PV, model) | $127M | $53M | `costing_constants.yaml` |
| Undiscounted decommissioning | $410M | $170M | PV basis: 40yr at 3% real discount |

**Decommissioning saving: ~$74M** (present-value provision in CAS50)

The model provisions decommissioning as a present-value lump sum in CAS50 (capitalized supplementary costs). The DT decommissioning scope includes characterization and disposal of activated reactor internals, tritium-contaminated systems, and the hot cell. pB11 decommissioning is conventional industrial demolition. See `CAS50_supplementary_costs.md`.

---

## 9. Staffing, O&M, and Owner's Costs

DT fusion requires approximately twice the operating staff of pB11, driven by nuclear-specific functions that pB11 does not need:

| Function | DT Staff | pB11 Staff | Source |
|----------|----------|------------|--------|
| Operations (incl. tritium plant) | 17-26 | 12-16 | `CAS70_staffing_and_om_costs.md` |
| Maintenance (rad-controlled + hot cell) | 40-75 | 20-35 | `CAS70_staffing_and_om_costs.md` |
| Administration (health physics, regulatory) | 12-25 | 7-10 | `CAS70_staffing_and_om_costs.md` |
| Technical (radwaste, licensing affairs) | 10-18 | 5-8 | `CAS70_staffing_and_om_costs.md` |
| Offsite (NRC interface) | 5-10 | 3-5 | `CAS70_staffing_and_om_costs.md` |
| **Total** | **~120** | **~60** | |

The staffing difference drives three cost accounts:

**CAS70 (Annual O&M):** The full annual O&M cost (staffing + materials + insurance + waste + regulatory) is $52M/yr for DT vs $24M/yr for pB11 at 1 GWe. Both scale with a power-law exponent of 0.5 for staffing economy of scale.

**CAS40 (Capitalized Owner's Costs):** Pre-operational recruitment, training, and housing costs for the same staff. The INL CAS40 methodology (1.5yr pre-op, 10% overhire, 25% recruiting, 58% benefits) gives $39M for DT vs $20M for pB11 at 1 GWe. See `CAS40_capitalized_owners_costs.md`.

| Account | DT (1 GWe) | pB11 (1 GWe) | Delta | Source |
|---------|-----------|-------------|-------|--------|
| CAS40 (owner's costs, one-time) | $39M | $20M | -$19M | `CAS40_capitalized_owners_costs.md` |
| CAS70 (annual O&M) | $52M/yr | $24M/yr | -$28M/yr | `CAS70_staffing_and_om_costs.md` |

**30-year levelized O&M delta: ~$28M/yr × 30yr ≈ $840M undiscounted**

Key DT-specific staff roles with no pB11 equivalent:
- Health physics department (10-15 FTE for 24/7 coverage)
- Tritium systems operators and technicians (10-20 FTE)
- Radwaste characterization and shipping technicians (5-10 FTE)
- Emergency preparedness coordinator (1-2 FTE)
- Tritium accountability officer

---

## 10. Interest During Construction (IDC)

The licensing timeline difference and regulatory ramp-up extend the DT project timeline, increasing IDC:

| Parameter | DT | pB11 | Source |
|-----------|-----|------|--------|
| Licensing time | 2.0 yr | 0.0-0.5 yr | `costing_constants.yaml` |
| Construction time | 6 yr | 6 yr | Reference estimate |
| Total pre-operation | ~8 yr | ~6 yr | |
| IDC (model output) | $1,022M | $693M | Model output (CAS60) |

**IDC saving: ~$329M**

On an overnight cost of ~$3.6-5.3B, the licensing time reduction and lower overnight cost compound to produce a substantial IDC difference.

---

## Summary: Where the Money Is

Ranked by magnitude of cost advantage for pB11 over DT (from current model output, 1 GWe NOAK tokamak):

| Rank | Cost Driver | Saving (est.) | CAS Account | Type |
|------|-------------|---------------|-------------|------|
| 1 | Component replacement (lifetime 5 vs 50 FPY) | $69M/yr annualized (~$2.1B undiscounted) | CAS72 | Opex + downtime |
| 2 | Annual O&M (120 vs 60 FTE, 30-year) | $840M undiscounted | CAS71 | Opex |
| 3 | No breeding blanket + tritium processing | $462M | C220101 + C220500 | Capex |
| 4 | Interest during construction | $329M | CAS60 | Capex |
| 5 | Buildings (tritium scaling + hot cell) | $295M | CAS21 | Capex |
| 6 | No neutron shielding | $235M | C220102 | Capex |
| 7 | Supplementary costs (spares, decom, startup) | $177M | CAS50 | Capex |
| 8 | Remote handling (rad-hardened vs conventional) | $140M | C220110 | Capex |
| 9 | Installation labor (smaller reactor subtotal) | $103M | C220111 | Capex |
| 10 | Indirect costs (scales with CAS20) | $251M | CAS30 | Capex |
| 11 | Compact siting (smaller magnets, buildings) | $50-150M | CAS22/CAS21 | Capex |
| 12 | Decommissioning provision (PV) | $74M | CAS50 | Liability |
| 13 | Owner's costs (pre-op staffing) | $19M | CAS40 | Capex |
| 14 | Special materials (blanket fill) | $15M | CAS27 | Capex |
| 15 | Regulatory (licensing cost + compliance) | $15-25M | CAS10 | Both |

**Total overnight cost delta: $1,709M** ($5,316M DT vs $3,607M pB11)
**Total capital delta: $2,037M** ($6,338M DT vs $4,301M pB11)

Against a DT overnight capital cost of ~$5.3B per GWe, this represents a **32% overnight cost reduction**.

---

## What pB11 Does NOT Save

Several major cost categories are identical or nearly identical between DT and pB11:

| Category | Why No Savings | DT Cost | pB11 Cost |
|----------|---------------|--------:|----------:|
| Magnets/coils (C220103) | Both need high-field superconducting magnets | $516M | $516M |
| Heating systems (C220104) | pB11 needs *more* heating (higher ignition T) | $353M | $353M |
| Turbine island (CAS23-26) | Both convert thermal energy via steam cycle | $428M | $424M |
| Power supplies (C220107) | Same magnet power requirements | $89M | $89M |
| Divertor (C220108) | Same thermal exhaust handling | $96M | $95M |
| Primary structure (C220105) | Same structural support | $28M | $28M |
| Vacuum system (C220106) | Same vessel requirements | $151M | $150M |
| Digital twin (CAS28) | Same complexity | $5M | $5M |

**Heating systems deserve a caution flag.** pB11 requires ion temperatures of ~200-300 keV (vs ~15 keV for DT). The heating power required to reach and sustain these temperatures is a major physics challenge that could increase CAS220104 costs. If pB11 requires 2-3x more heating power than DT, the heating system cost premium partially offsets savings elsewhere. The current model uses the same default heating power (50 MW NBI) for both fuels — a known simplification.

---

## Fuel Cost: The FOAK Penalty

Fuel cost is the one area where pB11 is more expensive than DT at first-of-a-kind. The model includes a burn-fraction correction: at 5% single-pass burn and 95% fuel recovery, the effective fuel cost multiplier is ~1.95×.

| Fuel | CAS80 (levelized, NOAK) | Source |
|------|------------------------:|--------|
| DT | $1.0M/yr | Model output |
| pB11 (NOAK, B-11 at $75/kg) | $0.2M/yr | Model output |
| pB11 (FOAK, B-11 at $10k/kg) | ~$30M/yr | Estimated from FOAK pricing |

The FOAK B-11 cost ($10,000/kg) reflects the absence of industrial-scale isotope enrichment. At NOAK ($75/kg, via chemical exchange distillation), pB11 fuel cost is negligible. This is a deployment-rate problem, not a fundamental cost problem — the first few plants pay the enrichment premium, but it disappears with scale. See `CAS80_annualized_fuel_cost.md` for the full derivation including burn-fraction and fuel-recovery parameters.

---

## LCOE Framing

Model outputs for a 1 GWe NOAK tokamak at 85% availability, 30-year life, 7% WACC, 2% inflation:

| Metric | DT | pB11 | Delta |
|--------|---:|-----:|------:|
| Overnight capital | $5,316M | $3,607M | -$1,709M |
| Total capital (incl. IDC) | $6,338M | $4,301M | -$2,037M |
| CAS40 (owner's costs) | $39M | $20M | -$19M |
| CAS71 (levelized O&M) | $72M/yr | $33M/yr | -$39M/yr |
| CAS72 (levelized replacement) | $69M/yr | $0/yr | -$69M/yr |
| CAS80 (levelized fuel) | $1.0M/yr | $0.2M/yr | -$0.8M/yr |
| **LCOE** | **$87.7/MWh** | **$51.0/MWh** | **-$36.7/MWh** |

The $36.7/MWh LCOE advantage is structural and spread across most CAS accounts:
- Lower direct costs (no breeding blanket, lighter shielding, conventional maintenance equipment)
- Lower indirect/owner/supplementary costs (smaller staff, lower decommissioning)
- Lower O&M (half the staff, no radwaste, no remote handling operations)
- No component replacement (50 FPY lifetime vs 5 FPY for DT)

**The 1c/kWh ($10/MWh) target** requires getting the pB11 LCOE from ~$51/MWh down by another ~5x. That is a separate question involving learning rates, nth-of-a-kind cost reduction, and whether the CHARM confinement concept delivers the compactness its proponents claim. But the structural advantage over DT — ~$1.7B per GW in avoided overnight cost — is the foundation that makes the aspiration worth pursuing.

---

## Caveats

1. **pB11 ignition is undemonstrated.** All cost savings are moot if the plasma physics doesn't work. The Lawson criterion for pB11 at 200-300 keV is far more demanding than DT at 15 keV.

2. **Bremsstrahlung losses.** pB11 radiates heavily via bremsstrahlung (Z_eff is higher). This power must be recovered thermally, reducing net electric efficiency. Ochs et al. (2025) and others are working on strategies to manage this.

3. **Alpha ash management.** Ochs, Kolmes, and Fisch (2025) show that alpha ash must be removed faster than it accumulates, or the plasma poisons itself. This is a solved problem in principle (demixing strategies exist) but undemonstrated in practice.

4. **Heating system cost.** The 10-20x higher ignition temperature for pB11 may require substantially more expensive heating systems, partially offsetting other savings. The current model uses the same 50 MW NBI default for all fuels.

5. **FOAK B-11 fuel cost.** The first few pB11 plants pay $10,000/kg for enriched B-11 (vs $75/kg at industrial scale). This FOAK penalty adds ~$30M/yr levelized — significant but temporary.

6. **Same coil/heating costs assumed.** The model currently uses the same default geometry (R0, b_max, r_coil) and heating power for all fuels. A pB11-optimized machine may differ substantially, affecting CAS220103 and CAS220104 in either direction.

---

## Sources

- `docs/account_justification/CAS10_preconstruction_costs.md` — licensing costs/times
- `docs/account_justification/CAS21_buildings.md` — building costs, fuel-dependent scaling
- `docs/account_justification/CAS22_reactor_components.md` — blanket, shield, coil costs
- `docs/account_justification/CAS22_plant_systems.md` — plant systems, installation labor
- `docs/account_justification/CAS220110_remote_handling.md` — remote handling cost build-up
- `docs/account_justification/CAS27_special_materials.md` — blanket fill materials
- `docs/account_justification/CAS40_capitalized_owners_costs.md` — pre-operational staffing costs
- `docs/account_justification/CAS50_supplementary_costs.md` — spares, decommissioning, startup fuel
- `docs/account_justification/CAS60_interest_during_construction.md` — IDC methodology
- `docs/account_justification/CAS70_staffing_and_om_costs.md` — staffing comparison (120 vs 60)
- `docs/account_justification/CAS80_annualized_fuel_cost.md` — fuel cost formula and burn fraction
- `docs/analysis/pb11_side_reactions_nrc_regulatory_risk.md` — regulatory risk assessment
- `costing_constants.yaml` — unit costs, lifetimes, fuel prices
- [Ochs, Kolmes, Fisch — "Preventing ash from poisoning proton-boron 11 fusion plasmas," Phys. Plasmas 32, 052506 (2025)](https://pubs.aip.org/aip/pop/article/32/5/052506/3347125/Preventing-ash-from-poisoning-proton-boron-11)
