# CAS80: Annualized Fuel Cost

**Date:** 2026-03-16
**Status:** Implemented

## Account Structure

CAS80 covers the annualized cost of fuel isotope consumables. Sub-accounts per GEN-IV EMWG (2007):

| CAS | Description |
|-----|-------------|
| 81 | Refueling Operations |
| 84 | Fuel |
| 86 | Processing Charges |
| 87 | Special Nuclear Materials |
| 89 | Contingency on Annualized Fuel Costs |

For fusion, the account simplifies dramatically compared to fission. There is no fuel fabrication chain (mining → conversion → enrichment → pellet fabrication → assembly). Fusion fuel is purchased as enriched isotopes at market prices and injected directly (gas puff, pellet injection, or neutral beam). The sub-account breakdown (C81-C89) is not modeled; CAS80 is computed as a single annual total.

**Relationship to CAS220112 (Isotope Separation Plant):**

CAS80 and CAS220112 are mutually exclusive models of the same cost. If CAS80 pays enriched market prices (e.g., $2,175/kg for deuterium), the supplier's extraction and enrichment costs are already included. An on-site separation plant (CAS220112) would double-count. We use market purchase → CAS220112 = 0. See `CAS220112_isotope_separation.md`.

**Relationship to CAS220500 (Fuel Handling):**

CAS220500 covers the on-site fuel handling infrastructure (injection systems, exhaust processing, tritium containment). CAS80 covers the annual consumable cost of the isotopes themselves. These are complementary, not overlapping: CAS220500 is capital cost of equipment, CAS80 is operating cost of fuel.

**Tritium is not a CAS80 item for DT fuel.** Tritium is bred on-site in the lithium breeding blanket (CAS220101) and processed by the fuel handling system (CAS220500). The CAS80 cost for DT is deuterium + Li-6 for breeding blanket replenishment. The tritium itself is self-produced, not purchased.

## Source Documents

### Primary sources (read directly)

1. **Woodruff, S.** "A Costing Framework for Fusion Power Plants," arXiv:2601.21724v2, January 2026.
   - Eqs. 24-27: DT fuel cost formula from fusion power, deuterium mass, and unit price.
   - Table 3: CF = $1.0M/yr for 637 MWe worked example.
   - Fuel cost treated as DT-only; other fuel cycles not shown.

2. **pyfecons source code.** `cas80_annualized_fuel.py` (MFE and IFE versions).
   - MFE: Same formula as 1costingfe — per-reaction cost × reactions/yr.
   - IFE: Uses target cost interpolation vs. implosion frequency (not adopted).
   - Unit costs hardcoded: D $2,175/kg, Li-6 $1,000/kg, He-3 $2M/kg, H $5/kg, B-11 $10k/kg.
   - No burn-fraction correction in pyfecons.

3. **GEN-IV EMWG (2007).** "Cost Estimating Guidelines for Generation IV Nuclear Energy Systems," Rev 4.2.
   - Account 80 defined for fission: fuel fabrication, reprocessing, special nuclear materials.
   - Fission CAS80 includes complex fuel cycle cost models (mining, enrichment SWU, fabrication, disposal).
   - Not directly applicable to fusion — structure retained for COA compatibility.

4. **Fusion fuel isotope sourcing analysis.** `fusion-tea/knowledge/research/approved/20260211-fusion-fuel-isotope-sourcing.md`.
   - Comprehensive 60KB analysis of D, T, He-3, B-11, H-1 production pathways, costs, and supply constraints.
   - Deuterium: $1,500-3,500/kg range, functionally inexhaustible (33g/m^3 seawater).
   - Li-6: $1,000/kg enriched, global capacity ~1-2 t/yr (fleet bottleneck).
   - He-3: $2M/kg, ~15 kg/yr terrestrial supply (show-stopper for DHe3 at scale).
   - B-11: $10,000/kg FOAK (custom lab), $75/kg NOAK (industrial chemical exchange).
   - Protium: $5/kg, commodity.

5. **CAS220112 isotope separation justification.** `docs/account_justification/CAS220112_isotope_separation.md`.
   - Documents the market-purchase decision and consumption analysis.

### Secondary sources

6. **STARFIRE (1980).** Deuterium baseline cost, inflation-adjusted via GDP IPD.
7. **LPP Fusion (2018).** B-11 procurement: 93g of 99.9% B-11 at $600/g from Russian isotopic purification.

## Formula

### Base annual fuel cost

For each fuel cycle, the annual fuel expenditure is:

```
A_fuel = N_mod × P_fus × (8760 × 3600) × f_avail × c_rxn / (Q_eff × e_MeV)
```

Where:
- `N_mod`: number of reactor modules
- `P_fus`: fusion power per module (MW)
- `8760 × 3600 = 31,536,000`: seconds per year
- `f_avail`: plant availability (capacity factor)
- `c_rxn`: cost per fusion reaction ($ — sum of consumed isotope masses × unit prices)
- `Q_eff`: effective energy released per fusion event (MeV)
- `e_MeV = 1.602 × 10^-13`: J per MeV

The formula converts fusion power (MW = MJ/s) into reactions per second via Q_eff, then multiplies by cost per reaction and seconds per year. The 10^6 (MW→W) and /10^6 ($→M$) cancel.

### Fuel-specific parameters

**DT: D + T → He-4(3.52 MeV) + n(14.06 MeV), Q = 17.58 MeV**

```
c_rxn = m_D × u_D + m_Li6 × u_Li6
```

Deuterium is consumed directly. Tritium is bred on-site from lithium-6 in the blanket — the CAS80 cost is the Li-6 replenishment, not tritium purchase.

| Parameter | Value | Source |
|-----------|-------|--------|
| m_D | 3.344 × 10^-27 kg | scipy CODATA (deuteron mass) |
| u_D | $2,175/kg | STARFIRE (1980), inflation-adjusted |
| m_Li6 | 9.988 × 10^-27 kg | 6.015 × AMU |
| u_Li6 | $1,000/kg | 90% enriched Li-6 |
| Q_DT | 17.58 MeV | Nuclear data |

**DD: Two equiprobable branches + secondary burn**

```
Branch 1: D + D → T(1.01) + p(3.02), Q = 4.03 MeV
Branch 2: D + D → He3(0.82) + n(2.45), Q = 3.27 MeV
```

Tritium and He-3 produced in primary reactions can undergo secondary fusion:
- Secondary DT: probability f_T (default 0.969 at T=50 keV, τ_p=5s)
- Secondary DHe3: probability f_He3 (default 0.689)

```
Q_eff = 0.5 × Q_DD_PT + 0.5 × Q_DD_NHe3 + 0.5 × f_T × Q_DT + 0.5 × f_He3 × Q_DHe3
D_per_event = 2 + 0.5 × f_T + 0.5 × f_He3
c_rxn = D_per_event × m_D × u_D
```

Only deuterium is consumed (as market purchase). The secondary products are produced and burned in-situ.

| Parameter | Value | Source |
|-----------|-------|--------|
| Q_DD_PT | 4.03 MeV | Nuclear data |
| Q_DD_NHe3 | 3.27 MeV | Nuclear data |
| f_T | 0.969 | Burn fraction at 50 keV, 5s confinement |
| f_He3 | 0.689 | Burn fraction at 50 keV, 5s confinement |

**DHe3: D + He-3 → He-4(3.6 MeV) + p(14.7 MeV), Q = 18.35 MeV**

```
c_rxn = m_D × u_D + m_He3 × u_He3
```

| Parameter | Value | Source |
|-----------|-------|--------|
| m_He3 | 5.006 × 10^-27 kg | scipy CODATA (helion mass) |
| u_He3 | $2,000,000/kg | Scarcity pricing ($2,000/g), optimistic self-production |

He-3 dominates DHe3 fuel cost by orders of magnitude. At terrestrial supply of ~15 kg/yr vs ~105 kg/GWe-yr consumption, He-3 is a supply-chain show-stopper for fleet deployment.

**pB11: p + B-11 → 3 He-4, Q = 8.68 MeV**

```
c_rxn = m_p × u_H + m_B11 × u_B11
u_B11 = u_b11_noak if NOAK else u_b11  (FOAK/NOAK pricing split)
```

| Parameter | Value | Source |
|-----------|-------|--------|
| m_p | 1.673 × 10^-27 kg | scipy CODATA (proton mass) |
| u_H | $5/kg | Commodity hydrogen |
| m_B11 | 1.828 × 10^-26 kg | 11.009 × AMU |
| u_B11 (FOAK) | $10,000/kg | Lab-scale enrichment |
| u_B11 (NOAK) | $75/kg | Industrial chemical exchange distillation |

B-11 has a dramatic FOAK/NOAK cost split. Current enriched B-11 is produced only at lab scale ($600/g, LPP Fusion 2018). At industrial scale, B-11 enrichment is really B-10 depletion — the same chemical exchange distillation used for B-10 production (proven at ~480 kg/yr by OSTI data) yields >99% B-11 as tails at commodity cost.

### Burn-fraction correction

The base formula assumes all injected fuel undergoes fusion. In practice, only a fraction burns per pass:

```
A_fuel_eff = A_fuel × [1 + (1 - β_burn) / β_burn × (1 - η_rec)]
```

Where:
- `β_burn`: fraction of injected fuel that fuses per pass (default 0.05)
- `η_rec`: fraction of unburned fuel recovered and recycled (default 0.95)

The multiplier equals 1 when either β=1 (complete burn) or η=1 (perfect recovery). At default values (β=0.05, η=0.95): multiplier = 1.95.

**Impact by fuel:**

| Fuel | Raw annual cost (1 GWe) | With burn correction | Dominant isotope |
|------|------------------------|---------------------|-----------------|
| DT | ~$1.0M/yr | ~$2.0M/yr | Deuterium (cheap) — negligible vs LCOE |
| DD | ~$0.5M/yr | ~$1.0M/yr | Deuterium only — negligible |
| DHe3 | ~$50M/yr | ~$98M/yr | He-3 ($2M/kg) — LCOE-dominant |
| pB11 (NOAK) | ~$1.5M/yr | ~$2.9M/yr | B-11 at NOAK pricing — modest |
| pB11 (FOAK) | ~$200M/yr | ~$390M/yr | B-11 at $10k/kg — LCOE-dominant |

For DT and DD, fuel cost is <1% of LCOE regardless of burn-fraction assumptions. For DHe3, fuel cost dominates LCOE and is extremely sensitive to He-3 pricing and burn fraction. For pB11, the FOAK/NOAK transition is the critical parameter.

### Levelization

The annual fuel cost is levelized using the same growing-annuity procedure as CAS70, accounting for nominal cost escalation over the plant lifetime. See `CAS70_levelized_annual_cost.md` for the derivation.

## Assessment

**What is well-grounded:**

1. The per-reaction cost formula is physics-based and exact — given Q-values and isotope masses from nuclear data, the only free parameters are unit prices.
2. Deuterium pricing ($2,175/kg from STARFIRE, inflation-adjusted) is robust: the range is $1,500-3,500/kg and fuel cost is <1% of DT LCOE even at the high end.
3. The market-purchase model (CAS80 at enriched prices, CAS220112 = 0) is the correct approach for pre-conceptual costing — avoids double-counting and does not require speculative on-site plant designs.
4. The DD secondary burn model (f_T, f_He3 fractions) correctly accounts for the catalyzed DD cycle's energy boost.

**What is uncertain:**

1. **He-3 pricing** ($2M/kg) is a placeholder. Terrestrial He-3 supply (~15 kg/yr from tritium decay) cannot support even one DHe3 fusion plant (~105 kg/yr). Lunar mining or catalyzed DD self-production would change the economics entirely, but neither exists. DHe3 LCOE projections should be treated as aspirational.
2. **B-11 NOAK pricing** ($75/kg) assumes industrial-scale chemical exchange distillation. This process is proven for B-10 production but not demonstrated at the scale needed for a fleet of pB11 plants (~811 kg/GWe-yr). The FOAK price ($10,000/kg) is better grounded.
3. **Burn fraction** (default 5%) and **fuel recovery** (default 95%) are generic MCF assumptions, not concept-specific. Actual values depend on confinement scheme, fueling method, and exhaust processing design. For DHe3, where fuel cost dominates LCOE, these parameters warrant concept-specific analysis.
4. **Li-6 enrichment capacity** (~1-2 t/yr globally) is a fleet-level bottleneck for DT fusion. Per-plant fuel cost is low, but the supply chain does not currently exist at fleet scale.

**What does NOT matter:**

1. Deuterium price uncertainty: $1,500-3,500/kg range changes DT LCOE by <0.5%.
2. Protium price: commodity hydrogen at $5/kg is negligible for pB11.
3. Sub-account precision (C81/C84/C86/C87): fusion fuel is purchased and injected, not fabricated through a multi-step chain. The fission sub-account structure is not meaningful for fusion.

## Sources

1. Woodruff, S. "A Costing Framework for Fusion Power Plants." arXiv:2601.21724v2, January 2026.
2. pyfecons source code: `cas80_annualized_fuel.py` (MFE and IFE versions).
3. GEN-IV EMWG (2007). "Cost Estimating Guidelines for Generation IV Nuclear Energy Systems," Rev 4.2.
4. STARFIRE (1980). Deuterium baseline cost. Inflation-adjusted via GDP IPD.
5. "Fusion Fuel Isotope Sourcing: Production, Costs, and Supply Chains." Internal research, 2026-02-11.
6. LPP Fusion (2018). B-11 procurement: 93g of 99.9% B-11 at $600/g.
7. OSTI technical reports: B-10 chemical exchange distillation at 40 kg/month (~480 kg/yr).
