# CAS22 Core Reactor Components: C220101–C220106

**Date:** 2026-03-16
**Status:** Justified — values validated, documented

## Overview

C220101–C220106 cover the reactor island core: the most fusion-specific
and technically complex components.  These 6 accounts use either
**volume-based** costing (blanket, shield, structure, vessel) or
**physics-based** scaling (coils, heating).

At reference parameters (1 GWe DT tokamak, CATF spherical tokamak
geometry with R0=3.0m, κ=3.0):

| Account | Description | Cost (M$) | Method |
|---------|-------------|----------:|--------|
| C220101 | First wall + blanket | 389 | Volume × thermal intensity |
| C220102 | Shield | 261 | Volume × fuel scale × thermal intensity |
| C220103 | Coils | 516 | Conductor kAm × $/kAm × markup |
| C220104 | Supplementary heating | 353 | Per-MW linear (NBI + ICRF + ECRH + LHCD) |
| C220105 | Primary structure | 28 | Volume × power scale |
| C220106 | Vacuum system | 151 | Volume × power scale |
| **Subtotal** | | **1,698** | |

---

## C220101: First Wall + Blanket + Neutron Multiplier

### Costing model

    C220101 = blanket_unit_cost(fuel) × V_blanket × (P_th / 2500)^0.6

where `V_blanket` is the combined volume of first wall + blanket +
reflector (from radial build geometry) and the 0.6 exponent captures
thermal intensity (higher power → better cooling, thicker walls,
higher-grade materials per unit volume).

### Unit costs (M$/m³)

| Fuel | Unit cost | Rationale |
|------|----------:|-----------|
| DT | 0.60 | Full breeding blanket (RAFM steel structure + PbLi/Li breeder + Be neutron multiplier + FW W armor). TBR > 1.05 required. Complex assembly: HIPed joints, cooling channels, tritium barrier coatings. |
| DD | 0.30 | Energy-capture blanket (no breeding). RAFM steel + coolant channels. Simpler than DT (no breeder, no multiplier). |
| DHe3 | 0.08 | Minimal blanket. ~5% neutron fraction → thin shielding layer. Simple steel structure. |
| pB11 | 0.05 | X-ray shielding only. Thin metallic liner, conventional materials. |

### Validation

For a DT tokamak blanket of ~650 m³ at 2535 MW thermal:
- Material mass: ~3,000–5,000 tonnes (RAFM steel + breeder + multiplier)
- Raw material cost: RAFM steel ~$30–50/kg, fabricated nuclear-grade
  components ~$100–200/kg (3–5× manufacturing/QA markup)
- At $150/kg average × 3,500 tonnes = $525M. Our $389M is conservative,
  reflecting NOAK learning-curve reduction.

ITER comparison: 440 blanket/shield modules, ~2,000 tonnes total.
ITER blanket is FOAK with bespoke international procurement; NOAK
serial production would be substantially cheaper per unit.

### TODO: wall_material cost multiplier

The code has a TODO for incorporating wall_material-specific cost
multipliers.  Different first-wall armor materials have significantly
different fabrication costs:
- **Tungsten (W) tiles**: $300–600/kg fabricated (PVD/CVD coating,
  castellated tiles, CuCrZr heat sink brazing)
- **Flowing lithium**: Minimal FW armor cost (flowing liquid metal),
  but complex manifolding and MHD insulation
- **SiC composites**: $500–1,000/kg fabricated (CVD/CVI process,
  limited suppliers)

This multiplier is deferred for future work.  The current volume-based
unit costs implicitly assume a tungsten-armored, steel-structured blanket
(the dominant concept for DT tokamaks).

---

## C220102: Shield

### Costing model

    C220102 = shield_unit_cost × V_shield × fuel_scale(fuel) × (P_th / 2500)^0.6

where `shield_unit_cost = 0.74 M$/m³` (DT reference) and fuel_scale
reduces the shield requirement for lower-neutron fuels.

### Fuel scaling factors

| Fuel | Factor | Rationale |
|------|-------:|-----------|
| DT | 1.0 | Full shield: HT shield (steel/WC) + LT shield (borated water) + bioshield (concrete). 14.1 MeV neutrons require ~1m total shielding. |
| DD | 0.7 | Reduced: 2.45 MeV neutrons, ~30% lower flux from side-reaction tritium. |
| DHe3 | 0.3 | Light: ~5% neutron fraction, thin neutron shield only. |
| pB11 | 0.1 | Minimal: X-ray shielding only, thin metallic/concrete layer. |

### Validation

At reference ($261M for DT): shield volume ~350 m³ of steel + borated
water.  Steel at ~$20–40/kg fabricated, total mass ~2,000 tonnes →
$40–80M raw material.  The higher cost reflects:
- Nuclear-grade welding and NDE (100% volumetric inspection)
- Borated water piping and containment
- Complex geometry (conformal to blanket/vessel)
- Integration/alignment with blanket modules

---

## C220103: Coils (Magnets)

### Costing model

    total_kAm = G × B_max × R_coil² / (μ₀ × 1000)
    C220103 = total_kAm × $/kAm × markup / 1e6

where:
- **G** = geometry factor (tokamak: 4π², stellarator: 4π²×path_factor,
  mirror: 4×4π)
- **B_max** = peak field on conductor (default 12 T)
- **R_coil** = effective coil radius (default 1.85 m)
- **$/kAm** = conductor cost per kilo-amp-meter
- **markup** = manufacturing complexity multiplier

### Conductor pricing

| Material | Default $/kAm | Context |
|----------|-------------:|---------|
| REBCO HTS | 50 | NOAK target. Current market: $150–300/kAm. ARPA-E target for fusion viability: ~$50/kAm. Long-term: $10/kAm with scale-up. |
| Nb₃Sn | 7 | Mature technology. ITER specification. |
| NbTi | 7 | Commodity superconductor. LHC heritage. |
| Copper | 1 | Resistive magnets (pulsed concepts). |

The $50/kAm REBCO assumption is aggressive but represents the NOAK
cost target articulated by CFS, Tokamak Energy, and ARPA-E BETHE
program.  At current prices ($200/kAm), coils would cost $2.1B for the
reference tokamak — a major cost driver that fusion magnet manufacturers
are actively working to reduce.

### Manufacturing markup

| Concept | Markup | Rationale |
|---------|-------:|-----------|
| Tokamak | 8.0× | TF + CS + PF coil systems. Complex D-shaped winding, insulation, quench protection, structural casing, cryostat integration. Conductor is ~10–15% of finished magnet cost. |
| Stellarator | 12.0× | Non-planar 3D coil geometry. Tighter tolerances, longer winding paths (2× path factor), higher manufacturing complexity. |
| Mirror | 2.5× | Simple solenoid coils. Well-established manufacturing. 4 independent coils. |

### Validation

At reference (B=12T, R=1.85m, REBCO @ $50/kAm, 8× markup):
- Total conductor: ~1.29M kAm → $64.5M raw conductor
- With 8× markup: $516M total coil system
- This includes: conductor, winding, insulation, quench protection,
  structural casing, cryostat, power leads, instrumentation, testing

CFS SPARC used ~300 km of REBCO tape for a ~2m-class magnet.
A full tokamak power plant coil set is ~5–10× larger.

---

## C220104: Supplementary Heating

### Costing model

    C220104 = Σ (cost_per_MW_i × P_i)  for i ∈ {NBI, ICRF, ECRH, LHCD}

These are **vendor-purchased turnkey systems**: the per-MW cost includes
the vendor's engineering, manufacturing, testing, and margin.

### Per-MW costs (M$/MW, 2023$)

| System | $/MW | Source | Scope |
|--------|-----:|--------|-------|
| NBI | 7.06 | ARIES/pyFECONS, calibrated to ITER NBI procurement | Ion source, accelerator, neutralizer, duct, cryo pumps, power supply |
| ICRF | 4.15 | ARIES/pyFECONS | RF generators, transmission lines, antenna, matching network |
| ECRH | 5.00 | ARIES/pyFECONS | Gyrotrons (1 MW each), transmission waveguides, launchers |
| LHCD | 4.00 | ARIES/pyFECONS | Klystrons, waveguide grills, power supply |

### Validation

Default: 50 MW NBI → $353M.

ITER NBI system: 2 injectors × 16.5 MW = 33 MW total. ITER NBI cost
is estimated at EUR 300–500M (FOAK, including test facility and R&D).
→ EUR 9–15M/MW (FOAK). Our $7.06M/MW is consistent with NOAK pricing
(FOAK-to-NOAK learning-curve discount of ~30–50%).

ITER ECRH: 24 gyrotrons providing 20 MW total. Gyrotrons cost ~$1–2M
each (vendor-purchased). Total ECRH system ~EUR 100–200M → EUR 5–10M/MW.
Our $5M/MW is mid-range.

---

## C220105: Primary Structure

### Costing model

    C220105 = structure_unit_cost × V_structure × (P_et / 1100)^0.5

where `structure_unit_cost = 0.15 M$/m³`.

This covers the structural steel framework supporting the reactor
components: gravity supports, thermal shields, inter-coil structure,
and machine base.  The 0.5 exponent (milder than blanket/shield)
reflects that structural loads scale sub-linearly with power.

### Validation

At reference ($28M): structure volume ~200 m³, ~1,500 tonnes of
structural steel at ~$10–20/kg fabricated.  This is a small account
dominated by conventional heavy structural steelwork.

---

## C220106: Vacuum System

### Costing model

    C220106 = vessel_unit_cost × V_vessel × (P_et / 1100)^0.6

where `vessel_unit_cost = 0.72 M$/m³`.

This covers the vacuum vessel (double-walled, welded stainless steel),
port extensions, cryopumps, vacuum gauges, and leak detection systems.

### Validation

At reference ($151M): vessel volume ~210 m³.

ITER vacuum vessel: 5,200 tonnes, 9 sectors.  Assembly contract alone
is $180M (Westinghouse, 2025).  Total ITER VV fabrication + assembly is
estimated at EUR 500M–1B (FOAK).  For a smaller NOAK vessel (~2,000
tonnes), $151M is reasonable with serial production learning.

ITER VV cost per tonne: ~$100–200k/tonne (FOAK).  NOAK fusion vessel:
~$50–75k/tonne is achievable → $100–150M for 2,000 tonnes.  Consistent
with our model.

---

## Cross-Cutting Considerations

### FOAK vs NOAK

All values represent **NOAK** (Nth-of-a-kind) costs.  ITER and current
fusion company costs are FOAK/prototype, typically 2–5× higher due to:
- Bespoke one-off engineering
- International collaboration overhead (ITER)
- Immature supply chains
- First-article qualification

### Fabrication markup principle

For custom-fabricated components (blanket, shield, vessel, structure):
**cost = material × mass × fabrication markup**.  Nuclear-grade
fabrication, assembly, inspection, and acceptance testing multiplies
raw material cost by 3–10×:
- Standard structural steel: 1.5–2.5× markup
- Nuclear-grade welded assemblies: 3–5× markup (100% NDE, QA/QC)
- Complex internals (blanket modules): 5–10× markup (HIP joints,
  cooling channels, tritium barriers)

### Volume-based vs power-based

Volume-based costing (C220101, C220102, C220105, C220106) captures
reactor size from geometry.  The thermal/electrical intensity exponent
(0.5–0.6) captures the fact that higher-power-density reactors need
better materials and cooling per unit volume.  This hybrid approach is
more physical than pure power-law scaling.

---

## References

- Waganer, L. M., "ARIES Cost Account Documentation," UCSD-CER-13-01,
  University of California San Diego, June 2013.
- ITER Organization, "Blanket," https://www.iter.org/machine/blanket
- ITER Organization, "Vacuum Vessel," https://www.iter.org/machine/vacuum-vessel
- ITER Organization, "External Heating Systems,"
  https://www.iter.org/machine/supporting-systems/external-heating-systems
- Westinghouse Electric Company, ITER Assembly Contract ($180M), 2025.
- ARPA-E, "Advanced HTS Conductors Customized for Fusion," BETHE program.
- CFS, "HTS Magnets," https://cfs.energy/technology/hts-magnets/
- Woodruff, S., "A Costing Framework for Fusion Power Plants,"
  arXiv:2601.21724, January 2026.
