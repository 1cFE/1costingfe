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
  mirror: n_coils×4π)
- **B_max** = peak field on conductor (default 12 T)
- **R_coil** = effective coil radius (default 1.85 m)
- **n_coils** = number of independent solenoid coils (mirror only;
  default 10, calibrated to a Realta HAMMIR-class tandem mirror with
  4 end-plug HTS coils plus ~6 LTS central-cell solenoid coils
  discretizing the 50 m central cell. Simple-mirror devices like
  WHAM/BEAM/Anvil use n_coils ≈ 4)
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
| Mirror | 2.5× | Simple solenoid coils. Well-established manufacturing. n_coils=10 default (HAMMIR-class tandem: 4 end-plug HTS + 6 central-cell LTS solenoids over a 50 m central cell). Simple-mirror devices (WHAM/BEAM/Anvil) use n_coils≈4. |
| Pulsed FRC | 1.5× | Theta-pinch formation coils. Simple, repetitive geometry. |
| Theta pinch | 1.5× | Compression coils. Simple solenoid geometry. |
| MagLIF | 2.0× | Axial field solenoid. Moderate complexity (pulsed duty). |
| Mag. target | 1.5× | Guide-field solenoid. Small, simple coils. |
| Plasma jet | 1.5× | Guide-field solenoid. Small, simple coils. |
| Orbitron | 1.5× | Electrostatic confinement coils. |
| Polywell | 2.0× | Polyhedral magrid. Moderate 3D complexity. |
| IFE / Z-pinch / DPF | — | $0: no confinement magnets. |

### Validation

At reference (B=12T, R=1.85m, REBCO @ $50/kAm, 8× markup):
- Total conductor: ~1.29M kAm → $64.5M raw conductor
- With 8× markup: $516M total coil system
- This includes: conductor, winding, insulation, quench protection,
  structural casing, cryostat, power leads, instrumentation, testing

CFS SPARC used ~300 km of REBCO tape for a ~2m-class magnet.
A full tokamak power plant coil set is ~5–10× larger.

For mirrors at the same reference (B=12T, R=1.85m, REBCO @ $50/kAm,
2.5× markup, n_coils=10): total conductor 4.1M kAm, raw conductor
$205M, finished coil system $513M. With the previous n_coils=4
assumption the coil system was $205M, which understated a tandem
mirror's actual coil burden by 2.5×. Realta's commercial design
target is HAMMIR, a tandem mirror with a 50 m central cell flanked by
two end plugs (each end plug carrying two HTS mirror coils in the
Hammer evolution); 10 is a conservative count covering the four
end-plug HTS coils plus ~6 LTS central-cell solenoids at coarse pitch
along the central cell. Simple-mirror devices (Anvil/WHAM/BEAM) use
2 HTS end coils plus a small LTS central solenoid and would set
n_coils≈4. The value lives in `_COIL_DEFAULTS[MIRROR]` in
`src/costingfe/layers/cas22.py`.

### Resistive (copper) coil mass build-up

The `$/kAm × markup` model is correct when an expensive superconductor
dominates cost, which holds for high-field SC coils. For a low-field
resistive copper coil (an FRC at near-unity beta runs external fields of
order 0.1 to 1 T) the conductor is cheap per kAm, so the `$/kAm` path
collapses to a near-zero, unphysical number while the machine still has
tonnes of copper to wind and support. For copper coils the cost is
therefore a **mass build-up**, re-pricing the same ampere-meters by mass:

    ampere_meters = total_kAm × 1000
    m_Cu    = (ρ_Cu / J) × ampere_meters          # J = current density
    m_steel = f_struct × m_Cu
    C220103 = (m_Cu × cu_$/kg × cu_markup
               + m_steel × steel_$/kg × steel_markup) / 1e6

| Parameter | Value | Basis |
|---|---|---|
| ρ_Cu | 8960 kg/m³ | physical |
| J (current density) | 5 A/mm² | water-cooled copper practice |
| cu_$/kg | 11 | LME 2026-class copper |
| cu_markup | 3.5 | winding, insulation, cooling, jointing, test (ARIES/pyFECONS class) |
| f_struct | 0.6 | steel support mass / copper mass |
| steel_$/kg | 6 | fabricated structural steel |
| steel_markup | 3.0 | coil-case / inter-coil support fabrication |

Worked example (steady FRC, B=0.5 T, R=1.85 m, n_coils=4): total_kAm
about 68,500, so ampere-meters about 6.85e7, m_Cu about 123 t, m_steel
about 74 t, **C220103 about $6.0M** (sourced range $1.4 to 14M). The
field model gave about $0.1M at the same field, which is unphysical.

The branch is selected by `coil_material == COPPER`, so it applies
uniformly to every copper concept (steady FRC, pulsed FRC, theta pinch,
polywell, orbitron) and leaves superconducting concepts (tokamak,
stellarator, mirror, dipole) on the `$/kAm` path unchanged. The mass
build-up reuses the concept's geometry factor `G`, so no new geometry
assumption is introduced.

---

## C220104: Supplementary Heating (MFE) / Primary Driver (pulsed)

This account covers different hardware depending on the confinement
family, analogous to how C220108 flips between divertor (MFE) and
target factory (IFE/MIF).

### Steady-state MFE: Supplementary Heating

    C220104 = Σ (cost_per_MW_i × P_i)  for i ∈ {NBI, ICRF, ECRH, LHCD}

These are **vendor-purchased turnkey systems**: the per-MW cost includes
the vendor's engineering, manufacturing, testing, and margin.

#### Power basis and the heating-split invariant

C220104 is priced per MW of **injected heating power**. The split
`{p_nbi, p_icrf, p_ecrh, p_lhcd}` is the same heating power the 0D power
balance uses (`q_sci = p_fus / p_input`); the wall-plug draw
`p_input / eta_pin` enters the recirculating power, not this capital line.
To keep the costed MW identical to the power-balance MW, the split is
treated as the heating **mix** and normalized so its total equals
`p_input`. Overriding `p_input` without the split therefore rescales the
split, and the costed heating always tracks the physics. A fully zero
split (electrostatic concepts whose input power is not NBI/RF heating,
e.g. orbitron, polywell) stays zero, so their NBI/RF capital is correctly
$0.

Pricing the supply-dominated NBI capital on wall-plug power
(`p_input / eta_pin`), which would auto-elevate low-coupling concepts
such as the FRC, is a deliberate further refinement that is not applied
here; the driver's recirculating-power burden is already represented via
`eta_pin` in the power balance.

#### Per-MW costs (M$/MW, 2023$)

| System | $/MW | Source | Scope |
|--------|-----:|--------|-------|
| NBI | 7.06 | ARIES/pyFECONS, calibrated to ITER NBI procurement | Ion source, accelerator, neutralizer, duct, cryo pumps, power supply |
| ICRF | 4.15 | ARIES/pyFECONS | RF generators, transmission lines, antenna, matching network |
| ECRH | 5.00 | ARIES/pyFECONS | Gyrotrons (1 MW each), transmission waveguides, launchers |
| LHCD | 4.00 | ARIES/pyFECONS | Klystrons, waveguide grills, power supply |

#### Validation

Default: 50 MW NBI → $353M.

ITER NBI system: 2 injectors × 16.5 MW = 33 MW total. ITER NBI cost
is estimated at EUR 300–500M (FOAK, including test facility and R&D).
→ EUR 9–15M/MW (FOAK). Our $7.06M/MW is consistent with NOAK pricing
(FOAK-to-NOAK learning-curve discount of ~30–50%).

ITER ECRH: 24 gyrotrons providing 20 MW total. Gyrotrons cost ~$1–2M
each (vendor-purchased). Total ECRH system ~EUR 100–200M → EUR 5–10M/MW.
Our $5M/MW is mid-range.

### Pulsed concepts: Primary Driver Capital

This is the pulsed analog of the magnet system (C220103) — the
hardware that provides confinement.  A tokamak confines with magnets;
laser IFE confines with a laser driver.  The electrical infrastructure
(capacitor banks, switches, charging circuits) is in C220107.

The cost basis differs by driver type:

- **Lasers, accelerators, and electromagnetic guns** are costed per joule of pulse
  energy:

      C220104 = driver_cost_per_MJ × E_driver

  Their capital is set by pulse energy (laser aperture, amplifier and pump-diode
  count, accelerator beam energy and storage-ring charge, coaxial-gun size and
  peak current), not by how often the driver fires.  Rep rate adds cooling and
  shortens consumable lifetime, but those are sub-dominant capital and O&M, not a
  linear multiplier on the driver itself.  An average-power basis (E_driver ×
  f_rep) would price the *same* laser 100× apart across the 0.1–15 Hz range
  concepts actually run at, which is unphysical.  A plasma-jet gun and a
  sheared-flow Z-pinch's coaxial gun follow the same logic: a single gun assembly
  is sized by its current and pulse energy, not its firing frequency.

- **Pneumatic and mechanical injectors** (mag-target) keep an average-power basis:

      C220104 = driver_cost_per_MW × P_driver,  P_driver = E_driver × f_rep

  because they accelerate target mass on every shot, and the handling and
  liquid-metal recirculation plant genuinely scales with throughput (average
  power), not just per-pulse energy.

The laser and heavy-ion per-MJ figures reproduce the prior $/W NOAK projections
at each driver's reference rep rate, then hold constant off-reference: laser
$8/W × 10 Hz = 80 M$/MJ; heavy-ion $12/W × 5 Hz = 60 M$/MJ.  The plasma-jet
coefficient (4 M$/MJ) likewise preserves its reference point (f_rep = 1 Hz).

#### Driver costs

| Concept | Basis | Coefficient | Hardware | Rationale |
|---------|-------|------------:|----------|-----------|
| Laser IFE | $/MJ | 80 | Diode-pumped solid-state laser | NIF-heritage optics at NOAK volume. Current DPSSL $20–50/W; NOAK target $8/W. Costed on pulse energy. |
| Heavy ion | $/MJ | 60 | RF linac + storage rings | Accelerator capital scales with per-pulse beam energy and ring charge. Higher than laser due to ring infrastructure. |
| MagLIF | $/MJ (preheat only) | 80 | Laser preheat system | Main driver is the electrical Z-pinch (C220107). C220104 carries only the preheat laser, costed per joule of preheat pulse energy (`e_preheat_mj`); same DPSSL class as the IFE driver. Set `e_preheat_mj = 0` for no-preheat magnetized compression (e.g. Pacific Fusion). |
| Plasma jet | $/MJ | 4 | Plasma gun array | Electromagnetic plasma guns, sized by per-pulse energy and current. More complex than pneumatics, simpler than lasers. |
| Staged Z-pinch | $/MJ | 1.5 | Coaxial gun + gas injection | Sheared-flow stabilization hardware: coaxial accelerator electrodes plus neutral-gas puff valves and fast pumping. Single, simpler coaxial assembly, so cheaper per MJ than the plasma-jet array. The Z-pinch cap bank itself is in C220107. |
| Mag. target | $/MW | 3 | Pneumatic pistons, liquid metal loop | Mechanical compression hardware. Throughput-scaled (mass moved each shot), so kept on average power deliberately. Mature industrial technology. |
| Z-pinch, DPF | — | 0 | — | Driver is purely electrical (capacitor bank), costed in C220107. No flow-formation gun (cf. staged Z-pinch). |
| Pulsed FRC, Theta pinch | — | 0 | — | Driver is magnetic coils, costed in C220103. |

#### Design rationale: avoiding double-counting

The split between C220104 (driver hardware) and C220107 (electrical
infrastructure) avoids double-counting:

- **Laser IFE**: C220104 = laser amplifiers + optics; C220107 = capacitor
  bank that fires the diodes (on $/J basis)
- **Z-pinch**: C220104 = $0; C220107 = full pulsed power system (Marx
  generators, transmission lines, on $/J basis)
- **Mag target**: C220104 = pneumatic compression hardware; C220107 =
  capacitor bank for guide-field pulsing (on $/J basis)
- **Staged Z-pinch**: C220104 = coaxial gun + gas injection; C220107 =
  full pulsed power system (on $/J basis)

#### Formation-electrode replacement (CAS72 O&M)

The plasma-facing coaxial-gun electrodes of the electromagnetic-gun concepts
(staged Z-pinch, plasma jet) erode under high current density and are replaced
periodically. This recurring cost is carried in CAS72 (scheduled replacement),
not in the C220104 capital line. Because the replacement interval can be
sub-annual, it is modeled as a levelized annual recurring cost rather than
discrete replacement events:

    annual = electrode_replace_frac × C220104 × n_shots_per_year / electrode_shot_lifetime

with `electrode_replace_frac = 0.5` (consumable-electrode share of the C220104
flow-drive capital; range 0.25 to 0.75) and `electrode_shot_lifetime = 1e8` shots
(high uncertainty, no NOAK data; range 1e7 to 1e9). At the reference staged
Z-pinch this is about $25M/yr levelized.

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

The account is a sum of two parts that scale on different drivers:

    C220106 = vessel_shell + pumping

### Vessel shell (volume-based)

    vessel_shell = vessel_unit_cost × V_vessel × (P_et / 1100)^0.6

where `vessel_unit_cost = 0.72 M$/m³`. This covers the welded stainless
chamber, port extensions, gauges, and leak detection, scaling with reactor
size.

### Pumping (gas-load-based)

Installed pumping speed is set by gas throughput and operating pressure,
`S_req = (Q_nbi + Q_fuel) / P_op`, costed at `pump_unit_cost` per (m³/s).
NBI neutral-gas load scales as `p_nbi/E_b`; fueling/exhaust load scales as
`(1-burn_fraction)/burn_fraction × p_fus/E_fus`. The per-concept operating
pressure `vac_op_pressure_pa` is the key driver: high-pressure tokamak
divertors pump cheaply, low-pressure linear devices (mirror, FRC) pump
expensively. Full derivation, calibration, per-concept results, and
uncertainties are in
[CAS220106_vacuum_pumping.md](CAS220106_vacuum_pumping.md).

### Validation

Vessel-shell at reference ($151M): vessel volume ~210 m³.

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
