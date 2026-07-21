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
| C220103 | Coils | 500 | Conductor kAm × $/kAm × markup |
| C220104 | Supplementary heating | 353 | Per-MW linear (NBI + ICRF + ECRH + LHCD) |
| C220105 | Primary structure | 28 | Volume × power scale |
| C220106 | Vacuum system | 151 | Volume × power scale |
| **Subtotal** | | **1,698** | |

---

## C220101: First Wall + Blanket + Neutron Multiplier

### Costing model

Blanketed machines (blanket_form other than `none`):

    C220101 = blanket_unit_cost(fuel) × structure_factor(form) × V_blanket × (P_th / 2500)^0.6

where `V_blanket` is the combined volume of first wall + blanket +
reflector (from radial build geometry) and the 0.6 exponent captures
thermal intensity (higher power → better cooling, thicker walls,
higher-grade materials per unit volume).

Aneutronic machines (blanket_form `none`, the p-B11/D-He3 fuel
normalization): there is no blanket structure, but the plasma-facing wall is
real hardware absorbing the photon/particle surface heat flux, so it is
priced as surface hardware:

    C220101 = firstwall_area × fw_unit_cost[fw_class]

with two discrete hardware classes, mirroring the fact that qualified wall
products exist as actively-cooled panels and as divertor-grade high-heat-flux
components, with nothing in between (`fw_class` is an explicit design input,
symmetric with `coil_material`):

| fw_class | NOAK unit cost | class limit | anchor |
|---|---:|---:|---|
| panel | 0.35 M$/m² | 2 MW/m² steady | ITER first-wall panel procurement: about 600 m² at order EUR 300-400M FOAK (about $0.6M/m²); NOAK 0.35 from panel-count repetition. Normal-heat-flux panel qualification class. |
| hhf | 2.5 M$/m² | 10 MW/m² steady | ITER divertor procurement: about 150-200 m² of W-monoblock plasma-facing units at order EUR 500-700M FOAK (about $3-3.5M/m²); NOAK 2.5. Qualified at 10 MW/m² steady state (Pitts et al. 2019, Nucl. Mater. Energy 20, 100696). |
| beam_dump | 1.0 M$/m² | 20 MW/m² through-wall | Bare-CuCrZr swirl-tube/hypervapotron panel wall (ITER NBI calorimeter/RID class, 20 MW/m² through-wall for hour-class pulses, Hemsworth et al. 2017; MAST-U hypervapotron test record 20 MW/m²). No public procurement value (F4E discloses supplier and scope, never award value), so this is a mass build-up, not a procurement anchor: about 0.9 t/m² fabricated CuCrZr (MITICA ERID 14 t / RID 15.3 m²) at qualified-copper rates gives 0.3-0.5 M$/m², doubled for manifolds/integration/QA and stated conservatively at 1.0 — a NOAK engineering judgment in the same evidence class as the capacitor $/J target. |

A cross-check the two anchors provide: per megawatt of absorbed capability
the classes land within a factor of about 1.5 of each other ($0.3-0.6M/MW
panel, about $0.35M/MW divertor), i.e. the $/m² rises sub-linearly with the
flux class. Pricing area at the class unit cost, with the class chosen
explicitly, therefore neither rewards nor punishes a high wall-flux design
beyond what the two procurement points support. `model.forward` audits
`q_surface_max` against the declared class's qualification limit and warns
on the concrete path if the cap exceeds it. C220101 is a
`replaceable_accounts` member, so the wall re-buys on the core-lifetime
replacement cycle (CAS72) under either branch. The DHe3/pB11 rows of the
fuel-keyed unit-cost table below apply only when an aneutronic fuel is run
with an explicit blanket-form override; at the normalized `none` form the
surface-priced branch above governs.

### Unit costs (M$/m³)

These unit costs are **structure only**: RAFM steel structure, tungsten
first-wall armor, and fabrication. The breeder and neutron-multiplier
material inventory (PbLi, Li, FLiBe, Be, ceramic) is priced once, separately,
in CAS27 as a volume-based mass build-up. It must not be baked into these unit
costs, or breeding blankets would be charged for the fill twice.

| Fuel | Unit cost | Rationale |
|------|----------:|-----------|
| DT | 0.35 | Breeding-blanket structure (RAFM steel structure + FW W armor + fabrication). TBR > 1.05 required, but the breeder/multiplier fill it holds is costed in CAS27. Complex assembly: HIPed joints, cooling channels, tritium barrier coatings. |
| DD | 0.30 | Energy-capture blanket (no breeding). RAFM steel + coolant channels. Slightly below DT (no W-armor/tritium-barrier premium; no breeder, no multiplier — and no CAS27 fill). |
| DHe3 | 0.08 | Minimal blanket. ~5% neutron fraction → thin shielding layer. Simple steel structure. |
| pB11 | 0.05 | X-ray shielding only. Thin metallic liner, conventional materials. |

### Validation

For a DT tokamak blanket of ~640 m³ (first wall + blanket + reflector) at
2535 MW thermal, structure only:
- Structural steel: ~1,150–1,600 t RAFM (FW panels + blanket channel walls and
  back plate + reflector block, ~10–20% of blanket volume plus a mostly-steel
  reflector); take ~1,300 t.
- W first-wall armor: ~4 mm over the ~430 m² first wall ≈ 33 t.
- Fabricated nuclear-grade cost: RAFM ~$30–50/kg raw → ~$120–150/kg fabricated
  (3–5× manufacturing/QA markup); W tiles ~$300–600/kg fabricated.
- ~1,300 t × ~$130/kg + 33 t × $400/kg ≈ $170–230M → 0.26–0.35 M$/m³.
  We take 0.35, the top of the steel build-up and consistent with the DD
  structure cost (0.30) plus a modest DT W-armor/tritium-barrier premium.

The old 0.60 anchored a ~3,500 t "structure + breeder + multiplier" mass at
9.4 t/m³ (the PbLi *fill* density), i.e. it folded the PbLi inventory into the
structure account — which CAS27 then charged again. 0.35 removes that fill mass.

ITER comparison: 440 blanket/shield modules, ~2,000 tonnes total (blanket +
shield combined). ITER is FOAK with bespoke international procurement; NOAK
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

The conductor quantity (ampere-meters) depends on device topology:

    Toroidal (tokamak, stellarator):
        total_kAm = G × B × R0 × r_coil / (μ₀ × 1000)
    Linear / loop (mirror, FRC, dipole, pulsed):
        total_kAm = G × B × r_coil² / (μ₀ × 1000)

    C220103 = total_kAm × $/kAm × markup / 1e6

The toroidal form is **bilinear**, not the square of one radius: the toroidal
field needs ampere-turns ~ B·R0 (Ampère's law around the torus) and the
conductor length per turn ~ the coil bore r_coil. The earlier model used a
single R² calibrated at SPARC's major radius (1.85 m); that made coil cost
grow as R0² and produced unphysically large costs for big machines (a stellarator
fed its major radius cost an order of magnitude too much). Real toroidal
conductor grows ~linearly in R0. For a true solenoid/ring loop (linear devices),
B = μ₀NI/(2R) with length 2πR gives R², which is correct, so those keep the
r² form.

where:
- **G** = geometry factor (tokamak: 4π², stellarator: 4π²×path_factor,
  mirror: see two-class model below, FRC: n_coils×4π)
- **B** = field at the loop center / on axis (NOT peak-on-conductor). Every
  concept that stores a plasma/design field `B` derives the coil-cost field
  from it at the point of consumption (`model._coil_center_field`): identity
  for tokamak, stellarator, mirror (central cell), and steady FRC;
  `coil_field_ratio × B` for dipole, polywell, and orbitron, whose coil-center
  field differs geometrically from the plasma-region field. Only the pulsed
  family (no stored plasma B) declares an explicit `b_center`. No concept
  stores two absolute fields that can drift apart.
- **R0** = major radius (toroidal devices); **r_coil** = coil-bore radius =
  `vessel_or` from the radial build (the TF/modular coils sit just outside the
  vessel). r_coil replaces the old "effective coil radius" calibration knob.
- **n_coils** = number of independent solenoid coils (FRC/other linear devices;
  not used for mirror, which uses the two-class model below)
- **$/kAm** = conductor cost per kilo-amp-meter
- **markup** = manufacturing complexity multiplier

#### Mirror two-class coil model

The mirror coil system is split into two physically distinct populations with
distinct bore derivations:

**Class 1: central-cell solenoids (large-bore, low-field):**

    r_bore_central = vessel_or + coil_standoff

The central solenoids sit outside the full radial build; their bore is the vessel
outer radius plus an assembly gap. At YAML defaults the radial build stacks to:

    plasma_t + vacuum_t + firstwall_t + blanket_t + reflector_t
        + ht_shield_t + structure_t + gap1_t + vessel_t
    = 1.5 + 0.10 + 0.05 + 0.80 + 0.20 + 0.20 + 0.15 + 0.10 + 0.10 = 3.20 m
    r_bore_central = 3.20 + 0.10 (coil_standoff) = 3.30 m

This is the physically correct coil radius for a tandem-mirror central cell:
WITAMIR-I central-cell solenoids are about 4 m radius for a comparable build.

    n_central = chamber_length / coil_spacing   (continuous; no rounding)
    G_central = n_central × 4π
    kAm_central = G_central × b_center × r_bore_central² / (μ₀ × 1000)

with `b_center = B`, the central-cell on-axis field: the central solenoid is
priced at the field it actually produces on axis (3 T at YAML defaults),
consistent with the plug's `R_m × B` throat-field basis. No second absolute
field is stored anywhere in the concept configuration.

`n_central` is a continuous costing aggregate (not a physical integer count); the
solenoid ensemble is treated as a distributed winding whose total conductor scales
linearly with machine length. YAML default: `coil_spacing: 5.0` m (Realta-class
solenoid pitch for a tandem-mirror central cell). Because `r_bore_central` is
derived from the radial build, coil cost now responds to `blanket_t` and other
build thicknesses.

**Class 2: end-plug HTS coils (small-bore, high-field):**

The plug coils sit at the throat where the machine necks down; flux conservation
gives the throat plasma radius:

    a_throat = plasma_t / sqrt(R_m)   (flux conservation)
    r_bore_plug = a_throat + plug_standoff

The plug bore is smaller than the central bore because the blanket does not extend
to the throat. At YAML defaults (plasma_t=1.5, R_m=10, plug_standoff=0.30):

    a_throat = 1.5 / sqrt(10) = 0.4743 m
    r_bore_plug = 0.4743 + 0.30 = 0.7743 m

`plug_standoff` covers the vacuum gap, throat structure, and cryostat at the mirror
throat. WHAM-class HTS plug coils have winding-pack bores well under 1 m at 17 T,
consistent with this value.

    b_plug = R_m × B          (throat field)
    G_plug = n_plug_coils × 4π
    kAm_plug = G_plug × b_plug × r_bore_plug² / (μ₀ × 1000)

`n_plug_coils` is the count of discrete end-plug coils (default 4: 2 per end,
HAMMIR Hammer-evolution class). Field at the plug throat: `b_plug = R_m × B`
where `R_m` is the mirror ratio and `B` is the central-cell field.

The bore ratio (0.77/3.30 about 0.23) means plug kAm is about 5% of central kAm
per coil at equal field; the high plug field (30 T vs 3 T central) partly offsets
this, giving a plug conductor fraction of about 36% of the total at the YAML
default machine (L = 20 m; the central share grows linearly with length).

**Total (per-class markups; see the manufacturing markup table):**

    C220103 = kAm_central × $/kAm × markup_central
            + kAm_plug × $/kAm × markup_plug

### Coil-bore radius from the radial build

`r_coil` is taken as `vessel_or` (the outer radius of the vacuum vessel
in the radial build), because the superconducting magnets sit outboard of
the entire plasma/blanket/shield/vessel stack. This is confirmed by the
published radial builds:

- **Tokamak (ARC, Sorbom et al. 2015, Fig. 2):** inboard order from the
  plasma inward is plasma → SOL → vacuum vessel → blanket → blanket tank →
  thermal shield → neutron shield → vacuum gap → TF coil → CS. The TF coil
  is on the far side of the vessel + blanket + shield; the stated inboard
  plasma-to-coil standoff is Δb = 0.85 m. (ARC is a liquid-immersion design,
  so its thin vessel sits inside the FLiBe blanket, but the magnet is still
  outboard of everything.)
- **Stellarator (ARIES-CS, Najmabadi et al. 2008):** 1.5–2 m between the
  plasma and the middle of the coil winding pack; 1.79 m for a regular
  breeding module (1.31 m with an optimized WC shield). The model's default
  stellarator build (vessel_or − plasma_t ≈ 1.7 m) matches this.
- **Mirror (tandem-mirror reactor studies):** "the coil radius is the sum
  of the plasma radius, blanket and shield thickness, and assembly gaps,"
  i.e. exactly vessel_or; WITAMIR-I central-cell solenoids are ~4 m radius.

Deriving `r_coil` from the radial build (rather than treating it as a free
calibration knob) ties the magnet cost to the same geometry that sets the
blanket and shield volumes. NOTE: the model places the LT shield outboard
of the coil, whereas ARIES-RS places it between vessel and coil; this does
not affect `r_coil = vessel_or`.

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
| Tokamak | 3.0× | TF + CS + PF coil systems. Complex D-shaped winding, insulation, quench protection, structural casing, cryostat integration. Derived from the ARC magnet bill of materials at qualified-fabrication rates (see "Tokamak markup from the ARC magnet BOM" below): tape at ARC's own price band (which brackets the $50/kA-m NOAK target), case steel at the ITER-FDR qualified-magnet-structure rate, giving BOM/conductor = 2.9 before winding content; the FDR TF class ratio (2.12, dominated by winding + radial plates + cases against much costlier Nb3Sn conductor) corroborates the fabrication content. At the SPARC-class reference (B=12T, R0=3.0m, r_coil=2.95m) the coil system lands at $500M under the bilinear model. |
| Stellarator | 5.7× | Non-planar 3D coil geometry. = 1.9× the tokamak markup, the NCSX modular-coil production cost overrun (90%; Neilson et al. 2010, PPPL-4455), the documented penalty for non-planar 3D coil fabrication (winding onto machined 3D forms, ±1.5mm tolerances, metrology). The longer 3D winding path is handled separately by the 2× path factor in G, so this 1.9× is fabrication complexity only. |
| Mirror (central cell) | 1.52× | PF-coil class from the ITER FDR magnet cost breakdown (see "Coil-class markups from the ITER FDR" below): large circular planar coils, the closest engineering analog to a tandem-mirror central-cell solenoid. |
| Mirror (end plug) | 1.81× | CS class from the same ITER FDR breakdown: high-field compact solenoid, the engineering analog of the HTS end-plug/choke coils. Stored as `mirror_plug_coil_markup`. |
| Pulsed FRC | 1.5× | Theta-pinch formation coils. Simple, repetitive geometry. |
| Theta pinch | 1.5× | Compression coils. Simple solenoid geometry. |
| Orbitron | 1.5× | Electrostatic confinement coils. |
| Polywell | 2.0× | Polyhedral magrid. Moderate 3D complexity. |
| MIF (MagLIF / Mag. target / Plasma jet) | — | $0: no plant-scale confinement magnets. MagLIF compresses with a Z-pinch liner; magnetized-target concepts compress mechanically/kinetically (the seed field lives in the pellet or CT injector); plasma-jet concepts merge plasma guns. The MFE conductor-scaling model does not apply. A small seed/guide coil, where one exists, is opt-in per concept via a `cost_overrides` entry on C220103. |
| IFE / Z-pinch / DPF | — | $0: no confinement magnets. |

### Validation

At the SPARC-class tokamak reference (B=12T, R0=3.0m, r_coil=2.95m, REBCO @
$50/kAm, 3.0× markup, bilinear model): $500M total coil system, consistent to
one significant figure with the legacy calibration's reference point (which
used the same B and $/kAm with R=1.85m and an 8× markup). This includes conductor, winding, insulation,
quench protection, structural casing, cryostat, power leads, instrumentation,
and testing. CFS SPARC used ~300 km of REBCO tape for a ~2m-class magnet; a full
tokamak power-plant coil set is ~5–10× larger. Because the model is now linear
in R0, a larger tokamak (e.g. ARC, R0=3.3m) scales its coil cost up
proportionally rather than as R0².

**Coil-class markups from the ITER FDR magnet cost breakdown:**

The per-class manufacturing markups (installed coil-class cost over finished
conductor) are taken from the ITER Final-Design-Report magnet cost estimate,
the only public source that breaks a full fusion magnet system down by coil
class AND separates conductor from winding and structure within each class:
Huguet, M., "The integrated design of the ITER magnets and their auxiliary
systems," 18th IAEA Fusion Energy Conference (IAEA-CSP-8/C, 2001), Fig. 4
("Magnet cost breakdown"; total 1,800 kIUA, 1 IUA = 1 US k$ 1989). The
pie-chart shares:

| item | share |
|---|---|
| TF conductor | 27.85% |
| TF winding | 10.75% |
| Radial plates | 9.1% |
| TF case and OIS | 11.4% |
| CS conductor | 6.95% |
| CS winding | 4.0% |
| CS structures | 1.65% |
| PF conductor | 11.4% |
| PF winding | 5.9% |
| Crowns and supports | 6.95% |
| Busbars and leads | 2.15% |

Per-class installed-cost / conductor ratios:

    TF (D-shaped, radial plates, cases):  (27.85+10.75+9.1+11.4)/27.85 = 2.12
    CS (high-field compact solenoid):     (6.95+4.0+1.65)/6.95        = 1.81
    PF (large circular planar coils):     (11.4+5.9)/11.4             = 1.52

The mirror central-cell solenoids take the PF-class ratio (large-bore circular
planar coils, simple winding) and the end plugs take the CS-class ratio
(high-field compact solenoid). Caveats carried knowingly: (a) these are
bottom-up FOAK engineering estimates for Nb3Sn/NbTi conductor, applied here
over REBCO tape at the $50/kA-m NOAK target, so the ratio, not the absolute
cost level, is what transfers; (b) the shared items (crowns/supports, busbars
and leads, about 9% of the system) are not allocated into either class ratio,
which makes both ratios slightly conservative-low; (c) the TF-class ratio
(2.12) sits below the model's tokamak markup (3.0) because the FDR
measured fabrication content against expensive Nb3Sn conductor economics,
while the tokamak markup applies over REBCO tape at the $50/kA-m NOAK
target; the tokamak markup is anchored independently on the ARC magnet
BOM (below), and the FDR TF class serves as the corroborating estimate of
winding/structure content, not as the anchor.

**Tokamak markup from the ARC magnet BOM:**

The tokamak markup is derived from the ARC design study's magnet bill of
materials (Sorbom et al. 2015, Table 11), repriced at qualified-fabrication
rates. ARC publishes 5,730 km of REBCO tape (about 2.6 million kA-m at the
about 450 A/tape its geometry implies; ARC's own tape cost of $103-206M is
$40-80/kA-m, bracketing this model's $50/kA-m NOAK target), 4,350 t of
SS316LN magnet case steel from its COMSOL stress model, a 358 t copper
former, and a 959 t tension ring. The case steel is load-bearing,
NDT-inspected, cryogenic-qualified structure, priced here at the ITER FDR's
own qualified-magnet-structure rate: the FDR's structures slices (TF case +
OIS 11.4%, crowns and supports 6.95%, CS structures 1.65% = 360 kIUA) over
its about 19,000 t of stainless magnet structures give $19/kg in 1989 $,
about $50/kg in 2025 $. Neither commodity welded plate (about $18/kg) nor
the heritage fusion-component multiplier (about $1,060/kg) is the right
basis for this material class.

    tape       2.6e6 kA-m x $50/kA-m            = $129M
    case steel 4,350 t x $50/kg (FDR qualified) = $218M
    Cu former  358 t x about $40/kg             = $14M
    tension ring 959 t x about $20/kg (simple)  = $19M
    BOM total                                   = $380M
    markup floor = 380 / 129                    = 2.9

Adding explicit winding/insulation/quench-protection labor content (the FDR
TF class carries about 0.4x of conductor in winding operations) closes the
remainder; the markup is stated as 3.0. Every link in this chain is citable
(ARC Table 11 masses and tape price; FDR structure pricing and winding
content), and none traces to a prior calibration of this model.

**Mirror two-class validation (YAML default machine, L=20 m):**

Central-cell bore:

    radial build: 1.5 + 0.10 + 0.05 + 0.80 + 0.20 + 0.20 + 0.15 + 0.10 + 0.10 = 3.20 m
    r_bore_central = 3.20 + 0.10 (coil_standoff) = 3.30 m

    n_central = 20 / 5 = 4
    G_central = 4 × 4π = 50.265
    kAm_central = 50.265 × 3 × 3.30² / (4π×10⁻⁷ × 1000) = 1,306,800 kAm
    conductor_central = 1,306,800 × 50 / 1e6 = 65.340 M$

Plug bore (flux conservation):

    a_throat = 1.5 / sqrt(10) = 0.47434 m
    r_bore_plug = 0.47434 + 0.30 (plug_standoff) = 0.77434 m

    b_plug = 10 × 3 = 30 T
    G_plug = 4 × 4π = 50.265
    kAm_plug = 50.265 × 30 × 0.77434² / (4π×10⁻⁷ × 1000) = 719,526 kAm
    conductor_plug = 719,526 × 50 / 1e6 = 35.976 M$

Account total:

    C220103 = 65.340 × 1.52 + 35.976 × 1.81 = 99.317 + 65.117 = 164.434 M$

Central conductor is 64% of the total at L = 20 m and grows linearly with
machine length; coil cost responds to blanket_t and every other radial-build
thickness that sets vessel_or, and to the central-cell field B (elasticity
formerly mis-attributed to the stored b_center knob).

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
| cu_markup | 3.5 | winding, insulation, cooling, jointing, test (ARIES class) |
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
| NBI | 7.06 | ARIES, calibrated to ITER NBI procurement | Ion source, accelerator, neutralizer, duct, cryo pumps, power supply |
| ICRF | 4.15 | ARIES | RF generators, transmission lines, antenna, matching network |
| ECRH | 5.00 | ARIES; ITER gyrotron cross-check | Gyrotrons (1 MW each), transmission waveguides, launchers |
| LHCD | 4.00 | ARIES | Klystrons, waveguide grills, power supply |

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

The heavy-ion and plasma-jet per-MJ figures reproduce their $/W reference points,
held constant off-reference (heavy-ion $12/W × 5 Hz = 60 M$/MJ; plasma-jet at
f_rep = 1 Hz).  The laser figure (205 M$/MJ) is grounded in published DPSSL NOAK
estimates; see the Laser IFE row and the double-counting note below.

#### Driver costs

| Concept | Basis | Coefficient | Hardware | Rationale |
|---------|-------|------------:|----------|-----------|
| Laser IFE (DPSSL) | $/MJ | 205 | Diode-pumped solid-state (optics + diodes) | Default architecture; optics + diode arrays. With the C220107 cap bank (around $5/J) the laser totals around $210/J, the aggressive end of the published DPSSL NOAK range ($210-700/J; diode roadmap toward $0.007/W packaged, where pulsed diode arrays are about 1/3 of drive-laser cost and need a 100x reduction — Haefner et al., LLNL IFE Workshop 2022). Selected by `laser_driver_type=dpssl`. |
| Laser IFE (KrF) | $/MJ | 40 | KrF excimer (e-beam pumped + gas) | NRL Electra / Xcimer heritage. 40 leans to the Xcimer/ASPEN large-aperture-optics claim ($10-20/J optical-on-target, with $5-10/J raw KrF light before SBS compression; an unproven long-pulse-scaling target — Galloway/Xcimer, LLNL IFE Workshop 2022, which itself cautions costs cannot yet be estimated); the NRL/Sethian engineering baseline is around $200/J (Sethian, Fusion Sci. Tech. 64, 2013). Range 20-200. `laser_driver_type=krf`. |
| Laser IFE (Nd:Glass) | $/MJ | 1000 | Flashlamp-pumped Nd:Glass (NIF-class) | NIF $3.5-4.2B / 1.1-1.9 MJ UV is around $2000/J facility, driver-only roughly half. Commercially marginal: see the flashlamp replacement note below. `laser_driver_type=nd_glass`. |
| Heavy ion | $/MJ | 60 | RF linac + storage rings | Accelerator capital scales with per-pulse beam energy and ring charge. Higher than laser due to ring infrastructure. |
| MagLIF | $/MJ (preheat only) | 205 | Laser preheat system | Main driver is the electrical Z-pinch (C220107). C220104 carries only the preheat laser, costed per joule of preheat pulse energy (`e_preheat_mj`); same DPSSL class as the IFE driver. Set `e_preheat_mj = 0` for no-preheat magnetized compression (e.g. Pacific Fusion). |
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

#### Laser-driver scheduled replacement (CAS72 O&M)

A rep-rated laser driver's replaceable subsystems wear at shot lifetimes
spanning sub-annual to multi-decade, so each is modeled as the level annual
cost of replacing it every `t = shot_lifetime / shots_per_year` years over the
plant life, summed via the shared geometric closed-form helper
(`levelized_replacement_cost`, the same machinery as core/DEC/cap-bank
replacement). The first set is capital (already in C220104); only replacements
beyond it are charged. Each subsystem cost is `replace_frac` times C220104. Shot
lifetimes are NOAK projections (LIFE / HiPER / NRL Electra), not demonstrated.

| Architecture | Subsystem | replace_frac | NOAK shot life | Demonstrated | Source |
|---|---|---:|---:|---:|---|
| DPSSL | Pump diodes | 0.50 | 1e10 | 1e8 (Mercury) | Orth/Bibeau DPSSL studies; Mercury (OSTI 1019071); Zuegel ARPA-E 2023 |
| DPSSL | KDP/DKDP crystals | 0.03 | 3e9 | NIF 14.3 J/cm2 spec | DKDP fatigue studies; NIF DKDP lifetime |
| DPSSL | Final optics (GIMM/transport/debris) | 0.05 | 3e8 | 1e5 (GIMM) | Latkowski fused-silica final optics (OSTI 20845924); GIMM (UCSD-CER-05-08; FST 56(1)) |
| KrF | Hibachi foil + windows | 0.04 | 3e8 | 1e4-1e5 | Sethian Electra (DTIC ADA480681) |
| KrF | E-beam diode + gas | 0.06 | 3e8 | engineering estimate | Sethian Electra |
| Nd:Glass | Xe flashlamps | 0.10 | 1e4 | O(1e3-1e4) | NIF flashlamp specs (lasers.llnl.gov; SPIE 8599) |

Three consequences fall out of the calibration, at a $1B driver, 10 Hz, 0.85
availability, 30-yr / 7% WACC plant:

- DPSSL diodes are capital, not O&M. At NOAK life (1e10 shots, around 37 yr) the
  replacement interval exceeds the plant life, so `n_rep = 0` and diodes
  contribute about $0. Diode shot life is the make-or-break sensitivity lever:
  at the demonstrated 1e8 shots they would wear sub-annually and dominate LCOE.
- DPSSL O&M is optics-dominated, landing around $48M/yr levelized, within the
  LIFE-projected band, driven by the final-optics line (around 1.1 yr interval).
- Nd:Glass is prohibitive. Flashlamps wear sub-annually (1e4 shots), so the
  replacement term explodes: the model surfaces NIF-class non-viability for
  rep-rated IFE, on top of its roughly 5x higher capital.

KrF and Nd:Glass subsystem cost shares are engineering estimates; no verified
component cost breakdown exists in the open literature.

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
`S_req = (Q_nbi + Q_fuel) / P_op`, costed at `pump_unit_cost[pump_basis]` per (m³/s) (fuel-keyed gas-phase species; see CAS220106_vacuum_pumping.md).
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
