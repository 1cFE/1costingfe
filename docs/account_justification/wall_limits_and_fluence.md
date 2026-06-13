# First-Wall Limits and Neutron Fluence Lifetime

**Date:** 2026-06-13
**Status:** Justified — sources recorded, values proposed for the fluence-based CAS72 basis

## Purpose

This document backs two related modeling choices in the mirror wall-loading
and radial-build workline:

1. **Fluence-based core lifetime (CAS72).** The fixed per-fuel core-lifetime
   constants (DT 5 full power years, DD 10, D-He3 30, p-B11 50) are replaced
   by per-fuel neutron fluence limits `Phi_max` [MW yr / m^2], from which the
   scheduled-replacement lifetime is computed as `Phi_max / q_n`, where `q_n`
   is the neutron wall loading the machine actually runs at. This makes a
   machine that pushes its wall loading replace its first wall and blanket
   more often, which the fixed constants did not capture. The change applies
   to all steady-state MFE concepts at once (settled decision d1). This
   document supplies the `Phi_max` values and the citations behind them.
2. **Surface heat-flux cap (`q_surface_max`).** The mirror sizing solve caps
   the steady-state photon-plus-radial-transport heat flux on the lateral
   first wall. This document supplies the 1.0 MW/m^2 default and its basis.

No cost value here is taken from or calibrated against any cost-modeling tool.
All numbers come from peer-reviewed papers, laboratory design reports, and
fusion power-plant design studies.

A `$` in this document is a plain dollar sign; there are no costs computed
here.

## 1. Neutron first-wall/blanket fluence limit

### The structural dpa limit and its fluence equivalent

The end-of-life limit for a reduced-activation ferritic/martensitic (RAFM)
steel first wall and blanket is set by displacement damage in the structural
steel. The widely used design limit is **200 dpa** for ferritic steel,
beyond which radiation-induced loss of fracture toughness and embrittlement
make the structure unacceptable.

The ARIES nuclear-assessment work states this limit and ties it directly to a
fluence:

- The ferritic-steel (FS) structure ends its service life at **200 dpa**,
  which for ARIES corresponds to **18 MW yr / m^2** of first-wall neutron
  fluence (El-Guebaly and the ARIES Team, "Nuclear performance assessment of
  ARIES-AT," and UWFDM-1108, "Nuclear Issues and Analysis for ARIES";
  Univ. Wisconsin Fusion Technology Institute). The same source notes that at
  a peak neutron wall loading of 5.3 MW/m^2 the resulting FS first-wall life
  is 3 FPY, consistent with 18 / 5.3 = 3.4 FPY.

This gives the controlling conversion for 14 MeV neutrons on steel:

    200 dpa / 18 MW yr/m^2 = 11.1 dpa per MW yr/m^2   (ARIES FS basis)

### Cross-checks across independent sources

- **ARIES-AT SiC structure.** The SiC/SiC structural option carries a 3%
  burnup limit, giving a first-wall/blanket life of about 4 FPY and an
  end-of-life fluence of **18.5 MW yr / m^2** (El-Guebaly, ARIES-AT
  neutronics; PPPL-5008, "The ARIES Advanced and Conservative Tokamak (ACT)
  Study"). The SiC limit is quoted only as a cross-check on the fluence
  window; the steel basis is the one adopted, because the cost model's
  blanket and shield accounts assume a tungsten-armored, steel-structured
  build (see CAS22_reactor_components.md).
- **EU-DEMO starter blanket.** A DEMO phase-I "starter blanket" is specified
  to withstand at least **2 MW yr / m^2** of fusion-neutron fluence (about 20
  dpa in the front-wall steel), an explicitly conservative first-generation
  target rather than an end-of-life structural limit (Boccaccini et al.,
  EU-DEMO breeding-blanket programme; the 2 MW yr/m^2 / 20 dpa figure recurs
  across the EUROFER97 qualification literature, e.g. the IOP Energy review
  "Irradiation damage concurrent challenges with RAFM and ODS steels for
  fusion reactor first-wall/blanket," doi:10.1088/2515-7655/ac6f7f). The 20
  dpa at 2 MW yr/m^2 ratio (10 dpa per MW yr/m^2) agrees with the ARIES FS
  conversion to within 10%.
- **316-SS rule of thumb.** An unprotected 316 stainless first wall sees
  about **10 dpa per FPY at 1 MW/m^2** neutron wall loading (FT/1-1Rb,
  "On the Potentiality of Using Ferritic/Martensitic Steels," IAEA), again
  consistent with the 10-11 dpa per MW yr/m^2 conversion.
- **EUROFER97 irradiation data.** EUROFER97 has been neutron-irradiated and
  characterized to 16.3 dpa (low-cycle fatigue) and tensile/Charpy/fracture-
  toughness data exist to 80 dpa (Gaganidze et al., "Assessment of neutron
  irradiation effects on RAFM steels," doi:10.1016/j.fusengdes.2012.04.052;
  and the EUROFER97 80-dpa irradiation campaign). These are the *measured*
  dose range; the 200 dpa structural limit is the *design* extrapolation, and
  the gap between them (about 80 dpa demonstrated vs 200 dpa assumed) is an
  honest open materials risk noted below.

The commonly cited RAFM-steel FW/blanket end-of-life fluence window is
therefore **15-20 MW yr / m^2**, and 18 MW yr/m^2 (the ARIES FS value) is
adopted as the DT reference.

### Advanced structural materials (caveat only)

SiC/SiC composites raise the burnup-limited fluence modestly (about 18.5
MW yr/m^2 at the 3% burnup limit above) and tolerate higher operating
temperatures, and ODS-EUROFER pushes the steel limit somewhat higher. These
are noted but **not adopted**: the cost model's blanket build is steel-based,
so the steel fluence limit is the correct constraint for the costed machine.
Adopting a SiC limit would also require re-pricing the blanket account, which
is out of scope here.

## 2. Per-fuel fluence limits

### What `Phi_max` is and is not a function of

`q_n` (neutron wall loading, MW/m^2) is already the neutron power per unit
first-wall area, so it carries the fuel's neutron fraction implicitly. The
per-fuel difference in `Phi_max` therefore comes from two things, neither of
which is the neutron *fraction*:

1. **Spectrum hardness.** A given MW yr/m^2 of 14.1 MeV DT neutrons does more
   displacement damage per unit fluence than the same MW yr/m^2 of 2.45 MeV
   DD neutrons, because the displacement cross-section and the recoil-energy
   spectrum are harder at 14 MeV. The fusion-materials literature is explicit
   that "a 14 MeV neutron can produce much more irradiation damage than
   lower-energy neutrons" (Konobeyev et al., iron NRT/arc displacement cross
   sections, doi:10.1016/j.nme.2017.07.006; "New evaluation of neutron-
   induced displacement damage cross section for EUROFER97,"
   doi:10.1016/j.nme.2022.101259). A softer spectrum at fixed MW yr/m^2 thus
   reaches the 200 dpa structural limit later, i.e. a *higher* `Phi_max`.
2. **Lifetime regime.** For the low-neutron advanced fuels the first wall and
   blanket are not fluence-limited at all on any realistic schedule; their
   replacement lifetime is set by surface and thermal cycling (the
   `q_surface` cap of section 5), and by the small neutron side-channels, not
   by structural dpa. For these fuels `Phi_max` is effectively a very large
   number and the plant-life cap binds first.

The model's own per-event neutron energy fractions (physics.py
`event_energies`) make this concrete: DT releases 14.06 of 17.58 MeV as
neutrons (about 80%); DD releases roughly a third; D-He3 releases only a few
percent (D-D side reactions only; about 5-6% of fusion power as 2.45 MeV
neutrons in helium-catalyzed D-D, less if tritium is bred out; corroborated
by the D-3He literature reporting "only about 5-10% of energy as fast
neutrons"); p-B11 is aneutronic to within a small `11B(alpha,n)` /
`11B(p,n)` side yield, essentially zero structural neutron load.

### Proposed values

The DT value is sourced directly (18 MW yr/m^2, ARIES FS). The remaining
values are set so the ladder, when divided by a representative per-fuel
neutron wall loading, reproduces the engineering judgment embedded in the old
5/10/30/50 FPY constants, while remaining consistent with the spectrum
argument above. This per-fuel split is a **modeling choice**, not a hard
measurement, and is flagged as such.

| Fuel | `fluence_limit_*` [MW yr/m^2] | Basis |
|------|------------------------------:|-------|
| DT | 18.0 | ARIES FS 200 dpa = 18 MW yr/m^2 (El-Guebaly; UWFDM-1108). Cross-checked by ARIES-AT SiC 18.5 and EU-DEMO 2 MW yr/m^2 / 20 dpa. |
| DD | 36.0 | 2x the DT limit: 2.45 MeV neutrons do roughly half the dpa per MW yr/m^2 of 14.1 MeV neutrons, so the steel reaches 200 dpa at about twice the fluence. Direction sourced (14 MeV >> 2.45 MeV dpa, Konobeyev et al.); the factor of 2 is a modeling choice within the qualitative cross-section ratio. |
| D-He3 | 108.0 | 6x the DT limit. Neutron load is the small D-D side channel (a few percent of fusion power, mostly 2.45 MeV); the steel almost never reaches its dpa limit, so the value is large and the plant-life cap dominates. The exact number is a modeling choice; only its ordering (>> DD) is physical. |
| p-B11 | 180.0 | 10x the DT limit. Aneutronic to within a small side yield; structurally the wall is never fluence-limited and surface/thermal cycling governs. Value chosen large; ordering is the physical content. |

### Honest statement on the per-fuel split

Only the DT value (18 MW yr/m^2) is anchored to a primary structural-limit
source. The DD 2x factor follows the sourced direction (softer DD spectrum
does less dpa) but its precise magnitude is a modeling choice. The D-He3 and
p-B11 values are deliberately large placeholders that encode "not fluence-
limited"; for those fuels the binding constraint is `q_surface` (section 5),
and any reasonable large `Phi_max` gives the same answer because the
plant-life cap clamps the lifetime first. These should be revisited if a
mirror result proves sensitive to them.

## 3. Consistency check against the old FPY ladder

The basis change is calibrated to the existing engineering judgment, not
arbitrary. Dividing each proposed `Phi_max` by the representative per-fuel
neutron wall loading that the old constants implied recovers the old ladder:

| Fuel | `Phi_max` [MW yr/m^2] | implied ref `q_n` [MW/m^2] | old lifetime [FPY] | `Phi_max` / `q_n` [FPY] |
|------|----------------------:|---------------------------:|-------------------:|------------------------:|
| DT | 18.0 | 3.6 | 5 | 5.0 |
| DD | 36.0 | 3.6 | 10 | 10.0 |
| D-He3 | 108.0 | 3.6 | 30 | 30.0 |
| p-B11 | 180.0 | 3.6 | 50 | 50.0 |

The DT reference wall loading that recovers exactly 5 FPY is

    q_n,ref = Phi_max,DT / 5 FPY = 18 / 5 = 3.6 MW/m^2

which lands squarely in the 3-4 MW/m^2 first-wall neutron-loading class for
DT power plants (ARIES designs run a peak NWL of 3-5 MW/m^2). The old DT
"5 FPY, about 20 dpa/yr" comment in `defaults.py` implies 18 MW yr/m^2 / 5 yr
= 3.6 MW yr/m^2/yr of fluence and, at 11 dpa per MW yr/m^2, about 40 dpa/yr;
the old "20 dpa/yr" comment was thus a softer assumption (closer to the
2 MW/m^2 NWL class). The new basis is internally consistent and traceable to
the 200 dpa structural limit, which the old constant was not.

**Whether the ladder is reproduced.** With these values the *ratios*
(1 : 2 : 6 : 10) preserve the old 5 : 10 : 30 : 50 ladder exactly when each
fuel runs at the same 3.6 MW/m^2 reference neutron loading. In practice the
fuels do not run at the same `q_n`: the advanced fuels have far lower neutron
loading, so their computed `Phi_max / q_n` lifetimes will be very long and
hit the plant-life cap (as intended). The DT and DD lifetimes will shift away
from the flat 5/10 FPY constants wherever the sized machine's actual `q_n`
differs from 3.6 MW/m^2; that shift is exactly the economic signal the change
introduces and is quantified in the implementation (the CAS72 before/after
table is appended there).

## 4. Surface heat-flux cap (`q_surface_max`)

The first wall takes two physically distinct loads. Neutrons pass through and
deposit volumetrically in the blanket (the fluence/lifetime problem above).
Photons (line and bremsstrahlung radiation) and cross-field charged-particle
transport deposit in the first microns of the wall surface and must be
removed in real time by the first-wall coolant. The surface cap governs the
second load.

### Sourced value

Large-area actively cooled first walls manage about **1 MW/m^2** of surface
heat flux in steady state:

- **EU-DEMO breeding blanket.** "The present DEMO breeding blanket design
  heat-load capability is limited to about 1 MW/m^2 for steady-state plasma
  loading," and DEMO scenarios are developed "to comply with the 1 MW/m^2
  heat-flux limit on the whole breeding-blanket first wall during the
  flat-top steady-state phases, including radiation and charged-particle
  loads" (Maviglia et al., "European DEMO first-wall shaping and limiters
  design," doi:10.1016/j.fusengdes.2020.111575; Arena et al., on DEMO
  first-wall protection). Helium-cooled first-wall variants are quoted at
  about 0.5 MW/m^2, and enhanced high-heat-flux first-wall options are studied
  to push beyond 1 MW/m^2 (Barrett et al., "Options for a high-heat-flux
  enabled helium-cooled first wall for DEMO,"
  doi:10.1016/j.fusengdes.2017.04.066). The 1 MW/m^2 figure is the design
  basis for the standard breeding blanket.
- **ARIES-AT first wall.** The actual *steady-state* surface heat flux on the
  ARIES-AT first wall is only about 0.26 MW/m^2 average and 0.34 MW/m^2 peak
  (El-Guebaly, ARIES-AT; PPPL-5008), because most of the exhaust power is
  radiated to, and removed at, the divertor. This confirms 1 MW/m^2 is
  conservative for the lateral first wall of a steady-state DT machine.

### Distinction from the divertor

The 1 MW/m^2 first-wall cap is not the divertor-target limit. Divertor
targets are small wetted-area, high-flux components designed for 10+ MW/m^2
(ARIES-ACT divertor peaks of about 13-15 MW/m^2; PPPL-5008). The lateral
first wall is a large-area, low-flux component, and 1 MW/m^2 is the correct
constraint for it.

### Application to the mirror

The 1.0 MW/m^2 default is conservative for the lateral first wall and is
applied to the mirror's cylindrical lateral wall, where the photon and radial
transport loads land. The mirror's open ends change the picture relative to a
tokamak: the axial end-loss power exits through the throats to the
expander/direct-energy-conversion plates and never touches the lateral wall,
so it is excluded from `q_surface`. For the advanced fuels, which radiate a
large fraction of fusion power as photons (D-He3 and p-B11 radiate the
majority of their power), this surface cap, not the neutron cap, is the
binding wall constraint.

## 5. Caveats and items to revisit

- **Geometry mismatch.** Every fluence and heat-flux number above is sourced
  from tokamak (ARIES, EU-DEMO) or generic-blanket studies. The mirror is a
  cylindrical, open-ended device. The *material* limits (dpa, surface flux)
  are material properties and carry over, but the *mapping* from plasma to
  wall (cylindrical lateral area, throat end-loss exclusion) is the model's,
  not the sources'.
- **Advanced-fuel spectra stretch the steel basis.** The 200 dpa / 18 MW yr/m^2
  anchor is a 14 MeV DT result. The D-He3 and p-B11 `Phi_max` values are
  large placeholders encoding "not fluence-limited"; their precise magnitude
  is unsourced and is a modeling choice. Their effect is governed by the
  plant-life cap and the surface cap, so the answer is insensitive to the
  exact number, but this should be confirmed if a mirror advanced-fuel result
  appears sensitive to `Phi_max`.
- **The per-fuel split is partly a modeling choice.** Only DT is anchored to a
  primary structural-limit source. The DD factor of 2 follows the sourced
  spectrum direction but is not a pinned cross-section ratio. Revisit with a
  dedicated displacement-cross-section calculation (ENDF/B-VIII.0 or TENDL)
  if DD lifetime becomes a cost driver.
- **Demonstrated vs assumed dose.** EUROFER97 is characterized to about 80 dpa;
  the 200 dpa structural limit is a design extrapolation. If the qualified
  limit settles lower, the DT `Phi_max` (and the whole ladder) scales down
  proportionally. This is the largest single materials uncertainty behind the
  fluence basis.

## Sources

- El-Guebaly, L. A. and the ARIES Team, "Nuclear performance assessment of
  ARIES-AT," ARIES-AT final report, UCSD/Univ. Wisconsin; FS 200 dpa = 18
  MW yr/m^2, SiC 3% burnup = 18.5 MW yr/m^2 at about 4 FPY, first-wall
  surface heat flux 0.26 (avg) / 0.34 (peak) MW/m^2.
- El-Guebaly, L. A., "Nuclear Issues and Analysis for ARIES," UWFDM-1108,
  Univ. Wisconsin Fusion Technology Institute; FS service life 200 dpa, FW
  life 3 FPY at 5.3 MW/m^2 peak NWL.
- Najmabadi, F. et al. (the ARIES Team), "The ARIES Advanced and Conservative
  Tokamak (ACT) Study," PPPL-5008, 2014; SiC limits, divertor peak heat flux
  13-15 MW/m^2.
- Boccaccini, L. V. et al. and the EU-DEMO breeding-blanket programme;
  EU-DEMO starter blanket 2 MW yr/m^2 (about 20 dpa front-wall steel).
- "Irradiation damage concurrent challenges with RAFM and ODS steels for
  fusion reactor first-wall/blanket," IOP Energy,
  doi:10.1088/2515-7655/ac6f7f; RAFM dpa context and starter-blanket figure.
- Gaganidze, E. et al., "Assessment of neutron irradiation effects on RAFM
  steels," Fusion Eng. Des., doi:10.1016/j.fusengdes.2012.04.052; EUROFER97
  irradiation to 16.3 dpa and 80 dpa data range.
- Konobeyev, A. Yu. et al., "Iron NRT- and arc-displacement cross sections
  and their covariances," Nucl. Mater. Energy,
  doi:10.1016/j.nme.2017.07.006; and "New evaluation of neutron-induced
  displacement damage cross section for EUROFER97," Nucl. Mater. Energy,
  doi:10.1016/j.nme.2022.101259; spectrum hardness, 14 MeV >> 2.45 MeV dpa.
- "On the Potentiality of Using Ferritic/Martensitic Steels" (FT/1-1Rb),
  IAEA; 316-SS about 10 dpa/FPY at 1 MW/m^2 NWL.
- Maviglia, F. et al., "European DEMO first-wall shaping and limiters design
  and analysis status," Fusion Eng. Des.,
  doi:10.1016/j.fusengdes.2020.111575; 1 MW/m^2 steady-state first-wall
  heat-flux limit.
- Barrett, T. R. et al., "Options for a high-heat-flux enabled helium-cooled
  first wall for DEMO," Fusion Eng. Des.,
  doi:10.1016/j.fusengdes.2017.04.066; helium-cooled FW about 0.5 MW/m^2,
  enhanced HHF options.
- Zinkle, S. J., "Materials challenges in nuclear energy," Acta Materialia
  61(3) (2013) 735-758; 50-200 dpa structural-damage requirement context for
  fusion DEMO.
