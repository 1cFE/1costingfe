# CAS220109: Direct Energy Converter — Add-On for Linear Devices

**Date:** 2026-04-03
**Status:** Justified — subsystem build-up from Hoffman 1977 scaling laws,
cross-referenced with Moir/Barr 1973-74 and MARS 1984. Efficiency and
survivability assessment informed by detailed analyses of all four DEC
types (venetian blind, TWDEC, ICC, MHD).

---

## Scope

Account C220109 covers add-on direct energy converters for linear
fusion devices (mirrors, steady-state FRCs) where charged-particle
exhaust escapes along open field lines and can be decelerated to
recover kinetic energy as electricity, bypassing or supplementing
the thermal cycle.

**This account does not cover:**
- Inductive DEC (Helion-type pulsed FRC) — the driver hardware
  IS the DEC; costs belong in C220103/C220104/C220107
- MHD generators integrated into blankets — these are part of the
  thermal conversion pathway, not an add-on
- Tokamaks or stellarators — isotropic plasma loss, no directed
  exhaust beam

**Applicability:**
- Tokamak / stellarator: **not applicable** (no directed exhaust)
- Mirror / FRC (any fuel): **active when f_dec > 0** — cost scales
  with DEC electric output (p_dee)
  - DHe3, pB11: DEC is the primary power path (majority charged
    energy fraction)
  - DT, DD: physically possible, but not economic at demonstrated
    DEC efficiencies (see "Economics by fuel" below)

The model does not gate C220109 on fuel. Whether to install a DEC
on a DT mirror is an economic judgment, not a physics constraint.
The cost model computes the answer either way.

---

## Physics of add-on DEC for linear devices

In a mirror machine or steady-state FRC, charged particles escape
axially through the magnetic mirrors into end tanks. The escaping
ion distribution is broad in energy (100-200 keV bulk with tails to
MeV for fusion products) and contains multiple species (fuel ions,
fusion products, impurities).

Three distinct DEC mechanisms have been proposed to capture this
directed exhaust. They differ in physics but share the same dominant
cost drivers (vacuum containment, cryogenic pumping, power
conditioning).

### Venetian blind (electrostatic)

Angled metal ribbon grids at successively higher retarding DC
potentials. Ions penetrate until they lack energy to reach the next
stage, reverse trajectory, and are collected on high-potential
electrodes. This passively sorts ions by energy with no active
electronics.

- **TRL 4-5** (highest of any DEC)
- Demonstrated 48% on real TMX mirror plasma (single-stage),
  65% in lab (2-stage), 86.5% on monoenergetic electron beam
  (22-stage)
- Theoretical maximum ~70% (Moir & Barr 1973)
- Detailed engineering design and cost data (Hoffman 1977,
  Barr 1974)
- One active development program (Realta Fusion, patented
  axisymmetric ferromagnetic variant)
- Passive DC collection — no single-point failures, graceful
  degradation

Primary failure modes: sputtering erosion of ribbon geometry,
helium blistering compromises voltage holding, secondary electron
emission, charge-exchange neutrals bypass the converter.

### Traveling wave (electromagnetic)

Reverse linear accelerator: an RF modulator bunches the ion beam,
then a series of grid electrodes with external RLC circuits extract
kinetic energy as RF AC power via a traveling deceleration field.
Self-excited — the beam drives the oscillation.

- **TRL 3-4**
- Demonstrated 58-60% on monoenergetic lab beams (Takeno 2008-09)
- Self-excitation confirmed on GAMMA 10 tandem mirror plasma,
  but broad energy spread degraded bunching quality
- No engineering design, no cost data, single research group
  (NIFS/Kobe), apparently dormant since ~2019
- Active RF circuitry (27+ RLC circuits) — fundamentally less
  reliable than passive venetian blind
- Handles higher ion energies than venetian blind (>1 MeV)
  via distributed RF deceleration

Primary failure modes: grid collision (~11% irreducible loss),
energy spread degrades bunching, RF circuit reliability (27+
impedance-matched circuits).

### Inverse cyclotron (electromagnetic)

Ions enter a magnetic cusp that separates electrons from ions.
Inside a hollow cylinder (~5 m long) with segmented electrodes,
an oscillating multipole field extracts rotational energy at the
ion cyclotron frequency (~5 MHz). Higher-energy ions orbit at
larger radii where the field is stronger, providing automatic
energy sorting.

- **TRL 2** (lowest of any named DEC)
- 90% efficiency claimed in patent filings — no supporting data,
  no experiments, no simulations in the open literature
- Single organization (TAE Technologies), which appears to have
  moved toward thermal conversion for their near-term Da Vinci
  device
- Requires active RF circuitry plus a superconducting magnet
  (0.6 T, 5 m bore)
- More compact than venetian blind

No experimental results exist for the ICC. For costing purposes,
it should not be used as a design basis.

### DEC type is not distinguished in the cost model

The three DEC types produce remarkably similar add-on cost
estimates (\$73-140M at ~400 MWe DEC electric output), because
the dominant costs
(vacuum containment, cryogenic pumping, power conditioning) are
shared infrastructure regardless of the conversion mechanism.
The actual converter elements (grids, electrodes, RF circuits)
constitute only 2-10% of total DEC cost.

Efficiency differences between DEC types are captured by the
`eta_de` parameter in the physics layer, not in C220109. The
cost model uses a single `dec_base` scaled by DEC electric
output.

---

## Economics by fuel

### DT and DD: not economic at demonstrated efficiencies

In D-T fusion, ~80% of energy leaves as 14.1 MeV neutrons, which
are invisible to any add-on DEC. The venetian blind only touches
the ~20% charged fraction:

| Conversion pathway | Net plant efficiency |
|---|---|
| Pure thermal (blanket + Rankine) | ~40% of P_fus |
| Hybrid (thermal + DEC at 48%) | 0.80 x 0.40 + 0.20 x 0.48 = 42% |
| Hybrid (thermal + DEC at 65%) | 0.80 x 0.40 + 0.20 x 0.65 = 45% |
| Pure thermal (blanket + sCO2 Brayton) | ~50% of P_fus |

At demonstrated DEC efficiencies (48%), the hybrid is worse than
a modern sCO2 Brayton cycle. At the theoretical venetian blind
maximum (65%), the hybrid gains ~5 percentage points over Rankine
but still does not beat sCO2 Brayton. Either way, the DEC roughly
doubles power conversion capital cost for marginal efficiency gain.

For DD (67% neutron fraction), the arithmetic is slightly better
but the conclusion is the same.

The model does not prevent the user from setting `f_dec > 0` for
DT or DD — it computes the cost and lets the LCOE speak for itself.

### DHe3 and pB11: DEC is the primary power path

For D-He3 (~60% charged fusion products) and p-B11 (~99%
charged), the DEC is the primary power conversion pathway. However,
the DEC only sees charged particles that reach the end tanks —
not radiation. Bremsstrahlung and synchrotron photons emitted by
the hot plasma hit the walls and must be captured thermally, just
like neutrons.

For p-B11, bremsstrahlung is a large fraction of total fusion
power because both Z_eff (boron is Z=5) and the required plasma
temperature (>100 keV for reasonable reactivity) are high. The
exact fraction depends on plasma conditions, but bremsstrahlung
alone can approach or exceed the fusion power at marginal
operating points (Rider 1995, Nevins & Swain 2000). Even at
optimistic operating points, the radiated fraction is large
enough to require a thermal plant of non-trivial size. The
turbine island is reduced but not eliminated.

---

## Demonstrated performance

| Configuration | Efficiency | DEC concept | Conditions | Source |
|---|---|---|---|---|
| 22-stage collector (lab) | 86.5% | Venetian blind | Monoenergetic e-beam | Moir UCRL-79015 |
| 2-stage venetian blind (lab) | 65% | Venetian blind | Ion beam, narrow spread | Barr et al. 1974 |
| 1-stage venetian blind | 48% | Venetian blind | Real mirror plasma (TMX) | Moir 1984 |
| 16-electrode simulator | 58-60% | Traveling wave | Monoenergetic ion beam | Takeno et al. 2008-09 |
| GAMMA 10 test | Self-excitation confirmed | Traveling wave | Real mirror plasma, broad spread | Takeno et al. 2011 |
| Venetian blind theoretical max | ~70% | Venetian blind | Ideal | Moir & Barr 1973 |

**Design point for costing:** `eta_de` is a user parameter
(mirror default 0.60). The cost model does not assume a specific
DEC type or efficiency — it uses whatever `eta_de` the user sets.

---

## Historical cost data

### Primary source: Hoffman 1977 (UCID-17560)

Most detailed DEC cost study available. Provides subsystem-level
scaling equations for both single-stage (SSDC) and 3-stage venetian
blind (VBDC) converters. All costs 1975 dollars, FOAK.

**3-stage VBDC handling 1112 MW mirror leakage (710 MWe output):**

| Subsystem | % of total | Cost (1975$) | Cost (2024$) |
|---|---|---|---|
| Vacuum tank | 37.1% | \$135M | \$783M |
| Cryo vacuum system | 37.5% | \$137M | \$795M |
| DC converter modules (grids) | 2.3% | \$8M | \$46M |
| DC power conditioning | 7.2% | \$26M | \$151M |
| Thermal panels | 8.2% | \$30M | \$174M |
| Bottoming plant | 7.7% | \$28M | \$162M |
| **Total** | **100%** | **\$364M** | **\$2,111M** |

Specific capital: \$418/kWe (1975\$) -> \$2,424/kWe (2024\$).
CPI escalation: 1975 CPI = 53.8, 2024 CPI = 312.3, ratio = 5.80x.

**Key insight:** Vacuum tank and cryo pumping dominate (75% of
total). The actual converter grids are only 2.3% of cost. This
is why all three DEC types produce similar cost estimates — they
share the same dominant infrastructure.

### Secondary: Moir & Barr 1974

Total DEC cost for 1000 MW handled: ~\$110M (1974\$), or \$110/kW
of power into the converter. Consistent with Hoffman range.

### MARS Study (1984)

MARS tandem mirror (2600 MW fusion, 1200 MWe net) used a gridless
DEC variant. DEC recovered 290 MWe. Total plant: \$2.9B (1983\$) =
~\$2,400/kWe. Barr & Moir (1980) described DEC as "a rather small
addition" to the tandem mirror since expander tanks and vacuum
already exist for plasma confinement.

---

## Integrated mirror machine: shared vs. DEC-specific costs

In a mirror machine, the magnetic expander and end tanks already
exist for plasma confinement. The DEC "add-on" cost is NOT the
full standalone system from Hoffman — it is only the incremental
hardware beyond what the mirror already requires.

### Shared with base mirror machine (NOT charged to C220109):
- Expander tank structure (exists for plasma exhaust handling)
- Expander/mirror coils (part of magnet system, C220103)
- Base vacuum system (required for plasma operation)

### DEC-specific add-on (charged to C220109):

| Subsystem | Scaling basis | Build-up (2024$, ~400 MWe DEC) |
|---|---|---|
| Grid/collector modules | Collector area | \$10-15M |
| DC-AC power conditioning | DEC electric output | \$35-50M |
| Heat collection system | Thermal load on grids | \$8-12M |
| Incremental vacuum (cryo) | DEC volume/gas load | \$10-20M |
| Incremental tank volume | Expansion ratio | \$15-25M |
| Neutron trap shielding | DT only | \$0-5M |
| DEC gate valve | Fixed | \$0.5M |
| **Total DEC add-on** | | **\$79-128M** |

### Build-up methodology

**Grid/collector modules (\$10-15M):**
Hoffman: \$4,000/m^2 (1975\$) = \$23,200/m^2 (2024\$). For a mirror
with ~300 MW charged particle throughput, expansion ratio ~100,
mirror exit area ~3 m^2, collector area ~300 m^2. Cost =
300 x \$23.2k = \$7M. Add 50% for support structure, alignment, and
electrical connections: \$10M. Three-stage system: x1.5 = \$15M.

**DC-AC power conditioning (\$35-50M):**
Hoffman: \$46/kWe (1975\$) = \$267/kWe (2024\$). Modern high-power
inverters (solar/wind industry) have driven costs down. NREL
utility-scale inverter benchmark (2024): \$30-50/kWe for >100 MW
systems. Fusion DEC operates at higher voltage (100-250 kV)
requiring specialized conversion, so use \$80-120/kWe. At 300-500
MWe DEC output: \$24-60M, central estimate \$40M.

**Heat collection system (\$8-12M):**
Helium-cooled thermal panels on grid supports. Similar technology
to divertor cooling but lower heat flux (grids are geometrically
transparent, so only ~10-20% of particle power deposits as heat).
Scale from divertor cost (\$60M/GWth) at ~1/6 the thermal load:
~\$10M.

**Incremental vacuum (\$10-20M):**
Cryo pumping for the larger DEC volume and charge-exchange gas
load. Hoffman: \$6,300/m^2 (1975\$) = \$36,500/m^2 (2024\$). Modern
cryopumps are commoditized (Edwards, Leybold). Use \$15,000/m^2
for industrial cryopanels at fusion scale. At ~1,000 m^2 pump
area: \$15M.

**Incremental tank volume (\$15-25M):**
The DEC needs a larger end tank than basic plasma exhaust handling.
Incremental steel: ~2,000-4,000 tonnes at \$5-8/kg fabricated and
installed. Cost: \$10-32M, central \$20M.

---

## Recommended cost model

### Scaling formula

    C220109 = dec_base * (p_dee / P_DEE_REF) ^ 0.7

where:
- `dec_base` = 100.0 M$ (total DEC add-on cost at reference output)
- `P_DEE_REF` = 400 MWe (reference DEC electric output)
- `p_dee` = DEC electric output in MW (from physics layer:
  `f_dec * eta_de * p_transport`)

C220109 is gated on `f_dec > 0`. No fuel gating — economic
viability is a user judgment, not a model constraint.

The 0.7 exponent reflects:
- Grid modules scale ~linearly with area (proportional to power)
- Power conditioning scales ~linearly with output
- Tank and vacuum scale sub-linearly (surface-to-volume)
- Consistent with other vendor-purchased power systems in CAS22

### Why a single dec_base, not per-fuel or per-DEC-type

**Not per-fuel:** The DEC hardware cost depends on the power it
handles (p_dee), not on which ions are in the beam. A venetian
blind collecting 400 MWe from D-He3 protons costs the same as
one collecting 400 MWe from D-T alphas — the grids, vacuum, and
power conditioning are the same. Fuel affects p_dee (via the
charged fraction and f_dec), and fuel affects grid lifetime (via
sputtering and neutron damage), but the capital cost per MWe of
DEC output is fuel-independent.

**Not per-DEC-type:** The three add-on DEC types produce cost
estimates in the range \$73-140M at ~400 MWe DEC electric output.
This overlap exists
because the dominant costs (vacuum: 37%, cryo: 38%, power
conditioning: 7%) are shared infrastructure. The converter
elements themselves (grids, electrodes, RF circuits) are 2-10% of
total cost. The cost difference between a venetian blind and a
TWDEC is smaller than the uncertainty on either estimate (+-40%).

Efficiency differences between DEC types are captured by `eta_de`,
which affects p_dee and therefore LCOE, without changing the
hardware cost model.

### DEC grid replacement (CAS72)

DEC grids degrade under charged particle bombardment. Unlike the
vacuum tank, cryopumps, and power conditioning (which are long-lived
industrial equipment), the grids are in the particle beam and suffer
progressive damage.

Grid replacement is modeled as a separate CAS72 term, independent
of blanket/divertor replacement:

    dec_grid_cost = 12.0 M$ at P_DEE_REF (replaceable fraction)
    annual_replace_dec = dec_grid_cost * (p_dee / P_DEE_REF) ^ 0.7
                         / dec_grid_lifetime(fuel)

**Grid lifetime (FPY) — HIGH UNCERTAINTY.** No reactor-scale data
exists for any DEC grid under sustained particle bombardment. These
are conservative estimates. Primary degradation is from charged
particle exhaust (sputtering erosion, helium blistering), with
neutron damage additive for DT/DD. Sensitivity range: 0.5x to 3x.

| Fuel | Lifetime (FPY) | Primary degradation |
|---|---|---|
| DT | 2.0 | Sputtering + 14.1 MeV neutron damage |
| DD | 3.0 | Sputtering + 2.45 MeV neutron damage |
| DHe3 | 4.0 | 14.7 MeV proton sputtering + He blistering |
| pB11 | 3.0 | 2.9 MeV alpha sputtering + severe He blistering (3 alpha/event) |

Note that DHe3 and pB11 grid lifetimes are short despite reduced
neutron damage, because the charged particle flux is the primary
degradation mechanism. For pB11, each fusion event produces three
~2.9 MeV alphas that embed in the grid material and cause helium
blistering. For DHe3, 14.7 MeV protons cause significant
sputtering despite their lighter mass.

### FOAK multiplier

Apply standard contingency (10%) for FOAK per CAS29 methodology.
No additional FOAK markup — DEC components are extensions of
existing industrial technology (vacuum vessels, cryopumps, power
electronics), not novel nuclear-grade systems.

---

## Survivability and failure modes

The cost estimate is for initial capital and periodic grid
replacement. The following failure modes degrade DEC performance
over time and inform the conservative grid lifetime estimates:

1. **Sputtering erosion** — ions at 100+ keV bombard grid surfaces,
   thinning and roughening ribbons. The venetian blind design
   intentionally maximizes oblique incidence on collection surfaces,
   which increases sputter yield. As ribbons thin and roughen,
   angular transmission properties degrade. Efficiency drops
   gradually and silently.

2. **Helium blistering** — alpha particles (He++) embed in grid
   material. Helium is insoluble in metals and accumulates at
   grain boundaries, forming pressurized bubbles that blister and
   flake the surface. Surface deterioration degrades voltage
   holding, causing arcing between stages. Progressive and
   irreversible. Particularly severe for pB11 (3 alphas per
   fusion event).

3. **Secondary electron emission** — energetic ions knock out
   secondary electrons from grid surfaces. These electrons are
   accelerated by the same electric fields meant for ions, flowing
   in the wrong direction and consuming power. Worsens as surfaces
   roughen. Magnetically suppressible but adds complexity.

4. **Charge-exchange neutrals** — ions that capture an electron
   in the expander become fast neutrals invisible to electric
   fields. They deposit energy as unrecoverable heat and
   sputtering damage. Irreducible loss channel (~1-10% depending
   on background gas pressure).

5. **Neutron damage** (DT and DD only) — 14.1 MeV (DT) or 2.45
   MeV (DD) neutrons cause displacement damage, transmutation,
   embrittlement, and activation. The converter becomes radioactive,
   complicating maintenance. Greatly reduced for DHe3 (~5% neutron
   fraction) and absent for pB11.

6. **Heat removal from thin grids** — grids must be thin for
   geometric transparency but absorb energy from collected ions,
   secondary processes, neutrals, and radiation. Thin grids have
   minimal thermal mass and poor conduction paths. Thermal
   management at reactor power levels is a serious engineering
   challenge that has never been addressed at scale.

7. **Space charge / Debye length** (Rosenbluth objection) — at
   reactor-relevant particle densities, the Debye length becomes
   very short. Maintaining quasi-neutrality while imposing strong
   electric fields across distances comparable to the Debye length
   is "very challenging in practice" (Rosenbluth). This problem
   worsens at higher densities (higher power).

For DHe3 and pB11, failure modes 4, 5 are reduced (lower neutron
flux, less charge exchange from cleaner plasma). However, mode 2
(He blistering) is worse for pB11 due to triple-alpha production.

---

## Comparison with pyFECONs

pyFECONs (cas220109_direct_energy_converter.py) lists subsystem
costs from Post 1970 totaling \$447M with a power x flux scaling
formula. This is not used because:

1. C220109 was set to \$0 in pyFECONs with a TODO noting the
   discrepancy. The \$447M total was never active.
2. The \$447M appears to be a standalone DEC system (including full
   expander tank and coils), not a mirror add-on. For an integrated
   mirror, shared infrastructure must not be double-counted.
3. The Post 1970 values predate the more detailed Hoffman 1977
   analysis and lack clear year-dollar basis.
4. The scaling formula `cost x system_power x (1/sqrt(flux_limit))^3`
   has no documented derivation and produces unreasonable values at
   typical parameters.

---

## Sensitivity and uncertainty

| Parameter | Range | Impact on C220109 |
|---|---|---|
| DEC efficiency (eta_de) | 48-65% | Higher eta -> more MWe recovered -> larger p_dee -> higher C220109, but better LCOE |
| Expansion ratio | 25-400 | Higher ER -> larger tank/vacuum cost, but lower heat flux on grids |
| Charge-exchange loss fraction | 1-10% | Higher tolerance -> smaller cryo system -> 27-46% cost reduction (Hoffman) |
| Grid material (Cu vs Mo vs W) | - | Affects sputtering lifetime; W most durable but hardest to fabricate |
| dec_grid_lifetime | 0.5x-3x base | Dominates LCOE sensitivity for high-f_dec plants |

**Overall uncertainty:** +-40% on base costs. This is appropriate
for TRL 4-5 technology that has never been built at reactor scale.
The uncertainty is comparable to other fusion-specific components
(blanket, divertor) at similar maturity.

The dec_grid_lifetime parameters carry the highest uncertainty of
any input in this account. If sensitivity analysis shows grid
lifetime dominating LCOE for a particular design, that is
physically meaningful — it means DEC grid survivability is a
make-or-break R&D question for that concept.

---

## References

- Hoffman, M.A., "Electrostatic Direct Energy Converter Performance
  and Cost Scaling Laws," UCID-17560, Lawrence Livermore Laboratory,
  August 1977. [OSTI 7218298](https://www.osti.gov/biblio/7218298)
- Moir, R.W. & Barr, W.L., "Venetian-blind direct energy converter
  for fusion reactors," Nuclear Fusion 13, 35-45, 1973.
  [OSTI 4563116](https://www.osti.gov/biblio/4563116)
- Barr, W.L. et al., "A Preliminary Engineering Design of a Venetian
  Blind Direct Energy Converter for Fusion Reactors," IEEE Trans.
  Plasma Sci. PS-2(2), 71-92, 1974.
  [OSTI 4232519](https://www.osti.gov/biblio/4232519)
- Moir, R.W., "Direct Conversion of Fusion Energy" (review),
  UCRL-79015, Lawrence Livermore Laboratory.
  [OSTI 7341986](https://www.osti.gov/biblio/7341986)
- Barr, W.L. & Moir, R.W., "Technology of direct conversion for
  mirror reactor end-loss plasma," 1980.
  [OSTI 6894579](https://www.osti.gov/biblio/6894579)
- Logan, B.G. et al., "Mirror Advanced Reactor Study (MARS),"
  UCRL-53480, July 1984.
  [OSTI 5981974](https://www.osti.gov/biblio/5981974)
- Post, R.F., "Mirror systems: fuel cycles, loss reduction and energy
  recovery," in Nuclear Fusion Reactors, pp. 99-111, Thomas Telford
  Publishing, 1970.
- Hoffman, M.A. & Hamilton, G.W., "Direct Energy Conversion Cost
  Estimates," J. Energy 2(5), 293, 1978.
- Momota, H. et al., "Conceptual design of the D-3He reactor
  ARTEMIS," Fusion Technology 21, 2307-2323, 1992.
- Takeno, H. et al., "Performance analysis of small-scale experimental
  facility of TWDEC," Energy Conversion and Management 49, 2008.
- Takeno, H. et al., "Simulation Experiments of TWDEC on GAMMA 10
  Tandem Mirror," Fusion Science and Technology 59(1T), 2011.
- US Patent 6,850,011 B2 (TAE Technologies), "Controlled fusion in a
  field reversed configuration and direct energy conversion."
- Realta Fusion, "Axisymmetric ferromagnetic venetian blinds,"
  US Patent 12,166,398 B2, 2025.
- Woodruff, S., "A Costing Framework for Fusion Power Plants,"
  arXiv:2601.21724, January 2026.
- NREL, "Utility-Scale Solar Photovoltaics — 2024 Cost Benchmark."
