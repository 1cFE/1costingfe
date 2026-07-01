# Concept Power-Plant Scaling Axes

How each confinement concept reaches plant-scale electric output, and which
concepts the cost model sizes from a power target. Classifications are grounded
in published developer/company material; where a developer states a scaling
approach, it overrides general expectation.

## The four scaling axes

A concept reaches a plant target along one of four axes:

- **Volume**: grow a single device (major radius, length, field). The model
  sizes these by solving geometry from the power target (`size_from_power`).
- **n_mod**: replicate a fixed module. The model sizes these by solving an
  integer module count `n_mod = ceil(target / module_net_mwe)` (this work).
- **Rep-rate**: fire a single pulsed chamber faster (shots per second).
- **Target yield**: increase the energy released per shot.

n_mod is also available to every concept as a plain plant multiplicity (build N
identical plants), independent of the scaling axis. The classification below is
about the concept's NATURAL single-plant scaling lever.

## Classification

| Concept | Axis | Sized from power? | Basis |
|---|---|---|---|
| Tokamak | Volume | Yes (solve R0) | Single large device; standard tokamak scaling |
| Mirror | Volume | Yes (solve length) | Tandem central cell grows with length |
| Stellarator | Volume | Not yet (future) | Type One Infinity Two, Thea Helios, Proxima Stellaris are all single large devices grown by major radius |
| Dipole | Volume | Not yet (future) | OpenStar single levitated device; pressure scales as R^(20/3), rewarding one larger device |
| Polywell | Volume | Not yet (future) | EMC2/Bussard published scaling is single-device (power scales as R^7); no developer proposes replicated polywell modules |
| **Orbitron** | **n_mod** | **Yes (this work)** | Avalanche 5 kWe "power pack" cells, "grouped together however needed" to grid scale |
| **Dense plasma focus** | **n_mod** | **Yes (this work)** | LPP Fusion: plants larger than 5 MW are "formed by simply stacking" 5 MW units |
| **Staged Z-pinch** | **n_mod** | **Yes (this work)** | Zap Energy sheared-flow pinch: "future power plants will have multiple modules" of 50 MW (DOE-approved design). See naming note below |
| **Steady FRC** | **n_mod** | **Yes (this work)** | TAE markets the FRC as "modular, perfect for mass production," 50 MWe to 350-500 MWe by replication. Physical basis: an FRC is MHD/tilt-unstable above a size limit, so it cannot scale by single-device volume growth |
| Pulsed FRC | Rep-rate | Not yet (future) | Helion: a single about 50 MWe machine driven to about 1 Hz |
| Magnetized target | Rep-rate | Not yet (future) | General Fusion: fixed-size sphere at 1 Hz (commercial plant is about two units) |
| Plasma jet (PJMIF) | Rep-rate | Not yet (future) | HyperJet/LANL: inexpensive plasma guns at about 1 Hz |
| Theta pinch | Rep-rate | Not yet (future) | Historical (Scylla); dormant |
| Z-pinch (IFE) | Target yield | Not yet (future) | Wire-array / driver-target z-pinch; yield per shot (distinct from Zap, see below) |
| Laser IFE | Mixed | Not yet (future) | Xcimer scales target yield (about 1 GJ/shot, sub-1 Hz); Focused Energy and Marvel scale rep-rate (10 Hz) |
| MagLIF | Target yield | Not yet (future) | Sandia: higher driver current for higher yield per shot |
| Heavy-ion IFE | Rep-rate + shared driver | Not yet (future) | Canonical HIBALL/HIDIF: one driver at about 10 Hz feeding several chambers |

## Naming note: STAGED_ZPINCH is Zap (sheared-flow)

The model's `STAGED_ZPINCH` enum is Zap Energy's SHEARED-FLOW z-pinch, confirmed
by where the model costs Zap's hardware: the sheared-flow coaxial gun plus gas
injection driver (`defaults.py` `driver_staged_zpinch_per_mj`, `cas22.py` C220104
mirror branch), its electrode erosion replacement (`costs.py` CAS72), and the
account doc `CAS22_reactor_components.md`. The plain `ZPINCH` enum is the IFE
wire-array/target z-pinch (yield/rep-rate). The academic "staged Z-pinch"
(MIFTI, liner-on-target) is a third, distinct concept not separately modeled and
has no published plant architecture. The enum names are pre-existing and left
unchanged. (`pulsed_staged_zpinch.yaml` carries a stale parameter citation to
LANL staged-z-pinch work; its header intent, Zap sheared-flow, is authoritative.)

## n_mod sizing mechanism and module_net_mwe anchors

For an `N_MOD_SIZED_CONCEPTS` member, `size_from_power=True` solves
`n_mod = max(1, ceil(target_net_mwe / module_net_mwe))`. Each module runs at its
design power `module_net_mwe`; the realized plant net is `n_mod * module_net_mwe`,
which overshoots the requested target by less than one module (a fractional
module cannot be built). `n_mod` is a physical integer and cannot be pinned in
this mode. The solved count is returned on `ForwardResult.solved_n_mod`.

| Concept | module_net_mwe | Source |
|---|---|---|
| Orbitron | 0.005 (5 kWe) | Avalanche Energy power-pack cell |
| Dense plasma focus | 5.0 | LPP Fusion 5 MW unit (200 Hz) |
| Staged Z-pinch (Zap) | 50.0 | Zap Energy module, DOE-approved design (about 10 Hz) |
| Steady FRC | 50.0 | TAE-class utility module |

**Consistency caveat.** `module_net_mwe` is the per-module design net power. The
non-0D steady-state and pulsed paths cost a module at whatever per-module power
they are given (audit-style), so the per-module costs are self-consistent with
`module_net_mwe` by construction. The value must remain a sensible design point
for the concept's geometry and plasma descriptors; if those are revised, revisit
`module_net_mwe`.

## Rep-rate shot design points

`REP_RATE_SIZED_CONCEPTS` (PULSED_FRC, MAG_TARGET, PLASMA_JET, THETA_PINCH,
LASER_IFE) each carry a per-shot design point in their concept-default YAML:
`e_driver_mj` (driver energy per shot), `yield_per_shot_mj` (fusion energy
released per shot), and `max_f_rep` (maximum rep rate, Hz). No solver
consumes these yet; a future rep-rate sizing solver (a separate workline)
would use them to solve shots-per-second or yield-per-shot from a power
target, the same way `module_net_mwe` sizes `N_MOD_SIZED_CONCEPTS` today.

Compiled 2026-07-01. All plant-scale yield/driver figures for these
pre-commercial concepts are design-target/simulation values, not
experimentally achieved, except THETA_PINCH's historical driver energy.
Two values are explicitly ILLUSTRATIVE placeholders, not sourced/disclosed
figures, and are flagged as such below and in their YAML comments.

| Concept | e_driver_mj | yield_per_shot_mj | max_f_rep (Hz) | Source |
|---|---|---|---|---|
| MAG_TARGET | 755 | 780 | 1 | SOURCED (Krotez et al., SOFE 2023 E-267, Sankey diagram) |
| PLASMA_JET | 31.3 | 736 | 1 | SOURCED (Langendorf & Hsu 2017 "case2"; yield = 31.3 x gain 23.5) |
| LASER_IFE | 2.5 | 250 | 10 | SOURCED (Ditmire, Roth, et al. 2023; yield = 2.5 x design-target gain G100), corrected from an unsupported 16 Hz |
| PULSED_FRC | 50 | 101.4 | 1 | driver + rate SOURCED (Helion Polaris, IEEE Spectrum); yield ILLUSTRATIVE (see below); rate corrected from an unsupported 10 Hz |
| THETA_PINCH | 3.5 | 35 | 1 | driver SOURCED (Scylla IV, historical); yield + rate ILLUSTRATIVE (see below) |

**PULSED_FRC yield derivation (illustrative, not disclosed by Helion).**
Helion has published no fusion-yield figure for Polaris or the commercial
plant. The only public plant number is Helion's stated target of about
50 MWe net electric at about 1 Hz (IEEE Spectrum). `yield_per_shot_mj` is
pinned to 101.4 MJ: the value of fusion power `p_fus` (at `f_rep=1`, so
`yield_per_shot_mj = p_fus` in MJ) for which the model's `pulsed_dec_forward`
(the concept's `pulsed_conversion: inductive_dec` path) reproduces
`p_net = 50` MW at `e_driver_mj=50`, `f_rep=1`, using
`pulsed_pulsed_frc.yaml`'s own DEC parameters (`eta_pin=0.95`,
`eta_dec=0.85`, `f_pdv=0.80`, `f_rad=0.05`, `p_coils=0.5`) and `eta_th=0`,
the default for INDUCTIVE_DEC concepts (no thermal bottoming cycle). This
is a modeling placeholder that makes the concept model-consistent with
Helion's one disclosed plant number; it is not a physical fusion yield and
should not be cited as one.

**THETA_PINCH yield + rate derivation (arbitrary/illustrative).** THETA_PINCH
is a dormant concept (no modern developer); all numbers are 1958-65 Los
Alamos Scylla-series experiments. `e_driver_mj = 3.5` is the sourced
Scylla IV capacitor-bank energy. Scylla IV never achieved meaningful fusion
gain: only diagnostic-level neutron counts are documented ("billions of
D-D fusion reactions per pulse"), which back-of-envelope converts to about
1e-10 MJ/shot (gain about 1e-10), negligible and not a published joule
figure. No primary source gives a Scylla shot cadence. `yield_per_shot_mj`
is pinned to 35 (an illustrative gain G=10 placeholder on the 3.5 MJ bank)
and `max_f_rep` to 1 (illustrative; no cadence documented), purely so the
concept has finite, non-zero modeling inputs.

**MAG_TARGET driver_recovery_frac derivation (anchored calibration).**
`e_driver_mj = 755` MJ/shot is gas-piston/liner compression KINETIC energy,
not a single-pass electrical driver load: General Fusion's own design
mechanically recovers most of it (liner rebound / working-fluid recovery)
each cycle rather than re-generating it from the grid. The shared pulsed
forward's driver recirculation term, `p_driver / eta_pin`, assumes a
single-pass electrical driver and so overcharges MAG_TARGET's recirculating
power by treating 755 MJ/shot as if it were all wall-plug electricity at
`eta_pin=0.30` (about 2517 MW of recirc at `f_rep=1`, versus 780 MW of
fusion power) -- driver-dominated and net-negative at any rep rate. The
model's `driver_recovery_frac` parameter (a required forward argument, set
explicitly to 0.0, a no-op, in every other pulsed concept's YAML) scales
that term to `p_driver * (1 - driver_recovery_frac) / eta_pin`.
`pulsed_mag_target.yaml`'s `driver_recovery_frac = 0.8191` is pinned to the
value for which the model's `pulsed_thermal_forward`, at this concept's own
shot design point (`f_rep=1`, `e_driver_mj=755`, `yield_per_shot_mj=780`)
and RANKINE `eta_th=0.40`, reproduces the GF SOFE 2023 Sankey's net electric
of about 150 MWe (single-chamber net at `f_rep=1` from bisection against
the model: 149.9 MW). As a rough consistency check against the Sankey's own
figures (gross about 363 MW = 907 MW thermal x 0.40, net 150 MW implies
driver electrical recirc about 213 MW versus the un-recovered
755/0.30 about 2517 MW, i.e. recovery about 0.91): the two paths agree to
within the precision of reading values off the published diagram, since the
model's thermal pool (neutron + ash + driver + pump) differs from the
diagram's lumped 907 MW node. This is a calibration anchor to GF's single
disclosed plant number, not an independently measured recovery efficiency;
if GF discloses a mechanical-recovery efficiency directly, this value should
be revisited against it.

### Rep-rate shot design point sources

- **PULSED_FRC (Helion):**
  - Helion "Polaris" page (bank energy 50+ MJ): https://www.helionenergy.com/polaris
  - Helion blog, "Helion's fusion system is (basically) an RLC circuit": https://www.helionenergy.com/blog/helions-fusion-system-is-basically-an-rlc-circuit
  - Scientific American, "Helion Energy is building a fusion power plant...": https://www.scientificamerican.com/article/helion-energy-is-building-a-fusion-power-plant-can-its-technology-deliver/
  - IEEE Spectrum, "Welcome to Fusion City, USA" (commercial reactor about 1 pulse/s, 50 MW): https://spectrum.ieee.org/fusion
  - Wikipedia, "Helion Energy" (Trenta about 1 pulse/10 min; Polaris about 1 Hz): https://en.wikipedia.org/wiki/Helion_Energy
- **MAG_TARGET (General Fusion):**
  - Krotez, Segas, Khalzov, Suponitsky, "Conceptual Design of a Magnetized Target Fusion Power Plant," IEEE SOFE 2023 paper E-267: https://generalfusion.com/wp-content/uploads/2023/07/SOFE2023_E267_daymon-krotez_v2.pdf
  - General Fusion public pages: https://generalfusion.com/
- **PLASMA_JET (HyperJet/LANL PJMIF):**
  - Langendorf & Hsu, Phys. Plasmas 24, 032704 (2017), arXiv:1612.07368: https://arxiv.org/pdf/1612.07368
  - Thio & Witherspoon, "Plasma-Jet-Driven Magneto-Inertial Fusion, A progress report," Open Access Government (2018): https://www.openaccessgovernment.org/plasma-jet-driven-magneto-inertial-fusion-2/63480/
  - Thio et al., "Plasma-Jet-Driven Magneto-Inertial Fusion," Fusion Sci. Technol. 75(7) (2019), DOI: 10.1080/15361055.2019.1598736
- **THETA_PINCH (Scylla):**
  - Quinn, Little, Ribe, Sawyer, "Stability, Heating and End Loss of a 3.5-Megajoule Theta Pinch (Scylla IV)," LASL LA-DC-7039 / CONF-650935-2 (1965): https://www.osti.gov/biblio/4624642
  - Wikipedia, "Theta pinch" (Scylla history, neutron counts, temperatures): https://en.wikipedia.org/wiki/Theta_pinch
  - Scylla IV-P (2 MJ, context): https://www.osti.gov/biblio/4127003
- **LASER_IFE (Focused Energy / Marvel Fusion):**
  - Ditmire, Roth, et al., "Focused Energy, A New Approach Towards Inertial Fusion Energy," J. Fusion Energy 42:27 (2023): https://doi.org/10.1007/s10894-023-00363-x
  - Marvel Fusion path to IFE (10 Hz DPSSL), Innovation News Network: https://www.innovationnewsnetwork.com/marvel-fusions-path-to-laser-based-inertial-fusion-energy/65989/
  - Ruhl & Korn, arXiv:2202.03170 (Marvel driver/yield claims; disputed, not used for values here): https://arxiv.org/abs/2202.03170
  - Rebuttal, arXiv:2204.00269: https://arxiv.org/abs/2204.00269

## Future-sizing roadmap

- Volume sizing for STELLARATOR (solve R0/a, plus 3D coil markup), DIPOLE, and
  POLYWELL (single-device growth) would follow the tokamak/mirror pattern.
- Rep-rate and target-yield sizing for the pulsed/IFE/MIF concepts would solve
  shots-per-second or yield-per-shot from the power target using the shot
  design points above; this is a separate future workline.

## Sources

- Orbitron: Avalanche Energy, avalanchefusion.com/technology (cell grouping to
  megawatt scale); NextBigFuture (CEO Langtry, "hundreds of little cells").
- Dense plasma focus: LPP Fusion business development plan and FAQ, lppfusion.com
  ("stacking smaller units").
- Staged Z-pinch (Zap): Zap Energy press release Oct 2024 (Century, 50 MW module,
  "multiple modules"); Physics of Plasmas 30, 090603 (2023).
- Steady FRC: TAE Technologies (modular, mass-production framing); SEC merger
  filings (50 MWe to 350-500 MWe). FRC MHD/tilt size limit is standard FRC
  physics.
- Polywell (volume): EMC2 emc2fusion.com and Bussard scaling (power scales as
  R^7); arXiv:2508.06761.
- Stellarators (volume): Type One Energy Infinity Two (J. Plasma Phys. 2025);
  Thea Energy Helios (arXiv:2512.08027); Proxima Stellaris (Fusion Eng. Des. 214,
  114868, 2025).
- Dipole (volume): OpenStar Technologies (arXiv:2602.20564).
- Pulsed FRC: Helion Energy, helionenergy.com (pulsed approach, about 1 Hz;
  see "Rep-rate shot design points" below for the corrected rate and citation).
- Magnetized target: General Fusion, generalfusion.com/commercialization-path
  (1 Hz, two-unit plant).
- Laser IFE: Xcimer (DOE-approved design, about 1 GJ/shot, sub-1 Hz); Focused
  Energy and Marvel Fusion (10 Hz).
- Heavy-ion IFE: HIBALL and HIDIF studies (shared driver, several chambers).
