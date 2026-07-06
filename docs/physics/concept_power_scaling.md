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
| Theta pinch | Rep-rate | Not yet (future) | No current developer, but a fully-worked D-T reactor design point exists: the LANL Reference Theta-Pinch Reactor (RTPR), 0.1 Hz, 1800 MWe |
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
`e_driver_mj` (driver energy delivered per shot), `yield_per_shot_mj` (fusion
energy released per shot), and `max_f_rep` (maximum rep rate, Hz). The
`size_from_power` solver holds this cited shot fixed and solves `f_rep` from
the power target, bumping the chamber count `n_mod` when the target exceeds one
chamber's net-electric ceiling at `max_f_rep`.

All five design points are anchored to reactor-class sources (a reactor design
study or a developer's commercial-plant target), not breakeven experiments: a
Q about 1 science device cannot represent a NOAK reactor. Two concepts,
MAG_TARGET and THETA_PINCH, use the `recovered_compression` forward, whose
driver energy is largely recovered each cycle rather than consumed; for them
the delivered `e_driver_mj` decouples from the cap-bank store `e_store_mj`
(sizing C220107) and the net grid draw `e_recirc_mj` (sizing recirculation).
See "The recovered-compression path" below.

| Concept | e_driver_mj (delivered) | yield_per_shot_mj | max_f_rep (Hz) | Source |
|---|---|---|---|---|
| MAG_TARGET | 755 | 780 | 1 | SOURCED, reactor: GF "Design Point - 150 MWe" Sankey (Krotez et al., SOFE 2023 E-267). recovered_compression: e_store=173, e_recirc=195 |
| PLASMA_JET | 31.3 | 736 | 1 | SOURCED, reactor design study (Langendorf & Hsu 2017 "case2"; yield = 31.3 x gain 23.5) |
| LASER_IFE | 2.5 | 250 | 10 | SOURCED, reactor plant (Ditmire, Roth, et al. 2023; yield = 2.5 x design-target gain G100), corrected from an unsupported 16 Hz |
| PULSED_FRC | 50 | 101.4 | 1 | driver SOURCED (Helion Polaris); yield reactor-target-anchored to Helion's sourced 50 MWe/1 Hz plant (derived, see below); rate third-party 1 Hz (range 1-10 Hz) |
| THETA_PINCH | 1190 | 28800 | 0.1 | SOURCED, reactor: LANL RTPR (LA-5121-MS 1972). recovered_compression: e_store=110250, e_recirc=2076 |

**PULSED_FRC yield (reactor-target-anchored to Helion's sourced 50 MWe).**
Helion has published no per-shot fusion yield for Polaris or the commercial
plant. The one sourced plant number is Helion's commercial target of at least
50 MWe net (Helion/Microsoft PPA, 2023). `yield_per_shot_mj` is pinned to
101.4 MJ: the value of `p_fus` (at `f_rep=1`, so `yield_per_shot_mj = p_fus`
in MJ) for which the model's `pulsed_dec_forward` (the concept's
`inductive_dec` path) reproduces `p_net = 50` MW at `e_driver_mj=50`,
`f_rep=1`, using `pulsed_pulsed_frc.yaml`'s own DEC parameters (`eta_pin=0.95`,
`eta_dec=0.85`, `f_pdv=0.80`, `f_rad=0.05`, `p_coils=0.5`) and `eta_th=0`.
This is the same "anchored to a sourced plant number" character as
MAG_TARGET's recovery, not a disclosed physical yield; do not cite it as one.
Helion's `eta_pin=0.95` is itself the reactive (RLC-ringdown) energy-recovery
efficiency Helion has demonstrated even without plasma, so the DEC path already
charges only the about 5% driver loss and PULSED_FRC needs no explicit
store/consume split (see "The recovered-compression path"). `max_f_rep = 1` Hz
is third-party (Polaris/NRC filing, IEEE Spectrum), not a Helion spec; the
roadmap points from 1 Hz toward about 10 Hz, so the honest range is 1-10 Hz.

**THETA_PINCH design point (SOURCED, LANL RTPR reactor study).** THETA_PINCH
has no modern developer, but a fully-worked D-T reactor design point exists:
the Los Alamos Reference Theta-Pinch Reactor (RTPR), LA-5121-MS (1972). This
replaces the earlier Scylla-experiment placeholder (a Q about 1e-10 science
device, not a reactor). RTPR is a 350 m toroidal high-beta theta pinch (a
straight theta pinch closed into a ring to eliminate axial end losses -- NOT a
tokamak: there is no ohmic-driven plasma current). LA-5121-MS Table IV states
energies per metre; totalled x 350 m:

- `e_driver_mj = 1190` (delivered to the plasma, (W_p)i = 3.41 MJ/m); sets
  gain `q_sci = yield/e_driver` about 24 (RTPR quotes plasma Q_p = 24.7).
- `yield_per_shot_mj = 28800` (bare D-T fusion, 82.3 MJ/m).
- `max_f_rep = 0.1` Hz (100 ms burn / 10 s cycle).
- `e_store_mj = 110250` (gross switched compression-field energy W_BO =
  315 MJ/m) and `e_recirc_mj = 2076` (net consumed drive Wi = 5.93 MJ/m); see
  "The recovered-compression path".

RTPR reference point: 1800 MWe net, 3600 MWth, plant gain Q about 17.5. In the
model, at the catalog-standard RANKINE `eta_th = 0.40` (RTPR's own 58%
potassium-topping figure is an optimistic 1970s claim not adopted here) with a
Li breeding blanket (`mn = 1.1`), one chamber nets about 1025 MW.

The 110 GJ `e_store_mj` makes C220107 (the pulsed-power store) dominate
THETA_PINCH's capital -- on the order of tens of billions of dollars per
chamber -- which is the correct, faithful signal that a reactive-store theta
pinch is uneconomic. Two caveats keep it an UPPER BOUND, not a point estimate:
C220107 prices the store at the CAPACITOR `c_cap_allin_per_joule` coefficient,
whereas RTPR used a cheaper inductive / homopolar-generator (METS) store; and
no economy-of-scale discount is applied to a GJ-class store. The magnitude is
directionally right (ETS-dominated), but the exact figure should be read as a
ceiling pending an inductive-store cost basis.

**MAG_TARGET recovered-compression design point (SOURCED, GF E-267 reactor
Sankey).** The design point is General Fusion's "Design Point - 150 MWe"
Sankey (Krotez et al., SOFE 2023 E-267) -- the REACTOR diagram, distinct from
the LM26 breakeven-experiment Sankey shown at the same conference. Verified
directly from the poster: `yield_per_shot_mj = 780` is the "Fusion 780 MJ"
node (not the 886 MJ "Converted Yield" arrow, which adds directly-converted
liner-vaporization energy); `e_driver_mj = 755` is the liner kinetic-energy
arrow; `max_f_rep = 1` Hz -> 150 MWe. Fusion gain 780/755 about 1.03 is the
genuine reactor point: liquid-liner MTF closes not through high gain but
through 85% mechanical recovery of the liner KE plus 40% thermal conversion.

`driver_recovery_frac = 0.85` is SOURCED from the poster's explicit constant
"Kinetic Energy Dissipation 1 - eta_mech = 15%". On the `recovered_compression`
path this fraction sets only the thermal deposition: (1 - 0.85) x 755 MJ
dissipates into the blanket heat pool. The electrical grid draw is the
separate `e_recirc_mj = 195` (pulsed power about 173 + gas compressor about 22,
per E-267), and the C220107 cap-bank capital scales with `e_store_mj = 173`
(the plasma-injector pulsed-power store), not the full liner KE. With these
sourced values one chamber nets about 161 MW at 1 Hz -- close to GF's headline
150 MWe; the residual is the model's blanket-multiplication accounting (`mn`
on the neutron fraction) versus the diagram's lumped high-grade-heat node, not
a calibration. A single-pass electrical driver at `eta_pin=0.30` would instead
charge the full 755 MJ as recirculation (about 2517 MW at `f_rep=1`) and land
net-negative -- the reason MAG_TARGET needs the recovered-compression path.

### The recovered-compression path

Most pulsed drivers are single-pass: a laser or plasma gun deposits its pulse
energy in the target once, it is gone, and the plant redraws it from the grid
next shot. For them the stored, consumed, and delivered per-pulse energies are
all roughly proportional, so one value (`e_driver_mj`, with the store derived
as `e_driver/eta_pin` and the recirc as `e_driver*(1-recovery)/eta_pin`)
suffices -- this is `pulsed_thermal_forward`.

Two concepts break that proportionality because their driver energy is largely
RECOVERED each cycle rather than consumed:

- **THETA_PINCH (reactive recovery).** The compression field is held in an
  inductive ETS that swings the full field energy into the coils and rings
  about 98% of it back out each cycle (an LC ringdown). The plant must STORE
  and switch about 110 GJ (real cap-bank/ETS hardware, sizing C220107) but only
  CONSUMES the about 2 GJ of resistive loss -- store and consumption differ by
  about 90x. That large ETS is the historical reason theta-pinch reactors are
  uneconomic; the split lets the cost model show it instead of hiding it.
- **MAG_TARGET (mechanical recovery).** The gas piston / liquid liner rebounds
  elastically, so 85% of the liner KE returns to the next stroke and only 15%
  dissipates as heat. Delivered (755 MJ), grid redraw (about 195 MJ), and
  dissipated heat (about 113 MJ) are three different energies.

`pulsed_recovered_compression_forward` handles both. It shares the
thermal-to-electric core (`_pulsed_thermal_core`) with `pulsed_thermal_forward`
-- the ash/neutron split and the p_th -> p_et -> q_eng -> p_net conversion are
identical -- and differs only in three driver-specific quantities taken from
explicit YAML inputs: `e_driver_mj` (delivered -> gain), `e_store_mj` (peak
store -> C220107), and `e_recirc_mj` (net grid draw -> recirculation), with
`driver_recovery_frac` setting the dissipated (thermalising) fraction of the
delivered energy. For a single-pass concept these collapse to the derived
defaults, so the eight thermal concepts are unaffected.

The same reactive recovery appears in Helion's D-He3 pulsed FRC, but there the
fusion output is charged particles recovered inductively (`inductive_dec`), so
the recovery is already captured by the high `eta_pin=0.95` in
`pulsed_dec_forward` and PULSED_FRC needs no explicit split. THETA_PINCH is the
hybrid case -- reactive-recovery drive but D-T neutron (thermal) output -- which
is why it, and not Helion, needs the recovered-compression forward.

### Rep-rate shot design point sources

- **PULSED_FRC (Helion):**
  - Helion "Polaris" page (bank energy 50+ MJ): https://www.helionenergy.com/polaris
  - Helion blog, "Helion's fusion system is (basically) an RLC circuit": https://www.helionenergy.com/blog/helions-fusion-system-is-basically-an-rlc-circuit
  - Scientific American, "Helion Energy is building a fusion power plant...": https://www.scientificamerican.com/article/helion-energy-is-building-a-fusion-power-plant-can-its-technology-deliver/
  - IEEE Spectrum, "Welcome to Fusion City, USA" (commercial reactor about 1 pulse/s, 50 MW): https://spectrum.ieee.org/fusion
  - Wikipedia, "Helion Energy" (Trenta about 1 pulse/10 min; Polaris about 1 Hz): https://en.wikipedia.org/wiki/Helion_Energy
- **MAG_TARGET (General Fusion, "Design Point - 150 MWe" reactor Sankey):**
  - Krotez, Segas, Khalzov, Suponitsky, "Conceptual Design of a Magnetized Target Fusion Power Plant," IEEE SOFE 2023 paper E-267: https://generalfusion.com/wp-content/uploads/2023/07/SOFE2023_E267_daymon-krotez_v2.pdf (the reactor design point, distinct from the LM26 breakeven-experiment Sankey shown at the same conference; e_driver=755, yield=780, 1 Hz -> 150 MWe, eta_mech=0.85, eta_th=0.40)
  - General Fusion public pages: https://generalfusion.com/
- **PLASMA_JET (HyperJet/LANL PJMIF):**
  - Langendorf & Hsu, Phys. Plasmas 24, 032704 (2017), arXiv:1612.07368: https://arxiv.org/pdf/1612.07368
  - Thio & Witherspoon, "Plasma-Jet-Driven Magneto-Inertial Fusion, A progress report," Open Access Government (2018): https://www.openaccessgovernment.org/plasma-jet-driven-magneto-inertial-fusion-2/63480/
  - Thio et al., "Plasma-Jet-Driven Magneto-Inertial Fusion," Fusion Sci. Technol. 75(7) (2019), DOI: 10.1080/15361055.2019.1598736
- **THETA_PINCH (LANL RTPR reactor study):**
  - Burnett, Ellis, Oliphant, Ribe, "A Reference Theta Pinch Reactor (RTPR): A Study of a Pulsed High-Beta Fusion Reactor Based on the Theta Pinch," LASL LA-5121-MS (1972), DOI 10.2172/4603908: https://www.osti.gov/biblio/4603908
  - Krakowski, Ribe, Coultas, Hatch, "An Engineering Design Study of a Reference Theta-Pinch Reactor (RTPR)," LA-5336/ANL-8019 (1974): https://www.osti.gov/biblio/4292057
  - Krakowski, Miller, Hagenson, "Operating Point Considerations for the RTPR," LA-UR-76-2166 / CONF-760935-24 (1976): https://www.osti.gov/biblio/7341997
  - Historical experiment (NOT the reactor source): Quinn, Little, Ribe, Sawyer, "Stability, Heating and End Loss of a 3.5-Megajoule Theta Pinch (Scylla IV)," LASL (1965): https://www.osti.gov/biblio/4624642
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
