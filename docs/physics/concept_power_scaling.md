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
| Pulsed FRC | Rep-rate | Not yet (future) | Helion: a single ~50 MWe machine driven to 10 Hz |
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

## Future-sizing roadmap

- Volume sizing for STELLARATOR (solve R0/a, plus 3D coil markup), DIPOLE, and
  POLYWELL (single-device growth) would follow the tokamak/mirror pattern.
- Rep-rate and target-yield sizing for the pulsed/IFE/MIF concepts would solve
  shots-per-second or yield-per-shot from the power target; these are separate
  future worklines.

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
- Pulsed FRC: Helion Energy, helionenergy.com (pulsed approach, 10 Hz).
- Magnetized target: General Fusion, generalfusion.com/commercialization-path
  (1 Hz, two-unit plant).
- Laser IFE: Xcimer (DOE-approved design, about 1 GJ/shot, sub-1 Hz); Focused
  Energy and Marvel Fusion (10 Hz).
- Heavy-ion IFE: HIBALL and HIDIF studies (shared driver, several chambers).
