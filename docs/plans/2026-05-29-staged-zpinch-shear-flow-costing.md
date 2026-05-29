# Shear-flow infrastructure costing for STAGED_ZPINCH

Issue: 1cFE/1costingfe#20
Date: 2026-05-29

## Problem

`STAGED_ZPINCH` (Zap-style sheared-flow stabilized Z-pinch) currently has
`C220104 = 0`. The coaxial gun and gas-injection hardware that establishes the
stabilizing velocity shear is not costed in any account; only the capacitor bank
(C220107) and the vessel/structure are captured. This understates a sheared-flow
Z-pinch relative to a bare `ZPINCH`, which physically lacks that hardware.

The same root cause touches a sibling: `PLASMA_JET`, an electromagnetic gun
formation driver, is still costed on rep-rate-scaled average power (M$/MW), the
basis issue #15 corrected for lasers and accelerators. And a defining recurring
cost of any plasma-facing gun, electrode erosion, is not modeled at all.

## Scope

Three changes, one root cause (electromagnetic-gun formation hardware was either
uncosted or costed on a rep-rate-scaled basis):

1. **STAGED_ZPINCH shear-flow drive (C220104, new).** Coaxial gun electrode
   assembly (inner/outer electrodes, acceleration + assembly regions) plus the
   neutral-gas injection system (fast puff valves, plenum, fast differential
   pumping). The capacitor bank that drives the current stays in C220107. Per-MJ
   basis.
2. **PLASMA_JET basis migration (C220104).** Move from $/MW to $/MJ; the
   per-pulse-capital logic that applies to the sheared-flow gun applies equally
   to a plasma-gun. Reference-point cost preserved exactly.
3. **Formation-electrode replacement O&M (CAS72).** Plasma-facing gun electrodes
   on `STAGED_ZPINCH` and `PLASMA_JET` erode and are periodically replaced.

`MAG_TARGET` deliberately stays on $/MW: pneumatic pistons plus liquid-metal
liner recirculation physically move mass on every shot, so the handling and
recirculation plant genuinely scales with throughput (average power). This
rationale will be written into the justification doc rather than left implicit.

## Cost model

### C220104 per-MJ drivers

Both move into the `_DRIVER_COST_PER_MJ` dispatch map (cost = coefficient x
`e_driver_mj`, rep-rate-independent):

- `driver_staged_zpinch_per_mj = 1.5` (M$/MJ). About $132M at the default
  `e_driver_mj` = 88 MJ, roughly 14% of the $913M plant. Cheaper per MJ than the
  PLASMA_JET array (about 4 M$/MJ implied) since it is a single, simpler coaxial
  assembly. Sensitivity 0.75 to 3.0 M$/MJ; no hard public source.
- `driver_plasma_jet_per_mj = 4.0` (M$/MJ), replacing `driver_plasma_jet_per_mw`.
  PLASMA_JET runs at f_rep = 1.0 Hz, so this preserves its current C220104
  ($436.6M at `e_driver_mj` = 109 MJ) exactly while removing the rep-rate
  scaling.

### Electrode-replacement O&M (CAS72)

Modeled as a levelized annual recurring cost, not discrete replacement events:
electrode lifetime can be sub-multi-year, and the existing discrete-event PV loop
in `cas70` caps at `MAX_REP = 20`, which would truncate a frequently-replaced
consumable. A continuous annual rate is the robust representation.

    n_shots_per_year = f_rep * 8760 * 3600 * availability
    annual_electrode  = electrode_replace_frac * C220104 * n_mod
                        * n_shots_per_year / electrode_shot_lifetime

This is levelized like CAS71 and added to CAS72. It applies only to the
eroding-electrode EM guns, `{STAGED_ZPINCH, PLASMA_JET}`.

New constants:

- `electrode_shot_lifetime = 1.0e8` (shots before electrode-assembly
  replacement; plasma-facing erosion, high uncertainty; range 1e7 to 1e9, same
  order as `cap_shot_lifetime`).
- `electrode_replace_frac = 0.5` (consumable-electrode share of the C220104
  flow-drive capital; the gas valves, plenum, and pumping are the non-consumable
  remainder; range 0.25 to 0.75).

At the STAGED_ZPINCH default point this is about $18M/yr
(0.5 x $132M x 2.68e7 shots/yr / 1e8). The magnitude is an estimate; both
constants are anchored with sensitivity ranges in the justification writeup.

## Implementation

1. `src/costingfe/defaults.py`: add `driver_staged_zpinch_per_mj = 1.5`; replace
   `driver_plasma_jet_per_mw` with `driver_plasma_jet_per_mj = 4.0`; add
   `electrode_shot_lifetime = 1.0e8` and `electrode_replace_frac = 0.5`.
2. `src/costingfe/layers/cas22.py`: add `STAGED_ZPINCH` and `PLASMA_JET` to
   `_DRIVER_COST_PER_MJ`; remove `PLASMA_JET` from `_DRIVER_COST_PER_MW`.
3. `src/costingfe/layers/costs.py` (`cas70`): add a formation-electrode annual
   replacement term for `{STAGED_ZPINCH, PLASMA_JET}`, levelized into CAS72.
4. `docs/account_justification/CAS22_reactor_components.md`: add the
   STAGED_ZPINCH row, update the PLASMA_JET row to $/MJ, add the MAG_TARGET
   "kept on average power" rationale, and an electrode-erosion O&M note.
5. `docs/papers/1costingfe_paper/config_schema.md`: add `driver_staged_zpinch_per_mj`,
   `driver_plasma_jet_per_mj` (replacing the per-MW entry), `electrode_shot_lifetime`,
   and `electrode_replace_frac` to the schema table with defaults.
6. `docs/papers/1costingfe_paper/1costingfe_paper.tex`: reflect the change in the
   manuscript (current-methodology framing only, no history). In the CAS22.01.04
   driver table (`tab:cas2204-driver`), add a STAGED_ZPINCH row (M$/MJ, 1.5) and
   move PLASMA_JET to the $/MJ basis (4.0), updating the surrounding prose so the
   per-MJ set reads laser / heavy-ion / staged-zpinch / plasma-jet and the
   average-power set reads mag-target only. In the CAS70 / O&M section, add a
   sentence on formation-electrode scheduled replacement for the EM-gun concepts.
   Recompile to confirm it still builds.
7. Tests (`tests/test_cas22.py`, `tests/test_economics.py` or
   `tests/test_costs.py`): STAGED_ZPINCH C220104 equals
   `driver_staged_zpinch_per_mj x e_driver_mj` and is > 0; bare ZPINCH C220104
   stays 0; PLASMA_JET C220104 unchanged at the default point after migration;
   electrode O&M raises CAS72 for STAGED_ZPINCH and PLASMA_JET and is 0 for a
   non-gun concept.

## Out of scope / follow-ups

- Migrating `MAG_TARGET` to $/MJ (deliberately retained on average power as a
  throughput-scaled mechanical driver).
- Electrode-erosion O&M for any non-gun concept.
