# Shear-flow infrastructure costing for STAGED_ZPINCH

Issue: 1cFE/1costingfe#20
Date: 2026-05-29

## Problem

`STAGED_ZPINCH` (Zap-style sheared-flow stabilized Z-pinch) currently has
`C220104 = 0`. The coaxial gun and gas-injection hardware that establishes the
stabilizing velocity shear is not costed in any account; only the capacitor bank
(C220107) and the vessel/structure are captured. This understates a sheared-flow
Z-pinch relative to a bare `ZPINCH`, which physically lacks that hardware, so the
two concepts are currently costed almost identically apart from their power
balance.

## Scope

Add a C220104 shear-flow drive line for `STAGED_ZPINCH` covering the full flow
formation mechanism:

- Coaxial accelerator/gun electrode assembly: inner and outer electrodes,
  acceleration region, and assembly region.
- Neutral-gas injection system: fast puff valves, gas plenum/manifold, and the
  fast differential pumping needed to clear neutral gas between shots.

The capacitor bank that drives the current stays in C220107 (avoids
double-counting the pulsed power). The vessel and primary structure stay in
C220106/C220105.

## Cost model

Basis: per joule of pulse energy, `C220104 = driver_staged_zpinch_per_mj x e_driver_mj`.

Rationale (consistent with issue #15): the coaxial gun's capital is set by its
physical size and peak current, which are fixed by the per-pulse energetics, not
by how often it fires. Higher repetition rate adds cooling and accelerates
electrode erosion (replacement = O&M), but the dominant electrode assembly is
unchanged. The fast-valve gas hardware is likewise per-pulse; only the
differential-pumping increment is throughput-scaled, and it is secondary. A
single coaxial gun is not an array whose count scales with throughput, so the
average-power story that keeps PLASMA_JET and MAG_TARGET on $/MW does not apply
here.

New constant: `driver_staged_zpinch_per_mj = 1.5` (M$/MJ).

- Cheaper per MJ than the PLASMA_JET gun *array* (about 4 M$/MJ implied,
  $437M / 109 MJ), since this is a single, simpler coaxial assembly.
- At the default operating point (`e_driver_mj` = 88 MJ) this gives
  `C220104` of about $132M, roughly 14% of the $913M plant, consistent with the
  low-cost-pulsed-power premise of the concept.
- No hard public source for the coefficient; sensitivity range 0.75 to 3.0
  M$/MJ (0.5x to 2x), to be stated in the justification writeup.

## Implementation

1. `src/costingfe/defaults.py`: add `driver_staged_zpinch_per_mj: float = 1.5`
   in the C220104 pulsed-driver block, with a comment explaining the basis.
2. `src/costingfe/layers/cas22.py`: add
   `ConfinementConcept.STAGED_ZPINCH: cc.driver_staged_zpinch_per_mj` to the
   `_DRIVER_COST_PER_MJ` map (the per-pulse-energy driver dispatch).
3. `docs/account_justification/CAS22_reactor_components.md`: add a
   `STAGED_ZPINCH` row to the driver table and a short rationale paragraph
   (hardware scope, per-MJ basis, magnitude and sensitivity).
4. `docs/papers/1costingfe_paper/config_schema.md` and
   `1costingfe_paper.tex`: add the new coefficient to the schema row and the
   CAS22.01.04 driver table.
5. Tests (`tests/test_cas22.py`): assert `STAGED_ZPINCH` C220104 equals
   `driver_staged_zpinch_per_mj x e_driver_mj` and is > 0, and that bare
   `ZPINCH` C220104 stays 0.

## Out of scope / follow-ups

- Migrating PLASMA_JET and MAG_TARGET from $/MW to $/MJ for full
  cross-driver consistency in C220104.
- A dedicated electrode-replacement O&M line (CAS70-side).
