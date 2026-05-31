# CAS22.01.06 Vacuum System: vessel shell + gas-load pumping

**Date:** 2026-05-31
**Status:** Justified — model implemented, all tests passing

## Overview

C220106 covers the vacuum system. It has two physically distinct parts that
scale on different drivers, so it is costed as a sum:

```
C220106 = vessel_shell(volume-based) + pumping(gas-load-based)
```

- **Vessel shell** (volume-based, inherited): the welded stainless chamber,
  port extensions, gauges, and leak detection. Cost `= vessel_unit_cost ×
  V_vessel × (P_et/1100)^0.6`, `vessel_unit_cost = 0.72 M$/m³`. This scales with
  reactor size, which is the right driver for the chamber itself.
- **Pumping** (gas-load-based): the installed cryopumping/turbopumping capacity.
  Its cost is set by the gas *throughput* the machine produces and the pressure
  it must hold, not by vessel volume.

The pumping term is applied uniformly to every concept. The same physics that
makes a beam-driven, low-pressure linear device pump hard also makes a
high-pressure tokamak divertor pump cheaply, so a single model handles all
concepts and the differences fall out of the inputs rather than per-concept
hand-tuning.

## Why pumping is not a volume account

A pump is rated by speed `S` (m³/s); the gas it removes is a throughput `Q`
(Pa·m³/s). They are tied by the pump equation `Q = S · P`, so the required
installed speed is

```
S_req = Q_gas / P_op
```

where `P_op` is the operating (plenum) pressure at the pump throat. The cost is
then `pump_unit_cost × S_req`. None of this depends on vessel volume; it depends
on how much gas the plasma throughput and the heating system inject, and how low
a pressure the concept must hold.

## Gas throughput model

Throughput is a particle removal rate times `kT` (`Q = N · kT`, gas at
`pump_gas_temp_k = 300 K`). Two sources dominate; wall outgassing scales with
surface area but is negligible for a baked UHV system and is omitted.

**NBI neutral gas.** A neutral beam of power `p_nbi` made of particles of energy
`E_b` injects particles at rate `p_nbi/E_b`. The un-trapped beam plus
neutralizer reflux is a gas amplification `g`:

```
Q_nbi = g · (p_nbi / E_b) · kT
```

Higher beam energy injects fewer particles per MW, so this term scales as `1/E_b`.

**Fueling/exhaust gas.** Sustaining fusion power `p_fus` requires a reaction
rate `p_fus/E_fus`; only `burn_fraction` of the fuel fed actually burns, so the
fuel feed is `reaction_rate/burn_fraction` and the unburned `(1 - burn_fraction)`
circulates through the pump:

```
Q_fuel = ions_per_reaction · (1 - burn_fraction)/burn_fraction
         · (p_fus / E_fus) · kT
```

Then `S_req = (Q_nbi + Q_fuel)/P_op` and `C220106_pump = pump_unit_cost · S_req`.

## Constants

| Constant | Value | Basis |
|---|---|---|
| `pump_unit_cost` | 0.015 M$/(m³/s) | $15/(L/s), commodity cryopump procurement |
| `pump_nbi_gas_amplification` (g) | 1.0 | gas particles pumped per beam particle; calibrated to C-2W |
| `pump_gas_temp_k` | 300 K | pumped-gas temperature for Q = N·kT |
| `pump_ion_per_reaction` | 2.0 | fuel ions consumed per fusion reaction |
| `nbi_beam_energy_kev` | 120 keV | reference reactor NBI energy; gas load scales 1/E_b |
| `e_fus_mev_{dt,dd,dhe3,pb11}` | 17.6 / 3.65 / 18.3 / 8.7 | energy released per reaction |

Per-concept operating pressure `vac_op_pressure_pa` lives in the concept YAMLs
(global default 1.0 Pa in `costing_constants.yaml`):

| Concept | P_op [Pa] | Rationale |
|---|---|---|
| Tokamak, Stellarator | 3.0 | divertor/island plenum designed to pump at high recycling pressure |
| Mirror | 0.02 | open field lines; low neutral pressure to limit charge-exchange losses |
| Steady FRC | 0.05 | open-field-line linear device, low neutral pressure |

## Calibration

The NBI gas amplification `g` is anchored to C-2W, which runs about
**2,000 m³/s** of divertor pumping to handle its neutral-beam gas load at roughly
21 MW injected (tae-c2w-machine-details.md). At `g = 1.0` with the reactor beam
energy, the NBI term reproduces a comparable specific pumping speed; the C-2W
figure also confirms that pumping is a real, large installed capacity for a
beam-driven device, not a rounding error.

The unit cost `$15/(L/s)` is the commodity cryopump range (CTI/Brooks On-Board
8F class at about 1,500 L/s for a few tens of $k per unit; ITER torus cryopumps
at 100 m³/s each are costlier per unit but similar per L/s at scale).

## Verified per-concept results

Run at 200 MWe net, NOAK (`/tmp` reproduction, all signs internally consistent):

| Concept | P_op | Q_nbi | Q_fuel | S [m³/s] | pump [M$] | vessel [M$] | C220106 [M$] |
|---|---|---|---|---|---|---|---|
| Tokamak DT | 3.00 | 10.8 | 41.5 | 17 | 0.3 | 72.3 | 72.5 |
| Stellarator DT | 3.00 | 0.0 | 37.6 | 13 | 0.2 | 24.9 | 25.1 |
| Mirror DT | 0.02 | 8.6 | 39.3 | 2,398 | 36.0 | 13.9 | 49.8 |
| FRC DT | 0.05 | 17.2 | 49.0 | 1,326 | 19.9 | 7.9 | 27.8 |
| FRC pB11 | 0.05 | 17.2 | 104.2 | 2,429 | 36.4 | 7.8 | 44.2 |

Behaviors that fall out of the model:

- **High-pressure concepts pump cheaply.** Tokamak and stellarator pumping is
  $0.2-0.3M; their C220106 stays vessel-dominated and essentially unchanged.
  A high-recycling divertor designed to run at a few Pa does not incur a large
  pumping bill, which is the correct physics.
- **Low-pressure concepts pump expensively.** Mirror and FRC pumping is
  $20-36M, driven by the open-field-line requirement to hold a low neutral
  pressure.
- **Fueling dominates over NBI.** Once beams are at reactor energy, the fueling
  throughput is the larger term. The NBI gas load that originally motivated this
  account is the minor contributor.
- **pB11 pumps more than DT** ($36M vs $20M at the same machine), purely through
  fueling: pB11 releases about half the energy per reaction (8.7 vs 17.6 MeV),
  so it runs about twice the reaction rate for the same power and feeds/exhausts
  about twice the fuel particles.

## Key uncertainties

1. **`P_op` is the dominant knob.** Cost scales as `1/P_op`, and plausible plenum
   pressures span more than an order of magnitude. It lumps in edge physics (how
   much of the recirculating flux reaches the pump versus recycling at the wall),
   so it is an effective pressure, not a measured one. This is the parameter most
   worth pinning per concept.
2. **`g` is calibrated, not derived.** The single C-2W anchor sets it; reactor
   beam-coupling and neutralizer gas at higher energy are unvalidated.
3. **pB11 fueling is a rough proxy.** `ions_per_reaction = 2` and a generic-gas
   treatment understate that pB11 plasmas run proton-rich (proton:boron about
   5:1 or higher to control boron bremsstrahlung) and that boron is injected as
   solid and largely deposits on walls rather than being gas-pumped. The pB11
   throughput is therefore a floor; the proton recycling load could be higher.
4. **Reactor-regime pumping data is sparse.** C-2W and ITER anchor the two
   regimes; reactor-scale gas-load scaling is extrapolated.

## Reporting convention

Because this account bundles two components with distinct cost bases, the model
emits informational sub-lines `C220106_vessel` and `C220106_pump` in
`cas22_detail` alongside the canonical `C220106` total. These follow a general
`<CODE>_<component>` convention: they are excluded from every aggregation path
(the canonical total already carries the sum) and exist for visibility and
sensitivity tracing. C220106 is currently the only account that uses sub-lines,
since it is the only one bundling materially different, concept-divergent bases.

## Implementation

- Model: `src/costingfe/layers/cas22.py`, C220106 block.
- Constants: `src/costingfe/data/defaults/costing_constants.yaml` and the
  `CostingConstants` dataclass in `src/costingfe/defaults.py`.
- Per-concept `vac_op_pressure_pa`: concept YAMLs in `src/costingfe/data/defaults/`.

## Sources

- C-2W divertor pumping (2,000 m³/s) and NBI: knowledge/concept_research/18-p-b11-frc
  source extracts (tae-c2w-machine-details.md).
- ITER torus cryopumps (8 × 100 m³/s): IAEA IT/P7-16.
- Commodity cryopump $/(L/s): CTI/Brooks On-Board 8F class procurement.
- Pump equation `Q = S·P` and throughput `Q = N·kT`: standard vacuum technology.
