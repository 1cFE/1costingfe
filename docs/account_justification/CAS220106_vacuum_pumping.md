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
circulates through the pump. The pump removes GAS-PHASE particles, which is
where the species accounting enters, keyed by fuel:

- hydrogenic atoms recombine to diatomic molecules before reaching the pump
  (two atoms per pumped particle);
- noble-gas ash (He-3, He-4) pumps as atoms, counted per reaction;
- condensable species never reach the pump: unburned boron is a refractory
  solid (melting point 2076 C, negligible vapor pressure) that plates onto
  end-tank/collector liners and is handled as a solids-recovery loop, not
  installed pumping speed.

```
Q_fuel = [ gas_mol_per_pair(fuel) · (1 - burn_fraction)/burn_fraction
           + ash_gas_per_reaction(fuel) ]
         · (p_fus / E_fus) · kT
```

with `gas_mol_per_pair` = {DT 1.0, DD 1.0, DHe3 1.5, PB11 0.5} and
`ash_gas_per_reaction` = {DT 1.0, DD 0.75, DHe3 1.5, PB11 3.0} (derivations
in `costing_constants.yaml`). At burn_fraction 0.05 this gives 20 pumped
particles per reaction for D-T (19 hydrogenic molecules + 1 He) and 12.5 for
p-B11 (9.5 H2 + 3 He, no boron gas).

Then `S_req = (Q_nbi + Q_fuel)/P_op` and
`C220106_pump = pump_unit_cost[pump_basis(concept)] · S_req`.

## Two pump technology bases

The cost per unit installed speed depends on whether the machine has free
wall area to pump on or must pump through ports; the per-concept assignment
is `pump_basis` in `costing_constants.yaml`.

**Discrete** ($15/(L/s) NOAK): valved, housed, ducted cryo/turbo/mechanical
pumps behind ports. Applies to closed toroidal machines (tokamak,
stellarator) and to pulsed chambers whose walls cannot carry bare panels.
The ITER torus cryopumps (8 tritium-rated, valved, regenerable units at
EUR 19M, about 50 m^3/s class each: about $50/(L/s)) bracket this from
above as the FOAK nuclear-component extreme; commodity cryopump procurement
brackets it from below.

**Cryopanel** ($2.5/(L/s) NOAK): bare cryopanel arrays lining an exhaust
tank on open-geometry machines, where speed scales with nearly-free wall
area (the pump is the tank wall). Demonstrated at scale: MFTF-B built about
1,100 m^2 of LHe cryopanels (VanSant et al., UCRL-83590; Margolies & Valby,
UCRL-87735), still the largest cryopanel system ever constructed. Four
independent costed mirror/DEC designs cluster tightly once escalated to
2025 dollars:

| study | basis | 2025 $/(L/s) |
|---|---|---|
| Hoffman, UCID-17560 (1977) | $6,300/m^2 cryopanel at demonstrated specific speeds | about 0.7 |
| WITAMIR-I, UWFDM-400 (1980) | cryopumps $20,000/m^2; account 22.01.06 total $26.8M | about 1.5 |
| MARS, UCRL-53480 (1984, tenth-of-a-kind) | vacuum systems account $6.97M over about 1e7 L/s | about 2 |
| TASKA, KfK-3311/UWFDM-500 (1982) | vacuum account $16.1M over 1.5e7 L/s | about 3.5 |

$2.5/(L/s) sits above the cluster midpoint to carry the helium-sorption
share: noble-gas ash does not cryocondense at 4.5 K and needs
charcoal-sorption panels with scheduled regeneration (WITAMIR carried the
split explicitly: about 5 L/s/cm^2 for D/T against about 2 for He). A
bottom-up cross-check: a panel at WITAMIR's escalated $76k/m^2 delivering
its 5 L/s/cm^2 is $1.5/(L/s) directly.

### Per-concept basis assignment

Open geometry (cryopanel): mirror, steady FRC, dipole, polywell, orbitron,
and pulsed FRC — the Helion-class machine exhausts along open field lines
to remote end divertor chambers (Helion patents US20170011811A1 /
EP3103119B1), the same pump-on-tank-wall geometry as the steady mirror.

Enclosed (discrete): tokamak and stellarator (port-limited), and the
remaining pulsed family: buffer-gas or liquid-wall IFE chambers at
Torr-class fill pressures (Sombrero 0.5 Torr Xe, DOE/ER-54100; LIFE about
1e16 cm^-3 Xe, turbo-pumped, Latkowski et al. 2010), Z-IFE/MagLIF chambers
at 10-20 Torr with condensable-flibe clearing and rep-rate pumping flagged
as unresolved R&D (SAND2006-7148), RTPR's radially-flushed toroidal modules
on mechanical Roots blowers (LA-5336 itemizes 704 pumps at 12.3 MWe, the
most fully engineered pulsed vacuum design on record), and MTF/DPF
pre-filled vessels.

## Constants

| Constant | Value | Basis |
|---|---|---|
| `pump_unit_cost_discrete` | 0.015 M$/(m³/s) | $15/(L/s), valved/ducted pumps behind ports (NOAK; ITER FOAK tritium-rated units about $50) |
| `pump_unit_cost_cryopanel` | 0.0025 M$/(m³/s) | $2.5/(L/s), bare panel arrays on open-geometry exhaust tanks (Hoffman/WITAMIR/MARS/TASKA cluster, MFTF-B-demonstrated technology) |
| `pump_basis` | per concept | open geometry -> cryopanel; enclosed -> discrete (see above) |
| `pump_nbi_gas_amplification` (g) | 1.0 | gas particles pumped per beam particle; calibrated to C-2W |
| `pump_gas_temp_k` | 300 K | pumped-gas temperature for Q = N·kT |
| `pump_gas_mol_per_pair` | per fuel | gas-phase particles per unburned consumed-ion-pair (DT 1.0, DD 1.0, DHe3 1.5, PB11 0.5) |
| `pump_ash_gas_per_reaction` | per fuel | gas-phase ash particles per reaction (DT 1.0, DD 0.75, DHe3 1.5, PB11 3.0) |
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

## Verified per-concept results

Run at 200 MWe net, NOAK:

| Concept | pump [M$] | vessel [M$] | C220106 [M$] |
|---|---|---|---|
| Tokamak DT | 0.1 | 118.9 | 119.0 |
| Stellarator DT | 0.1 | 24.9 | 25.0 |
| Mirror DT | 3.7 | 13.9 | 17.5 |
| Mirror pB11 | 4.5 | 13.6 | 18.1 |
| FRC DT | 1.4 | 7.2 | 8.6 |
| FRC pB11 | 1.8 | 7.1 | 8.9 |

Behaviors that fall out of the model:

- **High-pressure concepts pump cheaply.** Tokamak and stellarator pumping is
  order $0.1M; C220106 stays vessel-dominated. A high-recycling divertor
  designed to run at a few Pa does not incur a large pumping bill, which is
  the correct physics.
- **Low-pressure open machines carry the real pumping load, priced as
  panels.** Mirror and FRC install thousands of m^3/s to hold their
  open-field-line neutral pressure, but at cryopanel-array economics the
  bill is single-digit M$; the installed speeds are the same order as the
  costed mirror designs (MARS about 1e7 L/s, TASKA 1.5e7 L/s), which
  validates the throughput model.
- **Fueling dominates over NBI.** Once beams are at reactor energy, the
  fueling throughput is the larger term.
- **pB11's gas load per reaction is smaller than a naive ion count.** It
  runs about twice the reaction rate of D-T for the same power (8.7 vs 17.6
  MeV per reaction), but its unburned boron condenses (a solids-recovery
  loop on end-tank liners, not a pump load) and its protons pump as H2, so
  the pumped-gas count is 12.5 particles per reaction against D-T's 20; the
  two effects nearly cancel at the plant level.

## Key uncertainties

1. **`P_op` is the dominant knob.** Cost scales as `1/P_op`, and plausible plenum
   pressures span more than an order of magnitude. It lumps in edge physics (how
   much of the recirculating flux reaches the pump versus recycling at the wall),
   so it is an effective pressure, not a measured one. This is the parameter most
   worth pinning per concept. Bracketing evidence for the mirror: the classic
   DEC designs held their end tanks at 2-4e-3 Pa to control charge-exchange
   losses on the collector optics (MARS 2e-5 torr; WITAMIR 3e-5 torr; Hoffman
   UCID-17560 trades cost directly against the CX loss fraction and calls 1%
   loss "exceedingly costly" versus 10%), while GDT measured that its expander
   tolerates neutral densities up to 1e14 cm^-3, about 0.4 Pa, without
   confinement degradation (Soldatkina et al., Plasma Fusion Res. 14, 2402006
   (2019)). The mirror's 0.02 Pa sits between the two. Coupling caveat: P_op
   must not be relaxed for pump savings in a DEC machine without charging DEC
   efficiency, because background neutrals feed charge-exchange losses on the
   collector potential structure; that trade is Hoffman's, not a free knob.
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
