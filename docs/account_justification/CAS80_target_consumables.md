# Target Factory (CAS22.01.08) and Per-Shot Target Consumables (CAS80)

**Date:** 2026-06-08 (supersedes 2026-06-04)
**Status:** Implemented

## Scope

Inertial-fusion-energy (IFE) and magneto-inertial-fusion (MIF) concepts consume
a **fabricated target** on every shot: the capsule and hohlraum of a laser or
heavy-ion target, the metal liner and recyclable transmission line (RTL) of a
MagLIF or Z-pinch load. Costing this correctly requires **two** accounts, not one:

- **CAS22.01.08 (`target_factory_capex`)** -- the on-site, plant-dedicated
  factory that fabricates, fills, and fields the targets. Capital.
- **CAS80 (`target_unit_cost`)** -- the recurring per-shot hardware consumed in
  the burn (materials, factory operating cost, waste handling). Annualized.

These are gated by two independent knobs because their magnitudes are set by
different physics. The earlier single flat factory number ($244M, inherited
from the Miles/LIFE factory total-capital-cost via pyFECONS) both lacked a
documented derivation and **double-counted** the factory: the per-shot cost was
sourced from General Atomics / NAS studies that are *fully-loaded* (they already
amortize the factory capital), so banking a separate $244M capital line counted
the plant twice. This document records the de-sourced, bottom-up replacement.

## Why an on-site factory is plant capital (the fuel-assembly analogy, refined)

In the GEN-IV / fission cost structure, fuel fabrication is **not** a plant
capital line: a utility buys finished assemblies from a merchant vendor at a
fully-loaded $/kgU price, and that cost lives in the fuel-cycle account (the
CAS80 analogue). IFE breaks the analogy in one decisive way: the targets are
**DT-filled, cryogenically ice-layered, tritium-bearing, and consumed on-site at
high rep rate**. They cannot be shipped from a merchant vendor -- the DT ice has
a beta-self-heating-limited shelf life and the throughput is ~10^8/yr. So the
factory is a dedicated building inside the plant fence, and its capital belongs
in CAS22.01.08. The per-shot *hardware* still belongs in CAS80, as the fission
fuel-assembly convention puts the fabricated assembly (not just the isotope) in
the fuel account. The DT isotope itself is costed separately, in the recycled
fuel formula (`CAS80_annualized_fuel_cost.md`), and is not in `target_unit_cost`.

## Factory capital (CAS22.01.08): bottom-up build-up

`C220108 = target_factory_capex x (p_net / 1000)^0.7` -- the per-concept capex
is the factory capital at 1 GWe; the exponent scales with plant size.

The capsule factory (laser / heavy-ion) is a **two-zone** facility, and three
compounding drivers set its cost on top of the raw material bill:

1. **Precision.** Capsule surface finish, sphericity, and wall uniformity are
   held to "a few tenths of a percent" because surface defects seed
   Rayleigh-Taylor growth. That makes the fabrication/metrology zone a
   precision cleanroom (ISO 5-6), benchmarked at $1,500-7,500/sqft -- versus the
   $250/sqft generic construction the Miles model assumed (a ~5-30x premium on
   the building alone). The 51.6k-sqft Miles floorplan is ~$130M of cleanroom,
   not ~$13M.
2. **Tritium-confinement manufacturing.** The back half of the line (DT fill,
   ice-layering, contained assembly, characterization, injection) operates
   inside ITER-style dual confinement: process-in-gloveboxes, glovebox
   detritiation, dedicated ventilation, zoning, tritium accountancy. This is a
   multiplicative premium (~2-4x, engineering estimate -- no public $) on the
   hot-zone fraction (~40%) of the line, *on top of* precision.
3. **Yield.** At 10^8/yr targets are not individually characterized; the NOAK
   premise is a process repeatable enough for statistical sampling. The factory
   must be oversized by 1/Y to deliver good output; baseline Y = 0.80 (range
   0.6-0.95), deliberately off the optimistic edge since no pilot line exists.

The post-shot debris is **tritiated and neutron-activated**, so a contained
recovery / hot-cell back-end is added on the recycle path (FAFNIR $344M / IFMIF
$1,145M class).

### Calibration to LIFE

LIFE's published numbers -- total plant $4-6B, target $0.25 ($0.20-0.30)
fully-loaded -- back out to a ~$600M target factory. That lands on the
**optimistic corner** of the build-up below, so the floor of our range
reproduces the most credible IFE economics study rather than being free-floating.

### Per-concept values (1 GWe, dispose default)

| Concept | `target_factory_capex` (M$) | `target_unit_cost` ($/shot) | factory type |
|---|---|---|---|
| `laser_ife` | **725** (range 434-1564) | **0.50** (range 0.27-1.28) | precision + tritium capsule cleanroom |
| `heavy_ion` | **780** | **0.62** | capsule + lead-hohlraum press/assembly |
| `maglif` | **150** | **9.0** | metal liner stamping/machining + DT-fill shop |
| `zpinch` | **150** | **6.0** | wire-array/load fabrication + DT-fill shop |

MagLIF/Z-pinch are **not** CVD-diamond-and-cryo capsule cleanrooms -- they form
and fill metal liners, a far cheaper casting/machining process. Their previous
~$300M factory (the global laser base applied to every manufactured-target
concept) was a phantom; $150M reflects a real liner shop. Their per-shot cost is
already a bottom-up liner + RTL-remanufacturing number (the RTL remelt/recast
plant is a distinct recycling facility, amortized in `target_unit_cost`, not in
`target_factory_capex` -- no double count).

## Per-shot consumable (CAS80)

```
A_target = N_mod x N_shot_per_year x target_unit_cost          [$ /yr]
N_shot_per_year = f_rep x 8760 x 3600 x availability           [shots/yr per module]
```

`target_unit_cost` now carries only the **non-capital** per-shot cost --
materials + factory operating (labor, maintenance, detritiation, waste) divided
by yield -- because the capital lives in CAS22.01.08. The burn-fraction
correction does not apply: the target is consumed whole each shot. For laser
direct-drive the materials are sub-cent (a diamond ablator shell; the cost is
the CVD *process*, which is factory capex/opex); for indirect drive a lead
hohlraum (~3g, ~$0.008) adds an assembly step. The number rose from the old
$0.40 (an optimistic GA/NAS *requirement*) because the bottom-up opex carries
the tritium markup and the yield divisor.

## Recycle vs dispose

The model defaults to **dispose**. For LIFE-class **lead** hohlraums (chosen for
low cost, low-level activation, and a low melting point), recycling recovers
~$0.008 of metal per target while the contained recovery hot cell costs ~$0.2/
target amortized -- so material recycling is not worth it. The decision instead
hinges on radiological handling, which is incurred *either way*: all debris is
tritiated regardless of activation, and recovery requires a contained facility.
Computed both ways, dispose and recycle land within ~3 $/MWh of each other
(dispose marginally cheaper except at the conservative corner), inverting the
old "recyclable, net of recycling" default for capsules. The heavier *activated
steel* RTLs of MagLIF/Z-pinch are the case where recovery does pay, and their
per-shot RTL numbers are already net-of-recycling.

**This is a markup the prior model omitted entirely.** Post-shot LLW
conditioning/disposal and contained recovery are real costs that the
"net of recycling, cheap remanufacturing" credit left out.

## Result (1 GWe, vs MFE tokamak 86.0 $/MWh)

| concept | LCOE $/MWh | note |
|---|---|---|
| laser_ife | **107.8** | base case; band 90 (LIFE-optimistic) / 108 / 162 |
| heavy_ion | **100.1** | band 84 / 100 / 129 |
| maglif | 104.2 | phantom factory removed ($324M -> $150M) |
| zpinch | 82.3 | low-rep liner concept; competitive once de-phantomed |

Honest, de-double-counted costing **restores ICF > MFE** for the high-rep
capsule concepts -- the brief "parity" seen from de-double-counting alone was an
artifact, recovered once the precision + tritium + yield + radiological premiums
are included. The dominant residual uncertainties (band width) are the
tritium-facility markup and the yield tail, neither of which has a public dollar
basis; they are carried as the stated range rather than a false-precision point.

## Hard-NOAK costing principle

Target cost is a "hard NOAK" item: a ~10^4 reduction from today's ~$2,500
hand-built targets to a sub-dollar mass-produced unit is the central IFE
challenge. The old $0.40 was an *economic-viability requirement* (what targets
must cost for IFE to compete), which is circular -- it assumes the success being
evaluated. The bottom-up build-up here costs the **factory and the unit** from
their physics/manufacturing instead, with the NOAK aggressiveness confined to
projected unit prices carried as ranges. Same rule applies to REBCO $/kA·m and
capacitor $/J. See `feedback_hard_noak_costing` (project memory).

## Gating: two knobs

A concept that fabricates a consumed target declares **both**
`target_unit_cost > 0` (the per-shot consumable + factory power) **and**
`target_factory_capex > 0` (the factory capital). In-situ-formation concepts
(plasma-jet, liquid-liner MTF, FRC, theta-pinch, dense-plasma-focus, staged-Z)
leave both at 0 and carry no factory by default. A pellet-fed MTF (e.g. NearStar)
sets both in its own concept config. The factory magnitude is its own knob,
rather than a global base, because a capsule cleanroom and a liner shop differ
by ~5x.

## Sources

Target fabrication / mass production:
1. Miles et al., "Roadmap for Achieving IFE Target Mass Production," LLNL-TR-416932 (2009) -- factory tooling/floorspace build-up.
2. Goodin, D.T., Besenbruch, G.E., et al. (General Atomics, 2000-2004), *Nuclear Fusion* 44 (2004) S254 -- requirement ~$0.25-0.30/target; ~500,000 targets/day; current experimental ~$2,500/target.
3. Alexander, N.B. / Goodin, "Developing a commercial production process for 500,000 targets/day," *Phys. Plasmas* 13 (2006) 056305; statistical-sampling characterization regime.
4. National Academies (2013), *Assessment of the Prospects for IFE* -- mass-production goal $0.20-0.40/target.
5. Rickman & Schultz / Rickman, Goodin et al. (GA, 2003) -- indirect-drive hohlraum target costing, ~40.8 cents/target on-site.

LIFE plant + economics:
6. Anklam et al., "LIFE: The Case for Early Commercialization of Fusion Energy," LLNL-TR-480444 (2011); total plant $4-6B, target $0.20-0.30.
7. Dunne et al., FST 60 (2011) -- LIFE on-site target factory, ~$0.25/target.

Hohlraum material + recycling:
8. "Lead (Pb) Hohlraum: Target for Inertial Fusion Energy," *Sci. Rep.* 3, 1453 (2013) -- LIFE lead hohlraum, ~3g/target, low activation.
9. Goodin/Alexander, FPA 2007; "Recycling issues facing target and RTL materials of inertial fusion designs," *Nucl. Instrum. Methods A* 544 (2005) -- recycle vs dispose framing.
10. Cipiti & Rochau, "In-Zinerator," SAND2006-6590 (2006) -- RTL remanufacturing cost (steel $3.10-5.40/RTL); Olson, IAEA IFE CRP (2001).

Facility cost benchmarks:
11. Semiconductor cleanroom construction $1,500-7,500/sqft (ISO 5-3); precision-fab envelopes $1B+ at scale.
12. FAFNIR (~$344M, 2012) / IFMIF (~$1,145M) -- hot-cell + remote-handling facility class.
13. ITER tritium plant -- dual-confinement / glovebox-detritiation architecture (cost premium not publicly itemized; carried as 2-4x engineering estimate).

**FOAK contrast (not used):** today's targets are ~$2,500 (experimental,
hand-built, no factory); NIF-class precision capsules far more. The model is the
NOAK mass-production regime, with the factory's capital and radiological burden
made explicit rather than amortized into a deceptively precise sub-dollar number.
