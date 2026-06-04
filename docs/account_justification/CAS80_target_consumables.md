# CAS80: Per-Shot Target Consumables (IFE / MIF)

**Date:** 2026-06-04
**Status:** Implemented

## Scope

For inertial-fusion-energy (IFE) and magneto-inertial-fusion (MIF) concepts the
fuel-bearing consumable is not only the isotope but the **fabricated target
destroyed on every shot**: the capsule and hohlraum of a laser or heavy-ion
target, the metal liner and recyclable transmission line (RTL) of a MagLIF or
Z-pinch load. This document justifies the per-shot target cost
`target_unit_cost` ($/shot) added to CAS80, and the gating of the CAS22.01.08
target factory and its power draw on the same knob.

This is distinct from the fuel isotope cost (documented in
`CAS80_annualized_fuel_cost.md`): the D-T inside the capsule is the isotope; the
capsule/ablator/hohlraum/liner/RTL around it is the fabricated hardware costed
here. It is also distinct from the target *factory* capital (CAS22.01.08,
`CAS22_plant_systems.md`), which is the one-time manufacturing plant; this is its
recurring per-unit throughput.

## Why CAS80 (the fuel-assembly analogy)

In the GEN-IV / fission cost structure CAS80 ("fuel") holds the cost of the
fabricated fuel **assembly** -- cladding, structure, and fabrication -- not only
the raw uranium. The IFE/MIF target is the direct analogue: it is the
fuel-bearing unit delivered to the burn, and its fabricated hardware belongs in
the same account as the isotope it carries. Per-shot wear of *durable plant
hardware* (laser optics, electrodes, capacitor banks) is handled separately as
scheduled replacement in CAS72; the target is a consumed product, not plant
hardware, so it is a CAS80 consumable.

## Formula

```
A_target = N_mod × N_shot_per_year × target_unit_cost          [$ /yr]
N_shot_per_year = f_rep × 8760 × 3600 × availability           [shots/yr per module]
```

One target per shot per module. `A_target` is added to the annual fuel
expenditure and levelized with the same growing-annuity procedure (CAS70). The
burn-fraction correction does **not** apply: the target is consumed whole each
shot regardless of fuel burnup.

Because the rep rate enters linearly, the annual contribution is dominated by
high-rep concepts: a 10 Hz laser plant consumes ~2.7e8 targets/yr, so a $0.40
capsule contributes ~$107M/yr (a multiple-percent share of LCOE), while a 0.1 Hz
liner concept at ~2.7e6 shots/yr contributes far less in absolute terms despite
a much higher per-unit cost.

## Per-concept values (2024 USD, NOAK)

| Concept | `target_unit_cost` | Basis |
|---|---|---|
| `laser_ife` | **$0.40/shot** | NOAK fabricated capsule. Direct-drive ~$0.27, indirect (hohlraum) ~$0.70; $0.40 is a mid value. |
| `heavy_ion` | **$0.65/shot** | NOAK indirect-drive hohlraum + capsule; the metal hohlraum/radiator hardware roughly triples the bare capsule (~$0.11) and dominates the number. |
| `maglif` | **$9.0/shot** | NOAK beryllium liner (~$4.7) + steel RTL (~$5, net of recycling). Tin-RTL variant ~$6.5. |
| `zpinch` | **$6.0/shot** | NOAK dynamic-hohlraum wire-array load (~$4.9) + RTL (~$1.2, net of recycling). |
| `mag_target` | **$0.0** | Generic MTF (General Fusion / Helion-class) forms the target in-situ and recycles a liquid-metal liner: no manufactured target. A solid-liner MTF would be ~$2--15; a pellet-fed MTF (e.g. NearStar) sets a nonzero value in its own concept config. |
| `plasma_jet` | **$0.0** | PJMIF forms its imploding liner from plasma (standoff guns, "produces no hardware debris"): no fabricated target. Working gas is sub-cent and electrode wear is a CAS72 item. |
| `pulsed_frc`, `theta_pinch`, `dense_plasma_focus`, `staged_zpinch` | **$0.0** | In-situ plasma formation from gas; no manufactured target. |
| all steady-state (MFE) | **$0.0** | Continuous operation, no shots. |

These are projected NOAK mass-production costs (target factories at
10^5--10^6 units/day), not demonstrated. They are escalated to 2024 USD from the
study years (CPI ~1.5--1.7 from 2003--2006).

**FOAK contrast (not used):** today's one-off targets are orders of magnitude
more expensive -- NIF-class laser capsules cost $10^4 or more apiece, and
current experimental targets ~$2,500 -- versus the sub-dollar NOAK goal. The
factor ~10^4 reduction is the central target-fabrication challenge for IFE
economics; we model the NOAK reactor regime, not FOAK.

## RTL recycling caveat (MagLIF / Z-pinch)

For pulsed-power loads the recyclable transmission line is a per-shot consumable
comparable to or larger than the liner itself. The cited RTL dollar figures are
already **net of recycling**: RTL fragments are recovered and remanufactured, and
the cost is the remanufacturing-plant amortization (remelt + recast + refab), not
the value of virgin metal discarded each shot. The cost is therefore largely
mass-insensitive (a casting process). Do **not** additionally subtract a salvage
credit -- it is already applied. A raw-material replacement basis (no recycling)
would overstate a ~45 kg steel RTL by ~$135/shot of metal value that is in fact
recovered.

## Gating: one knob, three symptoms

A concept that manufactures no target should carry neither (a) the CAS22.01.08
target factory, (b) the target-factory power draw `p_target`, nor (c) a CAS80
per-shot consumable. These are three symptoms of one root -- *does the concept
fabricate a consumed target* -- so all three are driven by `target_unit_cost`:
the factory and its power are present iff `target_unit_cost > 0`. This makes the
in-situ-formation concepts (plasma-jet, liquid-liner MTF, FRC, theta-pinch,
dense-plasma-focus, staged-Z) correct by default in the framework, rather than
relying on per-concept `C220108 = 0` cost overrides in downstream analyses.

## Sources

Laser IFE:
1. Woodworth, J.G. & Meier, W.R. (1995). "Target production for inertial fusion energy," LLNL (OSTI 125415). ~16 cents/target (1995 USD), direct-drive, nth-of-a-kind; target fab ~6% of COE.
2. Goodin, D.T., Besenbruch, G.E., et al. (General Atomics, 2000--2004), incl. Goodin et al., *Nuclear Fusion* 44 (2004) S254. Requirement ~$0.25--0.30/target; ~500,000 targets/day; current experimental ~$2,500/target.
3. National Academies (2013), *Assessment of the Prospects for Inertial Fusion Energy*. Mass-production goal $0.20--0.40/target.

Heavy-ion IFE:
4. Rickman, W.S. & Schultz, K.R. / Rickman, Goodin et al. (2003), "Indirect Drive Target Costing Studies and Materials Selection," General Atomics (ARIES meeting). ~40.8 cents/target (1000 MWe, on-site); 22 cents (3000 MWe, central); ~$0.11 bare capsule; hohlraum metal dominates and assumes single-use.
5. Callahan-Miller & Tabak, *Phys. Plasmas* 7 (2000) 2083 (distributed-radiator target physics); Moir et al., HYLIFE-II, *Fusion Technology* (1995) (plant context, ~5--6 Hz).

MagLIF / Z-pinch:
6. Goodin, D.T., et al. / Alexander, N.B. et al., "IFE Target Fabrication, Delivery, and Cost Estimates," FPA 2007; *Nuclear Fusion* 44 (2004). ZFE dynamic-hohlraum target load ~$2.86/target (1000 MWe, nth-of-a-kind), explicitly excluding the RTL.
7. Cipiti, B.B., Rochau, G.E., et al., "In-Zinerator," SAND2006-6590 (2006). RTL net remanufacturing cost: steel $3.10--5.40/RTL, cast tin $1.21/RTL; fragments recovered and remanufactured.
8. Olson, C.L., "Z-Pinch Power Plant Design," IAEA IFE CRP (2001). RTL ~$0.70/shot (robotic casting, mass-insensitive); ~0.1 Hz, ~3 GJ/shot.
9. Slutz, S.A., Olson, C.L., et al., "Recyclable Transmission Line Concept," SAND2003-4551 (2003). RTL mass basis and recycling economics.

MTF / PJMIF (thin cost literature; values inferred from design intent):
10. Laberge, M. (General Fusion, 2013), "Acoustically Driven Magnetized Target Fusion"; LINUS lineage -- recycled liquid-metal liner, ~1 Hz, no manufactured target.
11. Degnan, J.H. et al. (AFRL FRCHX) / Siemon, Intrator, Lindemuth -- solid-liner MTF, 6061-T6 aluminum liner (~0.1--1 kg), single-shot experimental; no published per-shot cost.
12. Hsu, S.C., Thio, Y.C.F., et al. (LANL / HyperJet), PJMIF status reports (OSTI 1506631; arXiv:2401.11066). Plasma liner formed in-situ by standoff guns, "produces no hardware debris"; per-shot consumable is sub-cent working gas.
