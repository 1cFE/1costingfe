# CAS21: Buildings and Structures

**Date:** 2026-03-20
**Status:** Approved - Tal

## Overview

CAS21 covers all buildings and structures on the fusion plant site. The cost model prices each building individually based on its construction grade (industrial vs enhanced-industrial), physical scope (what equipment it houses), and fuel-dependent requirements (tritium containment, shielding, hot cell).

**Building totals:** DT = \$502/kW, DD = \$446/kW, DHe3 = \$339/kW, pB11 = \$308/kW (of gross electric).

**Key design principle:** Each building is priced for what it physically is, not scaled from a blanket multiplier. A pB11 plant has no hot cell — not half a hot cell. A DT plant has tritium confinement barriers in the reactor hall — but not a Part 50 containment structure.

## Construction Grade Framework

All fusion plants are regulated under 10 CFR Part 30 (byproduct materials), not Part 50 (reactor license). This means:
- No Seismic Category I requirements (use IBC/ASCE 7 industrial seismic)
- No NQA-1 QA program for most buildings (only tritium-containing systems for DT)
- No ACI 349 nuclear concrete (use ACI 318 commercial concrete)
- No fission-style containment structure (confinement approach: sealed building + negative pressure + filtered exhaust)

The relevant construction grades for fusion buildings are:

| Grade | Description | Typical $/sqft | When used |
|-------|-------------|---------------|-----------|
| Industrial | Pre-engineered steel, standard concrete, conventional HVAC | $100-250 | Most pB11 buildings, BOP for all fuels |
| Heavy industrial | Crane-served, reinforced concrete + steel, heavy foundations | $200-400 | Reactor hall, hot cell shell, assembly hall |
| Enhanced industrial | Industrial + selective upgrades for tritium or radiation | $300-500 | DT reactor hall, DT fuel storage, DT ventilation |

**Reference benchmarks ($/kW of gross electric):**

| Plant type | Buildings $/kW | Grade | Source |
|-----------|---------------|-------|--------|
| Gas CCGT | $150-250 | Industrial | Thunder Said Energy, EIA |
| Supercritical coal (NETL B12A) | $175-300 | Heavy industrial | DOE/NETL 2019 |
| Fusion pB11 (this model) | $308 | Industrial | Bottom-up, below |
| Fusion DHe3 (this model) | $339 | Light enhanced industrial | Bottom-up, below |
| Fusion DD (this model) | $446 | Enhanced industrial | Bottom-up, below |
| Fusion DT (this model) | $502 | Enhanced industrial | Bottom-up, below |
| Fission nuclear (Part 50) | $800-1,300 | Nuclear-grade | NREL ATB 2024 |

The fusion values form a gradient driven by neutron flux and tritium inventory:
- pB11 at $308/kW is 1.5x CCGT — industrial-grade with fusion-specific items (magnet hall, cryogenics, heavy crane)
- DHe3 at $339/kW is 1.7x CCGT — adds a small shielded maintenance area and light tritium monitoring
- DD at $446/kW is 2.2x CCGT — needs a reduced hot cell and moderate tritium handling
- DT at $502/kW is 2.5x CCGT — full hot cell, tritium confinement, shielding walls, rad-HVAC

## Source Documents

### Primary sources

1. **NETL Case B12A.** "Cost and Performance Baseline for Fossil Energy Plants," DOE/NETL-2015/1723, Rev 4, 2019. Account 13 (site improvements) and Account 14 (buildings). Supercritical coal, 550 MWe, 2019$.
2. **Cushman & Wakefield.** "Industrial Construction Cost Guide," 2025. Industrial building costs by type ($77-400/sqft).
3. **Cushman & Wakefield.** "Data Center Cost Guide," 2024. Data center costs $600-1,100/sqft.
4. **Breakthrough Institute.** "To Cut Nuclear Costs, Cut Concrete," 2022. Nuclear vs conventional concrete: 1.5x material, 33-105% longer installation, 23% QA overhead.
5. **Construction Physics.** "Why Are Nuclear Power Construction Costs So High?" Parts I & II, 2024. NQA-1 multipliers: 23% on concrete, 41% on steel, 5-50x on components.
6. **NREL ATB 2024.** Nuclear structures = 16.1% of ~$7,000/kW = ~$1,130/kW.
7. **SHINE Medical Technologies.** Isotope production facility, Janesville WI. ~$100M total for 91,000 sqft including hot cells. Part 30 licensed.
8. **ORNL SIPRC.** Stable Isotope Production & Research Center. $325M total, 64,000 sqft. DOE nuclear facility.

### Previous model sources (superseded)

9. **Waganer, L.M.** "ARIES Cost Account Documentation," UCSD-CER-13-01, 2013. Fusion heat island at $186.8/kW (nuclear-grade, superseded).
10. **pyFECONS.** cas21_buildings.py + CAS210000.tex. Mix of NETL B12A and ARIES values with undocumented modifications (superseded).

## Building-by-Building Cost Derivation

All values are reference costs at a 1 GWe net plant (~1.15 GW gross), in 2024\$, expressed as \$/kW of gross electric for comparability with literature. Each building scales with a different physical driver (see "Scales with" column in the Summary table) — not all scale with gross electric.

### Fuel-dependent buildings

**1. Site improvements** — DT: \$85M | DD: \$81M | DHe3: \$75M | pB11: \$69M

Scales with: site footprint (~fixed). Values at 1 GWe reference.

Scope: land clearing, grading, roads, parking, fencing, drainage, utility distribution, outdoor lighting, stormwater management.

- CCGT benchmark: \$50-100M for a 1 GWe site (NETL Account 13 for coal: ~\$25/kW in 2019\$, escalated)
- All fusion fuels are regulated under 10 CFR Part 30, which does not require a monitoring perimeter, protected-area fencing, or environmental monitoring stations. The proposed NRC fusion rule (91 FR 9476, Feb 2026) is performance-based — site infrastructure requirements are driven by the physical footprint of the plant, not by fuel-specific monitoring zones.
- DT (\$85M): larger site footprint than pB11 due to additional buildings (hot cell zone, tritium processing, rad-HVAC equipment pads). Emergency vehicle access routes for tritium emergency response. No monitoring perimeter or protected area — Part 30 does not require them.
- DD (\$81M): similar footprint to DT (hot cell, tritium handling areas) but slightly smaller tritium processing scope.
- DHe3 (\$75M): smaller footprint — no full hot cell, minimal tritium infrastructure. Standard industrial site with modest additional equipment pads.
- pB11 (\$69M): standard industrial site. Fence, roads, parking, drainage — like any large industrial facility.
- Previous value: \$268/kW (DT), \$134/kW (non-DT at 0.5x) — both substantially above benchmarks with no documented basis for the inflation

**2. Reactor building (fusion heat island)** — DT: \$138M | DD: \$127M | DHe3: \$98M | pB11: \$81M

Scales with: P\_fus (reactor size determines building volume). Values at 1 GWe reference (~2,300 MW fusion).

Scope: houses plasma vessel, magnets, blanket/shield, primary structure, in-vessel components. Includes crane hall, access corridors, equipment laydown.

- All fuels: heavy industrial crane hall (~30,000 sqft with 200+ ton overhead crane), reinforced concrete foundations for magnet loads, seismically designed per IBC (not Cat I)
- DT (\$138M): tritium primary confinement barriers on all penetrations, double-walled piping interfaces, glove-box connections, biological shielding walls (1-2m concrete near plasma for personnel access during shutdown), negative-pressure HVAC in reactor hall
- DD (\$127M): biological shielding needed (2.45 MeV neutrons from primary + 14.1 MeV from secondary DT), but less than DT (lower flux, ~1/3). Tritium confinement barriers needed but for smaller inventory.
- DHe3 (\$98M): light shielding (~5% neutron fraction — some activation exceeds occupational limits near the first wall, but far less than DD). Minor tritium barriers for DD-produced tritium. Closer to industrial than DT.
- pB11 (\$81M): standard heavy industrial building. No shielding walls (~0.1% of DT neutron flux), no tritium barriers, no double containment. Personnel walk-in access to all areas.
- Previous value: \$126/kW (DT), \$63/kW (non-DT at 0.5x)

**3. Hot cell** — DT: \$104M | DD: \$78M | DHe3: \$23M | pB11: \$0

Scales with: P\_fus (volume of activated components). Values at 1 GWe reference.

Scope: shielded zone for receiving activated components, remote disassembly, decontamination, size reduction, waste characterization, packaging. This is costed as a separate line item from the reactor building but is physically **integrated** within the same structure — not a separate building with a transfer corridor (the ITER configuration). The emerging power plant consensus (UKAEA STEP/PROCESS, ARC/CFS) is an integrated layout: a shielded bay adjacent to or beneath the reactor vessel, with downward or lateral extraction of activated components using the reactor building's crane system. This eliminates the shielded cask transport, transfer corridor, and duplicate crane systems of the ITER approach. The cost here covers the shielding walls, liner, remote handling equipment, and rad-HVAC for the hot cell zone — not a separate building envelope (that is in the reactor building line item).

For designs with demountable magnets (ARC concept), the entire vacuum vessel is replaced as a module — the hot cell scope shifts to a storage/processing bay for the removed vessel rather than in-situ component disassembly. This may reduce remote handling complexity but does not eliminate the need for shielded storage and waste processing.

- DT (\$90): full hot cell zone. 3-4 ft high-density concrete shielding walls, SS liner, leaded glass windows, remote manipulators, independent rad-HVAC with HEPA, fire suppression, Nuclear Quality Assurance, Level 1 (NQA-1) on tritium-wetted systems. First wall/blanket replaced every ~5 FPY at >10 Sv/hr contact dose. SHINE medical facility (Part 30, 91,000 sqft total) cost ~\$100M; fusion hot cell is smaller but more specialized.
- DD (\$68): reduced hot cell zone. Structural activation at ~7 dpa/yr requires component replacement on ~10 FPY cycles, but lower activation levels than DT mean components can be approached sooner after shutdown, simpler remote handling, and thinner shielding. NQA-1 on tritium-wetted systems (DD produces tritium via side reaction — smaller inventory than DT but same QA requirements on contaminated components). ~3/4 of DT scope — thinner shielding and simpler manipulators offset somewhat by the fixed NQA-1 program overhead.
- DHe3 (\$20): minimal shielded maintenance area. ~5% neutron fraction produces ~1 dpa/yr — components last 30+ FPY but some activation exceeds occupational limits for hands-on contact. A thick-walled room with manipulator arms for inspection and minor maintenance, not a full remote-handling hot cell.
- pB11 (\$0): **does not exist.** Near-zero neutron flux means no activation over the plant lifetime. All components maintained by contact (personnel with conventional tools). No shielded zone, no remote manipulators, no waste characterization.
- Previous value: \$93.4/kW (DT), \$46.7/kW (non-DT at 0.5x) — the 0.5x for pB11 was physically wrong
- References: [UKAEA PROCESS building model](https://ukaea.github.io/PROCESS/unique-models/buildings_sizes/), [ARC demountable magnet concept](https://www.sciencedirect.com/science/article/abs/pii/S0920379615302337), [STEP conceptual design](https://www.sciencedirect.com/science/article/pii/S0920379624000917)

**4. Fuel storage** — DT: \$9M | DD: \$7M | DHe3: \$3M | pB11: \$1M

Scales with: ~fixed (tritium inventory is set by processing loop, not plant power). Values at 1 GWe reference.

Scope: fuel isotope storage and handling.

- DT (\$9M): cryogenic tritium storage vessels (multi-barrier containment), deuterium gas bottles, tritium accountability vault, glovebox connections. Enhanced industrial construction for tritium areas.
- DD (\$7M): deuterium gas storage (main fuel) plus small-scale tritium handling. Side-reaction tritium inventory is grams — simpler storage than DT but still needs tritium-rated containment.
- DHe3 (\$3M): deuterium gas + He-3 gas storage. Minor tritium monitoring for DD side-reaction tritium. No breeding blanket tritium systems.
- pB11 (\$1M): boron-11 powder hopper/silo (commodity chemical storage), hydrogen gas bottle rack (standard compressed gas). Trivial industrial shed.
- Previous value: \$9.1/kW (DT), \$4.6/kW (non-DT at 0.5x)

**5. Reactor auxiliaries** — DT: \$29M | DD: \$25M | DHe3: \$21M | pB11: \$17M

Scales with: P\_fus (reactor size determines vacuum, gas, and diagnostics scope). Values at 1 GWe reference.

Scope: vacuum systems, gas handling, plasma diagnostics, water chemistry, heating system support equipment.

- DT (\$29M): some systems need radiation qualification (tritium-wetted vacuum pumps, contaminated gas handling, tritium-rated instruments)
- DD (\$25M): similar rad-qualified scope but smaller tritium-wetted area than DT
- DHe3 (\$21M): minor rad qualification — tritium-monitored vacuum only. Most systems are standard industrial.
- pB11 (\$17M): same equipment scope, fully industrial construction. No rad qualification needed on any system.
- Previous value: \$35/kW (all fuels, no fuel scaling) — pB11 was overpriced

**6. Control room** — DT: \$14M | DD: \$13M | DHe3: \$12M | pB11: \$12M

Scales with: ~fixed (building structure, not console count). Values at 1 GWe reference.

Scope: main control room, I&C equipment rooms, cable spreading room.

- The control room building cost is driven by the physical structure (reinforced room, cable trays, HVAC, fire suppression, UPS), not by the number of monitored systems. On-shift staffing (2-3 operators 24/7) is similar for any power plant regardless of fuel.
- DT (\$14M): adds a separate emergency control station for tritium release scenarios, and rad-hardened I&C in some areas.
- DD (\$13M): similar emergency control scope as DT, slightly reduced.
- DHe3 (\$12M) and pB11 (\$12M): standard industrial control room. No emergency control station, no rad-hardened I&C.
- Previous value: \$17/kW all fuels

**7. Security** — DT: \$3.5M | DD: \$3.5M | DHe3: \$2.3M | pB11: \$2.3M

Scales with: ~fixed (perimeter infrastructure). Values at 1 GWe reference.

Scope: perimeter security, access control, surveillance.

- Tritium is **not listed** in the 10 CFR Part 37 Category 1/Category 2 table (Appendix A). Part 37's physical protection requirements (armed response coordination, intrusion detection, vehicle barriers) do not apply to any fusion fuel type. Security for all fusion plants under Part 30 is standard industrial: perimeter fence, card access, cameras.
- Conventional power plant security is ~\$1-2M capital (fence + cameras + access control + guard service).
- DT and DD (\$3.5M): standard industrial security plus minor additions for tritium reporting requirements under 10 CFR 30.55 (incident reporting for >10 Ci tritium theft/diversion). This is a documentation requirement, not a physical protection requirement, but may warrant a slightly more robust access control system.
- DHe3 and pB11 (\$2.3M): standard industrial security. Fence, cameras, card access. No regulatory security requirements beyond normal industrial practice.
- Previous value: \$8/kW all fuels
- Note: the NRC's final fusion rule (due Dec 2027) could add tritium-specific security requirements. Current Part 30/37 framework does not require them.
- References: [Part 37 Appendix A — tritium not listed](https://www.nrc.gov/reading-rm/doc-collections/cfr/part037/part037-appa), [10 CFR 30.55 — tritium reports](https://www.law.cornell.edu/cfr/text/10/30.55)

**8. Ventilation/HVAC** — DT: \$17M | DD: \$15M | DHe3: \$6M | pB11: \$3.5M

Scales with: served floor area (driven by building sizes above). Values at 1 GWe reference.

- DT (\$17M): nuclear-grade HVAC for tritium zones — HEPA filtration banks, continuous air monitors (CAMs), stack monitoring, emergency isolation dampers, negative-pressure confinement zones. Rad-HVAC is 3-5x conventional (Breakthrough Institute).
- DD (\$15M): rad-HVAC needed — DD produces tritium via D(D,p)T branch, and any tritium-wetted surface requires HEPA, stack monitoring, negative pressure. Slightly smaller zones than DT due to lower tritium inventory, but still full rad-HVAC, not standard industrial.
- DHe3 (\$6M): enhanced industrial HVAC with stack monitoring for trace tritium. Not full HEPA banks — more like a monitored exhaust with a single-stage filter. Borderline between industrial and rad-HVAC.
- pB11 (\$3.5M): standard industrial HVAC. No radioactive effluent, no stack monitoring, no HEPA banks, no confinement zones. Conventional rooftop units.
- Previous value: \$9.2/kW all fuels (labeled "ventilation stack") — DT was too low, pB11 was too high

**9. Administration** — DT: \$9M | DD: \$8M | DHe3: \$6M | pB11: \$5M

Scales with: staff count. Values at 1 GWe reference.

- DT ~120 staff, DD ~94, DHe3 ~69, pB11 ~60 (or ~30 automated). Office space, training rooms, meeting rooms, locker rooms scale roughly with headcount.
- Previous value: \$10/kW all fuels

**10. Maintenance shops** — DT: \$17M | DD: \$16M | DHe3: \$15M | pB11: \$14M

Scales with: ~fixed + minor staff scaling. Values at 1 GWe reference.

- DT (\$17M): includes decontamination area, rad material staging space
- DD (\$16M): smaller decon area (lower activation levels)
- DHe3 (\$15M): minor decon capability
- pB11 (\$14M): standard industrial maintenance shop, no decon
- Previous value: \$25/kW all fuels

**11. Site services** — DT: \$5M | DD: \$5M | DHe3: \$3.5M | pB11: \$3.5M

Scales with: ~fixed. Values at 1 GWe reference.

- Warehouse, fire station, medical. Minor staff scaling.

### Fuel-independent buildings

These are the same cost for all fuels — they house equipment that doesn't depend on the fusion fuel cycle. Values at 1 GWe reference (~1.15 GW gross).

| Building | M\$ at ref | Scales with | Basis |
|----------|----------:|-------------|-------|
| Turbine building | \$58M | P\_the (thermal electric) | NETL B12A Account 14.3, pre-engineered steel. With DEC: scales with thermal electric only, not total gross. |
| Heat exchanger | \$17M | P\_th (thermal power) | Industrial building for coolant loops. Shrinks with DEC. |
| Power supply | \$17M | P\_et (gross electric) | Magnet power supplies, capacitor banks, DEC power conditioning. |
| Onsite AC power | \$12M | P\_et (gross electric) | Switchgear, diesel generators. |
| Cryogenics | \$14M | Magnet stored energy | LHe/LN2 plant for SC magnets. |
| Service water | \$9M | P\_th (thermal rejection) | Water treatment, circ water. |
| Assembly hall | \$21M | Component size (~fixed) | High-bay steel, reactor module pre-assembly. |

## Summary

| Building | DT | DD | DHe3 | pB11 | Scales with | Key driver |
|----------|---:|---:|-----:|-----:|-------------|------------|
| Site improvements | 74 | 70 | 65 | 60 | Site footprint (~fixed) | Building footprint differences |
| Reactor building | 120 | 110 | 85 | 70 | P\_fus (reactor size) | Shielding + tritium barriers |
| Hot cell | 90 | 68 | 20 | 0 | P\_fus (activated volume) | Activation level |
| Fuel storage | 8 | 6 | 3 | 1 | ~fixed | Tritium inventory |
| Turbine building | 50 | 50 | 50 | 50 | P\_the (thermal electric) | Steam cycle only, not DEC |
| Heat exchanger | 15 | 15 | 15 | 15 | P\_th (thermal power) | Coolant loop capacity |
| Power supply | 15 | 15 | 15 | 15 | P\_et (gross electric) | Magnet + DEC power conditioning |
| Onsite AC | 10 | 10 | 10 | 10 | P\_et (gross electric) | Switchgear capacity |
| Cryogenics | 12 | 12 | 12 | 12 | Magnet stored energy | SC magnet cooling load |
| Reactor auxiliaries | 25 | 22 | 18 | 15 | P\_fus (reactor size) | Rad qualification |
| Maintenance | 15 | 14 | 13 | 12 | ~fixed + staff | Decon area |
| Service water | 8 | 8 | 8 | 8 | P\_th (thermal rejection) | Cooling water treatment |
| Control room | 12 | 11 | 10 | 10 | ~fixed | Building structure, not consoles |
| Administration | 8 | 7 | 5 | 4 | Staff count | Office space |
| Site services | 4 | 4 | 3 | 3 | ~fixed | Warehouse, fire station |
| Security | 3 | 3 | 2 | 2 | ~fixed | Part 37 does not list tritium |
| Ventilation/HVAC | 15 | 13 | 5 | 3 | Served floor area | Rad-HVAC zones |
| Assembly hall | 18 | 18 | 18 | 18 | Component size | Pre-assembly space |
| **TOTAL** | **502** | **446** | **339** | **308** | | |

At 1 GWe (~1.15 GW gross): DT = \$577M, DD = \$513M, DHe3 = \$390M, pB11 = \$354M.

### DD rationale

DD produces ~1/3 the neutron flux of DT (2.45 MeV neutrons from primary reactions, plus 14.1 MeV from secondary DT at f\_T~0.97). Key building differences from DT:

- **Hot cell (\$78M vs \$104M):** Still needed — structural activation at ~7 dpa/yr requires component replacement on ~10 FPY cycles. But lower activation levels mean simpler remote handling, thinner shielding. NQA-1 still required on tritium-wetted systems (fixed overhead). ~3/4 of DT scope.
- **Reactor building (\$127M vs \$138M):** Needs biological shielding (2.45 MeV neutrons) but less than DT (14.1 MeV). Tritium confinement barriers needed but for smaller inventory.
- **Site improvements (\$81M vs \$85M):** Similar footprint to DT (hot cell, tritium processing areas). Slightly smaller tritium processing scope.
- **Ventilation/HVAC (\$15M vs \$17M):** Full rad-HVAC needed — DD produces tritium via D(D,p)T branch, and any tritium-wetted surface requires HEPA, stack monitoring, negative pressure. Slightly smaller zones than DT.
- **Security (\$3.5M, same as DT):** Part 37 does not list tritium. Standard industrial + minor access control for 10 CFR 30.55 reporting.

### DHe3 rationale

DHe3 produces ~5% of fusion energy as neutrons (from DD side reactions in the plasma). Minor tritium from D(D,p)T branch. Key building differences:

- **Hot cell (\$23M vs \$104M):** Minimal shielded maintenance area. ~1 dpa/yr means 30+ FPY component lifetime, but some activation exceeds occupational limits. A thick-walled room with manipulator arms, not a full remote-handling hot cell.
- **Reactor building (\$98M vs \$138M):** Light shielding (5% neutron fraction). Minor tritium barriers. Closer to industrial than DT.
- **Site improvements (\$75M vs \$85M):** Smaller footprint — no full hot cell, minimal tritium infrastructure means fewer equipment pads and buildings to connect.
- **Ventilation/HVAC (\$6M vs \$17M):** Enhanced industrial HVAC with stack monitoring for trace tritium. Not full HEPA banks.
- **Control room (\$12M vs \$14M):** Fewer safety-critical systems to monitor.
- **Security (\$2.3M vs \$3.5M):** Standard industrial. No tritium reporting requirements.

### Comparison with previous model

| Fuel | Old (\$/kW) | New (\$/kW) | New (M\$ at 1 GWe) | Change | Validation |
|------|-----------|-----------|-------------------|--------|------------|
| DT | 760 | 502 | \$577M | -34% | 2.5x CCGT (enhanced industrial for Part 30) |
| DD | 511 (0.5x) | 446 | \$513M | -13% | 2.2x CCGT (moderate rad scope) |
| DHe3 | 511 (0.5x) | 339 | \$390M | -34% | 1.7x CCGT (light rad scope) |
| pB11 | 511 (0.5x) | 308 | \$354M | -40% | 1.5x CCGT (industrial) |

The old model applied 0.5x to DD, DHe3, and pB11 identically. The new model correctly differentiates:
- DD needs most DT infrastructure at reduced scale (side-reaction tritium, lower neutron activation)
- DHe3 needs much less (minimal activation, trace tritium)
- pB11 needs none of the nuclear-specific infrastructure (zero hot cell, zero tritium, industrial HVAC)

Key corrections:
1. Site improvements: \$268/kW → \$60-100/kW (old value was 10x NETL coal, undocumented)
2. Hot cell: \$47/kW → \$0 for pB11 (was 0.5x, should be zero — nothing activates)
3. Hot cell: \$47/kW → \$20/kW for DHe3 (was same as DD, but DHe3 activation is 15x lower)
4. Ventilation: \$9/kW → \$3-15/kW (differentiated by fuel: rad-HVAC for DT, standard for pB11)
5. Security: \$8/kW → \$2-3/kW (Part 37 does not list tritium — no armed response required)
6. Many small buildings re-priced at industrial benchmarks rather than nuclear-adjacent

## References

- DOE/NETL, "Cost and Performance Baseline for Fossil Energy Plants," DOE/NETL-2015/1723, Rev 4, 2019.
- Cushman & Wakefield, "Industrial Construction Cost Guide," 2025.
- Cushman & Wakefield, "Data Center Cost Guide," 2024.
- Breakthrough Institute, "To Cut Nuclear Costs, Cut Concrete," 2022.
- Construction Physics, "Why Are Nuclear Power Construction Costs So High?" Parts I & II, 2024.
- NREL, "Annual Technology Baseline — Nuclear," 2024 edition.
- SHINE Medical Technologies, Janesville WI isotope facility (Part 30 licensed).
- ORNL SIPRC, Stable Isotope Production & Research Center, $88.8M construction contract.
- Waganer, L.M., "ARIES Cost Account Documentation," UCSD-CER-13-01, 2013 (superseded for building values).
- `CAS21_building_cost_research.md` — detailed benchmarking analysis.
