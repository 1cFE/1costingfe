# CAS220110: Remote Handling & Maintenance Equipment

**Date:** 2026-03-09
**Status:** Implemented

---

## Account Placement

CAS220110 fills the empty slot between 220109 (Direct Energy Converter) and 220111 (Installation Labor) in the ARIES/pyFECONs CAS22 hierarchy. pyFECONs does not currently have this account — remote handling costs are implicitly bundled into the hot cell building (CAS21) and installation labor (CAS220111). This account breaks them out explicitly because remote handling is:

1. A major DT-specific capital cost ($100-200M)
2. Significantly reduced but not negligible for aneutronic fuels
3. A key differentiator in the DT vs pB11 cost comparison

---

## What Remote Handling Covers

Remote handling (RH) refers to the robotic systems required to maintain, inspect, repair, and replace reactor internals that are too radioactive for human access. For a DT fusion plant, this includes:

### DT-Specific Remote Handling Systems

| System | Function | Reference |
|--------|----------|-----------|
| **In-vessel transporter** | Articulated robotic arms that enter the vacuum vessel through ports to detach, extract, and install blanket modules and first-wall panels. ITER's blanket RH system handles components up to 4.5 tonnes. | ITER BRHS |
| **Divertor cassette handler** | Specialized rail-mounted system to extract and insert divertor cassettes through equatorial ports. ITER plans up to 8 divertor replacements over 20 years. | ITER DRHS |
| **Cask & plug transfer** | Shielded casks that receive activated components at the vessel port and transport them to the hot cell. Must maintain containment of tritium-contaminated and activated components during transit. | ITER CPRHS |
| **Hot cell robotic systems** | Disassembly, inspection, and waste-processing robotics inside the hot cell. Includes remote cutting, welding, and characterization equipment. | UKAEA RACE |
| **In-pipe welding/cutting** | Robotic pipe welders for coolant connections that must be cut and re-welded during blanket replacement. UKAEA RACE (Culham) is developing these specifically for fusion. | UKAEA RACE |
| **Rad-hardened actuators** | All RH components must survive the neutron and gamma radiation environment. Rad-hardened motors, sensors, and electronics add significant cost over conventional robotics. | UKAEA Fusion Futures |
| **Tooling & end-effectors** | Specialized grippers, bolt runners, inspection cameras, leak-test probes. Each replaceable component type needs dedicated tooling. | General |

### Why DT Needs All This

14.1 MeV neutrons activate structural materials to the point where:
- Contact dose rates near the first wall reach ~10 Sv/hr within hours of shutdown
- Human access to the vacuum vessel interior is impossible for years after operation
- All maintenance of in-vessel components (blanket, divertor, first wall) must be fully remote
- Component transport must use shielded casks to protect building workers
- The hot cell must have its own robotic systems for component processing

### pB11: No Rad-Hardened RH, but Non-Trivial Maintenance Equipment

With <0.2% of fusion power in neutrons (~2 MeV, from side reactions):
- Structural activation is negligible (~0.1 dpa/yr vs ~20 dpa/yr for DT)
- Contact maintenance of all reactor internals is feasible after brief cooldown
- No shielded casks, no rad-hardened robotics, no remote cutting/welding
- Hot cell is not needed for component processing

However, pB11 still requires substantial maintenance equipment:
- **Vessel access tooling** — even without radiation, getting inside a sealed vacuum vessel through ports is mechanically challenging, especially for compact toroidal geometries (less so for FRC or mirror)
- **Heavy-lift capability** — first wall panels, divertor elements, and other internals are heavy regardless of activation status
- **Vessel opening/closing systems** — seals, flanges, alignment tooling for reassembly after maintenance
- **Confined-space equipment** — work platforms, lighting, ventilation for manned in-vessel access
- **Conventional welding/cutting** — coolant pipe reconnection, component attachment

The pB11 maintenance cost is concept-dependent: an FRC with a simple cylindrical vessel and easy access may need $10-15M in equipment, while a compact tokamak with narrow ports might need $25-40M for conventional but complex in-vessel tooling.

---

## Cost Estimates

### DT: $150M at 1 GWe reference

**Bottom-up basis:**

| Subsystem | Estimated Cost | Basis |
|-----------|---------------|-------|
| In-vessel transporter (2 units) | $40-60M | ITER BRHS is FOAK; commercial standardization reduces cost ~3x |
| Divertor cassette handler | $20-30M | Rail-mounted, simpler than blanket RH |
| Cask & plug transfer system | $15-25M | Shielded casks + automated transport |
| Hot cell robotic systems | $30-40M | Remote disassembly, inspection, waste processing |
| In-pipe welding/cutting robotics | $10-15M | UKAEA RACE-type systems, multiple units |
| Rad-hardened actuators & spares | $10-20M | Radiation-tolerant motors, sensors, electronics |
| Tooling & end-effectors | $5-10M | Component-specific grippers, probes |
| **Total** | **~$130-200M** | |
| **Central estimate** | **$150M** | |

**Cross-checks:**
- ITER's neutral beam RH system alone: ~EUR 70M contract (Amec Foster Wheeler / F4E)
- ITER has 4 major RH systems; total RH scope is a significant fraction of the ~$5B machine
- Commercial plant benefits from: standardization, learning curve, no FOAK penalty
- But commercial plant has larger components (1 GWe vs ITER's 500 MW thermal) and must handle full blanket replacement every 5-10 FPY

**Concept dependence:** The $150M estimate assumes a tokamak-class geometry (narrow ports, toroidal vessel). Simpler geometries reduce cost:

| Concept | DT RH Estimate | Rationale |
|---------|---------------|-----------|
| **Toroidal (tokamak, stellarator)** | $150M | Full suite through narrow ports; complex articulated robotic arms |
| **Linear (mirror, FRC)** | $70-100M | End-access simplifies transporter design; fewer port constraints |

**Power scaling:** `remote_handling_base * (p_et / 1000)^0.5` — sub-linear because RH systems are sized by port geometry and component count, not directly by power.

### DD: $100M

Lower activation (2.45 MeV neutrons, ~7 dpa/yr) but still too activated for contact maintenance:
- Still need full in-vessel RH capability (rad-hardened transporters, cask transfer)
- Simpler shielding on casks (2.45 MeV vs 14.1 MeV — less penetrating)
- Some components may allow semi-remote or delayed-access maintenance
- Rad-hardening requirements are less severe (lower gamma dose rates)
- Reduced vs DT because less shielding mass on casks and lower rad-hardening tier, not because of replacement frequency (frequency affects CAS72, not equipment capital)

### DHe3: $30M

~5% neutron fraction, ~1 dpa/yr. The D-D side reactions produce 2.45 MeV neutrons that activate structural materials. Even at 1 dpa/yr, first-wall activation will likely exceed occupational dose limits — if a human probably can't go in, rad safety says they don't go in. So DHe3 still needs remote handling, but:
- Lower activation means lighter shielding on casks and less demanding rad-hardening tier
- Neutron energy (2.45 MeV) produces less penetrating activation products than DT (14.1 MeV)
- RH equipment can be lighter-duty — less shielding mass, simpler containment
- Scope closer to DD than to pB11, but reduced because lower neutron flux means less secondary contamination and simpler decon

### pB11: $10-40M (concept-dependent)

No rad-hardened robotics needed, but non-trivial in-vessel maintenance tooling:

| Concept | Estimated Cost | Rationale |
|---------|---------------|-----------|
| **Linear (FRC, mirror)** | $10-15M | Simple cylindrical vessel, wide end-access, overhead crane suffices for most work |
| **Toroidal (tokamak, stellarator)** | $25-40M | Narrow ports, toroidal geometry constrains access; need articulated (but not rad-hardened) tooling, vessel opening systems |

**Central estimate: $20M** (weighted toward compact/linear concepts that pB11 proponents favor)

Common equipment across all pB11 concepts:
- Overhead cranes and heavy-lift rigging
- Personnel platforms and confined-space access
- Conventional welding and cutting equipment
- Vessel seal/flange tooling
- No radiation-specific requirements — conventional industrial grade

---

## Interaction with Other Accounts

| Account | Current Treatment | With C220110 |
|---------|-------------------|--------------|
| **CAS21 hot_cell** (93.4 $/kW) | Includes both building and equipment | Building only — RH equipment moves to C220110 |
| **CAS220111 installation** (14% of reactor subtotal) | Includes initial installation labor | Unchanged — C220110 is equipment capital, not labor |
| **CAS72 replacement** (annualized) | Component cost only | Should include RH operational cost per event (TBD — future enhancement) |
| **CAS71 O&M staffing** | RH operators included in maintenance staff | No concept factor in CAS70 (uniform staffing baseline). See `CAS70_staffing_and_om_costs.md`. |

**Important:** Adding C220110 does NOT double-count with the hot cell building cost. The hot cell building (CAS21) pays for the shielded structure. C220110 pays for the robotic equipment inside it and the in-vessel equipment that enters the tokamak.

---

## Availability Default

Default plant availability is 0.85 for all concepts (ARIES heritage).
No concept-specific availability premium is applied: no concept has a
sourced operational-availability basis to claim a higher figure (an
earlier mirror-only 0.87 default was removed as an unsourced,
mirror-only credit). Implemented in `default_availability(concept)` in
`src/costingfe/validation.py`; explicit `availability=...` kwargs
override the default.

---

## Sources

- [ITER Remote Handling](https://www.iter.org/machine/supporting-systems/remote-handling) — 4 major RH systems, components up to 45 tonnes
- [ITER Blanket RHS](https://www.sciencedirect.com/science/article/abs/pii/S0920379601002150) — 440 first walls, 2-year replacement campaign
- [ITER NB RH Contract (EUR 70M)](https://www.theengineer.co.uk/content/news/robotic-handling-system-to-maintain-key-components-of-iter-fusion-reactor) — Amec Foster Wheeler / F4E
- [UKAEA RACE Programme](https://www.ukaea.org/expertise/robotics/) — Fusion-specific remote maintenance R&D
- [UKAEA MASCOT System](https://www.eurekamagazine.co.uk/content/technology/re-tiling-a-fusion-reactor) — Haptic remote manipulator for in-vessel work
- [Maintenance Duration for DEMO](https://arxiv.org/abs/1412.4008) — Blanket replacement scheduling analysis
- [pyFECONs CAS22 Structure](https://github.com/Woodruff-Scientific-Ltd/PyFECONS) — Account 220110 is empty slot
- [Woodruff et al., "A Costing Framework for Fusion Power Plants" (2025)](https://arxiv.org/html/2601.21724) — RH listed as fusion-unique system
