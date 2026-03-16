# CAS22 Plant Systems: C220107–C220112 + C220200–C220700

**Date:** 2026-03-16
**Status:** Justified — values validated, C220109 formally excluded

## Overview

These accounts cover the remaining CAS22 sub-accounts beyond the core
reactor components (C220101–C220106).  They include a mix of:
- **Vendor-purchased systems**: power supplies, cryoplant, I&C
- **Custom-fabricated**: divertor cassettes
- **Site-constructed**: coolant piping, radwaste handling
- **Labor**: installation

At reference parameters (1 GWe DT tokamak):

| Account | Description | Cost (M$) | Method |
|---------|-------------|----------:|--------|
| C220107 | Power supplies | 89 | Power-scaled (0.7) |
| C220108 | Divertor / target factory | 96 | Power-scaled (0.5 MFE / 0.7 IFE) |
| C220109 | Direct energy converter | 0 | Excluded |
| C220110 | Remote handling | 162 | Fuel + concept dependent |
| C220111 | Installation labor | 286 | 14% of reactor subtotal |
| C220112 | Isotope separation | 0 | Zeroed (market purchase in CAS80) |
| C220200 | Main & secondary coolant | 200 | Power-scaled |
| C220300 | Auxiliary cooling + cryoplant | 14 | Power-scaled + cryo load |
| C220400 | Radioactive waste management | 5 | Thermal-scaled |
| C220500 | Fuel handling & storage | 120 | Fuel-dependent, power-scaled (0.7) |
| C220600 | Other reactor plant equipment | 12 | Power-scaled (0.8) |
| C220700 | Instrumentation & control | 69 | Thermal-scaled (0.65) |

---

## Per-Module Accounts

### C220107: Power Supplies

    C220107 = $80M × (P_et / 1000)^0.7

Covers high-current DC power supplies for magnets (TF, CS, PF coil
systems), pulsed power systems, and associated switchgear.  These are
**vendor-purchased** from specialized manufacturers (ABB, GE, Siemens).

ITER reference: ~1.6 GVA installed conversion capacity across 2 Magnet
Power Conversion buildings.  ITER procurement shared by Korea, China,
Russia.  Cost estimated at EUR 200–400M (FOAK, including buildings).
Our $89M at 1.17 GWe reflects NOAK pricing with standardized designs.

### C220108: Divertor (MFE) / Target Factory (IFE/MIF)

**MFE:**

    C220108 = $60M × (P_th / 1000)^0.5

Divertor cassettes (tungsten monoblock PFCs on CuCrZr heat sinks,
stainless steel body).  ITER: 60 cassettes, fabricated by Walter Tosto
and CNIM-SIMIC under F4E contracts.  ITER divertor procurement likely
EUR 100–300M.  NOAK with serial production: $60–100M at 1 GWe.

The 0.5 exponent reflects that divertor heat load scales sub-linearly
with total thermal power (divertor area scales with machine size, heat
flux concentration is a design parameter).

**IFE/MIF:**

    C220108 = $244M × (P_et / 1000)^0.7

Target factory for inertial/magneto-inertial concepts.  Higher cost
than MFE divertor due to high-repetition-rate target manufacturing
infrastructure.

### C220109: Direct Energy Converter

**Value: $0 (formally excluded from standard configurations).**

Direct energy conversion (DEC) applies only to mirror and FRC concepts
where charged-particle exhaust can be directly converted to electricity.
For the default tokamak/stellarator configurations, f_dec = 0 and this
account is zero.

DEC implementation for mirror/FRC concepts is deferred to concept-
specific extensions.  The user can override via `cost_overrides["C220109"]`
for custom configurations.

### C220110: Remote Handling & Maintenance Equipment

**Already justified.** See `docs/account_justification/CAS220110_remote_handling.md`.

Fuel-dependent (DT: $150M, DD: $100M, DHe3: $30M, pB11: $20M at 1 GWe)
and concept-dependent (0.55× for mirror geometry).

### C220111: Installation Labor

    C220111 = 14% × reactor_subtotal (C220101 through C220110)

Covers on-site labor for reactor plant installation: rigging, welding,
alignment, connection, testing, and commissioning of all reactor
components.

**Validation against industry norms:**
- Nuclear EPC: installation labor is typically 15–25% of equipment cost
  (including nuclear-grade QA/QC requirements)
- Conventional power plants: 10–15% of equipment cost
- Our 14% is appropriate for NOAK fusion plants with standardized
  modular construction (below nuclear, above conventional)

World Nuclear Association data: ~80% of overnight cost is EPC, with
~70% of EPC being direct costs (equipment + labor).  For a $2.8B
reactor plant, $286M installation labor (10% of total CAS22) is
consistent.

### C220112: Isotope Separation Plant

**Already justified.** See `docs/account_justification/CAS220112_isotope_separation.md`.

Zeroed — all isotope procurement is modeled as market purchase in CAS80
at enriched unit prices.

---

## Plant-Wide Accounts

These accounts are computed once for the entire plant (not per module)
using total plant power: `p_th_total = n_mod × p_th`.

### C220200: Main & Secondary Coolant

    C220201 = $166M × (P_net_total / 1000)   [primary coolant]
    C220202 = $40.6M × (P_th_total / 3500)^0.55  [intermediate coolant]
    C220200 = C220201 + C220202

Covers primary coolant loops (pumps, piping, heat exchangers),
intermediate heat transfer system, and secondary coolant to the steam
generators.  At reference: $200M.

Fission comparison: primary coolant systems for a 1 GWe PWR are
typically $150–250M.  Fusion coolant systems are more complex (higher
temperatures, larger volumes, potentially liquid metal or molten salt)
but use similar piping and pump technology.

### C220300: Auxiliary Cooling + Cryoplant

    C220301 = $1.1×10⁻³ × P_th_total   [auxiliary cooling]
    C220302 = $200M × (P_cryo / 30)^0.7  [cryoplant]

The cryoplant is calibrated to ITER data: ITER's cryoplant provides
75 kW at 4.5K using 3 helium refrigeration units (Air Liquide,
EUR 83M) plus 2 nitrogen refrigerators (EUR 65M) = ~EUR 148M total.
At 30 MW electrical cryo power, our model gives $200M — consistent
with ITER.

For the CATF reference tokamak (P_cryo = 0.5 MW): $6.8M.  Much smaller
than ITER due to compact HTS magnets with lower cryogenic load.

### C220400: Radioactive Waste Management

    C220400 = $1.96M × (P_th_total / 1000)

Covers radwaste processing, storage, and packaging systems.  Small
account ($5M at reference) because fusion produces low-level activated
waste, not high-level fission products.  Consistent with pyFECONS.

### C220500: Fuel Handling & Storage

    C220500 = fuel_handling_base(fuel) × (P_net_total / 1000)^0.7

| Fuel | Base (M$) | At 1 GWe | Scope |
|------|----------:|----------:|-------|
| DT | 120 | 120 | Full tritium processing: fuel injection, exhaust processing, isotope separation, tritium storage, accountability |
| DD | 60 | 60 | Small-scale tritium (from side reactions) + deuterium handling |
| DHe3 | 40 | 40 | He-3 recovery and recycling |
| pB11 | 15 | 15 | Boron powder/pellet injection, hydrogen supply |

DT fuel handling is the most expensive due to tritium's radioactivity,
permeation behavior, and regulatory requirements.

### C220600: Other Reactor Plant Equipment

    C220600 = $11.5M × (P_net_total / 1000)^0.8

Catch-all for miscellaneous reactor equipment not covered by other
accounts: fluid systems, maintenance equipment beyond remote handling,
tooling, fixtures.  Small account.

### C220700: Instrumentation & Control

    C220700 = $85M × (P_th_total / 3500)^0.65

Covers plant I&C: reactor control system, plasma diagnostics, safety
interlock systems, operator interfaces, data acquisition, and plant
computer systems.  At reference: $69M.

Nuclear I&C for fission plants typically costs $100–200M (including
safety-class redundancy).  Fusion I&C is complex (plasma control) but
doesn't require the same safety-class redundancy as fission.

---

## References

- ITER Organization, "Power Supply,"
  https://www.iter.org/machine/supporting-systems/power-supply
- ITER Organization, "Cryogenics,"
  https://www.iter.org/machine/supporting-systems/cryogenics
- ITER Organization, "EUR 83 million contract signed for liquid helium
  plant," Newsline, 2012.
- Fusion for Energy, "European prototypes for ITER Divertor Cassette
  completed," 2021.
- World Nuclear Association, "Economics of Nuclear Power," 2025.
- Waganer, L. M., "ARIES Cost Account Documentation," UCSD-CER-13-01,
  University of California San Diego, June 2013.
- Woodruff, S., "A Costing Framework for Fusion Power Plants,"
  arXiv:2601.21724, January 2026.
