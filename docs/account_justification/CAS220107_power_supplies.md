# CAS220107: Power Supplies

**Date:** 2026-04-07
**Status:** Justified — two scaling modes. MFE mode calibrated to
vendor-quoted magnet power supply costs. Pulsed mode anchored to
\$/J stored energy basis from CATF IWG extension (arXiv:2602.19389),
with aggressive NOAK assumption for capacitor costs.

---

## Scope

Account C220107 covers the electrical power supply systems that
drive the fusion core: magnet power supplies, pulsed power drivers
(capacitor banks, switches, charging circuits, buswork), and
associated switchgear. It does not cover the grid-tie inverter
(C220109), building electrical systems (CAS24), or auxiliary plant
power.

The account has two modes, selected by confinement family:

1. **Steady-state MFE**: High-current DC supplies for
   superconducting magnets, plus pulsed power for heating
   systems. Cost scales with gross electric output.
2. **All pulsed concepts**: The capacitor bank / pulsed power
   system is the dominant electrical infrastructure cost.
   Cost scales with stored energy per pulse on a $/J basis.
   For inductive DEC concepts, the same bank also serves as
   the energy recovery system.

---

## Mode 1: Steady-State MFE

### Scaling formula

    C220107 = power_supplies_base * (p_et / 1000) ^ 0.7

where:
- `power_supplies_base` = 80.0 M\$ (at 1 GWe gross electric)
- `p_et` = gross electric output (MW)
- Exponent 0.7 reflects sub-linear scaling of vendor-purchased
  power electronics

### Cost basis

The \$80M reference at 1 GWe originates from the ARIES-CS
power plant design study, which estimated C220107 at \$70.6M
(2004\$) for a 1.3 GWe stellarator — approximately \$78/kW in
2024 dollars. This is a design study estimate, not a vendor
quote or procurement record.

Cross-checks against real-world industrial power electronics:

| Analog | Scale | \$/kW | Source |
|---|---|---|---|
| HVDC converter stations | GW | \$100–250/kW installed | Siemens ULTRANET, MISO MTEP24 |
| Solar grid-tie inverters | 100+ MW | \$30–50/kW | NREL 2024 benchmarks |
| Aluminum smelter rectifiers | 100+ MW DC | Not public | Closest regime (low V, high A) |

Fusion magnet supplies operate at lower voltage and higher
current than HVDC (1–10 kV, 10–100 kA vs 100+ kV), which is
generally the cheaper regime for power electronics. The \$80/kW
figure sits between the commodity floor (solar inverters) and
industrial FOAK (HVDC stations).

At typical parameters:

| P_gross | C220107 | \$/kW_gross |
|---|---|---|
| 50 MW | 7.8 M\$ | \$156/kW |
| 200 MW | 22 M\$ | \$110/kW |
| 1,000 MW | 80 M\$ | \$80/kW |

### Uncertainty

+-20%. The \$80M figure is a design study estimate (ARIES-CS),
not a vendor quote. No direct procurement data exists for
100+ MW DC power supply systems at fusion-relevant parameters
(low voltage, very high current). The HVDC and solar inverter
cross-checks bracket the figure but are not the same product
category. A vendor quote from an industrial rectifier
manufacturer (ABB, GE, Siemens) at the relevant specifications
would reduce this uncertainty.

---

## Mode 2: All Pulsed Concepts

### Why a different cost basis

For pulsed concepts (IFE, MIF, pulsed MCF), the capacitor bank
or pulsed power system is the dominant electrical infrastructure
cost — not a secondary system. The MFE scaling formula
(`power_supplies_base * p_et^0.7`) was calibrated for magnet
power supplies and substantially underestimates pulsed-power
costs, because it scales with output power rather than stored
energy. For inductive DEC concepts, the same bank also serves
as the energy recovery system.

The correct basis is \$/J of stored energy per pulse, as
recommended by the CATF IWG extension to the fusion costing
standard (arXiv:2602.19389):

> "For MIFE concepts, pulsed power is frequently the dominant
> capital driver ... The implementation is anchored to installed
> \$/J (stored energy) cost bases, explicit module counts, and
> lifetime/replacement logic driven by shot rate and component
> derating assumptions."

### Scaling formula

    C220107 = c_cap_allin_per_joule * e_stored_mj

where:
- `c_cap_allin_per_joule` = 0.50 \$/J stored (NOAK all-in)
- `e_stored_mj` = capacitor bank stored energy per pulse (MJ),
  computed by the physics layer from P_net, f_rep, Q_sci, and
  conversion efficiency

The formula is linear in stored energy because both cost and
power scale with E_stored x f_rep. The \$/kW_gross figure is
therefore insensitive to plant scale — the principal levers are
\$/J and Q_eng.

### Cost build-up: what "all-in" includes

The \$/J figure covers the complete installed pulsed power driver:

| Component | Fraction of all-in cost | Basis |
|---|---|---|
| Capacitors (BOPP dielectric or advanced) | 55–65% | Energy storage at rated voltage |
| Switch stacks (IGBTs, thyristors, or SiC) | 15–20% | Rated for E_stored discharge current |
| Charging power supplies | 8–12% | DC from grid to cap bank at f_rep rate |
| Buswork, transmission lines, dump circuits | 5–8% | Copper bus, coaxial connections |
| Diagnostics and baseline controls | 3–5% | Per-module instrumentation |

### The \$0.50/J assumption

The model uses \$0.50/J as an aggressive NOAK target. This
requires context:

| Price point | \$/J stored | Status |
|---|---|---|
| Lab-scale (2024) | \$20–50/J | Small-volume BOPP film caps, no volume manufacturing |
| Near-term FOAK | \$5–15/J | First plant, limited supply chain |
| CATF IWG recommendation | \$1.5–4/J | NOAK with mature supply chain |
| **1costingfe default** | **\$0.50/J** | **Aggressive NOAK — assumes advanced dielectric and high-volume manufacturing** |

The \$0.50/J figure assumes:

1. **Advanced dielectric materials** beyond current BOPP film —
   higher energy density reduces material volume per joule.
2. **High-volume manufacturing** at automotive or grid-storage
   scale — millions of units per year, not thousands.
3. **Mature supply chain** with multiple competing vendors.

This is a 40–100x reduction from today's lab prices. For
comparison, the solar PV industry achieved a 100x cost reduction
over roughly two decades of scaling (from \$30/W in the early
1980s to \$0.30/W today). The capacitor cost reduction required
for fusion pulsed power is comparable in magnitude, but the
starting technology (pulsed-power capacitors rated for 10^8
cycles) does not yet exist as a commercial product category.

**The \$0.50/J figure is a commercial viability requirement, not
a projection.** If capacitor costs remain above \$2–5/J, pulsed
MCF/MIF concepts face a structural cost disadvantage against
thermal conversion. Sensitivity analysis should sweep
c_cap_allin_per_joule from 0.5 to 5.0 to bracket this risk.

### E_stored derivation

In the physics layer, `e_stored_mj = e_driver_mj / eta_pin`,
where `eta_pin` is the driver wall-plug efficiency (fraction
of stored energy delivered to the plasma). The inverse power
balance solves for `e_driver_mj` from the top-level P_net
requirement, then divides by `eta_pin` to get stored energy.

The stored energy determines both the driver cost (C220107)
and the cap bank replacement schedule (CAS72). Near breakeven
(Q_eng < 1.2), small changes in efficiency produce large
changes in E_stored and therefore driver cost — a physically
real sensitivity that the model surfaces as a warning.

### Scheduled replacement (CAS72)

At f_rep = 1 Hz and 85% availability, the capacitor bank executes
approximately 27 million charge-discharge cycles per year.
Capacitor lifetime is modeled as a NOAK requirement:

    cap_shot_lifetime = 1.0e8 shots (default)

At this lifetime, replacement occurs every 3.7 years. The
replacement cost is captured in CAS72 as a present-value sum
of discrete replacements over the plant lifetime, discounted
at the project interest rate and annualized by the capital
recovery factor:

    t_replace = cap_shot_lifetime / (f_rep * 8760 * 3600 * availability)
    n_replacements = ceil(lifetime_yr / t_replace) - 1
    PV = sum over k=1..n_replacements of: C220107 / (1 + r)^(k * t_replace)
    CAS72_cap = PV * CRF

This O&M term is large and is the dominant source of LCOE
sensitivity for high-rep-rate pulsed plants.

**Lifetime uncertainty:** +-1 order of magnitude (1e7 to 1e9).
At 1e7 shots (12 days between replacements), the concept is
not viable as a power plant. At 1e9 shots (37 years), cap
replacement is a minor O&M cost. The 1e8 default is a
commercial viability requirement, like the \$/J target.

---

## Comparison: MFE vs Pulsed at representative scales

| Parameter | MFE (1 GWe tokamak) | Pulsed (1 GWe FRC) |
|---|---|---|
| C220107 | \$80M | Depends on E_stored |
| Scaling basis | p_et^0.7 | E_stored * \$/J |
| Dominant component | Magnet DC supplies | Capacitor bank |
| \$/kW_gross | \$80/kW | Architecture-dependent |
| CAS72 (replacement) | None | 27% of C220107/yr |

For a 1 GWe pulsed FRC with E_stored = 100 MJ at \$0.50/J:
C220107 = \$50M, comparable to the MFE scaling. At the CATF
recommendation of \$2/J, C220107 = \$200M — 2.5x the MFE value.
At today's lab prices (\$20/J), C220107 = \$2,000M — 25x.

---

## Comparison with pyFECONs

pyFECONs (cas220107_power_supplies.py) uses a single scaling
formula for all concepts:

    C220107 = power_supplies_base * (p_et / 1000) ^ 0.7

This is the MFE mode retained in 1costingfe. The pyFECONs
implementation does not distinguish pulsed from steady-state
concepts, does not scale with stored energy, and does not
model cap bank replacement. The \$/J basis and CAS72 cap
replacement logic are new to 1costingfe.

---

## References

- CATF IWG, "Extension of the Fusion Power Plant Costing
  Standard," arXiv:2602.19389 (2026). — \$/J basis, pulsed
  power account treatment, module counts and replacement logic.
- Kirtley, D. et al., "Generalized burn cycle efficiency
  framework," APS DPP 2024, Abstract GO05.8. — Burn cycle
  efficiency decomposition for pulsed inductive systems.
- Davis, S.W. et al. (Helion Energy), "Energy Recovery in
  Electrical Systems," US Patent Application 2024/0275198 A1
  (filed 2022, published 2024).
  [Link](https://patents.google.com/patent/US20240275198A1/)
  — Bidirectional pulsed power circuit for energy recovery at
  >90% per cycle. Directly relevant to the switch and circuit
  architecture that enables inductive DEC. Claims peak currents
  >1 MA, pulse durations 1–500 us, recovery 85–97%.
- "IMGs, Capacitors, and the Supply Chain Gap Facing Pulsed
  Power Fusion," The Fusion Report, March 2025. — Cap \$/J
  costs, 5–10x reduction requirement, lifetime targets.
- NREL, "Utility-Scale Solar Photovoltaics — 2024 Cost
  Benchmark." — Grid-tie inverter \$/kW benchmarks.
- Helion Energy, helionenergy.com — Polaris 50 MJ cap bank
  reference design point.
- Woodruff, S., "A Costing Framework for Fusion Power Plants,"
  arXiv:2601.21724 (2026). — pyFECONS methodology that
  1costingfe builds on.
