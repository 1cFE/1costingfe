# CAS 71-73: Plant Staffing Estimate for a pB11 Fusion Power Plant

This cost account justification begins with reference data from the fission regulation, 10 CFR part 50. Even at the worse case scenario pB11 fusion would be regulated under 10 CFR part 30, which imposes a significantly lighter regulatory load.

## 1. Reference Data: Fission Plant Staffing

### S-PRISM (Sodium Fast Reactor) — Primary Reference
Source: Boardman et al., "Economic Assessment of S-PRISM Including Development and Generating Costs," ICONE-9, 2000. ([IAEA](https://inis.iaea.org/collection/NCLCollectionStore/_Public/33/020/33020128.pdf))

Plant: 1,520 MWe net, 4 reactor modules, 2 power blocks.

| Division | Headcount | Annual Cost (1996 $k) |
|---|---|---|
| Administration | 115 | 4,640 |
| Operating | 68 | 4,392 |
| Maintenance | 189 | 9,632 |
| Technical | 61 | 3,724 |
| **On-Site Subtotal** | **434** | **22,531** |
| Off-Site | 60 | 4,352 |
| Payroll Tax & Insurance @10% | — | 2,688 |
| **Total** | **494** | **29,571** |

### INL Scaling Analysis (2024)
Source: Prosser et al., "First-Principles Cost Estimation of a Sodium Fast Reactor Nuclear Plant," INL/RPT-23-74316 Rev 1, Jan 2024. ([INL Digital Library](https://inldigitallibrary.inl.gov/sites/sti/sti/Sort_67398.pdf))

Uses S-PRISM as baseline and scales by reactor count. Operations and maintenance staff scale with reactor quantity; admin/technical/offsite staff scale 50% fixed + 50% proportional to O&M headcount change.

| Division | 165 MWe (1 rx) | 311 MWe (1 rx) | 1,243 MWe (4 rx) | 3,108 MWe (10 rx) |
|---|---|---|---|---|
| Operations | 32 | 32 | 68 | 158 |
| Maintenance | 48 | 48 | 189 | 473 |
| Administration | 76 | 76 | 115 | 199 |
| Technical | 40 | 40 | 60 | 104 |
| Offsite | 40 | 40 | 60 | 104 |
| **Total** | **236** | **236** | **493** | **1,040** |

Average salaries (2023 USD): Operations $128k, Maintenance $101k, Admin $80k, Technical $121k, Offsite $143k. Benefits add ~$61k/employee.

### Key Fission-Specific Staffing Drivers
Fission plants carry substantial staffing overhead (~200+ of ~493 positions) due to radiation protection, radwaste management, armed security (30-50+ people), regulatory compliance, fuel handling, emergency preparedness, and nuclear QA. These burdens are concentrated in the Administration and Technical divisions and are largely a consequence of fission reactor regulation and radiological hazard — not inherent to power generation.

---

## 2. Reference Data: Conventional (Gas CCGT) Plant Staffing

| Plant Type | Capacity | Staff | Source |
|---|---|---|---|
| Gas CCGT | 565 MW | ~27 | Power Engineering |
| Gas CCGT | ~860 MW | ~33 | Power Engineering |
| Gas (DTE) | 1,100 MW | 35 | IEEE Spectrum |
| Simple Cycle (Elwood) | large | 15 | IEEE Spectrum |
| Coal | 300 MW | ~53 | EIA (1997) |
| Coal | 2,000 MW | 200-250 | Industry estimates |

Key insight: A modern ~1,000 MW CCGT plant operates with **25-35 staff**, roughly **1/15th** the staffing of a comparable nuclear plant. Coal plants fall in between at roughly 0.18 staff/MW.

The dramatic difference is driven by:
- No nuclear regulatory/security overhead
- High degree of automation in gas turbine controls
- Simpler maintenance (no radiation controls, no sodium/coolant chemistry)
- No fuel handling complexity
- No emergency preparedness requirements

---

## 3. Estimated Staffing for a pB11 Fusion Power Plant

### Regulatory Framework: 10 CFR Part 30

Per the NRC's Feb 2026 proposed rule, all fusion machines — including pB11 — are regulated under **10 CFR Part 30** (byproduct material). Even in the most pessimistic regulatory interpretation, this is the case: pB11 does not involve a self-sustaining fission chain reaction and produces only minor activation products from side-reaction neutrons. See `docs/analysis/pb11_side_reactions_nrc_regulatory_risk.md` for full analysis.

Part 30 compliance for pB11 involves byproduct material licensing, radiation safety for activated components, and associated recordkeeping — comparable to the regulatory burden on an industrial accelerator facility or a hospital with a cyclotron. A Radiation Safety Officer (RSO) and minimal radiation protection program suffices. There are no licensed operator requirements, no armed security force, no emergency planning zones, no resident inspectors — none of the staffing-intensive regulatory infrastructure associated with fission plants applies.

This is the single largest reason pB11 staffing is closer to a conventional plant than to fission. The fission reference data in Sections 1-2 above shows that regulatory and security overhead accounts for roughly half of fission plant staffing (~200+ of ~493 positions). Under Part 30, essentially all of that is eliminated.

### Remaining Complexity Beyond Conventional

A pB11 plant retains or adds complexity in other areas beyond a conventional plant:

| Area | Staffing Need |
|---|---|
| Fusion reactor systems (magnets, plasma heating, beam lines, vacuum) | Specialized maintenance and operations staff |
| Power conversion (direct energy conversion or steam cycle) | Similar to conventional, possibly simpler if direct conversion |
| Electrical systems, switchyard, grid connection | Same as conventional |
| Cooling systems | Same as conventional |
| Control systems / instrumentation | Comparable, possibly more complex plasma diagnostics |
| General plant (buildings, grounds, warehouse, admin) | Same as conventional |

### Proposed Staffing Estimate (~1,000 MWe pB11 Plant)

Under Part 30, the staffing comparison is not really an interpolation between fission and conventional — it's a conventional plant with additional complexity for fusion-specific systems. The fission numbers are provided as context for what is *not* required.

| Division | Fission Ref (1,243 MWe) | Conventional Ref (~1 GW) | pB11 Estimate (~1 GW) | Notes |
|---|---|---|---|---|
| Operations | 68 | ~8-10 | **12-16** | 24/7 control room coverage (4 shifts x 2-3 operators + supervision). Plasma operations more complex than a gas turbine. Plant-specific training, not NRC-licensed. |
| Maintenance | 189 | ~10-12 | **20-35** | Fusion-specific systems (magnets, vacuum, beam lines, plasma-facing components) need specialized maintenance beyond a CCGT. No rad-controlled work zones, no refueling outage surge. Industrial QA, not nuclear QA. |
| Administration | 115 | ~5-6 | **7-10** | Management, training, industrial safety, admin services, HR. RSO for Part 30 compliance (one person, possibly shared with Technical). No security force. |
| Technical | 61 | ~3-4 | **5-8** | Process/plasma engineering, controls/instrumentation, water chemistry (if steam cycle), industrial QA. Plasma diagnostics adds ~1-2 specialists vs conventional. |
| Offsite | 3-5 | ~2-3 | **3-5** | Minimal Part 30 reporting. Corporate engineering and fleet support. |
| **Total** | **493** | **~30** | **47-74** | |

**Central estimate: ~60 staff** for a ~1 GW pB11 fusion plant.

This is roughly **1/8th of a fission plant** and **~2x a gas CCGT plant** of comparable size. The premium over conventional is driven entirely by fusion-specific system complexity (vacuum, magnets/beams, plasma-facing components, plasma diagnostics), not regulatory burden. Under Part 30, regulatory staffing is essentially one RSO plus minimal recordkeeping. See `docs/analysis/pb11_side_reactions_nrc_regulatory_risk.md` for detailed regulatory analysis.

### Comparison: DHe3 Fusion Staffing (~1,000 MWe Plant)

DHe3 (deuterium-helium-3) produces ~5% of fusion energy as neutrons from DD side reactions in the plasma. This is far less than DT but not negligible — it produces real structural activation over a plant lifetime and small quantities of tritium from the D(D,p)T branch.

| Area | Impact vs pB11 |
|---|---|
| Neutron activation | ~50x more neutrons than pB11 (~0.05 of DT). Some rad-controlled maintenance zones needed. |
| Tritium (secondary) | Small amounts produced by DD side reactions. No breeding blanket, but tritium monitoring and leak detection needed. |
| Radiation protection | Modest HP program beyond RSO — area monitoring, some personal dosimetry for maintenance in activated areas. |
| Radwaste | Some activated components (first wall), but volume and activity far less than DT. |

| Division | pB11 Estimate | DHe3 Additions | DHe3 Estimate | Notes |
|---|---|---|---|---|
| Operations | 12-16 | +1-2 | **13-18** | Similar to pB11; slightly more radiation awareness protocols. |
| Maintenance | 20-35 | +3-8 | **23-43** | Some rad-controlled maintenance; minor decon procedures. No remote handling needed. |
| Administration | 7-10 | +1-3 | **8-13** | Small HP program (2-3 technicians beyond RSO). Tritium monitoring. |
| Technical | 5-8 | +1-2 | **6-10** | Minor radwaste characterization. Tritium accountability (small scale). |
| Offsite | 3-5 | +0-1 | **3-6** | Slightly more regulatory reporting than pB11. |
| **Total** | **47-74** | **+6-16** | **53-90** | |

**DHe3 central estimate: ~69 staff.**

### Comparison: DD Fusion Staffing (~1,000 MWe Plant)

DD (deuterium-deuterium) has two branches: D(D,n)He3 producing 2.45 MeV neutrons, and D(D,p)T producing tritium. If the produced tritium is burned in-situ (as most designs assume), secondary DT reactions add 14.1 MeV neutrons. Total neutron flux is roughly 1/3 of a pure DT plant.

| Area | Impact vs pB11 |
|---|---|
| Neutron activation | ~300x more neutrons than pB11 (~0.3 of DT). Significant structural activation. Component replacement on ~10 FPY cycles. |
| Tritium | Produced as byproduct. If burned: small in-plasma inventory, no breeding blanket. If extracted: tritium processing system needed. Either way, tritium handling and accountability required. |
| Radiation protection | Full HP program needed — dosimetry, area monitoring, ALARA for maintenance. Less intense than DT but same framework. |
| Radwaste | Activated structural components require characterization, packaging, disposal. Less volume than DT (lower flux, longer component life). |
| Remote handling | Some activated components may require remote handling, though less than DT (lower activation levels, longer replacement intervals). |

| Division | pB11 Estimate | DD Additions | DD Estimate | Notes |
|---|---|---|---|---|
| Operations | 12-16 | +3-6 | **15-22** | Tritium monitoring. Radiation protocols for operators. |
| Maintenance | 20-35 | +12-25 | **32-60** | Rad-controlled maintenance zones. Some remote handling. Tritium system maintenance. Longer outages for component replacement. |
| Administration | 7-10 | +3-8 | **10-18** | HP department (5-8 staff). Tritium accountability. EP planning. |
| Technical | 5-8 | +3-6 | **8-14** | Radwaste chemistry. Licensing/regulatory affairs. Activation analysis. |
| Offsite | 3-5 | +1-3 | **4-8** | More active NRC interface. Waste disposal coordination. |
| **Total** | **47-74** | **+22-48** | **69-122** | |

**DD central estimate: ~94 staff.**

### Comparison: DT Fusion Staffing (~1,000 MWe Plant)

Running DT fuel in the same or similar plant fundamentally changes the staffing picture. DT is still regulated under Part 30 per the Feb 2026 proposed rule — not Part 50 — so it avoids the full fission regulatory apparatus. But the practical operational and regulatory burden is far heavier than pB11:

**What DT adds vs pB11:**

| Area | Impact | Staffing Consequence |
|---|---|---|
| **14.1 MeV neutron flux** | 80% of fusion power carried by neutrons. ~10,000x more neutrons than pB11. Severe structural activation, requiring component replacement on multi-year cycles. | Rad-controlled maintenance zones. ALARA planning for every maintenance activity. Remote handling systems and hot cell operations. Adds health physics technicians, dosimetry, decontamination staff. |
| **Tritium breeding & processing** | DT requires an on-site tritium plant: breeding blankets (Li-based), tritium extraction, purification, storage, accountability, and fuel injection. Tritium inventory on the order of kg. | Dedicated tritium systems operations and maintenance staff. Tritium accountability officer. Hazardous material handling and leak detection. Estimated 10-20 additional staff. |
| **Radwaste management** | Activated structural components (first wall, blanket modules, divertor) become intermediate-level waste. Tritium-contaminated components require special handling. | Radwaste technicians, waste characterization, packaging, shipping, disposal coordination. Estimated 5-10 additional staff. |
| **Radiation protection program** | Full health physics program required — not just an RSO. Personal dosimetry for all workers in controlled areas, area monitoring, bioassay for tritium exposure, contamination surveys. | Health physics department: HP manager, HP technicians (2-4 per shift for 24/7 coverage), dosimetry specialist. Estimated 10-15 total HP staff. |
| **Part 30 regulatory burden (risk-informed)** | Much heavier than pB11. Detailed safety analysis for tritium release scenarios. More extensive NRC reporting, inspections, and compliance. Possible tritium-specific requirements in final rule. | Licensing/regulatory affairs staff (2-3), compliance documentation, NRC inspection support. |
| **Maintenance in activated environments** | First wall and blanket module replacement requires remote handling equipment, hot cells, shielded transport. Maintenance tasks take longer due to radiation controls and remote tooling. | Larger maintenance division. More specialized crafts (remote handling operators, hot cell technicians). Longer outage durations require surge staffing. |
| **Emergency preparedness** | Tritium release scenarios (kg-scale inventory) likely require site emergency plans, drills, and coordination with local authorities — beyond what pB11 needs, though far less than fission EPZ requirements. | Dedicated EP coordinator (1-2 staff), periodic drill costs. |
| **Security** | Tritium is not SNM; no armed security force required. Tritium is a controlled material with modest security considerations (dual-use concerns). | Industrial security measures. Minimal staffing impact (0-2). |

**DT Staffing Estimate (~1,000 MWe):**

| Division | pB11 Estimate | DT Additions | DT Estimate | Notes |
|---|---|---|---|---|
| Operations | 12-16 | +5-10 | **17-26** | Tritium plant operations (breeding, extraction, purification). Radiation protocols for control room and field operators. More complex startup/shutdown procedures. |
| Maintenance | 20-35 | +20-40 | **40-75** | Remote handling operations for activated components. Hot cell maintenance. Tritium system maintenance. All maintenance in rad-controlled areas requires ALARA planning, HP coverage, and decon. Blanket module replacement outages. |
| Administration | 7-10 | +5-15 | **12-25** | Health physics department (10-15 staff for 24/7 rad protection coverage). EP coordinator. Possible modest security additions. No armed force. |
| Technical | 5-8 | +5-10 | **10-18** | Tritium accountability. Radwaste characterization and chemistry. Licensing/regulatory affairs for Part 30 compliance. Neutronics/activation analysis. |
| Offsite | 3-5 | +2-5 | **5-10** | NRC regulatory interface (more active than pB11). Waste disposal coordination. Corporate tritium program support. |
| **Total** | **47-74** | **+37-80** | **84-154** | |

**DT central estimate: ~120 staff** for a ~1 GW DT fusion plant.

This is roughly **2x the pB11 estimate** and about **1/4 of a fission plant**. DT is still Part 30 and avoids fission's largest staffing drivers (armed security force, licensed operators, EPZ organization). But it picks up substantial staffing from:
- Tritium handling infrastructure (~10-20 staff)
- Health physics for a high-neutron environment (~10-15 staff)
- Radwaste management (~5-10 staff)
- Radiation-controlled maintenance (longer tasks, more personnel per task, HP coverage requirements)

**Key insight:** The staffing gap between pB11 and DT (~60 vs ~120) is almost entirely driven by neutron-related consequences — radiation protection, activated component handling, tritium systems, and radwaste. This is the operational cost of the ~10,000x difference in neutron production. The fusion-specific complexity (magnets, vacuum, plasma control) is roughly the same for both fuels.

---

## 4. Potential for AI-Driven Staffing Reduction

### Where AI Can Reduce Staffing

| Function | Current Role | AI Replacement Potential | Estimated Reduction |
|---|---|---|---|
| **Predictive maintenance** | Maintenance planners, reliability engineers analyze equipment data manually | AI sensor analytics predict failures, auto-generate work orders; demonstrated 25-30% maintenance cost reduction (NextEra Energy) and 70-75% fewer breakdowns | 20-30% of maintenance staff |
| **Automated inspection** | Visual inspections by maintenance technicians, scheduled rounds | Drones, cameras, and computer vision for routine inspections; vibration/thermal monitoring replaces human rounds | 10-20% of maintenance staff |
| **Control room operations** | 24/7 operator coverage, manual monitoring of plant parameters | AI-assisted monitoring reduces cognitive load; advanced automation allows fewer operators per shift (already demonstrated in gas plants) | 1-2 fewer operators per shift (25-30% of ops) |
| **Plasma control & optimization** | Plasma physicists / operators tune machine parameters | ML-based plasma control (already demonstrated in tokamak experiments); AI can optimize burn parameters in real time | Reduces need for dedicated plasma specialists |
| **Administrative / compliance** | Document management, reporting, training administration | AI document generation, automated compliance reporting, AI-assisted training programs | 20-30% of admin staff |
| **Technical / engineering** | Process engineering, data analysis, chemistry monitoring | AI-driven process optimization, automated water chemistry control, AI-assisted engineering analysis | 15-25% of technical staff |
| **Remote operations** | Some staff needed on-site for physical presence | Remote monitoring centers can serve multiple plants; a single operations center could oversee multiple fusion plants | Enables offsite consolidation |

### AI-Reduced Staffing Estimate

| Division | Base Estimate | AI Reduction | AI-Reduced Estimate |
|---|---|---|---|
| Operations | 12-16 | 25-30% | **9-12** |
| Maintenance | 20-35 | 25-35% | **13-25** |
| Administration | 7-10 | 25-30% | **5-7** |
| Technical | 5-8 | 20-25% | **4-6** |
| Offsite | 3-5 | 30-40% | **2-3** |
| **Total** | **47-74** | | **33-53** |

**AI-reduced central estimate: ~42 staff** for a ~1 GW pB11 fusion plant.

### Additional AI Considerations

1. **Multi-plant operations centers**: If a fleet of fusion plants is deployed, a centralized AI-assisted operations center could provide remote monitoring and engineering support for multiple sites, reducing per-plant offsite and technical staff further.

2. **Autonomous maintenance scheduling**: AI can optimize maintenance windows and workforce allocation, reducing the need for dedicated planning staff and enabling a leaner, more flexible maintenance team.

3. **Digital twin technology**: A real-time digital twin of the fusion plant enables AI to simulate scenarios, predict failures, and optimize operations without dedicated engineering staff for each analysis.

4. **Regulatory simplification feedback loop**: If AI-driven safety monitoring demonstrates consistently safe operation, regulators may further simplify oversight requirements, reducing compliance staffing.

5. **New roles created**: Data scientists, ML engineers, and robotics/drone specialists may be needed — but these can be shared across a fleet rather than dedicated per-plant. Estimated 2-5 per plant or fewer in a fleet model.

---

## 5. Salary Estimates and Annual O&M Cost Build-Up

### Salary Assumptions (2023 USD)

Source: INL/RPT-23-74316 Table 22 (salaries) and BLS March 2023 (benefits).

| Division | Annual Salary | Benefits | Total Compensation |
|---|---|---|---|
| Operations | $128,000 | $61,000 | $189,000 |
| Maintenance | $101,000 | $61,000 | $162,000 |
| Administration | $80,000 | $61,000 | $141,000 |
| Technical | $121,000 | $61,000 | $182,000 |
| Offsite | $143,000 | $61,000 | $204,000 |

Benefits include: paid leave ($6.88/hr), supplemental pay ($2.59/hr), insurance ($7.27/hr), retirement/savings ($7.54/hr), legally required benefits ($5.27/hr) = $29.55/hr × 2,080 hr/yr ≈ $61k/yr.

### Annual O&M Cost Build-Up at 1 GWe Reference (2023 USD)

| Cost Component | pB11 | DHe3 | DD | DT |
|---|---|---|---|---|
| Staff (headcount) | 59 | 69 | 94 | 117 |
| Labor + benefits | $10.1M | $11.7M | $15.9M | $19.7M |
| Maintenance materials (% of labor) | $6.0M (60%) | $6.4M (55%) | $11.1M (70%) | $15.8M (80%) |
| Insurance (property + liability) | $3.0M | $3.0M | $4.0M | $5.0M |
| Supplies & consumables | $2.5M | $2.5M | $3.0M | $4.0M |
| Regulatory fees (Part 30) | $0.3M | $0.3M | $0.7M | $1.0M |
| Waste treatment/disposal | $0.2M | $0.5M | $2.0M | $4.0M |
| General admin overhead | $1.5M | $1.5M | $2.0M | $2.5M |
| **Total annual O&M** | **$23.6M** | **$26.0M** | **$38.7M** | **$52.0M** |
| **O&M per MW/yr** | **$24k** | **$26k** | **$39k** | **$52k** |

Notes on non-labor components:
- **Maintenance materials** scale with neutron damage — DT requires more frequent replacement of activated components and specialized remote-handling consumables. pB11/DHe3 use conventional industrial materials.
- **Insurance** is property and general liability only — no Price-Anderson nuclear insurance required under Part 30. DT is higher due to tritium inventory risk.
- **Waste treatment** ranges from near-zero (pB11, minimal activation) to $4M/yr (DT, activated first wall/blanket modules, tritium-contaminated components).
- **Regulatory fees** under Part 30 are modest for all fuels. DT is highest due to more extensive NRC oversight of tritium operations.

### Comparison with Reference Data

| Plant Type | O&M (M$/yr) | O&M (k$/MW/yr) |
|---|---|---|
| S-PRISM fission (1,520 MWe) | $75.9M | $50k |
| Gas CCGT (~1 GWe) | ~$12M | ~$12k |
| pB11 fusion (1 GWe, est.) | $23.6M | $24k |
| DHe3 fusion (1 GWe, est.) | $26.0M | $26k |
| DD fusion (1 GWe, est.) | $38.7M | $39k |
| DT fusion (1 GWe, est.) | $52.0M | $52k |

The pB11 estimate ($24k/MW/yr) is ~2x gas CCGT and ~half of fission — consistent with the staffing analysis showing ~2x CCGT headcount and Part 30 regulation eliminating nuclear-specific overhead. DT ($52k/MW/yr) approaches fission O&M levels due to neutron-related operational costs, despite avoiding the fission security and regulatory apparatus.

### Values for Cost Model

The following fuel-specific O&M coefficients are at a 1 GWe reference point:

| Parameter | Value (M$/yr at 1 GWe) | Justification |
|---|---|---|
| `om_cost_dt` | 52.0 | Full neutron + tritium operational overhead |
| `om_cost_dd` | 39.0 | Reduced neutron flux (~1/3 DT), smaller tritium inventory |
| `om_cost_dhe3` | 26.0 | ~5% neutron fraction, minimal tritium, light HP program |
| `om_cost_pb11` | 24.0 | Aneutronic, no tritium, RSO-only rad protection |

### Power-Law Scaling with Plant Size

Staffing does not scale linearly with plant capacity. The INL SFR data (Sort_67398) shows significant economy of scale:

| Plant size | Staff | Staff/GWe |
|-----------|-------|-----------|
| 165 MWe (1 rx) | 236 | 1,430 |
| 311 MWe (1 rx) | 236 | 759 |
| 1,243 MWe (4 rx) | 493 | 397 |
| 3,108 MWe (10 rx) | 1,040 | 335 |

INL models this as: operations and maintenance staff scale with reactor count, while administration, technical, and offsite staff are 50% fixed + 50% proportional to the O&M headcount change. Fitting a power law (staff ∝ P^α) to the endpoints (165 → 3,108 MWe) gives **α ≈ 0.5**.

The cost model uses this scaling:

```
annual_om = om_cost(fuel) * (P_net / 1 GWe)^0.5
```

No concept-dependent factor is applied. Health physics, tritium
accountability, engineering staff, and security dominate the
fuel-baseline FTE count and are fuel-driven rather than geometry-driven,
and no concept has a sourced maintenance-ergonomics basis to claim an
O&M discount. (An earlier mirror-only 0.85 factor was removed as an
unsourced, inconsistently-applied credit: it singled out the mirror
among the non-toroidal concepts, none of which share it.) Geometry-
specific maintenance access is captured where it has a clearer hardware
basis: the C220110 remote-handling capex carries a 0.55x discount for
all non-toroidal concepts (see `CAS220110_remote_handling.md`).

This produces the expected economy-of-scale behavior — smaller plants
have higher per-MW O&M costs due to the fixed staffing component:

| Plant size | DT annual O&M | Effective k$/MW/yr |
|-----------|--------------|-------------------|
| 200 MWe | $23.3M/yr | $116 |
| 500 MWe | $36.8M/yr | $74 |
| 1,000 MWe | $52.0M/yr | $52 |
| 2,000 MWe | $73.5M/yr | $37 |

The same exponent is used for CAS40 (capitalized owner's costs), which is also staffing-driven. See `CAS40_capitalized_owners_costs.md`.

---

## 6. Summary Comparison

| Plant Type | Capacity | Total Staff | Staff/GW | O&M (k$/MW/yr) |
|---|---|---|---|---|
| Nuclear SFR (S-PRISM) | 1,243 MWe | 493 | ~397 | $50k |
| Nuclear SFR (S-PRISM) | 1,520 MWe | 494 | ~325 | $50k |
| **DT Fusion (est.)** | **~1,000 MWe** | **~117** | **~117** | **$52k** |
| **DD Fusion (est.)** | **~1,000 MWe** | **~94** | **~94** | **$39k** |
| Coal (historical) | 300 MW | 53 | ~177 | — |
| Coal (large) | 2,000 MW | 225 | ~113 | — |
| **DHe3 Fusion (est.)** | **~1,000 MWe** | **~69** | **~69** | **$26k** |
| **pB11 Fusion (est.)** | **~1,000 MWe** | **~59** | **~59** | **$24k** |
| Gas CCGT | 565 MW | 27 | ~48 | ~$12k |
| **pB11 Fusion + AI (est.)** | **~1,000 MWe** | **~42** | **~42** | — |
| Gas CCGT | 1,100 MW | 35 | ~32 | ~$12k |

The fusion estimates form a clear gradient driven by neutron production: pB11 ($24k/MW/yr, ~2x CCGT) → DHe3 ($26k/MW/yr) → DD ($39k/MW/yr) → DT ($52k/MW/yr, approaching fission). All fusion plants benefit from Part 30 regulation, but DT's radiological reality (full neutron flux, tritium handling, activated component management) adds substantial practical staffing and O&M cost regardless of regulatory framework.

---

## 6. Sources

1. Boardman, C. E. et al. "Economic Assessment of S-PRISM Including Development and Generating Costs." ICONE-9, 2000.
2. Prosser, J. H. et al. "First-Principles Cost Estimation of a Sodium Fast Reactor Nuclear Plant." INL/RPT-23-74316 Rev 1, January 2024.
3. IEEE Spectrum, "Automation Is Engineering the Jobs Out of Power Plants," 2017.
4. Power Engineering, "An Inside Look at Gas-Fired O&M."
5. NRC 10 CFR 50.54(m) — Licensed Operator Staffing Requirements.
6. IAEA, "Staffing of Nuclear Power Plants and the Recruitment, Training and Authorization of Operating Personnel," 1991.
7. US Bureau of Labor Statistics, Employer Costs for Employee Compensation, March 2023.
8. US EPA, "Methodology for Power Sector-Specific Employment Analysis," 2023.
