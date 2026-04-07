# Direct Energy Conversion and the Cost Floor

In the [previous dispatch](Lower%20bound%20for%20fusion%20energy.md), we found that the balance-of-plant — buildings, turbines, staffing, and financing — creates a hard floor on fusion electricity cost, even if the fusion core is free. For a p-B11 plant at 1 GWe, that floor is \$17/MWh with a supercritical CO2 Brayton cycle. The post ended with a promise: direct energy conversion could delete the floor entirely. This dispatch delivers on that promise — or rather, tempers it with numbers.

## What Direct Energy Conversion Does

A fusion plant with a thermal cycle converts fusion energy to heat, then heat to steam (or hot gas), then steam to spinning metal, then spinning metal to electricity. Each step has losses. A supercritical CO2 Brayton cycle converts at ~47%. A combined cycle reaches ~53%. These are mature technologies — but they come with turbine halls, cooling towers, condensers, and the buildings to house them. That hardware is the floor.

Direct energy conversion skips the thermal intermediary. Charged particles from fusion — alphas, protons — are decelerated and their kinetic energy is captured as electricity, electromagnetically or electrostatically. No turbine, no steam, no cooling towers. The question is whether this actually lowers the cost.

DEC only works on charged particles. Neutrons pass through any DEC system unaffected. Bremsstrahlung and synchrotron photons radiate to the walls as heat. This means DEC is only useful for fuels where most of the fusion energy comes out as charged particles: deuterium-helium-3 (~60% charged) and proton-boron (~99.8% charged). For deuterium-tritium (~20% charged), DEC is a marginal add-on that doesn't justify its cost. The design space for DEC is aneutronic fuel.

There is a further constraint. Even for p-B11, a significant fraction of fusion power is radiated as bremsstrahlung, driven by the high plasma temperature (>100 keV) and the Z=5 boron nucleus. The exact fraction depends on plasma conditions — [Ochs et al. (2022)](https://doi.org/10.1103/PhysRevE.106.055215) showed that in a thermonuclear p-B11 plasma at optimal conditions, bremsstrahlung consumes roughly **97% of fusion power** — the margin between fusion energy production and radiative loss is only ~3%. Alpha channeling (redirecting fusion product energy back into fuel ions) can widen this margin: channeling to fast protons raises the net power to ~17% of fusion power, but bremsstrahlung still dominates at ~83% of P_fus. The radiated fraction is not a minor correction — it is most of the energy. These photons are invisible to any DEC. They hit the walls and become heat. This heat must either be converted thermally (a small turbine) or rejected as waste. The DEC handles the charged particles; the bremsstrahlung sets a floor on the thermal plant that remains.

## Three Architectures

We analyzed three DEC approaches using the [1costingfe](https://github.com/1cfe/1costingfe) model, all on a p-B11 mirror with a free fusion core. All numbers in this post can be reproduced with the [companion script](https://github.com/1cfe/1costingfe/blob/master/examples/dec_blog_numbers.py).

### 1. Steady-State Mirror + Venetian Blind DEC

The [venetian blind](https://www.osti.gov/biblio/4563116) is the most mature DEC concept (TRL 4–5). Angled metal ribbon grids at successively higher retarding potentials sort ions by energy and collect them on high-potential electrodes — entirely passive, no moving parts. Demonstrated at 48% efficiency on real mirror plasma at LLNL's TMX (1982); theoretical maximum ~70%. [Realta Fusion](https://www.realtafusion.com/) is the sole active developer, with a [patented](https://patents.google.com/patent/US12166398B2) axisymmetric variant.

In a mirror machine, the venetian blind is an add-on to the end tanks that already exist for plasma confinement. The DEC hardware — grids, power conditioning, incremental vacuum and tank volume — costs [\$79–128M](https://github.com/1cfe/1costingfe/blob/master/docs/account_justification/CAS220109_direct_energy_converter.md) at ~400 MWe DEC electric output. This is comparable to the turbine island it partially replaces (~\$115M for sCO2 at 1 GWe).

When the DEC handles 90% of the charged particle transport and the remaining energy goes through a small thermal cycle:

| Configuration (1 GWe, pB11, free core) | Floor (\$/MWh) | Overnight (\$/kW) |
| --- | --- | --- |
| Thermal only (sCO2 Brayton, 47%) | 17 | 1,040 |
| VB DEC at 48% + thermal (hybrid) | 17 | 1,080 |
| VB DEC at 60% + thermal (hybrid) | 17 | 1,050 |
| VB DEC at 60%, no turbine (waste brem) | 17 | 1,030 |

The venetian blind barely moves the floor. At the demonstrated 48% efficiency, the DEC hardware (\$194M) costs more than the turbine (\$115M) it replaces. The efficiency gain is one percentage point over sCO2 Brayton — not enough to matter. Even at 60% (never demonstrated on real plasma), the savings are \$1/MWh. The DEC replaces the turbine in kind, not in cost.

### 2. Pulsed Inductive DEC

Helion Energy's approach is [architecturally different](https://doi.org/10.1088/0029-5515/51/5/053008). Two field-reversed configuration plasmoids collide and merge in a compression chamber surrounded by magnetic coils. The hot plasma expands against the confining field, pushing flux through the coils and inducing an EMF — Faraday's law, directly converting plasma kinetic energy to electrical current. The same coils that compress the plasma recover energy during expansion, like regenerative braking. No plasma-surface contact. No grids, no ribbons, no electrodes.

This eliminates the turbine entirely. The power conversion system is a grid-tie inverter (\$150/kW, [NREL benchmarks](https://www.nrel.gov/solar/market-research-analysis.html)) plus power conditioning. No cooling towers for the charged fraction — only for the bremsstrahlung waste heat.

| Configuration (1 GWe, pB11, free core) | Floor (\$/MWh) | Overnight (\$/kW) |
| --- | --- | --- |
| Thermal only (sCO2 Brayton, 47%) | 17 | 1,040 |
| Pulsed inductive (85% η, no turbine) | 16 | 930 |

Better than the venetian blind, but still only \$1/MWh below thermal. Better than the venetian blind — but the gap between all approaches is narrow. The reason: the turbine island is not the dominant cost. Buildings (\$242M), the electrical plant (\$96M), O&M (\$36M/yr), and financing charges together dwarf the \$115M turbine. Eliminating the turbine removes about 10% of overnight capital. The other 90% doesn't care what converts the heat.

The pulsed inductive approach has a separate, harder problem: the round-trip efficiency of the compression-expansion cycle is [claimed at ~95%](https://www.helionenergy.com/) but unpublished. The [theoretical framework](https://youtu.be/5nHmqk1cI2E?t=505) (Kirtley et al., APS DPP 2024) suggests 68–87% is more realistic for plasma-present operation, depending on the burn cycle. If the efficiency is 85%, the system works. If it's 70%, it still works but the advantage over a turbine narrows. If it drops below ~60%, the economics favor just building a turbine. Polaris — Helion's seventh device, under construction — is the make-or-break test.

### 3. MHD Generator

A [magnetohydrodynamic generator](https://doi.org/10.1016/B978-0-08-025566-5.50008-1) passes a conducting fluid through a transverse magnetic field, driving current through electrodes via the Lorentz force. No moving parts. The coal MHD programs of the 1960s–1990s demonstrated this at up to [32 MW gross output](https://www.osti.gov/biblio/6380343), and liquid-metal MHD (LMMHD) generators have shown [>71% efficiency](https://www.osti.gov/search/semantic:liquid%20metal%20MHD%20generator%20efficiency) at small scale.

MHD doesn't pair naturally with the directed charged-particle exhaust of a mirror — that's what the venetian blind is for. But it could replace the thermal cycle for the fraction of energy that ends up as heat: bremsstrahlung absorbed by walls, neutron energy captured in blankets, or thermalized charged particles. A liquid metal coolant flowing from the first wall through an MHD channel would convert heat to electricity at projected >60% efficiency — better than sCO2 Brayton — with no turbine.

The opportunity for p-B11 is a hybrid: DEC on the charged particles, MHD on the brem-heated coolant, no turbine at all. For D-T, LMMHD integrated into the lithium blanket converts both neutron and thermal energy, again without a turbine.

The obstacle is that no fusion-MHD system has ever been built. The coal MHD programs achieved TRL 5–6, but the killer problem — [electrode erosion from slag](https://www.gao.gov/products/emd-80-14) — doesn't apply to clean fusion coolants. The new critical challenge is MHD pressure drop: liquid metal flowing through the strong magnetic fields of a fusion device (5–15 T) experiences enormous drag, potentially consuming more pumping power than the MHD generates. This is an [active research problem](https://doi.org/10.3390/en14206640) at ORNL and KIT Karlsruhe, with no demonstrated solution at reactor scale. For the purposes of this analysis, MHD remains a what-if — promising physics, no engineering basis for costing.

## The Floor Doesn't Move Much

Here is the comparison at aggressive conditions — 2 GWe, 95% availability, 3% WACC, 50-year life, 3-year construction — where every lever is pulled:

| Approach (2 GWe, pB11, free core) | Floor (\$/MWh) | Overnight (\$/kW) | Budget for core |
| --- | --- | --- | --- |
| Thermal only (sCO2 Brayton, 47%) | 7.3 | 710 | +\$2.7/MWh |
| VB DEC 60% + thermal (hybrid) | 7.5 | 680 | +\$2.5/MWh |
| VB DEC 60%, no turbine | 7.4 | 660 | +\$2.6/MWh |
| Pulsed inductive (85%, no turbine) | 7.4 | 650 | +\$2.6/MWh |

At aggressive conditions, the spread between all approaches is **less than \$0.2/MWh**. The thermal floor is actually the cheapest because the venetian blind hardware (\$309M at 2 GWe) costs more than the turbine it replaces (\$218M). The power conversion hardware — whether a turbine, a venetian blind, or an inverter — is a small fraction of total cost at this scale. The dominant terms are buildings (\$370–490M), the electrical plant (\$180M), O&M (\$66M/yr), indirect costs, and financing. None of these care whether the electricity comes from spinning metal or decelerating ions.

The floor is not set by the power conversion choice. It is set by the industrial cost of building, staffing, and financing a large power plant.

## Where DEC Actually Helps

If DEC doesn't move the floor, why does it matter?

The floor is what you pay with a free fusion core. The core is not free. DEC's real benefit is that higher conversion efficiency means **you need a smaller core for the same net electric output**. A plant converting at 85% (pulsed inductive) needs roughly half the fusion power of a plant converting at 47% (sCO2 Brayton). Half the fusion power means fewer magnets, a smaller vacuum vessel, less heating power, and a proportionally cheaper core.

In the fully costed p-B11 mirror at 1 GWe baseline conditions:

| Configuration | LCOE (\$/MWh) | Core cost (CAS22) |
| --- | --- | --- |
| Thermal only (47%) | 36 | \$1,221M |
| VB DEC 60% (hybrid) | 33 | \$1,203M |

The core cost itself barely changes — the DEC hardware (\$197M) offsets part of the savings from smaller magnets and heating. The \$3/MWh LCOE improvement comes mainly from better utilization of the fusion power that is produced, not from a cheaper core.

For the pulsed inductive case at 85% efficiency, the cascade is larger: roughly half the fusion power, roughly half the core, minus the inverter cost. But this depends entirely on the unverified round-trip efficiency. If the efficiency is 70% instead of 85%, the core size savings are modest and the pulsed power system (capacitor banks, switches — [estimated at \$2–4/J stored](https://arxiv.org/abs/2602.19389) NOAK) may cost more than the turbine it replaces.

## The Bremsstrahlung Constraint

Every DEC architecture for p-B11 runs into the same wall: bremsstrahlung. At the plasma conditions required for p-B11 fusion (>100 keV, Z_eff elevated by the Z=5 boron nucleus), the vast majority of fusion power is radiated as X-ray photons that pass through any DEC system and deposit on the walls as heat. Even with alpha channeling — the most favorable scenario studied — bremsstrahlung is ~83% of fusion power. Without it, the margin between fusion and radiation is only ~3%.

This is a more severe constraint than the previous dispatch acknowledged. DEC does not handle the majority of the energy in a p-B11 plant — radiation does. The charged-particle transport that a venetian blind or pulsed system can capture is the net margin after bremsstrahlung, not the gross fusion power. The thermal plant is not a small bottoming cycle — it is the primary power conversion pathway.

The options for the radiated fraction are:

1. **Thermal cycle** — a full turbine island converts brem heat at 47–53%. This is what the "hybrid" configurations above assume. It works but preserves essentially the full thermal BOP.
2. **MHD on the coolant** — if the brem-heated wall coolant is liquid metal, an MHD channel could convert at >60% with no turbine. Undemonstrated at reactor scale.
3. **Reject as waste** — accept the efficiency loss and dump the brem heat to cooling towers. At 83% brem fraction, this means wasting most of the fusion energy. The overall plant efficiency collapses — a DEC at 85% efficiency on the 17% net charged fraction yields only ~14% of fusion power as electricity, far worse than thermal-only (47%).

Option 3 does not work for p-B11 at these bremsstrahlung fractions. The thermal plant is not optional — it handles most of the energy. DEC captures the margin.

## D-He3: Better Physics, Different Problem

Deuterium-helium-3 changes the bremsstrahlung picture. D-He3 operates at lower temperature (~50–60 keV vs >100 keV for p-B11) with lower effective charge (Z_eff ~1.3 vs ~2.5), so bremsstrahlung is a much smaller fraction of fusion power. The primary reaction produces a 14.7 MeV proton and a 3.6 MeV alpha — both charged particles, ~60% of total fusion energy. D-D side reactions contribute ~5% neutrons, but the bulk of the energy is available to DEC. This is why [Helion Energy](https://www.helionenergy.com/) chose D-He3 for their pulsed inductive architecture.

The BOP floors for D-He3 are comparable to p-B11 once fuel cost is separated out:

| Approach (1 GWe, D-He3, free core) | BOP floor (\$/MWh) | He-3 fuel (\$/MWh) | Total (\$/MWh) |
| --- | --- | --- | --- |
| Thermal only (sCO2 Brayton, 47%) | 18 | 79 | 97 |
| VB DEC 60% + thermal (hybrid) | 15 | 69 | 84 |
| Pulsed inductive (85%, no turbine) | 17 | 49 | 66 |

The BOP floors — \$15–18/MWh — are in the same range as p-B11. But the He-3 fuel cost dominates everything at \$49–79/MWh, five to eight times the 1-cent target by itself. At the [DOE-allocated price](https://www.everycrsreport.com/reports/R41419.html) of ~\$4.5M/kg, the fuel cost roughly doubles. Purchased He-3 makes 1-cent D-He3 impossible regardless of what the BOP costs.

Helion's answer is to breed He-3 internally from D-D side reactions, avoiding the need to purchase it. If the fuel cost drops to zero, the D-He3 BOP floor is \$15–18/MWh at baseline conditions and \$5–8/MWh at aggressive conditions — essentially the same as p-B11. The economics then hinge on (a) whether self-breeding produces enough He-3 to sustain the burn, and (b) whether the pulsed inductive DEC achieves 85%+ round-trip efficiency. Neither has been demonstrated.

The advantage of D-He3 over p-B11 for DEC is not the floor — it's the bremsstrahlung. With most of the energy in charged particles and a manageable radiated fraction, DEC can be the primary power conversion pathway, not just a margin-capture add-on. The pulsed inductive architecture eliminates the turbine entirely. Whether this translates to cheaper electricity depends on the fusion core, not the BOP.

## Conclusions

**1. DEC does not delete the floor.** At baseline conditions, the p-B11 cost floor moves from \$17/MWh (thermal) to \$16–17/MWh (DEC), depending on the approach. At aggressive conditions, all approaches converge to \$7–8/MWh. The floor is dominated by buildings, electrical systems, O&M, and financing — not the power conversion equipment.

**2. DEC's real value is efficiency, not BOP savings.** Higher conversion efficiency means a smaller fusion core for the same net output. This doesn't show up in a free-core analysis. It shows up in the fully costed plant, where the core is typically half of total LCOE.

**3. The venetian blind is an even trade.** At demonstrated efficiency (48%), the DEC hardware costs more than the turbine it replaces and converts at roughly the same efficiency as sCO2 Brayton. At 60%+ (undemonstrated on real plasma), it's modestly better. The venetian blind does not justify itself on BOP savings alone — it must deliver >60% sustained efficiency to beat the thermal path.

**4. The pulsed inductive approach is all-or-nothing.** If the round-trip efficiency is 85%+, the pulsed architecture halves the required fusion power and eliminates the turbine — a genuine transformation. If it's below 70%, the economics favor building a turbine. The gap between "transformative" and "not viable" is 15 percentage points of unverified efficiency.

**5. Aneutronic fuel is required, but not sufficient.** DEC only makes sense when most of the fusion energy is in charged particles. But p-B11 radiates 83–97% of its fusion power as bremsstrahlung. The net charged-particle energy that DEC can capture is the thin margin that remains — at most 17% of fusion power with alpha channeling, and only 3% without it. The turbine is not a small bottoming cycle; it handles most of the energy.

**6. D-He3 is the better DEC fuel — if you can get the He-3.** D-He3 has manageable bremsstrahlung and ~60% charged fraction, making DEC the primary power path rather than a margin add-on. But purchased He-3 costs \$49–79/MWh — far more than the entire BOP floor. Self-breeding eliminates this, but whether a D-He3 device can breed enough He-3 to sustain itself is undemonstrated.

**7. MHD could close the gap — someday.** An MHD generator on the brem-heated coolant could convert the radiated fraction at >60% efficiency with no moving parts, eliminating the last piece of the conventional thermal plant. The physics is demonstrated. The engineering for fusion environments is not.

The path to 1-cent fusion energy does not run through direct energy conversion alone. DEC is one lever — it raises the budget for the core by improving efficiency — but it does not remove the industrial cost structure that creates the floor. Reaching \$10/MWh still requires the same conditions identified in the previous dispatch: large plants, high availability, low-cost financing, fast construction, and long plant life. DEC makes those conditions slightly easier to achieve, not unnecessary.

## References

1. Moir, R.W. & Barr, W.L., "Venetian-blind direct energy converter for fusion reactors," *Nuclear Fusion* 13, 35–45 (1973). [Link](https://www.osti.gov/biblio/4563116)
2. Hoffman, M.A., "Electrostatic Direct Energy Converter Performance and Cost Scaling Laws," UCID-17560, LLNL (1977). [Link](https://www.osti.gov/biblio/7218298)
3. Slough, J. et al., "Creation of a high-temperature plasma through merging and compression of supersonic field reversed configuration plasmoids," *Nuclear Fusion* 51(5), 053008 (2011). [Link](https://doi.org/10.1088/0029-5515/51/5/053008)
4. Ochs, I.E. et al., "Improving the feasibility of economical proton-boron-11 fusion via alpha channeling with a hybrid fast and thermal proton scheme," *Physical Review E* 106, 055215 (2022). [Link](https://doi.org/10.1103/PhysRevE.106.055215)
5. Kirtley, D. et al., "Generalized burn cycle efficiency framework," APS DPP 2024, Abstract GO05.8. [Presentation](https://youtu.be/5nHmqk1cI2E?t=505)
6. CATF IWG, "Extension of the Fusion Power Plant Costing Standard," arXiv:2602.19389 (2026). [Link](https://arxiv.org/abs/2602.19389)
7. Rosa, R.J., *Magnetohydrodynamic Energy Conversion*, Hemisphere Publishing (1987). [Link](https://doi.org/10.1016/B978-0-08-025566-5.50008-1)
8. GAO, "Magnetohydrodynamics: A Promising Technology for Efficiently Generating Electricity From Coal," EMD-80-14 (1980). [Link](https://www.gao.gov/products/emd-80-14)
9. Shea, D.A. & Morgan, D., "The Helium-3 Shortage: Supply, Demand, and Options for Congress," Congressional Research Service, R41419 (2010). [Link](https://www.everycrsreport.com/reports/R41419.html)
10. Realta Fusion, US Patent 12,166,398 B2, "Axisymmetric ferromagnetic venetian blinds" (2025). [Link](https://patents.google.com/patent/US12166398B2)
11. 1cFE, "1costingfe: Open-source fusion techno-economic model." [GitHub](https://github.com/1cfe/1costingfe)
