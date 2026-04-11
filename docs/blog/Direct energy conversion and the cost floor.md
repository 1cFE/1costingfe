# Direct Energy Conversion and the Cost Floor

In the [previous dispatch](https://1cf.energy/fusions-cost-floor-what-if-the-core-were-free/), we found that the balance-of-plant (buildings, turbines, staffing, and financing) creates a hard floor on fusion electricity cost, even if the fusion core is free. For a p-B11 plant at 1 GWe, that floor is \$18/MWh with a supercritical CO2 Brayton cycle. The post ended with a promise: direct energy conversion (DEC) could circumvent the floor entirely. This dispatch delivers on that promise, or rather, tempers it with numbers.

## What Direct Energy Conversion Does

A fusion plant with a thermal cycle converts fusion heat to spinning turbine to electricity. A supercritical CO2 Brayton cycle converts at 47% efficiency; a combined cycle reaches 53%. These are mature technologies with known capital costs: turbines, generators, cooling towers, condensers, and the buildings to house them. That hardware is the floor.

Direct energy conversion skips the thermal intermediary. Charged fusion products (alphas, protons) are decelerated and their kinetic energy captured as electricity. The turbines and cooling towers are replaced by DEC hardware. The question is whether this lowers the cost.

DEC only works on charged particles. Neutrons pass through unaffected, and photons can only push electrons via the photoelectric effect, which works at 1 eV, not the keV energies of fusion bremsstrahlung. DEC therefore requires fuel where most of the energy is in charged particles: deuterium-helium-3 (5% of fusion energy in neutrons) or proton-boron (0.2% neutrons). Deuterium-tritium (80% neutrons) is not a DEC fuel.

Photons are a separate problem. For p-B11, bremsstrahlung is severe: the high plasma temperature (>100 keV) and the Z=5 boron nucleus drive intense radiation. In a best-case thermonuclear p-B11 plasma, bremsstrahlung carries away roughly 97% of fusion power; the margin between production and loss is only 3% ([Putvinski et al., 2019](https://doi.org/10.1088/1741-4326/ab1a60)). Driving the plasma out of thermal equilibrium can widen this to about 13% ([Ochs et al., 2022](https://doi.org/10.1103/PhysRevE.106.055215)), but bremsstrahlung still dominates at 87% of fusion power. These photons hit the walls and become heat. The DEC handles the charged-particle margin; the thermal plant handles the rest.

![p-B11 thermal only (f_dec=0), 1 GWe net output, Heating: 80 MW wall-plug, 40 MW delivered to plasma](<p-B11 Thermal Cycle.svg>)

## Three Architectures

We modeled three DEC architectures with a free fusion core, using the same methodology as the previous dispatch. Numbers can be reproduced with the [companion script](https://github.com/1cfe/1costingfe/blob/master/examples/dec_blog_numbers.py).

### 1. Steady-State Mirror + Venetian Blind DEC

The [venetian blind](https://iopscience.iop.org/article/10.1088/0029-5515/13/1/005) is the most mature DEC concept (TRL 4–5). Angled metal ribbon grids at successively higher retarding potentials sort ions by energy and collect them on high-potential electrodes, entirely passive, no moving parts. Demonstrated at [48% efficiency on real mirror plasma](https://www.osti.gov/biblio/7341986) at LLNL's TMX (1982); theoretical maximum [70%](https://iopscience.iop.org/article/10.1088/0029-5515/13/1/005). [Realta Fusion](https://www.realtafusion.com/) is the sole active developer, with a [patented](https://patents.google.com/patent/US12166398B2) axisymmetric variant.

In a mirror machine, the venetian blind is an add-on to the expander tanks that already exist for plasma confinement. The DEC hardware (grids, power conditioning, incremental vacuum and tank volume) costs [$79–128M](https://www.osti.gov/biblio/7218298) at 400 MWe DEC electric output. This is comparable to the turbine island it partially replaces (\$168M for sCO2 at 1 GWe).

When the DEC handles 90% of the charged particle transport and the remaining energy goes through a small thermal cycle:

| Configuration (1 GWe, pB11, free core) | Floor (\$/MWh) | Overnight (\$/kW) |
| --- | --- | --- |
| Thermal only (sCO2 Brayton, 47%) | 18 | 1,234 |
| VB DEC at 48% + thermal (hybrid) | 18 | 1,233 |
| VB DEC at 60% + thermal (hybrid) | 18 | 1,226 |

The venetian blind barely moves the floor. With 87% of fusion energy radiated as bremsstrahlung, the DEC only captures the 13% charged-particle margin. At the demonstrated 48% efficiency, the DEC hardware adds cost without meaningfully improving conversion. Even at 60% (never demonstrated on real plasma), the hybrid floor stays at \$18/MWh. Eliminating the turbine entirely and wasting the brem is not viable for p-B11: without a thermal cycle, the plant needs 16 GW of fusion power for 1 GWe net, driving the fully costed LCOE to \$60/MWh, far worse than the thermal path (\$37/MWh). For D-He3 with its higher charged fraction, the VB DEC 60% floor is \$17/MWh, but the He-3 fuel adds \$74/MWh on top.

### 2. Pulsed Inductive DEC

[Helion Energy's](https://www.helionenergy.com/) approach is [architecturally different](https://doi.org/10.1088/0029-5515/51/5/053008) and uses a different fuel. Two D-He3 field-reversed configuration plasmoids collide and merge in a compression chamber. The hot plasma expands against the confining magnetic field, inducing an EMF in the surrounding coils via Faraday's law. The same coils that compress the plasma recover energy during expansion. No plasma-surface contact.

Helion chose D-He3 rather than p-B11 for good reason. D-He3 operates at lower temperature (50–60 keV vs >100 keV) with lower effective charge (Z_eff 1.3 vs 2.5), so bremsstrahlung is a much smaller fraction of fusion power. The primary reaction produces a 14.7 MeV proton and a 3.6 MeV alpha, both charged, 60% of total fusion energy. D-D side reactions contribute 5% neutrons, but the bulk of the energy is available to DEC. With most energy in charged particles, the turbine can genuinely be replaced by a grid-tie inverter ($150/kW, [NREL benchmarks](https://www.nrel.gov/solar/market-research-analysis.html)) that converts pulsed DC to grid AC.

| Configuration (1 GWe, D-He3, free core) | BOP floor (\$/MWh) | He-3 fuel (\$/MWh) | Total (\$/MWh) |
| --- | --- | --- | --- |
| Thermal only (sCO2 Brayton, 47%) | 19 | 81 | 100 |
| Pulsed inductive (85% η, no turbine) | 17 | 58 | 74 |

The BOP floor is comparable to thermal: \$17/MWh vs \$19/MWh. The turbine island is not the dominant cost. Buildings (\$354M), the electrical plant (\$96M), O&M (\$37M/yr), and financing charges together dwarf the turbine. Eliminating it removes about 10% of overnight capital.

The He-3 fuel cost is a different problem. At \$2M/kg (optimistic), He-3 adds \$58/MWh, nearly six times the 1-cent target by itself. At the [DOE-allocated price](https://www.everycrsreport.com/reports/R41419.html) of \$4.5M/kg, it roughly doubles. Helion's answer is to breed He-3 internally from D-D side reactions. If the fuel cost drops to zero, the D-He3 pulsed DEC floor is \$17/MWh at baseline and \$7.8/MWh at aggressive conditions, comparable to p-B11.

The round-trip efficiency of the compression-expansion cycle is the other open question. Helion [claims 95%](https://www.helionenergy.com/) but has not published data. The [theoretical framework](https://youtu.be/5nHmqk1cI2E?t=505) (Kirtley et al., APS DPP 2024) suggests 68–87% is more realistic for plasma-present operation, depending on the burn cycle. If the efficiency is 85%, the system works. If it's 70%, the advantage over a turbine narrows. If it drops below 60%, the economics favor just building a turbine. Polaris (Helion's seventh device, under construction) is the make-or-break test.

![D-He3 pulsed inductive DEC at 85%, f_dec=0.95, 95% of charged transport to DEC, 5% + neutrons to thermal](<D-He3 Pulsed Inductive DEC.svg>)

### 3. MHD Generator

A [magnetohydrodynamic generator](https://doi.org/10.1016/B978-0-08-025566-5.50008-1) passes a conducting fluid through a transverse magnetic field, driving current through electrodes via the Lorentz force. No moving parts. From the 1960s through 1993, the US, Soviet, Japanese, and Australian governments [spent hundreds of millions](https://www.gao.gov/products/emd-80-14) developing MHD generators that used hot coal combustion gas as the working fluid. These programs demonstrated up to [32 MW gross output](https://www.osti.gov/biblio/6380343) and proved the basic physics. Separately, liquid-metal MHD (LMMHD) generators have shown [>71% efficiency](https://doi.org/10.1016/0196-8904(81)90006-6) at small scale.

MHD is not a charged-particle DEC; it replaces the Brayton cycle. Any fusion plant could route its coolant through an MHD channel instead of a turbine. Liquid metal flowing from the blanket or first wall converts heat to electricity at projected >60% efficiency with no moving parts. MHD is still Carnot-limited: a heat engine without rotating machinery, not a way around thermodynamics. For D-T, LMMHD integrated into the lithium breeding blanket handles neutron and thermal energy. For aneutronic fuels, it handles the bremsstrahlung fraction that charged-particle DEC cannot touch.

The obstacle is that no fusion-MHD system has ever been built. The coal programs achieved TRL 5–6, but the problem that killed them ([electrode erosion from coal slag](https://www.gao.gov/products/emd-80-14)) doesn't apply to clean fusion coolants. The new critical challenge is MHD pressure drop: liquid metal flowing through the strong magnetic fields of a fusion device (5–15 T) experiences enormous drag, potentially consuming more pumping power than the MHD generates. This is an [active research problem](https://doi.org/10.3390/en14206640) at ORNL and KIT Karlsruhe, with no demonstrated solution at reactor scale. For the purposes of this analysis, MHD remains a what-if: promising physics, no engineering basis for costing.

## The Floor Doesn't Move Much

Here is the comparison at aggressive conditions (2 GWe, 95% availability, 3% WACC, 50-year life, 3-year construction) where every parameter is pushed:

| Approach (2 GWe, pB11, free core) | Floor (\$/MWh) | Overnight (\$/kW) | Budget for core |
| --- | --- | --- | --- |
| Thermal only (sCO2 Brayton, 47%) | 7.6 | 826 | +\$2.4/MWh |
| VB DEC 60% + thermal (hybrid) | 7.8 | 840 | +\$2.2/MWh |

At aggressive conditions, the hybrid DEC and thermal approaches are within \$0.2/MWh. The no-turbine configurations are excluded for p-B11 because wasting 87% of fusion energy as brem heat requires an enormous core that more than offsets the turbine savings (see above). The dominant terms remain buildings (\$370-490M), the electrical plant (\$180M), O&M (\$62M/yr), indirect costs, and financing.

The floor is not set by the power conversion choice. It is set by the industrial cost of building, staffing, and financing a large power plant.

## Where DEC Actually Helps

If DEC doesn't move the floor, why does it matter?

The floor is what you pay with a free fusion core. The core is not free. DEC's real benefit is that higher conversion efficiency means **you need less fusion power for the same net electric output**, and for D-He3, less fusion power also means less He-3 fuel consumed.

In the fully costed D-He3 mirror at 1 GWe baseline conditions:

| Configuration | LCOE (\$/MWh) | Core cost | He-3 fuel (\$/MWh) |
| --- | --- | --- | --- |
| Thermal only (47%) | 120 | \$1,276M | 81 |
| VB DEC 60% (hybrid) | 111 | \$1,284M | 74 |
| Pulsed inductive (85%) | 94 | \$1,278M | 58 |

The core cost barely changes: DEC hardware (counted as part of the core in standard fusion accounting) offsets savings from smaller heating and blanket systems. But LCOE drops \$9/MWh with the venetian blind and \$26/MWh with pulsed inductive. The savings come mostly from fuel: higher efficiency means less fusion power per MWh, which means less He-3 burned. For D-He3, DEC is as much a fuel-saving technology as a power conversion technology.

For p-B11 (negligible fuel cost), the effect is smaller: thermal LCOE is \$37/MWh, VB DEC 60% is \$37/MWh (core \$1,189M vs \$1,209M). Because bremsstrahlung dominates, the DEC captures only the 13% charged-particle margin and does not meaningfully reduce the required fusion power.

The pulsed inductive case depends entirely on the unverified round-trip efficiency. If the efficiency is 70% instead of 85%, the core size savings are modest and the pulsed power system (capacitor banks, switches; the model assumes $0.50/J stored NOAK, vs [$2-4/J](https://arxiv.org/abs/2602.19389) in the literature) may cost more than the turbine it replaces. Capacitor lifetime is a separate risk: the model assumes 10^8 shot lifetime (NOAK), but current high-energy-density capacitors achieve roughly 10^7 cycles. At a 1 Hz rep rate, 10^7 shots is four months of operation, requiring frequent bank replacements that add both capital cost and downtime.

## The Bremsstrahlung Constraint

Every charged-particle DEC architecture for p-B11 runs into bremsstrahlung. As noted above, even in the most favorable scenario, bremsstrahlung is 87% of fusion power. Without driving the plasma out of equilibrium, the margin is only 3%.

This is more severe than the previous dispatch acknowledged. In a p-B11 plant, charged-particle DEC captures the margin; radiation handles the majority. The thermal plant is not a small bottoming cycle; it is the primary power conversion pathway.

The options for the radiated fraction are:

1. **Thermal cycle**: a full turbine island converts brem heat at 47–53%. This is what the "hybrid" configurations above assume. It works but preserves essentially the full thermal BOP.
2. **MHD on the coolant**: if the brem-heated wall coolant is liquid metal, an MHD channel could convert at >60% with no turbine. Undemonstrated at reactor scale.
3. **Reject as waste**: accept the efficiency loss and dump the brem heat to cooling towers. At 87% brem fraction, this means wasting most of the fusion energy. The overall plant efficiency collapses: a DEC at 85% efficiency on the 13% net charged fraction yields only 11% of fusion power as electricity, far worse than thermal-only (47%).
4. **X-ray photovoltaic DEC**: convert bremsstrahlung directly to electricity via cascading Auger emission in nanometric high-Z/low-Z layers ([Binderbauer & Tajima, 2018](https://patents.google.com/patent/US9893226B2/)). If it works, this would capture the bremsstrahlung fraction without a thermal cycle, but no prototype has been built, no efficiency has been measured, and the radiation damage environment (keV photons at GW/m² flux for years) is extreme. This is a TRL 1 concept.

Options 3 and 4 do not have a demonstrated path. Today, the thermal plant is the only demonstrated path for the radiated fraction. DEC, where it applies, captures only the charged-particle margin.

![p-B11 venetian blind DEC at 60%, f_dec=0.9, 90% of transport to DEC, 10% hits walls as bremsstrahlung heat](<p-B11 VB DEC.svg>)

## Learning Rates

The numbers above are snapshots. Deployed technologies get cheaper. Turbines have been manufactured for over a century with thousands of GW deployed; there is little room for costs to fall. The DEC concepts are much lower TRL, with larger cost uncertainties but more room for learning.

The venetian blind is the most mature DEC. Its dominant costs (vacuum vessels, cryopumps) are mature industrial hardware, 75% of total cost. The grids are 2%. Little room to learn down.

The pulsed inductive path is different. Capacitor banks and power electronics dominate, and both are early in their deployment curves. The [NOAK assumption](https://arxiv.org/abs/2602.19389) in the 1costingfe model is $0.50/J stored (all-in: capacitors, switches, charging, buswork). Today's lab pricing is $20–50/J, a 40–100x gap, comparable to what solar PV achieved over two decades. Grid-tie inverters have already followed that curve ($1,000/kW in the early 2000s to $100–150/kW today). But the fusion-specific components (capacitors rated for 100 million cycles at high energy density) do not yet exist as commercial products. The $0.50/J figure is a target, not a price anyone can buy today.

The pulsed floors in this post already assume the 40–100x reduction has happened. If capacitors stay at $5–20/J, those floors rise substantially and the economics favor a turbine. The thermal and venetian blind floors carry no comparable risk.

## Conclusions

**1. DEC does not move the p-B11 floor.** At baseline conditions, the p-B11 cost floor is \$18/MWh regardless of whether a venetian blind DEC is added. With 87% of fusion energy radiated as bremsstrahlung, the thermal plant handles the majority of conversion and cannot be eliminated without an enormous increase in fusion power. At aggressive conditions, thermal and hybrid DEC converge to \$7.6-7.8/MWh. The floor is dominated by buildings, electrical systems, O&M, and financing.

**2. DEC's potential value is efficiency, not BOP savings.** No DEC has demonstrated efficiency above 48% on real plasma, comparable to sCO2 Brayton. If higher efficiencies are achieved, the payoff is a smaller fusion core for the same net output. This shows up in the fully costed plant, not the free-core floor.

**3. The venetian blind is an even trade.** At demonstrated efficiency (48%), the DEC hardware costs more than the turbine it replaces and converts at roughly the same efficiency as sCO2 Brayton. At 60%+ (undemonstrated on real plasma), it's modestly better. The venetian blind does not justify itself on BOP savings alone; it must deliver >60% sustained efficiency to beat the thermal path.

**4. The pulsed inductive approach is all-or-nothing.** If the round-trip efficiency is 85%+, the pulsed architecture halves the required fusion power and eliminates the turbine, a genuine transformation. If it's below 70%, the economics favor building a turbine. The gap between "transformative" and "not viable" is 15 percentage points of unverified efficiency.

**5. Aneutronic fuel is required, but not sufficient.** DEC needs most fusion energy in charged particles. But p-B11 radiates 87-97% as bremsstrahlung. The DEC captures the thin margin: at most 13% with alpha channeling, 3% without. The turbine handles the majority.

**6. D-He3 is the better DEC fuel, if you can get the He-3.** D-He3 has manageable bremsstrahlung (35% of fusion power) and a large charged fraction, making DEC a significant power path rather than a margin add-on. But purchased He-3 costs \$58-81/MWh, far more than the entire BOP floor. Self-breeding eliminates this, but whether a D-He3 device can breed enough He-3 to sustain itself is undemonstrated.

**7. MHD could close the gap, someday.** An MHD generator on the brem-heated coolant could convert the radiated fraction at >60% with no moving parts. The physics is demonstrated; the engineering for fusion environments is not.

The path to 1-cent fusion energy does not run through direct energy conversion alone. DEC raises the budget for the core by improving efficiency, but it does not remove the industrial cost structure that creates the floor. Reaching \$10/MWh still requires the same conditions identified in the previous dispatch: large plants, high availability, low-cost financing, fast construction, and long plant life. DEC makes those conditions slightly easier to achieve, not unnecessary.

## References

1. Moir, R.W. & Barr, W.L., "Venetian-blind direct energy converter for fusion reactors," *Nuclear Fusion* 13, 35–45 (1973). [Link](https://iopscience.iop.org/article/10.1088/0029-5515/13/1/005)
2. Hoffman, M.A., "Electrostatic Direct Energy Converter Performance and Cost Scaling Laws," UCID-17560, LLNL (1977). [Link](https://www.osti.gov/biblio/7218298)
3. Slough, J. et al., "Creation of a high-temperature plasma through merging and compression of supersonic field reversed configuration plasmoids," *Nuclear Fusion* 51(5), 053008 (2011). [Link](https://doi.org/10.1088/0029-5515/51/5/053008)
4. Putvinski, S.V., Ryutov, D.D. & Yushmanov, P.N., "Fusion reactivity of the p-B11 plasma revisited," *Nuclear Fusion* 59(7), 076018 (2019). [Link](https://doi.org/10.1088/1741-4326/ab1a60)
5. Ochs, I.E. et al., "Improving the feasibility of economical proton-boron-11 fusion via alpha channeling with a hybrid fast and thermal proton scheme," *Physical Review E* 106, 055215 (2022). [Link](https://doi.org/10.1103/PhysRevE.106.055215)
6. Kirtley, D. et al., "Generalized burn cycle efficiency framework," APS DPP 2024, Abstract GO05.8. [Presentation](https://youtu.be/5nHmqk1cI2E?t=505)
7. CATF IWG, "Extension of the Fusion Power Plant Costing Standard," arXiv:2602.19389 (2026). [Link](https://arxiv.org/abs/2602.19389)
8. Rosa, R.J., *Magnetohydrodynamic Energy Conversion*, Hemisphere Publishing (1987). [Link](https://doi.org/10.1016/B978-0-08-025566-5.50008-1)
9. GAO, "Magnetohydrodynamics: A Promising Technology for Efficiently Generating Electricity From Coal," EMD-80-14 (1980). [Link](https://www.gao.gov/products/emd-80-14)
10. Shea, D.A. & Morgan, D., "The Helium-3 Shortage: Supply, Demand, and Options for Congress," Congressional Research Service, R41419 (2010). [Link](https://www.everycrsreport.com/reports/R41419.html)
11. Barr, W.L. et al., "Experimental Results from a Beam Direct Converter at 100 kV," *Journal of Fusion Energy* 2, 131-143 (1982). [Link](https://www.osti.gov/biblio/7341986)
12. Fabris, G. & Hantman, R.G., "Interaction of fluid dynamics phenomena and generator efficiency in two-phase liquid-metal gas magnetohydrodynamic power generators," *Energy Conversion and Management* 21(1), 49-60 (1981). [Link](https://doi.org/10.1016/0196-8904(81)90006-6)
13. Pinkhasov, D. et al., "CDIF 32-MWt MHD Program Results," *Proc. 31st Symposium on Engineering Aspects of MHD* (1993). [Link](https://www.osti.gov/biblio/6380343)
14. Smolentsev, S. et al., "MHD Thermofluid Issues of Liquid-Metal Blankets: Phenomena and Advances," *Energies* 14(20), 6640 (2021). [Link](https://doi.org/10.3390/en14206640)
15. Binderbauer, M.W. & Tajima, T., US Patent 9,893,226 B2, "Photon-to-electric direct conversion" (2018). [Link](https://patents.google.com/patent/US9893226B2/)
16. NREL, "U.S. Solar Photovoltaic System and Energy Storage Cost Benchmarks." [Link](https://www.nrel.gov/solar/market-research-analysis.html)
17. Realta Fusion, US Patent 12,166,398 B2, "Axisymmetric ferromagnetic venetian blinds" (2025). [Link](https://patents.google.com/patent/US12166398B2)
18. 1cFE, "1costingfe: Open-source fusion techno-economic model." [GitHub](https://github.com/1cfe/1costingfe)
