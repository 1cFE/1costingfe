# The Lower Bound for Fusion Energy Cost

If the deuterium-tritium fusion core were free — every magnet, every blanket, every heating system, handed to you at zero cost — the   plant around it would still produce electricity at $29/MWh. That is nearly three times the 1-cent target. The uncertainty is not in the   turbines or the buildings. It is in the fusion core. And the range of estimates for that core is enormous.

Published estimates for a commercial fusion plant range from [$47/MWh](https://www.sciencedirect.com/science/article/abs/pii/S0920379605007210) for an optimistic nth-of-a-kind advanced tokamak (ARIES-AT, Najmabadi et al., 2006) to [$160+/MWh](https://www.sciencedirect.com/science/article/pii/S0360544218305395) for a first-generation EU-DEMO (Entler et al., 2018), with most serious estimates [landing between $50 and $130/MWh](https://www.tandfonline.com/doi/abs/10.13182/FST15-157) (Sheffield & Milora, 2016). Recent analysis from Cambridge [suggests early fusion plants will exceed $150/MWh](https://www.sciencedirect.com/science/article/abs/pii/S0301421523000964) even with production learning (Lindley et al., 2023).

Cost estimates for fusion projects are kept under wraps, and even experts have a hard time estimating the eventual costs. A [recent Nature Energy analysis](https://www.nature.com/articles/s41560-026-02023-8) compiled first-of-a-kind CAPEX estimates from experts and the literature and found a range of **$1,400 to $43,000 per kilowatt** — a 30x spread (Tang et al., 2026). That study determined that the experience rates for a deuterium-tritium tokamak fusion core would likely end up in the range of 2-8% cost reduction per doubling of deployment. This revises the 8-20% estimate assumed previously. These rates are a result of the size and complexity of the technology.

A fusion plant has to convert the energy generated in the fusion core to useful electricity. For this purpose, most projects envision using a heat engine — turbines, generators, cooling towers — in addition to a switchyard, a control room, and a building to put it all in. These components are mature industrial hardware, with no real price uncertainty or learning rates.

We are seeking corridors for fusion to have a truly revolutionary price point — 1 cent per kilowatt-hour. If we start by accounting for the components we're certain about and that exist today at commercial scale, we could find the lower bound on what fusion electricity can cost.

To find the lower bound on fusion energy cost we can ask: what is the cost of the plant if the fusion core were free?

## The Heat Engine — Generating Electricity from Heat

The main product of a fusion core is heat, and every fusion plant has to convert that heat into electricity. Most designs envision a thermal cycle — steam Rankine or supercritical CO2 Brayton. In either case, the plant needs buildings, electrical infrastructure, a grid connection, a control room, and staff.

Using the [1costingfe](https://github.com/1cfe/1costingfe) techno-economic model, we can price these components for a 1 GWe plant. The table below uses a supercritical CO2 Brayton cycle, which gives the lowest floor of the three options we modeled (Rankine, sCO2, and combined cycle). All numbers in this post can be reproduced with the [companion script](https://github.com/1cfe/1costingfe/blob/master/examples/lower_bound_blog_numbers.py).

The power conversion equipment is the same regardless of fuel choice:

| Component | Cost | What it is |
| --- | --- | --- |
| Turbine & generator | $180M | sCO2 turbomachinery, comparable to gas plant hardware |
| Electrical plant | $97M | Switchgear, transformers, grid connection |
| Miscellaneous equipment | $59M | Cranes, compressed air, fire protection |
| Heat rejection | $25M | Cooling towers and circulating water (smaller with higher efficiency) |
| **BOP subtotal** | **$360M** |  |

## The Rest of the Plant

Buildings are not fuel-independent — they range from $320M (industrial-grade, for aneutronic fuel) to $570M (enhanced-industrial, for DT) depending on the radiation protection and tritium containment requirements. We return to this below.

The BOP equipment is priced from [NETL's Cost and Performance Baseline for Fossil Energy Plants](https://www.osti.gov/servlets/purl/1893822) (DOE/NETL-2022/3575) and the [ARIES cost account documentation](https://cer.ucsd.edu/_files/publications/UCSD-CER-13-01.pdf) (Waganer, UCSD-CER-13-01, 2013), adjusted to 2024 dollars.

On top of the direct costs, there are indirect costs (engineering, project management — roughly 20% of directs), owner's costs, supplementary costs (shipping, spares, insurance, decommissioning provisions), and financing charges. A plant also needs staff — roughly 30 to 80 full-time employees at 1 GWe scale depending on fuel choice and the associated radiation protection requirements, costing $18-40M/year in loaded operations and maintenance (O&M). All of these costs are included in the LCOE floors reported below.

## The Balance of Plant for Deuterium-Tritium

Deuterium-tritium (DT) is the mainstream fuel choice for fusion. It produces 80% of its energy as 14.1 MeV neutrons, which activate structural components and require heavy shielding, and it uses radioactive tritium — an environmental hazard if it leaks into groundwater. Even with a free fusion core, a DT plant still needs buildings with radiation shielding, tritium containment with secondary barriers, hot cells for remote handling of activated components, and nuclear-rated HVAC. These requirements are driven by the hazards themselves — neutron activation and tritium inventory — not by regulatory classification. The buildings cost $570M at 1 GWe, and the plant requires roughly 80 staff for radiation protection, tritium handling, and the maintenance procedures these hazards demand.

For a 1 GWe DT tokamak at standard financial assumptions (7% weighted average cost of capital (WACC), 85% availability, 30-year life, 6-year construction), the Levelized Cost of Electricity (LCOE) floor with a free fusion core is **$29/MWh** — nearly three times the 1-cent target ($10/MWh). The overnight capital cost of this core-free plant is $1,700/kW.

For context, the fully costed DT tokamak at the same conditions comes in at $82/MWh. The fusion core accounts for about two-thirds of that. But the remaining third — $29/MWh of buildings, BOP, staffing, and financing — already exceeds the 1-cent target by itself. No improvement in the fusion core can close a gap that lives outside the core.

## Fuel Choice Reshapes the Floor

The $29/MWh DT floor is not an intrinsic property of fusion power. It is a consequence of the neutrons and tritium that DT produces, and the buildings, staff, and procedures required to handle them. Different fuel choices produce different floors.

| Fuel | Buildings (1 GWe) | BOP floor (excl. fuel) | Staffing |
| --- | --- | --- | --- |
| DT | $570M | $29/MWh | 78 FTE |
| D-He3 | $380M | $19/MWh | 39 FTE |
| pB11 | $322M | $17/MWh | 36 FTE |

Deuterium-helium-3 (D-He3) produces about 5% of its energy as neutrons from DD side reactions — far less than DT, but not zero. The buildings require some radiation shielding and modest tritium monitoring, bringing them to $380M. The BOP floor drops to $19/MWh. However, D-He3 carries a separate problem: helium-3 fuel at current prices ($2M/kg) contributes $79/MWh to the levelized electricity cost, pushing the total free-core floor to $98/MWh. The BOP is competitive; the fuel supply is not.

Proton-boron (pB11) fuel is aneutronic — 99.8% of its fusion energy comes out as charged alpha particles, not neutrons. It uses no tritium and does not activate structural components. The buildings can be built to conventional industrial standards — no shielding, no hot cells, no tritium containment infrastructure. The result is a BOP floor of **$17/MWh**, roughly half the DT floor.

The $250M building cost gap between DT and pB11 is larger than most fusion core cost-reduction scenarios in the literature. This gap is not a statement about plasma physics difficulty. It is a statement about what the expectation of handling neutrons and tritium does to the cost of the building you put the plant in. The fuel choice reshapes the floor before the fusion core enters the picture.

## The Floor Is Not Fixed

The floors above assume a single 1 GWe plant built with commercial project finance. The parameters that determine this cost are:

- **Scale** spreads fixed costs (buildings, staff, indirects) over more megawatt-hours
- **Availability** produces more energy per dollar of installed capital
- **Financing cost (WACC)** determines the annual capital charge rate
- **Construction time** determines interest during construction
- **Plant lifetime** determines how many years of revenue amortize the capital

The point of the table below is not the floor itself, but what budget it leaves for the fusion core — shown in the last column. This is the headroom between the floor and the $10/MWh target. At baseline, the budget is negative: the floor alone overshoots. At aggressive but not extreme conditions, real budget appears.

| Scenario (pB11) | Floor ($/MWh) | Overnight ($/kW) | Budget left for core |
| --- | --- | --- | --- |
| Baseline: 1 GWe, 85%, 7% WACC, 30yr, 6yr build | 17 | 1,100 | -$7/MWh |
| 2 GWe, 85%, 7%, 30yr, 6yr | 14 | 970 | -$4/MWh |
| 2 GWe, 95%, 3% WACC, 50yr, 3yr build | 7.3 | 770 | +$2.7/MWh |
| 3 GWe, 95%, 3%, 50yr, 3yr | 6.4 | 730 | +$3.6/MWh |
| 5 GWe, 95%, 2% WACC, 50yr, 3yr | 5.1 | 690 | +$4.9/MWh |

At 2 GWe with 95% availability, 3% WACC, 3-year construction, and 50-year life, the floor drops to $7/MWh. Is 3-year construction realistic for the BOP? Modern gas combined-cycle plants of comparable scale are built in 2-3 years. The BOP of a fusion plant is similar in scope and complexity. In the scenario we are solving for, we do not construct the fusion core, only the turbine halls, cooling systems, switchyards, and buildings.

At these conditions, the remaining budget for the fusion core is $3/MWh — roughly $850/kW of overnight capital for magnets, heating, vacuum vessel, structure, power supplies, and installation. For reference, the entire overnight cost of a natural gas combined-cycle plant is $900-1,200/kW. The fusion core has to fit inside that budget while being none of those things — mature, mass-produced, or built at scale — yet.

O&M staffing accounts for $4/MWh of the $7/MWh floor — more than half. The model assumes roughly 60 employees for a 1 GWe pB11 plant, built up from a conventional gas plant baseline (30 staff) plus fusion-specific additions for magnets, vacuum systems, plasma diagnostics, and beamlines. In the free-core scenario, those fusion-specific roles disappear, leaving a conventional thermal plant. Lights-out operation — already deployed in semiconductor fabs, chemical plants, and automated factories — could bring that down to 15 full-time employees, dropping the floor from $7 to $5/MWh and nearly doubling the fusion core's budget. This is not speculative; it exists in other industries at scale today.

## Conclusions

**1. Fuel choice is a BOP decision, not just a core decision.** The $250M building cost gap between DT ($570M) and pB11 ($320M) is larger than most fusion core cost-reduction scenarios. Aneutronic fuel downgrades the buildings from nuclear to industrial, eliminates scheduled replacement campaigns for the fusion core, and removes tritium infrastructure entirely. These savings compound through indirect costs and financing. The DT floor ($29/MWh) is nearly double the pB11 floor ($17/MWh). D-He3 falls in between on BOP ($19/MWh) but is burdened by helium-3 fuel costs.

**2. No single lever reaches 1 cent.** At standard financial conditions, even the pB11 LCOE floor is 1.7x the target with a free fusion core. Reaching $10/MWh requires at least four favorable conditions simultaneously: scale (2+ GWe), high availability (95%+), low-cost financing (3% WACC or below), and fast construction (3 years or less). Each of these is individually achievable. Together they represent a systems integration challenge as much as an engineering one.

**3. Scale matters more than unit cost.** Going from 1 GWe to 2 GWe buys more LCOE reduction than any other single lever. Buildings, staff, and indirect costs do not double when the plant doubles — staffing scales as P^0.5, so a 2 GWe plant needs 1.4x the staff of a 1 GWe plant, not 2x. This economy of scale does not depend on any technology breakthrough.

This analysis assumes a thermal cycle. For aneutronic fuels like pB11, where 99.8% of fusion energy is in charged particles, direct energy conversion could bypass the heat engine entirely — eliminating the turbine, cooling towers, and much of the building. That would not lower the floor; it would remove it. Direct energy conversion is less mature, and its cost uncertainty is much larger. We will explore this in a future post.

## The Path Forward

The lower bound is a useful limit case. No fusion core will be free. But the exercise reveals where the constraints lie: not in the fusion core, but in the industrial cost structure around it.

A pB11 fusion plant at 2 GWe scale, with 95% availability, 3% cost of capital, 3-year construction, and 50-year life, has a balance-of-plant floor of $7/MWh and a fusion core budget of $850/kW. That budget is tight but not zero, and it grows with scale.

Reaching 1-cent fusion energy is a whole-plant problem. It requires large plants, high availability, fast construction, low-cost financing, long plant life, and a fuel that does not burden the balance of plant with radioactivity safety requirements. Every component of this path — industrial buildings, large turbines, high-availability operations, government-backed financing, long-lived civil structures — exists today, in other industries, at the required scale.

Tang et al.'s Nature Energy analysis arrives at a compatible conclusion from the opposite direction: mainstream fusion designs have technological characteristics — enormous unit size, extraordinary complexity, heavy customization — that predict slow cost learning. Their recommendation is to explore alternative fusion concepts with more favorable attributes, including smaller unit size, reduced complexity, and alternative fuels like proton-boron. The lower-bound analysis shows why: simpler, smaller, aneutronic designs have a lower floor to build on, not just a better shot at a cheap fusion core.

The fusion core is the hard part. The floor tells us how much room it has to work with.

## References

1. Najmabadi, F. et al. "The ARIES-AT advanced tokamak, Advanced technology fusion power plant." *Fusion Engineering and Design*, 80, 3-23 (2006). [Link](https://www.sciencedirect.com/science/article/abs/pii/S0920379605007210)
2. Entler, S. et al. "Approximation of the economy of fusion energy." *Energy*, 152, 489-497 (2018). [Link](https://www.sciencedirect.com/science/article/pii/S0360544218305395)
3. Sheffield, J. & Milora, S. "Generic Magnetic Fusion Reactor Revisited." *Fusion Science and Technology*, 70(1), 14-35 (2016). [Link](https://www.tandfonline.com/doi/abs/10.13182/FST15-157)
4. Lindley, B. A. et al. "Can fusion energy be cost-competitive and commercially viable? An analysis of magnetically confined reactors." *Energy Policy*, 177 (2023). [Link](https://www.sciencedirect.com/science/article/abs/pii/S0301421523000964)
5. Tang, L. et al. "Fusion power experience rates are overestimated." *Nature Energy* (2026). [Link](https://www.nature.com/articles/s41560-026-02023-8)
6. U.S. DOE/NETL. "Cost and Performance Baseline for Fossil Energy Plants, Volume 1." DOE/NETL-2022/3575 (2022). [Link](https://www.osti.gov/servlets/purl/1893822)
7. Waganer, L. "ARIES Cost Account Documentation." UCSD-CER-13-01 (2013). [Link](https://cer.ucsd.edu/_files/publications/UCSD-CER-13-01.pdf)
8. 1cFE. "1costingfe: Open-source fusion techno-economic model." [GitHub](https://github.com/1cfe/1costingfe)