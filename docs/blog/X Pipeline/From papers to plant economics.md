# From papers to plant economics: turning fusion concepts into executable cost models

Published fusion power plant designs almost never include a complete cost estimate. One compact tokamak paper (Sorbom et al., 2015) costs three fabricated components and explicitly excludes the balance of plant. A recent stellarator preconceptual design ([arXiv:2512.08027](https://arxiv.org/abs/2512.08027)) publishes a target LCOE with no cost breakdown at all. This is typical. Fusion papers describe physics performance and engineering geometry; the economics are left as an exercise for the reader.

The result is that comparing the cost of different fusion concepts requires assembling a consistent cost picture from incomplete, inconsistent, and often incomparable data. One design publishes magnet costs in 2014 dollars. Another publishes net electric output and availability but no capital costs. A third asserts an LCOE target with no supporting analysis. To compare them, you need a common framework that fills the gaps systematically: the same cost account structure, the same financial assumptions, the same indirect cost fractions, the same escalation basis.

This dispatch demonstrates how [1costingfe](https://github.com/1cfe/1costingfe) does that. The framework underpins our [first dispatch](https://1cf.energy/blog/fusions-cost-floor) on the industrial floor of fusion electricity cost and our [second dispatch](https://1cf.energy/blog/direct-energy-conversion) on direct energy conversion architectures; here we turn it on published concept designs. We take two very different fusion concepts, an HTS compact tokamak and a planar coil stellarator, extract their published parameters, and run them through the same cost framework. The point is not the LCOE numbers themselves, which carry large uncertainty. The point is what you learn when you apply a consistent methodology across concepts that differ in confinement, geometry, scale, and data availability.

## The workflow

Applying 1costingfe to a published design has three stages. Today this is a manual workflow with tool support; a separate dispatch will describe the automated ingestion and code-generation layer we are building on top of it.

**Stage one: parameter extraction.** A published design paper or report is read and its LCOE-relevant parameters extracted into a structured analysis document. For each parameter, the extraction records the value, the source (with section and table reference), the units, and a confidence tag: known (published and independently verifiable), uncertain (published but with large range or weak basis), or truly unknown (not published; estimated from analogous concepts or first principles). This stage is labour-intensive and concept-specific. It cannot be fully automated because the quality of fusion design publications varies enormously: one paper gives a complete radial build with materials and thicknesses; another gives a major radius and a fusion power and nothing else.

**Stage two: model setup.** The extracted parameters are written into a [1costingfe](https://github.com/1cfe/1costingfe) model script. 1costingfe is a JAX-based costing framework that computes LCOE from the Code of Accounts Structure (CAS) established by the Generation IV Economic Modeling Working Group and adapted for fusion by Schulte et al. and the ARIES program, building on and replacing the [pyFECONS](https://github.com/woodruff-scientific-ltd/pyfecons) methodology (Woodruff, 2025).

The CAS is a standardized decomposition of every cost a power plant incurs, developed for fission and adapted for fusion by the ARIES program. Capital accounts (CAS10 through CAS60) cover everything from land and licensing through buildings, reactor equipment, turbines, cooling systems, indirect services, and construction financing. Annual accounts cover operations and maintenance (CAS70), fuel (CAS80), and the annualized capital charge (CAS90). The LCOE is CAS70 + CAS80 + CAS90 divided by annual electricity production. The structure forces completeness: when a fusion paper reports only the cost of the magnets, the CAS tells you exactly which accounts are missing.

The user specifies what the published design tells them (geometry, power balance, thermal efficiency, specific component costs if available) and the framework fills in the rest using its internally calibrated defaults. Every parameter that comes from the framework rather than the source paper is tagged as DEFAULT in the script, so a reader can see exactly which numbers are concept-specific and which are borrowed.

For the compact tokamak, the model setup looks like this:

```python
from costingfe import ConfinementConcept, CostModel, Fuel

CPI_2014_TO_2024 = 1.34  # BLS CPI-U: ~236 (2014) -> ~315 (2024)

model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)

result = model.forward(
    net_electric_mw=261.0,       # Published aggressive pilot output
    availability=0.75,           # UNCERTAIN: not published
    lifetime_yr=30,
    construction_time_yr=5.0,    # Compact geometry
    interest_rate=0.07,
    inflation_rate=0.0245,
    noak=True,

    # Geometry (from published design paper)
    R0=3.3,                      # Major radius [m]
    plasma_t=1.13,               # Minor radius [m]
    elon=1.84,                   # Elongation

    # Power balance
    p_input=38.6,                # 25 MW LHCD + 13.6 MW ICRF
    eta_th=0.46,                 # Supercritical Rankine at 250 bar / 540C

    # Published component costs (2014 USD x CPI)
    cost_overrides={
        "C220103": 5150.0 * CPI_2014_TO_2024,  # REBCO magnets + structure
        "C220101":  260.0 * CPI_2014_TO_2024,  # FLiBe blanket
        "C220106":   92.0 * CPI_2014_TO_2024,  # Vacuum vessel (Inconel-718)
        "CAS27":    146.0,                      # FLiBe inventory: 950t x $154/kg NOAK
    },
)
```

Three component costs come from the published design paper (magnets, blanket, vacuum vessel), inflated from 2014 to 2024 dollars. Everything else, buildings, turbine, electrical plant, cooling, indirect costs, owner's costs, O&M, comes from the framework's internally calibrated defaults.

For the stellarator, the setup is different. No published component costs exist, so the framework defaults are used for most accounts. The exception is the magnet cost: the default stellarator coil model assumes conventional 3D-wound coils, but this concept uses 12 tokamak-like encircling coils plus 324 smaller, simpler planar shaping coils. We estimate the cost by running the tokamak coil model at the stellarator's field strength (which captures the encircling coils), then applying a 1.5x factor as a rough estimate for the additional shaping coils and their control infrastructure:

```python
model = CostModel(concept=ConfinementConcept.STELLARATOR, fuel=Fuel.DT)

# Estimate planar coil cost from the tokamak coil model
coil_ref = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
ref = coil_ref.forward(net_electric_mw=390.0, availability=0.88, lifetime_yr=40,
                        b_max=20.0, r_coil=1.85)
planar_coil_cost = float(ref.cas22_detail["C220103"]) * 1.5  # 1.5x for 336 coils

result = model.forward(
    net_electric_mw=390.0,       # Published design point
    availability=0.88,           # Stated in the design paper
    lifetime_yr=40,              # Magnet design lifetime
    construction_time_yr=8.0,    # DEFAULT: no published timeline
    noak=False,                  # FOAK: first plant is the scenario

    # Geometry (from preconceptual design paper)
    R0=8.0,                      # Major radius [m]
    plasma_t=1.8,                # Minor radius [m]
    elon=1.0,                    # QA stellarator, circular cross-section
    blanket_t=0.50,              # Published blanket thickness

    # Power balance
    p_input=1.0,                 # Ignited plasma; 1 MW ECRH for impurity control
    eta_th=0.40,                 # Three-stage Rankine, 635C steam
    p_cryo=15.0,                 # UNCERTAIN: 336 REBCO coils at 20K

    cost_overrides={"C220103": planar_coil_cost},
)
```

**Stage three: cost computation and sensitivity.** 1costingfe computes the full CAS hierarchy (CAS10 through CAS90), the power balance, and the LCOE. Because the framework is built on JAX, it also computes sensitivity elasticities via automatic differentiation: the percentage change in LCOE per percentage change in each input parameter, in a single backward pass. This identifies which parameters have the most influence on cost without requiring manual parameter sweeps.

The costing framework, its methodology, and worked examples are documented in the [1costingfe repository](https://github.com/1cfe/1costingfe) and its [companion paper](https://github.com/1cfe/1costingfe/blob/master/tex/paper.tex). The concept-specific model setup scripts used in this analysis were written from the analysis documents, typically with an AI coding agent drafting the initial setup and a human reviewing the parameter choices and source citations.

## Two concepts, one framework

Before diving into specific published designs, it is worth showing what the framework produces for a generic D-T tokamak at 1 GWe with no concept-specific overrides:

```python
model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
result = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
```

This gives an LCOE of $87/MWh, an overnight cost of $6,166/kW, and a total capital of $6.2 billion. The magnets are $516M (using the framework's REBCO coil model at default field and bore), buildings are $626M, and the balance of plant is $488M. This sits within the CATF Investors Working Group NOAK range of $60-100/MWh, confirming that the framework's calibration is reasonable for a conventional D-T tokamak at scale. The two concept-specific analyses below deviate from this baseline because of their published parameters and cost data, not because the framework is miscalibrated.

### Concept A: HTS compact tokamak

This is a compact, high-field D-T tokamak designed around REBCO high-temperature superconducting magnets. The published design (Sorbom et al., 2015) specifies a major radius of 3.3 m, minor radius 1.13 m, and 9.2 T on-axis field. It produces 525 MW of fusion power and 261 MWe net electric via a supercritical Rankine cycle at 46% efficiency.

The design paper includes fabricated component costs for three subsystems: magnets plus structural support ($5.1-5.2 billion in 2014 dollars), FLiBe blanket ($260 million), and vacuum vessel ($92 million). These are the only published cost data. No buildings, no balance of plant, no indirect costs, no O&M.

| Account | Cost (M$) | Source |
| --- | --- | --- |
| CAS22 Reactor plant equipment | 8,364 | Published paper (3 components) + defaults |
| of which: magnets + structure | 6,901 | Published, CPI-adjusted to 2024 |
| of which: FLiBe blanket | 348 | Published, CPI-adjusted |
| of which: vacuum vessel | 123 | Published, CPI-adjusted |
| CAS21 Buildings | 301 | Framework default (D-T enhanced industrial) |
| CAS23-26 Balance of plant | 155 | Framework default |
| CAS27 Special materials (FLiBe) | 146 | 950t at $154/kg NOAK |
| CAS30 Indirect costs | 1,495 | 20% of directs, construction-time scaled |
| CAS50 Supplementary | 434 | D-T decommissioning, spares, shipping |
| CAS60 Interest during construction | 1,642 | 7% WACC, 5-year build |
| **Total capital** | **12,579** | |
| CAS70 O&M (annualized) | 86 | D-T staffing-based |
| CAS80 Fuel (annualized) | 0.3 | Deuterium + Li-6 market purchase |
| **LCOE** | **$642/MWh** | 261 MWe, 75% CF, 30 yr, 7% WACC |

At $642/MWh, this is not a prediction of what electricity from this plant will cost. It is what happens when you take the published 2015 component costs at face value, inflate them to 2024 dollars, and wrap a complete plant around them using calibrated defaults.

The result is high because the paper's $5.15 billion magnet estimate reflects 2014 REBCO tape prices. REBCO has since fallen to roughly $100/kA-m, but we are analyzing based on the 2014 figure. The $6.9 billion CPI-adjusted figure may overstate the magnet cost by a large factor. But it is the published number, and we use published numbers.

The FOAK scenario (2x magnet cost plus 10% contingency) produces $1,205/MWh. For context, the CATF Investors Working Group (2025) estimates FOAK fusion LCOE at $150-200/MWh and NOAK at $60-100/MWh.

### Concept B: planar coil stellarator

This is a quasi-axisymmetric stellarator with 12 encircling and 324 individually-controlled planar REBCO shaping coils. The preconceptual design ([arXiv:2512.08027](https://arxiv.org/abs/2512.08027)) specifies a major radius of 8 m, minor radius 1.8 m, and 6 T on-axis field. It produces 1,036 MW of fusion power and 390 MWe net electric via a three-stage steam Rankine cycle at 40% efficiency. The plasma is effectively ignited (Q of approximately 958), requiring only 1 MW of operational ECRH for impurity control.

No bottom-up capital cost breakdown has been published. The only public cost figure is an asserted LCOE target of $150/MWh for the first plant, declining to $60/MWh at scale. The model uses framework defaults for most accounts. The one exception is the magnet cost, where the default stellarator coil model (calibrated to conventional 3D-wound coils with a 12x manufacturing markup and a 2x path factor for non-planar winding) is inappropriate for planar coils. The entire point of the planar coil architecture is that individual coils are simpler to manufacture, at the cost of needing more of them (336 vs. 18 for a conventional stellarator).

The magnet system has 12 tokamak-like encircling coils that provide the main field, plus 324 smaller planar shaping coils. We estimate the cost by using the tokamak coil model (which captures the encircling coils) with a 1.5x factor for the shaping coils and their control infrastructure. This is a rough estimate; no published cost data exists for this architecture. The result is $1,290M, roughly half the default stellarator coil model.

| Account | Cost (M$) | Source |
| --- | --- | --- |
| CAS22 Reactor plant equipment | 2,818 | Planar coil override + defaults |
| of which: magnets | 1,290 | Tokamak geometry, 12x markup for planar coils |
| CAS21 Buildings | 381 | Framework default (D-T enhanced industrial) |
| CAS23-26 Balance of plant | 188 | Framework default |
| CAS29 Contingency | 340 | 10% FOAK |
| CAS30 Indirect costs | 997 | 20% of directs, construction-time scaled |
| CAS50 Supplementary | 259 | D-T decommissioning, spares, shipping |
| CAS60 Interest during construction | 1,428 | 7% WACC, 8-year build |
| **Total capital** | **6,481** | |
| CAS70 O&M (annualized) | 96 | D-T staffing-based |
| CAS80 Fuel (annualized) | 0.6 | Deuterium + Li-6 market purchase |
| **LCOE** | **$194/MWh** | 390 MWe, 88% CF, 40 yr, 7% WACC, FOAK |

At $194/MWh, this is a FOAK estimate (10% contingency on all direct costs) because the published target of $150/MWh refers to the first plant. The remaining gap between $194/MWh and $150/MWh could reflect a lower coil markup than our estimate, a shorter construction time than the 8-year default, or both.

## Cross-check against published estimates

Our compact tokamak LCOE of $642/MWh is far above the range of published fusion cost estimates. This is expected, and the gap is instructive.

The CATF Investors Working Group (2025) surveyed expert estimates for fusion LCOE and found a range of $150-200/MWh for FOAK plants and $60-100/MWh for NOAK. The FECONS reference D-T tokamak (Woodruff, 2025) produces $55/MWh at 637 MWe with 90% availability. The ARPA-E ALPHA compact fusion re-costing (Woodruff Scientific, 2020) gives $43/MWh at 500 MWe with 90% availability and zero contingency. Araiinejad & Shirvan (2025) published an independent TEA for this same compact tokamak design.

| Source | LCOE ($/MWh) | Key assumptions |
| --- | --- | --- |
| This analysis (NOAK) | 642 | 261 MWe, 75% CF, 2015 magnet costs at face value |
| CATF IWG NOAK range | 60-100 | Expert survey, multiple concepts |
| FECONS reference DT tokamak | 55 | 637 MWe, 90% CF, ARIES-calibrated |
| ARPA-E ALPHA compact fusion | 43 | 500 MWe, 90% CF, zero contingency, non-HTS |
| CATF IWG FOAK range | 150-200 | Expert survey, multiple concepts |
| This analysis (FOAK) | 1,205 | 2x magnet cost, 10% contingency |

The gap between our $642/MWh and the $60-100/MWh NOAK range is almost entirely explained by two factors: the magnet cost and the plant scale. The 2015 paper's $5.15 billion magnet estimate (2014 dollars) reflects REBCO tape prices that have since fallen by roughly 5x. And 261 MWe is small; the FECONS reference plant is 2.4x larger, and LCOE scales sublinearly with size. Araiinejad & Shirvan (2025), using the same design but updated assumptions for REBCO learning and regulatory treatment, arrive at estimates within the CATF range.

This is exactly the kind of insight 1costingfe is designed to surface. The model does not hide the gap or explain it away. It shows you the published input, the computed output, and the sensitivity analysis that tells you which inputs would need to change (and by how much) to reach a different answer. The magnet cost and availability dominate; everything else is noise.

## Sensitivity: what moves the needle

The framework computes sensitivity elasticities via automatic differentiation: the percentage change in LCOE for a 1% change in each input parameter. The tornado chart below shows the compact tokamak's top sensitivities.

```
Parameter                    Elasticity
─────────────────────────────────────────────────────
Availability                    -0.96  ██████████████████████████████
Cost of capital (WACC)          +0.77                                ███████████████████████
Construction time               +0.30                                █████████
NBI heating cost                +0.04                                █
Thermal efficiency              -0.03
Coil winding radius             +0.01
Major radius                    +0.01
Heating wall-plug eff.          -0.01
```

Availability is nearly 1:1: a 10% increase in capacity factor reduces LCOE by 9.6%. Cost of capital is the second largest, followed by construction time. Everything below those three is noise at this design point, because the magnets dominate the capital cost so completely that changing the thermal efficiency or the heating system barely registers.

Note what is absent: the magnets themselves. The tokamak's magnet cost is a fixed dollar override from the published paper, not a computed parameter. The sensitivity analysis perturbs input parameters and measures the LCOE response; a fixed override has no knob to turn, so it does not appear. This is correct but important to understand: the largest single cost item is invisible to the sensitivity analysis precisely because it is anchored to published data rather than computed from a model.

For the stellarator, the picture is different. Availability is the top sensitivity (-0.94), but the coil winding radius (elasticity +1.02) is close behind. The coil winding radius and peak field are proxies for the magnet cost: the coil cost scales as r_coil^2 * b_max, so these parameters capture the magnet sensitivity that is absent from the tokamak chart. This is because the stellarator's magnets are computed parametrically (with our planar coil estimate) rather than overridden from a published cost. The tokamak's cost is anchored to published component data, so the sensitivity is about operations and finance; the stellarator's cost is anchored to parametric estimates, so the sensitivity is about the magnet manufacturing assumptions.

The [companion script](tornado_hts_tokamak.py) reproduces this chart.

## What the comparison reveals

Running both concepts through the same framework produces several observations that would not be visible from either concept's publications alone.

**The magnet cost dominates both concepts, but for different reasons.** For the compact tokamak, the magnets are 82% of CAS22 and the single largest uncertainty in the model. The 2015 cost estimate predates the REBCO manufacturing scaleup that has reduced tape prices by roughly 5x. For the stellarator, the magnets are 46% of CAS22, using our planar coil estimate (tokamak coil model with a 1.5x factor for the shaping coils). In both cases, the magnet account is where concept-specific engineering data would most reduce uncertainty.

**Availability has the highest elasticity for both, but what comes next differs.** The tokamak's sensitivity analysis shows availability (elasticity -0.96) as the parameter with the most influence on LCOE, but the design has never published a capacity factor estimate. The stellarator's availability elasticity is similar (-0.94), followed by the coil winding radius (+1.02), reflecting the magnet cost scaling. For the tokamak, cost of capital is second (+0.77); for the stellarator, it is the coil geometry. These are different kinds of uncertainty: the tokamak's is operational and financial, the stellarator's is manufacturing.

**The balance of plant is cheap and boring.** For the tokamak, CAS21 (buildings) plus CAS23-26 (BOP equipment) total $456 million, roughly 5% of the $8.4 billion reactor plant equipment. For the stellarator, the same accounts total $569 million, 20% of CAS22. The turbines, buildings, and cooling towers are not where the cost uncertainty lives. This is consistent with the finding from our [first dispatch](https://1cf.energy/blog/fusions-cost-floor): the cost floor is real but low, and the fusion core dominates the total.

**Construction time and financing amplify everything.** The tokamak's 5-year construction generates $1.6 billion of interest during construction at 7% WACC; the total capital investment is 36% higher than the overnight cost. The stellarator's 8-year construction (a default assumption since no timeline is published) generates $1.4 billion of IDC, making the total capital 28% higher than overnight. These are not engineering costs; they are financial costs that scale with everything else. Faster construction reduces cost for every concept.

**Scale matters.** The tokamak at 261 MWe is small for a power plant. The staffing economy of scale (power-law exponent 0.5) means that a 1 GWe plant with the same technology would have roughly 60% lower LCOE per MWh from O&M alone. The stellarator at 390 MWe benefits less from scaling up because it is already in a more favourable size range.

## What the framework does not tell you

1costingfe produces numbers. Numbers invite false precision. Several caveats apply.

The model uses published parameters at face value. If a design paper's magnet cost estimate is obsolete (and for the tokamak it likely is, given the REBCO price trajectory since 2014), the LCOE will be correspondingly wrong. The framework does not improve the source data; it makes the source data's implications explicit.

Framework defaults for accounts where concept-specific data is unavailable are calibrated to the ARIES program and NETL fossil energy baselines. They are reasonable for conventional D-T plants at GWe scale. For a 261 MWe compact tokamak or a first-of-a-kind planar coil stellarator, the defaults may be too high or too low. The model flags every default-sourced parameter so the user can see exactly where the analysis is relying on analogues rather than concept-specific data.

The sensitivity analysis identifies which parameters have the most influence, but it does not tell you how much those parameters can realistically change. Availability has an elasticity of -0.96 for the tokamak, but whether it can achieve 90% availability depends on divertor lifetime, FLiBe maintenance procedures, and remote handling technology readiness, none of which are published.

A related risk is overfitting. The framework has many parameters, and it is tempting to adjust several of them to match a desired LCOE. The sensitivity analysis helps here: for the compact tokamak, only three parameters (availability, cost of capital, construction time) have elasticities above 0.1. Everything else is below the noise floor at this design point. A credible analysis should focus on getting those few high-sensitivity parameters right and leave the rest at defaults, rather than tuning many small parameters whose individual effects are unmeasurable against the uncertainty in the dominant ones.

## The comparative value

The value of running multiple concepts through the same framework is not the individual LCOE numbers. It is the structured comparison. Where does the cost live? Which parameters matter most? Where is the published data strong, and where is it absent?

For the compact tokamak, the answer is: the cost lives in the magnets, the highest sensitivity is availability, and the published data covers fabricated components but not operations.

For the planar coil stellarator, the answer is: the cost lives in the coil system, the highest sensitivities are availability and coil manufacturing, and the published data covers physics performance but not costs.

These are different kinds of engineering challenges, requiring different kinds of evidence to resolve. A comparative framework makes that visible.

The [1costingfe](https://github.com/1cfe/1costingfe) framework and its [companion paper](https://github.com/1cfe/1costingfe/blob/master/tex/paper.tex) document the methodology, cost account structure, and physics models. All numbers in this post can be reproduced from the code snippets shown above.

Writing these scripts by hand works for two concepts. It does not scale to the 36 concepts in our spanning set. The follow-up dispatch describes the automated ingestion and code-generation pipeline we are building on top of 1costingfe so that a new fusion concept enters as a published paper and exits as a fully attributed LCOE estimate with sensitivity analysis.

## References

1. Sorbom, B. N. et al. "ARC: A compact, high-field, fusion nuclear science facility and demonstration power plant with demountable magnets." *Fusion Engineering and Design*, 100, 378-405 (2015). [Link](https://doi.org/10.1016/j.fusengdes.2015.07.008)
2. "The Helios Stellarator." [arXiv:2512.08027](https://arxiv.org/abs/2512.08027) (2025). [Link](https://arxiv.org/abs/2512.08027)
3. Araiinejad, A. & Shirvan, K. "Techno-economic analysis of the ARC fusion reactor." *Applied Energy* (2025). [Link](https://doi.org/10.1016/j.apenergy.2025.01.039)
4. Schwartz, J. A. et al. "The value of fusion energy to a decarbonized United States electric grid." arXiv:2405.01514 (2024). [Link](https://arxiv.org/abs/2405.01514)
5. CATF Investors Working Group. "Assessing the cost of fusion energy." arXiv:2602.19389 (2025). [Link](https://arxiv.org/abs/2602.19389)
6. Woodruff, S. "A Costing Framework for Fusion Power Plants." arXiv:2601.21724 (2025). [Link](https://arxiv.org/abs/2601.21724)
7. Waganer, L. "ARIES Cost Account Documentation." UCSD-CER-13-01 (2013). [Link](https://cer.ucsd.edu/_files/publications/UCSD-CER-13-01.pdf)
8. 1cFE. "1costingfe: Open-source fusion techno-economic model." [GitHub](https://github.com/1cfe/1costingfe)
