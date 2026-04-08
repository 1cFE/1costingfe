# Learning Rates and the Cost of Fusion

Fusion will start expensive. The [previous dispatch](https://1cf.energy/fusions-cost-floor-what-if-the-core-were-free/) showed that even with a free fusion core, the balance-of-plant floor is $17/MWh for p-B11 and $29/MWh for D-T. The fully costed plant is $36-83/MWh depending on fuel and concept. These are NOAK numbers: they already assume that component costs have fallen to mature-industry levels. The question is which components still have room to fall, how far, and what accelerates the journey.

## Components That Will Not Get Cheaper

Turbines, generators, cooling towers, buildings, switchgear, and transformers are mature industrial hardware. Thousands of GW of turbines are installed worldwide. The construction industry has built power plants for over a century. These components are at the bottom of their learning curves. Cost improvements are measured in low single-digit percentages per decade, driven by incremental process optimization, not technology change.

These mature components constitute the balance-of-plant floor. They are 30-50% of total plant cost at NOAK. No amount of deployment will make them substantially cheaper.

## Components Where Learning Is Steep

The fusion core is a different story. Its dominant cost drivers are early in their manufacturing curves, and the gaps between today's prices and NOAK targets are large:

| Component | Today | NOAK target | Factor | Status |
| --- | --- | --- | --- | --- |
| REBCO HTS conductor | $150-300/kAm | $50/kAm | 3-6x | Product exists, volume does not |
| Pulsed power capacitors | $20-50/J | $0.50/J | 40-100x | Product category does not exist |
| B-11 fuel enrichment | $10,000/kg | $75/kg | 133x | Process proven, not at fusion scale |
| He-3 (purchased) | $2-4.5M/kg | $0 (self-bred) | Infinite | Undemonstrated |
| Remote handling | ITER FOAK | Commercial standard | 3x | Standardization |
| NBI heating | EUR 9-15M/MW | $7M/MW | 1.3-2x | ITER reference to NOAK |

The LCOE figures in the previous dispatches already assume these NOAK targets have been met. The floors are post-learning costs. If any of these targets are not achieved, the actual costs will be higher.

**REBCO superconducting tape** is commercially produced today. The $50/kAm target requires scaling production from hundreds of kilometers per year to tens of thousands, driven by CFS, Tokamak Energy, and the ARPA-E BETHE program. This is a manufacturing scale-up for an existing product, not a speculative technology.

**Pulsed power capacitors** rated for 100 million charge-discharge cycles at high energy density do not exist as a commercial product category. The $0.50/J target is a 40-100x reduction from lab pricing, comparable to what solar PV achieved over two decades. The learning curve has not started.

**B-11 fuel enrichment** at $75/kg assumes chemical exchange distillation at industrial scale. The process is proven for B-10 production; B-11 is the byproduct. But it has never been operated at fusion fleet volumes.

## The Risk Is Asymmetric

Because the model uses NOAK targets, the risk runs one way. If targets are met, the floors hold. If REBCO stays at $150/kAm instead of reaching $50, or capacitors stay at $5-20/J instead of $0.50, costs rise. There is more room to be wrong on the upside than the downside.

A recent [Nature Energy analysis](https://www.nature.com/articles/s41560-026-02023-8) (Tang et al., 2026) argues that fusion's plant-level experience rates will be 2-8%, much lower than the 8-20% assumed in most projections. They are right that plant-level rates will be low: buildings and turbines do not learn fast. But a plant-level rate misses the point. Magnets, capacitors, and fuel enrichment have their own learning trajectories, and those are where the cost reduction lives.

## AI Won't Design Your Reactor, but It Will Make It Cheaper to Build and Run

There is a widespread expectation that AI will accelerate fusion development. Some of this is real. Most of it is not.

### What AI will not do

AI will not replace computational physics. Plasma simulation requires solving kinetic or magnetohydrodynamic equations in regimes where training data does not exist. A neural network trained on existing tokamak data cannot predict the behavior of a novel configuration. Unlike natural language or image recognition, plasma physics has no large corpus of labeled examples. Surrogate models can interpolate within known parameter spaces; fusion needs extrapolation to new ones. Numerical solvers do not hallucinate. ML surrogates do.

AI will not skip materials qualification. Computational screening of candidate materials is useful, but the bottleneck is physical testing: irradiation campaigns that take years, creep tests that run for thousands of hours, joining qualification that requires destructive examination. AI can suggest what to test next. It cannot replace the testing.

### What AI will actually do

The value of AI for fusion cost reduction is in the factory and the control room.

**Manufacturing quality control.** REBCO tape production is a continuous deposition process where defects reduce performance. AI-driven in-line inspection (computer vision, anomaly detection) is standard in semiconductor fabrication and directly applicable. Higher yield and fewer defective meters translate to lower $/kAm. This compresses the REBCO learning curve.

**Process optimization.** Chemical exchange distillation for B-11, cryogenic processes, and capacitor dielectric manufacturing all involve complex process control where AI-driven optimization outperforms manual tuning. These are the same problems AI solves in chemical plants and refineries today.

**Predictive maintenance.** Demonstrated at scale: NextEra Energy reports 25-30% maintenance cost reduction and 70-75% fewer unplanned breakdowns using ML-based monitoring. For a fusion plant, this means higher availability (more MWh per dollar of capital) and lower O&M. Both flow directly into LCOE.

**Control room automation.** AI agents for grid dispatch, alarm management, and routine operations. The [DeepMind/EPFL tokamak plasma control work](https://doi.org/10.1038/s41586-021-04301-9) (2022) demonstrated reinforcement learning for real-time magnetic control. Extending this to plant-level operations is engineering, not research.

**Staffing reduction.** Combining predictive maintenance, autonomous inspection, and control room automation, a 30% staffing reduction is achievable with current technology. For a p-B11 plant at 1 GWe, this saves $1-2/MWh. The capital cost of automation ($15-30M) adds less than $0.2/MWh.

**Construction scheduling.** AI-optimized logistics for modular fabrication and site assembly. The impact is on construction time: compressing a 6-year schedule to 3-4 years reduces interest during construction and flows into LCOE.

## The Net Effect

AI does not change the physics of fusion or the thermodynamics of power conversion. It does not make plasma confinement easier or materials more radiation-resistant. What it does is compress manufacturing learning curves, reduce operational costs, and shorten construction schedules.

The components where AI has the most impact are the same ones where learning potential is highest: REBCO production (manufacturing QC), capacitor manufacturing (process optimization), plant operations (predictive maintenance, staffing). AI makes the steep part of the learning curve steeper. It does not make the flat parts (buildings, turbines, concrete) any less flat.

For the LCOE floors in the previous dispatches: AI contributes to achieving the NOAK targets, not to changing them. The floors already assume learning has happened. AI is part of how it happens.

## Conclusions

**1. The model already assumes substantial learning.** NOAK targets for magnets (3-6x), capacitors (40-100x), and fuel enrichment (133x) are built into the LCOE figures. The dispatches report post-learning costs, not first-plant costs.

**2. The risk is asymmetric.** If NOAK targets are met, the floors hold. If not, costs rise. There is more upside risk than downside.

**3. Some components will not get cheaper.** Turbines, buildings, and switchgear are at the bottom of their learning curves. Their costs are the floor.

**4. Design choices matter more than experience rates.** Modular, factory-fabricated, standardized designs learn fast. Bespoke, site-built, customized designs learn slowly. The learning rate is a property of the design and manufacturing strategy, not of "fusion."

**5. AI compresses learning curves, it does not bypass them.** The value is in manufacturing QC, process optimization, predictive maintenance, and construction scheduling. Not in replacing physics simulation or skipping materials testing.

## References

1. Tang, L. et al. "Fusion power experience rates are overestimated." *Nature Energy* (2026). [Link](https://www.nature.com/articles/s41560-026-02023-8)
2. Degrave, J. et al. "Magnetic control of tokamak plasmas through deep reinforcement learning." *Nature* 602, 414-419 (2022). [Link](https://doi.org/10.1038/s41586-021-04301-9)
3. Woodruff, S. "A Costing Framework for Fusion Power Plants." arXiv:2601.21724 (2025). [Link](https://arxiv.org/abs/2601.21724)
4. CATF IWG, "Extension of the Fusion Power Plant Costing Standard." arXiv:2602.19389 (2026). [Link](https://arxiv.org/abs/2602.19389)
5. 1cFE. "1costingfe: Open-source fusion techno-economic model." [GitHub](https://github.com/1cfe/1costingfe)
