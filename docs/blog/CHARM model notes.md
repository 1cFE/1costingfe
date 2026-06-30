# What the model says about CHARM (Pale Blue Fusion)

## The concept

CHARM (Centrifugal Hybrid Axisymmetric Rotating Mirror) is Pale Blue Fusion's approach to proton-boron fusion. It is a centrifugally confined magnetic mirror that uses supercritical plasma rotation to improve confinement, combined with RF alpha channeling to extract energy from fusion-born alpha particles. The fuel is p-B11, which is nearly aneutronic: less than 1% of the fusion energy appears as neutrons.

Pale Blue Fusion has published no plant design, no engineering parameters, and no cost breakdown of any kind. The ARPA-E OPEN 2024 award presentation (Fisch, 2025) describes the physics concept and the CMFX proof-of-principle experiment (6.7 m, small bore), but gives no reactor-scale parameters. Every quantitative parameter in this analysis is estimated from analogous concepts or first principles. The LCOE output is indicative only and should not be cited as a Pale Blue Fusion estimate.

## Model configuration

The analysis uses 1costingfe's MIRROR concept with PB11 fuel. The mirror maps the cylindrical geometry, solenoid coil cost model, and DEC-capable power balance. The p-B11 fuel activates the correct near-aneutronic cost scaling: no breeding blanket, no tritium processing, minimal remote handling, near-zero licensing, and industrial-grade buildings.

| Parameter | Value | Basis |
| --- | --- | --- |
| Net electric output | 500 MWe | Reference scale (no published design point) |
| Availability | 80% | No maintenance study; slightly below DT mirror default |
| Plant lifetime | 30 yr | Standard reference |
| Construction time | 5 yr | Mirror geometry simpler than tokamak |
| WACC | 7% | Standard reference |
| Chamber length | 30 m | Estimated upscale from 6.7 m CMFX; truly unknown |
| Plasma radius | 1.5 m | Framework default; no published machine radius |
| Auxiliary heating | 60 MW | Rotation sustainment (~30 MW) + RF alpha channeling (~20 MW) + misc (~10 MW) |
| Neutron multiplier | 1.0 | Near-aneutronic; no breeding blanket |
| Thermal efficiency | 20% | Only radiation losses (bremsstrahlung, synchrotron) enter thermal cycle |
| DEC fraction | 85% | Most charged-particle power available for direct conversion |
| DEC efficiency | 70% | Physics limit per Rax, Kolmes, Fisch (PRX Energy, 2025); above historical MARS measurement of 54% |
| Heating wall-plug efficiency | 60% | RF + biased electrode systems |

The power balance reflects a fundamentally different architecture from a thermal-cycle plant. Most of the fusion energy exits as charged alpha particles (p-B11 is 99.8% aneutronic), and 85% of the transport power is routed through direct energy conversion at 70% efficiency. Only the radiation losses (bremsstrahlung and synchrotron, roughly 15-25% of fusion power) enter the thermal cycle at 20% efficiency. The result is a hybrid DEC-plus-thermal plant where DEC provides the majority of the electricity.

The recirculating power is substantial: 60 MW of auxiliary heating at 60% wall-plug efficiency draws 100 MW from the grid, plus coils, cooling, and cryogenics. The engineering Q is 4.3 (recirculating fraction 23%). This is the central uncertainty: rotation sustainment power has never been characterized at reactor scale.

## Results

| Account | Cost (M$) | Notes |
| --- | --- | --- |
| CAS21 Buildings | 200 | Override: reduced from default $354M (no hot cell, no tritium building) |
| CAS22 Reactor plant equipment | 1,340 | Framework defaults for mirror geometry |
| of which: magnets | 513 | Solenoid coils (mirror markup 2.5x, n_coils=10 HAMMIR-class tandem default) |
| of which: heating systems | 283 | 60 MW NBI at framework default $/MW |
| of which: DEC hardware | 30 | Electrostatic DEC for directed axial exhaust |
| of which: installation | 141 | 14% of reactor subtotal |
| CAS23-26 Balance of plant | 216 | Turbine, electrical, misc, heat rejection |
| CAS30 Indirect costs | 294 | 20% of directs |
| CAS50 Supplementary | 104 | p-B11 decommissioning, spares, shipping |
| CAS60 Interest during construction | 328 | 7% WACC, 5-year build |
| **Total capital** | **2,513** | |
| CAS70 O&M (annualized) | 22 | p-B11 staffing: 36 FTE at 1 GWe reference, scaled to 500 MWe; 0.85x mirror concept_scale |
| CAS80 Fuel (annualized) | 0.1 | Protium + B-11 at NOAK prices |
| **LCOE** | **$64/MWh** | 500 MWe, 80% CF, 30 yr, 7% WACC, NOAK |
| **Scaled to 1 GWe** | **$49/MWh** | Economy-of-scale exponent 0.6 |

The overnight capital cost is $5,025/kW. For comparison, ARC's published component costs produce $48,194/kW and Helios's framework defaults produce $21,529/kW. CHARM's low overnight cost reflects three things: simpler mirror coils (n_coils=10 HAMMIR-class tandem at $513M vs. $6,901M for ARC's REBCO tokamak coils), aneutronic fuel (no hot cell, no tritium infrastructure, industrial-grade buildings), and the absence of published component costs (the framework's parametric defaults may underestimate novel subsystems).

## Where the cost lives

The reactor plant equipment (CAS22) is $1.3 billion, split roughly evenly between the magnets ($513M), heating system ($283M), installation ($141M), and everything else. The magnets are now the largest single CAS22 line item, reflecting the framework's HAMMIR-class tandem-mirror default of 10 independent solenoid coils (4 end-plug HTS + 6 LTS central-cell solenoids over a 50 m central cell). Unlike ARC, where magnets are 82% of CAS22, CHARM's simpler solenoid geometry still keeps the magnets to 38% of CAS22 because mirror manufacturing markup (2.5x) is much lower than tokamak (8x). The other large cost center is the heating and sustainment system: 60 MW of auxiliary power for rotation sustainment and RF alpha channeling.

The balance of plant ($216M) and buildings ($200M) together are $416M, roughly 31% of the reactor plant equipment. This is a much larger fraction than ARC (5%) or Helios (15%), reflecting CHARM's relatively cheap core. The p-B11 fuel choice eliminates the hot cell, tritium building, radiation-rated HVAC, and biological shielding, dropping buildings from $354M (p-B11 default) to an overridden $200M.

O&M is low ($22M/yr annualized) because p-B11 staffing is near-industrial: no health physics department, no tritium processing personnel, no radwaste handlers. The 36-FTE reference (at 1 GWe) is roughly half the D-T staffing level, and the 0.85x mirror concept_scale on CAS70 reflects the smaller scheduled-maintenance crew rotation enabled by axial-extraction maintenance access.

Fuel cost is negligible ($0.2M/yr). Protium is commodity hydrogen; B-11 at NOAK prices ($10,000/kg, enriched from natural boron) contributes less than $0.1/MWh. This is a structural advantage over D-He3, where helium-3 at $2M/kg contributes $81/MWh.

## Sensitivity

| Parameter | Elasticity | Interpretation |
| --- | --- | --- |
| Availability | -0.99 | 1% higher CF reduces LCOE by 1%; dominant lever |
| Construction time | +0.29 | Faster build reduces IDC and capital charge |
| Coil winding radius | +0.24 | Larger bore raises magnet cost |
| NBI heating cost | +0.17 | Heating system is a large cost center |
| Peak field | +0.12 | Higher field raises conductor cost |
| Thermal efficiency | -0.12 | Modest; only radiation losses go through thermal cycle |
| DEC efficiency | -0.06 | Matters, but less than availability or construction time |
| Heating wall-plug efficiency | -0.06 | Higher efficiency reduces recirculating power |
| DEC fraction | -0.06 | Higher fraction shifts more power to DEC pathway |

Availability dominates, as it does for every concept. The next tier is construction time and the coil/heating system costs. DEC efficiency and DEC fraction are in the third tier (elasticities around 0.06), meaning a 10% improvement in DEC efficiency reduces LCOE by only 0.6%. This is because DEC competes with the thermal cycle: at 85% DEC fraction, most of the energy is already going through DEC, and improving its efficiency has diminishing returns on the total.

The heating system (NBI cost, heating wall-plug efficiency, auxiliary power level) collectively matters more than DEC. This is the signature of a high-recirculating-fraction plant: the cost of sustaining the plasma is a larger lever than the efficiency of converting its output.

## What is truly unknown

Every plasma parameter in this analysis is estimated, not measured. The blocking gaps, from the concept analysis:

- **Alpha channeling efficiency**: analytical only, never measured experimentally in any device
- **Rotation sustainment power**: not characterized at any scale; the 60 MW estimate is a central guess with no published basis
- **DEC efficiency for rotation energy recovery**: physics bounds exist (PRX Energy, 2025), but no engineering design or hardware demonstration
- **p-B11 nonthermal plasma**: never demonstrated in any experiment; the concept relies on maintaining a non-Maxwellian ion distribution to suppress bremsstrahlung losses
- **CHARM multi-chamber architecture**: never tested; the multi-mirror concept is distinct from both simple mirrors and tandem mirrors
- **Machine geometry at reactor scale**: the CMFX experiment is 6.7 m long with a small bore; a 30 m, 1.5 m radius reactor is a 100x volume extrapolation

The $64/MWh LCOE is what the model produces when you make optimistic but physically-bounded assumptions about all of these unknowns. It should be read as: if the physics works and the engineering parameters land near these estimates, the plant economics are in this range. The "if" is doing most of the work.

## Comparison with the cost floor

From the [first dispatch](https://1cf.energy/blog/fusions-cost-floor): the p-B11 BOP cost floor (free fusion core) at 500 MWe, 80% CF, 7% WACC is roughly $20/MWh. CHARM's $58/MWh includes $38/MWh for the fusion core (CAS22 + its share of indirects, supplementary costs, and financing). The core budget is roughly $2,200/kW of overnight capital. For context, the p-B11 free-core floor at aggressive conditions (2 GWe, 95% CF, 3% WACC, 50-yr life, 3-yr build) is $7.6/MWh, leaving a core budget of roughly $920/kW. CHARM at aggressive conditions would need to fit within that budget to reach the 1-cent target.
