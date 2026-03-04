# DT Magnetic Mirror — 500 MWe Reference Case

## Summary

| Metric | Value |
|--------|-------|
| LCOE | 80.2 $/MWh (8.02 ¢/kWh) |
| Overnight cost | 5,862 $/kW |
| Fusion power | 1,137 MW |
| Net electric | 500 MW |
| Q_eng | 4.6 |
| Recirculating fraction | 22.0% |

**Plant assumptions:** 85% availability, 30 yr lifetime, 5 yr construction, NOAK
**Cost overrides:** CAS21 (Buildings) = $250M

## Cost Breakdown

| Code | Account | M$ |
|------|---------|----|
| CAS10 | Preconstruction | 17.2 |
| CAS21 | Buildings | 250.0 |
| CAS22 | Reactor Plant Equipment | 1,578.4 |
| CAS23 | Turbine Plant | 126.6 |
| CAS24 | Electrical Plant | 53.9 |
| CAS25 | Miscellaneous | 32.8 |
| CAS26 | Heat Rejection | 21.9 |
| CAS28 | Digital Twin | 5.0 |
| CAS29 | Contingency | 0.0 |
| CAS30 | Indirect Costs | 344.8 |
| CAS40 | Owner's Costs | 103.4 |
| CAS50 | Supplementary | 14.4 |
| CAS60 | IDC | 382.7 |
| | **Total Capital** | **2,931.2** |
| CAS70 | O&M (annualized) | 62.0 |
| CAS80 | Fuel (annualized) | 0.3 |
| CAS90 | Financial (annualized) | 236.2 |

## CAS22 Reactor Plant Equipment Breakdown

| Sub-account | Description | M$ |
|-------------|-------------|----|
| C220101 | Blanket + FW + neutron multiplier | 83.8 |
| C220102 | Shielding | 61.9 |
| C220103 | Coils | 516.1 |
| C220104 | Supplementary heating | 353.2 |
| C220105 | Primary structure | 5.9 |
| C220106 | Vacuum system | 19.3 |
| C220107 | Power supplies | 58.6 |
| C220108 | Divertor / end plugs | 65.9 |
| C220111 | Installation labor | 163.0 |
| C220200 | Main & secondary coolant | 105.6 |
| C220300 | Auxiliary cooling + cryoplant | 19.8 |
| C220400 | Radioactive waste | 2.4 |
| C220500 | Fuel handling | 73.9 |
| C220600 | Other equipment | 6.6 |
| C220700 | Instrumentation & control | 42.5 |
| **C220000** | **Total CAS22** | **1,578.4** |

## Sensitivity Analysis (Elasticity = %LCOE / %param)

### Engineering Levers

| Parameter | Elasticity |
|-----------|------------|
| availability | -0.9405 |
| construction_time_yr | +0.2665 |
| eta_th | -0.0934 |
| plasma_t | +0.0693 |
| blanket_t | +0.0666 |
| p_input | +0.0481 |
| eta_pin | -0.0479 |
| ht_shield_t | +0.0140 |
| p_cool | +0.0120 |
| f_sub | +0.0115 |
| vessel_t | +0.0081 |
| f_dec | -0.0062 |
| p_trit | +0.0060 |
| eta_de | -0.0059 |
| p_cryo | +0.0051 |
| mn | +0.0048 |
| structure_t | +0.0033 |
| p_coils | +0.0030 |
| p_house | +0.0024 |
| p_pump | +0.0009 |

### Financial

| Parameter | Elasticity |
|-----------|------------|
| interest_rate | +0.6602 |
| inflation_rate | +0.0396 |
