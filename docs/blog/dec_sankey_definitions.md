# Sankey Diagram Definitions for "Direct Energy Conversion and the Cost Floor"

Generated using [SankeyMatic](https://sankeymatic.com/build/). Paste each block into the editor.

All energy values in MW. All cases are 1 GWe net output, baseline conditions
(85% availability, 7% WACC, 30-year life, 6-year construction).

Power balance from `1costingfe` model (mirror concept).

## Figure 1: p-B11 Thermal Cycle

Placement: end of "What Direct Energy Conversion Does"

```
// p-B11 thermal only (f_dec=0), 1 GWe net output
// Heating: 80 MW wall-plug, 40 MW delivered to plasma

Fusion Reactions [2397] Charged Particles
Heating System [40] Charged Particles

Charged Particles [2434] Thermal Cycle (sCO2)

Thermal Cycle (sCO2) [1146] Gross Electric
Thermal Cycle (sCO2) [1292] Waste Heat

Gross Electric [1000] Net Electric
Gross Electric [146] Recirculating

:Fusion Reactions #d45
:Charged Particles #2a7fff
:Thermal Cycle (sCO2) #888
:Gross Electric #6a3
:Net Electric #0a0
:Waste Heat #c33
:Recirculating #c80
:Heating System #c80
```

## Figure 2: D-He3 Pulsed Inductive DEC 85%

Placement: end of "Pulsed Inductive DEC"

```
// D-He3 pulsed inductive DEC at 85%, f_dec=0.95
// 95% of charged transport to DEC, 5% + neutrons to thermal

Fusion Reactions [1301] Charged Particles
Fusion Reactions [63] Neutrons
Heating System [40] Charged Particles

Charged Particles [1271] Pulsed Inductive DEC
Charged Particles [67] Bremsstrahlung → Thermal (sCO2)
Neutrons [69] Bremsstrahlung → Thermal (sCO2)

Pulsed Inductive DEC [1080] Gross Electric
Pulsed Inductive DEC [191] Waste Heat

Bremsstrahlung → Thermal (sCO2) [66] Gross Electric
Bremsstrahlung → Thermal (sCO2) [74] Waste Heat

Gross Electric [1000] Net Electric
Gross Electric [146] Recirculating

:Fusion Reactions #d45
:Charged Particles #2a7fff
:Neutrons #a5a
:Pulsed Inductive DEC #07a
:Bremsstrahlung → Thermal (sCO2) #e8a030
:Gross Electric #6a3
:Net Electric #0a0
:Waste Heat #c33
:Recirculating #c80
:Heating System #c80
```

## Figure 3: p-B11 VB DEC 60% Hybrid

Placement: end of "The Bremsstrahlung Constraint"

```
// p-B11 venetian blind DEC at 60%, f_dec=0.9
// 90% of transport to DEC, 10% hits walls as bremsstrahlung heat

Fusion Reactions [1912] Charged Particles
Heating System [40] Charged Particles

Charged Particles [1754] Venetian Blind DEC
Charged Particles [195] Bremsstrahlung → Thermal (sCO2)

Venetian Blind DEC [1052] Gross Electric
Venetian Blind DEC [702] Waste Heat

Bremsstrahlung → Thermal (sCO2) [94] Gross Electric
Bremsstrahlung → Thermal (sCO2) [105] Waste Heat

Gross Electric [1000] Net Electric
Gross Electric [146] Recirculating

:Fusion Reactions #d45
:Charged Particles #2a7fff
:Venetian Blind DEC #07a
:Bremsstrahlung → Thermal (sCO2) #e8a030
:Gross Electric #6a3
:Net Electric #0a0
:Waste Heat #c33
:Recirculating #c80
:Heating System #c80
```
