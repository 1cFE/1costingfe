# Pulsed Power Balance and Inductive DEC Revamp

Design spec for restructuring the IFE and MIF models in 1costingfe. The goal
is to make pulsed fusion concepts — including Helion-style inductive direct
energy conversion — first-class citizens that don't require extensive cost
overrides.

---

## Confinement family restructure

Replace the current three families with two:

```python
class ConfinementFamily(Enum):
    STEADY_STATE = "steady_state"  # tokamak, stellarator, mirror
    PULSED = "pulsed"              # all current IFE + MIF concepts
```

The existing MFE power balance (including mirror DEC via `f_dec`/`eta_de`)
is unchanged. All current IFE and MIF concepts move to a unified `PULSED`
family with new power balance functions.

`CONCEPT_TO_FAMILY` mapping updated accordingly. Concept-specific YAML
defaults remain per-concept (laser, z-pinch, heavy_ion, mag_target,
plasma_jet); the concept enum is not changed in this work (expanding the
concept list is a separate task).


## Conversion mode: user-selected, per-concept default

```python
class PulsedConversion(Enum):
    THERMAL = "thermal"              # laser, z-pinch, heavy ion, MTF
    INDUCTIVE_DEC = "inductive_dec"  # Helion FRC, theta-pinch, magnetized concepts
```

Each pulsed concept's YAML sets a default conversion mode. The user can
override via `CostModel(pulsed_conversion=...)` or `forward(pulsed_conversion=...)`.

Physically, `INDUCTIVE_DEC` requires a magnetic compression-expansion cycle.
This is documented but not enforced — the model does not prevent the user
from setting `INDUCTIVE_DEC` on a laser concept (it would produce
nonsensical results, same as many other invalid parameter combinations).


## Two pulsed power balance functions

Both operate in **per-pulse energy** (MJ/pulse), converted to average power
(MW) at the end via `f_rep`. Both share the same parameter interface except
for DEC-specific additions.

### Shared parameters

| Parameter | Description | Notes |
|---|---|---|
| `e_driver_mj` | Energy delivered to plasma per pulse (MJ) | First-class input |
| `f_rep` | Pulse repetition rate (Hz) | First-class input |
| `eta_pin` | Driver wall-plug efficiency | Single value; replaces split `eta_pin1`/`eta_pin2` |
| `fuel` | Fuel enum | Determines `f_ch` (charged-particle fraction) and ash/neutron split |
| `f_rad` | Radiation fraction of charged-particle energy | Default: DT 0.10, DD 0.08, DHe3 0.05, pB11 0.15 (bremsstrahlung-dominated; pB11 high due to Z^2 scaling) |
| `mn` | Neutron energy multiplier | Same as existing |
| `eta_th` | Thermal cycle efficiency | 0 if no thermal BOP |
| `f_sub` | Subsystem power fraction | Same as existing |
| `p_trit` | Tritium processing power (MW) | Same as existing |
| `p_house` | Housekeeping power (MW) | Same as existing |
| `p_cryo` | Cryogenic system power (MW) | Same as existing |
| `p_target` | Target/liner factory power (MW) | Same as existing |
| `p_coils` | Guide field coil power (MW) | Same as existing, 0 for concepts without guide fields |
| `p_pump` | Coolant pumping power (MW) | Only in recirculating power when `eta_th > 0` |

### DEC-only parameters

| Parameter | Description | Notes |
|---|---|---|
| `eta_dec` | Electrical recovery efficiency (coil-to-cap circuit) | |
| `f_pdv` | Fraction of charged-particle energy that does PdV work | Default in YAML with comment: `f_pdv = 1 - (1/r)^(gamma-1)` for adiabatic expansion at ratio `r`, gamma = 5/3 |

### Derived intermediates (exposed as outputs)

| Value | Formula | Used by |
|---|---|---|
| `e_stored_mj` | `e_driver_mj / eta_pin` | Cost layer (C220107 on $/J basis) |
| `p_driver` | `e_driver_mj * f_rep` | Average MW for power table |
| `f_ch` | From `Fuel` enum via `ash_neutron_split` | Energy split |


### `pulsed_thermal_power_balance()`

For laser, Z-pinch, heavy ion, General Fusion MTF — driver energy is
absorbed by the target and enters the thermal pool.

Per-pulse energy balance:

```
E_fus            = Q_sci * E_driver    (or from inverse)
E_neutron        = (1 - f_ch) * E_fus
E_charged        = f_ch * E_fus
E_rad            = f_rad * E_charged
E_charged_net    = E_charged - E_rad

E_thermal        = E_neutron * mn + E_rad + E_charged_net + E_driver
E_electric       = eta_th * E_thermal
```

Convert to average power: `P_x = E_x * f_rep` for all terms.

Recirculating power:

```
P_recirc = E_driver * f_rep / eta_pin
         + p_pump             (only if eta_th > 0)
         + f_sub * P_et
         + p_trit + p_house   (= p_aux)
         + p_cryo
         + p_target
         + p_coils
```

Gross electric: `P_et = P_electric`
Net electric: `P_net = P_et - P_recirc`


### `pulsed_dec_power_balance()`

For Helion FRC, theta-pinch, magnetized concepts with inductive recovery —
driver energy is in an electromagnetic loop (cap bank to coils to plasma
to coils to cap bank).

Per-pulse energy balance:

```
E_fus            = Q_sci * E_driver    (or from inverse)
E_neutron        = (1 - f_ch) * E_fus
E_charged        = f_ch * E_fus
E_rad            = f_rad * E_charged
E_charged_net    = E_charged - E_rad

DEC recovery:
  E_pdv          = f_pdv * E_charged_net
  E_recovered    = eta_dec * (E_driver + E_pdv)
  E_dec_net      = E_recovered - E_driver

Thermal (optional, for neutron energy when eta_th > 0):
  E_dec_waste    = (1 - eta_dec) * (E_driver + E_pdv)
  E_undirected   = E_charged_net - E_pdv
  E_thermal      = E_neutron * mn + E_rad + E_undirected + E_dec_waste
  E_th_electric  = eta_th * E_thermal
```

Convert to average power: `P_x = E_x * f_rep` for all terms.

Gross electric: `P_et = P_dec_net * f_rep + P_th_electric`

Recirculating power:

```
P_recirc = (E_driver * f_rep / eta_pin) - (E_driver * f_rep)
           ^^^ wall-plug charging losses only; E_driver itself is in the cap loop
         + p_pump             (only if eta_th > 0)
         + f_sub * P_et
         + p_trit + p_house
         + p_cryo
         + p_target
         + p_coils
```

Note: The recirculating power for the DEC driver is only the charging
losses `E_driver * f_rep * (1/eta_pin - 1)`, not the full wall-plug draw,
because the cap bank energy is recovered each cycle.

Net electric: `P_net = P_et - P_recirc`


### Inverse power balance

Both functions need inverse forms (target `P_net` to required `e_driver_mj`).
The thermal case is a linear inversion similar to the existing IFE/MIF
inverse. The DEC case is also linear in `E_driver` once `Q_sci`, `f_ch`,
and efficiencies are fixed — `E_driver` is the single unknown scaling all
energy terms proportionally.


### PowerTable changes

`PowerTable` gains new fields for pulsed concepts:

```python
e_driver_mj: float   # Per-pulse driver energy (MJ)
e_stored_mj: float   # Per-pulse cap bank energy (MJ) = e_driver_mj / eta_pin
f_rep: float          # Repetition rate (Hz)
f_ch: float           # Charged-particle fraction (from fuel)
```

Existing fields (`p_dee`, `p_dec_waste`, `p_th`, `p_the`, etc.) are
populated by the pulsed functions in the same way as MFE. Fields that
don't apply (e.g., `p_coils` for laser IFE) are set to 0.


## Costing changes

### `INDUCTIVE_DEC` mode

**C220107 — Power supplies (pulsed driver):**
Reclassified to $/J_stored basis when `INDUCTIVE_DEC`:

```python
c220107 = c_cap_allin_per_joule * e_stored_mj * 1e6   # M$
```

New costing constant: `c_cap_allin_per_joule: 2.0` ($/J_stored, NOAK
all-in driver cost including caps, switches, charging, buswork; sensitivity
range 1.5-4.0).

The existing `power_supplies_base * p_et^0.7` formulation remains active
for `THERMAL` pulsed concepts and for `STEADY_STATE`.

**C220109 — Direct energy converter (inductive DEC):**
Populated from circuit-derived markups on the pulsed driver cost:

```python
markup_cap   = eta_dec * (1 + Q_sci * f_ch) - 1
delta_cap    = c220107 * markup_cap
delta_switch = c220107 * markup_switch_bidir
delta_inv    = c_inv_per_kw_net * p_net    # grid inverter, $/kW_net basis
delta_ctrl   = c220107 * markup_controls

c220109 = delta_cap + delta_switch + delta_inv + delta_ctrl
```

New costing constants:
- `markup_switch_bidir: 0.06` (bidirectional switch premium, % of driver cost)
- `markup_controls: 0.04` (FPGA/energy management upgrade, % of driver cost)
- `c_inv_per_kw_net: 150.0` ($/kW_net, grid-tie inverter + DC-link buffer)

**CAS23 — Turbine plant:**
Scales with thermal electric power `p_the`, not gross electric. When
`eta_th = 0` (no thermal BOP), `p_the = 0` and CAS23 = 0 automatically.

**CAS26 — Heat rejection:**
Scales with total thermal power `p_th`. When `eta_th = 0`, there is no
steam cycle condenser. Waste heat (DEC circuit losses, radiation to walls,
undirected charged-particle energy) must still be rejected, but at reduced
scale. The formula `p_th * heat_rej_per_mw` handles this automatically
since `p_th` includes all non-electric thermal loads regardless of whether
a steam cycle converts them.

**CAS21 — Buildings:**
The turbine building scales with `p_the`. When `eta_th = 0`, it zeros
automatically. The power supply building gets an increased scaling factor
for `INDUCTIVE_DEC` to reflect the large physical footprint of a pulsed
capacitor bank (reference: Helion Polaris, 2800 m^2 for 50 MJ stored
energy).

**CAS72 — Scheduled replacement (cap bank):**
New replacement term for the capacitor bank under `INDUCTIVE_DEC`:

```python
n_shots_per_year  = f_rep * 8760 * 3600 * availability
t_replace_yr      = cap_shot_lifetime / n_shots_per_year
annual_cap_replace = c220107 / t_replace_yr
```

New costing constant: `cap_shot_lifetime: 1.0e8` (shots, NOAK baseline;
sensitivity range 1e7-1e9). Added to the existing CAS72 replacement
framework alongside blanket and DEC grid replacements.


### `THERMAL` pulsed concepts

Costing unchanged from today's IFE/MIF behavior. The new pulsed parameters
(`f_rep`, `e_driver_mj`) are available in the power table but do not alter
the cost account structure.


### CAS23 and CAS26 scaling basis change

Currently both CAS23 and CAS26 scale with `p_et` (gross electric). This
should change to scale with the **thermal** power throughput for both
pulsed and steady-state concepts:

- CAS23: `p_the * turbine_per_mw` (thermal electric, i.e., the steam
  turbine output)
- CAS26: `p_th * heat_rej_per_mw` (total thermal, i.e., the heat that
  must be rejected)

This makes them automatically correct for mixed DEC+thermal plants
(e.g., D-T with inductive DEC and a steam cycle for neutrons) and for
steady-state mirrors with partial DEC.


## YAML defaults structure

### New shared pulsed defaults

Each pulsed concept YAML gains:

```yaml
# Pulsed parameters
e_driver_mj: 12.0       # Energy delivered to plasma per pulse [MJ]
f_rep: 1.0               # Repetition rate [Hz]
pulsed_conversion: thermal   # or inductive_dec
f_rad: 0.05              # Radiation fraction of charged-particle energy

# PdV work fraction for inductive DEC
# For adiabatic expansion: f_pdv = 1 - (1/r)^(gamma-1), gamma=5/3
# r=10 → 0.78, r=20 → 0.86, r=50 → 0.91
f_pdv: 0.80
```

DEC-specific constants (only relevant when `pulsed_conversion: inductive_dec`):

```yaml
eta_dec: 0.85            # Electrical recovery efficiency (coil-to-cap)
```

### New costing constants (in costing_constants.yaml)

```yaml
# Pulsed inductive DEC — driver cost basis
c_cap_allin_per_joule: 2.0   # $/J_stored, NOAK all-in (caps+switches+charging+buswork)
                              # Sensitivity range: 1.5-4.0

# Pulsed inductive DEC — C220109 incremental markups
markup_switch_bidir: 0.06    # Bidirectional switch premium (% of driver cost)
markup_controls: 0.04        # FPGA/energy management upgrade (% of driver cost)
c_inv_per_kw_net: 150.0      # Grid-tie inverter ($/kW_net)

# Pulsed inductive DEC — CAS72 cap replacement
cap_shot_lifetime: 1.0e8     # Shots, NOAK baseline. Sensitivity: 1e7-1e9
```


## Migration: `dhe3_pulsed_frc.py` example

The existing example uses `ConfinementConcept.MAG_TARGET` with 9 cost
overrides to approximate a Helion-like FRC. After this revamp:

```python
model = CostModel(
    concept=ConfinementConcept.MAG_TARGET,
    fuel=Fuel.DHE3,
    pulsed_conversion=PulsedConversion.INDUCTIVE_DEC,
    costing_constants=cc,
)

result = model.forward(
    net_electric_mw=1000.0,
    n_mod=20,
    e_driver_mj=12.0,
    f_rep=1.0,
    eta_pin=0.95,
    eta_dec=0.85,
    eta_th=0.0,        # no thermal BOP (pure DEC, D-He3)
    f_pdv=0.80,
    mn=1.0,            # no breeding blanket
    p_cryo=0.0,        # copper coils
    p_target=0.0,      # in-situ FRC formation
    p_coils=0.5,
    # No cost_overrides needed for:
    #   CAS23 (auto-zeroed: eta_th=0 → p_the=0)
    #   CAS26 (auto-scaled to waste heat)
    #   C220107 (auto $/J_stored basis from INDUCTIVE_DEC)
    #   C220109 (auto-populated from DEC markups)
)
```

Remaining overrides, if any, would be for concept-specific geometry
choices (copper coils cost, building layout), not for the DEC mechanism
itself.


## Split driver deprecation

The IFE split driver (`p_implosion`, `p_ignition`, `eta_pin1`, `eta_pin2`)
is replaced by a single `e_driver_mj` + `eta_pin`. For the rare case of
genuinely different driver technologies, the user sets `e_driver_mj` to
the total per-pulse energy and `eta_pin` to the weighted-average
efficiency. The split-driver YAML keys and function signatures are removed.


## Scope boundaries

**In scope:**
- New `ConfinementFamily` enum (two families)
- `PulsedConversion` enum and per-concept defaults
- Two new pulsed power balance functions (forward + inverse)
- Costing changes for `INDUCTIVE_DEC` (C220107, C220109, CAS72)
- CAS23/CAS26 scaling basis change (p_et to p_the/p_th)
- CAS21 building scaling for power supply building under INDUCTIVE_DEC
- New costing constants and YAML defaults
- PowerTable extensions
- Updated `dhe3_pulsed_frc.py` example
- Removal of old `ife_*_power_balance` and `mif_*_power_balance` functions

**Out of scope:**
- Expanding the `ConfinementConcept` enum (separate task)
- Radiation physics model for pulsed burns (future upgrade from `f_rad`)
- Computing `f_pdv` from expansion ratio (future, just a parameter for now)
- Changes to the MFE power balance or steady-state concepts
- Changes to the 0D tokamak model
