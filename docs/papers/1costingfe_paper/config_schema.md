# 1costingfe configuration schema (v1)

**Generated**: 2026-05-17
**Source**: `src/costingfe/types.py`, `src/costingfe/validation.py`, `src/costingfe/defaults.py`, `src/costingfe/data/defaults/*.yaml`

Every option that can be supplied to `CostModel(...)` or `model.forward(...)`. Override precedence: `forward(**overrides)` > YAML defaults > dataclass defaults. The "Where" column indicates whether the option is a constructor argument, a `forward()` keyword, an engineering YAML key, or a `CostingConstants` field.

---

## 1. `CostModel(...)` constructor arguments

| Column | Description | Vocabulary |
|---|---|---|
| **concept** | Confinement concept (required). Selects YAML template + physics path. | `tokamak` · `stellarator` · `mirror` · `laser_ife` · `zpinch` · `heavy_ion` · `mag_target` · `plasma_jet` · `pulsed_frc` · `maglif` · `theta_pinch` · `dense_plasma_focus` · `staged_zpinch` · `orbitron` · `polywell` |
| **fuel** | Primary fusion fuel cycle (required). | `dt` · `dd` · `dhe3` · `pb11` |
| **costing_constants** | Optional `CostingConstants` instance overriding all costing coefficients. | dataclass or `None` (YAML default) |
| **power_cycle** | Thermal conversion technology. Sets `eta_th`, `turbine_per_mw`, `heat_rej_per_mw` defaults. | `rankine` (eta_th=0.40) · `brayton_sco2` (eta_th=0.47) · `combined` (eta_th=0.53) |
| **pulsed_conversion** | Energy capture mode for pulsed concepts. Defaults per `CONCEPT_DEFAULT_CONVERSION`. | `thermal` · `inductive_dec` · `None` (concept default) |

Concept-to-family mapping (`CONCEPT_TO_FAMILY`):

| Family | Concepts |
|---|---|
| `steady_state` | tokamak, stellarator, mirror, orbitron, polywell |
| `pulsed` | laser_ife, zpinch, heavy_ion, mag_target, plasma_jet, pulsed_frc, maglif, theta_pinch, dense_plasma_focus, staged_zpinch |

Concept-default pulsed conversion (`CONCEPT_DEFAULT_CONVERSION`):

| Concept | Default conversion |
|---|---|
| laser_ife, zpinch, heavy_ion, mag_target, plasma_jet, maglif, dense_plasma_focus, staged_zpinch | `thermal` |
| pulsed_frc, theta_pinch | `inductive_dec` |

---

## 2. `model.forward(...)` customer parameters

These appear in the explicit signature and are validated by `CostingInput`.

| Column | Description | Vocabulary / Range |
|---|---|---|
| **net_electric_mw** | Target net electric output (required, MW). | `> 0` |
| **availability** | Capacity factor fraction. `None` -> concept default: 0.87 (mirror), 0.85 (others). | `(0, 1]` or `None` |
| **lifetime_yr** | Plant operating life (years). | `> 0`, default `40` |
| **n_mod** | Number of co-located reactor modules. | int `>= 1`, default `1` |
| **construction_time_yr** | EPC construction time (years). Drives IDC and labor scaling. | `> 0`, default `6.0` |
| **interest_rate** | Annual discount/financing rate. | `> 0`, default `0.07` |
| **inflation_rate** | Annual inflation rate. | float, default `0.02` |
| **noak** | NOAK vs FOAK costing flag (contingency, plant studies, recovery, B-11 supply). | `True` (NOAK) / `False` (FOAK) |
| **cost_overrides** | Per-account capital cost override (M$, keyed by CAS code, e.g. `"CAS22"` or `"C220101"`). | `dict[str, float]` |
| **override_reference_mw** | If set, `cost_overrides` are treated as absolute M$ at this reference power and rescaled. | float or `None` |
| **costing_overrides** | Per-coefficient override of any `CostingConstants` field. | `dict[str, float]` |

---

## 3. Engineering parameters (YAML keys, all overridable via `forward(**kwargs)`)

Loaded from `src/costingfe/data/defaults/{family}_{concept}.yaml` and merged into the params dict at `model.py:419`. Override precedence: kwargs win over YAML.

### 3a. Common (all concepts)

| Column | Description | Units / Vocabulary |
|---|---|---|
| **mn** | Neutron energy multiplier (blanket exothermic). | dimensionless, typical 1.0-1.5 |
| **f_sub** | Subsystem auxiliary load fraction. | dimensionless, typical < 0.3 |
| **eta_pin** | Heating-system wall-plug efficiency. | `(0, 1)` |
| **p_pump** | Pumping power (coolant). | MW |
| **p_trit** | Tritium plant power. | MW |
| **p_house** | Housekeeping/HVAC power. | MW |
| **p_cryo** | Cryogenic system power. | MW (`0` for copper coils) |
| **blanket_t** | Blanket radial thickness. | m |
| **ht_shield_t** | High-temperature shield thickness. | m |
| **structure_t** | Primary structure thickness. | m |
| **vessel_t** | Vacuum vessel wall thickness. | m |
| **plasma_t** | Plasma minor radius (steady-state) / radial half-extent (pulsed). | m |
| **R0** | Major radius (or characteristic length). | m |
| **construction_time_yr** | YAML default for construction time (overridable by `forward()` arg). | yr |
| **blanket_form** | Blanket cassette/canister architecture. Drives CAS22.01 structure cost via `BlanketForm.structure_factor`. | `liquid_metal` (1.0x) · `molten_salt` (1.3x) · `solid_breeder` (1.2x) · `none` (0.0x) |
| **blanket_fill** | Bulk material chemistry that fills the cassettes. Drives CAS27 inventory cost via `BlanketFill.fill_factor`. | `pbli` (1.0x) · `li` (2.0x) · `flibe` (5.0x) · `be_ceramic` (13.0x) · `ceramic_only` (3.0x) · `none` (0.0x) |

The form-fill pair is validated by `CostingInput.check_blanket_compatibility`. Compatibility:

| BlanketForm | Compatible fills | Default fill (via `form.default_fill`) |
|---|---|---|
| `liquid_metal` | `pbli`, `li` | `pbli` |
| `molten_salt` | `flibe` | `flibe` |
| `solid_breeder` | `be_ceramic`, `ceramic_only` | `be_ceramic` |
| `none` | `none` | `none` |

Additional validation rules:
- ERROR: DT fuel requires non-`none` form and non-`none` fill (no breeding -> no DT plant).
- WARNING: DHe3 / pB11 with non-`none` blanket (aneutronic fuels do not need breeding).

Per-concept defaults: 10 concepts default to `(liquid_metal, pbli)` (tokamak, stellarator, mirror, laser_ife, zpinch, heavy_ion, mag_target, plasma_jet, maglif, staged_zpinch); 5 default to `(none, none)` (orbitron, polywell, pulsed_frc, theta_pinch, dense_plasma_focus). See `docs/account_justification/CAS27_special_materials.md` for the multiplier calibration.

### 3b. Steady-state (MFE) only

| Column | Description | Units / Vocabulary |
|---|---|---|
| **p_input** | Total auxiliary heating power. | MW |
| **p_nbi** | Neutral-beam heating share. | MW (sum with others = p_input) |
| **p_ecrh** | ECRH heating share. | MW |
| **p_icrf** | ICRF heating share. | MW |
| **p_lhcd** | Lower-hybrid current-drive share. | MW |
| **eta_p** | Pumping efficiency. | `(0, 1)` |
| **eta_de** | DEC efficiency (charged-particle conversion). | `(0, 1)` |
| **f_dec** | DEC fraction of charged-particle power. | `[0, 1]` (`0` disables DEC) |
| **p_coils** | Coil resistive/refrigerator parasitic. | MW (large for copper, small for SC) |
| **p_cool** | Primary loop cooling power. | MW |
| **elon** | Elongation kappa. | dimensionless |
| **b_max** | Peak field on conductor. | T |
| **r_coil** | Effective winding bore radius (pyFECONs calibration). | m |
| **coil_material** | Conductor technology. Sets default cost/kAm and cryo expectation. | `rebco_hts` ($50/kAm) · `nb3sn` ($7) · `nbti` ($7) · `copper` ($1) |
| **n_coils** | Override coil count (else inferred from concept). | int or `None` |

### 3c. Steady-state (MFE) plasma / impurity model

Used by `compute_p_rad`. Required when the 0D physics path is not active.

| Column | Description | Units / Vocabulary |
|---|---|---|
| **n_e** | Volume-averaged electron density. | m^-3 |
| **T_e** | Volume-averaged electron temperature. | keV |
| **Z_eff** | Effective charge. | dimensionless, `>= 1` |
| **plasma_volume** | Plasma volume. | m^3 |
| **B** | On-axis magnetic field. | T |
| **wall_material** | First-wall material (sets sputtering / impurity model). | `W` · `C` · `Be` · `Mo` · `SiC` · `Li` |
| **T_edge** | Edge ion temperature at wall. | keV (default 0.05 = 50 eV detached) |
| **tau_ratio** | tau_imp / tau_E. | dimensionless, default 3 |
| **seeded_impurities** | Deliberately injected species, e.g. `{Ar: 0.002}`. | `dict[str, float]` |
| **R_w** | Wall reflectivity (synchrotron). | `[0, 1]`, typical 0.6-0.8 |
| **chamber_length** | Mirror/orbitron axial length (used in CAS22 vessel + DEC sizing). | m |

### 3d. Tokamak 0D model

Activated when `use_0d_model: true` (tokamak only).

| Column | Description | Units / Vocabulary |
|---|---|---|
| **use_0d_model** | Toggle the 0D plasma model. | bool, default `False` |
| **q95** | Safety factor at 95% flux surface. | dimensionless |
| **f_GW** | Greenwald density fraction. | `(0, 1.2]` typically |
| **M_ion** | Average ion mass. | AMU |
| **lambda_q** | SOL power width at midplane. | m |
| **disruption_rate_base** | Disruptions per FPY far from limits. | dim'less |
| **disruption_steepness** | Exponential rise toward limits. | dim'less |
| **disruption_damage** | Component-life fraction per disruption. | `[0, 1]` |
| **disruption_downtime** | Hours of downtime per disruption. | hours |

### 3e. Pulsed-family only

| Column | Description | Units / Vocabulary |
|---|---|---|
| **q_eng** | Target engineering Q (used by pulsed inverse). | dimensionless |
| **f_rep** | Pulse repetition rate. | Hz |
| **f_rad** | Radiation fraction of charged-particle power. | `[0, 1]` |
| **f_pdv** | PdV work fraction of charged-particle energy. | `[0, 1]`, default 0.80 |
| **eta_dec** | DEC efficiency for pulsed inductive conversion. | `[0, 1]` |
| **p_target** | Target factory power load. | MW |
| **p_coils** | (Pulsed) coil/driver coil parasitic. | MW (some concepts only) |
| **pulsed_conversion** | YAML hint (the canonical setter is the constructor arg). | `thermal` · `inductive_dec` |
| **e_driver_mj** | Per-pulse driver energy (set by physics path, not user-facing). | MJ |
| **e_stored_mj** | Per-pulse cap-bank energy (inductive DEC). | MJ |

### 3f. Fuel-cycle burn fractions (all families)

| Column | Description | Default |
|---|---|---|
| **burn_fraction** | Single-pass burn fraction. Per concept; MFE 0.05, ICF 0.25-0.30, magneto-inertial 0.01-0.15. | per-concept |
| **fuel_recovery** | Fraction of unburned fuel recovered and recycled (NOAK). | 0.99 |
| **dd_f_T** | DD branch: tritium-bearing fraction. | 0.969 |
| **dd_f_He3** | DD branch: He-3 fraction. | 0.689 |
| **dhe3_dd_frac** | D-He3: parasitic DD reaction fraction. | 0.131 |
| **dhe3_f_T** | D-He3: residual tritium fraction. | 0.5 |
| **dhe3_f_He3** | D-He3: He-3 burnup fraction. | 0.1 |
| **pb11_f_alpha_n** | p-B11 secondary `alpha + alpha -> n` branch. | 0.0 |
| **pb11_f_p_n** | p-B11 secondary `p + alpha -> n` branch. | 0.0 |

---

## 4. `CostingConstants` (loaded from `costing_constants.yaml`; field-level override via `costing_overrides=` or `replace()`)

Categorized by CAS account. Every field is overridable.

### CAS10 -- Preconstruction

| Field | Description | Default |
|---|---|---|
| `site_permits` | Site permitting (M$). | 3.0 |
| `plant_studies_foak` / `plant_studies_noak` | EIA/plant studies (M$). | 20.0 / 4.0 |
| `plant_permits` · `plant_reports` · `other_precon` | Misc preconstruction (M$). | 2.0 / 1.0 / 1.0 |
| `land_intensity` · `land_cost` | Acres per MWe; $/acre. | 0.25 · 10 000 |
| `licensing_cost_<fuel>` | Licensing capital (M$). | dt 5 · dd 3 · dhe3 1 · pb11 0.1 |
| `licensing_time_<fuel>` | Licensing duration (yr). | dt 2.0 · dd 1.5 · dhe3 0.75 · pb11 0.0 |

### CAS21 -- Buildings

| Field | Description | Default |
|---|---|---|
| `building_costs` | Per-building, per-fuel M$ at 1 GWe + scaling exponent. | loaded from YAML |

### CAS22 -- Reactor plant equipment (volume + per-MW costs)

| Field | Description | Default |
|---|---|---|
| `blanket_unit_cost_<fuel>` | Blanket M$/m^3. | dt 0.60 · dd 0.30 · dhe3 0.08 · pb11 0.05 |
| `shield_unit_cost` | Shield M$/m^3. | 0.74 |
| `heating_nbi_per_mw` · `heating_icrf_per_mw` · `heating_ecrh_per_mw` · `heating_lhcd_per_mw` | M$/MW of aux heating. | 7.06 · 4.15 · 5.0 · 4.0 |
| `driver_laser_per_mj` · `driver_heavy_ion_per_mj` · `driver_plasma_jet_per_mj` · `driver_staged_zpinch_per_mj` | Laser / accelerator / EM-gun driver M$/MJ of pulse energy (rep-rate-independent). | 80 / 60 / 4 / 1.5 |
| `driver_mag_target_per_mw` | Mechanical-injector driver M$/MW of average power (throughput-scaled). | 3 |
| `laser_preheat_per_mj` | Laser preheat add-on M$/MJ of preheat pulse energy (MagLIF; scaled by per-concept `e_preheat_mj`). | 80 |
| `structure_unit_cost` · `vessel_unit_cost` | M$/m^3. | 0.15 · 0.72 |
| `power_supplies_base` · `divertor_base` · `target_factory_base` | Base capital (M$ at 1 GWe). | 80 · 60 · 244 |
| `dec_base` · `dec_grid_cost` | DEC base + grid module cost (M$). | 125 · 12 |
| `dec_grid_lifetime_<fuel>` | Grid replacement interval (FPY). | dt 2 · dd 3 · dhe3 4 · pb11 3 |
| `c_cap_allin_per_joule` | Pulsed cap bank $/J_stored. | 0.5 |
| `markup_switch_bidir` · `markup_controls` · `c_inv_per_kw_net` | Inductive-DEC markups + inverter $/kW. | 0.06 · 0.04 · 150 |
| `cap_shot_lifetime` | Capacitor shot life. | 1e8 |
| `electrode_shot_lifetime` · `electrode_replace_frac` | EM-gun electrode shot life and consumable share of C220104 (CAS72 replacement, staged Z-pinch / plasma jet). | 1e8 / 0.5 |
| `remote_handling_<fuel>_base` | Remote handling (M$ at 1 GWe). | dt 150 · dd 100 · dhe3 30 · pb11 20 |
| `installation_frac` | C220111 labor as fraction of subtotal. | 0.14 |
| `multi_unit_labor_factor` | Labor cost of module N beyond #1. | 0.92 |
| `core_lifetime_<fuel>` | Core component life (FPY). | dt 5 · dd 10 · dhe3 30 · pb11 50 |
| `replaceable_accounts` | CAS22 sub-accounts that get replaced. | `("C220101", "C220108")` |
| `f_rad_<fuel>` (pulsed) | Pulsed radiation fraction. | dt 0.10 · dd 0.08 · dhe3 0.05 · pb11 0.15 |
| `f_rad_fus_pb11` · `f_rad_fus_dhe3` | Steady-state brem fraction of P_fus. | 0.83 · 0.24 |
| `f_pdv` | PdV work fraction (pulsed). | 0.80 |
| `fuel_handling_<fuel>_base` | CAS22.05 fuel handling (M$ at 1 GWe). | dt 120 · dd 60 · dhe3 40 · pb11 15 |

### CAS23-26 -- Balance of plant

| Field | Description | Default |
|---|---|---|
| `turbine_per_mw` | Turbine + condenser ($M/MWe gross). | 0.19764 |
| `electric_per_mw` | Switchyard + transformers ($M/MWe). | 0.08418 |
| `misc_per_mw` | HVAC + fire + air ($M/MWe). | 0.05124 |
| `heat_rej_per_mw` | Cooling tower + circ water ($M/MWe). | 0.03416 |

### CAS27 -- Special materials

| Field | Description | Default |
|---|---|---|
| `special_materials_<fuel>` | Initial reactor material inventory (M$ at 1 GWe). | dt 15 · dd 2 · dhe3 1 · pb11 0 |

### CAS28 -- Digital twin

| Field | Description | Default |
|---|---|---|
| `digital_twin` | Fixed (M$). | 5.0 |

### CAS29 -- Contingency

| Field | Description | Default |
|---|---|---|
| `contingency_rate_foak` · `contingency_rate_noak` | Fraction of direct costs. | 0.10 / 0.0 |

### CAS30 -- Indirect

| Field | Description | Default |
|---|---|---|
| `indirect_fraction` | Fraction of direct costs. | 0.20 |
| `reference_construction_time` | Reference build time for indirect scaling. | 6.0 yr |

### CAS40 -- Owner's costs

| Field | Description | Default |
|---|---|---|
| `owner_cost_<fuel>` | M$ at 1 GWe. | dt 39 · dd 31 · dhe3 23 · pb11 20 |

### CAS50 -- Supplementary

| Field | Description | Default |
|---|---|---|
| `shipping_frac` | Frac of CAS20. | 0.015 |
| `spare_parts_frac_<fuel>` | Frac of CAS22-28. | dt 0.030 · dd 0.025 · dhe3 0.015 · pb11 0.010 |
| `tax_frac` | Frac of CAS20. | 0.01 |
| `construction_insurance_frac` | Frac of (CAS20+CAS30). | 0.015 |
| `startup_fuel_<fuel>` | M$ at 1 GWe. | dt 40 · dd 0.1 · dhe3 10 · pb11 0.1 |
| `decom_provision_<fuel>` | M$ at 1 GWe. | dt 127 · dd 93 · dhe3 65 · pb11 53 |

### CAS70 -- O&M (annual)

| Field | Description | Default |
|---|---|---|
| `om_cost_<fuel>` | M$/yr at 1 GWe ref (scales as P^0.5). | dt 52 · dd 39 · dhe3 26 · pb11 24 |

### CAS80 -- Fuel (annual)

| Field | Description | Default |
|---|---|---|
| `u_deuterium` · `u_li6` · `u_he3` · `u_protium` · `u_b11` (FOAK) · `u_b11_noak` | Isotope unit cost ($/kg). | 2 175 · 1 000 · 2 000 000 · 5 · 10 000 · 75 |

---

## 5. Result objects (read-only)

What `forward()` returns. Not configurable, listed for completeness.

| Object | Field | Description |
|---|---|---|
| `ForwardResult` | `power_table` · `costs` · `params` · `overridden` · `cas22_detail` · `plasma_state` | Top-level container. |
| `PowerTable` | `p_fus, p_ash, p_neutron, p_rad, p_wall, p_dee, p_dec_waste, p_th, p_the, p_et, p_loss, p_net, p_pump, p_sub, p_aux, p_input, p_coils, p_cool, p_cryo, p_target, q_sci, q_eng, rec_frac, e_driver_mj, e_stored_mj, f_rep, f_ch` | All power flows from Layer 2. |
| `CostResult` | `cas10..cas90, cas20, cas71, cas72, total_capital, lcoe, overnight_cost` | Cost breakdown in M$ + $/MWh + $/kW. |

---

## 6. Editability rule (summary)

```python
CostModel(concept=..., fuel=..., power_cycle=..., pulsed_conversion=..., costing_constants=cc)
    .forward(net_electric_mw=..., availability=..., ..., **engineering_overrides,
             cost_overrides={...}, costing_overrides={...})
```

- Every key in `data/defaults/<concept>.yaml` -> override as a `forward()` kwarg.
- Every field on `CostingConstants` -> override via `costing_overrides={...}` or by constructing `CostingConstants(...)` and passing it.
- Every CAS line item -> override the M$ directly via `cost_overrides={"CAS22": 2200.0}` or sub-account `cost_overrides={"C220101": 400.0}`.
- Power-cycle defaults (`eta_th`, `turbine_per_mw`, `heat_rej_per_mw`) come from the `PowerCycle` enum but can be overridden as explicit kwargs to `forward()`.
