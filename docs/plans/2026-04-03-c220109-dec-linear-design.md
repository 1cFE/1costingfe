# C220109: Direct Energy Converter for Linear Devices — Design Spec

**Date:** 2026-04-03
**Status:** Draft
**Scope:** Populate C220109 for add-on DECs on linear devices (mirrors, steady-state FRCs). Does not cover inductive DEC for pulsed concepts (task 2).

---

## Summary

C220109 is currently hardcoded to $0. This spec populates it with a
cost model for add-on direct energy converters (venetian blind, TWDEC,
inverse cyclotron) on linear devices with directed axial exhaust.

The model is:
- Gated on `f_dec > 0` (no fuel gating — user decides whether DEC is worth it)
- Scaled by `p_dee` (DEC electric output), not by total plant power
- DEC type (venetian blind vs TWDEC vs ICC) is not distinguished —
  cost ranges overlap ($73-140M at 1 GWe); efficiency differences
  flow through `eta_de`
- Grid replacement is a separate CAS72 term with fuel-dependent lifetime

---

## Physics layer

No changes. The MFE power balance already computes `p_dee`:

```
p_transport = p_ash + p_input_eff - p_rad
p_dee = f_dec * eta_de * p_transport
p_wall = (1 - f_dec) * p_transport
```

`f_dec` is the fraction of transport power routed to the DEC.
`eta_de` is the DEC conversion efficiency. Both are already in the
mirror defaults (`f_dec=0.3`, `eta_de=0.60`).

---

## Cost layer changes

### New constants in `CostingConstants` (`defaults.py`)

```python
# C220109: DEC add-on base cost (M$ at 400 MWe DEC electric output)
# Source: docs/account_justification/CAS220109_direct_energy_converter.md
# Subsystem build-up: grids + power conditioning + incremental vacuum/tank + heat collection
# Range: $79-128M at ~400 MWe DEC output (1 GWe DHe3 mirror reference)
dec_base: float = 100.0  # M$ at P_DEE_REF

# DEC grid/collector module cost (M$ at 400 MWe DEC output)
# The replaceable portion of the DEC — grids degrade under particle bombardment.
# Rest of DEC (vacuum, cryo, power conditioning, tank) is long-lived.
# Source: Hoffman 1977 grid module costs, escalated and adjusted for mirror add-on
dec_grid_cost: float = 12.0  # M$ at P_DEE_REF

# DEC grid lifetime (FPY) — HIGH UNCERTAINTY
# No reactor-scale data exists for any DEC grid type. Conservative estimates.
# Primary degradation: sputtering erosion + helium blistering from charged
# particle exhaust. Neutron damage is additive for DT/DD.
# Sensitivity range: 0.5x to 3x these values.
dec_grid_lifetime_dt: float = 2.0    # Sputtering + 14.1 MeV neutron damage
dec_grid_lifetime_dd: float = 3.0    # Sputtering + 2.45 MeV neutron damage
dec_grid_lifetime_dhe3: float = 4.0  # 14.7 MeV proton sputtering + He blistering
dec_grid_lifetime_pb11: float = 3.0  # 2.9 MeV alpha sputtering + severe He blistering (3 alpha per event)
```

Plus a `dec_grid_lifetime(fuel)` accessor method, same pattern as `core_lifetime(fuel)`.

### C220109 computation in `cas22.py`

New inputs to `cas22_reactor_plant_equipment`: `f_dec`, `p_dee`.

```python
# C220109: Direct Energy Converter (add-on for linear devices)
# Scales with DEC electric output. Covers: grid/collector modules,
# DC-AC power conditioning, incremental vacuum/cryo, incremental
# tank volume, heat collection. Applicable to venetian blind, TWDEC,
# or ICC — cost ranges overlap; efficiency differences in eta_de.
P_DEE_REF = 400.0  # MW reference DEC electric output
if f_dec > 0 and p_dee > 0:
    c220109 = cc.dec_base * (p_dee / P_DEE_REF) ** 0.7
else:
    c220109 = 0.0
```

The 0.7 exponent reflects:
- Grid modules scale ~linearly with area (proportional to power)
- Power conditioning scales ~linearly with output
- Tank and vacuum scale sub-linearly (surface-to-volume)
- Consistent with other vendor-purchased power systems in CAS22

### CAS72: DEC grid replacement

Additive to existing blanket/divertor replacement. Independent cycle.

```python
# DEC grid replacement (if DEC present)
if c220109 > 0:
    dec_grid = cc.dec_grid_cost * (p_dee / P_DEE_REF) ** 0.7
    dec_grid_life = cc.dec_grid_lifetime(fuel)
    # Convert FPY to calendar years using availability
    t_replace_dec = dec_grid_life / availability
    annual_replace_dec = n_mod * dec_grid / t_replace_dec
else:
    annual_replace_dec = 0.0
```

The grid cost uses the same `(p_dee / P_DEE_REF) ** 0.7` scaling as
the total DEC — grids are a fixed fraction of DEC hardware.

### Model plumbing (`model.py`)

- Pass `f_dec` and `p_dee` (from PowerTable) to `cas22_reactor_plant_equipment`
- Add DEC grid replacement term to CAS72 computation
- Ensure `cost_overrides["C220109"]` still works (existing override mechanism)

---

## Defaults impact

### Mirror with `f_dec=0.3`, `eta_de=0.60`

At 1 GWe net DT tokamak-equivalent, a DHe3 mirror would have
significant `p_dee`. C220109 would auto-populate, adding $50-150M
depending on plant size and DEC output. DEC grid replacement adds
an annualized O&M term.

### Tokamak/stellarator (`f_dec=0.0`)

No change — C220109 remains $0.

### IFE/MIF (`f_dec` not in power balance)

No change — IFE/MIF power balances don't have `f_dec`. This is
correct for now; task 2 (IFE/MIF revamp) will address whether
pulsed concepts get DEC capability.

---

## Justification doc update

Update `docs/account_justification/CAS220109_direct_energy_converter.md`:

1. Remove fuel-gating language ("$0 for DT/DD"). Replace with: C220109
   scales with DEC electric output regardless of fuel; economic
   viability is a user judgment, not a model constraint.
2. Replace fuel-specific base costs table with the `dec_base * (p_dee / P_DEE_REF) ** 0.7` formula.
3. Add grid lifetime section with the conservative fuel-dependent values
   and high-uncertainty flag.
4. Note that DEC type (venetian blind, TWDEC, ICC) is not distinguished
   in the cost model — cost ranges overlap, efficiency differences
   flow through `eta_de`.

---

## What this does NOT change

- Physics layer (power balance routing already correct)
- IFE/MIF power balances (no DEC pathway — task 2)
- Inductive DEC for pulsed concepts (task 2)
- DEC type enum (not needed — cost ranges overlap)
- `f_dec` or `eta_de` default values for any concept

---

## Files modified

| File | Change |
|---|---|
| `src/costingfe/defaults.py` | Add `dec_base`, `dec_grid_cost`, `dec_grid_lifetime_*`, accessor method |
| `src/costingfe/data/defaults/costing_constants.yaml` | Add same constants to YAML |
| `src/costingfe/layers/cas22.py` | Add `f_dec`, `p_dee` params; compute C220109 |
| `src/costingfe/model.py` | Pass `f_dec`, `p_dee` to cas22; add DEC grid replacement to CAS72 |
| `docs/account_justification/CAS220109_direct_energy_converter.md` | Remove fuel gating, add scaling formula, add grid lifetime |
| `tests/test_cas22.py` | Test C220109 populates when f_dec > 0, zeros when f_dec = 0 |
| `tests/test_model.py` | Test DEC grid replacement in CAS72 |
