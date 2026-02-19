# Verification Layer Design

## Goal

Add a pydantic-based input validation layer to 1costingfe, similar to pyfecons's `validation.py`. Every path through the system (forward(), run_costing(), direct construction) validates inputs before calculations run.

## Approach

Pydantic `CostingInput` model with built-in validators. Preserves the current `forward()` kwargs API and YAML template defaults. Validation runs automatically on `CostingInput` construction.

## CostingInput Model

### Required (no defaults)

| Field | Type | Constraint |
|-------|------|------------|
| concept | ConfinementConcept | valid enum |
| fuel | Fuel | valid enum |
| net_electric_mw | float | > 0 |

### Customer parameters (with defaults)

| Field | Type | Default | Constraint |
|-------|------|---------|------------|
| availability | float | 0.85 | > 0, <= 1 |
| lifetime_yr | float | 40.0 | > 0 |
| n_mod | int | 1 | >= 1, integer enforced |
| construction_time_yr | float | 6.0 | > 0 |
| interest_rate | float | 0.07 | > 0 |
| inflation_rate | float | 0.02 | (no constraint) |
| noak | bool | True | |
| cost_overrides | dict[str, float] | {} | |
| costing_overrides | dict[str, float] | {} | |

### Engineering parameters (optional, None = use YAML template)

**Common (all families):**
mn, eta_th, eta_p, f_sub, p_pump, p_trit, p_house, p_cryo, blanket_t, ht_shield_t, structure_t, vessel_t, plasma_t

**MFE only:**
p_input, eta_pin, eta_de, f_dec, p_coils, p_cool, axis_t, elon

**IFE only:**
p_implosion, p_ignition, eta_pin1, eta_pin2, p_target

**MIF only:**
p_driver, eta_pin, p_target, p_coils

## Validation Tiers

### Tier 1 — Field-level (pydantic built-in)

Type checking, range constraints via `Field()`. Fires automatically on construction.

### Tier 2 — Family-aware required fields

`@model_validator(mode='after')` checks that all required engineering params for the given confinement family are present (not None) after YAML template merge.

Family-required maps:
- MFE: p_input, eta_pin, eta_de, f_dec, p_coils, p_cool
- IFE: p_implosion, p_ignition, eta_pin1, eta_pin2, p_target
- MIF: p_driver, eta_pin, p_target, p_coils

### Tier 3 — Cross-field and physics checks

`@model_validator(mode='after')` runs derived physics checks using existing pure physics functions.

| Check | Severity | Condition |
|-------|----------|-----------|
| p_net < 0 | error | Plant consumes more than it produces |
| Q_sci < 1 | warning | Fusion power < injected heating |
| Recirculating fraction > 0.5 | warning | Excessive parasitic load |
| eta_th > 0.65 | warning | Unusually high thermal efficiency |
| eta_p > 0.95 | warning | Unusually high pumping efficiency |
| mn outside [1.0, 1.5] | warning | Atypical neutron multiplier |
| f_sub > 0.3 | warning | High subsystem fraction |

Errors raise `ValidationError`. Warnings emit via `warnings.warn()`.

## Integration Points

### CostModel.forward()

Keeps current kwargs signature. Internally:
1. Load YAML template defaults
2. Merge user overrides on top
3. Construct `CostingInput(**merged)` — pydantic validates
4. Proceed with costing

### run_costing(FusionTeaInput)

1. Convert FusionTeaInput fields to dict
2. Construct `CostingInput(**dict)` — validates
3. Pass to CostModel.forward()

### Direct construction

```python
inp = CostingInput(concept="tokamak", fuel="dt", net_electric_mw=1000)
# validates immediately
```

## File Structure

```
src/costingfe/
  validation.py      # NEW — CostingInput pydantic model + all validators
  exceptions.py      # NEW — FieldError, ValidationWarning
  model.py           # MODIFIED — construct CostingInput inside forward()
  adapter.py         # MODIFIED — construct CostingInput inside run_costing()
tests/
  test_validation.py # NEW — validation-specific tests
```

## Dependencies

Add `pydantic` to pyproject.toml.
