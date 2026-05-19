# Blanket configuration: two-axis form/fill schema

**Date:** 2026-05-17
**Status:** Design — approved for implementation
**Scope:** Add `BlanketForm` and `BlanketFill` enums; wire into CAS22.01 (structure) and CAS27 (fill inventory); validate via Pydantic.

## Motivation

The cost model currently has one parameter per fuel for the blanket: `blanket_unit_cost_<fuel>` (M$/m³ structure, CAS22.01) and `special_materials_<fuel>` (M$ at 1 GWe fill inventory, CAS27). These per-fuel values silently bake in a specific blanket configuration assumption (PbLi for DT, no breeder for aneutronic). The fusion concept ontology (`1costingfe_fusion_tea_interaction/CONCEPT_ONTOLOGY.md`) distinguishes four blanket families — liquid metal, molten salt, solid breeder, none — and the cost spread between them is large (HCPB blanket fill is ~13× PbLi). Today the only way to capture that spread is a manual `cost_overrides={"CAS27": 200.0}`. This proposal makes the choice a first-class configuration option.

## Schema

Two orthogonal enums in `src/costingfe/types.py`:

```python
class BlanketForm(Enum):
    LIQUID_METAL = "liquid_metal"
    MOLTEN_SALT = "molten_salt"
    SOLID_BREEDER = "solid_breeder"
    NONE = "none"

    @property
    def structure_factor(self) -> float:
        return _BLANKET_STRUCTURE_FACTOR[self]


class BlanketFill(Enum):
    PBLI = "pbli"
    LI = "li"
    FLIBE = "flibe"
    BE_CERAMIC = "be_ceramic"      # HCPB: Be pebbles + Li ceramic
    CERAMIC_ONLY = "ceramic_only"  # WCCB: Li ceramic only (no Be multiplier)
    NONE = "none"

    @property
    def fill_factor(self) -> float:
        return _BLANKET_FILL_FACTOR[self]


_BLANKET_FORM_VALID_FILLS = {
    BlanketForm.LIQUID_METAL:  {BlanketFill.PBLI, BlanketFill.LI},
    BlanketForm.MOLTEN_SALT:   {BlanketFill.FLIBE},
    BlanketForm.SOLID_BREEDER: {BlanketFill.BE_CERAMIC, BlanketFill.CERAMIC_ONLY},
    BlanketForm.NONE:          {BlanketFill.NONE},
}

_BLANKET_FORM_DEFAULT_FILL = {
    BlanketForm.LIQUID_METAL:  BlanketFill.PBLI,
    BlanketForm.MOLTEN_SALT:   BlanketFill.FLIBE,
    BlanketForm.SOLID_BREEDER: BlanketFill.BE_CERAMIC,
    BlanketForm.NONE:          BlanketFill.NONE,
}
```

**Decomposition rationale.** `BlanketForm` selects the cassette/canister architecture and drives CAS22.01 (the manufactured hardware). `BlanketFill` selects the bulk material chemistry and drives CAS27 (inventory). They are orthogonal physically, with a compatibility constraint expressed by `_BLANKET_FORM_VALID_FILLS` (you cannot put PbLi in a solid-breeder pebble bed).

**Surface area.** Follows the `coil_material` pattern, not the `power_cycle` pattern: both axes are YAML defaults per concept and `forward()` kwarg overrides. They do not switch whole defaults dicts; they just feed two cost multipliers.

## Cost wiring

Multipliers on the existing per-fuel constants, preserving the existing calibration for the `(LIQUID_METAL, PBLI)` baseline.

### CAS22.01 — blanket structure

At `cas22.py:136-145`, the line becomes:

```python
c220101 = (
    blanket_unit[fuel]
    * blanket_form.structure_factor
    * blanket_vol
    * (p_th / P_TH_REF) ** 0.6
)
```

`cas22_reactor_plant_equipment(...)` takes `blanket_form: BlanketForm` as a required keyword parameter.

### CAS27 — fill inventory

At `costs.py:144-163`:

```python
def cas27_special_materials(cc, p_net, fuel, blanket_fill):
    base = {
        Fuel.DT: cc.special_materials_dt,
        Fuel.DD: cc.special_materials_dd,
        Fuel.DHE3: cc.special_materials_dhe3,
        Fuel.PB11: cc.special_materials_pb11,
    }
    return base[fuel] * blanket_fill.fill_factor * (p_net / 1000.0)
```

`blanket_fill: BlanketFill` is required (no Python keyword default per the project rule).

### Materialization in `model.py`

Adjacent to the existing `coil_material` lookup at `model.py:562`:

```python
blanket_form = BlanketForm(params["blanket_form"])
blanket_fill = BlanketFill(params["blanket_fill"])
```

Missing YAML key → KeyError (surfaces YAML coverage gaps loudly). Invalid string → ValueError from the Enum constructor. Compatibility and fuel-physics checks happen earlier in the `CostingInput` validation step.

## Multiplier values

All factors are relative to the DT/PbLi baseline (`blanket_unit_cost_dt = $0.60/m³`, `special_materials_dt = $15M`), which gives the `(LIQUID_METAL, PBLI)` combination factors `(1.0, 1.0)` and preserves the existing calibration.

### `structure_factor` (CAS22.01)

| BlanketForm | structure_factor | Rationale |
|---|---:|---|
| `liquid_metal` | 1.0 | Baseline. RAFM steel flow channels + manifolds + W FW armor. |
| `molten_salt` | 1.3 | Hastelloy-N corrosion liner on FLiBe-wetted surfaces. Source: INEEL/EXT-99-00331. |
| `solid_breeder` | 1.2 | Pebble-bed canisters: separate breeder/multiplier zones, He coolant manifolds. Source: EUROfusion HCPB cost basis. |
| `none` | 0.0 | No blanket structure. |

### `fill_factor` (CAS27)

| BlanketFill | fill_factor | Rationale (baseline = DT/PbLi $15M) |
|---|---:|---|
| `pbli` | 1.0 | Baseline. $12M PbLi at $3/kg × 4 000 t + $3M enriched-Li-6 premium. |
| `li` | 2.0 | Self-cooled Li. Lower inventory mass (~300 t) but Li-6 enrichment to ~30%. Center ~$30M. |
| `flibe` | 5.0 | 2LiF-BeF2 melt, ~1 000 t at $50-150/kg with Li-7 enrichment. Center ~$75M. |
| `be_ceramic` | 13.0 | HCPB. ~300 t Be at $600/kg + Li-ceramic pebbles. ~$200M. |
| `ceramic_only` | 3.0 | WCCB. Li-ceramic pebbles without Be multiplier. ~$45M. |
| `none` | 0.0 | Aneutronic, no breeder. |

**Calibration caveat.** These are first-pass values calibrated against limited public data. Spread is wide for some entries (`li` 0.8-10×, `flibe` 2-20×) driven by enrichment choices. Center-of-range values shipped, same convention as `CoilMaterial.default_cost_per_kAm` and `c_cap_allin_per_joule`. The existing `docs/account_justification/CAS27_special_materials.md` will be extended with a "Blanket configuration multipliers" section documenting these derivations.

## Defaults per concept YAML

Both keys are added explicitly to every concept YAML. No Python-side default; missing key → KeyError. Per-concept defaults reflect each concept's natural blanket choice rather than blindly defaulting to PbLi everywhere.

| Concept YAML | `blanket_form` | `blanket_fill` | Rationale |
|---|---|---|---|
| `steady_state_tokamak` | `liquid_metal` | `pbli` | EU-DEMO baseline |
| `steady_state_stellarator` | `liquid_metal` | `pbli` | Type One, Proxima, Renaissance |
| `steady_state_mirror` | `liquid_metal` | `pbli` | Realta / GDT-class |
| `steady_state_orbitron` | `none` | `none` | Zephyr D-He3; charged-particle DEC |
| `steady_state_polywell` | `none` | `none` | Electrostatic, no blanket |
| `pulsed_laser_ife` | `liquid_metal` | `pbli` | HYLIFE-style |
| `pulsed_zpinch` | `liquid_metal` | `pbli` | ZAP liquid Li-Pb |
| `pulsed_heavy_ion` | `liquid_metal` | `pbli` | HIBALL |
| `pulsed_mag_target` | `liquid_metal` | `pbli` | GFU Pb-Li |
| `pulsed_plasma_jet` | `liquid_metal` | `pbli` | PLX Li |
| `pulsed_pulsed_frc` | `none` | `none` | Helion D-He3 |
| `pulsed_maglif` | `liquid_metal` | `pbli` | PAC D-T |
| `pulsed_theta_pinch` | `none` | `none` | Aneutronic-oriented |
| `pulsed_dense_plasma_focus` | `none` | `none` | LPP p-B11 |
| `pulsed_staged_zpinch` | `liquid_metal` | `pbli` | D-T |

## Backward compatibility

The 10 concepts that retain `(liquid_metal, pbli)` produce bit-identical output to current code across all four fuels — multipliers are `(1.0, 1.0)` no-ops.

The 5 concepts moving to `(none, none)` (orbitron, polywell, pulsed_frc, theta_pinch, dense_plasma_focus) experience a deliberate change: CAS22.01 blanket-term contribution drops to 0 and CAS27 drops to 0. This is correct — the previous numbers silently assumed a PbLi blanket on concepts that physically do not have one. For DHe3/pB11 the absolute delta is small (the per-fuel base is already aneutronic-flavored at $0.05-0.08/m³ and $0-1M). For a DT sensitivity sweep on these concepts, the user must explicitly override to a non-`none` blanket.

## Validation

All checks live in `CostingInput` (`validation.py`) as Tier 3 model validators. `forward()` already calls `CostingInput` at `model.py:467`, so these checks fire in the normal path. The adapter API (`adapter.py:76`) also gets coverage for free.

Two new fields:

```python
blanket_form: BlanketForm | None = None
blanket_fill: BlanketFill | None = None
```

Pydantic v2 auto-coerces YAML strings into the enums. `None` at the Pydantic layer is fine because presence is enforced by the YAML lookup at the materialization point.

New model_validator:

```python
@model_validator(mode="after")
def check_blanket_compatibility(self):
    if self.blanket_form is None or self.blanket_fill is None:
        return self  # presence checked at materialization, not here

    valid = _BLANKET_FORM_VALID_FILLS[self.blanket_form]
    if self.blanket_fill not in valid:
        raise ValueError(
            f"blanket_fill={self.blanket_fill.value!r} not valid for "
            f"blanket_form={self.blanket_form.value!r}. "
            f"Valid: {sorted(f.value for f in valid)}"
        )

    if self.fuel == Fuel.DT and (
        self.blanket_form == BlanketForm.NONE
        or self.blanket_fill == BlanketFill.NONE
    ):
        raise ValueError(
            "DT fuel requires a breeding blanket "
            f"(got blanket_form={self.blanket_form.value!r}, "
            f"blanket_fill={self.blanket_fill.value!r})."
        )

    if self.fuel in (Fuel.DHE3, Fuel.PB11) and self.blanket_form != BlanketForm.NONE:
        warnings.warn(
            f"{self.fuel.value} with blanket_form={self.blanket_form.value!r}: "
            "aneutronic fuels do not need a breeding blanket.",
            stacklevel=2,
        )

    return self
```

Rules summary: schema-compatibility (ERROR), DT-needs-breeder (ERROR), aneutronic-with-blanket (WARNING).

**Out of scope.** Wiring `coil_material` and `wall_material` into `CostingInput` has the same shape and is recommended as a follow-up, but is not part of this change so the two cleanups do not tangle.

## Tests

### `tests/test_blanket_config.py` (new file)

1. **`test_enum_factors`** — pin every `structure_factor` and `fill_factor` value, plus every entry of `_BLANKET_FORM_VALID_FILLS` and `_BLANKET_FORM_DEFAULT_FILL`. Catches accidental drift.
2. **`test_dt_tokamak_blanket_cost_scaling`** — parameterized over the six valid form/fill pairs. Asserts that `cas22_detail["C220101"]` and `costs.cas27` scale by the documented multipliers vs the `(liquid_metal, pbli)` baseline. This is the primary correctness test.

### `tests/test_validation.py` (extend existing file)

3. **`test_blanket_fill_must_match_form`** — `(solid_breeder, pbli)` raises ValidationError.
4. **`test_dt_requires_breeding_blanket`** — `DT + (none, none)` raises.
5. **`test_dt_requires_non_none_fill`** — `DT + (liquid_metal, none)` raises.
6. **`test_aneutronic_with_blanket_warns`** — `pB11 + (liquid_metal, pbli)` emits UserWarning, no error.
7. **`test_all_valid_form_fill_pairs_accepted_for_dt`** — every entry of `_BLANKET_FORM_VALID_FILLS` (except `NONE`, which DT cannot use) is accepted without error.

### `tests/test_defaults.py` (extend existing file)

8. **`test_every_concept_yaml_declares_blanket`** — parameterized over all 15 concept YAMLs. Asserts both keys present and the fill is in the valid set for its form.

### Not tested (deliberately)

- Full backward-compat snapshot of all 60 (concept × fuel) outputs. Brittle to floating-point drift. The targeted multiplier-ratio test covers the wiring; existing `test_costs.py` / `test_economics.py` cover the rest.

## Implementation order

1. `types.py`: add the two enums, the `_BLANKET_FORM_VALID_FILLS` and `_BLANKET_FORM_DEFAULT_FILL` tables, the two factor lookup dicts.
2. All 15 concept YAMLs: add `blanket_form` and `blanket_fill` keys.
3. `validation.py`: add the two fields and the `check_blanket_compatibility` validator. Make sure `_PLASMA_0D_FIELDS`-style import lists are updated if needed.
4. `costs.py`: change `cas27_special_materials` signature to require `blanket_fill`.
5. `cas22.py`: thread `blanket_form` through `cas22_reactor_plant_equipment` and multiply at the c220101 line.
6. `model.py`: materialize `blanket_form` and `blanket_fill` from `params`; pass to the two downstream calls.
7. Write the new test file, extend the two existing test files.
8. Update `docs/account_justification/CAS27_special_materials.md` with the "Blanket configuration multipliers" section.

The order is chosen so that each step compiles and existing tests pass before the next step begins.
