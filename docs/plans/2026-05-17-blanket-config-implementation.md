# Blanket Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `BlanketForm` and `BlanketFill` enums to 1costingfe, wire structure_factor into CAS22.01 and fill_factor into CAS27, with Pydantic validation and YAML defaults per concept.

**Architecture:** Two orthogonal enums materialised from concept YAML keys (no Python keyword defaults), feeding two cost multipliers on existing per-fuel constants. Validation lives in `CostingInput` (Tier 3 model_validator). The `(LIQUID_METAL, PBLI)` pair has multipliers `(1.0, 1.0)` to preserve calibration; the five concepts that move to `(NONE, NONE)` (orbitron, polywell, pulsed_frc, theta_pinch, dense_plasma_focus) experience a deliberate behaviour change.

**Tech Stack:** Python 3.11+, Pydantic v2, JAX (existing), pytest. Spec: `docs/plans/2026-05-17-blanket-config-design.md`.

**File map:**
- Modify: `src/costingfe/types.py` — add two enums and four lookup tables
- Modify: `src/costingfe/data/defaults/*.yaml` — add `blanket_form` and `blanket_fill` keys to all 15 concept YAMLs
- Modify: `src/costingfe/validation.py` — add two fields and one model_validator
- Modify: `src/costingfe/layers/costs.py:144` — change `cas27_special_materials` signature, apply `fill_factor`
- Modify: `src/costingfe/layers/cas22.py:84-117, 136-145` — change `cas22_reactor_plant_equipment` signature, apply `structure_factor`
- Modify: `src/costingfe/model.py` — materialise from `params`, pass to cost functions
- Modify: `src/costingfe/backcasting_bridge.py:88` — pass `blanket_form` through to cas22 call
- Modify: `tests/test_cas22.py` — add `blanket_form` to the 8 direct callers
- Create: `tests/test_blanket_config.py` — new file, enum + cost-scaling tests
- Modify: `tests/test_validation.py` — add 5 validation tests
- Modify: `tests/test_defaults.py` — add YAML coverage test
- Modify: `docs/account_justification/CAS27_special_materials.md` — append "Blanket configuration multipliers" section

---

## Task 1: Add `BlanketForm` and `BlanketFill` enums to `types.py`

**Files:**
- Create: `tests/test_blanket_config.py`
- Modify: `src/costingfe/types.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_blanket_config.py` with this content:

```python
"""Tests for BlanketForm and BlanketFill enums and their cost wiring."""

import pytest

from costingfe.types import (
    BlanketFill,
    BlanketForm,
    _BLANKET_FORM_DEFAULT_FILL,
    _BLANKET_FORM_VALID_FILLS,
)


def test_blanket_form_structure_factors():
    """Pin structure_factor values; drift here changes costs across the board."""
    assert BlanketForm.LIQUID_METAL.structure_factor == 1.0
    assert BlanketForm.MOLTEN_SALT.structure_factor == 1.3
    assert BlanketForm.SOLID_BREEDER.structure_factor == 1.2
    assert BlanketForm.NONE.structure_factor == 0.0


def test_blanket_fill_factors():
    """Pin fill_factor values; drift here changes CAS27 across the board."""
    assert BlanketFill.PBLI.fill_factor == 1.0
    assert BlanketFill.LI.fill_factor == 2.0
    assert BlanketFill.FLIBE.fill_factor == 5.0
    assert BlanketFill.BE_CERAMIC.fill_factor == 13.0
    assert BlanketFill.CERAMIC_ONLY.fill_factor == 3.0
    assert BlanketFill.NONE.fill_factor == 0.0


def test_blanket_form_valid_fills():
    """Compatibility table: only physical pairs allowed."""
    assert _BLANKET_FORM_VALID_FILLS[BlanketForm.LIQUID_METAL] == {
        BlanketFill.PBLI,
        BlanketFill.LI,
    }
    assert _BLANKET_FORM_VALID_FILLS[BlanketForm.MOLTEN_SALT] == {BlanketFill.FLIBE}
    assert _BLANKET_FORM_VALID_FILLS[BlanketForm.SOLID_BREEDER] == {
        BlanketFill.BE_CERAMIC,
        BlanketFill.CERAMIC_ONLY,
    }
    assert _BLANKET_FORM_VALID_FILLS[BlanketForm.NONE] == {BlanketFill.NONE}


def test_blanket_form_default_fills():
    """Each form has exactly one default fill."""
    assert _BLANKET_FORM_DEFAULT_FILL[BlanketForm.LIQUID_METAL] == BlanketFill.PBLI
    assert _BLANKET_FORM_DEFAULT_FILL[BlanketForm.MOLTEN_SALT] == BlanketFill.FLIBE
    assert _BLANKET_FORM_DEFAULT_FILL[BlanketForm.SOLID_BREEDER] == BlanketFill.BE_CERAMIC
    assert _BLANKET_FORM_DEFAULT_FILL[BlanketForm.NONE] == BlanketFill.NONE
```

- [ ] **Step 2: Run the test, see it fail with ImportError**

```bash
cd /mnt/c/Users/talru/1cfe/1costingfe
uv run pytest tests/test_blanket_config.py -v
```

Expected: `ImportError: cannot import name 'BlanketForm' from 'costingfe.types'`.

- [ ] **Step 3: Add the enums to `types.py`**

In `src/costingfe/types.py`, add after the existing `CoilMaterial` block (after the `_COIL_MATERIAL_COST` dict ends, around line 99):

```python
class BlanketForm(Enum):
    LIQUID_METAL = "liquid_metal"
    MOLTEN_SALT = "molten_salt"
    SOLID_BREEDER = "solid_breeder"
    NONE = "none"

    @property
    def structure_factor(self) -> float:
        """Multiplier on the per-fuel blanket_unit_cost_<fuel> in CAS22.01."""
        return _BLANKET_STRUCTURE_FACTOR[self]


class BlanketFill(Enum):
    PBLI = "pbli"
    LI = "li"
    FLIBE = "flibe"
    BE_CERAMIC = "be_ceramic"
    CERAMIC_ONLY = "ceramic_only"
    NONE = "none"

    @property
    def fill_factor(self) -> float:
        """Multiplier on the per-fuel special_materials_<fuel> in CAS27."""
        return _BLANKET_FILL_FACTOR[self]


_BLANKET_STRUCTURE_FACTOR = {
    BlanketForm.LIQUID_METAL: 1.0,
    BlanketForm.MOLTEN_SALT: 1.3,
    BlanketForm.SOLID_BREEDER: 1.2,
    BlanketForm.NONE: 0.0,
}

_BLANKET_FILL_FACTOR = {
    BlanketFill.PBLI: 1.0,
    BlanketFill.LI: 2.0,
    BlanketFill.FLIBE: 5.0,
    BlanketFill.BE_CERAMIC: 13.0,
    BlanketFill.CERAMIC_ONLY: 3.0,
    BlanketFill.NONE: 0.0,
}

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

- [ ] **Step 4: Run the test, see it pass**

```bash
uv run pytest tests/test_blanket_config.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Run the full test suite to verify no regressions**

```bash
uv run pytest -x
```

Expected: all existing tests still pass (this task only adds; nothing is consumed yet).

- [ ] **Step 6: Commit**

```bash
git -C /mnt/c/Users/talru/1cfe/1costingfe add tests/test_blanket_config.py src/costingfe/types.py
git -C /mnt/c/Users/talru/1cfe/1costingfe commit -m "Add BlanketForm and BlanketFill enums with factor tables"
```

---

## Task 2: Add `blanket_form` and `blanket_fill` keys to all 15 concept YAMLs

**Files:**
- Modify: `src/costingfe/data/defaults/steady_state_tokamak.yaml`
- Modify: `src/costingfe/data/defaults/steady_state_stellarator.yaml`
- Modify: `src/costingfe/data/defaults/steady_state_mirror.yaml`
- Modify: `src/costingfe/data/defaults/steady_state_orbitron.yaml`
- Modify: `src/costingfe/data/defaults/steady_state_polywell.yaml`
- Modify: `src/costingfe/data/defaults/pulsed_laser_ife.yaml`
- Modify: `src/costingfe/data/defaults/pulsed_zpinch.yaml`
- Modify: `src/costingfe/data/defaults/pulsed_heavy_ion.yaml`
- Modify: `src/costingfe/data/defaults/pulsed_mag_target.yaml`
- Modify: `src/costingfe/data/defaults/pulsed_plasma_jet.yaml`
- Modify: `src/costingfe/data/defaults/pulsed_pulsed_frc.yaml`
- Modify: `src/costingfe/data/defaults/pulsed_maglif.yaml`
- Modify: `src/costingfe/data/defaults/pulsed_theta_pinch.yaml`
- Modify: `src/costingfe/data/defaults/pulsed_dense_plasma_focus.yaml`
- Modify: `src/costingfe/data/defaults/pulsed_staged_zpinch.yaml`
- Modify: `tests/test_defaults.py`

- [ ] **Step 1: Write the failing test**

Append this to `tests/test_defaults.py`:

```python
import yaml
from pathlib import Path

import pytest

from costingfe.types import BlanketFill, BlanketForm, _BLANKET_FORM_VALID_FILLS

_DEFAULTS_DIR = (
    Path(__file__).parent.parent / "src" / "costingfe" / "data" / "defaults"
)
_CONCEPT_YAMLS = sorted(
    p for p in _DEFAULTS_DIR.glob("*.yaml") if p.name != "costing_constants.yaml"
)


@pytest.mark.parametrize("yaml_path", _CONCEPT_YAMLS, ids=lambda p: p.stem)
def test_every_concept_yaml_declares_blanket(yaml_path):
    """Every concept YAML must declare blanket_form and blanket_fill, and the
    pair must be valid per _BLANKET_FORM_VALID_FILLS."""
    data = yaml.safe_load(yaml_path.read_text())
    assert "blanket_form" in data, f"{yaml_path.name} missing blanket_form"
    assert "blanket_fill" in data, f"{yaml_path.name} missing blanket_fill"

    form = BlanketForm(data["blanket_form"])
    fill = BlanketFill(data["blanket_fill"])
    assert fill in _BLANKET_FORM_VALID_FILLS[form], (
        f"{yaml_path.name}: blanket_fill={fill.value!r} not valid for "
        f"blanket_form={form.value!r}"
    )
```

- [ ] **Step 2: Run the test, see it fail**

```bash
uv run pytest tests/test_defaults.py::test_every_concept_yaml_declares_blanket -v
```

Expected: 15 failures (one per YAML), `AssertionError: <yaml> missing blanket_form`.

- [ ] **Step 3: Add `blanket_form: liquid_metal` and `blanket_fill: pbli` to the 10 PbLi-default YAMLs**

Append the following two lines (preserve the existing trailing blank line if any) to each of these YAMLs:

```yaml

# Blanket configuration
blanket_form: liquid_metal
blanket_fill: pbli
```

YAMLs to edit (PbLi default):
- `steady_state_tokamak.yaml`
- `steady_state_stellarator.yaml`
- `steady_state_mirror.yaml`
- `pulsed_laser_ife.yaml`
- `pulsed_zpinch.yaml`
- `pulsed_heavy_ion.yaml`
- `pulsed_mag_target.yaml`
- `pulsed_plasma_jet.yaml`
- `pulsed_maglif.yaml`
- `pulsed_staged_zpinch.yaml`

- [ ] **Step 4: Add `blanket_form: none` and `blanket_fill: none` to the 5 none-default YAMLs**

Append the following two lines to each:

```yaml

# Blanket configuration (aneutronic-oriented concept: no breeding blanket)
blanket_form: none
blanket_fill: none
```

YAMLs to edit (none default):
- `steady_state_orbitron.yaml`
- `steady_state_polywell.yaml`
- `pulsed_pulsed_frc.yaml`
- `pulsed_theta_pinch.yaml`
- `pulsed_dense_plasma_focus.yaml`

- [ ] **Step 5: Run the YAML coverage test, see it pass**

```bash
uv run pytest tests/test_defaults.py::test_every_concept_yaml_declares_blanket -v
```

Expected: 15 passed.

- [ ] **Step 6: Run the full test suite to verify no regressions**

```bash
uv run pytest -x
```

Expected: all existing tests still pass. (The YAML keys are silently ignored by `CostingInput` until Task 3 declares them as fields; nothing consumes the multipliers yet.)

- [ ] **Step 7: Commit**

```bash
git -C /mnt/c/Users/talru/1cfe/1costingfe add src/costingfe/data/defaults/ tests/test_defaults.py
git -C /mnt/c/Users/talru/1cfe/1costingfe commit -m "Declare blanket_form and blanket_fill in every concept YAML"
```

---

## Task 3: Add `CostingInput` fields and `check_blanket_compatibility` validator

**Files:**
- Modify: `src/costingfe/validation.py`
- Modify: `tests/test_validation.py`

- [ ] **Step 1: Write the failing tests**

Append this to `tests/test_validation.py`:

```python
from costingfe.types import BlanketFill, BlanketForm


def test_blanket_fill_must_match_form():
    """Schema check: solid_breeder cannot use pbli fill."""
    with pytest.raises(ValidationError, match="not valid for blanket_form"):
        CostingInput(
            concept=ConfinementConcept.TOKAMAK,
            fuel=Fuel.DT,
            net_electric_mw=1000.0,
            blanket_form=BlanketForm.SOLID_BREEDER,
            blanket_fill=BlanketFill.PBLI,
        )


def test_dt_requires_breeding_blanket():
    """Physics check: DT without breeding blanket raises."""
    with pytest.raises(ValidationError, match="DT fuel requires a breeding blanket"):
        CostingInput(
            concept=ConfinementConcept.TOKAMAK,
            fuel=Fuel.DT,
            net_electric_mw=1000.0,
            blanket_form=BlanketForm.NONE,
            blanket_fill=BlanketFill.NONE,
        )


def test_dt_requires_non_none_fill():
    """Physics check: DT with form=liquid_metal but fill=none still raises."""
    with pytest.raises(ValidationError, match="not valid for blanket_form"):
        # fill=NONE is not in valid_fills for LIQUID_METAL, so the schema check
        # fires first. Verify the schema-incompatibility error is what fires.
        CostingInput(
            concept=ConfinementConcept.TOKAMAK,
            fuel=Fuel.DT,
            net_electric_mw=1000.0,
            blanket_form=BlanketForm.LIQUID_METAL,
            blanket_fill=BlanketFill.NONE,
        )


def test_aneutronic_with_blanket_warns():
    """Economics check: p-B11 with non-none blanket emits UserWarning."""
    with pytest.warns(UserWarning, match="aneutronic fuels do not need a breeding blanket"):
        CostingInput(
            concept=ConfinementConcept.TOKAMAK,
            fuel=Fuel.PB11,
            net_electric_mw=1000.0,
            blanket_form=BlanketForm.LIQUID_METAL,
            blanket_fill=BlanketFill.PBLI,
        )


def test_all_valid_form_fill_pairs_accepted_for_dt():
    """Every valid pair (except NONE/NONE which DT rejects) is accepted."""
    from costingfe.types import _BLANKET_FORM_VALID_FILLS

    for form, fills in _BLANKET_FORM_VALID_FILLS.items():
        if form == BlanketForm.NONE:
            continue
        for fill in fills:
            CostingInput(
                concept=ConfinementConcept.TOKAMAK,
                fuel=Fuel.DT,
                net_electric_mw=1000.0,
                blanket_form=form,
                blanket_fill=fill,
            )
```

Note: `ConfinementConcept`, `Fuel`, `CostingInput`, `ValidationError`, and `pytest` are already imported at the top of `test_validation.py` (lines 1-15 of the existing file).

- [ ] **Step 2: Run the new tests, see them fail**

```bash
uv run pytest tests/test_validation.py -v -k "blanket or aneutronic"
```

Expected: failures because `CostingInput` does not yet have `blanket_form` / `blanket_fill` fields (extra kwargs are forbidden or silently dropped depending on Pydantic config; in either case the assertions about the validator's error messages will not fire).

- [ ] **Step 3: Add the two fields to `CostingInput`**

In `src/costingfe/validation.py`, add this import near the top with the other `costingfe.types` imports (around line 13-18):

```python
from costingfe.types import (
    CONCEPT_TO_FAMILY,
    BlanketFill,
    BlanketForm,
    ConfinementConcept,
    ConfinementFamily,
    Fuel,
    _BLANKET_FORM_VALID_FILLS,
)
```

Add the two fields near the end of the `CostingInput` field block, just before the `_COMMON_REQUIRED` class-level list (around line 145):

```python
    # Blanket configuration (YAML-driven, validated below)
    blanket_form: BlanketForm | None = None
    blanket_fill: BlanketFill | None = None
```

- [ ] **Step 4: Add the `check_blanket_compatibility` model_validator**

In `src/costingfe/validation.py`, add this method to `CostingInput` after the existing `check_physics` method (after line 263):

```python
    @model_validator(mode="after")
    def check_blanket_compatibility(self):
        """Tier 3: blanket form/fill compatibility and fuel-physics coupling.

        Errors:
          - blanket_fill must be in _BLANKET_FORM_VALID_FILLS[blanket_form]
          - DT fuel with blanket_form=NONE or blanket_fill=NONE is unphysical
        Warnings:
          - Aneutronic fuels (DHe3, pB11) with non-none blanket are wasteful
        """
        if self.blanket_form is None or self.blanket_fill is None:
            return self  # presence is enforced at the materialization point

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

        if (
            self.fuel in (Fuel.DHE3, Fuel.PB11)
            and self.blanket_form != BlanketForm.NONE
        ):
            warnings.warn(
                f"{self.fuel.value} with blanket_form={self.blanket_form.value!r}: "
                "aneutronic fuels do not need a breeding blanket.",
                stacklevel=2,
            )

        return self
```

- [ ] **Step 5: Run the new validation tests, see them pass**

```bash
uv run pytest tests/test_validation.py -v -k "blanket or aneutronic"
```

Expected: 5 passed.

- [ ] **Step 6: Run the full test suite to verify no regressions**

```bash
uv run pytest -x
```

Expected: all existing tests still pass. The new validator only runs when both fields are non-None, and since model.py does not yet pass them through `params`, the existing forward() flow still skips it.

- [ ] **Step 7: Commit**

```bash
git -C /mnt/c/Users/talru/1cfe/1costingfe add src/costingfe/validation.py tests/test_validation.py
git -C /mnt/c/Users/talru/1cfe/1costingfe commit -m "Validate blanket_form and blanket_fill in CostingInput"
```

---

## Task 4: Wire `blanket_form` and `blanket_fill` through the cost pipeline

This is the atomic wiring change: `cas27_special_materials` and `cas22_reactor_plant_equipment` get new required parameters, `model.py` materialises from `params` and passes them, `backcasting_bridge.py` is updated, and the 8 direct callers in `test_cas22.py` get the new keyword argument.

**Files:**
- Modify: `src/costingfe/layers/costs.py:144-163`
- Modify: `src/costingfe/layers/cas22.py:84-117, 136-145`
- Modify: `src/costingfe/model.py` (two call sites + new materialization)
- Modify: `src/costingfe/backcasting_bridge.py:88-113`
- Modify: `tests/test_cas22.py` (8 direct callers)
- Modify: `tests/test_blanket_config.py` (add the end-to-end scaling test)

- [ ] **Step 1: Write the failing end-to-end test**

Append this to `tests/test_blanket_config.py`:

```python
from costingfe import ConfinementConcept, CostModel, Fuel
from costingfe.types import BlanketFill, BlanketForm


@pytest.mark.parametrize(
    "form, fill, exp_structure_factor, exp_fill_factor",
    [
        ("liquid_metal", "pbli",         1.0,  1.0),
        ("liquid_metal", "li",           1.0,  2.0),
        ("molten_salt",  "flibe",        1.3,  5.0),
        ("solid_breeder","be_ceramic",   1.2, 13.0),
        ("solid_breeder","ceramic_only", 1.2,  3.0),
    ],
)
def test_dt_tokamak_blanket_cost_scaling(
    form, fill, exp_structure_factor, exp_fill_factor
):
    """Picking a non-default blanket scales CAS22.01 and CAS27 by the
    documented multipliers vs the (liquid_metal, pbli) baseline."""
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    kw = dict(
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
    )
    baseline = model.forward(**kw, blanket_form="liquid_metal", blanket_fill="pbli")
    result = model.forward(**kw, blanket_form=form, blanket_fill=fill)

    c220101_base = float(baseline.cas22_detail["C220101"])
    c220101_new = float(result.cas22_detail["C220101"])
    assert c220101_new == pytest.approx(
        c220101_base * exp_structure_factor, rel=1e-6
    ), f"C220101 multiplier wrong for {form}/{fill}"

    cas27_base = float(baseline.costs.cas27)
    cas27_new = float(result.costs.cas27)
    assert cas27_new == pytest.approx(
        cas27_base * exp_fill_factor, rel=1e-6
    ), f"CAS27 multiplier wrong for {form}/{fill}"
```

- [ ] **Step 2: Run the test, see it fail**

```bash
uv run pytest tests/test_blanket_config.py::test_dt_tokamak_blanket_cost_scaling -v
```

Expected: failure for non-pbli cases, because nothing yet multiplies by the factors.

- [ ] **Step 3: Modify `cas27_special_materials` in `src/costingfe/layers/costs.py`**

Find the function at line 144. Change its definition and body to:

```python
def cas27_special_materials(cc, p_net, fuel, blanket_fill):
    """CAS27: Special materials, initial reactor material inventory. Returns M$.

    Covers non-fuel reactor materials: breeding blanket fill (PbLi, Li, FLiBe),
    neutron multiplier (Be if HCPB concept), and other special inventory.
    CAS220101 covers the blanket *structure*; CAS27 covers the *material fill*.

    The per-fuel base sets the baseline (PbLi-style assumption); blanket_fill
    scales by fill_factor to capture alternative chemistries (HCPB Be, FLiBe,
    pure Li, ceramic-only, none).

    See docs/account_justification/CAS27_special_materials.md
    """
    base = {
        Fuel.DT: cc.special_materials_dt,
        Fuel.DD: cc.special_materials_dd,
        Fuel.DHE3: cc.special_materials_dhe3,
        Fuel.PB11: cc.special_materials_pb11,
    }
    return base[fuel] * blanket_fill.fill_factor * (p_net / 1000.0)
```

At the top of `costs.py`, add `BlanketFill` to the existing `from costingfe.types import ...` block.

- [ ] **Step 4: Modify `cas22_reactor_plant_equipment` in `src/costingfe/layers/cas22.py`**

At the top of `cas22.py`, add `BlanketForm` to the existing `from costingfe.types import ...` block.

In the function signature at line 84-117, add `blanket_form: BlanketForm` as a required parameter. Place it adjacent to `coil_material` and `n_coils` for readability. Concretely, change the signature block to include this line after `coil_material: CoilMaterial,` at line 102:

```python
    coil_material: CoilMaterial,
    blanket_form: BlanketForm,
```

In the function body at line 136-145, change the c220101 calculation. The existing code is:

```python
    blanket_unit = {
        Fuel.DT: cc.blanket_unit_cost_dt,
        Fuel.DD: cc.blanket_unit_cost_dd,
        Fuel.DHE3: cc.blanket_unit_cost_dhe3,
        Fuel.PB11: cc.blanket_unit_cost_pb11,
    }
    # TODO: incorporate wall_material cost multiplier into C220101
    # (W tiles vs flowing Li systems vs SiC composites have very different
    # fabrication costs — requires dedicated research)
    c220101 = blanket_unit[fuel] * blanket_vol * (p_th / P_TH_REF) ** 0.6
```

Change the assignment to:

```python
    c220101 = (
        blanket_unit[fuel]
        * blanket_form.structure_factor
        * blanket_vol
        * (p_th / P_TH_REF) ** 0.6
    )
```

(Keep the existing `# TODO` comment immediately above this line.)

- [ ] **Step 5: Update `model.py` to materialize and pass both values**

In `src/costingfe/model.py`, add `BlanketForm` and `BlanketFill` to the existing `from costingfe.types import ...` block near the top.

At line 562 (where `coil_material` is materialized), add immediately after the existing `coil_material = ...` and `n_coils = ...` lines:

```python
        blanket_form = BlanketForm(params["blanket_form"])
        blanket_fill = BlanketFill(params["blanket_fill"])
```

At line 582 (the `cas22_reactor_plant_equipment(...)` call), add `blanket_form=blanket_form,` to the kwargs list (place it adjacent to `coil_material=...` for readability).

At line 675, change the `cas27_special_materials(...)` call from:

```python
        c27 = co.get("CAS27", cas27_special_materials(cc, pt.p_net, self.fuel))
```

to:

```python
        c27 = co.get(
            "CAS27",
            cas27_special_materials(cc, pt.p_net, self.fuel, blanket_fill),
        )
```

- [ ] **Step 6: Update `backcasting_bridge.py`**

At the top of `src/costingfe/backcasting_bridge.py`, add `BlanketForm` to the existing `from costingfe.types import ...` block.

At line 88-113 where `cas22_reactor_plant_equipment(...)` is called, add this line in the kwargs (adjacent to `coil_material=...`):

```python
        blanket_form=BlanketForm(params["blanket_form"]),
```

Note: `params["blanket_form"]` (not `params.get(...)`) — KeyError on missing YAML key is the intended failure mode, matching the project rule that defaults live in YAML, not in code.

- [ ] **Step 7: Update the 9 direct callers in `tests/test_cas22.py`**

In `tests/test_cas22.py`, change the import at line 6 to include `BlanketForm`:

```python
from costingfe.types import (
    BlanketForm,
    CoilMaterial,
    ConfinementConcept,
    ConfinementFamily,
    Fuel,
)
```

Then for each of the 9 `cas22_reactor_plant_equipment(...)` call sites at lines 23, 169, 170, 232, 323, 411, 491, 518, 555 (verify with `grep -n cas22_reactor_plant_equipment tests/test_cas22.py` if the line numbers have shifted since this plan was written), add this argument to the kwargs (place adjacent to `coil_material=CoilMaterial.REBCO_HTS,` for readability):

```python
        blanket_form=BlanketForm.LIQUID_METAL,
```

Each test fixture in `test_cas22.py` currently sets `fuel=Fuel.DT` (or a non-DT fuel via parameter). Using `BlanketForm.LIQUID_METAL` is the correct default for DT tests; for the non-DT parameterized tests (`pb11`, `dhe3`), `LIQUID_METAL` is also fine because those tests assert specific numeric outputs that were calibrated against the implicit-liquid-metal assumption (factor 1.0 preserves them).

- [ ] **Step 8: Run the scaling test, see it pass**

```bash
uv run pytest tests/test_blanket_config.py::test_dt_tokamak_blanket_cost_scaling -v
```

Expected: 5 passed.

- [ ] **Step 9: Run the full test suite**

```bash
uv run pytest -x
```

Expected: all tests pass. Existing test_cas22.py tests should be bit-identical because LIQUID_METAL.structure_factor == 1.0. Existing economic tests should be bit-identical for the 10 PbLi-default concepts × 4 fuels (factors 1.0 × 1.0); the 5 none-default concepts will produce different CAS22.01 and CAS27 values, which is correct per the spec.

If any economic regression test fails for one of the 5 affected concepts (orbitron, polywell, pulsed_frc, theta_pinch, dense_plasma_focus), the failure is expected and the test must be updated to reflect the new physics. For each such failure: read the test, confirm it is hitting one of the 5 concepts, and update its expected value to match the new output (which excludes the previously-implicit PbLi-blanket contribution).

- [ ] **Step 10: Commit**

```bash
git -C /mnt/c/Users/talru/1cfe/1costingfe add \
    src/costingfe/layers/costs.py \
    src/costingfe/layers/cas22.py \
    src/costingfe/model.py \
    src/costingfe/backcasting_bridge.py \
    tests/test_cas22.py \
    tests/test_blanket_config.py
git -C /mnt/c/Users/talru/1cfe/1costingfe commit -m "Wire blanket_form and blanket_fill multipliers into CAS22.01 and CAS27"
```

---

## Task 5: Update `CAS27_special_materials.md` with multiplier table

**Files:**
- Modify: `docs/account_justification/CAS27_special_materials.md`

- [ ] **Step 1: Append the new section to the existing doc**

At the end of `docs/account_justification/CAS27_special_materials.md` (after the existing References section), append:

```markdown

## Blanket configuration multipliers (added 2026-05-17)

CAS27 is now multiplied by `BlanketFill.fill_factor`, and CAS22.01 is multiplied
by `BlanketForm.structure_factor`. Both factors are relative to the existing
DT/PbLi baseline (which keeps factor 1.0 × 1.0 = unchanged).

### `BlanketFill.fill_factor` (CAS27)

| BlanketFill | fill_factor | Source / rationale |
|---|---:|---|
| `pbli` | 1.0 | Baseline. $12M PbLi + $3M Li-6 premium, as documented above. |
| `li` | 2.0 | Self-cooled Li, ~300 t inventory at $300-1000/kg enriched. Center ~$30M. |
| `flibe` | 5.0 | 2LiF-BeF2 melt at $50-150/kg with Li-7 enrichment. ~$75M. See FLiBe section above. |
| `be_ceramic` | 13.0 | HCPB. ~300 t Be at $600/kg + Li-ceramic pebbles. ~$200M. See "HCPB Beryllium Override" section. |
| `ceramic_only` | 3.0 | WCCB. Li-ceramic pebbles without Be multiplier. ~$45M for 300 t at $150/kg synthesis. |
| `none` | 0.0 | Aneutronic, no breeder. |

### `BlanketForm.structure_factor` (CAS22.01)

| BlanketForm | structure_factor | Source / rationale |
|---|---:|---|
| `liquid_metal` | 1.0 | Baseline. RAFM steel flow channels + W FW armor. |
| `molten_salt` | 1.3 | Hastelloy-N corrosion liner on FLiBe-wetted surfaces. Source: INEEL/EXT-99-00331. |
| `solid_breeder` | 1.2 | Pebble-bed canisters: separate breeder/multiplier zones, He coolant manifolds. Source: EUROfusion HCPB cost basis. |
| `none` | 0.0 | No blanket structure. |

### Calibration caveat

These are first-pass values calibrated against limited public data. Spread is
wide for some entries (e.g., `li` 0.8-10x, `flibe` 2-20x) driven by enrichment
choices and supply assumptions. Users should override the multipliers via
`cost_overrides={"CAS27": ...}` for project-specific sensitivity analyses, the
same way they would for any other coefficient with a documented uncertainty
band.
```

- [ ] **Step 2: Commit**

```bash
git -C /mnt/c/Users/talru/1cfe/1costingfe add docs/account_justification/CAS27_special_materials.md
git -C /mnt/c/Users/talru/1cfe/1costingfe commit -m "Document blanket configuration multipliers in CAS27 justification"
```

---

## Final verification

After all tasks complete:

- [ ] Run the full test suite one more time:

```bash
cd /mnt/c/Users/talru/1cfe/1costingfe
uv run pytest -v
```

Expected: every test passes. If any pre-existing tests for orbitron, polywell, pulsed_frc, theta_pinch, or dense_plasma_focus fail with cost mismatches, that is the deliberate model improvement per the design spec, and those tests should have been updated in Task 4 Step 9.

- [ ] Verify the example scripts still run (they exercise the public API end-to-end):

```bash
uv run python examples/dt_tokamak.py
uv run python examples/dt_mirror.py
uv run python examples/dt_tokamak_copper.py
```

Expected: all three complete without error and produce LCOE/cost output. The first two should produce numbers identical to before this change; the third (created in this conversation) should work because it does not override blanket_form/fill.

- [ ] Verify git log shows 5 commits for this feature:

```bash
git -C /mnt/c/Users/talru/1cfe/1costingfe log --oneline -7
```

Expected (most recent first):
1. Document blanket configuration multipliers in CAS27 justification
2. Wire blanket_form and blanket_fill multipliers into CAS22.01 and CAS27
3. Validate blanket_form and blanket_fill in CostingInput
4. Declare blanket_form and blanket_fill in every concept YAML
5. Add BlanketForm and BlanketFill enums with factor tables
6. Design: blanket configuration two-axis form/fill schema
