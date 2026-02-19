# Verification Layer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a pydantic-based CostingInput validation layer that validates all inputs before costing calculations, integrated at both CostModel.forward() and run_costing() entry points.

**Architecture:** Pydantic BaseModel (CostingInput) with Field() constraints for Tier 1, @model_validator for Tier 2 (family-required params) and Tier 3 (physics checks). Constructed internally by forward() and run_costing() after merging YAML defaults with user overrides. Errors raise ValidationError; warnings emit via warnings.warn().

**Tech Stack:** Python 3.10+, pydantic, JAX (existing), pytest

---

### Task 1: Add pydantic dependency

**Files:**
- Modify: `pyproject.toml:6-11`

**Step 1: Add pydantic to dependencies**

In `pyproject.toml`, add `"pydantic>=2.0"` to the dependencies list:

```toml
dependencies = [
    "jax>=0.4.0",
    "jaxlib>=0.4.0",
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "scipy>=1.10.0",
]
```

**Step 2: Install**

Run: `cd /mnt/c/Users/talru/1cfe/1costingfe && uv sync`
Expected: pydantic installed successfully

**Step 3: Verify existing tests still pass**

Run: `cd /mnt/c/Users/talru/1cfe/1costingfe && uv run pytest tests/ -q`
Expected: All 91 tests pass

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Add pydantic dependency for input validation"
```

---

### Task 2: Create CostingInput with Tier 1 field constraints

**Files:**
- Create: `src/costingfe/validation.py`
- Create: `tests/test_validation.py`

**Step 1: Write failing tests for Tier 1 field-level validation**

```python
"""Tests for CostingInput validation."""

import pytest
import warnings
from pydantic import ValidationError

from costingfe.validation import CostingInput
from costingfe.types import ConfinementConcept, Fuel


class TestTier1FieldConstraints:
    """Tier 1: pydantic Field() constraints."""

    def test_valid_minimal_input(self):
        """Required fields only — should succeed."""
        inp = CostingInput(
            concept=ConfinementConcept.TOKAMAK,
            fuel=Fuel.DT,
            net_electric_mw=1000.0,
        )
        assert inp.net_electric_mw == 1000.0
        assert inp.availability == 0.85  # default
        assert inp.lifetime_yr == 40.0  # default

    def test_net_electric_mw_must_be_positive(self):
        with pytest.raises(ValidationError, match="net_electric_mw"):
            CostingInput(
                concept=ConfinementConcept.TOKAMAK,
                fuel=Fuel.DT,
                net_electric_mw=-100.0,
            )

    def test_net_electric_mw_zero_rejected(self):
        with pytest.raises(ValidationError, match="net_electric_mw"):
            CostingInput(
                concept=ConfinementConcept.TOKAMAK,
                fuel=Fuel.DT,
                net_electric_mw=0.0,
            )

    def test_availability_must_be_in_range(self):
        with pytest.raises(ValidationError, match="availability"):
            CostingInput(
                concept=ConfinementConcept.TOKAMAK,
                fuel=Fuel.DT,
                net_electric_mw=1000.0,
                availability=1.5,
            )

    def test_availability_zero_rejected(self):
        with pytest.raises(ValidationError, match="availability"):
            CostingInput(
                concept=ConfinementConcept.TOKAMAK,
                fuel=Fuel.DT,
                net_electric_mw=1000.0,
                availability=0.0,
            )

    def test_lifetime_must_be_positive(self):
        with pytest.raises(ValidationError, match="lifetime_yr"):
            CostingInput(
                concept=ConfinementConcept.TOKAMAK,
                fuel=Fuel.DT,
                net_electric_mw=1000.0,
                lifetime_yr=-5.0,
            )

    def test_n_mod_must_be_integer(self):
        with pytest.raises(ValidationError, match="n_mod"):
            CostingInput(
                concept=ConfinementConcept.TOKAMAK,
                fuel=Fuel.DT,
                net_electric_mw=1000.0,
                n_mod=1.5,
            )

    def test_n_mod_must_be_at_least_one(self):
        with pytest.raises(ValidationError, match="n_mod"):
            CostingInput(
                concept=ConfinementConcept.TOKAMAK,
                fuel=Fuel.DT,
                net_electric_mw=1000.0,
                n_mod=0,
            )

    def test_interest_rate_must_be_positive(self):
        with pytest.raises(ValidationError, match="interest_rate"):
            CostingInput(
                concept=ConfinementConcept.TOKAMAK,
                fuel=Fuel.DT,
                net_electric_mw=1000.0,
                interest_rate=-0.01,
            )

    def test_inflation_rate_can_be_negative(self):
        """Deflation is valid."""
        inp = CostingInput(
            concept=ConfinementConcept.TOKAMAK,
            fuel=Fuel.DT,
            net_electric_mw=1000.0,
            inflation_rate=-0.01,
        )
        assert inp.inflation_rate == -0.01

    def test_construction_time_must_be_positive(self):
        with pytest.raises(ValidationError, match="construction_time_yr"):
            CostingInput(
                concept=ConfinementConcept.TOKAMAK,
                fuel=Fuel.DT,
                net_electric_mw=1000.0,
                construction_time_yr=0.0,
            )

    def test_concept_string_accepted(self):
        """Concept can be passed as string (adapter path)."""
        inp = CostingInput(
            concept="tokamak",
            fuel="dt",
            net_electric_mw=1000.0,
        )
        assert inp.concept == ConfinementConcept.TOKAMAK

    def test_invalid_concept_rejected(self):
        with pytest.raises(ValidationError, match="concept"):
            CostingInput(
                concept="not_a_concept",
                fuel=Fuel.DT,
                net_electric_mw=1000.0,
            )

    def test_all_customer_defaults(self):
        """All customer params have sensible defaults."""
        inp = CostingInput(
            concept=ConfinementConcept.TOKAMAK,
            fuel=Fuel.DT,
            net_electric_mw=1000.0,
        )
        assert inp.availability == 0.85
        assert inp.lifetime_yr == 40.0
        assert inp.n_mod == 1
        assert inp.construction_time_yr == 6.0
        assert inp.interest_rate == 0.07
        assert inp.inflation_rate == 0.02
        assert inp.noak is True
        assert inp.cost_overrides == {}
        assert inp.costing_overrides == {}
```

**Step 2: Run tests to verify they fail**

Run: `cd /mnt/c/Users/talru/1cfe/1costingfe && uv run pytest tests/test_validation.py -v`
Expected: ImportError — validation module doesn't exist yet

**Step 3: Implement CostingInput with Tier 1 constraints**

Create `src/costingfe/validation.py`:

```python
"""Input validation for the costing model.

Pydantic-based CostingInput with three validation tiers:
- Tier 1: Field-level constraints (pydantic Field)
- Tier 2: Family-aware required engineering parameters
- Tier 3: Cross-field physics checks
"""

import warnings
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from costingfe.types import (
    CONCEPT_TO_FAMILY,
    ConfinementConcept,
    ConfinementFamily,
    Fuel,
)


class CostingInput(BaseModel):
    """Validated input for the costing model.

    Required fields: concept, fuel, net_electric_mw.
    Customer parameters have defaults.
    Engineering parameters default to None (filled from YAML templates).
    """

    # --- Required (no defaults) ---
    concept: ConfinementConcept
    fuel: Fuel
    net_electric_mw: float = Field(gt=0)

    # --- Customer parameters (with defaults) ---
    availability: float = Field(default=0.85, gt=0, le=1)
    lifetime_yr: float = Field(default=40.0, gt=0)
    n_mod: int = Field(default=1, ge=1, strict=True)
    construction_time_yr: float = Field(default=6.0, gt=0)
    interest_rate: float = Field(default=0.07, gt=0)
    inflation_rate: float = 0.0245
    noak: bool = True
    cost_overrides: dict[str, float] = Field(default_factory=dict)
    costing_overrides: dict[str, float] = Field(default_factory=dict)

    # --- Engineering parameters (None = use YAML template) ---
    # Common (all families)
    mn: Optional[float] = None
    eta_th: Optional[float] = None
    eta_p: Optional[float] = None
    f_sub: Optional[float] = None
    p_pump: Optional[float] = None
    p_trit: Optional[float] = None
    p_house: Optional[float] = None
    p_cryo: Optional[float] = None
    blanket_t: Optional[float] = None
    ht_shield_t: Optional[float] = None
    structure_t: Optional[float] = None
    vessel_t: Optional[float] = None
    plasma_t: Optional[float] = None

    # MFE only
    p_input: Optional[float] = None
    eta_pin: Optional[float] = None
    eta_de: Optional[float] = None
    f_dec: Optional[float] = None
    p_coils: Optional[float] = None
    p_cool: Optional[float] = None
    axis_t: Optional[float] = None
    elon: Optional[float] = None

    # IFE only
    p_implosion: Optional[float] = None
    p_ignition: Optional[float] = None
    eta_pin1: Optional[float] = None
    eta_pin2: Optional[float] = None
    p_target: Optional[float] = None  # shared with MIF

    # MIF only
    p_driver: Optional[float] = None
    # eta_pin: already declared above (shared MFE/MIF)
    # p_target: already declared above (shared IFE/MIF)
    # p_coils: already declared above (shared MFE/MIF)

    # Plasma parameters (MFE radiation calculation)
    n_e: Optional[float] = None
    T_e: Optional[float] = None
    Z_eff: Optional[float] = None
    plasma_volume: Optional[float] = None
    B: Optional[float] = None
```

**Step 4: Run tests to verify they pass**

Run: `cd /mnt/c/Users/talru/1cfe/1costingfe && uv run pytest tests/test_validation.py -v`
Expected: All Tier 1 tests pass

**Step 5: Commit**

```bash
git add src/costingfe/validation.py tests/test_validation.py
git commit -m "Add CostingInput with Tier 1 field-level validation"
```

---

### Task 3: Add Tier 2 — family-aware required engineering parameters

**Files:**
- Modify: `src/costingfe/validation.py`
- Modify: `tests/test_validation.py`

**Step 1: Write failing tests for Tier 2**

Append to `tests/test_validation.py`:

```python
class TestTier2FamilyRequiredParams:
    """Tier 2: After template merge, all family-required params must be present."""

    def test_mfe_missing_p_input_rejected(self):
        """MFE requires p_input — should fail if None after merge."""
        with pytest.raises(ValidationError, match="p_input"):
            CostingInput(
                concept=ConfinementConcept.TOKAMAK,
                fuel=Fuel.DT,
                net_electric_mw=1000.0,
                # Provide all common params but omit p_input
                mn=1.1, eta_th=0.46, eta_p=0.5, f_sub=0.03,
                p_pump=1.0, p_trit=10.0, p_house=4.0, p_cryo=0.5,
                blanket_t=0.7, ht_shield_t=0.2, structure_t=0.15,
                vessel_t=0.1, plasma_t=2.0,
                # MFE params — p_input intentionally omitted
                eta_pin=0.5, eta_de=0.85, f_dec=0.0,
                p_coils=2.0, p_cool=13.7, axis_t=6.2, elon=1.7,
            )

    def test_ife_missing_p_implosion_rejected(self):
        with pytest.raises(ValidationError, match="p_implosion"):
            CostingInput(
                concept=ConfinementConcept.LASER_IFE,
                fuel=Fuel.DT,
                net_electric_mw=1000.0,
                mn=1.1, eta_th=0.46, eta_p=0.5, f_sub=0.03,
                p_pump=1.0, p_trit=10.0, p_house=4.0, p_cryo=0.5,
                blanket_t=0.8, ht_shield_t=0.25, structure_t=0.15,
                vessel_t=0.1, plasma_t=4.0,
                # IFE params — p_implosion intentionally omitted
                p_ignition=0.1, eta_pin1=0.1, eta_pin2=0.1, p_target=1.0,
            )

    def test_mif_missing_p_driver_rejected(self):
        with pytest.raises(ValidationError, match="p_driver"):
            CostingInput(
                concept=ConfinementConcept.MAG_TARGET,
                fuel=Fuel.DT,
                net_electric_mw=1000.0,
                mn=1.1, eta_th=0.4, eta_p=0.5, f_sub=0.03,
                p_pump=1.0, p_trit=10.0, p_house=4.0, p_cryo=0.2,
                blanket_t=0.7, ht_shield_t=0.2, structure_t=0.15,
                vessel_t=0.1, plasma_t=3.0,
                # MIF params — p_driver intentionally omitted
                eta_pin=0.3, p_target=2.0, p_coils=0.5,
            )

    def test_none_engineering_params_ok_when_template_will_fill(self):
        """When no engineering params given (all None), Tier 2 is skipped.
        Template merge happens in forward(), not in CostingInput directly.
        Tier 2 only fires when at least one engineering param is set."""
        inp = CostingInput(
            concept=ConfinementConcept.TOKAMAK,
            fuel=Fuel.DT,
            net_electric_mw=1000.0,
        )
        assert inp.mn is None  # Will be filled by template later

    def test_mfe_complete_params_accepted(self):
        """All MFE params provided — should pass."""
        inp = CostingInput(
            concept=ConfinementConcept.TOKAMAK,
            fuel=Fuel.DT,
            net_electric_mw=1000.0,
            mn=1.1, eta_th=0.46, eta_p=0.5, f_sub=0.03,
            p_pump=1.0, p_trit=10.0, p_house=4.0, p_cryo=0.5,
            blanket_t=0.7, ht_shield_t=0.2, structure_t=0.15,
            vessel_t=0.1, plasma_t=2.0,
            p_input=50.0, eta_pin=0.5, eta_de=0.85, f_dec=0.0,
            p_coils=2.0, p_cool=13.7, axis_t=6.2, elon=1.7,
        )
        assert inp.p_input == 50.0
```

**Step 2: Run tests to verify they fail**

Run: `cd /mnt/c/Users/talru/1cfe/1costingfe && uv run pytest tests/test_validation.py::TestTier2FamilyRequiredParams -v`
Expected: FAIL — no validator yet, missing params accepted

**Step 3: Implement Tier 2 validator**

Add to `src/costingfe/validation.py`, inside the `CostingInput` class, after the field declarations:

```python
    # --- Family-required engineering parameters ---
    _COMMON_REQUIRED = [
        "mn", "eta_th", "eta_p", "f_sub",
        "p_pump", "p_trit", "p_house", "p_cryo",
        "blanket_t", "ht_shield_t", "structure_t", "vessel_t", "plasma_t",
    ]
    _MFE_REQUIRED = [
        "p_input", "eta_pin", "eta_de", "f_dec",
        "p_coils", "p_cool", "axis_t", "elon",
    ]
    _IFE_REQUIRED = [
        "p_implosion", "p_ignition", "eta_pin1", "eta_pin2", "p_target",
    ]
    _MIF_REQUIRED = [
        "p_driver", "eta_pin", "p_target", "p_coils",
    ]

    @model_validator(mode="after")
    def check_family_required_params(self):
        """Tier 2: If any engineering param is set, all family-required params must be present."""
        family = CONCEPT_TO_FAMILY[self.concept]

        # Check if user provided any engineering params explicitly
        all_eng = (
            self._COMMON_REQUIRED
            + self._MFE_REQUIRED + self._IFE_REQUIRED + self._MIF_REQUIRED
        )
        any_set = any(getattr(self, k) is not None for k in all_eng)
        if not any_set:
            return self  # All None — template will fill later

        # Determine required params for this family
        family_required = {
            ConfinementFamily.MFE: self._MFE_REQUIRED,
            ConfinementFamily.IFE: self._IFE_REQUIRED,
            ConfinementFamily.MIF: self._MIF_REQUIRED,
        }
        required = self._COMMON_REQUIRED + family_required.get(family, [])

        missing = [k for k in required if getattr(self, k) is None]
        if missing:
            raise ValueError(
                f"Missing required engineering parameters for "
                f"{family.value}: {', '.join(missing)}"
            )
        return self
```

**Step 4: Run tests to verify they pass**

Run: `cd /mnt/c/Users/talru/1cfe/1costingfe && uv run pytest tests/test_validation.py -v`
Expected: All Tier 1 and Tier 2 tests pass

**Step 5: Commit**

```bash
git add src/costingfe/validation.py tests/test_validation.py
git commit -m "Add Tier 2 family-aware required param validation"
```

---

### Task 4: Add Tier 3 — cross-field and physics checks

**Files:**
- Modify: `src/costingfe/validation.py`
- Modify: `tests/test_validation.py`

**Step 1: Write failing tests for Tier 3**

Append to `tests/test_validation.py`:

```python
class TestTier3PhysicsChecks:
    """Tier 3: Cross-field and physics validation."""

    def _make_mfe_input(self, **overrides):
        """Helper: complete MFE tokamak input with all params."""
        defaults = dict(
            concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT,
            net_electric_mw=1000.0,
            mn=1.1, eta_th=0.46, eta_p=0.5, f_sub=0.03,
            p_pump=1.0, p_trit=10.0, p_house=4.0, p_cryo=0.5,
            blanket_t=0.7, ht_shield_t=0.2, structure_t=0.15,
            vessel_t=0.1, plasma_t=2.0,
            p_input=50.0, eta_pin=0.5, eta_de=0.85, f_dec=0.0,
            p_coils=2.0, p_cool=13.7, axis_t=6.2, elon=1.7,
        )
        defaults.update(overrides)
        return CostingInput(**defaults)

    def test_eta_th_warning_when_high(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._make_mfe_input(eta_th=0.70)
            assert any("eta_th" in str(warning.message) for warning in w)

    def test_eta_th_no_warning_when_normal(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._make_mfe_input(eta_th=0.46)
            assert not any("eta_th" in str(warning.message) for warning in w)

    def test_eta_p_warning_when_high(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._make_mfe_input(eta_p=0.98)
            assert any("eta_p" in str(warning.message) for warning in w)

    def test_mn_warning_when_outside_range(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._make_mfe_input(mn=2.0)
            assert any("mn" in str(warning.message) for warning in w)

    def test_f_sub_warning_when_high(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._make_mfe_input(f_sub=0.35)
            assert any("f_sub" in str(warning.message) for warning in w)

    def test_p_net_negative_raises_error(self):
        """p_net < 0 is a hard error — plant consumes more than it produces."""
        with pytest.raises(ValidationError, match="p_net"):
            self._make_mfe_input(
                net_electric_mw=1.0,
                p_input=500.0,
                eta_pin=0.1,
            )

    def test_q_sci_warning_when_below_one(self):
        """Q_sci < 1 means fusion power < injected heating."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._make_mfe_input(p_input=5000.0, eta_pin=0.9)
            assert any("Q_sci" in str(warning.message) for warning in w)

    def test_rec_frac_warning_when_high(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._make_mfe_input(eta_pin=0.05)
            assert any("rec" in str(warning.message).lower() for warning in w)
```

**Step 2: Run tests to verify they fail**

Run: `cd /mnt/c/Users/talru/1cfe/1costingfe && uv run pytest tests/test_validation.py::TestTier3PhysicsChecks -v`
Expected: FAIL — no physics validator yet

**Step 3: Implement Tier 3 validator**

Add to `src/costingfe/validation.py`, inside the `CostingInput` class, after the Tier 2 validator:

```python
    @model_validator(mode="after")
    def check_physics(self):
        """Tier 3: Cross-field physics checks (warnings + errors).

        Only runs when all engineering params are present (not None).
        When params are None, template merge hasn't happened yet.
        """
        family = CONCEPT_TO_FAMILY[self.concept]

        # --- Simple field warnings (no computation needed) ---
        if self.eta_th is not None and self.eta_th > 0.65:
            warnings.warn(
                f"eta_th = {self.eta_th} is unusually high (> 0.65)",
                stacklevel=2,
            )
        if self.eta_p is not None and self.eta_p > 0.95:
            warnings.warn(
                f"eta_p = {self.eta_p} is unusually high (> 0.95)",
                stacklevel=2,
            )
        if self.mn is not None and not (1.0 <= self.mn <= 1.5):
            warnings.warn(
                f"mn = {self.mn} is outside typical range [1.0, 1.5]",
                stacklevel=2,
            )
        if self.f_sub is not None and self.f_sub > 0.3:
            warnings.warn(
                f"f_sub = {self.f_sub} is unusually high (> 0.3)",
                stacklevel=2,
            )

        # --- Physics checks requiring power balance computation ---
        # Only possible when all required engineering params are present
        if any(getattr(self, k) is None for k in self._COMMON_REQUIRED):
            return self

        if family == ConfinementFamily.MFE:
            self._check_mfe_physics()
        elif family == ConfinementFamily.IFE:
            self._check_ife_physics()
        elif family == ConfinementFamily.MIF:
            self._check_mif_physics()

        return self

    def _check_mfe_physics(self):
        from costingfe.layers.physics import (
            mfe_inverse_power_balance,
            mfe_forward_power_balance,
        )

        mfe_params = [self.p_input, self.eta_pin, self.eta_de, self.f_dec,
                      self.p_coils, self.p_cool]
        if any(v is None for v in mfe_params):
            return

        p_net_per_mod = self.net_electric_mw / self.n_mod
        p_fus = mfe_inverse_power_balance(
            p_net_target=p_net_per_mod, fuel=self.fuel,
            p_input=self.p_input, mn=self.mn, eta_th=self.eta_th,
            eta_p=self.eta_p, eta_pin=self.eta_pin, eta_de=self.eta_de,
            f_sub=self.f_sub, f_dec=self.f_dec, p_coils=self.p_coils,
            p_cool=self.p_cool, p_pump=self.p_pump, p_trit=self.p_trit,
            p_house=self.p_house, p_cryo=self.p_cryo,
        )
        pt = mfe_forward_power_balance(
            p_fus=p_fus, fuel=self.fuel,
            p_input=self.p_input, mn=self.mn, eta_th=self.eta_th,
            eta_p=self.eta_p, eta_pin=self.eta_pin, eta_de=self.eta_de,
            f_sub=self.f_sub, f_dec=self.f_dec, p_coils=self.p_coils,
            p_cool=self.p_cool, p_pump=self.p_pump, p_trit=self.p_trit,
            p_house=self.p_house, p_cryo=self.p_cryo,
        )
        self._check_power_table(pt)

    def _check_ife_physics(self):
        from costingfe.layers.physics import (
            ife_inverse_power_balance,
            ife_forward_power_balance,
        )

        ife_params = [self.p_implosion, self.p_ignition,
                      self.eta_pin1, self.eta_pin2, self.p_target]
        if any(v is None for v in ife_params):
            return

        p_net_per_mod = self.net_electric_mw / self.n_mod
        p_fus = ife_inverse_power_balance(
            p_net_target=p_net_per_mod, fuel=self.fuel,
            p_implosion=self.p_implosion, p_ignition=self.p_ignition,
            mn=self.mn, eta_th=self.eta_th, eta_p=self.eta_p,
            eta_pin1=self.eta_pin1, eta_pin2=self.eta_pin2,
            f_sub=self.f_sub, p_pump=self.p_pump, p_trit=self.p_trit,
            p_house=self.p_house, p_cryo=self.p_cryo, p_target=self.p_target,
        )
        pt = ife_forward_power_balance(
            p_fus=p_fus, fuel=self.fuel,
            p_implosion=self.p_implosion, p_ignition=self.p_ignition,
            mn=self.mn, eta_th=self.eta_th, eta_p=self.eta_p,
            eta_pin1=self.eta_pin1, eta_pin2=self.eta_pin2,
            f_sub=self.f_sub, p_pump=self.p_pump, p_trit=self.p_trit,
            p_house=self.p_house, p_cryo=self.p_cryo, p_target=self.p_target,
        )
        self._check_power_table(pt)

    def _check_mif_physics(self):
        from costingfe.layers.physics import (
            mif_inverse_power_balance,
            mif_forward_power_balance,
        )

        mif_params = [self.p_driver, self.eta_pin, self.p_target]
        if any(v is None for v in mif_params):
            return

        p_net_per_mod = self.net_electric_mw / self.n_mod
        p_fus = mif_inverse_power_balance(
            p_net_target=p_net_per_mod, fuel=self.fuel,
            p_driver=self.p_driver, mn=self.mn, eta_th=self.eta_th,
            eta_p=self.eta_p, eta_pin=self.eta_pin, f_sub=self.f_sub,
            p_pump=self.p_pump, p_trit=self.p_trit, p_house=self.p_house,
            p_cryo=self.p_cryo, p_target=self.p_target,
            p_coils=self.p_coils or 0.0,
        )
        pt = mif_forward_power_balance(
            p_fus=p_fus, fuel=self.fuel,
            p_driver=self.p_driver, mn=self.mn, eta_th=self.eta_th,
            eta_p=self.eta_p, eta_pin=self.eta_pin, f_sub=self.f_sub,
            p_pump=self.p_pump, p_trit=self.p_trit, p_house=self.p_house,
            p_cryo=self.p_cryo, p_target=self.p_target,
            p_coils=self.p_coils or 0.0,
        )
        self._check_power_table(pt)

    def _check_power_table(self, pt):
        """Check derived physics values from power balance."""
        if float(pt.p_net) < 0:
            raise ValueError(
                f"p_net = {float(pt.p_net):.1f} MW is negative — "
                f"plant consumes more power than it produces"
            )
        if float(pt.q_sci) < 1:
            warnings.warn(
                f"Q_sci = {float(pt.q_sci):.3f} < 1 — "
                f"fusion power is less than injected heating",
                stacklevel=4,
            )
        if float(pt.rec_frac) > 0.5:
            warnings.warn(
                f"Recirculating fraction = {float(pt.rec_frac):.3f} > 0.5 — "
                f"excessive parasitic power load",
                stacklevel=4,
            )
```

**Step 4: Run tests to verify they pass**

Run: `cd /mnt/c/Users/talru/1cfe/1costingfe && uv run pytest tests/test_validation.py -v`
Expected: All Tier 1, 2, and 3 tests pass

**Step 5: Commit**

```bash
git add src/costingfe/validation.py tests/test_validation.py
git commit -m "Add Tier 3 cross-field and physics validation"
```

---

### Task 5: Integrate into CostModel.forward()

**Files:**
- Modify: `src/costingfe/model.py:181-211`
- Modify: `tests/test_validation.py`

**Step 1: Write failing integration test**

Append to `tests/test_validation.py`:

```python
from costingfe.model import CostModel


class TestForwardIntegration:
    """Validation fires when calling CostModel.forward()."""

    def test_forward_rejects_negative_net_electric(self):
        model = CostModel(
            concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT,
        )
        with pytest.raises(ValidationError, match="net_electric_mw"):
            model.forward(net_electric_mw=-100, availability=0.85, lifetime_yr=40)

    def test_forward_rejects_invalid_availability(self):
        model = CostModel(
            concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT,
        )
        with pytest.raises(ValidationError, match="availability"):
            model.forward(net_electric_mw=1000, availability=2.0, lifetime_yr=40)

    def test_forward_still_works_with_valid_input(self):
        model = CostModel(
            concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT,
        )
        result = model.forward(
            net_electric_mw=1000, availability=0.85, lifetime_yr=40,
        )
        assert result.costs.lcoe > 0
```

**Step 2: Run tests to verify they fail**

Run: `cd /mnt/c/Users/talru/1cfe/1costingfe && uv run pytest tests/test_validation.py::TestForwardIntegration -v`
Expected: FAIL — forward() doesn't validate yet

**Step 3: Integrate CostingInput into forward()**

In `src/costingfe/model.py`, add import at top:

```python
from costingfe.validation import CostingInput
```

Insert validation in `forward()` after the `params.update(...)` block (after line 211) and before `# Layer 2: Power balance` (line 213). Add:

```python
        # Validate merged parameters
        CostingInput(
            concept=self.concept,
            fuel=self.fuel,
            net_electric_mw=net_electric_mw,
            availability=availability,
            lifetime_yr=lifetime_yr,
            n_mod=n_mod,
            construction_time_yr=construction_time_yr,
            interest_rate=interest_rate,
            inflation_rate=inflation_rate,
            noak=noak,
            cost_overrides=cost_overrides or {},
            **{k: v for k, v in params.items()
               if k in CostingInput.model_fields
               and k not in {
                   "concept", "fuel", "net_electric_mw", "availability",
                   "lifetime_yr", "n_mod", "construction_time_yr",
                   "interest_rate", "inflation_rate", "noak",
                   "cost_overrides",
               }},
        )
```

**Step 4: Run ALL tests to verify nothing broke**

Run: `cd /mnt/c/Users/talru/1cfe/1costingfe && uv run pytest tests/ -v`
Expected: All tests pass (91 existing + new validation tests)

**Step 5: Commit**

```bash
git add src/costingfe/model.py tests/test_validation.py
git commit -m "Integrate CostingInput validation into CostModel.forward()"
```

---

### Task 6: Integrate into run_costing() adapter

**Files:**
- Modify: `src/costingfe/adapter.py:48-73`
- Modify: `tests/test_validation.py`

**Step 1: Write failing integration test**

Append to `tests/test_validation.py`:

```python
from costingfe.adapter import FusionTeaInput, run_costing


class TestAdapterIntegration:
    """Validation fires when calling run_costing()."""

    def test_adapter_rejects_negative_net_electric(self):
        inp = FusionTeaInput(
            concept="tokamak", fuel="dt",
            net_electric_mw=-100, availability=0.85, lifetime_yr=40,
        )
        with pytest.raises(ValidationError, match="net_electric_mw"):
            run_costing(inp)

    def test_adapter_rejects_invalid_availability(self):
        inp = FusionTeaInput(
            concept="tokamak", fuel="dt",
            net_electric_mw=1000, availability=2.0, lifetime_yr=40,
        )
        with pytest.raises(ValidationError, match="availability"):
            run_costing(inp)

    def test_adapter_still_works_with_valid_input(self):
        inp = FusionTeaInput(
            concept="tokamak", fuel="dt",
            net_electric_mw=1000, availability=0.85, lifetime_yr=40,
        )
        output = run_costing(inp)
        assert output.lcoe > 0
```

**Step 2: Run tests to verify they fail**

Run: `cd /mnt/c/Users/talru/1cfe/1costingfe && uv run pytest tests/test_validation.py::TestAdapterIntegration -v`
Expected: FAIL — run_costing() doesn't validate yet

**Step 3: Integrate CostingInput into run_costing()**

In `src/costingfe/adapter.py`, add import:

```python
from costingfe.validation import CostingInput
```

Insert validation after line 55 (`fuel = Fuel(inp.fuel)`) and before `cc = load_costing_constants()`:

```python
    # Validate inputs before costing
    CostingInput(
        concept=concept,
        fuel=fuel,
        net_electric_mw=inp.net_electric_mw,
        availability=inp.availability,
        lifetime_yr=inp.lifetime_yr,
        n_mod=inp.n_mod,
        construction_time_yr=inp.construction_time_yr,
        interest_rate=inp.interest_rate,
        inflation_rate=inp.inflation_rate,
        noak=inp.noak,
        cost_overrides=inp.cost_overrides or {},
        costing_overrides=inp.costing_overrides or {},
        **{k: v for k, v in inp.overrides.items()
           if k in CostingInput.model_fields},
    )
```

**Step 4: Run ALL tests**

Run: `cd /mnt/c/Users/talru/1cfe/1costingfe && uv run pytest tests/ -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/costingfe/adapter.py tests/test_validation.py
git commit -m "Integrate CostingInput validation into run_costing() adapter"
```

---

### Task 7: Export CostingInput from package and final verification

**Files:**
- Modify: `src/costingfe/__init__.py:1-9`

**Step 1: Add CostingInput to public API**

In `src/costingfe/__init__.py`, add to the imports:

```python
from costingfe.validation import CostingInput
```

**Step 2: Run full test suite**

Run: `cd /mnt/c/Users/talru/1cfe/1costingfe && uv run pytest tests/ -v --tb=short`
Expected: All tests pass (91 existing + ~20 new validation tests)

**Step 3: Run ruff to check formatting**

Run: `cd /mnt/c/Users/talru/1cfe/1costingfe && uv run ruff check src/ tests/`
Expected: No lint errors

**Step 4: Commit**

```bash
git add src/costingfe/__init__.py
git commit -m "Export CostingInput from package public API"
```
