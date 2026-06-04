# Laser-IFE Driver Capital + Scheduled Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make laser-IFE driver architecture (DPSSL / KrF / Nd:Glass) first-class — branch both C220104 capital and CAS72 scheduled replacement by type — and unify replacement costing through one shared closed-form helper.

**Architecture:** A single `levelized_replacement_cost` helper (geometric closed-form of the existing discrete PV loop, no iteration cap) replaces the three inline `MAX_REP` loops (core, DEC grid, cap bank) and powers a new per-subsystem laser-driver replacement block. A `LaserDriverType` enum selects, for `LASER_IFE`, the C220104 $/MJ coefficient and the set of `(replace_frac, shot_lifetime)` subsystem pairs. Driver type is a parameter (concept-YAML default + override), not a separate concept.

**Tech Stack:** Python, JAX (`jax.numpy`), pytest, YAML config. Source under `src/costingfe/`, tests under `tests/`.

**Spec:** `docs/superpowers/specs/2026-06-04-laser-ife-driver-replacement-design.md`

**Branch:** create `feat/laser-ife-driver-replacement-v2` off `master` before Task 1.

---

### Task 1: Shared `levelized_replacement_cost` helper

**Files:**
- Modify: `src/costingfe/layers/economics.py` (add helper after `levelized_annual_cost`, ~line 51)
- Test: `tests/test_economics.py` (create if absent; else append)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_economics.py` (create file with these imports if it does not exist):

```python
import math

import pytest

from costingfe.layers.economics import levelized_replacement_cost


def _manual_repl(event_cost, t_replace, i, n):
    s = (1.0 + i) ** (-t_replace)
    n_rep = max(0, math.ceil(n / t_replace) - 1)
    pv = sum(event_cost * s ** k for k in range(1, n_rep + 1))
    crf = (i * (1 + i) ** n) / ((1 + i) ** n - 1)
    return pv * crf


def test_repl_matches_geometric_sum_multi_year():
    # interval 5 yr over 30 yr plant -> 5 replacements
    got = levelized_replacement_cost(100.0, 5.0, 0.07, 30)
    assert got == pytest.approx(_manual_repl(100.0, 5.0, 0.07, 30), rel=1e-6)


def test_repl_no_cap_at_many_events():
    # sub-annual interval -> ~80 events; closed form must NOT truncate at 20
    t = 1.0e8 / (10.0 * 8760.0 * 3600.0 * 0.85)  # cap_shot_lifetime / shots_per_yr at 10 Hz
    got = levelized_replacement_cost(100.0, t, 0.07, 30)
    assert got == pytest.approx(_manual_repl(100.0, t, 0.07, 30), rel=1e-6)


def test_repl_zero_when_item_outlives_plant():
    # interval 40 yr > 30 yr plant -> 0 replacements beyond the initial set
    assert levelized_replacement_cost(100.0, 40.0, 0.07, 30) == pytest.approx(0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_economics.py -v`
Expected: FAIL with `ImportError: cannot import name 'levelized_replacement_cost'`

- [ ] **Step 3: Write minimal implementation**

In `src/costingfe/layers/economics.py`, after the `levelized_annual_cost` function (before `compute_lcoe`), add:

```python
def levelized_replacement_cost(
    event_cost: float,
    t_replace: float,
    interest_rate: float,
    plant_lifetime: float,
) -> float:
    """Level annual cost of replacing an item every ``t_replace`` years.

    Closed form of the discrete replacement-PV series used by the core,
    DEC-grid, and cap-bank blocks, with no iteration cap so it is exact for
    sub-annual to multi-decade intervals. Nominal discount only, PV at
    operation start, annualized by CRF. The first set is capital, so only
    replacements beyond it are charged: n_rep = ceil(n / t) - 1.

    pv = event_cost * sum_{k=1}^{n_rep} s^k = event_cost * s (1 - s^n_rep)/(1 - s),
    with s = (1 + i)^(-t_replace). n_rep = 0 gives pv = 0.
    """
    i = interest_rate
    n = plant_lifetime
    s = (1.0 + i) ** (-t_replace)  # per-interval discount, < 1 for i > 0
    n_rep = jnp.maximum(0.0, jnp.ceil(n / t_replace) - 1.0)
    pv = event_cost * s * (1.0 - s ** n_rep) / (1.0 - s)
    return pv * compute_crf(i, n)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_economics.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/costingfe/layers/economics.py tests/test_economics.py
git commit -m "Add levelized_replacement_cost geometric closed-form helper"
```

---

### Task 2: Route core / DEC grid / cap bank through the helper

Behavior-preserving refactor: removes three `MAX_REP` loops and the standalone `crf`/`MAX_REP`. Shipped numbers are unchanged (all shipped `n_rep <= 20`); fixes the latent cap-bank truncation when `n_rep > 20`.

**Files:**
- Modify: `src/costingfe/layers/costs.py:303-355` (CAS72 core, DEC, cap-bank blocks)
- Test: `tests/test_costs.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_costs.py`:

```python
import math as _math

from costingfe.layers.economics import levelized_replacement_cost
from costingfe.types import PulsedConversion


def _manual_repl_check(event_cost, t_replace, i, n):
    s = (1.0 + i) ** (-t_replace)
    n_rep = max(0, _math.ceil(n / t_replace) - 1)
    pv = sum(event_cost * s ** k for k in range(1, n_rep + 1))
    crf = (i * (1 + i) ** n) / ((1 + i) ** n - 1)
    return pv * crf


def test_cas72_core_matches_helper():
    """Core replacement equals the geometric helper for the summed accounts."""
    detail = {"C220101": 100.0, "C220104": 0.0, "C220108": 50.0}
    _, _, c72 = cas70_om(
        CC,
        cas22_detail=detail,
        replaceable_accounts=CC.replaceable_accounts,
        n_mod=1,
        p_net=1000.0,
        availability=0.85,
        inflation_rate=0.02,
        interest_rate=0.07,
        lifetime_yr=30,
        core_lifetime=CC.core_lifetime(Fuel.DT),
        construction_time=6,
        fuel=Fuel.DT,
        noak=True,
        concept=ConfinementConcept.TOKAMAK,
    )
    cost_per_event = sum(detail[k] for k in CC.replaceable_accounts if k in detail)
    t_core = CC.core_lifetime(Fuel.DT) / 0.85
    assert float(c72) == pytest.approx(
        _manual_repl_check(cost_per_event, t_core, 0.07, 30), rel=1e-6
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_costs.py::test_cas72_core_matches_helper -v`
Expected: FAIL (current loop value differs from helper only if truncated; for this case it should already be close — if it PASSES, that is acceptable, the refactor below must keep it passing). If the assertion is exact it will pass post-refactor; proceed regardless.

> Note: this is a characterization test. Its job is to stay green across the refactor. If it is already green, that confirms the math; the refactor must not break it.

- [ ] **Step 3: Refactor the three blocks**

In `src/costingfe/layers/costs.py`, replace the core/DEC/cap-bank section (the block from `# CAS72: Annualized scheduled replacement` through the end of the cap-bank `if`, currently lines ~303-355) with:

```python
    # CAS72: Annualized scheduled replacement. Each term is the level annual
    # cost of replacing an item every t_replace years over the plant life,
    # computed by the shared geometric closed-form helper (exact for any
    # interval; the first set is capital, so only replacements beyond it count).
    core_lifetime_cal = core_lifetime / availability  # FPY → calendar years
    cost_per_event = sum(cas22_detail[k] for k in replaceable_accounts) * n_mod
    cas72 = levelized_replacement_cost(
        cost_per_event, core_lifetime_cal, interest_rate, lifetime_yr
    )

    # DEC grid replacement (additive, independent cycle).
    # jnp.maximum(p_dee, 1e-6) keeps the power-law gradient finite at p_dee=0;
    # the outer jnp.where masks the result to zero when p_dee == 0.
    P_DEE_REF = 400.0
    p_dee_safe = jnp.maximum(p_dee, 1e-6)
    dec_grid = cc.dec_grid_cost * jnp.where(
        p_dee > 0, (p_dee_safe / P_DEE_REF) ** 0.7, 0.0
    )
    dec_grid_life_cal = cc.dec_grid_lifetime(fuel) / availability
    cas72 = cas72 + levelized_replacement_cost(
        dec_grid * n_mod, dec_grid_life_cal, interest_rate, lifetime_yr
    )

    # Cap bank scheduled replacement (INDUCTIVE_DEC only).
    if pulsed_conversion == PulsedConversion.INDUCTIVE_DEC and f_rep > 0:
        n_shots_per_year = f_rep * 8760.0 * 3600.0 * availability
        t_replace_cap = cc.cap_shot_lifetime / n_shots_per_year
        cap_cost = cas22_detail.get("C220107", 0.0) * n_mod
        cas72 = cas72 + levelized_replacement_cost(
            cap_cost, t_replace_cap, interest_rate, lifetime_yr
        )
```

Then add `levelized_replacement_cost` to the existing economics import near the top of `costs.py` (the `from costingfe.layers.economics import (...)` block at lines 14-17):

```python
from costingfe.layers.economics import (
    compute_crf,
    levelized_annual_cost,
    levelized_replacement_cost,
)
```

(`compute_crf` is still used by the electrode block's `levelized_annual_cost`; leave it imported.)

- [ ] **Step 4: Run the full suite to verify no number changes**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS — all existing tests green (the refactor is behavior-preserving for shipped configs), plus `test_cas72_core_matches_helper`.

- [ ] **Step 5: Commit**

```bash
git add src/costingfe/layers/costs.py tests/test_costs.py
git commit -m "Route core/DEC/cap-bank CAS72 through shared replacement helper (fixes >20-event cap-bank truncation)"
```

---

### Task 3: `LaserDriverType` enum

**Files:**
- Modify: `src/costingfe/types.py:30-33` (after `PulsedConversion`)
- Test: `tests/test_types.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_types.py`:

```python
def test_laser_driver_type_values():
    from costingfe.types import LaserDriverType

    assert LaserDriverType("dpssl") == LaserDriverType.DPSSL
    assert LaserDriverType("krf") == LaserDriverType.KRF
    assert LaserDriverType("nd_glass") == LaserDriverType.NDGLASS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_types.py::test_laser_driver_type_values -v`
Expected: FAIL with `ImportError: cannot import name 'LaserDriverType'`

- [ ] **Step 3: Add the enum**

In `src/costingfe/types.py`, after the `PulsedConversion` enum (line 32) add:

```python
class LaserDriverType(Enum):
    """Laser-IFE driver architecture.

    Selects, for LASER_IFE, both the C220104 capital coefficient ($/MJ) and the
    CAS72 scheduled-replacement subsystem set. Chosen via the laser_driver_type
    parameter (concept-YAML default + per-run override), not a separate concept.

    DPSSL is the commercial baseline (LIFE / HiPER / Focused Energy / Marvel);
    KRF carries the NRL Electra / Xcimer heritage; NDGLASS (NIF-class) is
    flagged commercially marginal — flashlamp shot life is Xe-arc-limited.
    """

    DPSSL = "dpssl"        # diode-pumped solid-state
    KRF = "krf"            # KrF excimer
    NDGLASS = "nd_glass"   # flashlamp-pumped Nd:Glass (NIF-class)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_types.py::test_laser_driver_type_values -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/costingfe/types.py tests/test_types.py
git commit -m "Add LaserDriverType enum (DPSSL/KrF/Nd:Glass)"
```

---

### Task 4: Capital + subsystem replacement defaults (dataclass + YAML)

**Files:**
- Modify: `src/costingfe/defaults.py:98` (capital coeffs) and `:178` (after electrode fields)
- Modify: `src/costingfe/data/defaults/costing_constants.yaml:239` (after `cap_shot_lifetime`)
- Test: `tests/test_costs.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_costs.py`:

```python
def test_laser_driver_capital_and_subsystem_defaults():
    assert CC.driver_laser_per_mj == pytest.approx(205.0)
    assert CC.driver_krf_per_mj == pytest.approx(40.0)
    assert CC.driver_ndglass_per_mj == pytest.approx(1000.0)
    # DPSSL subsystem NOAK pairs
    assert CC.dpssl_diode_replace_frac == pytest.approx(0.50)
    assert CC.dpssl_diode_shot_lifetime == pytest.approx(1.0e10)
    assert CC.dpssl_crystal_shot_lifetime == pytest.approx(3.0e9)
    assert CC.dpssl_optics_shot_lifetime == pytest.approx(3.0e8)
    # KrF + Nd:Glass
    assert CC.krf_foil_shot_lifetime == pytest.approx(3.0e8)
    assert CC.ndglass_lamp_shot_lifetime == pytest.approx(1.0e4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_costs.py::test_laser_driver_capital_and_subsystem_defaults -v`
Expected: FAIL with `AttributeError: ... 'driver_krf_per_mj'`

- [ ] **Step 3a: Add capital coefficients to the dataclass**

In `src/costingfe/defaults.py`, immediately after the `driver_laser_per_mj` field and its trailing comment lines (after line 100, before `driver_heavy_ion_per_mj`), add:

```python
    # KrF excimer (NRL Electra / Xcimer) and flashlamp Nd:Glass (NIF-class)
    # driver capital, $/MJ of pulse energy, selected by laser_driver_type for
    # LASER_IFE. KrF 40 leans to the Xcimer/ASPEN large-aperture-optics claim
    # (range 20-200; NRL/Sethian engineering baseline ~200). Nd:Glass 1000 from
    # NIF $3.5-4.2B / 1.1-1.9 MJ UV (~$2000/J facility, driver-only ~half).
    # See docs/account_justification/CAS22_reactor_components.md (C220104).
    driver_krf_per_mj: float = 40.0  # M$/MJ KrF excimer driver hardware
    driver_ndglass_per_mj: float = 1000.0  # M$/MJ flashlamp Nd:Glass driver
```

- [ ] **Step 3b: Add subsystem replacement pairs to the dataclass**

In `src/costingfe/defaults.py`, after the electrode fields (after `electrode_replace_frac`, line 178), add:

```python
    # CAS72 laser-IFE driver scheduled replacement, per architecture. Each
    # subsystem: replace_frac = share of C220104; shot_lifetime = NOAK
    # projection (shots), NOT demonstrated. Dispatched by laser_driver_type via
    # the shared geometric replacement helper. KrF/Nd:Glass cost shares are
    # engineering estimates. See CAS22_reactor_components.md (CAS72 O&M).
    # DPSSL (LIFE/HiPER): diodes are ~plant-life (≈capital), optics dominate O&M.
    dpssl_diode_replace_frac: float = 0.50  # pump diodes (dominant cost share)
    dpssl_diode_shot_lifetime: float = 1.0e10  # NOAK ≈ plant life; demonstrated ~1e8
    dpssl_crystal_replace_frac: float = 0.03  # KDP/DKDP tripler crystals (small)
    dpssl_crystal_shot_lifetime: float = 3.0e9  # long-lived
    dpssl_optics_replace_frac: float = 0.05  # final optics / GIMM / debris shields
    dpssl_optics_shot_lifetime: float = 3.0e8  # GIMM NOAK target; demonstrated ~1e5
    # KrF excimer (engineering estimates)
    krf_foil_replace_frac: float = 0.04  # hibachi foil + windows
    krf_foil_shot_lifetime: float = 3.0e8  # Electra durability target; demonstrated ~1e4-1e5
    krf_ebeam_replace_frac: float = 0.06  # e-beam diode + gas system
    krf_ebeam_shot_lifetime: float = 3.0e8
    # Nd:Glass (NIF-class): flashlamps are Xe-arc-limited; glass slabs are capital
    ndglass_lamp_replace_frac: float = 0.10  # Xe flashlamps
    ndglass_lamp_shot_lifetime: float = 1.0e4  # demonstrated O(1e3-1e4); arc-limited
```

- [ ] **Step 3c: Mirror the tunable values into the YAML**

In `src/costingfe/data/defaults/costing_constants.yaml`, after the `cap_shot_lifetime` line (239), add:

```yaml

# CAS22.01.04 laser-IFE driver capital by architecture (M$/MJ, selected by
# laser_driver_type). DPSSL 205 (existing); KrF 40 (Xcimer/ASPEN-leaning,
# range 20-200); Nd:Glass 1000 (NIF-class). See CAS22_reactor_components.md.
driver_laser_per_mj: 205.0
driver_krf_per_mj: 40.0
driver_ndglass_per_mj: 1000.0

# CAS72 laser-IFE driver scheduled replacement (per architecture).
# replace_frac = share of C220104; shot_lifetime = NOAK projection (shots).
dpssl_diode_replace_frac: 0.50
dpssl_diode_shot_lifetime: 1.0e+10
dpssl_crystal_replace_frac: 0.03
dpssl_crystal_shot_lifetime: 3.0e+9
dpssl_optics_replace_frac: 0.05
dpssl_optics_shot_lifetime: 3.0e+8
krf_foil_replace_frac: 0.04
krf_foil_shot_lifetime: 3.0e+8
krf_ebeam_replace_frac: 0.06
krf_ebeam_shot_lifetime: 3.0e+8
ndglass_lamp_replace_frac: 0.10
ndglass_lamp_shot_lifetime: 1.0e+4
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_costs.py::test_laser_driver_capital_and_subsystem_defaults -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/costingfe/defaults.py src/costingfe/data/defaults/costing_constants.yaml tests/test_costs.py
git commit -m "Add laser driver capital + per-subsystem replacement defaults (dataclass + YAML)"
```

---

### Task 5: Branch C220104 capital by driver type in `cas22.py`

**Files:**
- Modify: `src/costingfe/layers/cas22.py` (imports; signature `:165`; C220104 block `:368-386`)
- Test: `tests/test_costs.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_costs.py`:

```python
from costingfe.model import CostModel
from costingfe.types import LaserDriverType


def _laser_c220104(driver_type):
    model = CostModel(
        concept=ConfinementConcept.LASER_IFE,
        fuel=Fuel.DT,
        laser_driver_type=driver_type,
    )
    result = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    return float(result.cas22_detail["C220104"])


def test_c220104_branches_by_driver_type():
    """C220104 = coeff * E_drv; KrF/Nd:Glass scale vs DPSSL by 40/205 and 1000/205."""
    dpssl = _laser_c220104(LaserDriverType.DPSSL)
    krf = _laser_c220104(LaserDriverType.KRF)
    ndglass = _laser_c220104(LaserDriverType.NDGLASS)
    assert krf == pytest.approx(dpssl * 40.0 / 205.0, rel=1e-6)
    assert ndglass == pytest.approx(dpssl * 1000.0 / 205.0, rel=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_costs.py::test_c220104_branches_by_driver_type -v`
Expected: FAIL — `CostModel.__init__() got an unexpected keyword argument 'laser_driver_type'` (added in Task 7) OR capital not branched. Proceed; this task adds the cas22 branch, Task 7 adds the model kwarg. Re-run at end of Task 7.

- [ ] **Step 3a: Import the enum in `cas22.py`**

Find the `from costingfe.types import (...)` block near the top of `src/costingfe/layers/cas22.py` and add `LaserDriverType` to it.

- [ ] **Step 3b: Add the parameter to the function signature**

In `src/costingfe/layers/cas22.py`, in `cas22_reactor_plant_equipment`, after the `e_preheat_mj: float = 0.0,` parameter (line 166), add:

```python
    laser_driver_type=None,  # LaserDriverType for LASER_IFE; selects C220104 $/MJ
```

- [ ] **Step 3c: Branch the coefficient**

In the `else` branch of the C220104 block (the pulsed branch, lines ~368-386), replace:

```python
        c220104 = (
            _DRIVER_COST_PER_MJ.get(concept, 0.0) * e_driver_mj
            + _DRIVER_COST_PER_MW.get(concept, 0.0) * p_driver
            + cc.laser_preheat_per_mj * e_preheat_mj
        )
```

with:

```python
        # For LASER_IFE the per-MJ coefficient is set by the driver architecture
        # (DPSSL/KrF/Nd:Glass). The MagLIF preheat line stays DPSSL-class.
        _LASER_DRIVER_PER_MJ = {
            LaserDriverType.DPSSL: cc.driver_laser_per_mj,
            LaserDriverType.KRF: cc.driver_krf_per_mj,
            LaserDriverType.NDGLASS: cc.driver_ndglass_per_mj,
        }
        if concept == ConfinementConcept.LASER_IFE and laser_driver_type is not None:
            drv_per_mj = _LASER_DRIVER_PER_MJ[laser_driver_type]
        else:
            drv_per_mj = _DRIVER_COST_PER_MJ.get(concept, 0.0)
        c220104 = (
            drv_per_mj * e_driver_mj
            + _DRIVER_COST_PER_MW.get(concept, 0.0) * p_driver
            + cc.laser_preheat_per_mj * e_preheat_mj
        )
```

- [ ] **Step 4: Defer verification to Task 7**

Run: `.venv/bin/python -m pytest tests/ -q -k "not test_c220104_branches_by_driver_type and not laser_driver"`
Expected: PASS (no regressions; the new model-level test still fails until Task 7 wires the kwarg).

- [ ] **Step 5: Commit**

```bash
git add src/costingfe/layers/cas22.py tests/test_costs.py
git commit -m "Branch C220104 laser capital by driver architecture"
```

---

### Task 6: Laser-driver replacement dispatch in `cas70_om`

**Files:**
- Modify: `src/costingfe/layers/costs.py` (import; `cas70_om` signature `:273`; new block before `return` `:380`)
- Test: `tests/test_costs.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_costs.py`:

```python
_LASER_KWARGS = dict(
    cas22_detail={"C220101": 0.0, "C220104": 1000.0, "C220108": 0.0},
    replaceable_accounts=CC.replaceable_accounts,
    n_mod=1,
    p_net=1000.0,
    availability=0.85,
    inflation_rate=0.02,
    interest_rate=0.07,
    lifetime_yr=30,
    core_lifetime=CC.core_lifetime(Fuel.DT),
    construction_time=6,
    fuel=Fuel.DT,
    noak=True,
    f_rep=10.0,
    concept=ConfinementConcept.LASER_IFE,
)


def _laser_cas72(driver_type):
    _, _, c72 = cas70_om(CC, laser_driver_type=driver_type, **_LASER_KWARGS)
    _, _, c72_none = cas70_om(CC, laser_driver_type=None, **_LASER_KWARGS)
    return float(c72 - c72_none)


def test_cas72_dpssl_replacement_magnitude():
    """DPSSL replacement = sum over (diodes, crystals, optics) of the geometric
    helper applied to replace_frac*C220104 with t = shot_life / shots_per_yr."""
    n_shots = 10.0 * 8760.0 * 3600.0 * 0.85
    c220104 = 1000.0
    expected = (
        levelized_replacement_cost(
            CC.dpssl_diode_replace_frac * c220104,
            CC.dpssl_diode_shot_lifetime / n_shots, 0.07, 30)
        + levelized_replacement_cost(
            CC.dpssl_crystal_replace_frac * c220104,
            CC.dpssl_crystal_shot_lifetime / n_shots, 0.07, 30)
        + levelized_replacement_cost(
            CC.dpssl_optics_replace_frac * c220104,
            CC.dpssl_optics_shot_lifetime / n_shots, 0.07, 30)
    )
    assert _laser_cas72(LaserDriverType.DPSSL) == pytest.approx(float(expected), rel=1e-6)


def test_cas72_dpssl_diodes_outlive_plant_contribute_zero():
    """At NOAK diode life (1e10 ≈ 37 yr > plant) the diode term is ~0; dropping
    diode life to demonstrated 1e8 makes it dominate (sensitivity lever)."""
    n_shots = 10.0 * 8760.0 * 3600.0 * 0.85
    diode_noak = levelized_replacement_cost(
        CC.dpssl_diode_replace_frac * 1000.0,
        CC.dpssl_diode_shot_lifetime / n_shots, 0.07, 30)
    diode_demo = levelized_replacement_cost(
        CC.dpssl_diode_replace_frac * 1000.0, 1.0e8 / n_shots, 0.07, 30)
    assert float(diode_noak) == pytest.approx(0.0)
    assert float(diode_demo) > 100.0  # huge if diodes wear sub-annually


def test_cas72_ndglass_dominates_dpssl():
    """Flashlamp replacement is prohibitive -> Nd:Glass CAS72 >> DPSSL."""
    assert _laser_cas72(LaserDriverType.NDGLASS) > 10.0 * _laser_cas72(LaserDriverType.DPSSL)


def test_cas72_no_laser_replacement_guards():
    """No-op for non-laser concept, f_rep=0, or laser_driver_type=None."""
    tok = {**_LASER_KWARGS, "concept": ConfinementConcept.TOKAMAK}
    _, _, a = cas70_om(CC, laser_driver_type=LaserDriverType.DPSSL, **tok)
    _, _, b = cas70_om(CC, laser_driver_type=None, **tok)
    assert float(a) == float(b)
    zero = {**_LASER_KWARGS, "f_rep": 0.0}
    _, _, c = cas70_om(CC, laser_driver_type=LaserDriverType.DPSSL, **zero)
    _, _, d = cas70_om(CC, laser_driver_type=None, **zero)
    assert float(c) == float(d)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_costs.py -k "cas72_dpssl or ndglass or laser_replacement_guards" -v`
Expected: FAIL — `cas70_om() got an unexpected keyword argument 'laser_driver_type'`

- [ ] **Step 3a: Add the parameter and import**

In `src/costingfe/layers/costs.py`, add `laser_driver_type=None,` to the `cas70_om` signature (after `concept=None,`, line 272). Add `LaserDriverType` to the top-level `from costingfe.types import ...` line (line 33):

```python
from costingfe.types import BlanketFill, ConfinementConcept, Fuel, LaserDriverType, PulsedConversion
```

- [ ] **Step 3b: Add the dispatch block**

In `cas70_om`, immediately before the final `return cas71 + cas72, cas71, cas72` line, add:

```python
    # Laser-IFE driver scheduled replacement (DPSSL / KrF / Nd:Glass).
    # Each architecture has its own replaceable subsystems with distinct shot
    # lifetimes; each is summed via the shared geometric helper. Diodes whose
    # NOAK life exceeds the plant contribute ~0 (capital); flashlamps wear
    # sub-annually and dominate. See CAS22_reactor_components.md (CAS72 O&M).
    _LASER_SUBSYSTEMS = {
        LaserDriverType.DPSSL: (
            ("dpssl_diode_replace_frac", "dpssl_diode_shot_lifetime"),
            ("dpssl_crystal_replace_frac", "dpssl_crystal_shot_lifetime"),
            ("dpssl_optics_replace_frac", "dpssl_optics_shot_lifetime"),
        ),
        LaserDriverType.KRF: (
            ("krf_foil_replace_frac", "krf_foil_shot_lifetime"),
            ("krf_ebeam_replace_frac", "krf_ebeam_shot_lifetime"),
        ),
        LaserDriverType.NDGLASS: (
            ("ndglass_lamp_replace_frac", "ndglass_lamp_shot_lifetime"),
        ),
    }
    if (
        concept == ConfinementConcept.LASER_IFE
        and f_rep > 0
        and laser_driver_type is not None
    ):
        n_shots_per_year = f_rep * 8760.0 * 3600.0 * availability
        c220104 = cas22_detail.get("C220104", 0.0) * n_mod
        for frac_attr, life_attr in _LASER_SUBSYSTEMS[laser_driver_type]:
            event_cost = getattr(cc, frac_attr) * c220104
            t_replace = getattr(cc, life_attr) / n_shots_per_year
            cas72 = cas72 + levelized_replacement_cost(
                event_cost, t_replace, interest_rate, lifetime_yr
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_costs.py -k "cas72_dpssl or ndglass or laser_replacement_guards" -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/costingfe/layers/costs.py tests/test_costs.py
git commit -m "Add laser-IFE driver scheduled replacement dispatch in cas70_om"
```

---

### Task 7: Wire `laser_driver_type` through the model + concept YAML default

**Files:**
- Modify: `src/costingfe/model.py` (import; `__init__` `:75-92`; cas22 call `:744`; cas70_om call `:880`)
- Modify: `src/costingfe/data/defaults/pulsed_laser_ife.yaml` (add default)
- Test: `tests/test_model.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_model.py`:

```python
def test_laser_driver_type_defaults_to_dpssl_from_yaml():
    from costingfe.types import LaserDriverType

    model = CostModel(concept=ConfinementConcept.LASER_IFE, fuel=Fuel.DT)
    assert model.laser_driver_type == LaserDriverType.DPSSL


def test_laser_driver_type_override_changes_capital_and_om():
    from costingfe.types import LaserDriverType

    dpssl = CostModel(
        ConfinementConcept.LASER_IFE, Fuel.DT, laser_driver_type=LaserDriverType.DPSSL
    ).forward(1000.0, 0.85, 30)
    krf = CostModel(
        ConfinementConcept.LASER_IFE, Fuel.DT, laser_driver_type=LaserDriverType.KRF
    ).forward(1000.0, 0.85, 30)
    assert krf.cas22_detail["C220104"] < dpssl.cas22_detail["C220104"]  # 40 < 205 $/MJ
    assert krf.costs.lcoe != dpssl.costs.lcoe
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_model.py -k laser_driver_type -v`
Expected: FAIL — `__init__() got an unexpected keyword argument 'laser_driver_type'`

- [ ] **Step 3a: Import + constructor**

In `src/costingfe/model.py`, add `LaserDriverType` to the `from costingfe.types import (...)` block (near line 60). Add the parameter to `__init__` (after `pulsed_conversion: PulsedConversion = None,`, line 78):

```python
        laser_driver_type: "LaserDriverType" = None,
```

At the end of `__init__`, after `self._eng_defaults = load_engineering_defaults(...)` (line ~92), add:

```python
        # Driver architecture for LASER_IFE. Default comes from the concept YAML
        # (laser_driver_type: dpssl), not hardcoded; explicit arg overrides it.
        if laser_driver_type is None:
            _ld = self._eng_defaults.get("laser_driver_type")
            if _ld is not None:
                laser_driver_type = LaserDriverType(_ld)
        self.laser_driver_type = laser_driver_type
```

- [ ] **Step 3b: Pass into both layer calls**

In the `cas22_reactor_plant_equipment(...)` call, after `e_preheat_mj=e_preheat_mj,` (line ~740), add:

```python
            laser_driver_type=self.laser_driver_type,
```

In the `cas70_om(...)` call, after `concept=self.concept,` (line ~880), add:

```python
            laser_driver_type=self.laser_driver_type,
```

- [ ] **Step 3c: Concept YAML default**

In `src/costingfe/data/defaults/pulsed_laser_ife.yaml`, after the `pulsed_conversion: thermal` line, add:

```yaml
laser_driver_type: dpssl   # dpssl | krf | nd_glass (selects C220104 $/MJ + CAS72 replacement)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_model.py -k laser_driver_type tests/test_costs.py::test_c220104_branches_by_driver_type -v`
Expected: PASS (3 tests, including the deferred Task 5 capital test)

- [ ] **Step 5: Commit**

```bash
git add src/costingfe/model.py src/costingfe/data/defaults/pulsed_laser_ife.yaml tests/test_model.py
git commit -m "Wire laser_driver_type through model; default from concept YAML"
```

---

### Task 8: Adapter input + resolution

**Files:**
- Modify: `src/costingfe/adapter.py` (import `:13`; `FusionTeaInput` `:44`; `run_costing` `:96-104`)
- Test: `tests/test_adapter.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_adapter.py` (mirror the existing laser input at line ~27 — supply the same required fields that file already uses for `laser_ife`):

```python
def test_adapter_resolves_laser_driver_type():
    from costingfe.adapter import FusionTeaInput, run_costing

    base = dict(
        concept="laser_ife",
        fuel="dt",
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
    )
    dpssl = run_costing(FusionTeaInput(**base, laser_driver_type="dpssl"))
    krf = run_costing(FusionTeaInput(**base, laser_driver_type="krf"))
    assert krf.costs["CAS22"] != dpssl.costs["CAS22"]
```

(If `FusionTeaOutput.costs` uses different keys in this repo, assert on `krf.lcoe != dpssl.lcoe` instead — both are populated.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_adapter.py::test_adapter_resolves_laser_driver_type -v`
Expected: FAIL — `FusionTeaInput.__init__() got an unexpected keyword argument 'laser_driver_type'`

- [ ] **Step 3a: Import the enum**

In `src/costingfe/adapter.py`, add `LaserDriverType` to the `from costingfe.types import (...)` block (lines 13-18).

- [ ] **Step 3b: Add the input field**

In `FusionTeaInput`, after the `pulsed_conversion: str = (...)` field (line ~41), add:

```python
    laser_driver_type: str = ""  # "" = concept default; "dpssl" | "krf" | "nd_glass"
```

- [ ] **Step 3c: Resolve and pass to CostModel**

In `run_costing`, after the `pulsed_conv` resolution block (lines 96-97), add:

```python
    laser_drv = None
    if inp.laser_driver_type:
        laser_drv = LaserDriverType(inp.laser_driver_type)
```

Then add `laser_driver_type=laser_drv,` to the `CostModel(...)` constructor call (after `pulsed_conversion=pulsed_conv,`, line ~104).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_adapter.py::test_adapter_resolves_laser_driver_type -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/costingfe/adapter.py tests/test_adapter.py
git commit -m "Expose laser_driver_type through the fusion-tea adapter"
```

---

### Task 9: Account-justification doc (CAS22_reactor_components.md)

**Files:**
- Modify: `docs/account_justification/CAS22_reactor_components.md` (driver-cost table; new CAS72 section after the electrode section)

- [ ] **Step 1: Extend the `#### Driver costs` table**

In the `#### Driver costs` table, change the existing `Laser IFE` row's Coefficient/Rationale to note the architecture branch, and add two rows below it:

```markdown
| Laser IFE (DPSSL) | $/MJ | 205 | Diode-pumped solid-state (optics + diodes) | Default architecture; optics + diode arrays. With the C220107 cap bank (~$5/J) the laser totals ~$210/J = aggressive DPSSL NOAK (range $210-700/J; diode roadmap to ~$0.007/W). Selected by `laser_driver_type=dpssl`. |
| Laser IFE (KrF) | $/MJ | 40 | KrF excimer (e-beam pumped + gas) | NRL Electra / Xcimer heritage. 40 leans to the Xcimer/ASPEN large-aperture-optics claim ($10-20/J optical-on-target, unproven long-pulse scaling); NRL/Sethian engineering baseline is ~$200/J (FST 64, 2013). Range 20-200. `laser_driver_type=krf`. |
| Laser IFE (Nd:Glass) | $/MJ | 1000 | Flashlamp-pumped Nd:Glass (NIF-class) | NIF $3.5-4.2B / 1.1-1.9 MJ UV ≈ $2000/J facility, driver-only ~half. Commercially marginal: see the flashlamp replacement note below. `laser_driver_type=nd_glass`. |
```

- [ ] **Step 2: Add the scheduled-replacement section**

Immediately after the `#### Formation-electrode replacement (CAS72 O&M)` section (before the `---` that precedes `## C220105`), add:

```markdown
#### Laser-driver scheduled replacement (CAS72 O&M)

A rep-rated laser driver's replaceable subsystems wear at shot lifetimes
spanning sub-annual to multi-decade, so each is modeled as the level annual
cost of replacing it every `t = shot_lifetime / shots_per_year` years over the
plant life, summed via the shared geometric closed-form helper
(`levelized_replacement_cost`, the same machinery as core/DEC/cap-bank
replacement). The first set is capital (already in C220104); only replacements
beyond it are charged. Each subsystem cost is `replace_frac × C220104`. Shot
lifetimes are NOAK projections (LIFE / HiPER / NRL Electra), not demonstrated.

| Architecture | Subsystem | replace_frac | NOAK shot life | Demonstrated | Source |
|---|---|---:|---:|---:|---|
| DPSSL | Pump diodes | 0.50 | 1e10 | ~1e8 (Mercury) | Orth/Bibeau DPSSL studies; Mercury (OSTI 1019071); Zuegel ARPA-E 2023 |
| DPSSL | KDP/DKDP crystals | 0.03 | 3e9 | NIF 14.3 J/cm² spec | DKDP fatigue studies; NIF DKDP lifetime |
| DPSSL | Final optics (GIMM/transport/debris) | 0.05 | 3e8 | ~1e5 (GIMM) | Latkowski fused-silica final optics (OSTI 20845924); GIMM (UCSD-CER-05-08; FST 56(1)) |
| KrF | Hibachi foil + windows | 0.04 | 3e8 | ~1e4-1e5 | Sethian Electra (DTIC ADA480681) |
| KrF | E-beam diode + gas | 0.06 | 3e8 | — (engineering estimate) | Sethian Electra |
| Nd:Glass | Xe flashlamps | 0.10 | 1e4 | O(1e3-1e4) | NIF flashlamp specs (lasers.llnl.gov; SPIE 8599) |

Three consequences fall out of the calibration, at a $1B driver, 10 Hz, 0.85
availability, 30-yr / 7% WACC plant:

- **DPSSL diodes are capital, not O&M.** At NOAK life (1e10 shots ≈ 37 yr) the
  replacement interval exceeds the plant life, so `n_rep = 0` and diodes
  contribute ~$0. Diode shot life is the make-or-break sensitivity lever: at the
  demonstrated ~1e8 shots they would wear sub-annually and dominate LCOE.
- **DPSSL O&M is optics-dominated**, landing ~$48M/yr levelized — within the
  LIFE-projected band, driven by the final-optics line (~1.1 yr interval).
- **Nd:Glass is prohibitive.** Flashlamps wear sub-annually (~1e4 shots), so the
  replacement term explodes — the model surfaces NIF-class non-viability for
  rep-rated IFE, on top of its ~5× higher capital.

KrF and Nd:Glass subsystem cost shares are engineering estimates; no verified
component cost breakdown exists in the open literature.
```

- [ ] **Step 3: Verify Markdown renders (no broken table)**

Run: `.venv/bin/python -c "import pathlib; t=pathlib.Path('docs/account_justification/CAS22_reactor_components.md').read_text(); assert 'Laser-driver scheduled replacement' in t and t.count('|') > 0"`
Expected: no output (assertion passes)

- [ ] **Step 4: Commit**

```bash
git add docs/account_justification/CAS22_reactor_components.md
git commit -m "Document laser driver capital branch + CAS72 replacement (account justification)"
```

---

### Task 10: Paper (1costingfe_paper.tex §CAS22.01.04)

**Files:**
- Modify: `docs/papers/1costingfe_paper/1costingfe_paper.tex` (§`sec:cas2204`, around the 205 M$/MJ paragraph and after `tab:cas2204-driver`)

- [ ] **Step 1: Add the architecture-branch sentence**

In `1costingfe_paper.tex`, find the sentence beginning `The laser coefficient of $205$~M\$/MJ is set from published diode-pumped solid-state`. At the end of that paragraph (after `consistent with a diode roadmap toward \$0.007/W.`), append:

```latex
The laser coefficient branches on driver architecture, selected by
\texttt{laser\_driver\_type}: diode-pumped solid-state (DPSSL, the default) at
$205$~M\$/MJ; KrF excimer at $40$~M\$/MJ, leaning to the large-aperture-optics
projection for e-beam-pumped excimer (engineering range $20$--$200$~M\$/MJ); and
flashlamp-pumped Nd:glass (NIF-class) at $1000$~M\$/MJ, set from the NIF
\$3.5--4.2$\,$B / $1.1$--$1.9$~MJ-UV facility cost.
```

- [ ] **Step 2: Add the scheduled-replacement paragraph**

After the `\end{table}` for `tab:cas2204-driver`, add:

```latex
\paragraph{Driver scheduled replacement.}
A rep-rated laser driver's replaceable subsystems wear at shot lifetimes
spanning sub-annual to multi-decade, so each is charged as the level annual cost
of replacing it every $t = N_{\text{life}} / \dot N_{\text{shot}}$ years over the
plant life, summed by the same geometric replacement model used for the core,
direct-energy-converter grid, and capacitor bank. The first set is capital
(already in CAS22.01.04); only replacements beyond it are charged. For DPSSL the
pump diodes carry a NOAK shot life ($\sim$$10^{10}$) that exceeds a 30-year plant
at $10$~Hz, so they are effectively a capital item and the operating burden is
dominated by the final optics; for flashlamp-pumped Nd:glass the lamp shot life
($\sim$$10^4$) is sub-annual and the replacement term is prohibitive, which is
the standard reason NIF-class drivers are not viable for rep-rated energy.
```

- [ ] **Step 3: Verify LaTeX compiles**

Run: `cd docs/papers/1costingfe_paper && (latexmk -pdf -interaction=nonstopmode 1costingfe_paper.tex >/tmp/tex.log 2>&1 || pdflatex -interaction=nonstopmode 1costingfe_paper.tex >/tmp/tex.log 2>&1); tail -5 /tmp/tex.log`
Expected: build completes (a generated PDF appears). If `latexmk`/`pdflatex` is unavailable in the environment, skip and note it; do not block the task.

- [ ] **Step 4: Commit**

```bash
git add docs/papers/1costingfe_paper/1costingfe_paper.tex
git commit -m "Document laser driver architecture branch + CAS72 replacement (paper)"
```

---

### Task 11: Full verification

- [ ] **Step 1: Run the entire suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all green (375 existing + new tests). Note the new total in the commit if useful.

- [ ] **Step 2: Lint / format per repo convention**

Run: `.venv/bin/python -m ruff check src/ tests/ && .venv/bin/python -m ruff format --check src/ tests/` (or the repo's configured pre-commit: `pre-commit run --all-files`).
Expected: clean. Fix any findings and amend the relevant commit.

- [ ] **Step 3: Sanity-check the headline number**

Run:
```bash
.venv/bin/python -c "
from costingfe.model import CostModel
from costingfe.types import ConfinementConcept, Fuel, LaserDriverType
for d in (LaserDriverType.DPSSL, LaserDriverType.KRF, LaserDriverType.NDGLASS):
    r = CostModel(ConfinementConcept.LASER_IFE, Fuel.DT, laser_driver_type=d).forward(1000.0, 0.85, 30)
    print(d.value, 'C220104=%.1f' % float(r.cas22_detail['C220104']), 'LCOE=%.2f' % float(r.costs.lcoe))
"
```
Expected: C220104 scales 205:40:1000; Nd:Glass LCOE is by far the largest (flashlamp replacement); DPSSL and KrF are sane.

- [ ] **Step 4: Final commit / branch ready for PR**

```bash
git status   # confirm clean tree
git log --oneline master..HEAD   # review the task commits
```

Open a PR from `feat/laser-ife-driver-replacement-v2` into `master` (do not merge directly). Reference that it supersedes PR #28.

---

## Self-Review

**Spec coverage:**
- Capital branch (DPSSL/KrF/Nd:Glass) → Tasks 4, 5, 7. ✓
- Geometric shared helper + refactor of core/DEC/cap-bank → Tasks 1, 2. ✓
- Per-subsystem replacement dispatch → Tasks 4, 6. ✓
- Selection via YAML default + adapter/constructor override; remove hardcoded model default → Tasks 7, 8. ✓
- Defaults in YAML → Task 4 (costing_constants.yaml) + Task 7 (pulsed_laser_ife.yaml). ✓
- Docs in CAS22 justification + paper (incl. KrF 40) → Tasks 9, 10. ✓
- Tests incl. diode no-double-count, Nd:Glass prohibitive, guards, refactor regression, cap-bank truncation → Tasks 1, 2, 6. ✓

**Placeholder scan:** No TBD/TODO; all code blocks complete; the only conditional instruction (Task 8 key name, Task 10 LaTeX tool availability) gives an explicit fallback, not a placeholder.

**Type consistency:** `levelized_replacement_cost(event_cost, t_replace, interest_rate, plant_lifetime)` used identically in Tasks 1, 2, 6. `LaserDriverType` members `DPSSL/KRF/NDGLASS` and values `dpssl/krf/nd_glass` consistent across Tasks 3-8. Dataclass field names (`driver_krf_per_mj`, `dpssl_diode_shot_lifetime`, etc.) identical in Tasks 4, 5, 6. `laser_driver_type` parameter name identical across `cas22_reactor_plant_equipment`, `cas70_om`, `CostModel`, `FusionTeaInput`.
