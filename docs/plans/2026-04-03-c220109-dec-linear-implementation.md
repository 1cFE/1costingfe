# C220109 DEC for Linear Devices — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Populate C220109 with a cost model for add-on direct energy converters on linear devices (mirrors, steady-state FRCs), gated on `f_dec > 0`, scaled by DEC electric output (`p_dee`), with separate DEC grid replacement in CAS72.

**Architecture:** Add DEC constants to `CostingConstants`, pass `f_dec` and `p_dee` to `cas22_reactor_plant_equipment`, compute C220109 using power-law scaling, and add a separate DEC grid replacement term to `cas70_om`. No physics layer changes needed.

**Tech Stack:** Python, JAX, pytest

**Spec:** `docs/plans/2026-04-03-c220109-dec-linear-design.md`

---

### Task 1: Add DEC constants to CostingConstants

**Files:**
- Modify: `src/costingfe/defaults.py:60-85` (add constants after `divertor_base`)
- Modify: `src/costingfe/data/defaults/costing_constants.yaml:36-70` (add YAML entries)
- Test: `tests/test_defaults.py`

- [ ] **Step 1: Write failing test for new constants**

Add to `tests/test_defaults.py`:

```python
def test_dec_constants_exist():
    """DEC add-on constants should be loadable from defaults."""
    from costingfe.defaults import load_costing_constants

    cc = load_costing_constants()
    assert cc.dec_base == 100.0
    assert cc.dec_grid_cost == 12.0
    assert cc.dec_grid_lifetime_dt == 2.0
    assert cc.dec_grid_lifetime_dd == 3.0
    assert cc.dec_grid_lifetime_dhe3 == 4.0
    assert cc.dec_grid_lifetime_pb11 == 3.0


def test_dec_grid_lifetime_accessor():
    """dec_grid_lifetime(fuel) should return fuel-specific values."""
    from costingfe.defaults import load_costing_constants
    from costingfe.types import Fuel

    cc = load_costing_constants()
    assert cc.dec_grid_lifetime(Fuel.DT) == 2.0
    assert cc.dec_grid_lifetime(Fuel.DD) == 3.0
    assert cc.dec_grid_lifetime(Fuel.DHE3) == 4.0
    assert cc.dec_grid_lifetime(Fuel.PB11) == 3.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_defaults.py::test_dec_constants_exist tests/test_defaults.py::test_dec_grid_lifetime_accessor -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Add constants to CostingConstants dataclass**

In `src/costingfe/defaults.py`, add after `target_factory_base` (line ~63):

```python
    # C220109: DEC add-on for linear devices
    # Source: docs/account_justification/CAS220109_direct_energy_converter.md
    # Subsystem build-up: grids + power conditioning + incremental vacuum/tank
    dec_base: float = 100.0       # M$ at 400 MWe DEC electric output (P_DEE_REF)
    dec_grid_cost: float = 12.0   # M$ replaceable grid/collector modules at P_DEE_REF

    # DEC grid lifetime (FPY) — HIGH UNCERTAINTY, no reactor-scale data.
    # Conservative estimates. Primary degradation: sputtering + He blistering
    # from charged particle exhaust. Neutron damage additive for DT/DD.
    # Sensitivity range: 0.5x to 3x these values.
    dec_grid_lifetime_dt: float = 2.0    # Sputtering + 14.1 MeV neutron damage
    dec_grid_lifetime_dd: float = 3.0    # Sputtering + 2.45 MeV neutron damage
    dec_grid_lifetime_dhe3: float = 4.0  # 14.7 MeV proton sputtering + He blistering
    dec_grid_lifetime_pb11: float = 3.0  # 2.9 MeV alpha sputtering + severe He blistering
```

Add the accessor method after `core_lifetime()` (around line ~234):

```python
    def dec_grid_lifetime(self, fuel):
        """DEC grid replacement interval in FPY for a given fuel type."""
        from costingfe.types import Fuel

        return {
            Fuel.DT: self.dec_grid_lifetime_dt,
            Fuel.DD: self.dec_grid_lifetime_dd,
            Fuel.DHE3: self.dec_grid_lifetime_dhe3,
            Fuel.PB11: self.dec_grid_lifetime_pb11,
        }.get(fuel, self.dec_grid_lifetime_dt)
```

- [ ] **Step 4: Add constants to costing_constants.yaml**

In `src/costingfe/data/defaults/costing_constants.yaml`, add after `target_factory_base: 244.0` (line ~38):

```yaml
# C220109: DEC add-on for linear devices (M$ at 400 MWe DEC electric output)
# Source: docs/account_justification/CAS220109_direct_energy_converter.md
dec_base: 100.0              # Total DEC add-on cost at P_DEE_REF
dec_grid_cost: 12.0          # Replaceable grid/collector modules at P_DEE_REF

# DEC grid lifetime (FPY) — HIGH UNCERTAINTY, no reactor-scale data.
# Conservative estimates. Sensitivity range: 0.5x to 3x.
dec_grid_lifetime_dt: 2.0    # Sputtering + 14.1 MeV neutron damage
dec_grid_lifetime_dd: 3.0    # Sputtering + 2.45 MeV neutron damage
dec_grid_lifetime_dhe3: 4.0  # 14.7 MeV proton sputtering + He blistering
dec_grid_lifetime_pb11: 3.0  # 2.9 MeV alpha sputtering + severe He blistering
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_defaults.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/costingfe/defaults.py src/costingfe/data/defaults/costing_constants.yaml tests/test_defaults.py
git commit -m "feat: add DEC add-on costing constants (dec_base, dec_grid_cost, dec_grid_lifetime)"
```

---

### Task 2: Populate C220109 in cas22.py

**Files:**
- Modify: `src/costingfe/layers/cas22.py:53-76` (function signature) and `183-186` (C220109 block)
- Test: `tests/test_cas22.py`

- [ ] **Step 1: Write failing tests for C220109**

Add to `tests/test_cas22.py`:

```python
def _make_cas22_dec(f_dec=0.3, p_dee=300.0):
    """Helper for DEC tests — mirror with DEC."""
    rb = RadialBuild(R0=6.2, plasma_t=2.0, elon=1.7, blanket_t=0.70)
    geo = compute_geometry(rb, ConfinementConcept.TOKAMAK)
    return cas22_reactor_plant_equipment(
        CC,
        p_net=1000.0,
        p_th=2500.0,
        p_et=1100.0,
        p_fus=2300.0,
        p_cryo=0.5,
        n_mod=1,
        fuel=Fuel.DHE3,
        noak=True,
        blanket_vol=geo.firstwall_vol + geo.blanket_vol + geo.reflector_vol,
        shield_vol=geo.ht_shield_vol + geo.lt_shield_vol,
        structure_vol=geo.structure_vol,
        vessel_vol=geo.vessel_vol,
        family=ConfinementFamily.MFE,
        concept=ConfinementConcept.MIRROR,
        b_max=12.0,
        r_coil=1.85,
        coil_material=CoilMaterial.REBCO_HTS,
        p_nbi=50.0,
        p_icrf=0.0,
        p_ecrh=0.0,
        p_lhcd=0.0,
        f_dec=f_dec,
        p_dee=p_dee,
    )


def test_c220109_nonzero_when_dec_active():
    """C220109 should be nonzero when f_dec > 0 and p_dee > 0."""
    result = _make_cas22_dec(f_dec=0.3, p_dee=300.0)
    assert result["C220109"] > 0


def test_c220109_zero_when_no_dec():
    """C220109 should be zero when f_dec = 0."""
    result = _make_cas22_dec(f_dec=0.0, p_dee=0.0)
    assert result["C220109"] == 0.0


def test_c220109_scales_with_p_dee():
    """Higher DEC output should increase C220109."""
    low = _make_cas22_dec(f_dec=0.3, p_dee=200.0)
    high = _make_cas22_dec(f_dec=0.3, p_dee=600.0)
    assert high["C220109"] > low["C220109"]


def test_c220109_scaling_exponent():
    """C220109 should scale as (p_dee / P_DEE_REF) ** 0.7."""
    result = _make_cas22_dec(f_dec=0.3, p_dee=400.0)
    # At p_dee = P_DEE_REF = 400, scaling factor is 1.0
    expected = CC.dec_base * 1.0
    assert abs(result["C220109"] - expected) < 0.01


def test_c220109_included_in_total():
    """C220109 should be included in C220000 total."""
    with_dec = _make_cas22_dec(f_dec=0.3, p_dee=400.0)
    without_dec = _make_cas22_dec(f_dec=0.0, p_dee=0.0)
    assert with_dec["C220000"] > without_dec["C220000"]
    diff = with_dec["C220000"] - without_dec["C220000"]
    # Difference should include C220109 plus its share of installation labor
    assert diff > with_dec["C220109"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cas22.py::test_c220109_nonzero_when_dec_active -v`
Expected: FAIL with `TypeError` (unexpected keyword argument `f_dec`)

- [ ] **Step 3: Add f_dec and p_dee to cas22_reactor_plant_equipment**

In `src/costingfe/layers/cas22.py`, add two parameters to the function signature after `p_lhcd`:

```python
def cas22_reactor_plant_equipment(
    cc: CostingConstants,
    p_net: float,
    p_th: float,
    p_et: float,
    p_fus: float,
    p_cryo: float,
    n_mod: int,
    fuel: Fuel,
    noak: bool,
    blanket_vol: float,
    shield_vol: float,
    structure_vol: float,
    vessel_vol: float,
    family: ConfinementFamily,
    concept: ConfinementConcept,
    b_max: float,
    r_coil: float,
    coil_material: CoilMaterial,
    p_nbi: float,
    p_icrf: float,
    p_ecrh: float,
    p_lhcd: float,
    f_dec: float,
    p_dee: float,
) -> dict[str, float]:
```

- [ ] **Step 4: Replace C220109 computation**

Replace the C220109 block (lines ~183-186) with:

```python
    # -----------------------------------------------------------------------
    # 220109: Direct Energy Converter — add-on for linear devices
    # (mirrors, steady-state FRCs) with directed axial exhaust.
    # Covers: grid/collector modules, DC-AC power conditioning,
    # incremental vacuum/cryo, incremental tank volume, heat collection.
    # Applicable to venetian blind, TWDEC, or ICC — cost ranges overlap;
    # efficiency differences flow through eta_de in the physics layer.
    # Gated on f_dec > 0 (no fuel gating — user decides economic viability).
    # See docs/account_justification/CAS220109_direct_energy_converter.md
    # -----------------------------------------------------------------------
    P_DEE_REF = 400.0  # MW reference DEC electric output
    if f_dec > 0 and p_dee > 0:
        c220109 = cc.dec_base * (p_dee / P_DEE_REF) ** 0.7
    else:
        c220109 = 0.0
```

- [ ] **Step 5: Run new DEC tests to verify they pass**

Run: `python -m pytest tests/test_cas22.py::test_c220109_nonzero_when_dec_active tests/test_cas22.py::test_c220109_zero_when_no_dec tests/test_cas22.py::test_c220109_scales_with_p_dee tests/test_cas22.py::test_c220109_scaling_exponent tests/test_cas22.py::test_c220109_included_in_total -v`
Expected: All 5 new tests PASS. Existing tests will FAIL because `f_dec` and `p_dee` are now required — that's expected and fixed in Task 2.5.

- [ ] **Step 6: Commit**

```bash
git add src/costingfe/layers/cas22.py tests/test_cas22.py
git commit -m "feat: populate C220109 DEC cost, scaled by p_dee with 0.7 exponent"
```

---

### Task 2.5: Update all existing callers to pass f_dec and p_dee

`f_dec` and `p_dee` are required parameters (no defaults) on `cas22_reactor_plant_equipment`. All existing callers and test helpers must be updated to pass them explicitly.

**Files:**
- Modify: `tests/test_cas22.py` (all helpers: `_make_cas22`, `_make_cas22_with_family`, `_make_cas22_coil`, `_make_cas22_heating`)
- Modify: `src/costingfe/model.py:535-558` (the call site — will be done properly in Task 3, but needed now for test suite to pass)
- Modify: any other callers found by running the full test suite

- [ ] **Step 1: Update all test helpers in test_cas22.py**

Add `f_dec=0.0, p_dee=0.0` to every call to `cas22_reactor_plant_equipment` in the existing helpers.

In `_make_cas22` (around line 21), add after `p_lhcd=0.0`:
```python
        f_dec=0.0,
        p_dee=0.0,
```

In `_make_cas22_with_family` (around line 141), add after `p_lhcd=0.0`:
```python
        f_dec=0.0,
        p_dee=0.0,
```

In `_make_cas22_coil` (around line 230), add after `p_lhcd=0.0`:
```python
        f_dec=0.0,
        p_dee=0.0,
```

In `_make_cas22_heating` (around line 310), add after `p_lhcd=p_lhcd`:
```python
        f_dec=0.0,
        p_dee=0.0,
```

In `test_cas220110_concept_scales` (around line 389), add `f_dec=0.0, p_dee=0.0` to both direct calls to `cas22_reactor_plant_equipment` (the tokamak call around line 412 and the mirror call around line 437), after `p_lhcd=0.0`.

- [ ] **Step 2: Update model.py call site**

In `src/costingfe/model.py`, add `f_dec` and `p_dee` to the `cas22_reactor_plant_equipment` call (around line 535-558). Add after `p_lhcd=p_lhcd`:

```python
            f_dec=params.get("f_dec", 0.0),
            p_dee=float(pt.p_dee),
```

- [ ] **Step 3: Find and fix any other callers**

Run: `python -m pytest tests/ -v --tb=short 2>&1 | head -80`

Check for any remaining `TypeError: cas22_reactor_plant_equipment() missing required positional argument` errors. Fix any callers found (e.g., in `backcasting_bridge.py` or `adapter.py`).

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "refactor: pass f_dec and p_dee explicitly to all cas22 callers"
```

---

### Task 3: Integration tests for model-level DEC wiring

Model.py was already updated in Task 2.5 to pass `f_dec` and `p_dee`. This task adds tests to verify the wiring works end-to-end.

**Files:**
- Test: `tests/test_model.py`

- [ ] **Step 1: Write tests**

Add to `tests/test_model.py`:

```python
def test_mirror_dec_populates_c220109():
    """Mirror with f_dec > 0 should have nonzero C220109."""
    from costingfe import ConfinementConcept, CostModel, Fuel

    model = CostModel(concept=ConfinementConcept.MIRROR, fuel=Fuel.DHE3)
    result = model.forward(net_electric_mw=500.0, availability=0.85, lifetime_yr=30)
    # Mirror defaults: f_dec=0.3, eta_de=0.60 → p_dee > 0
    assert result.cas22_detail["C220109"] > 0, (
        "DHe3 mirror with f_dec=0.3 should have nonzero DEC cost"
    )


def test_tokamak_no_dec():
    """Tokamak with f_dec=0 should have zero C220109."""
    from costingfe import ConfinementConcept, CostModel, Fuel

    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    result = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    assert result.cas22_detail["C220109"] == 0.0
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/test_model.py::test_mirror_dec_populates_c220109 tests/test_model.py::test_tokamak_no_dec -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_model.py
git commit -m "test: verify model passes f_dec and p_dee to cas22"
```

---

### Task 4: Add DEC grid replacement to CAS72

**Files:**
- Modify: `src/costingfe/layers/costs.py:254-300` (cas70_om function)
- Modify: `src/costingfe/model.py:663-677` (cas70_om call site)
- Test: `tests/test_costs.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_costs.py`:

```python
def test_cas72_includes_dec_grid_replacement():
    """CAS72 should include DEC grid replacement when p_dee > 0."""
    from costingfe.defaults import load_costing_constants
    from costingfe.layers.costs import cas70_om
    from costingfe.types import Fuel

    cc = load_costing_constants()
    # Fake CAS22 detail with nonzero C220109
    cas22_detail = {
        "C220101": 500.0,
        "C220108": 60.0,
        "C220109": 100.0,
    }
    P_DEE_REF = 400.0
    p_dee = 400.0
    dec_grid_scaled = cc.dec_grid_cost * (p_dee / P_DEE_REF) ** 0.7

    _, _, cas72_with = cas70_om(
        cc,
        cas22_detail=cas22_detail,
        replaceable_accounts=("C220101", "C220108"),
        n_mod=1,
        p_net=1000.0,
        availability=0.85,
        inflation_rate=0.02,
        interest_rate=0.07,
        lifetime_yr=30,
        core_lifetime=5.0,
        construction_time=6.0,
        fuel=Fuel.DHE3,
        noak=True,
        p_dee=p_dee,
    )
    _, _, cas72_without = cas70_om(
        cc,
        cas22_detail=cas22_detail,
        replaceable_accounts=("C220101", "C220108"),
        n_mod=1,
        p_net=1000.0,
        availability=0.85,
        inflation_rate=0.02,
        interest_rate=0.07,
        lifetime_yr=30,
        core_lifetime=5.0,
        construction_time=6.0,
        fuel=Fuel.DHE3,
        noak=True,
        p_dee=0.0,
    )
    assert cas72_with > cas72_without, (
        "CAS72 should be higher with DEC grid replacement"
    )


def test_cas72_no_dec_grid_when_p_dee_zero():
    """CAS72 should not include DEC grid term when p_dee = 0."""
    from costingfe.defaults import load_costing_constants
    from costingfe.layers.costs import cas70_om
    from costingfe.types import Fuel

    cc = load_costing_constants()
    cas22_detail = {
        "C220101": 500.0,
        "C220108": 60.0,
        "C220109": 0.0,
    }
    _, _, cas72_a = cas70_om(
        cc,
        cas22_detail=cas22_detail,
        replaceable_accounts=("C220101", "C220108"),
        n_mod=1,
        p_net=1000.0,
        availability=0.85,
        inflation_rate=0.02,
        interest_rate=0.07,
        lifetime_yr=30,
        core_lifetime=5.0,
        construction_time=6.0,
        fuel=Fuel.DT,
        noak=True,
        p_dee=0.0,
    )
    _, _, cas72_b = cas70_om(
        cc,
        cas22_detail=cas22_detail,
        replaceable_accounts=("C220101", "C220108"),
        n_mod=1,
        p_net=1000.0,
        availability=0.85,
        inflation_rate=0.02,
        interest_rate=0.07,
        lifetime_yr=30,
        core_lifetime=5.0,
        construction_time=6.0,
        fuel=Fuel.DT,
        noak=True,
    )
    assert cas72_a == cas72_b, (
        "CAS72 should be identical when p_dee=0 vs not provided"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_costs.py::test_cas72_includes_dec_grid_replacement -v`
Expected: FAIL with `TypeError` (unexpected keyword argument `p_dee`)

- [ ] **Step 3: Add p_dee parameter and DEC grid replacement to cas70_om**

In `src/costingfe/layers/costs.py`, modify `cas70_om` signature to add `p_dee=0.0` after `noak`:

```python
def cas70_om(
    cc,
    cas22_detail,
    replaceable_accounts,
    n_mod,
    p_net,
    availability,
    inflation_rate,
    interest_rate,
    lifetime_yr,
    core_lifetime,
    construction_time,
    fuel,
    noak,
    p_dee=0.0,
):
```

After the existing CAS72 PV calculation (after `cas72 = pv * crf`, line ~298), add the DEC grid replacement:

```python
    # DEC grid replacement (additive, independent cycle)
    P_DEE_REF = 400.0
    if p_dee > 0:
        dec_grid = cc.dec_grid_cost * (p_dee / P_DEE_REF) ** 0.7
        dec_grid_life_cal = cc.dec_grid_lifetime(fuel) / availability
        n_rep_dec = jnp.maximum(0.0, jnp.ceil(lifetime_yr / dec_grid_life_cal) - 1.0)
        dec_cost = dec_grid * n_mod
        pv_dec = 0.0
        for k in range(1, MAX_REP + 1):
            discount = (1 + interest_rate) ** (k * dec_grid_life_cal)
            pv_dec = pv_dec + jnp.where(k <= n_rep_dec, dec_cost / discount, 0.0)
        cas72 = cas72 + pv_dec * crf

    return cas71 + cas72, cas71, cas72
```

- [ ] **Step 4: Pass p_dee from model.py to cas70_om**

In `src/costingfe/model.py`, modify the `cas70_om` call (around line 663) to add `p_dee`:

```python
        c70, c71, c72 = cas70_om(
            cc,
            cas22_detail=c22_detail,
            replaceable_accounts=repl_accounts,
            n_mod=n_mod,
            p_net=pt.p_net,
            availability=avail_eff,
            inflation_rate=inflation_rate,
            interest_rate=interest_rate,
            lifetime_yr=lifetime_yr,
            core_lifetime=core_lt,
            construction_time=construction_time_yr,
            fuel=self.fuel,
            noak=noak,
            p_dee=float(pt.p_dee),
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_costs.py -v`
Expected: All PASS

Run full suite:
Run: `python -m pytest tests/ -v --tb=short`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/costingfe/layers/costs.py src/costingfe/model.py tests/test_costs.py
git commit -m "feat: add DEC grid replacement to CAS72, fuel-dependent lifetime"
```

---

### Task 5: Update justification doc

**Files:**
- Modify: `docs/account_justification/CAS220109_direct_energy_converter.md`

- [ ] **Step 1: Update the recommended cost model section**

Replace the "Recommended Cost Model" section (starting around line 180) with:

```markdown
## Recommended Cost Model

### Scaling formula

    C220109 = dec_base * (p_dee / P_DEE_REF) ^ 0.7

where:
- `dec_base` = 100.0 M$ (total DEC add-on cost at reference output)
- `P_DEE_REF` = 400 MWe (reference DEC electric output)
- `p_dee` = DEC electric output in MW (from physics layer: `f_dec * eta_de * p_transport`)

C220109 is gated on `f_dec > 0`. No fuel gating is applied — the model
computes DEC cost for any fuel if the user sets `f_dec > 0`. Economic
viability is a user judgment (the DEC is not cost-effective for DT/DD
at demonstrated efficiencies, but the model does not prevent the user
from exploring this).

The 0.7 exponent reflects:
- Grid modules scale ~linearly with area (proportional to power)
- Power conditioning scales ~linearly with output
- Tank and vacuum scale sub-linearly (surface-to-volume)
- Consistent with other vendor-purchased power systems in CAS22

DEC type (venetian blind, TWDEC, ICC) is not distinguished in the cost
model. The add-on cost ranges overlap ($73-140M at ~400 MWe) because
the dominant costs (vacuum, cryo, power conditioning) are shared
infrastructure. Efficiency differences between DEC types flow through
`eta_de` in the physics layer.

### DEC grid replacement (CAS72)

DEC grids degrade under charged particle bombardment (sputtering,
helium blistering) and neutron damage. Grid replacement is modeled
as a separate CAS72 term, independent of blanket/divertor replacement:

    annual_replace_dec = dec_grid_cost * (p_dee / P_DEE_REF) ^ 0.7 / dec_grid_lifetime

**Grid lifetime (FPY) — HIGH UNCERTAINTY:** No reactor-scale data
exists for any DEC grid type. These are conservative estimates.
Primary degradation is from charged particle exhaust (sputtering,
He blistering), with neutron damage additive for DT/DD.
Sensitivity range: 0.5x to 3x.

| Fuel | dec_grid_lifetime (FPY) | Primary degradation |
|---|---|---|
| DT | 2.0 | Sputtering + 14.1 MeV neutron damage |
| DD | 3.0 | Sputtering + 2.45 MeV neutron damage |
| DHe3 | 4.0 | 14.7 MeV proton sputtering + He blistering |
| pB11 | 3.0 | 2.9 MeV alpha sputtering + severe He blistering (3 alpha per event) |
```

- [ ] **Step 2: Remove the fuel-specific base costs table**

Delete the existing table (around line 183-189) that lists per-fuel C220109 base costs ($0/$0/$90/$120). This is replaced by the scaling formula above.

- [ ] **Step 3: Commit**

```bash
git add docs/account_justification/CAS220109_direct_energy_converter.md
git commit -m "docs: update C220109 justification — remove fuel gating, add scaling formula and grid lifetime"
```

---

### Task 6: Integration test — full mirror plant with DEC

**Files:**
- Test: `tests/test_model.py`

- [ ] **Step 1: Write integration test**

Add to `tests/test_model.py`:

```python
def test_mirror_dhe3_dec_full_integration():
    """Full DHe3 mirror plant should have DEC costs in capital and O&M."""
    from costingfe import ConfinementConcept, CostModel, Fuel

    model = CostModel(concept=ConfinementConcept.MIRROR, fuel=Fuel.DHE3)
    result = model.forward(
        net_electric_mw=500.0,
        availability=0.85,
        lifetime_yr=30,
    )
    pt = result.power_table
    c = result.costs

    # Physics: DEC should produce electric power
    assert pt.p_dee > 0, "DHe3 mirror should have DEC electric output"

    # Capital: C220109 should be nonzero
    c220109 = result.cas22_detail["C220109"]
    assert c220109 > 0, "C220109 should be nonzero for DHe3 mirror"

    # C220109 should be included in CAS22 total
    assert c.cas22 > c220109, "CAS22 should include C220109"

    # O&M: CAS72 should include DEC grid replacement
    # Run same plant without DEC to compare
    model_no_dec = CostModel(concept=ConfinementConcept.MIRROR, fuel=Fuel.DHE3)
    result_no_dec = model_no_dec.forward(
        net_electric_mw=500.0,
        availability=0.85,
        lifetime_yr=30,
        f_dec=0.0,
    )
    assert c.cas72 > result_no_dec.costs.cas72, (
        "CAS72 should be higher with DEC grid replacement"
    )


def test_mirror_dt_no_dec_by_default():
    """DT mirror with default f_dec=0.3 should still compute DEC costs.

    The model does not gate on fuel — f_dec > 0 is sufficient.
    Mirror defaults have f_dec=0.3 for all fuels.
    """
    from costingfe import ConfinementConcept, CostModel, Fuel

    model = CostModel(concept=ConfinementConcept.MIRROR, fuel=Fuel.DT)
    result = model.forward(net_electric_mw=500.0, availability=0.85, lifetime_yr=30)
    # Mirror default f_dec=0.3 applies even for DT
    assert result.cas22_detail["C220109"] > 0
```

- [ ] **Step 2: Run integration tests**

Run: `python -m pytest tests/test_model.py::test_mirror_dhe3_dec_full_integration tests/test_model.py::test_mirror_dt_no_dec_by_default -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_model.py
git commit -m "test: integration tests for DEC costs on mirror plants"
```
