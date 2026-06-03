# Radiation Peaking Factor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `radiation_peaking_factor` that corrects the 0-D bremsstrahlung+line over-count for highly-peaked plasmas (dipole), fixing issue #24.

**Architecture:** `compute_p_rad` multiplies the two collisional volume-integral terms (brems + line) by an emission-measure factor while leaving synchrotron untouched. The factor is threaded from the YAML defaults through `model.py`'s `rad_kw` into the MFE power-balance functions. Library functions default the factor to 1.0 (mathematical identity, consistent with existing physics defaults like `kappa=1.7`); `model.py` reads the canonical value strictly from YAML.

**Tech Stack:** Python, JAX (`jax.numpy`), pytest, PyYAML.

---

### Task 1: Emission-measure factor in `compute_p_rad`

**Files:**
- Modify: `src/costingfe/layers/radiation.py:243-293`
- Test: `tests/test_physics.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_physics.py` (near the other `compute_p_rad` tests, after `test_no_impurities_unchanged`):

```python
def test_peaking_factor_scales_brems_linearly():
    """radiation_peaking_factor multiplies the bremsstrahlung term linearly."""
    n_e, T_e, Z_eff, V, B = 1e20, 15.0, 1.5, 500.0, 0.0  # B=0 -> no sync via Albajar gate
    p_full = compute_p_rad(n_e, T_e, Z_eff, V, B, R=0.0, a=0.0)
    p_half = compute_p_rad(n_e, T_e, Z_eff, V, B, R=0.0, a=0.0, radiation_peaking_factor=0.5)
    assert abs(float(p_half) - 0.5 * float(p_full)) < 1e-6 * float(p_full)


def test_peaking_factor_leaves_synchrotron_untouched():
    """The factor scales brems+line but NOT synchrotron."""
    # Geometry enabled (R,a>0) so synchrotron is nonzero.
    n_e, T_e, Z_eff, V, B = 1e20, 15.0, 1.5, 500.0, 5.0
    kw = dict(R=6.0, a=2.0, kappa=1.7, R_w=0.6)
    p_full = compute_p_rad(n_e, T_e, Z_eff, V, B, **kw)
    p_scaled = compute_p_rad(n_e, T_e, Z_eff, V, B, radiation_peaking_factor=0.5, **kw)
    # Synchrotron is invariant, so halving the factor must remove LESS than half
    # the total (the sync part is unscaled).
    assert float(p_scaled) > 0.5 * float(p_full)


def test_peaking_factor_default_is_one():
    """Default factor reproduces the pre-change result exactly."""
    n_e, T_e, Z_eff, V, B = 1e20, 15.0, 1.5, 500.0, 5.0
    p_a = compute_p_rad(n_e, T_e, Z_eff, V, B, R=6.0, a=2.0)
    p_b = compute_p_rad(n_e, T_e, Z_eff, V, B, R=6.0, a=2.0, radiation_peaking_factor=1.0)
    assert float(p_a) == float(p_b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_physics.py -k peaking -v`
Expected: FAIL with `TypeError: compute_p_rad() got an unexpected keyword argument 'radiation_peaking_factor'`

- [ ] **Step 3: Add the parameter and apply it**

In `src/costingfe/layers/radiation.py`, extend the `compute_p_rad` signature. Change:

```python
def compute_p_rad(
    n_e: float,
    T_e: float,
    Z_eff: float,
    volume: float,
    B: float = 0.0,
    impurities: ImpurityMix | None = None,
    R: float = 0.0,
    a: float = 0.0,
    kappa: float = 1.7,
    R_w: float = 0.6,
    alpha_n: float = 0.5,
    alpha_T: float = 1.0,
) -> float:
```

to add one parameter at the end:

```python
def compute_p_rad(
    n_e: float,
    T_e: float,
    Z_eff: float,
    volume: float,
    B: float = 0.0,
    impurities: ImpurityMix | None = None,
    R: float = 0.0,
    a: float = 0.0,
    kappa: float = 1.7,
    R_w: float = 0.6,
    alpha_n: float = 0.5,
    alpha_T: float = 1.0,
    radiation_peaking_factor: float = 1.0,
) -> float:
```

Update the docstring body (the block beginning "Volume-averaged n_e, T_e...") by appending:

```python
    radiation_peaking_factor scales the collisional volume-integral terms
    (bremsstrahlung + line) to correct peak-value-times-full-volume over-counting
    for peaked profiles (emission-measure ratio). It does NOT scale synchrotron,
    whose profile dependence is already carried by alpha_n/alpha_T. Default 1.0
    (uniform profile, no correction).
```

Change the final return statement from:

```python
    p_line = compute_p_line(n_e, T_e, impurities, volume)
    return p_brem + p_sync + p_line
```

to:

```python
    p_line = compute_p_line(n_e, T_e, impurities, volume)
    return radiation_peaking_factor * (p_brem + p_line) + p_sync
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_physics.py -k peaking -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/costingfe/layers/radiation.py tests/test_physics.py
git commit -m "Add radiation_peaking_factor to compute_p_rad (brems+line only)"
```

---

### Task 2: Thread the factor through the MFE power-balance functions

**Files:**
- Modify: `src/costingfe/layers/physics.py:177-241` (forward) and `src/costingfe/layers/physics.py:349-407` (inverse)
- Test: `tests/test_power_balance.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_power_balance.py`:

```python
def test_forward_peaking_factor_reduces_radiation():
    """A peaking factor < 1 lowers p_rad in the forward balance."""
    base = dict(CATF_PARAMS)
    pt_full = mfe_forward_power_balance(**base)
    pt_peaked = mfe_forward_power_balance(**{**base, "radiation_peaking_factor": 0.1})
    assert float(pt_peaked.p_rad) < float(pt_full.p_rad)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_power_balance.py::test_forward_peaking_factor_reduces_radiation -v`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'radiation_peaking_factor'`

- [ ] **Step 3: Add the parameter to both functions and forward it**

In `src/costingfe/layers/physics.py`, in `mfe_forward_power_balance`, the optional-override block currently reads:

```python
    # Genuine optional overrides (keep defaults)
    p_rad_override: float | None = None,
    f_rad_fus: float | None = None,
    seeded_impurities: dict[str, float] | None = None,
) -> PowerTable:
```

Change it to add the factor:

```python
    # Genuine optional overrides (keep defaults)
    p_rad_override: float | None = None,
    f_rad_fus: float | None = None,
    seeded_impurities: dict[str, float] | None = None,
    radiation_peaking_factor: float = 1.0,
) -> PowerTable:
```

In the same function, the `compute_p_rad` call currently reads:

```python
        p_rad = compute_p_rad(
            n_e,
            T_e,
            Z_eff,
            plasma_volume,
            B,
            impurities,
            R=R_major,
            a=a_minor,
            kappa=kappa,
            R_w=R_w,
        )
```

Change to add the factor:

```python
        p_rad = compute_p_rad(
            n_e,
            T_e,
            Z_eff,
            plasma_volume,
            B,
            impurities,
            R=R_major,
            a=a_minor,
            kappa=kappa,
            R_w=R_w,
            radiation_peaking_factor=radiation_peaking_factor,
        )
```

In `mfe_inverse_power_balance`, the optional-override block currently reads:

```python
    # Genuine optional overrides (keep defaults)
    p_rad_override: float | None = None,
    f_rad_fus: float | None = None,
    seeded_impurities: dict[str, float] | None = None,
) -> float:
```

Change to:

```python
    # Genuine optional overrides (keep defaults)
    p_rad_override: float | None = None,
    f_rad_fus: float | None = None,
    seeded_impurities: dict[str, float] | None = None,
    radiation_peaking_factor: float = 1.0,
) -> float:
```

In the same function, the `compute_p_rad` call currently reads:

```python
            p_rad_raw = compute_p_rad(
                n_e,
                T_e,
                Z_eff,
                plasma_volume,
                B,
                impurities,
                R=R_major,
                a=a_minor,
                kappa=kappa,
                R_w=R_w,
            )
```

Change to add the factor:

```python
            p_rad_raw = compute_p_rad(
                n_e,
                T_e,
                Z_eff,
                plasma_volume,
                B,
                impurities,
                R=R_major,
                a=a_minor,
                kappa=kappa,
                R_w=R_w,
                radiation_peaking_factor=radiation_peaking_factor,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_power_balance.py -v`
Expected: PASS (all tests, including the new one)

- [ ] **Step 5: Commit**

```bash
git add src/costingfe/layers/physics.py tests/test_power_balance.py
git commit -m "Thread radiation_peaking_factor through MFE power balance"
```

---

### Task 3: Wire the YAML value into `model.py`'s rad_kw

**Files:**
- Modify: `src/costingfe/model.py:184-191`

- [ ] **Step 1: Add the param to rad_kw**

In `src/costingfe/model.py`, the `rad_kw` dict currently reads:

```python
            rad_kw = dict(
                n_e=_to_num(params["n_e"]),
                T_e=_to_num(params["T_e"]),
                Z_eff=_to_num(params["Z_eff"]),
                plasma_volume=_to_num(params["plasma_volume"]),
                B=_to_num(params["B"]),
                f_rad_fus=params.get("f_rad_fus", self.cc.f_rad_fus(self.fuel)),
            )
```

Change to add the factor (read strictly from YAML, no `.get` fallback, so a missing
entry raises rather than silently defaulting):

```python
            rad_kw = dict(
                n_e=_to_num(params["n_e"]),
                T_e=_to_num(params["T_e"]),
                Z_eff=_to_num(params["Z_eff"]),
                plasma_volume=_to_num(params["plasma_volume"]),
                B=_to_num(params["B"]),
                radiation_peaking_factor=_to_num(params["radiation_peaking_factor"]),
                f_rad_fus=params.get("f_rad_fus", self.cc.f_rad_fus(self.fuel)),
            )
```

This `rad_kw` is spread into both the `mfe_inverse_power_balance` call
(`model.py:203`) and the `mfe_forward_power_balance` call (`model.py:225`), so both
steady-state paths receive the factor. The 0-D tokamak forward path
(`model.py:371`) and `tokamak.py`'s internal `compute_p_rad` are intentionally left
on the 1.0 default (tokamaks use a flat profile).

- [ ] **Step 2: Verify nothing is broken yet (YAMLs not updated, so steady-state models will KeyError)**

Run: `pytest tests/test_model.py -k tokamak -v`
Expected: FAIL with `KeyError: 'radiation_peaking_factor'` (confirms the value is now required from YAML; fixed in Task 4)

- [ ] **Step 3: Commit**

```bash
git add src/costingfe/model.py
git commit -m "Read radiation_peaking_factor from YAML into rad_kw"
```

---

### Task 4: Add the field to all 7 steady-state YAML defaults; fix dipole volume

**Files:**
- Modify: `src/costingfe/data/defaults/steady_state_tokamak.yaml`
- Modify: `src/costingfe/data/defaults/steady_state_stellarator.yaml`
- Modify: `src/costingfe/data/defaults/steady_state_mirror.yaml`
- Modify: `src/costingfe/data/defaults/steady_state_steady_frc.yaml`
- Modify: `src/costingfe/data/defaults/steady_state_orbitron.yaml`
- Modify: `src/costingfe/data/defaults/steady_state_polywell.yaml`
- Modify: `src/costingfe/data/defaults/steady_state_dipole.yaml`

- [ ] **Step 1: Add `radiation_peaking_factor: 1.0` next to the `plasma_volume:` line in each of the six non-dipole files**

For each of `steady_state_tokamak.yaml`, `steady_state_stellarator.yaml`,
`steady_state_mirror.yaml`, `steady_state_steady_frc.yaml`,
`steady_state_orbitron.yaml`, `steady_state_polywell.yaml`: locate the line
`plasma_volume: <value>` and insert immediately after it:

```yaml
radiation_peaking_factor: 1.0   # ~uniform profile, no emission-measure correction
```

- [ ] **Step 2: Edit the dipole file: correct the volume and add the factor**

In `src/costingfe/data/defaults/steady_state_dipole.yaml`, the radiation block currently reads:

```yaml
plasma_volume: 200.0 # Plasma volume [m^3]
B: 2.0              # Characteristic plasma-region field [T] (falls off from coil)
```

Change to:

```yaml
plasma_volume: 13600.0 # Geometric plasma volume [m^3] (Simpson 2026 Reactor A,
                       #   Table 6). No longer a radiation fudge: the peaking
                       #   factor below carries the profile correction.
radiation_peaking_factor: 0.05 # Hasegawa-Mauel profile (n~R^-4, T~R^-8/3); the
                               #   hot/dense radiating core is ~5% of the
                               #   geometric volume (Simpson 2026 sec 2.1.1).
B: 2.0              # Characteristic plasma-region field [T] (falls off from coil)
```

- [ ] **Step 3: Run the full steady-state model suite**

Run: `pytest tests/test_model.py -v`
Expected: PASS (the `KeyError` from Task 3 is resolved; dipole and tokamak both run)

- [ ] **Step 4: Commit**

```bash
git add src/costingfe/data/defaults/steady_state_*.yaml
git commit -m "Add radiation_peaking_factor to steady-state YAMLs; fix dipole volume (issue #24)"
```

---

### Task 5: DIPOLE end-to-end regression test (recirculating fraction sane)

**Files:**
- Modify: `tests/test_model.py:214-224`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_model.py` after `test_dipole_concept_runs`:

```python
def test_dipole_radiation_peaking_keeps_recirc_sane():
    """With the peaking factor, the dipole's bremsstrahlung no longer explodes,
    so the recirculating fraction stays physical (issue #24). Without it, the
    13,600 m^3 geometric volume drives p_rad above the fusion power and pushes
    recirc toward ~86%."""
    model = CostModel(concept=ConfinementConcept.DIPOLE, fuel=Fuel.DT)
    result = model.forward(net_electric_mw=208.0, availability=0.85, lifetime_yr=30)
    assert float(result.power_table.rec_frac) < 0.5
    # p_fus must be the right order of magnitude (hundreds of MW, not thousands).
    assert 300.0 < float(result.power_table.p_fus) < 1500.0
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `pytest tests/test_model.py::test_dipole_radiation_peaking_keeps_recirc_sane -v`
Expected: PASS (the implementation from Tasks 1-4 makes this green immediately;
this is a regression guard, so confirm it passes now and would fail if the factor
were reverted to 1.0)

- [ ] **Step 3: Sanity-check the guard by temporary mutation (optional verification)**

Temporarily set `radiation_peaking_factor: 1.0` in `steady_state_dipole.yaml`, rerun
the test, and confirm it FAILS (rec_frac jumps high). Then restore `0.05`. This
proves the test actually guards the fix.

Run: `pytest tests/test_model.py::test_dipole_radiation_peaking_keeps_recirc_sane -v`
Expected after restore: PASS

- [ ] **Step 4: Run the full suite**

Run: `pytest -q`
Expected: PASS (no regressions; non-dipole concepts unchanged at factor 1.0)

- [ ] **Step 5: Commit**

```bash
git add tests/test_model.py
git commit -m "Test: dipole recirculating fraction stays sane with peaking factor (issue #24)"
```

---

## Self-Review

**Spec coverage:**
- Library factor on brems+line only, sync untouched -> Task 1.
- Threading through forward+inverse balance -> Task 2.
- Wiring from YAML via model.py -> Task 3.
- Field in 7 steady-state YAMLs (1.0; dipole 0.05) + dipole volume 200->13,600 -> Task 4.
- DIPOLE sane recirc / p_fus ballpark test + brems-scaling unit test + sync-invariance unit test -> Tasks 1 and 5.
- Backward compatibility (1.0 = no-op) -> covered by `test_peaking_factor_default_is_one` (Task 1) and full-suite green (Task 5).

**Placeholder scan:** none; every code step shows complete before/after.

**Type/name consistency:** parameter name `radiation_peaking_factor` is identical across `compute_p_rad`, `mfe_forward_power_balance`, `mfe_inverse_power_balance`, `rad_kw`, and all YAML keys.

**Out of scope (unchanged):** pulsed YAMLs (no `plasma_volume`, use f_rad path); the 0-D tokamak path and `tokamak.py` internal `compute_p_rad` (stay at 1.0 default); fusion-tea OpenStar concept spec and validators (PR #37).
