# Unified Pulsed Inverse via Q_eng — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `e_driver_mj` and `q_sci` with `q_eng` as the universal input for all pulsed power balance inverses, making the parameterization consistent across thermal and DEC conversion pathways.

**Architecture:** Both `pulsed_thermal_inverse` and `pulsed_dec_inverse` are rewritten to accept `q_eng` (engineering gain). From `q_eng` and `p_net`, we derive `p_recirc` and `p_et`, then solve for `p_driver` using the pathway-specific recirculating definition. Forward passes are unchanged. YAMLs switch from `e_driver_mj` to `q_eng`.

**Tech Stack:** Python, numpy (numpy-only branch), pytest

**Branches:** Implement on `numpy-only`, then copy `costingfe/` to `fusion-backcasting` lcoe-dashboard branch.

---

### Task 1: Rewrite `pulsed_thermal_inverse` to accept `q_eng`

**Files:**
- Modify: `src/costingfe/layers/physics.py:641-712`
- Test: `tests/test_pulsed_power_balance.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pulsed_power_balance.py`:

```python
def test_thermal_inverse_qeng_roundtrip():
    """Inverse with q_eng should recover the same forward results."""
    pt = pulsed_thermal_forward(**THERMAL_PARAMS)
    # Build inverse params: drop p_fus, add q_eng
    inv_params = {
        k: v
        for k, v in THERMAL_PARAMS.items()
        if k not in ("p_fus", "fuel", "e_driver_mj")
    }
    p_fus_recovered, e_driver_recovered = pulsed_thermal_inverse(
        p_net_target=pt.p_net, fuel=Fuel.DT, q_eng=pt.q_eng, **inv_params
    )
    assert abs(p_fus_recovered - 2500.0) < 0.5, f"Expected ~2500, got {p_fus_recovered}"
    assert abs(e_driver_recovered - 100.0) < 0.5, f"Expected ~100, got {e_driver_recovered}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pulsed_power_balance.py::test_thermal_inverse_qeng_roundtrip -v`
Expected: FAIL (signature mismatch — `q_eng` not accepted, returns float not tuple)

- [ ] **Step 3: Rewrite `pulsed_thermal_inverse`**

Replace the function at `physics.py:641-712` with:

```python
def pulsed_thermal_inverse(
    p_net_target: float,
    fuel: Fuel,
    q_eng: float,
    f_rep: float,
    mn: float,
    eta_th: float,
    eta_pin: float,
    f_rad: float,
    f_sub: float,
    p_pump: float,
    p_trit: float,
    p_house: float,
    p_cryo: float,
    p_target: float,
    p_coils: float = 0.0,
    dd_f_T: float = DD_F_T_DEFAULT,
    dd_f_He3: float = DD_F_HE3_DEFAULT,
    dhe3_dd_frac: float = 0.07,
    dhe3_f_T: float = 0.97,
    pb11_f_alpha_n: float = 0.0,
    pb11_f_p_n: float = 0.0,
) -> tuple[float, float]:
    """Inverse pulsed thermal: target P_net + Q_eng -> required P_fus and e_driver_mj.

    From Q_eng we derive P_et and P_recirc, then solve for P_driver from the
    recirculating definition: P_recirc = P_driver/eta_pin + fixed_loads.
    Then e_driver_mj = P_driver / f_rep, and P_fus from the forward balance.

    Returns (p_fus, e_driver_mj).
    """
    # Step 1: Derive gross and recirculating from q_eng
    p_et = p_net_target * q_eng / (q_eng - 1.0)
    p_recirc = p_et / q_eng

    # Step 2: Fixed loads (independent of p_driver)
    pump_term = jnp.where(eta_th > 0, p_pump, 0.0)
    p_aux = p_trit + p_house
    p_sub = f_sub * p_et
    fixed_loads = pump_term + p_sub + p_aux + p_cryo + p_target + p_coils

    # Step 3: Solve for p_driver
    # p_recirc = p_driver / eta_pin + fixed_loads
    p_driver = (p_recirc - fixed_loads) * eta_pin

    # Step 4: Derive e_driver_mj
    e_driver_mj = p_driver / f_rep

    # Step 5: Derive p_fus from the thermal power balance
    # p_th = (mn * neutron_frac + ash_frac) * p_fus + p_driver + pump_term
    # p_et = eta_th * p_th
    # => p_fus = (p_et / eta_th - p_driver - pump_term) / (mn * neutron_frac + ash_frac)
    ash_frac, _ = ash_neutron_split(
        1.0, fuel,
        dd_f_T, dd_f_He3, dhe3_dd_frac, dhe3_f_T,
        pb11_f_alpha_n, pb11_f_p_n,
    )
    neutron_frac = 1.0 - ash_frac
    c_th = mn * neutron_frac + ash_frac
    p_fus = (p_et / eta_th - p_driver - pump_term) / c_th

    return p_fus, e_driver_mj
```

- [ ] **Step 4: Update old roundtrip test**

Replace `test_thermal_inverse_roundtrip` in the test file:

```python
def test_thermal_inverse_roundtrip():
    pt = pulsed_thermal_forward(**THERMAL_PARAMS)
    inv_params = {
        k: v
        for k, v in THERMAL_PARAMS.items()
        if k not in ("p_fus", "fuel", "e_driver_mj")
    }
    p_fus_recovered, e_driver_recovered = pulsed_thermal_inverse(
        p_net_target=pt.p_net, fuel=Fuel.DT, q_eng=pt.q_eng, **inv_params
    )
    assert abs(p_fus_recovered - 2500.0) < 0.5
    assert abs(e_driver_recovered - 100.0) < 0.5
```

(This replaces the old test; the new `test_thermal_inverse_qeng_roundtrip` can be removed as it's now redundant.)

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_pulsed_power_balance.py -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/costingfe/layers/physics.py tests/test_pulsed_power_balance.py
git commit -m "feat: rewrite pulsed_thermal_inverse to accept q_eng"
```

---

### Task 2: Rewrite `pulsed_dec_inverse` to accept `q_eng`

**Files:**
- Modify: `src/costingfe/layers/physics.py:868-960`
- Test: `tests/test_pulsed_power_balance.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pulsed_power_balance.py`:

```python
def test_dec_inverse_qeng_roundtrip():
    """DEC inverse with q_eng should recover same forward results."""
    pt = pulsed_dec_forward(**DEC_PARAMS)
    inv_params = {
        k: v
        for k, v in DEC_PARAMS.items()
        if k not in ("p_fus", "fuel", "e_driver_mj")
    }
    p_fus_recovered, e_driver_recovered = pulsed_dec_inverse(
        p_net_target=pt.p_net, fuel=Fuel.DHE3, q_eng=pt.q_eng, **inv_params
    )
    assert abs(p_fus_recovered - 500.0) < 0.5, f"Expected ~500, got {p_fus_recovered}"
    assert abs(e_driver_recovered - 12.0) < 0.5, f"Expected ~12, got {e_driver_recovered}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pulsed_power_balance.py::test_dec_inverse_qeng_roundtrip -v`
Expected: FAIL (signature still uses `q_sci`)

- [ ] **Step 3: Rewrite `pulsed_dec_inverse`**

Replace the function at `physics.py:868-960` with:

```python
def pulsed_dec_inverse(
    p_net_target: float,
    fuel: Fuel,
    q_eng: float,
    f_rep: float,
    mn: float,
    eta_th: float,
    eta_pin: float,
    eta_dec: float,
    f_pdv: float,
    f_rad: float,
    f_sub: float,
    p_pump: float,
    p_trit: float,
    p_house: float,
    p_cryo: float,
    p_target: float,
    p_coils: float = 0.0,
    dd_f_T: float = DD_F_T_DEFAULT,
    dd_f_He3: float = DD_F_HE3_DEFAULT,
    dhe3_dd_frac: float = 0.07,
    dhe3_f_T: float = 0.97,
    pb11_f_alpha_n: float = 0.0,
    pb11_f_p_n: float = 0.0,
) -> tuple[float, float]:
    """Inverse pulsed DEC: target P_net + Q_eng -> required P_fus and e_driver_mj.

    From Q_eng we derive P_et and P_recirc, then solve for P_driver from the
    DEC recirculating definition: P_recirc = P_driver*(1/eta_pin - 1) + fixed_loads.
    Then e_driver_mj = P_driver / f_rep, and P_fus from the forward balance.

    Returns (p_fus, e_driver_mj).
    """
    # Step 1: Derive gross and recirculating from q_eng
    p_et = p_net_target * q_eng / (q_eng - 1.0)
    p_recirc = p_et / q_eng

    # Step 2: Fixed loads (independent of p_driver)
    pump_term = jnp.where(eta_th > 0, p_pump, 0.0)
    p_aux = p_trit + p_house
    p_sub = f_sub * p_et
    fixed_loads = pump_term + p_sub + p_aux + p_cryo + p_target + p_coils

    # Step 3: Solve for p_driver
    # DEC: p_recirc = p_driver * (1/eta_pin - 1) + fixed_loads
    p_driver = (p_recirc - fixed_loads) / (1.0 / eta_pin - 1.0)

    # Step 4: Derive e_driver_mj
    e_driver_mj = p_driver / f_rep

    # Step 5: Derive p_fus from the DEC energy balance
    # Need to solve for p_fus given p_driver, p_et, and the DEC physics.
    # p_et = p_dee + p_the
    # Work backwards from known p_driver and p_et.
    fuel_frac_kw = dict(
        dd_f_T=dd_f_T, dd_f_He3=dd_f_He3, dhe3_dd_frac=dhe3_dd_frac,
        dhe3_f_T=dhe3_f_T, pb11_f_alpha_n=pb11_f_alpha_n, pb11_f_p_n=pb11_f_p_n,
    )
    f_ch = _charged_particle_fraction(fuel, **fuel_frac_kw)
    neutron_frac = 1.0 - f_ch

    # DEC electric: p_dee = eta_dec * (p_driver + p_pdv) - p_driver
    # where p_pdv = f_pdv * p_charged_net = f_pdv * f_ch * (1 - f_rad) * p_fus
    # Thermal electric: p_the = eta_th * p_th
    # p_th = mn * neutron_frac * p_fus + f_ch * f_rad * p_fus
    #      + f_ch * (1-f_rad) * (1-f_pdv) * p_fus  (undirected charged)
    #      + (1-eta_dec) * (p_driver + p_pdv)        (DEC waste)
    #      + p_driver                                 (driver thermalises)
    #      + pump_term
    # Wait — driver does NOT thermalise in DEC. It circulates electromagnetically.
    # Let me use the per-P_fus approach instead.
    #
    # Express all terms per unit p_fus:
    # p_charged_net_per = f_ch * (1 - f_rad)
    # p_pdv_per = f_pdv * p_charged_net_per
    # p_dee_per = eta_dec * (p_driver/p_fus + p_pdv_per) - p_driver/p_fus
    #
    # Actually simpler: we know p_driver and need p_fus.
    # p_dee = eta_dec * (p_driver + f_pdv * f_ch * (1-f_rad) * p_fus) - p_driver
    # p_th = mn * neutron_frac * p_fus + f_ch * f_rad * p_fus
    #      + f_ch * (1-f_rad) * (1-f_pdv) * p_fus
    #      + (1-eta_dec) * (p_driver + f_pdv * f_ch * (1-f_rad) * p_fus)
    #      + pump_term
    # p_the = eta_th * p_th
    # p_et = p_dee + p_the  (we know this value)
    #
    # Collect p_fus terms:
    q_cn = f_ch * (1.0 - f_rad)
    q_pdv = f_pdv * q_cn

    # p_dee = eta_dec * p_driver + eta_dec * q_pdv * p_fus - p_driver
    #       = p_driver * (eta_dec - 1) + eta_dec * q_pdv * p_fus
    c_dee = eta_dec * q_pdv  # coefficient of p_fus in p_dee
    k_dee = p_driver * (eta_dec - 1.0)  # constant in p_dee

    # p_th terms with p_fus coefficient:
    c_th_fus = (
        mn * neutron_frac
        + f_ch * f_rad
        + q_cn * (1.0 - f_pdv)
        + (1.0 - eta_dec) * q_pdv
    )
    # p_th constant terms:
    k_th = (1.0 - eta_dec) * p_driver + pump_term

    c_the = eta_th * c_th_fus  # coefficient of p_fus in p_the
    k_the = eta_th * k_th  # constant in p_the

    # p_et = (c_dee + c_the) * p_fus + (k_dee + k_the)
    c_et_total = c_dee + c_the
    k_et_total = k_dee + k_the

    p_fus = (p_et - k_et_total) / c_et_total

    return p_fus, e_driver_mj
```

- [ ] **Step 4: Update old roundtrip test**

Replace `test_dec_inverse_roundtrip` in the test file:

```python
def test_dec_inverse_roundtrip():
    pt = pulsed_dec_forward(**DEC_PARAMS)
    inv_params = {
        k: v
        for k, v in DEC_PARAMS.items()
        if k not in ("p_fus", "fuel", "e_driver_mj")
    }
    p_fus_recovered, e_driver_recovered = pulsed_dec_inverse(
        p_net_target=pt.p_net, fuel=Fuel.DHE3, q_eng=pt.q_eng, **inv_params
    )
    assert abs(p_fus_recovered - 500.0) < 0.5, f"Expected ~500, got {p_fus_recovered}"
    assert abs(e_driver_recovered - 12.0) < 0.5, f"Expected ~12, got {e_driver_recovered}"
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_pulsed_power_balance.py -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/costingfe/layers/physics.py tests/test_pulsed_power_balance.py
git commit -m "feat: rewrite pulsed_dec_inverse to accept q_eng"
```

---

### Task 3: Update `model.py` call site

**Files:**
- Modify: `src/costingfe/model.py:191-248`

- [ ] **Step 1: Update the pulsed branch in `_power_balance`**

Replace lines 191-248 of `model.py` (the `elif self.family == ConfinementFamily.PULSED:` block):

```python
        elif self.family == ConfinementFamily.PULSED:
            fuel_frac_kw = dict(
                dd_f_T=params["dd_f_T"],
                dd_f_He3=params["dd_f_He3"],
                dhe3_dd_frac=params["dhe3_dd_frac"],
                dhe3_f_T=params["dhe3_f_T"],
                pb11_f_alpha_n=params["pb11_f_alpha_n"],
                pb11_f_p_n=params["pb11_f_p_n"],
            )
            q_eng = params["q_eng"]
            common_kw = dict(
                fuel=self.fuel,
                f_rep=params["f_rep"],
                mn=params["mn"],
                eta_th=params["eta_th"],
                eta_pin=params["eta_pin"],
                f_rad=params.get("f_rad", self.cc.f_rad(self.fuel)),
                f_sub=params["f_sub"],
                p_pump=params["p_pump"],
                p_trit=params["p_trit"],
                p_house=params["p_house"],
                p_cryo=params["p_cryo"],
                p_target=params.get("p_target", 0.0),
                p_coils=params.get("p_coils", 0.0),
                **fuel_frac_kw,
            )

            if self.pulsed_conversion == PulsedConversion.INDUCTIVE_DEC:
                dec_kw = dict(
                    eta_dec=params["eta_dec"],
                    f_pdv=params.get("f_pdv", self.cc.f_pdv),
                )
                p_fus, e_driver_solved = pulsed_dec_inverse(
                    p_net_target=p_net_per_mod,
                    q_eng=q_eng,
                    **common_kw,
                    **dec_kw,
                )
                common_kw["e_driver_mj"] = e_driver_solved
                pt = pulsed_dec_forward(
                    p_fus=p_fus,
                    **common_kw,
                    **dec_kw,
                )
            else:
                p_fus, e_driver_solved = pulsed_thermal_inverse(
                    p_net_target=p_net_per_mod,
                    q_eng=q_eng,
                    **common_kw,
                )
                common_kw["e_driver_mj"] = e_driver_solved
                pt = pulsed_thermal_forward(
                    p_fus=p_fus,
                    **common_kw,
                )
```

Key changes:
- `e_driver_mj` removed from `common_kw` (it's now derived by both inverses)
- `q_eng` read from params and passed to both inverses
- Both branches now set `common_kw["e_driver_mj"] = e_driver_solved` before the forward pass

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: Some failures in integration tests that pass `e_driver_mj` directly, and validation tests. We'll fix those in the next tasks.

- [ ] **Step 3: Commit**

```bash
git add src/costingfe/model.py
git commit -m "feat: model.py pulsed branch uses q_eng for both inverses"
```

---

### Task 4: Update all pulsed YAML defaults

**Files:**
- Modify: All 10 files in `src/costingfe/data/defaults/pulsed_*.yaml`

- [ ] **Step 1: Replace `e_driver_mj` and `q_sci` with `q_eng` in each file**

For each file, remove the `e_driver_mj` and `q_sci` lines and add `q_eng`. The `q_eng` default should be concept-appropriate:

**pulsed_laser_ife.yaml:** `q_eng: 4.0`
**pulsed_heavy_ion.yaml:** `q_eng: 4.0`
**pulsed_zpinch.yaml:** `q_eng: 4.0`
**pulsed_mag_target.yaml:** `q_eng: 3.0`
**pulsed_plasma_jet.yaml:** `q_eng: 3.0`
**pulsed_maglif.yaml:** `q_eng: 3.0`
**pulsed_pulsed_frc.yaml:** `q_eng: 3.0`
**pulsed_theta_pinch.yaml:** `q_eng: 3.0`
**pulsed_dense_plasma_focus.yaml:** `q_eng: 3.0`
**pulsed_staged_zpinch.yaml:** `q_eng: 3.0`

Example for `pulsed_mag_target.yaml` — replace:
```yaml
e_driver_mj: 50.0       # Pulsed power per pulse [MJ]
...
q_sci: 5.0               # Scientific gain target (used for inductive DEC inverse)
```
with:
```yaml
q_eng: 3.0               # Engineering gain target (P_et / P_recirc)
```

Keep `f_rep` unchanged in all files.

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: Closer to passing — model can now find `q_eng` in params. Some tests may still fail if they construct params manually with `e_driver_mj`.

- [ ] **Step 3: Commit**

```bash
git add src/costingfe/data/defaults/pulsed_*.yaml
git commit -m "feat: replace e_driver_mj/q_sci with q_eng in all pulsed YAMLs"
```

---

### Task 5: Update validation and remaining tests

**Files:**
- Modify: `src/costingfe/validation.py`
- Modify: `tests/test_validation.py`
- Modify: `tests/test_pulsed_dec_costing.py`
- Modify: `tests/test_types.py` (if needed)

- [ ] **Step 1: Update validation.py**

In `src/costingfe/validation.py`, replace `e_driver_mj` with `q_eng` in:
- The `CostingInput` model field: change `e_driver_mj: float | None = None` to `q_eng: float | None = None`
- The pulsed required fields list (around line 147): change `"e_driver_mj"` to `"q_eng"`
- Any references in validation logic that check `e_driver_mj`

- [ ] **Step 2: Update test_validation.py**

Update `test_pulsed_missing_e_driver_mj_rejected` to test for `q_eng` instead:

```python
def test_pulsed_missing_q_eng_rejected(self):
    """Pulsed concept missing q_eng should be rejected."""
    with pytest.raises(ValidationError, match="q_eng"):
        # ... test body with q_eng removed from params
```

Update `test_q_sci_warning_when_low` — remove or adapt if `q_sci` is no longer a user input.

- [ ] **Step 3: Update test_pulsed_dec_costing.py**

Replace `q_sci=5.0` with `q_eng=3.0` (or appropriate value) in all test calls that pass pulsed params. Replace any `e_driver_mj` overrides with `q_eng` overrides.

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: All 280+ tests pass

- [ ] **Step 5: Commit**

```bash
git add src/costingfe/validation.py tests/test_validation.py tests/test_pulsed_dec_costing.py
git commit -m "feat: update validation and tests for q_eng parameterization"
```

---

### Task 6: Update `paper.tex`

**Files:**
- Modify: `tex/paper.tex`

- [ ] **Step 1: Update pulsed power balance section**

Find the pulsed power balance description (around lines 899-960) and update:

- Replace the description of `E_{\text{drv}}` as a user input with a description of `Q_{\text{eng}}` as the primary input
- Add the inversion formula: `P_{\text{et}} = P_{\text{net}} \cdot Q_{\text{eng}} / (Q_{\text{eng}} - 1)`
- Note that `E_{\text{drv}}` and `Q_{\text{sci}}` are now derived quantities
- Update the C220104 driver section (around line 1305) if it references `E_{\text{drv}}` as a user-specified input

- [ ] **Step 2: Verify paper compiles**

Run: `cd tex && pdflatex -interaction=nonstopmode paper.tex`
Expected: Compiles without errors

- [ ] **Step 3: Commit**

```bash
git add tex/paper.tex
git commit -m "docs: update paper pulsed section for q_eng parameterization"
```

---

### Task 7: Push numpy-only, update fusion-backcasting

**Files:**
- Push: `numpy-only` branch
- Modify: `fusion-backcasting/costingfe/` (copy from numpy-only)
- Modify: `fusion-backcasting/frontend/src/components/EngineeringPanel.tsx`

- [ ] **Step 1: Push numpy-only**

```bash
git push
```

- [ ] **Step 2: Copy updated costingfe to fusion-backcasting**

```bash
cp -r /mnt/c/Users/talru/1cfe/1costingfe/src/costingfe/*.py /mnt/c/Users/talru/1cfe/fusion-backcasting/costingfe/
cp -r /mnt/c/Users/talru/1cfe/1costingfe/src/costingfe/layers/*.py /mnt/c/Users/talru/1cfe/fusion-backcasting/costingfe/layers/
cp -r /mnt/c/Users/talru/1cfe/1costingfe/src/costingfe/data/defaults/*.yaml /mnt/c/Users/talru/1cfe/fusion-backcasting/costingfe/data/defaults/
```

Note: fusion-backcasting uses `import numpy as jnp` (same as numpy-only), so files copy directly.

- [ ] **Step 3: Update EngineeringPanel.tsx**

In `fusion-backcasting/frontend/src/components/EngineeringPanel.tsx`, replace the `e_driver_mj` entry in `PARASITIC_PULSED` (line 38):

```typescript
const PARASITIC_PULSED: ParamDef[] = [
  { key: 'q_eng', label: 'Engineering Q', min: 1.5, max: 10, step: 0.1, format: (v) => `${v.toFixed(1)}` },
  { key: 'f_rep', label: 'Rep Rate', min: 0.1, max: 20, step: 0.1, format: (v) => `${v.toFixed(1)} Hz` },
  { key: 'eta_pin', label: 'Driver Efficiency', min: 0.05, max: 0.98, step: 0.01, format: formatPct },
  { key: 'f_rad', label: 'Radiation Fraction', min: 0.01, max: 0.30, step: 0.01, format: formatPct },
  { key: 'p_target', label: 'Target Factory', min: 0, max: 10, step: 0.1, format: formatMW },
  { key: 'p_coils', label: 'Coil Power', min: 0, max: 10, step: 0.5, format: formatMW },
];
```

- [ ] **Step 4: Run fusion-backcasting tests**

```bash
cd /mnt/c/Users/talru/1cfe/fusion-backcasting && python -m pytest tests/ -x -q
```

- [ ] **Step 5: Commit and push fusion-backcasting**

```bash
cd /mnt/c/Users/talru/1cfe/fusion-backcasting
git add costingfe/ frontend/src/components/EngineeringPanel.tsx
git commit -m "feat: unified pulsed inverse via q_eng, update dashboard slider"
git push
```

---

### Task 8: Cherry-pick to master (1costingfe)

**Files:**
- Cherry-pick physics.py, model.py, YAMLs, validation, tests, paper changes to master

- [ ] **Step 1: Switch to master and cherry-pick**

```bash
cd /mnt/c/Users/talru/1cfe/1costingfe
git checkout master
git cherry-pick <commit-range-from-numpy-only>
```

Resolve any conflicts (likely just `import jax.numpy as jnp` vs `import numpy as jnp`).

- [ ] **Step 2: Run full test suite on master**

```bash
pytest tests/ -x -q
```

- [ ] **Step 3: Push master**

```bash
git push
```
