# Mirror Wall-Loading, Radial-Build Coil Bores, and Fluence Lifetime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mirror coil bores derived from the radial build (large-bore central solenoids, small-bore throat plugs), neutron and surface-flux wall constraints in the mirror sizing solve, and fluence-based core lifetime in CAS72 for all MFE concepts.

**Architecture:** Three independent parts per the spec (READ IT FIRST: `docs/superpowers/specs/2026-06-12-mirror-wall-loading-and-radial-build.md` — all four decisions are settled there). Part 1 is costing-only (cas22 + model plumbing + one markup recalibration, calibration-neutral by construction). Part 2 extends the mirror sizing density resolution to `min(n_beta, n_wall, n_surf)` with both caps YAML-sourced. Part 3 replaces the per-fuel `core_lifetime` constants with per-fuel fluence limits for the steady-state MFE family; this DELIBERATELY moves default results (decision d1) and updates the LCOE pin in one documented commit.

**Tech Stack:** Python, JAX (float32, no x64), pytest via `uv run pytest`, ruff (pre-commit hook). Branch: create `feat/mirror-wall-fluence` off master.

**Invariants for every task:**
- Full suite green; ruff clean. Baseline at branch time: 554 passed (run once to confirm before Task 1).
- Densities in kernels follow the 1e20-unit convention (see `_TAU_II_PREFACTOR_20` in mirror.py and the reactivity.py hazard docstring).
- No Python keyword defaults for physics params (YAML only; None sentinels and behavior flags excepted).
- Commits: single line, no body, NO Co-Authored-By.
- pytest on this filesystem is slow (WSL /mnt/c): single file 30s-4min, full suite ~15 min. Run the full suite ONCE per task right before committing. Never kill a run.
- The coil calibration pin (C220103 = 513.375 M$ at mirror YAML defaults) holds through EVERY task. The LCOE pin (93.643616) holds through Tasks 1-5 and is updated ONCE in Task 6 with the documented basis change.

---

### Task 0: Branch and baseline

- [ ] Step 0.1: `git checkout -b feat/mirror-wall-fluence master`
- [ ] Step 0.2: `uv run pytest -q` — confirm 554 passed (553 + the mirror_power_sizing example added none; if the count differs, record the actual number as the baseline and proceed).
- [ ] Step 0.3: No commit (branch only).

---

### Task 1: Coil bores from the radial build (Part 1)

**Files:**
- Modify: `src/costingfe/layers/cas22.py` (signature ~line 180-184, mirror two-class branch ~line 320-360)
- Modify: `src/costingfe/model.py` (cas22 call site ~line 1219-1223)
- Modify: `src/costingfe/data/defaults/steady_state_mirror.yaml` (new keys; r_bore comment)
- Modify: `src/costingfe/data/defaults/costing_constants.yaml` (mirror coil markup, one value)
- Modify: `docs/account_justification/CAS22_reactor_components.md`
- Test: `tests/test_mirror.py` (TestMirrorCoilLengthScaling)

READ FIRST: the spec's Part 1; the current two-class branch in cas22.py; the call site in model.py (note `r_coil=geo.vessel_or` is ALREADY passed — the central bore builds on it).

- [ ] Step 1.1: Write the failing tests in `TestMirrorCoilLengthScaling` (tests/test_mirror.py). Add:

```python
def test_central_bore_from_radial_build(self):
    # r_bore_central = vessel_or + coil_standoff. At YAML defaults the
    # build stacks to vessel_or = 3.20 m (1.5 plasma + 0.10 vacuum +
    # 0.05 FW + 0.80 blanket + 0.20 reflector + 0.20 HT shield +
    # 0.15 structure + 0.10 gap1 + 0.10 vessel) and coil_standoff = 0.10,
    # so r_bore_central = 3.30 m. Probe via the model's stored detail.
    m = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
    r = m.forward(net_electric_mw=500.0, availability=0.87, lifetime_yr=40.0)
    assert r.cas22_detail["r_bore_central"] == pytest.approx(3.30, abs=1e-6)
    assert r.cas22_detail["r_bore_plug"] == pytest.approx(
        1.5 / math.sqrt(10.0) + 0.30, abs=1e-6
    )

def test_coil_cost_responds_to_blanket_t(self):
    # Thicker blanket -> larger central bore -> costlier coils.
    m = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
    lo = m.forward(net_electric_mw=500.0, availability=0.87, lifetime_yr=40.0)
    hi = m.forward(
        net_electric_mw=500.0, availability=0.87, lifetime_yr=40.0,
        blanket_t=1.2,
    )
    assert hi.cas22_detail["C220103"] > lo.cas22_detail["C220103"]

def test_plug_central_split_pinned(self):
    # At YAML defaults the conductor split is central 261.4 / plug 36.0 M$
    # class (exact values pinned after the implementation derives them).
    m = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
    r = m.forward(net_electric_mw=500.0, availability=0.87, lifetime_yr=40.0)
    split = r.cas22_detail["C220103_central"] / r.cas22_detail["C220103_plug"]
    assert split == pytest.approx(7.267, rel=1e-2)  # re-pin exact at impl time
```

Also KEEP (unchanged, they must still pass): `test_calibration_neutrality_pin` (513.375 to 1e-6), `test_lcoe_pin_unchanged` (93.643616), `test_doubling_length_doubles_central_contribution`, `test_jax_grad_chamber_length_finite_positive`.

NOTE: if `cas22_detail` does not currently carry `r_bore_central` / `C220103_central` keys, add them in Step 1.3 (diagnostic detail entries are the established pattern for pinning sub-account structure — grep `cas22_detail\[` in tests for examples). If detail-dict entries are NOT the established pattern, pin via direct `cas22_reactor_plant_equipment` calls instead, mirroring how existing coil tests in tests/test_cas22_n_coils.py construct calls.

- [ ] Step 1.2: Run `uv run pytest tests/test_mirror.py::TestMirrorCoilLengthScaling -q` — new tests FAIL (missing keys/params), old ones pass.
- [ ] Step 1.3: Implement.

cas22.py — replace the two mirror bore uses. Signature: add after `B: float = 0.0`:

```python
    r_bore_central: float = 0.0,  # Central-cell solenoid bore [m] (vessel_or + coil_standoff)
    r_bore_plug: float = 0.0,  # End-plug coil bore [m] (a/sqrt(R_m) + plug_standoff)
```

Two-class branch guard gains `and r_bore_central > 0 and r_bore_plug > 0`. Replace the kAm lines:

```python
            kAm_central = G_central * b_center * r_bore_central**2 / (_MU0 * 1000)
            kAm_plug = G_plug * b_plug * r_bore_plug**2 / (_MU0 * 1000)
```

Update the branch's comment block: central bore from the radial build (vessel outer radius plus assembly standoff, large-bore/low-field); plug bore from the flux-conservation throat radius plus standoff (small-bore/high-field, no blanket at the throat); r_bore no longer read by the mirror branch. Present-state wording only.

model.py call site — replace the flat r_bore plumbing for the mirror params block:

```python
            r_bore_central=(
                geo.vessel_or + params["coil_standoff"]
                if self.concept == ConfinementConcept.MIRROR
                else 0.0
            ),
            r_bore_plug=(
                params["plasma_t"] / math.sqrt(params["R_m"])
                + params["plug_standoff"]
                if self.concept == ConfinementConcept.MIRROR
                else 0.0
            ),
```

(`r_bore=r_bore` stays — tokamak/stellarator still use it.)

steady_state_mirror.yaml — add near the coil keys:

```yaml
coil_standoff: 0.10  # Assembly gap, vessel outer surface to central solenoid winding pack [m]
plug_standoff: 0.30  # Throat plasma to plug-coil winding pack [m]: vacuum gap + throat structure + cryostat (no blanket at the throat)
```

and reword the `r_bore: 1.85` comment to state it is no longer read by the mirror coil branch (kept for cross-concept schema compatibility), or delete the key if nothing else reads it for the mirror (grep first: `grep -rn "r_bore" src/costingfe/ | grep -v test`).

- [ ] Step 1.4: Recalibrate the markup analytically. At YAML defaults: n_central = 4, G = 4*4*pi each class; kAm_central = G * 12 * 3.30^2 / (mu0*1000) = 5.2279e6; kAm_plug = G * 30 * (1.5/sqrt(10) + 0.30)^2 / (mu0*1000) = 7.195e5; conductor = (sum) * 50 / 1e6 = 297.3 M$ class. markup = 513.375 / conductor_exact. Compute the exact value with a scratch expression (do NOT hand-round; carry full float64) and write it into costing_constants.yaml replacing 1.7857142857142858, with a comment pointing to the account doc. Then re-pin `test_plug_central_split_pinned` with the exact split.
- [ ] Step 1.5: Update `docs/account_justification/CAS22_reactor_components.md`: replace the two-class calibration algebra block with the new bore derivations (radial-build stack to 3.20 m + 0.10 standoff; throat 1.5/sqrt(10) + 0.30 = 0.774 m), the new markup derivation, and the note that the costing now responds to blanket thickness. House style; no pyFECONS values; bare $.
- [ ] Step 1.6: `uv run pytest tests/test_mirror.py -q` (all green, including both pins) then full suite, then `uv run ruff check`.
- [ ] Step 1.7: Commit: `Derive mirror coil bores from radial build and throat flux conservation`

---

### Task 2: First-wall area basis (a + vacuum_t)

**Files:**
- Modify: `src/costingfe/layers/mirror.py` (`mirror_0d_forward` signature + geometry step; `_net_at_L_T` fw_area; `mirror_0d_inverse` passthrough)
- Modify: `src/costingfe/model.py` (`_power_balance_mirror_0d` passes vacuum_t)
- Modify: `src/costingfe/data/defaults/steady_state_mirror.yaml` (explicit `vacuum_t: 0.10`)
- Test: `tests/test_mirror.py`

- [ ] Step 2.1: Failing test first:

```python
def test_fw_area_matches_geometry_layer(self):
    # Cross-layer consistency: the 0D state's first-wall area must equal
    # the geometry layer's (2 pi (a + vacuum_t) L), closing the audit
    # finding that the 0D diagnostic sat on the bare plasma surface.
    from costingfe.layers.geometry import RadialBuild, compute_geometry
    rb = RadialBuild(plasma_t=1.5, chamber_length=20.0, vacuum_t=0.10)
    geo = compute_geometry(rb, ConfinementConcept.MIRROR)
    ps = _forward(L=20.0, a=1.5, vacuum_t=0.10)
    assert float(ps.fw_area) == pytest.approx(geo.firstwall_area, rel=1e-6)
```

- [ ] Step 2.2: Run — fails (`mirror_0d_forward` has no vacuum_t param).
- [ ] Step 2.3: Implement: `mirror_0d_forward` gains a required keyword `vacuum_t: float` (no Python default — YAML carries it). Geometry step becomes:

```python
    V_plasma = jnp.pi * a**2 * L
    fw_area = 2.0 * jnp.pi * (a + vacuum_t) * L
```

`wall_loading` and the `collisionality`/state assembly are unchanged otherwise. `mirror_0d_inverse` gains `vacuum_t` and passes it through (both to the forward call and anywhere it recomputes fw_area — grep `fw_area` in mirror.py; `_net_at_L_T` line ~689 has its own `fw_area = 2.0 * math.pi * a * L` for the power balance: update to `(a + vacuum_t)`). model.py `_power_balance_mirror_0d` passes `vacuum_t=params["vacuum_t"]`. `_net_at_L_T` reads `params["vacuum_t"]`; `_size_mirror`'s solve_params must include it (check how solve_params is built; it copies params, so adding the YAML key suffices). YAML: add `vacuum_t: 0.10  # Plasma-to-first-wall vacuum gap [m] (was implicit via the RadialBuild default)` — same value as the RadialBuild default, so geometry results are bit-identical.
- [ ] Step 2.4: Existing 0D tests that assert fw_area/wall_loading/p_end bookkeeping change by the 1.6/1.5 factor — update the affected expected values DELIBERATELY (each with a one-line comment: first-wall basis moved to a + vacuum_t). The two load-bearing pins (LCOE 93.643616 non-0D; coil 513.375) must NOT move (non-0D path untouched). The GDT/WHAM anchor tests use the kernels directly (not fw_area) — verify they pass unchanged.
- [ ] Step 2.5: `uv run pytest tests/test_mirror.py -q`, full suite, ruff.
- [ ] Step 2.6: Commit: `Move mirror first-wall area basis to a + vacuum_t`

---

### Task 3: Neutron wall-load cap in sizing (Part 2)

**Files:**
- Modify: `src/costingfe/layers/mirror.py` (`_WALL_LOADING_MAX` removal; `_density_from_f_beta` callers; new `_density_from_wall_cap`; `net_electric_at_L`; `mirror_size_from_power` error message; gate warning threshold)
- Modify: `src/costingfe/data/defaults/steady_state_mirror.yaml` (`q_wall_max: 5.0`)
- Modify: `src/costingfe/model.py` (plumb q_wall_max into the inverse + solve params if not already in params flow)
- Test: `tests/test_mirror.py` (new TestWallConstraint class)

- [ ] Step 3.1: Failing tests first:

```python
class TestWallConstraint:
    def test_sizing_respects_q_wall_max(self):
        # Default machine sized at q_wall_max=5: solution is wall-bound,
        # q_n at the cap and beta below the f_beta boundary.
        params = dict(_SIZING_PARAMS, q_wall_max=5.0)
        L, pnet, state = _size(params)  # use/extend the existing sizing test helper
        assert float(state.wall_loading) <= 5.0 * 1.001
        assert float(state.beta) < 0.85 * 0.5  # below f_beta * beta_max

    def test_loose_cap_recovers_beta_bound_solution(self):
        # q_wall_max=50 must reproduce the pre-cap (beta-bound) solution.
        tight = _size(dict(_SIZING_PARAMS, q_wall_max=50.0))
        legacy = _size_expected_beta_bound()  # pin from a pre-change run
        assert tight[0] == pytest.approx(legacy[0], rel=1e-4)

    def test_infeasible_under_cap_raises_naming_cap(self):
        with pytest.raises(SizingInfeasible, match=r"q_wall_max"):
            _size(dict(_SIZING_PARAMS, q_wall_max=0.5, p_net_target=600.0))

    @pytest.mark.parametrize("fuel", [Fuel.DT, Fuel.DD, Fuel.DHE3, Fuel.PB11])
    def test_n_wall_jit_matches_eager_all_fuels(self, fuel):
        # Same harness as test_density_from_f_beta_jit_matches_eager_all_fuels:
        # copy its per-fuel mix kwargs verbatim.
        def f(T):
            return _density_from_wall_cap(
                T, 20.0, 5.0, 1.5, 0.10, fuel, _MIX_KWARGS_FOR[fuel]
            )

        eager = float(f(20.0))
        jitted = float(jax.jit(f)(20.0))
        assert jnp.isfinite(jitted)
        assert jitted == pytest.approx(eager, rel=1e-4)
        g = float(jax.grad(f)(20.0))
        assert jnp.isfinite(g)
```

(Write the helper `_size` against `mirror_size_from_power` exactly the way TestMirrorSizing's tests call it — read them first and reuse their `_SIZING_PARAMS`-style dict; capture the legacy beta-bound pin by running the CURRENT tip once before implementing.)

- [ ] Step 3.2: Run — fails (no q_wall_max key/branch).
- [ ] Step 3.3: Implement `_density_from_wall_cap` in mirror.py beside `_density_from_f_beta`:

```python
def _density_from_wall_cap(T_i, T_e, q_wall_max, a, vacuum_t, fuel, params_mix):
    """Density at the neutron wall-load cap [m^-3].

    q_n = f_n * C_fus(T) * n_e^2 * V / A_fw with V/A_fw = a^2 / (2 (a + vacuum_t))
    (cylinder, L cancels), so

        n_e = sqrt( 2 * q_wall_max * (a + vacuum_t) / (f_n * C_fus(T) * a^2) )

    C_fus(T) is the per-n_e^2 fusion power density at T (reactivity kernel at
    n_e = 1e20, rescaled), f_n the neutron fraction from ash_neutron_split at
    the same fuel mix. Work in n20 units: compute p_density_20 = fusion power
    density at n20 = 1 and divide q-side by it, so no 1e40-class intermediates
    appear (densities-squared hazard, see reactivity.py docstring).
    """
```

Implementation note (binding): evaluate `fusion_power` at the reference density 1e20 with unit volume to get `C_fus_20` (power per m^3 at n20 = 1), then `n20 = jnp.sqrt(2.0 * q_wall_max * (a + vacuum_t) / (f_n * C_fus_20 * a**2))` and return `n20 * 1e20`. The neutron fraction comes from `ash_neutron_split` with the same effective D-He3 fraction logic as the forward (pin-aware). Then in `net_electric_at_L`'s per-T evaluation (`_net_at_L_T`), the density line becomes:

```python
    n_beta = _density_from_f_beta(...)
    n_wall = _density_from_wall_cap(...)
    n_e = jnp.minimum(n_beta, n_wall)
```

`mirror_size_from_power`'s SizingInfeasible message gains the cap when the wall branch was binding at L_max: include `q_wall_max` and the achievable p_net in the message text (decision b). Replace `_WALL_LOADING_MAX` everywhere with the YAML-sourced `q_wall_max` (the audit-mode warning in `mirror_0d_inverse` reads it as a required kwarg; model.py passes `params["q_wall_max"]`). YAML: `q_wall_max: 5.0  # Neutron wall-load cap [MW/m^2]; sizing constraint, audit-mode warning threshold`.
- [ ] Step 3.4: `uv run pytest tests/test_mirror.py -q`, full suite, ruff. The mirror_power_sizing example output will change (wall-bound at high f_beta) — rerun `uv run python examples/mirror_power_sizing.py` and confirm it still executes cleanly end-to-end (its narration is updated in Task 7).
- [ ] Step 3.5: Commit: `Cap mirror sizing density at the neutron wall-load limit`

---

### Task 4: Surface heat-flux constraint (Part 2, decision c)

**Files:**
- Modify: `src/costingfe/layers/mirror.py` (MirrorPlasmaState `q_surface` field; forward computes it; `_density_from_surface_cap` bisection; third min() branch; gate warning)
- Modify: `src/costingfe/data/defaults/steady_state_mirror.yaml` (`q_surface_max: 1.0`)
- Modify: `src/costingfe/model.py` (plumb q_surface_max)
- Test: `tests/test_mirror.py`

- [ ] Step 4.1: Failing tests first:

```python
def test_state_reports_q_surface(self):
    ps = _forward()
    expected = (float(ps.p_rad) + float(ps.p_radial)) / float(ps.fw_area)
    assert float(ps.q_surface) == pytest.approx(expected, rel=1e-6)

def test_pb11_sizing_is_surface_bound(self):
    # p-B11 radiates 83% of fusion power; the surface cap must bind
    # before both the beta boundary and the neutron cap.
    L, pnet, state = _size(dict(_PB11_SIZING_PARAMS, q_surface_max=1.0))
    assert float(state.q_surface) <= 1.0 * 1.001
    assert float(state.wall_loading) < 5.0      # neutron cap slack
    assert float(state.beta) < 0.85 * 0.5       # beta boundary slack

def test_dt_audit_mode_warns_above_q_surface_max(self):
    with pytest.warns(UserWarning, match=r"surface"):
        _inverse(p_net_target=..., q_surface_max=0.05)  # force the warning
```

- [ ] Step 4.2: Run — fails (no field/branch).
- [ ] Step 4.3: Implement:
  - `MirrorPlasmaState` gains `q_surface: float` (placed after `wall_loading`); `mirror_0d_forward` computes `q_surface = (p_rad + p_radial) / fw_area` and stores it.
  - `_density_from_surface_cap(T_i, T_e, q_surface_max, ..., forward_kwargs)`: uniform monotone bisection on n (40 iterations, brackets [1e17, 1e22]) solving `q_surface(n; T) == q_surface_max`, evaluating the forward's radiation at each probe. Uniform bisection for ALL fuels (supersedes the spec's closed-form/bisection split — one code path, p_radial included exactly; note this in the Task 7 spec sync). This runs in the eager sizing path only (the GSS loop is already eager Python), so a Python loop is fine; guard the early exit: if q_surface at the bracket top is still below the cap, return the bracket top (cap not binding).
  - `net_electric_at_L` density line becomes `n_e = min(n_beta, n_wall, n_surf)` (jnp.minimum chained).
  - Gate in `mirror_0d_inverse`: add a `q_surface` warning parallel to the wall-loading one, threshold `q_surface_max` (required kwarg from YAML, model.py passes it).
  - SizingInfeasible message names `q_surface_max` when the surface branch was binding at L_max.
  - YAML: `q_surface_max: 1.0  # Surface heat-flux cap on the first wall [MW/m^2] (photons + radial transport); provenance: account doc (Task 5)`.
- [ ] Step 4.4: Update the MirrorPlasmaState field-count assertions if any test pins the dataclass shape (grep `dhe3_dd_frac_eff` in tests for the field-table test). `uv run pytest tests/test_mirror.py -q`, full suite, ruff.
- [ ] Step 4.5: Commit: `Add surface heat-flux constraint to mirror wall limits`

---

### Task 5: Provenance writeup — fluence limits and surface-flux cap (research)

**Files:**
- Create: `docs/account_justification/wall_limits_and_fluence.md`
- Modify: `src/costingfe/data/defaults/steady_state_mirror.yaml` + `src/costingfe/data/defaults/costing_constants.yaml` (comments referencing the doc; values confirmed or revised per sources)

- [ ] Step 5.1: Research via WebSearch/WebFetch, primary sources only (NO pyFECONS): (a) RAFM-steel FW/blanket neutron fluence limits (15-20 MW yr/m^2 class — ARIES studies, EU-DEMO blanket literature, Abdou et al. blanket reviews); (b) large-area actively cooled first-wall surface heat-flux capability (about 1 MW/m^2 — ARIES-AT/ACT design basis); (c) per-fuel spectrum scaling rationale (14.1 MeV vs 2.45 MeV dpa per MW yr/m^2) to justify per-fuel Phi_max values consistent with the existing lifetime ratios (DT 5 / DD 10 / DHE3 30 / PB11 50 FPY). Record EVERY number with citation; honest caveats where the mirror's cylindrical geometry or the advanced-fuel spectra stretch the sources.
- [ ] Step 5.2: Choose and document: `fluence_limit_dt/dd/dhe3/pb11` [MW yr/m^2] and `q_surface_max` default. Source honestly — do NOT pick values to preserve pins; Task 6 quantifies whatever shifts result.
- [ ] Step 5.3: House style check (bare $, no em dashes, no prose tildes, no history narration beyond methodology rationale). Commit: `Document wall fluence limits and surface-flux cap provenance`

---

### Task 6: Fluence-based core lifetime in CAS72 (Part 3, decision d1)

**Files:**
- Modify: `src/costingfe/defaults.py` (fluence_limit_* fields + `fluence_limit(fuel)` accessor; keep `core_lifetime(fuel)` for non-MFE)
- Modify: `src/costingfe/data/defaults/costing_constants.yaml` (fluence_limit_* keys per Task 5; core_lifetime_* comments scoped to IFE/MIF)
- Modify: `src/costingfe/model.py` (~line 1334: q_n-dependent lifetime for the steady-state MFE family)
- Test: `tests/test_mirror.py` + wherever CAS72 is tested (grep `cas72` in tests/)

- [ ] Step 6.1: Failing tests first:

```python
def test_fluence_lifetime_continuity_at_reference(self):
    # DT at exactly q_n = Phi_max / 5 FPY reproduces the legacy 5 FPY.
    cc = load_costing_constants()
    q_ref = cc.fluence_limit(Fuel.DT) / 5.0
    # _core_lifetime_fpy is a new module-level helper in model.py wrapping
    # the clip expression (import it in the test from costingfe.model).
    lifetime = float(_core_lifetime_fpy(cc, Fuel.DT, q_n=q_ref,
                                        lifetime_yr=40.0, availability=0.87))
    assert lifetime == pytest.approx(5.0, rel=1e-9)

def test_cas72_grows_with_wall_loading(self):
    m = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
    lo = m.forward(net_electric_mw=300.0, availability=0.87, lifetime_yr=40.0)
    hi = m.forward(net_electric_mw=700.0, availability=0.87, lifetime_yr=40.0)
    # Higher power at fixed geometry -> higher q_n -> shorter life -> larger CAS72
    assert hi.cost_table.cas72 > lo.cost_table.cas72

def test_lifetime_clamped(self):
    cc = load_costing_constants()
    assert _core_lifetime_fpy(cc, Fuel.DT, q_n=1e6) >= 0.5       # floor
    assert _core_lifetime_fpy(cc, Fuel.PB11, q_n=1e-9) <= 40.0   # plant-life cap
```

- [ ] Step 6.2: Run — fails. Implement:
  - defaults.py: add `fluence_limit_dt/dd/dhe3/pb11: float` fields (values from Task 5's doc) and a `fluence_limit(self, fuel)` accessor mirroring `core_lifetime(self, fuel)`. KEEP core_lifetime_* (IFE/MIF concepts continue using them; update their comments to say so).
  - model.py: add a module-level helper (importable by tests):

```python
def _core_lifetime_fpy(cc, fuel, q_n, lifetime_yr, availability):
    """Fluence-based core lifetime [FPY]: Phi_max / q_n, clamped so the
    replacement-cost gradient stays finite at extreme wall loadings."""
    return jnp.clip(
        cc.fluence_limit(fuel) / jnp.maximum(q_n, 1e-6),
        0.5,
        lifetime_yr * availability,
    )
```

  At the `core_lt = cc.core_lifetime(self.fuel)` site (~1334): for `self.family == ConfinementFamily.STEADY_STATE`, compute `q_n = pt.p_neutron * <per-module scaling exactly as the surrounding cost code treats per-module powers — read the n_mod handling at this site first> / geo.firstwall_area` and `core_lt = _core_lifetime_fpy(cc, self.fuel, q_n, lifetime_yr, availability)`. Non-steady-state families keep `cc.core_lifetime(self.fuel)`. Confirm `pt.p_neutron` is the power-table attribute (grep `p_neutron` in types.py/physics.py; if the name differs, use the actual one).
  - This is differentiable end-to-end (clip + division), so the f_beta and f_GW optimizers feel it — that is the point.
- [ ] Step 6.3: THE SANCTIONED PIN UPDATE. Run the full suite; the following move BY DESIGN: the mirror LCOE pin 93.643616 (`test_default_path_bit_identical`, `test_lcoe_pin_unchanged`), any tokamak/stellarator LCOE/CAS72 pins, and dependent example-value assertions. For EACH: record old -> new in a before/after table appended to `docs/account_justification/wall_limits_and_fluence.md` (concept, q_n, implied lifetime old/new, CAS72 old/new, LCOE old/new), then update the pinned literal with a comment `# re-pinned 2026-06-12: fluence-based CAS72 basis change, see wall_limits_and_fluence.md`. The coil calibration pin 513.375 must NOT move (capital account; if it moves, STOP — something is wrong).
- [ ] Step 6.4: Full suite green post-repin; ruff. Commit: `Replace fixed core lifetime with fluence-based CAS72 for MFE concepts`

---

### Task 7: Examples, docs, spec sync, final review

**Files:**
- Modify: `examples/mirror_power_sizing.py` (narration: wall-bound regime, optimizer no longer rides f_beta to 1.0)
- Modify: `examples/dt_mirror_0d.py` (if its printed numbers/narration are invalidated — run it and read the output)
- Modify: `docs/superpowers/specs/2026-06-12-mirror-wall-loading-and-radial-build.md` (status -> Implemented; record the uniform-bisection unification of the n_surf branch)
- Modify: `docs/papers/1costingfe_paper/1costingfe_paper.tex` (CAS22 markup table row: new markup value + two-bore description; mirror appendix: density resolution min() and the wall caps; CAS72 description: fluence basis) + recompile the tracked PDF
- Test: full suite

- [ ] Step 7.1: Run both mirror examples end to end; update printed narration to match the new behavior (the f_beta optimizer should now return an interior or cap-saturated optimum — print q_wall and q_surface at the solution). Paper: NO history narration (absolute rule); present-state equations for the three density branches and the fluence lifetime; update the Mirror markup-table row (the Task 1 value) and any 1.786x mentions; recompile (two pdflatex passes), commit the PDF.
- [ ] Step 7.2: `grep -rn "1.786\|25/14\|core_lifetime\|q_wall\|5.0 MW" docs/ src/ --include="*.md" --include="*.tex" | grep -v node_modules` — fix stragglers (account docs, YAML comments) so no stale constant survives.
- [ ] Step 7.3: Full suite + ruff; final whole-branch review; merge decision per the user.
- [ ] Step 7.4: Commit: `Document wall constraints and fluence lifetime: examples, spec, paper`

---

## Self-Review Notes

- Spec coverage: Part 1 -> Task 1; Part 2 neutron cap -> Task 3; A_fw basis -> Task 2 (ordered BEFORE Task 3 so the cap formula is born on the correct area); decision (c) surface flux -> Task 4; Part 3 -> Tasks 5+6 (research before values); spec's test-plan items all appear in Tasks 1-4/6.
- The n_surf uniform-bisection choice supersedes the spec's closed-form/bisection split deliberately (one code path, p_radial handled exactly); Task 7 records it in the spec.
- Pin discipline is explicit per task: 513.375 never moves; 93.643616 moves only in Task 6 Step 6.3 with the documented table.
- The Task 1 split-pin value (7.267) and Step 1.4 markup are derived-at-implementation numbers; the plan marks them re-pin-at-impl rather than trusting transcription (lesson from the tau_ii prefactor).
- Task 6 depends on Task 2 only through fw_area consistency for the MIRROR q_n; the tokamak/stellarator q_n uses geo.firstwall_area, which is untouched.
