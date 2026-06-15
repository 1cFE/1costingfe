# Mirror Energy-Balance, Confinement-Regime, and Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. All subagents in this workline run on the opus model per user direction.

**Goal:** Make the mirror 0D model physically self-consistent so its sizing/optimize path lands at a realistic operating temperature: collisionality-gate the confinement-regime combination (always-on kernel fix), close the steady-state energy balance in sizing mode by charging confinement-derived auxiliary power (copying the tokamak pattern), and add stability/validity diagnostics with an explicit constraint only if observation shows it is needed.

**Architecture:** All changes are mirror-local. The confinement bridge replaces the broken harmonic sum in `mirror_0d_forward`. The energy-balance closure adds a mirror analog of the tokamak's `aux_heating_from_confinement` and feeds it as `p_input` to the unchanged shared `mfe_forward_power_balance`, whose existing recirculating (`p_input_eff/eta_pin`) and DEC (`f_dec*eta_de*p_transport`) terms do the netting. The shared power balance and the tokamak/stellarator/IFE paths are not modified. Spec: `docs/superpowers/specs/2026-06-14-mirror-energy-balance-and-stability-design.md` (READ IT FIRST).

**Tech Stack:** Python, JAX (float32, no x64), pytest via `uv run pytest` (xdist enabled: `-n auto` for the full gate, `-m "not slow"` for the dev loop), ruff (pre-commit hook). Branch: `feat/mirror-energy-balance` off `feat/mirror-wall-fluence`.

**Invariants for every task:**
- Full gate green; ruff clean. Baseline at branch time: 578 passed (confirm in Task 0).
- Densities follow the 1e20-unit convention (`_TAU_II_PREFACTOR_20`, `_density_from_f_beta`/`_density_from_wall_cap` return [10^20 m^-3], single SI conversion at the `_net_at_L_T` boundary).
- No Python keyword defaults for physics params (YAML only; None sentinels and behavior flags excepted).
- The coil calibration pin (C220103 = 513.375 M$, capital) must NOT move in any task. Non-mirror concept results (tokamak, stellarator, IFE/MIF) must NOT move until Task 5 (tokamak cross-check is read-only) and never as cost values.
- Commits: single line, no body, NO Co-Authored-By.
- pytest on /mnt/c is slow: iterate with single tests or `-m "not slow"`; run the full `-n auto` gate once before each commit. Never kill a run.

## File structure

- `src/costingfe/layers/mirror.py` — confinement bridge (replace the harmonic sum), the `mirror_aux_heating` helper, the energy-balance wiring in `_net_at_L_T` / `mirror_0d_inverse`, the new diagnostics on `MirrorPlasmaState` and in the forward, the optional stability constraint.
- `src/costingfe/layers/tokamak.py` — read-only reference in Task 5; not modified.
- `src/costingfe/data/defaults/steady_state_mirror.yaml` — new knobs: `p_aux_floor`, regime-bridge constants if any, collisionality/DCLC thresholds, optional stability-bound knob.
- `docs/account_justification/mirror_confinement_regimes.md` — new: the regime-bridge sourcing + the sanctioned re-pin before/after table.
- `docs/account_justification/tokamak_validation.md` — new (Task 5): tokamak-vs-literature cross-check.
- `tests/test_mirror.py` — all new mirror tests.
- `tests/test_tokamak.py` (or a new `tests/test_tokamak_anchors.py` — match where tokamak tests live) — the literature cross-check.
- `docs/papers/1costingfe_paper/1costingfe_paper.tex`, `examples/mirror_power_sizing.py`, `examples/dt_mirror_0d.py` — Task 6.

---

### Task 0: Branch and baseline

- [ ] Step 0.1: `git checkout -b feat/mirror-energy-balance feat/mirror-wall-fluence`
- [ ] Step 0.2: `uv run pytest -q -n auto` — confirm 578 passed (record the actual number if it differs; that is the baseline).
- [ ] Step 0.3: Capture the CURRENT (buggy) D-T sizing optimum for the before/after table later: run `uv run python examples/mirror_power_sizing.py` and record the size-mode T_i (about 59.8 keV), beta, LCOE, and the optimizer f_beta. Save these numbers in a scratch note (not committed) for the Task 6 re-pin table. No commit.

---

### Task 1: Confinement-regime collisionality bridge (always-on)

**Files:**
- Create: `docs/account_justification/mirror_confinement_regimes.md`
- Modify: `src/costingfe/layers/mirror.py` (the combination at line 286; add a bridge function near the confinement kernels), `src/costingfe/data/defaults/steady_state_mirror.yaml` (any new bridge constant)
- Test: `tests/test_mirror.py`

READ FIRST: the spec Part 1; `mirror.py` confinement kernels (`compute_tau_pastukhov`, `compute_tau_gas_dynamic`, lines 124-159), the combination (line 286), and the `collisionality` definition (line 329: `collisionality = L / (v_thi * tau_ii)`, i.e. L/mean-free-path; >>1 collisional, <<1 collisionless); `docs/account_justification/mirror_confinement.md` (the GDT/WHAM anchors and their (n,T)).

- [ ] Step 1.1: RESEARCH (write the doc first, like the fluence task). Via WebSearch/WebFetch, primary sources only (Ryutov gas-dynamic confinement and its collisional validity boundary; Pastukhov 1974 / Cohen et al. 1978 kinetic limit; Ivanov & Prikhodko GDT reviews; Mirnov & Ryutov). Establish: (a) the gas-dynamic confinement applies only when the ion mean-free-path is at or below the device length (collisional, loss cone kept full); (b) in the collisionless limit confinement is loss-cone/Pastukhov limited and the gas-dynamic flow time does NOT apply. Determine the published transition criterion (the collisionality at which the regimes cross). Write `mirror_confinement_regimes.md` documenting the chosen smooth bridge: the functional form that makes the gas-dynamic loss rate vanish smoothly as collisionality drops below the transition (so Pastukhov governs when collisionless and gas-dynamic when collisional), its validity window, and honest caveats. If no turnkey closed-form bridge exists, document the constructed smooth gate over the published boundary. House style: bare $, no em dashes, no prose tildes (write "about"), present-state, no pyFECONS.

- [ ] Step 1.2: Write the failing tests in `tests/test_mirror.py` (class `TestRegimeBridge`):

```python
class TestRegimeBridge:
    def test_collisionless_uses_pastukhov_branch(self):
        # Deeply collisionless (low n, high T): tau_axial must approach the
        # Pastukhov time, NOT the much shorter gas-dynamic time. This is the
        # bug guard: the old harmonic sum gave ~tau_GD here.
        from costingfe.layers.mirror import compute_tau_axial
        n, T, A, R_m, L, B = 1.0e20, 60.0, 2.5, 10.0, 20.0, 3.0
        tii = float(compute_tau_ii(n, T, A))
        phi = float(compute_ambipolar_potential(T, A))
        tau_p = float(compute_tau_pastukhov(tii, R_m, phi, T))
        tau_gd = float(compute_tau_gas_dynamic(R_m, L, T, A))
        # sanity: this point is collisionless (tau_gd << tau_p)
        assert tau_gd < 0.01 * tau_p
        tau_axial = float(compute_tau_axial(tii, R_m, L, T, A, phi, n))
        # bridge must pick (near) Pastukhov, within a small factor, not tau_GD
        assert tau_axial > 0.5 * tau_p

    def test_collisional_uses_gas_dynamic_branch(self):
        # High n, low T: collisional, gas-dynamic governs (tau_axial ~ tau_GD).
        from costingfe.layers.mirror import compute_tau_axial
        n, T, A, R_m, L, B = 5.0e20, 1.0, 2.5, 30.0, 7.0, 0.35
        tii = float(compute_tau_ii(n, T, A))
        phi = float(compute_ambipolar_potential(T, A))
        tau_gd = float(compute_tau_gas_dynamic(R_m, L, T, A))
        tau_axial = float(compute_tau_axial(tii, R_m, L, T, A, phi, n))
        assert tau_axial == pytest.approx(tau_gd, rel=0.5)

    @pytest.mark.parametrize("fuel", [Fuel.DT, Fuel.DD, Fuel.DHE3, Fuel.PB11])
    def test_bridge_jit_matches_eager_and_differentiable(self, fuel):
        from costingfe.layers.mirror import compute_tau_axial
        n, A, R_m, L, B = 1.0e20, 2.5, 10.0, 20.0, 3.0

        def f(T):
            tii = compute_tau_ii(n, T, A)
            phi = compute_ambipolar_potential(T, A)
            return compute_tau_axial(tii, R_m, L, T, A, phi, n)

        eager = float(f(30.0))
        jitted = float(jax.jit(f)(30.0))
        assert jnp.isfinite(jitted)
        assert jitted == pytest.approx(eager, rel=1e-4)
        g = float(jax.grad(f)(30.0))
        assert jnp.isfinite(g)
```

(Adapt the exact `compute_tau_axial` signature to what you implement; it must take the inputs needed to compute collisionality internally so callers do not have to. The collisionality the bridge uses is `L / (v_thi * tau_ii)` exactly as `mirror.py:329` defines it.)

- [ ] Step 1.3: Run `uv run pytest tests/test_mirror.py::TestRegimeBridge -q` — fails (no `compute_tau_axial`).

- [ ] Step 1.4: Implement `compute_tau_axial(tau_ii, R_m, L, T_i, A, phi_keV, n_i)` in `mirror.py` near the confinement kernels: compute `tau_Pastukhov` and `tau_GD`, compute collisionality `L/(v_thi*tau_ii)`, and combine them with the smooth bridge from Step 1.1 so the gas-dynamic rate is suppressed when collisionless. Keep it pure JAX, float32-safe (constants folded float64), differentiable. Then replace `mirror_0d_forward`'s line 286 `inv_tau_axial = 1.0/tau_Pastukhov + 1.0/tau_GD` with `tau_axial = compute_tau_axial(...)` and `inv_tau_axial = 1.0/tau_axial` (keep `tau_Pastukhov`/`tau_GD` computed for the state diagnostics). The downstream `inv_tau_p`, `tau_E`, power split, etc. are unchanged.

- [ ] Step 1.5: Re-anchor. Run `uv run pytest tests/test_mirror.py -k "Anchor or anchor" -q` — the GDT and WHAM anchor tests in `TestAnchors` must still pass within their 2x tolerance on the new bridge. If an anchor now fails, the bridge formula is wrong for that regime: STOP, re-derive in the doc, do not widen the tolerance. (GDT is collisional -> gas-dynamic branch; WHAM is collisionless -> Pastukhov branch.)

- [ ] Step 1.6: `uv run pytest tests/test_mirror.py -q` then full gate `uv run pytest -q -n auto`. Expect mirror forward/inverse/sizing VALUES to shift (tau_E changed for any collisionless point). The coil pin 513.375 and non-mirror pins must hold. The mirror LCOE/sizing pins WILL move; this is part of the sanctioned re-pin but defer the bulk re-pin to Task 6 — for now, update only the mirror tests that BREAK due to the tau change, each with a comment `# tau_E corrected by collisionality-gated regime bridge, see mirror_confinement_regimes.md`, and record old->new in the doc's before/after table (start it now). If a NON-mirror test moves, that is a regression: STOP.
- [ ] Step 1.7: `uv run ruff check`. Commit: `Gate mirror confinement by collisionality (gas-dynamic vs Pastukhov bridge)`

---

### Task 2: Energy-balance closure in sizing mode

**Files:**
- Modify: `src/costingfe/layers/mirror.py` (add `mirror_aux_heating`; wire it in `_net_at_L_T`; audit-mode diagnostic in `mirror_0d_inverse`), `src/costingfe/data/defaults/steady_state_mirror.yaml` (`p_aux_floor`)
- Test: `tests/test_mirror.py`

READ FIRST: the spec Part 2 and Mode distinction; `physics.py` `mfe_forward_power_balance` lines 290-324 (`p_input_eff = max(p_input, p_rad - p_ash)`, `p_transport = p_ash + p_input_eff - p_rad`, `p_dee = f_dec*eta_de*p_transport`, `recirculating = ... + p_input_eff/eta_pin`); `tokamak.py:213-242` (`aux_heating_from_confinement`, the pattern to copy) and `tokamak.py:904-943` (how the tokamak sizing feeds `p_input=p_aux`); the current `_net_at_L_T` in `mirror.py` (it builds `ps` then calls `mfe_forward_power_balance` with `p_input=params["p_input"]`).

CONTROLLER NOTE (verified): feeding `p_input = P_aux` where `P_aux = P_end + P_radial + P_rad - P_alpha` makes `p_transport = p_ash + P_aux - p_rad`; with `p_ash ~ p_alpha` this collapses to `P_end + P_radial` (the real transport loss), so the existing `f_dec*eta_de*p_transport` DEC term and `P_aux/eta_pin` recirculating term net correctly. Q4 option A (reuse existing machinery) is therefore expected to work; the explicit double-duty fallback (option B) is the contingency only if the identity below fails.

- [ ] Step 2.1: Write the failing tests (`TestEnergyBalanceClosure` in `tests/test_mirror.py`):

```python
class TestEnergyBalanceClosure:
    def test_mirror_aux_heating_closes_balance(self):
        # P_aux = max(floor, P_end + P_radial + P_rad - P_alpha) from the state.
        from costingfe.layers.mirror import mirror_aux_heating
        ps = _forward(T_i=15.0)  # a physical, collisional-ish point post-Task-1
        p_aux = float(mirror_aux_heating(ps, p_aux_floor=2.0))
        expected = max(2.0, float(ps.p_end) + float(ps.p_radial)
                       + float(ps.p_rad) - float(ps.p_alpha))
        assert p_aux == pytest.approx(expected, rel=1e-6)

    def test_p_transport_identity_in_sizing(self):
        # When sizing feeds p_input=P_aux, the shared balance's p_transport
        # equals P_end + P_radial to a small tolerance (the netting works in
        # the existing machinery; if this fails, use the explicit fallback).
        # Build a sized point and read the power table + state.
        m = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
        r = m.forward(net_electric_mw=400.0, availability=0.87,
                      lifetime_yr=40.0, size_from_power=True, f_beta=0.85)
        ps = r.plasma_state
        pt = r.power_table  # adapt to the actual attribute name
        p_transport_expected = float(ps.p_end) + float(ps.p_radial)
        assert float(pt.p_transport) == pytest.approx(p_transport_expected, rel=0.05)

    def test_sized_dt_optimum_is_realistic_temperature(self):
        # THE headline regression: with the balance closed and tau fixed, the
        # D-T sizing optimum lands in a realistic band, not ~60 keV.
        m = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
        r = m.forward(net_electric_mw=400.0, availability=0.87,
                      lifetime_yr=40.0, size_from_power=True, f_beta=0.85)
        assert 8.0 <= float(r.plasma_state.T_i) <= 25.0

    def test_tau_E_physical_p_end_below_p_fus(self):
        m = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
        r = m.forward(net_electric_mw=400.0, availability=0.87,
                      lifetime_yr=40.0, size_from_power=True, f_beta=0.85)
        assert float(r.plasma_state.p_end) < float(r.plasma_state.p_fus)
```

(Adapt `r.power_table`/`pt.p_transport`/`r.plasma_state` to the actual result attribute names — grep how existing mirror tests read the state and power table.)

- [ ] Step 2.2: Run the new tests — fail (no `mirror_aux_heating`; sizing still passes fixed p_input so T is still ~60 keV).
- [ ] Step 2.3: Implement `mirror_aux_heating(state, p_aux_floor)` returning `jnp.maximum(p_aux_floor, state.p_end + state.p_radial + state.p_rad - state.p_alpha)`. In `_net_at_L_T`, after building `ps` from the forward, compute `p_aux = mirror_aux_heating(ps, params["p_aux_floor"])` and pass `p_input=p_aux` (instead of `params["p_input"]`) to the `mfe_forward_power_balance` call. The forward/inverse AUDIT path (`mirror_0d_inverse` direct, not via sizing) keeps `p_input=params["p_input"]` AND additionally computes the sustainment-consistency diagnostic `p_input / mirror_aux_heating(ps, floor)` (store it on the state or return it; if storing, add one field `sustainment_ratio` to `MirrorPlasmaState` and a test). YAML: `p_aux_floor: 2.0  # MW; control/startup floor on auxiliary sustainment power`.
- [ ] Step 2.4: Run `test_p_transport_identity_in_sizing`. If it FAILS (the existing machinery does not net cleanly), switch to the explicit double-duty fallback per the spec: compute the mirror recirculating contribution as `p_aux/eta_pin - eta_de*axial_frac*p_end` in a mirror-specific path and document the choice in `mirror_confinement_regimes.md`. Report which path was taken.
- [ ] Step 2.5: `uv run pytest tests/test_mirror.py -q`, full gate `uv run pytest -q -n auto`, `uv run ruff check`. Mirror sizing/optimize values move again (now realistic); update broken mirror pins with `# energy-balance closed: sustainment charged from confinement` and extend the before/after table. Coil pin and non-mirror pins hold. Commit: `Close mirror steady-state energy balance with confinement-derived sustainment`

---

### Task 3: Diagnostics and the observe step

**Files:**
- Modify: `src/costingfe/layers/mirror.py` (validity flag from collisionality; DCLC diagnostic), `src/costingfe/data/defaults/steady_state_mirror.yaml` (thresholds)
- Test: `tests/test_mirror.py`

READ FIRST: spec Part 3; the `collisionality` field already on `MirrorPlasmaState`.

- [ ] Step 3.1: Failing tests (`TestStabilityDiagnostics`):

```python
class TestStabilityDiagnostics:
    def test_collisionality_validity_flag_fires_when_collisionless(self):
        # collisionality = L/mfp; below the Pastukhov-Maxwellian validity
        # threshold the diagnostic flags overestimated confinement.
        ps_hot = _forward(T_i=60.0, n_e=1.0e20)   # collisionless
        ps_cold = _forward(T_i=2.0, n_e=5.0e20)   # collisional
        assert ps_hot.pastukhov_valid is False or float(ps_hot.collisionality) < 1.0
        assert float(ps_cold.collisionality) > float(ps_hot.collisionality)

    def test_dclc_diagnostic_present_and_finite_all_fuels(self):
        for fuel in (Fuel.DT, Fuel.DD, Fuel.DHE3, Fuel.PB11):
            ps = _forward(fuel=fuel)
            assert math.isfinite(float(ps.dclc_parameter))
```

(Adapt field names. Choose the DCLC diagnostic per the spec candidate, e.g. a warm-plasma/loss-cone parameter; document its definition in `mirror_confinement_regimes.md`. These are diagnostics only, not constraints.)

- [ ] Step 3.2: Run — fail (no such fields). Implement: add `collisionality`-based validity reporting (a `pastukhov_valid` bool-as-float or a threshold compare; threshold a YAML knob `collisionality_min`) and a `dclc_parameter` field computed from the loss-cone/warm-plasma proxy. Add fields to `MirrorPlasmaState`, populate in the forward. Update the field-table test if one pins the dataclass shape.
- [ ] Step 3.3: `uv run pytest tests/test_mirror.py -q`, full gate, ruff. Commit: `Add mirror collisionality-validity and DCLC diagnostics`
- [ ] Step 3.4: THE OBSERVE STEP (no code; produces a decision and a committed note). Run the corrected model across all four fuels in sizing/optimize mode (use `examples/mirror_power_sizing.py` extended or a scratch run via the model API). Record for each fuel the settled optimum: T_i, collisionality, the DCLC parameter, beta, Q_eng, LCOE. Write the observations into `mirror_confinement_regimes.md` under a "Settled-regime observation" section. DECIDE: does each fuel settle in a sensible, validity-respecting regime (collisionality not absurd, DCLC parameter within the documented stable window, T realistic)? Commit the doc: `Record settled-regime observation across fuels`. Then:
  - If ALL fuels settle sensibly: mark Task 4 SKIPPED in the plan (note it in the report) and proceed to Task 5.
  - If any fuel walks outside the trustworthy regime: proceed to Task 4.

---

### Task 4: Conditional explicit stability constraint (only if Task 3.4 requires it)

**Files:**
- Modify: `src/costingfe/layers/mirror.py` (stability bound in sizing/optimize), `src/costingfe/data/defaults/steady_state_mirror.yaml` (bound knob), `docs/account_justification/mirror_confinement_regimes.md` (sourcing)
- Test: `tests/test_mirror.py`

SKIP THIS TASK ENTIRELY if Task 3.4 concluded all fuels settle sensibly. Otherwise:

- [ ] Step 4.1: Source the operating bound (leading candidate: minimum warm-plasma fraction to fill the loss cone and stabilize DCLC; GDT/WHAM both rely on this). Document the value and citation in `mirror_confinement_regimes.md`.
- [ ] Step 4.2: Failing test (`TestStabilityConstraint`): a point violating the bound raises `OperatingPointInfeasible` with a stability message; the sizing/optimize search restricts to the stable window; a stable point sizes normally. Write the test against the actual offending regime found in Task 3.4.
- [ ] Step 4.3: Implement the bound as a feasibility constraint in the sizing/optimize path (raise `OperatingPointInfeasible` or restrict the T/n window), with a YAML knob for the limit. Keep it differentiable-friendly where it enters the GSS (a smooth restriction is preferable to a hard reject inside the optimizer; a hard reject is acceptable at the audit gate).
- [ ] Step 4.4: `uv run pytest tests/test_mirror.py -q`, full gate, ruff. Commit: `Add mirror DCLC stability operating bound`

---

### Task 5: Tokamak-vs-literature cross-check (read-only)

**Files:**
- Create: `docs/account_justification/tokamak_validation.md`
- Test: `tests/test_tokamak.py` (or `tests/test_tokamak_anchors.py` — match where tokamak tests live; grep first)

READ FIRST: spec Validation item 4; `tokamak.py` forward/sizing; how existing tokamak tests instantiate `CostModel(TOKAMAK, ...)`.

- [ ] Step 5.1: RESEARCH + write `tokamak_validation.md`. Gather published design points (ITER, SPARC, ARC, ARIES-AT, EU-DEMO) for operating temperature, Q (fusion gain), and recirculating/auxiliary-power fraction, primary sources, every number cited (no pyFECONS). Pick 2-3 of them as cross-check anchors. House style.
- [ ] Step 5.2: Write `TestTokamakAnchors`: run the UNCHANGED tokamak model at each chosen design point's machine parameters and assert operating T, Q, and recirculating fraction within a documented tolerance (start at 2x like the mirror anchors; tighten only with justification). These pin the tokamak (the reference pattern we copied) to literature.

```python
class TestTokamakAnchors:
    def test_arc_class_operating_point(self):
        # ARC (Sorbom 2015): R0~3.3 m, B~9.2 T, ... assert the model's
        # operating T and Q land within 2x of the published values.
        # (fill machine params + published T/Q from tokamak_validation.md)
        ...
```

- [ ] Step 5.3: Run `uv run pytest -k TokamakAnchors -q`. If the tokamak deviates MATERIALLY from literature, do NOT change tokamak behavior under this spec: record the deviation in the doc and report it (it would mean the reference itself needs scrutiny, a separate decision). If within tolerance, the reference is validated. Full gate, ruff. Commit: `Validate tokamak 0D against ITER/SPARC/ARC literature anchors`

---

### Task 6: Sanctioned re-pin, docs, paper, examples, final review

**Files:**
- Modify: `docs/account_justification/mirror_confinement_regimes.md` (finalize before/after table), `docs/papers/1costingfe_paper/1costingfe_paper.tex` (+ PDF), `examples/mirror_power_sizing.py`, `examples/dt_mirror_0d.py`, any remaining mirror test pins
- Test: full gate

- [ ] Step 6.1: Finalize the sanctioned re-pin. Ensure every moved mirror pin carries a `# re-pinned 2026-06-14: energy-balance + regime fix, see mirror_confinement_regimes.md` comment, and the before/after table in the doc is complete: columns concept, fuel, operating T, tau_E, P_aux, Q_eng, recirculating fraction, LCOE (old from Task 0 scratch note, new from the corrected model). Confirm the coil pin 513.375 and all non-mirror pins are unmoved (`uv run pytest -k "calibration_neutrality or lcoe_pin or bit_identical" -q` plus the tokamak/stellarator cost pins).
- [ ] Step 6.2: Examples: run both end to end; update narration to the corrected behavior (realistic T, the recirculating cost of sustainment, the optimizer's settled f_beta). No fabricated numbers; print what the code returns. Present-state, no em dashes, no prose tildes.
- [ ] Step 6.3: Paper (`1costingfe_paper.tex`): present-state ONLY (no history/was/changed/bug narration). Update the mirror appendix confinement section to the collisionality-gated bridge and the closed energy balance (sustainment charged from confinement); add the tokamak literature-anchor note parallel to the mirror anchors. Recompile (two pdflatex passes), commit the tracked PDF.
- [ ] Step 6.4: `grep -rn "harmonic sum\|60 keV\|p_input.*fixed" docs/ src/ --include="*.md" --include="*.tex"` and fix any stale claim about the old behavior. Full gate `uv run pytest -q -n auto` green; `uv run ruff check` clean.
- [ ] Step 6.5: Final whole-branch review (spec compliance then quality). Commit: `Document mirror energy-balance and regime fix: re-pin, paper, examples`. Then the merge decision per the user (this branch + the held wall-fluence work merge together).

---

## Self-Review Notes

- Spec coverage: Part 1 -> Task 1; Part 2 -> Task 2 (with the verified p_transport identity and the documented fallback); Part 3 diagnostics -> Task 3, observe-step -> Task 3.4, conditional constraint -> Task 4 (skippable by design); Mode distinction -> Task 2.3 (sizing closure vs audit p_input + consistency diagnostic); tokamak cross-check -> Task 5; the sanctioned re-pin -> threaded through Tasks 1/2 and finalized in Task 6.
- The headline success criterion (D-T optimum at realistic T, not 60 keV) is `test_sized_dt_optimum_is_realistic_temperature` in Task 2 and re-confirmed in the Task 3.4 observation.
- Pin discipline is explicit per task: coil 513.375 never moves; non-mirror never moves; mirror pins move across Tasks 1-2 and are finalized in Task 6 with the before/after table.
- Task 1's bridge formula and Task 5's literature values are derived-at-implementation (research steps), pinned after derivation, following the fluence-task precedent rather than transcription.
- compute_tau_axial / mirror_aux_heating / the new state fields are the cross-task type contracts; later tasks use exactly these names.

---

## Revision 2026-06-14: tandem reframe (user-approved)

Verification after Tasks 1-2 found the corrected model spuriously ignites (tau_E ~15-24x too long at the collisionless operating point) so the D-T optimum stays ~60 keV. Root cause: the CAS22 coil account already costs a TANDEM (n_plug_coils=4, Hammir class), but the confinement uses an unbounded simple-mirror Boltzmann ambipolar potential. See the spec's "Revision 2026-06-14 (tandem reframe)". Tasks 1 and 2 STAND (the bridge is correct; the energy-balance closure is correct). The remaining tasks are revised:

### Task 2b (NEW): Tandem plug-limited confinement calibrated to Hammir Q>5

**Files:** Modify `src/costingfe/layers/mirror.py` (the confining potential / plug confinement), `src/costingfe/data/defaults/steady_state_mirror.yaml` (plug-confinement calibration knob), `docs/account_justification/mirror_confinement_regimes.md` (Hammir + tandem-lit sourcing); Test: `tests/test_mirror.py`.

- [ ] Step 2b.1: RESEARCH (write to mirror_confinement_regimes.md first). Primary sources for tandem-mirror central-cell confinement: Realta Hammir Q>5 design point (50 m central cell; published fields, density, temperature, Q -- Forest et al. / Realta announcements), and the classic tandem-mirror literature (Fowler & Logan tandem concept; Baldwin & Logan thermal barrier; MFTF-B and TMX-U design/results; Fowler & Ryutov). Establish a tandem central-cell confining potential or n*tau that, at the Hammir reference, gives Q>5 (NOT ignition). No pyFECONS.
- [ ] Step 2b.2: Failing tests (`TestTandemConfinement`): (a) Hammir anchor -- at the published Hammir machine the model's central-cell confinement reproduces the Q>5 design point within 2x; (b) the D-T sizing optimum now lands in the realistic band (this is the xfail'd `test_sized_dt_optimum_is_realistic_temperature` from Task 2 -- it should now PASS; remove the xfail); (c) the plasma is not spuriously ignited (Q tandem-realistic ~1-few at the optimum); (d) jit==eager + finite grad for the new confining-potential function across fuels.
- [ ] Step 2b.3: Implement: replace the unbounded `compute_ambipolar_potential` use in the Pastukhov-branch confinement with a tandem plug-limited confining potential -- a bounded form (cap the effective e*phi/T_i ratio) or an explicit plug-confining-potential YAML parameter, calibrated so the Hammir anchor reproduces Q>5. Keep `compute_ambipolar_potential` available as the simple-mirror diagnostic. Keep Task 1's collisionality bridge (collisionless -> plugged branch). Document the calibration in the doc. YAML knob for the plug confinement.
- [ ] Step 2b.4: Revisit the Task 2 `f_dec_eff` fallback: with the plasma no longer ignited, confirm the clean p_transport identity (p_transport ~ P_end + P_radial) now holds and REMOVE the f_dec_eff fallback if so (flip the xfail'd `test_p_transport_identity_in_sizing` to passing). If it still does not hold, keep the corrected fallback (compute against the shared function's actual internal p_rad, fixing the 32% error the reviewer found).
- [ ] Step 2b.5: Full gate `uv run pytest -q -n auto`; ruff. Mirror values move again (realistic now). Coil pin 513.375 and non-mirror pins hold. Extend the before/after table. Commit: `Calibrate mirror plug confinement to Hammir Q>5 tandem design point`

### Task 3 (revised): diagnostics + observe against the tandem model
Unchanged in intent (collisionality validity flag + DCLC diagnostic), but the OBSERVE step now runs against the tandem-calibrated model. Expectation: D-T settles realistic; surface the per-fuel observation to the user. Decide if Task 4 is still needed.

### Task 4 (likely SKIP): conditional stability constraint
The tandem calibration (Task 2b) is now the lever, so an explicit DCLC feasibility bound is likely unnecessary. Skip unless Task 3's observation shows a fuel still walking outside the trustworthy regime.

### Tasks 5-6: unchanged
Tokamak literature cross-check (Task 5) and the sanctioned re-pin + docs + paper + final review (Task 6) proceed as written, now over the tandem-calibrated results.
