# burn_fraction differentiation by concept — design

## Motivation

The framework currently carries a single global `burn_fraction = 0.05` (and `fuel_recovery = 0.99`) in `costing_constants.yaml`, applied to every confinement concept. The paper itself flags this as MFE-centric ("pulsed-compression and ICF concepts achieve much higher single-pass burnup, 10–40%+"). The result is that concept-resolved LCOE comparisons currently use an MFE-class burnup for pulsed and inertial concepts, inflating their fuel costs unrealistically and obscuring real economic differences between regimes.

Within-regime burnup variation in pulsed concepts is large enough (factor 2–10x between, e.g., dense-plasma-focus and heavy-ion fusion) that a single bucket value can't represent them honestly. A hybrid approach — one MFE-regime value, individual values per pulsed concept — is the smallest change that captures the physics.

Separately: `burn_fraction` and `fuel_recovery` are the only physics quantities living in `costing_constants.yaml`, which otherwise holds $/cost values. Moving them to per-concept YAMLs also tidies that layering.

## Decisions

1. **`burn_fraction` becomes per-concept.** 5 steady-state MFE concepts share a regime default (0.05). Each of the 10 pulsed concepts gets its own value.
2. **`fuel_recovery` stays uniform at 0.99 numerically**, but is declared per-concept so all physics knobs live in the same place. Justification: the within-concept spread is ≤ 5 percentage points, and the cumulative-fusion sensitivity to `fuel_recovery` is highest exactly where concepts are most architecturally similar (low-burnup MFE, all gas-phase exhaust).
3. **Both fields move out of `CostingConstants` entirely.** No concept-less code path exists (the `CostingModel` constructor requires `concept`), so a global fallback is dead weight. Keeping the dataclass field hides the policy that values must be concept-specific.
4. **No new dispatch code.** The existing concept-YAML → `CostingConstants` merge order in `model.py:433–439` already favors YAML values; moving the keys is sufficient.

## Architecture

### Where the values live

Per-concept YAMLs (`src/costingfe/data/defaults/<family>_<concept>.yaml`, all 15 files):

- Each YAML declares `burn_fraction:` with its concept-specific value.
- Each YAML declares `fuel_recovery: 0.99` (uniform value, declared by policy).

`costing_constants.yaml` and `CostingConstants` (defaults.py):

- The two lines are removed from `costing_constants.yaml`.
- The two `float` fields are removed from the `CostingConstants` dataclass.
- `cc_float_fields()` consequently stops reporting them, and the `setdefault` loop in `model.py:437` stops injecting them.

`CostingInput` (validation.py):

- Both fields are added to the engineering-parameter block as `float | None = None` with pydantic `Field(gt=0, le=1)` constraints.
- Both names are added to `_COMMON_REQUIRED` so missing-from-YAML failures raise a clear "Missing required engineering parameters" error rather than a KeyError in the physics layer.

`_engineering_keys` (model.py:890):

- Both names are added to the `common` list, next to `dd_f_T`, `dhe3_f_T`, etc., so they continue to appear in the JAX-autodiff sensitivity tornado.

### Call path

Unchanged. The existing `model.py:433–439` flow — `params = self._eng_defaults; setdefault from cc; params.update(overrides)` — keeps working. The YAML provides the value; user kwargs still override.

## Per-concept values

### Steady-state MFE (5 concepts, regime default 0.05)

| Concept | Range | Recommended | Basis |
|---|---|---|---|
| tokamak | 0.03–0.10 | 0.05 | ARIES-AT, ITER-class operating point |
| stellarator | 0.03–0.10 | 0.05 | HSX/W7-X reactor projections; same n·τ regime |
| mirror | 0.02–0.08 | 0.05 | Modern axisymmetric tandem mirror (Realta-class); legacy MFTF studies quote 1–3% but high-mirror-ratio designs catch up |
| orbitron | n/a | 0.05 | No public reactor study; document as MFE-class assumption |
| polywell | n/a | 0.05 | No public reactor study; document as MFE-class assumption |

### Inertial (2 concepts)

| Concept | Range | Recommended | Basis |
|---|---|---|---|
| laser_ife | 0.10–0.35 | 0.25 | NIF ignition shots 2–4% (Lawrence 2024); reactor-target designs (LIFE, HYLIFE, HAPL) assume 25–35% |
| heavy_ion | 0.20–0.40 | 0.30 | HIBALL/HIBLIC indirect-drive uniformity; higher than laser |

### Magneto-inertial / pulsed magnetic (8 concepts)

| Concept | Range | Recommended | Basis |
|---|---|---|---|
| maglif | 0.08–0.25 | 0.15 | Slutz 2010 / Knapp 2019 2D LASNEX projections at full Z-driver |
| mag_target | 0.05–0.15 | 0.10 | General Fusion plasma-compression projections; MIF concept literature |
| plasma_jet | 0.05–0.15 | 0.10 | PJMIF MHD sims (Witherspoon, Hsu) |
| pulsed_frc | 0.10–0.25 | 0.15 | Helion-class staged-compression FRC; vendor does not publish a burn fraction (figure of merit is Q ≈ 6–11 at peak compression). 0.15 is a regime midpoint from the magneto-inertial fusion literature (10–25% target for economic operation) |
| staged_zpinch | 0.05–0.15 | 0.10 | LANL Rahman/Wessel staged-pinch designs |
| zpinch | 0.05–0.15 | 0.10 | Standard Z-pinch reactor concepts, Rayleigh-Taylor disassembly limit |
| theta_pinch | 0.03–0.08 | 0.05 | Faster expansion than Z-pinch; closer to MFE bound |
| dense_plasma_focus | 0.005–0.02 | 0.01 | LPP Focus Fusion projections; sub-microsecond pinch lifetime is the binding constraint |

### Notable points

- The `dense_plasma_focus` value (0.01) is a real outlier and will move LCOE materially in any comparison involving DPF.
- `pulsed_frc` at 0.15 is a regime midpoint; Helion does not publish a burn fraction (their figure of merit is Q). Lookup-confirmed during this design.
- `orbitron` and `polywell` get 0.05 as an MFE-class placeholder; account justification doc must call this out explicitly.

## Documentation

`docs/account_justification/burn_fraction.md` (new file):

- One section per concept (or per regime for the MFE 5).
- For each: range, recommended value, primary source(s), provenance line.
- The `pulsed_frc` / Helion section starts pre-populated with the lookup results:
  - [Helion FAQ](https://www.helionenergy.com/faq/)
  - [More on Helion's pulsed approach](https://www.helionenergy.com/articles/more-on-helions-pulsed-approach-to-fusion/)
  - [Slough et al., "A compact fusion reactor based on staged compression of an FRC" (Nucl. Fusion 2024)](https://iopscience.iop.org/article/10.1088/1741-4326/ae034d) — Q-based metrics, no burn fraction
  - [Wurden, "Magneto-Inertial Fusion" 2-pager (PPPL)](https://fire.pppl.gov/IFE_NAS_MTF_Wurden_2pager.pdf) — generic MIF regime statement
  - [Hybrid simulations of FRC merging and compression (arxiv 2501.03425)](https://arxiv.org/pdf/2501.03425)
  - [Quasi-static magnetic compression of FRC (arxiv 2204.07978)](https://arxiv.org/pdf/2204.07978)
  - Note: vendor publishes Q, not burn fraction; 0.15 is a MIF literature regime midpoint.
- Expanding the per-concept sourcing for the other 14 concepts is a follow-up; see the Follow-ups section below.

## Tests, validation, and other touch points

`_VALIDATION_PHYSICS` (validation.py:43–64):

- The placeholder `dhe3_f_He3=0.84  # bf=0.05, fr=0.99` is a synthetic value used for cross-field warning checks. Value stays at 0.84; comment updated to "representative MFE-class value; concept-specific values flow through compute()."

Existing tests:

- No `tests/` files reference `burn_fraction` or `fuel_recovery` directly (verified by grep). Likely-affected areas to verify after the move:
  - Anything that instantiates `CostingConstants(...)` with explicit kwargs.
  - Anything that loops over `cc_float_fields()` and expects these two keys.
  - `test_backcast.py` — references legacy ARC/ARIES configs; check whether they pass values through or rely on the global default.

Examples:

- `examples/dhe3_pulsed_frc.py` and `examples/path_to_1cent.py` currently hardcode `burn_fraction=0.10, fuel_recovery=0.95` as kwargs. These continue to override post-move (kwarg always wins), so example outputs are unchanged. Whether to keep the hardcoded kwargs or let YAML defaults apply is a separate cleanup, out of scope.

Paper:

- The current footnote (`paper.tex` lines ~819–824) that the NOAK defaults are MFE-centric stops being accurate once values are per-concept. Replace with a sentence noting values are concept-specific and pointing at the account_justification doc.

Companion scripts:

- `benchmark_arc.py`, `benchmark_aries_at.py` target D-T tokamak at 0.05; numbers should be identical post-move, but rerun to confirm.
- Any blog companion script touching a non-MFE concept (DEC blog, LCOE comparison scripts) should be rerun and any drift documented.

## Implementation sequencing

Two phases with a green-tests checkpoint between them, so the codebase stays runnable throughout.

### Phase A — additive

1. Create `docs/account_justification/burn_fraction.md` with the Helion section pre-populated and stubs for the other 14 concepts.
2. Add `burn_fraction:` and `fuel_recovery:` lines to all 15 concept YAMLs (values per Section "Per-concept values").
3. Add `burn_fraction` and `fuel_recovery` fields to `CostingInput` with `Field(gt=0, le=1)`.
4. Add both names to `_COMMON_REQUIRED`.
5. Add both names to the `common` list in `_engineering_keys` (model.py:890).
6. Run pytest. Should be green: YAML values override the still-present CC defaults, validation passes, sensitivity tornado picks them up.

### Phase B — remove the fallback

7. Delete the two lines from `costing_constants.yaml`.
8. Delete the two `float` fields from the `CostingConstants` dataclass.
9. Update the `_VALIDATION_PHYSICS` comment (just the comment; value stays 0.84).
10. Run pytest again. Same green result.

### Phase C — paper text and companion scripts

11. Update `1costingfe_paper.tex` footnote about the MFE-centric NOAK default.
12. Re-run `benchmark_arc.py` and `benchmark_aries_at.py`. Numbers should be identical; flag any drift.
13. Re-run any blog companion script that touches a non-MFE concept; document any drift.

## Out of scope

- Reassessing the FOAK/NOAK switch's interaction with `fuel_recovery` (e.g., should FOAK push fuel_recovery to 0.95?). The current scope keeps fuel_recovery=0.99 uniform.
- Per-concept fuel_recovery values driven by exhaust architecture (gas-phase vs target-factory). Within-concept spread is ≤ 5pp, and where it would matter most (low-burnup MFE) the concepts are most uniform.
- Cleanup of hardcoded kwargs in `examples/dhe3_pulsed_frc.py` and `examples/path_to_1cent.py`.
- Per-concept fuel_recovery research; the regime statement in this spec is the working position.

## Follow-ups

- Expanding `burn_fraction.md` with per-concept sources/citations for the 14 concepts other than `pulsed_frc`. This is research work to be done as a separate task once the framework wiring is in place.
