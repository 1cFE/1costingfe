# burn_fraction differentiation by concept — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `burn_fraction` and `fuel_recovery` from the global `CostingConstants` into per-concept YAMLs, with concept-specific `burn_fraction` values (hybrid: regime default for steady-state MFE, individual values per pulsed concept) and uniform `fuel_recovery: 0.99`.

**Architecture:** Phase A is purely additive (YAML keys, pydantic fields, sensitivity-tornado registration) and leaves the existing `CostingConstants` fallback in place so nothing breaks. Phase B removes the fallback now that every concept YAML declares the values. Phase C updates the paper text and reruns benchmark/companion scripts to confirm no unintended LCOE drift.

**Tech Stack:** Python 3.10+, pydantic v2 for validation, JAX for sensitivity autodiff, pytest for tests, PyYAML for config loading.

**Reference spec:** `docs/plans/2026-05-21-burn-fraction-by-concept-design.md`.

**Working directory:** `/mnt/c/Users/talru/1cfe/1costingfe`. All paths below are relative to this directory unless otherwise noted.

**Test command:** `pytest tests/`. Run from repo root.

---

## File structure

**Created:**
- `docs/account_justification/burn_fraction.md` — provenance for each concept's burn_fraction value, with the pulsed_frc / Helion section pre-populated.

**Modified:**
- All 15 `src/costingfe/data/defaults/<family>_<concept>.yaml` — add `burn_fraction:` and `fuel_recovery: 0.99` lines.
- `src/costingfe/data/defaults/costing_constants.yaml` — remove `burn_fraction` and `fuel_recovery` lines (Phase B).
- `src/costingfe/defaults.py` — remove `burn_fraction` and `fuel_recovery` fields from `CostingConstants` (Phase B).
- `src/costingfe/validation.py` — add fields to `CostingInput`; add to `_COMMON_REQUIRED`; update `_VALIDATION_PHYSICS` comment.
- `src/costingfe/model.py` — add `burn_fraction` and `fuel_recovery` to `_engineering_keys` common list.
- `tests/test_defaults.py` — new parametrized test verifying every concept YAML declares both fields with values in (0, 1].
- `docs/papers/1costingfe_paper/1costingfe_paper.tex` — replace the MFE-centric NOAK-default footnote (Phase C).

---

## Per-concept values reference

Used in Phase A Task 2 (filling YAMLs). Sourced from `docs/plans/2026-05-21-burn-fraction-by-concept-design.md`.

| YAML file | burn_fraction | fuel_recovery |
|---|---|---|
| `steady_state_tokamak.yaml` | 0.05 | 0.99 |
| `steady_state_stellarator.yaml` | 0.05 | 0.99 |
| `steady_state_mirror.yaml` | 0.05 | 0.99 |
| `steady_state_orbitron.yaml` | 0.05 | 0.99 |
| `steady_state_polywell.yaml` | 0.05 | 0.99 |
| `pulsed_laser_ife.yaml` | 0.25 | 0.99 |
| `pulsed_heavy_ion.yaml` | 0.30 | 0.99 |
| `pulsed_maglif.yaml` | 0.15 | 0.99 |
| `pulsed_mag_target.yaml` | 0.10 | 0.99 |
| `pulsed_plasma_jet.yaml` | 0.10 | 0.99 |
| `pulsed_pulsed_frc.yaml` | 0.15 | 0.99 |
| `pulsed_staged_zpinch.yaml` | 0.10 | 0.99 |
| `pulsed_zpinch.yaml` | 0.10 | 0.99 |
| `pulsed_theta_pinch.yaml` | 0.05 | 0.99 |
| `pulsed_dense_plasma_focus.yaml` | 0.01 | 0.99 |

---

# Phase A — Additive (codebase remains green throughout)

## Task A1: Write parametrized test for per-concept declaration

**Files:**
- Modify: `tests/test_defaults.py` (append a new test)

- [ ] **Step 1: Add the failing test**

Append at the end of `tests/test_defaults.py`:

```python
@pytest.mark.parametrize("yaml_path", _CONCEPT_YAMLS, ids=lambda p: p.stem)
def test_every_concept_yaml_declares_burn_fraction_and_fuel_recovery(yaml_path):
    """Every concept YAML must declare burn_fraction and fuel_recovery; both
    must lie in (0, 1].

    Background: these two physics knobs used to live in costing_constants.yaml
    as global defaults but are concept-specific (burn_fraction) or declared
    per-concept by policy (fuel_recovery). See
    docs/plans/2026-05-21-burn-fraction-by-concept-design.md.
    """
    data = yaml.safe_load(yaml_path.read_text())
    for key in ("burn_fraction", "fuel_recovery"):
        assert key in data, f"{yaml_path.name} missing {key}"
        value = data[key]
        assert isinstance(value, (int, float)), (
            f"{yaml_path.name}: {key}={value!r} is not numeric"
        )
        assert 0 < value <= 1, (
            f"{yaml_path.name}: {key}={value} is outside (0, 1]"
        )
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `pytest tests/test_defaults.py::test_every_concept_yaml_declares_burn_fraction_and_fuel_recovery -v`

Expected: 15 parametrized cases, all FAIL with `AssertionError: <file>.yaml missing burn_fraction` because no YAML declares these keys yet.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_defaults.py
git commit -m "Add failing test: every concept YAML must declare burn_fraction and fuel_recovery"
```

---

## Task A2: Add burn_fraction and fuel_recovery to all 15 concept YAMLs

**Files:**
- Modify: all 15 files in `src/costingfe/data/defaults/<family>_<concept>.yaml`

Each YAML already has a comment block beginning with `# Fuel burn fraction defaults` (or similar) followed by `dd_f_T: 0.969`. Insert `burn_fraction:` and `fuel_recovery:` lines immediately *before* the `dd_f_T:` line within that block. The two new lines belong logically with the fuel-utilization knobs.

For example, in `steady_state_tokamak.yaml` change:

```yaml
# Fuel burn fraction defaults (physics.py ash_neutron_split)
dd_f_T: 0.969
```

to:

```yaml
# Fuel burn fraction defaults (physics.py ash_neutron_split)
burn_fraction: 0.05         # Single-pass burn fraction (MFE-class)
fuel_recovery: 0.99         # Fraction of unburned fuel recovered and recycled (NOAK)
dd_f_T: 0.969
```

- [ ] **Step 1: Edit steady_state_tokamak.yaml**

Add `burn_fraction: 0.05` and `fuel_recovery: 0.99` immediately before the `dd_f_T: 0.969` line.

- [ ] **Step 2: Edit steady_state_stellarator.yaml**

Add `burn_fraction: 0.05` and `fuel_recovery: 0.99` immediately before the `dd_f_T: 0.969` line.

- [ ] **Step 3: Edit steady_state_mirror.yaml**

Add `burn_fraction: 0.05` and `fuel_recovery: 0.99` immediately before the `dd_f_T: 0.969` line.

- [ ] **Step 4: Edit steady_state_orbitron.yaml**

Add `burn_fraction: 0.05` and `fuel_recovery: 0.99` immediately before the `dd_f_T: 0.969` line.

- [ ] **Step 5: Edit steady_state_polywell.yaml**

Add `burn_fraction: 0.05` and `fuel_recovery: 0.99` immediately before the `dd_f_T: 0.969` line.

- [ ] **Step 6: Edit pulsed_laser_ife.yaml**

Add `burn_fraction: 0.25` and `fuel_recovery: 0.99` immediately before the `dd_f_T: 0.969` line. Comment for burn_fraction: `# Single-pass burn fraction (inertial; reactor-target value, NIF shots get 2-4%)`.

- [ ] **Step 7: Edit pulsed_heavy_ion.yaml**

Add `burn_fraction: 0.30` and `fuel_recovery: 0.99` immediately before the `dd_f_T: 0.969` line. Comment: `# Single-pass burn fraction (heavy-ion indirect-drive uniformity, HIBALL/HIBLIC)`.

- [ ] **Step 8: Edit pulsed_maglif.yaml**

Add `burn_fraction: 0.15` and `fuel_recovery: 0.99` immediately before the `dd_f_T: 0.969` line. Comment: `# Single-pass burn fraction (MagLIF; Slutz 2010 / Knapp 2019 2D LASNEX at full Z-driver)`.

- [ ] **Step 9: Edit pulsed_mag_target.yaml**

Add `burn_fraction: 0.10` and `fuel_recovery: 0.99` immediately before the `dd_f_T: 0.969` line. Comment: `# Single-pass burn fraction (MTF/MIF concept literature)`.

- [ ] **Step 10: Edit pulsed_plasma_jet.yaml**

Add `burn_fraction: 0.10` and `fuel_recovery: 0.99` immediately before the `dd_f_T: 0.969` line. Comment: `# Single-pass burn fraction (PJMIF MHD simulations, Witherspoon, Hsu)`.

- [ ] **Step 11: Edit pulsed_pulsed_frc.yaml**

Add `burn_fraction: 0.15` and `fuel_recovery: 0.99` immediately before the `dd_f_T: 0.969` line. Comment: `# Single-pass burn fraction (Helion-class staged-compression FRC; MIF literature regime midpoint, vendor publishes Q, not burnup)`.

- [ ] **Step 12: Edit pulsed_staged_zpinch.yaml**

Add `burn_fraction: 0.10` and `fuel_recovery: 0.99` immediately before the `dd_f_T: 0.969` line. Comment: `# Single-pass burn fraction (LANL Rahman/Wessel staged Z-pinch designs)`.

- [ ] **Step 13: Edit pulsed_zpinch.yaml**

Add `burn_fraction: 0.10` and `fuel_recovery: 0.99` immediately before the `dd_f_T: 0.969` line. Comment: `# Single-pass burn fraction (Z-pinch reactor concepts; Rayleigh-Taylor disassembly limit)`.

- [ ] **Step 14: Edit pulsed_theta_pinch.yaml**

Add `burn_fraction: 0.05` and `fuel_recovery: 0.99` immediately before the `dd_f_T: 0.969` line. Comment: `# Single-pass burn fraction (theta-pinch; faster expansion than Z-pinch, MFE-bound)`.

- [ ] **Step 15: Edit pulsed_dense_plasma_focus.yaml**

Add `burn_fraction: 0.01` and `fuel_recovery: 0.99` immediately before the `dd_f_T: 0.969` line. Comment: `# Single-pass burn fraction (DPF; sub-microsecond pinch lifetime limits dwell)`.

- [ ] **Step 16: Run the parametrized test to confirm it now passes**

Run: `pytest tests/test_defaults.py::test_every_concept_yaml_declares_burn_fraction_and_fuel_recovery -v`

Expected: 15/15 PASS.

- [ ] **Step 17: Commit**

```bash
git add src/costingfe/data/defaults/*.yaml
git commit -m "Declare burn_fraction and fuel_recovery in every concept YAML"
```

---

## Task A3: Add burn_fraction and fuel_recovery to CostingInput

**Files:**
- Modify: `src/costingfe/validation.py:97-110` (engineering-parameter block, common section)
- Modify: `src/costingfe/validation.py:153-165` (`_COMMON_REQUIRED` list)

- [ ] **Step 1: Add fields to the common engineering-parameter block**

In `src/costingfe/validation.py`, locate the block at lines 96-110:

```python
    # --- Engineering parameters (None = use YAML template) ---
    # Common (all families)
    mn: float | None = None
    eta_th: float | None = None
    eta_p: float | None = None
    f_sub: float | None = None
    p_pump: float | None = None
    p_trit: float | None = None
    p_house: float | None = None
    p_cryo: float | None = None
    blanket_t: float | None = None
    ht_shield_t: float | None = None
    structure_t: float | None = None
    vessel_t: float | None = None
    plasma_t: float | None = None
```

Add two new lines at the end of this block (before the blank line and the `# MFE only` comment):

```python
    burn_fraction: float | None = Field(default=None, gt=0, le=1)
    fuel_recovery: float | None = Field(default=None, gt=0, le=1)
```

- [ ] **Step 2: Add to _COMMON_REQUIRED**

In the same file, locate lines 153-165:

```python
    _COMMON_REQUIRED = [
        "mn",
        "f_sub",
        "p_pump",
        "p_trit",
        "p_house",
        "p_cryo",
        "blanket_t",
        "ht_shield_t",
        "structure_t",
        "vessel_t",
        "plasma_t",
    ]
```

Add two entries at the end of the list:

```python
    _COMMON_REQUIRED = [
        "mn",
        "f_sub",
        "p_pump",
        "p_trit",
        "p_house",
        "p_cryo",
        "blanket_t",
        "ht_shield_t",
        "structure_t",
        "vessel_t",
        "plasma_t",
        "burn_fraction",
        "fuel_recovery",
    ]
```

- [ ] **Step 3: Run the validation test suite to confirm no regressions**

Run: `pytest tests/test_defaults.py tests/test_adapter.py tests/test_model.py -v`

Expected: all PASS. (Concept YAMLs now declare the values, so `_COMMON_REQUIRED` is satisfied for every concept.)

- [ ] **Step 4: Run the full suite as a checkpoint**

Run: `pytest tests/`

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/costingfe/validation.py
git commit -m "Validate burn_fraction and fuel_recovery as common engineering parameters"
```

---

## Task A4: Add burn_fraction and fuel_recovery to sensitivity tornado

**Files:**
- Modify: `src/costingfe/model.py:892-916` (the `common` list inside `_engineering_keys`)

- [ ] **Step 1: Add both names to the common engineering keys**

In `src/costingfe/model.py`, locate the `_engineering_keys` method around lines 890-916. The relevant `common` list currently ends:

```python
            # Fuel burn fractions (physics model)
            "dd_f_T",
            "dd_f_He3",
            "dhe3_dd_frac",
            "dhe3_f_T",
            "pb11_f_alpha_n",
            "pb11_f_p_n",
        ]
```

Add `burn_fraction` and `fuel_recovery` at the top of the "Fuel burn fractions (physics model)" group, immediately after the section comment:

```python
            # Fuel burn fractions (physics model)
            "burn_fraction",
            "fuel_recovery",
            "dd_f_T",
            "dd_f_He3",
            "dhe3_dd_frac",
            "dhe3_f_T",
            "pb11_f_alpha_n",
            "pb11_f_p_n",
        ]
```

- [ ] **Step 2: Run the sensitivity / adapter tests**

Run: `pytest tests/test_adapter.py tests/test_model.py -v`

Expected: PASS. The adapter test `test_adapter_tokamak_dt` asserts `"eta_th" in out.sensitivity["engineering"]`; once burn_fraction and fuel_recovery are in `_engineering_keys`, they'll be present in `out.sensitivity["engineering"]` too (no explicit assertion to add — silent inclusion is enough).

- [ ] **Step 3: Run the full suite**

Run: `pytest tests/`

Expected: all PASS. This is the Phase A green checkpoint.

- [ ] **Step 4: Commit**

```bash
git add src/costingfe/model.py
git commit -m "Add burn_fraction and fuel_recovery to sensitivity tornado engineering keys"
```

---

## Task A5: Create account_justification doc with Helion section pre-populated

**Files:**
- Create: `docs/account_justification/burn_fraction.md`

- [ ] **Step 1: Create the file**

Write `docs/account_justification/burn_fraction.md` with the following content:

```markdown
# burn_fraction account justification

Per-concept single-pass burn fraction `burn_fraction` values, with sourcing
and provenance notes.

See also: `docs/plans/2026-05-21-burn-fraction-by-concept-design.md` for the
broader design rationale.

## Method

`burn_fraction` is the fraction of injected fuel atoms that undergo fusion
before being exhausted from the reaction volume. Combined with
`fuel_recovery` (the fraction of unburned fuel recovered and recycled), it
gives the cumulative fusion probability of an injected atom as
`burn_fraction / (1 - fuel_recovery * (1 - burn_fraction))`, which drives
CAS80 fuel-cost accounting.

The value is bounded above by physics (kinematic and confinement) and
chosen by design (operating-point selection). Values below are reactor-
target (NOAK) design points, not experimental records.

## Steady-state MFE (regime default 0.05)

The MFE concepts (tokamak, stellarator, mirror, orbitron, polywell) all
operate at moderate n·τ confinement and share the same single-pass burnup
regime. Range across designs: 0.03-0.10. Standard reactor-target value:
0.05.

- **tokamak: 0.05** — ARIES-AT, ITER-class operating point. Sourcing
  pending detailed citation.
- **stellarator: 0.05** — HSX/W7-X reactor projections, same n·τ regime as
  tokamak. Sourcing pending.
- **mirror: 0.05** — Modern axisymmetric tandem mirror (Realta-class).
  Legacy MFTF/TARA studies quote 1-3%; high-mirror-ratio designs catch up.
  Sourcing pending.
- **orbitron: 0.05** — No public reactor-scale study. **MFE-class
  placeholder** pending design-specific data.
- **polywell: 0.05** — No public reactor-scale study. **MFE-class
  placeholder** pending design-specific data.

## Inertial confinement (ICF)

Hot-spot ρR sets the burn fraction; reactor-target designs assume
substantially higher values than NIF ignition shots (which achieve 2-4%).

- **laser_ife: 0.25** — Reactor-target value from direct-drive ICF reactor
  studies (LIFE, HYLIFE, HAPL NRL). Range: 0.10-0.35. Sourcing pending.
- **heavy_ion: 0.30** — Indirect-drive heavy-ion fusion (HIBALL, HIBLIC).
  Higher uniformity than direct-drive laser. Range: 0.20-0.40. Sourcing
  pending.

## Magneto-inertial / pulsed magnetic

A heterogeneous bucket where burn fraction depends strongly on
compression dwell time, peak ρ, and disassembly mechanism. Each concept
gets an individual value.

### pulsed_frc: 0.15 (Helion-class staged-compression FRC)

**Vendor does not publish a burn fraction.** Helion and the broader
compressed-FRC literature use **fusion gain Q** as the figure of merit
(Q ≈ 6-11 at peak compression for D-T staged-compression FRC).

The 0.15 value is a regime midpoint from the magneto-inertial fusion
literature, which generically targets 10-25% single-pass burnup for
economic operation. This will need updating if Helion or a peer
publishes a concept-specific burnup figure.

Sources reviewed (none quote a burn fraction directly):

- [Helion FAQ](https://www.helionenergy.com/faq/)
- [More on Helion's pulsed approach](https://www.helionenergy.com/articles/more-on-helions-pulsed-approach-to-fusion/)
- [Slough et al., "A compact fusion reactor based on staged compression of an FRC" (Nucl. Fusion 2024)](https://iopscience.iop.org/article/10.1088/1741-4326/ae034d) — Q-based metrics, no burn fraction
- [Wurden, "Magneto-Inertial Fusion" 2-pager (PPPL)](https://fire.pppl.gov/IFE_NAS_MTF_Wurden_2pager.pdf) — generic MIF regime statement
- [Hybrid simulations of FRC merging and compression (arxiv 2501.03425)](https://arxiv.org/pdf/2501.03425)
- [Quasi-static magnetic compression of FRC (arxiv 2204.07978)](https://arxiv.org/pdf/2204.07978)

### Other magneto-inertial / pulsed concepts (sourcing pending)

- **maglif: 0.15** — Slutz 2010 / Knapp 2019 2D LASNEX projections at full
  Z-driver. Range: 0.08-0.25. Sourcing pending.
- **mag_target: 0.10** — General Fusion plasma-compression projections,
  MIF concept literature. Range: 0.05-0.15. Sourcing pending.
- **plasma_jet: 0.10** — PJMIF MHD sims (Witherspoon, Hsu). Range:
  0.05-0.15. Sourcing pending.
- **staged_zpinch: 0.10** — LANL Rahman/Wessel staged-pinch designs.
  Range: 0.05-0.15. Sourcing pending.
- **zpinch: 0.10** — Standard Z-pinch reactor concepts; Rayleigh-Taylor
  disassembly limit. Range: 0.05-0.15. Sourcing pending.
- **theta_pinch: 0.05** — Faster expansion than Z-pinch; closer to MFE
  bound. Range: 0.03-0.08. Sourcing pending.
- **dense_plasma_focus: 0.01** — LPP Focus Fusion projections;
  sub-microsecond pinch lifetime is the binding constraint. **Genuinely
  low** — drives LCOE materially for DPF comparisons. Sourcing pending.

## fuel_recovery: 0.99 (uniform across all concepts)

NOAK fuel-cycle recycling efficiency. ITER targets ~99% for tritium and
serves as the mature reference. Within-concept architectural spread (gas-
phase exhaust vs. target-factory residue) is ≤ 5 percentage points at
NOAK; sensitivity to `fuel_recovery` is highest at low `burn_fraction`,
exactly where concepts are most architecturally similar (all gas-phase
MFE). A single uniform value is therefore defensible.

A future refinement would treat `fuel_recovery` as FOAK/NOAK-toggled
(e.g., FOAK ~0.95, NOAK 0.99). Out of scope for this iteration.
```

- [ ] **Step 2: Commit**

```bash
git add docs/account_justification/burn_fraction.md
git commit -m "Add burn_fraction account justification with Helion section sourced"
```

---

## Task A6: Phase A green checkpoint

- [ ] **Step 1: Confirm full test suite is green**

Run: `pytest tests/`

Expected: all PASS. This is the Phase A checkpoint. The codebase is now in a state where:
- Every concept YAML declares `burn_fraction` and `fuel_recovery`.
- `CostingInput` validates these as common required engineering parameters.
- The sensitivity tornado includes them.
- `CostingConstants` *still* declares them as a fallback (about to be removed in Phase B), so the merge order in `model.py:437` continues to work without any explicit ordering dependence.

If any test fails here, do **not** proceed to Phase B; diagnose and fix first.

---

# Phase B — Remove the fallback

## Task B1: Remove burn_fraction and fuel_recovery from costing_constants.yaml

**Files:**
- Modify: `src/costingfe/data/defaults/costing_constants.yaml:199-201`

- [ ] **Step 1: Delete the two lines**

In `src/costingfe/data/defaults/costing_constants.yaml`, locate the CAS80 fuel-utilization section (around lines 199-201):

```yaml
# CAS80 — fuel utilization
burn_fraction: 0.05          # Fraction of injected fuel that undergoes fusion per pass
fuel_recovery: 0.99          # Fraction of unburned fuel recovered and recycled (NOAK; mature fuel-cycle recycling, ITER tritium plants target ~99%)
```

Delete these three lines (the section header comment plus the two value lines). Preserve any blank lines that separate this section from neighbors.

- [ ] **Step 2: Verify the file still parses**

Run: `python -c "import yaml; yaml.safe_load(open('src/costingfe/data/defaults/costing_constants.yaml').read())"`

Expected: no output, no error.

---

## Task B2: Remove burn_fraction and fuel_recovery from CostingConstants

**Files:**
- Modify: `src/costingfe/defaults.py:245-251`

- [ ] **Step 1: Delete the two fields**

In `src/costingfe/defaults.py`, locate lines 245-251:

```python
    # CAS80 — fuel utilization
    burn_fraction: float = (
        0.05  # Fraction of injected fuel that undergoes fusion per pass
    )
    fuel_recovery: float = (
        0.99  # Fraction of unburned fuel recovered and recycled (NOAK default)
    )
```

Delete all seven lines (the section comment plus the two field definitions).

- [ ] **Step 2: Run the full test suite**

Run: `pytest tests/`

Expected: all PASS. Concept YAMLs supply the values; `_COMMON_REQUIRED` ensures they're present; the merge loop in `model.py:437` (`for name in cc_float_fields(): params.setdefault(...)`) no longer touches these two keys because they're no longer CC fields.

- [ ] **Step 3: Commit B1 and B2 together**

```bash
git add src/costingfe/data/defaults/costing_constants.yaml src/costingfe/defaults.py
git commit -m "Remove burn_fraction and fuel_recovery from CostingConstants; per-concept YAML is now sole source"
```

---

## Task B3: Update _VALIDATION_PHYSICS comment

**Files:**
- Modify: `src/costingfe/validation.py:43-64` (specifically the `dhe3_f_He3` line)

- [ ] **Step 1: Update the comment**

In `src/costingfe/validation.py`, locate line 53:

```python
    dhe3_f_He3=0.84,  # bf=0.05, fr=0.99 -> q = bf/(1-fr(1-bf)) ~ 0.84
```

Change the comment to reflect that bf and fr are now concept-specific:

```python
    dhe3_f_He3=0.84,  # representative MFE-class value (bf=0.05, fr=0.99); concept-specific values flow through compute()
```

The value stays at 0.84. `_VALIDATION_PHYSICS` is a synthetic dict for cross-field warning checks, not a runtime default.

- [ ] **Step 2: Run the full test suite**

Run: `pytest tests/`

Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add src/costingfe/validation.py
git commit -m "Update _VALIDATION_PHYSICS comment: bf/fr are concept-specific"
```

---

## Task B4: Phase B green checkpoint

- [ ] **Step 1: Confirm full test suite is green**

Run: `pytest tests/`

Expected: all PASS.

- [ ] **Step 2: Sanity-check the adapter for two non-MFE concepts**

Run:

```bash
python -c "
from costingfe.adapter import FusionTeaInput, run_costing
for concept in ('laser_ife', 'pulsed_frc', 'dense_plasma_focus'):
    inp = FusionTeaInput(concept=concept, fuel='dt', net_electric_mw=1000.0)
    out = run_costing(inp)
    print(f'{concept}: LCOE = {out.lcoe:.2f} \$/MWh')
"
```

Expected: three LCOE values printed without errors. Compare the `pulsed_frc` and `dense_plasma_focus` numbers against what they were at the start of the session (a pre-change comparison should be captured by running this same snippet on the branch's merge-base before starting Phase A; if not captured, just verify the three values are sensible — DPF should be noticeably higher than the others because its burn_fraction=0.01).

If any concept errors out with "Missing required engineering parameters: burn_fraction" or similar, Task A2 missed that YAML — go back and fix.

---

# Phase C — Paper text and companion scripts

## Task C1: Update the paper footnote about MFE-centric NOAK defaults

**Files:**
- Modify: `docs/papers/1costingfe_paper/1costingfe_paper.tex` (the MFE-centric NOAK-default footnote)

- [ ] **Step 1: Locate the footnote**

Run: `grep -n "MFE-centric\|MFE centric" docs/papers/1costingfe_paper/1costingfe_paper.tex`

Expected: a hit around line 822 (line number may have drifted slightly; the content is the binding reference).

- [ ] **Step 2: Read the surrounding context**

Read the paragraph containing "MFE-centric" (typically a 5-10 line span). The current text reads approximately:

```latex
... The NOAK defaults are $f_b = 0.05$ and $f_r = 0.99$,
which are MFE-centric: pulsed-compression and ICF concepts achieve much
higher single-pass burnup (10--40\%+), and recovery efficiency depends on
how mature the plant's fuel-cycle infrastructure is. Concept-specific
values should be supplied for those families when concept-resolved
comparisons are made.
```

- [ ] **Step 3: Replace with concept-specific framing**

Replace the existing sentences (from "The NOAK defaults are" through "concept-resolved comparisons are made.") with:

```latex
Both $f_b$ and $f_r$ are now declared per-concept in the engineering
YAMLs rather than as global defaults. Steady-state MFE concepts share
a regime default $f_b = 0.05$; inertial concepts use 0.25--0.30; the
heterogeneous magneto-inertial bucket uses concept-specific values
ranging from 0.01 (dense plasma focus) to 0.30 (heavy-ion fusion). The
$f_r = 0.99$ NOAK value is uniform across concepts but declared per-
concept by policy. Per-concept values and provenance are documented in
\texttt{docs/account\_justification/burn\_fraction.md}.
```

- [ ] **Step 4: Recompile the paper**

Run: `cd docs/papers/1costingfe_paper && pdflatex -interaction=nonstopmode -halt-on-error 1costingfe_paper.tex 2>&1 | grep -E "Error|undefined" | head -10`

Expected: empty output (no errors). If you see "undefined reference," the LaTeX cross-references resolve on the second pass — rerun the same command once.

- [ ] **Step 5: Commit**

```bash
git add docs/papers/1costingfe_paper/1costingfe_paper.tex
git commit -m "Paper: replace MFE-centric NOAK-default footnote with per-concept framing"
```

---

## Task C2: Re-run paper benchmark scripts

**Files:**
- Run-only: `docs/papers/1costingfe_paper/scripts/benchmark_arc.py`
- Run-only: `docs/papers/1costingfe_paper/scripts/benchmark_aries_at.py`

Both target D-T tokamaks at the standard MFE burn_fraction (0.05), so the post-move numbers should be **identical** to the pre-move numbers. This task confirms that.

- [ ] **Step 1: Run benchmark_arc.py**

Run: `python docs/papers/1costingfe_paper/scripts/benchmark_arc.py`

Expected: completes without error; LCOE / capex outputs match the version on the previous commit. If the script writes outputs to disk, `git diff` against `HEAD~N` to confirm no drift.

- [ ] **Step 2: Run benchmark_aries_at.py**

Run: `python docs/papers/1costingfe_paper/scripts/benchmark_aries_at.py`

Expected: same as above — completes cleanly, outputs match prior values.

- [ ] **Step 3: If any output drift is observed**

Investigate first; the only legitimate source of drift is if the benchmark script picks up a concept-specific burn_fraction it didn't pick up before. ARC and ARIES-AT are both tokamak D-T, so this should not happen.

- [ ] **Step 4: Commit any regenerated outputs (only if they changed)**

```bash
# Only if outputs changed legitimately:
git add docs/papers/1costingfe_paper/<changed-outputs>
git commit -m "Refresh ARC and ARIES-AT benchmark outputs after per-concept burn_fraction"
```

---

## Task C3: Re-run blog / companion scripts touching non-MFE concepts

**Files:**
- Run-only: any `docs/blog/*/companion-*.py` or `docs/papers/1costingfe_paper/scripts/*.py` script that uses a non-MFE concept.

- [ ] **Step 1: Find candidate scripts**

Run: `grep -rln "laser_ife\|heavy_ion\|maglif\|pulsed_frc\|dense_plasma_focus\|mag_target\|plasma_jet\|staged_zpinch\|zpinch\|theta_pinch" docs/blog/ docs/papers/1costingfe_paper/scripts/`

Expected: a list of scripts touching pulsed/inertial concepts. Note each one.

- [ ] **Step 2: For each script, run it and compare outputs**

For each script in the list from Step 1:

```bash
python <script-path>
```

If the script writes files (CSVs, PNGs, JSON), run `git diff` on the outputs and document any drift. Drift in LCOE for non-MFE concepts is **expected** when the previous global default 0.05 differed from the new concept-specific value — that's the entire point of this work. The verification is that the drift is in the *direction* and *magnitude* the spec predicts (e.g., LCOE down for high-burnup ICF, up for DPF).

- [ ] **Step 3: Document any LCOE drift in the commit message**

If outputs change, commit them with a message that names which concepts moved and roughly by how much:

```bash
git add docs/blog/<paths> docs/papers/1costingfe_paper/<paths>
git commit -m "Refresh companion script outputs after per-concept burn_fraction (LCOE drift: <summary>)"
```

If no script touches non-MFE concepts (or all touch them only via hardcoded `burn_fraction=` kwargs which override the YAML), no commit is needed and no drift is expected.

---

## Task C4: Final green checkpoint

- [ ] **Step 1: Run the full test suite one more time**

Run: `pytest tests/`

Expected: all PASS.

- [ ] **Step 2: Verify the paper still compiles cleanly**

Run: `cd docs/papers/1costingfe_paper && pdflatex -interaction=nonstopmode -halt-on-error 1costingfe_paper.tex 2>&1 | grep -E "Error|undefined" | head -10`

Expected: empty output.

- [ ] **Step 3: Review the commit log**

Run: `git log --oneline -20`

Expected: roughly 6-9 new commits from this branch's work:
1. Failing test for per-concept YAML declaration (Task A1)
2. Declare burn_fraction and fuel_recovery in every concept YAML (Task A2)
3. Validate as common required engineering parameters (Task A3)
4. Sensitivity tornado registration (Task A4)
5. Account justification doc (Task A5)
6. Remove from CostingConstants (Tasks B1 + B2)
7. _VALIDATION_PHYSICS comment (Task B3)
8. Paper footnote (Task C1)
9. Refreshed companion outputs (Task C3) — only if drift was observed

The plan is complete.

---

# Out of scope (do not do these in this branch)

- Per-concept `fuel_recovery` values driven by exhaust architecture.
- FOAK/NOAK interaction with `fuel_recovery`.
- Cleanup of hardcoded `burn_fraction=` / `fuel_recovery=` kwargs in `examples/dhe3_pulsed_frc.py` and `examples/path_to_1cent.py` (their kwargs continue to override YAML, so example outputs are unchanged).
- Expanding `docs/account_justification/burn_fraction.md` with sourcing for the 14 concepts other than `pulsed_frc` — separate research task.
