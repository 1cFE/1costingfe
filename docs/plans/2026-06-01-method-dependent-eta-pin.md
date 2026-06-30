# Method-Dependent Heating Efficiency (eta_pin = source x coupling) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the heating wall-plug efficiency `eta_pin` depend on the heating method, by decomposing it into a per-method source efficiency and a per-concept coupling efficiency: `eta_pin = eta_source(method) x eta_couple(concept)`.

**Architecture:** `eta_source_{nbi,icrf,ecrh,lhcd}` are global constants (wall-plug to delivered power). `eta_couple` is a per-concept YAML parameter (delivered to plasma-absorbed). The effective `eta_pin` for a heating mix is `sum_i p_i / sum_i (p_i / (eta_source_i * eta_couple))`, computed once in `CostModel._power_balance` and injected so all power-balance call sites pick it up. Concepts without an NBI/RF method (electrostatic orbitron/polywell; all pulsed drivers) keep an explicit `eta_pin`. Per-concept `eta_couple` values are chosen to reproduce each benchmarked concept's current `eta_pin` exactly, so the change is regression-neutral; only the FRC is anchored to source data (TAE C-2W: 0.60 x 0.43 = 0.26).

**Tech Stack:** Python, JAX (power balance is traced for AD/uncertainty), Pydantic (validation), PyYAML (concept defaults), pytest.

---

## Design Tables

**eta_source (global, method-level, wall-plug to delivered):**

| Method | eta_source | Basis |
|---|---|---|
| NBI | 0.60 | negative-ion source wall-plug (OSTI 2441289 / ITER NBI) |
| ICRF | 0.70 | RF tetrode transmitter (ITER ICRF final-amplifier ~70%) |
| ECRH | 0.50 | gyrotron wall-plug |
| LHCD | 0.50 | klystron wall-plug |

**eta_couple (per concept, delivered to plasma-absorbed) — chosen to preserve current eta_pin:**

| Concept | Method | Current eta_pin | eta_couple | eta_source x eta_couple |
|---|---|---|---|---|
| tokamak | NBI | 0.50 | 0.8333 | 0.60 x 0.8333 = 0.50 |
| mirror | NBI | 0.50 | 0.8333 | 0.50 |
| stellarator | ECRH | 0.50 | 1.0 | 0.50 x 1.0 = 0.50 |
| dipole | ICRF | 0.70 | 1.0 | 0.70 x 1.0 = 0.70 |
| steady FRC | NBI | 0.26 | 0.43 | 0.60 x 0.43 = 0.258 (sourced, not back-fit) |

Electrostatic (orbitron `eta_pin=0.80`, polywell `eta_pin=0.70`) and all pulsed concepts keep explicit `eta_pin`, no `eta_couple`.

**Baseline LCOEs to preserve (200 MWe net, NOAK):** tokamak 233.41, mirror 186.47, stellarator 359.94, dipole 182.79, polywell 52.98.

**Known simplification (document, do not implement now):** `eta_couple` is a single per-concept value, so within one device all methods share the coupling and the method distinction comes only through `eta_source`. A PFRC-style FRC (RF/RMF, better coupling ~0.60) is handled by overriding `eta_couple` alongside the RF driver, not automatically. Per-(concept, method) coupling is a noted future refinement.

---

## File Structure

- `src/costingfe/defaults.py` — add 4 `eta_source_*` fields to `CostingConstants`.
- `src/costingfe/data/defaults/costing_constants.yaml` — add 4 `eta_source_*` values.
- `src/costingfe/validation.py` — add `eta_couple` field; drop `eta_pin` from `_MFE_REQUIRED`; add eta_pin-or-eta_couple check.
- `src/costingfe/model.py` — add `_effective_eta_pin`; inject in `_power_balance`.
- `src/costingfe/data/defaults/steady_state_{tokamak,mirror,stellarator,dipole,steady_frc}.yaml` — replace `eta_pin` with `eta_couple`.
- `docs/papers/1costingfe_paper/1costingfe_paper.tex` — document in the Steady-State Power Balance subsection (Physics Module section).
- `tests/test_eta_pin_coupling.py` — new regression + behavior test.

---

### Task 1: Add eta_source constants

**Files:**
- Modify: `src/costingfe/defaults.py` (CostingConstants, after `heating_lhcd_per_mw`)
- Modify: `src/costingfe/data/defaults/costing_constants.yaml` (after the heating per-MW block)

- [ ] **Step 1: Add fields to CostingConstants dataclass**

In `src/costingfe/defaults.py`, immediately after the line `heating_lhcd_per_mw: float = 4.0  # Lower Hybrid Current Drive (klystrons)` add:

```python
    # Heating wall-plug source efficiency by method (wall-plug -> delivered
    # power, before plasma coupling). Combined with a per-concept eta_couple
    # (in the concept YAML) to form eta_pin = eta_source x eta_couple.
    # See docs/account_justification/... and the Physics Module of the paper.
    eta_source_nbi: float = 0.60  # negative-ion NBI source (OSTI 2441289 / ITER)
    eta_source_icrf: float = 0.70  # RF tetrode transmitter (ITER ICRF ~70%)
    eta_source_ecrh: float = 0.50  # gyrotron wall-plug
    eta_source_lhcd: float = 0.50  # klystron wall-plug
```

- [ ] **Step 2: Add values to costing_constants.yaml**

In `src/costingfe/data/defaults/costing_constants.yaml`, immediately after `heating_lhcd_per_mw: 4.0  # Lower Hybrid Current Drive (klystrons)` add:

```yaml
# Heating source efficiency by method (wall-plug -> delivered). Combined with
# per-concept eta_couple (concept YAML) as eta_pin = eta_source x eta_couple.
eta_source_nbi: 0.60         # negative-ion NBI source (OSTI 2441289 / ITER)
eta_source_icrf: 0.70        # RF tetrode transmitter (ITER ICRF ~70%)
eta_source_ecrh: 0.50        # gyrotron wall-plug
eta_source_lhcd: 0.50        # klystron wall-plug
```

(Exact anchor line text may differ slightly; place the block adjacent to the existing `heating_*_per_mw` entries.)

- [ ] **Step 3: Verify constants load**

Run: `python -c "from costingfe.defaults import load_costing_constants as L; c=L(); print(c.eta_source_nbi, c.eta_source_icrf, c.eta_source_ecrh, c.eta_source_lhcd)"`
Expected: `0.6 0.7 0.5 0.5`

- [ ] **Step 4: Commit**

```bash
git add src/costingfe/defaults.py src/costingfe/data/defaults/costing_constants.yaml
git commit -m "Add per-method heating source efficiency constants (eta_source)"
```

---

### Task 2: Add eta_couple validation field and relax eta_pin requirement

**Files:**
- Modify: `src/costingfe/validation.py:118` (add field), `:172-182` (`_MFE_REQUIRED`), `:197-234` (`check_family_required_params`)

- [ ] **Step 1: Add the eta_couple field**

In `src/costingfe/validation.py`, immediately after `eta_pin: float | None = None` (the MFE block, ~line 118) add:

```python
    eta_couple: float | None = None  # heating delivered->plasma coupling (concept)
```

- [ ] **Step 2: Drop eta_pin from _MFE_REQUIRED**

In `_MFE_REQUIRED`, remove the `"eta_pin",` entry so the list is:

```python
    _MFE_REQUIRED = [
        "p_input",
        "eta_p",
        "eta_de",
        "f_dec",
        "p_coils",
        "p_cool",
        "R0",
        "elon",
    ]
```

- [ ] **Step 3: Add eta_pin-or-eta_couple check**

In `check_family_required_params`, immediately before `# 0D model requires q95 and f_GW`, add:

```python
        # Steady-state heating efficiency: either an explicit eta_pin
        # (electrostatic / direct-injection concepts) or an eta_couple that
        # combines with the per-method eta_source to form eta_pin.
        if (
            family == ConfinementFamily.STEADY_STATE
            and self.eta_pin is None
            and self.eta_couple is None
        ):
            raise ValueError(
                "steady-state concept requires either eta_pin or eta_couple"
            )
```

- [ ] **Step 4: Run existing validation/cas22 tests (should still pass)**

Run: `python -m pytest tests/test_cas22.py tests/test_costs.py -q`
Expected: PASS (no behavior change yet; yamls still carry eta_pin)

- [ ] **Step 5: Commit**

```bash
git add src/costingfe/validation.py
git commit -m "Accept eta_couple as an alternative to eta_pin for steady-state concepts"
```

---

### Task 3: Derive eta_pin from source x coupling in the power balance

**Files:**
- Modify: `src/costingfe/model.py` — add `_effective_eta_pin` method; first line of `_power_balance` (~line 104-106)

- [ ] **Step 1: Add the _effective_eta_pin method**

In `src/costingfe/model.py`, add this method to `CostModel` immediately before `def _power_balance(self, params, n_mod):`:

```python
    def _effective_eta_pin(self, params):
        """Heating wall-plug efficiency eta_pin = eta_source(method) x eta_couple.

        For NBI/RF-heated concepts (those that set eta_couple), the effective
        eta_pin is the mix-weighted product
            eta_pin = sum_i p_i / sum_i p_i / (eta_source_i * eta_couple).
        Concepts without an NBI/RF method (electrostatic orbitron/polywell;
        pulsed drivers) keep their explicit eta_pin. An explicit eta_pin always
        wins. Dict-key checks are static (JAX-safe under AD tracing); the
        arithmetic traces cleanly.
        """
        if "eta_couple" not in params or "eta_pin" in params:
            return params["eta_pin"]
        ec = params["eta_couple"]
        cc = self.cc
        p_input = params.get("p_input", 0.0)
        p_nbi = params.get("p_nbi", p_input)
        p_icrf = params.get("p_icrf", 0.0)
        p_ecrh = params.get("p_ecrh", 0.0)
        p_lhcd = params.get("p_lhcd", 0.0)
        num = p_nbi + p_icrf + p_ecrh + p_lhcd
        den = (
            p_nbi / (cc.eta_source_nbi * ec)
            + p_icrf / (cc.eta_source_icrf * ec)
            + p_ecrh / (cc.eta_source_ecrh * ec)
            + p_lhcd / (cc.eta_source_lhcd * ec)
        )
        return num / den
```

- [ ] **Step 2: Inject the derived eta_pin at the top of _power_balance**

In `_power_balance`, immediately after the docstring and before `p_net_per_mod = params["net_electric_mw"] / n_mod`, add:

```python
        # Derive eta_pin from per-method source x per-concept coupling, so all
        # downstream power-balance sites (and the 0D branch) use the same value.
        params = {**params, "eta_pin": self._effective_eta_pin(params)}
```

- [ ] **Step 3: Verify it is a no-op while yamls still carry eta_pin**

Run: `python -c "import warnings; warnings.filterwarnings('ignore'); from costingfe import ConfinementConcept as C, CostModel, Fuel; r=CostModel(concept=C.TOKAMAK,fuel=Fuel.DT).forward(net_electric_mw=200.0,availability=0.85,lifetime_yr=30,n_mod=1,construction_time_yr=6.0,interest_rate=0.07,inflation_rate=0.02,noak=True); print(round(float(r.costs.lcoe),2))"`
Expected: `233.41` (yaml still has eta_pin, so `_effective_eta_pin` returns it unchanged)

- [ ] **Step 4: Run full suite (no behavior change yet)**

Run: `python -m pytest tests/ -q`
Expected: PASS (356)

- [ ] **Step 5: Commit**

```bash
git add src/costingfe/model.py
git commit -m "Derive eta_pin from eta_source x eta_couple in the power balance"
```

---

### Task 4: Switch the 5 heated MFE concepts to eta_couple

**Files:**
- Modify: `src/costingfe/data/defaults/steady_state_tokamak.yaml`
- Modify: `src/costingfe/data/defaults/steady_state_mirror.yaml`
- Modify: `src/costingfe/data/defaults/steady_state_stellarator.yaml`
- Modify: `src/costingfe/data/defaults/steady_state_dipole.yaml`
- Modify: `src/costingfe/data/defaults/steady_state_steady_frc.yaml`

- [ ] **Step 1: Replace eta_pin with eta_couple in each heated MFE yaml**

In each file, replace the `eta_pin: <value> ...` line with the `eta_couple` line below (delete the `eta_pin` line entirely; electrostatic and pulsed concepts are untouched):

- `steady_state_tokamak.yaml`: `eta_couple: 0.8333   # NBI coupling; eta_pin = 0.60 x 0.8333 = 0.50`
- `steady_state_mirror.yaml`: `eta_couple: 0.8333   # NBI coupling; eta_pin = 0.60 x 0.8333 = 0.50`
- `steady_state_stellarator.yaml`: `eta_couple: 1.0      # ECRH coupling; eta_pin = 0.50 x 1.0 = 0.50`
- `steady_state_dipole.yaml`: `eta_couple: 1.0      # ICRF coupling; eta_pin = 0.70 x 1.0 = 0.70`
- `steady_state_steady_frc.yaml`: replace the `eta_pin: 0.26 ...` block with:
  ```yaml
  eta_couple: 0.43    # NBI delivered->plasma coupling at C-2W (shine-through +
                      # tangential-duct losses; OSTI 2441289). eta_pin = eta_source_nbi
                      # 0.60 x 0.43 = 0.26. A PFRC-style RF FRC overrides p_icrf and
                      # eta_couple (RMF coupling ~0.60).
  ```

- [ ] **Step 2: Verify benchmark LCOEs are unchanged**

Run:
```bash
python - <<'PY'
import warnings; warnings.filterwarnings("ignore")
from costingfe import ConfinementConcept as C, CostModel, Fuel
exp={"tokamak":233.41,"mirror":186.47,"stellarator":359.94,"dipole":182.79}
cs={"tokamak":(C.TOKAMAK,Fuel.DT),"mirror":(C.MIRROR,Fuel.DT),
    "stellarator":(C.STELLARATOR,Fuel.DT),"dipole":(C.DIPOLE,Fuel.DHE3)}
for n,(c,f) in cs.items():
    r=CostModel(concept=c,fuel=f).forward(net_electric_mw=200.0,availability=0.85,
        lifetime_yr=30,n_mod=1,construction_time_yr=6.0,interest_rate=0.07,
        inflation_rate=0.02,noak=True)
    got=round(float(r.costs.lcoe),2); ok=abs(got-exp[n])<0.05
    print(f"{n:<12} {got}  expect {exp[n]}  {'OK' if ok else 'MISMATCH'}")
PY
```
Expected: all four `OK`.

- [ ] **Step 3: Verify the FRC is unchanged and the RF override raises eta_pin**

Run:
```bash
python - <<'PY'
import warnings; warnings.filterwarnings("ignore")
from costingfe import ConfinementConcept as C, CostModel, Fuel
m=CostModel(concept=C.STEADY_FRC,fuel=Fuel.PB11)
base=dict(net_electric_mw=105.0,availability=0.85,lifetime_yr=30,n_mod=1,
          construction_time_yr=6.0,interest_rate=0.07,inflation_rate=0.02,noak=True)
nbi=m.forward(**base)
rf =m.forward(p_nbi=0.0,p_icrf=26.0,p_ecrh=0.0,p_lhcd=0.0,eta_couple=0.60,**base)
print("FRC NBI rec_frac:",f"{float(nbi.power_table.rec_frac):.1%}")
print("FRC RF  rec_frac:",f"{float(rf.power_table.rec_frac):.1%}  (should be lower)")
assert float(rf.power_table.rec_frac) < float(nbi.power_table.rec_frac)
print("OK")
PY
```
Expected: RF rec_frac < NBI rec_frac, prints `OK`.

- [ ] **Step 4: Run full suite**

Run: `python -m pytest tests/ -q`
Expected: PASS (356)

- [ ] **Step 5: Commit**

```bash
git add src/costingfe/data/defaults/steady_state_tokamak.yaml \
        src/costingfe/data/defaults/steady_state_mirror.yaml \
        src/costingfe/data/defaults/steady_state_stellarator.yaml \
        src/costingfe/data/defaults/steady_state_dipole.yaml \
        src/costingfe/data/defaults/steady_state_steady_frc.yaml
git commit -m "Express heated MFE concepts via eta_couple (eta_pin = source x coupling)"
```

---

### Task 5: Regression + behavior test

**Files:**
- Create: `tests/test_eta_pin_coupling.py`

- [ ] **Step 1: Write the test**

```python
"""eta_pin = eta_source(method) x eta_couple(concept): regression + behavior."""
import warnings
import pytest
from costingfe import ConfinementConcept as C, CostModel, Fuel

warnings.filterwarnings("ignore")

_BASE = dict(availability=0.85, lifetime_yr=30, n_mod=1, construction_time_yr=6.0,
             interest_rate=0.07, inflation_rate=0.02, noak=True)


@pytest.mark.parametrize("concept,fuel,expected", [
    (C.TOKAMAK, Fuel.DT, 233.41),
    (C.MIRROR, Fuel.DT, 186.47),
    (C.STELLARATOR, Fuel.DT, 359.94),
    (C.DIPOLE, Fuel.DHE3, 182.79),
    (C.POLYWELL, Fuel.PB11, 52.98),
])
def test_benchmark_lcoe_preserved(concept, fuel, expected):
    """Coupling factors are chosen to reproduce each concept's prior eta_pin."""
    r = CostModel(concept=concept, fuel=fuel).forward(net_electric_mw=200.0, **_BASE)
    assert float(r.costs.lcoe) == pytest.approx(expected, abs=0.05)


def test_frc_rf_driver_couples_better_than_nbi():
    """Swapping the FRC driver to RF (with RMF coupling) cuts recirculating power."""
    m = CostModel(concept=C.STEADY_FRC, fuel=Fuel.PB11)
    nbi = m.forward(net_electric_mw=105.0, **_BASE)
    rf = m.forward(net_electric_mw=105.0, p_nbi=0.0, p_icrf=26.0, p_ecrh=0.0,
                   p_lhcd=0.0, eta_couple=0.60, **_BASE)
    assert float(rf.power_table.rec_frac) < float(nbi.power_table.rec_frac)


def test_explicit_eta_pin_override_wins():
    """An explicit eta_pin override bypasses the source x coupling derivation."""
    m = CostModel(concept=C.TOKAMAK, fuel=Fuel.DT)
    a = m.forward(net_electric_mw=200.0, **_BASE)
    b = m.forward(net_electric_mw=200.0, eta_pin=0.30, **_BASE)
    assert float(b.power_table.rec_frac) > float(a.power_table.rec_frac)
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_eta_pin_coupling.py -v`
Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_eta_pin_coupling.py
git commit -m "Test eta_pin source x coupling: benchmark preservation + FRC driver swap"
```

---

### Task 6: Document in the paper Physics Module

**Files:**
- Modify: `docs/papers/1costingfe_paper/1costingfe_paper.tex` (Steady-State Power Balance subsection, `\subsection{Steady-State Power Balance}` near line 1023)

- [ ] **Step 1: Add a paragraph + equation on method-dependent eta_pin**

Find the point in `\subsection{Steady-State Power Balance}` where the recirculating heating term `P_{\text{in}}/\eta_{\text{pin}}` is introduced, and add immediately after it:

```latex
The heating wall-plug efficiency is not a single device constant but a product
of a method-level source efficiency and a device-level coupling efficiency,
\begin{equation}
  \eta_{\text{pin}} = \frac{\sum_i P_i}{\sum_i P_i / \bigl(\eta_{\text{src}}(i)\,\eta_{\text{cpl}}\bigr)},
\end{equation}
where $i$ runs over the heating methods (NBI, ICRF, ECRH, LHCD) in the mix,
$\eta_{\text{src}}(i)$ is the wall-plug-to-delivered source efficiency of method
$i$ (negative-ion NBI $0.60$, ICRF $0.70$, ECRH $0.50$, LHCD $0.50$), and
$\eta_{\text{cpl}}$ is the concept's delivered-to-absorbed coupling. This makes
the recirculating heating load track the actual driver: a beam-driven FRC,
whose tangential injection through long ducts and short plasma path give a
coupling of only $\eta_{\text{cpl}} \approx 0.43$ (C-2W), lands at
$\eta_{\text{pin}} = 0.60 \times 0.43 = 0.26$, whereas an RF-driven device
couples far better. Concepts whose input power is not delivered by an NBI/RF
heating system (electrostatic confinement; pulsed drivers) specify
$\eta_{\text{pin}}$ directly. The per-concept $\eta_{\text{cpl}}$ values are
chosen so the product reproduces each benchmarked concept's calibrated
$\eta_{\text{pin}}$, leaving the benchmarks unchanged while making the
method-dependence explicit.
```

- [ ] **Step 2: Sanity-check LaTeX (optional local build)**

If a LaTeX toolchain is available: build the paper and confirm no errors around the new equation. Otherwise verify braces/`align` balance by inspection.

- [ ] **Step 3: Commit**

```bash
git add docs/papers/1costingfe_paper/1costingfe_paper.tex
git commit -m "Document method-dependent eta_pin (source x coupling) in the physics module"
```

---

## Self-Review Notes

- **Spec coverage:** source constants (T1), validation for the new param (T2), the derivation + injection (T3), the per-concept switch with benchmark proof (T4), tests (T5), paper docs (T6). All covered.
- **JAX safety:** `_effective_eta_pin` branches only on dict-key presence (static at trace time); the arithmetic is plain and traces. No `isinstance`/concrete guard needed, unlike the heating-split normalization (which compares magnitudes).
- **Override semantics:** an explicit `eta_pin` (in `params`) short-circuits the derivation (T3 method first line), so users can still force a value; verified by `test_explicit_eta_pin_override_wins`.
- **Type/name consistency:** `eta_couple` and `eta_source_{nbi,icrf,ecrh,lhcd}` are spelled identically across constants, validation, model, and yamls.
- **Out of scope (documented):** per-(concept, method) coupling; the FRC RF case is handled by override, not automation.
</content>
