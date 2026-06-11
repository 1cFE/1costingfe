# 1costingfe-explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A live forward-costing explorer on Vercel: pick concept + fuel, drag sliders, see LCOE, capex, CAS breakdown, and an elasticity tornado computed live by a numpy-only port of costingfe frozen at `4c46468`.

**Architecture:** New repo `1costingfe-explorer` containing (1) a numpy-only copy of `costingfe` stripped from the freeze tag, (2) a FastAPI backend served as a single Vercel Python function, (3) a Vite/React/Tailwind frontend with Zustand state and Recharts charts. The numpy strip is validated against the JAX original via a reference-JSON parity test.

**Tech Stack:** Python 3.10+, numpy, FastAPI, pydantic v2, React 18, TypeScript, Vite 5, Zustand, Recharts, Tailwind 3, Vercel.

**Hard constraint:** A parallel session owns the `1costingfe` working tree. NEVER switch branches, modify tracked files, or commit in `/mnt/c/Users/talru/1cfe/1costingfe`. Allowed there: creating a tag, creating a `git worktree` (separate directory, does not affect the main checkout).

---

### Task 1: Freeze tag + worktree

**Files:** none created in 1costingfe (tag + worktree only)

- [ ] **Step 1: Create and push the tag**

```bash
cd /mnt/c/Users/talru/1cfe/1costingfe
git tag v0.1.0-alpha.1 4c46468
git push origin v0.1.0-alpha.1
```

Expected: tag visible in `git tag -l 'v0.1.0*'` and on origin.

- [ ] **Step 2: Create a worktree of the freeze commit**

```bash
git worktree add /mnt/c/Users/talru/1cfe/1costingfe-freeze v0.1.0-alpha.1
```

Expected: `/mnt/c/Users/talru/1cfe/1costingfe-freeze` exists in detached-HEAD state at `4c46468`. The other session's checkout is untouched (`git -C /mnt/c/Users/talru/1cfe/1costingfe status` shows no change to its current branch).

- [ ] **Step 3: Install the freeze worktree env (JAX version, used for parity reference + blog numbers)**

```bash
cd /mnt/c/Users/talru/1cfe/1costingfe-freeze
uv sync --all-extras
uv run python -c "from costingfe import CostModel, ConfinementConcept, Fuel; m = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT); r = m.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30); print(round(float(r.costs.lcoe), 2))"
```

Expected: prints a finite LCOE (USD/MWh). Record this value — it is the parity anchor.

### Task 2: Scaffold the new repo

**Files:**
- Create: `/mnt/c/Users/talru/1cfe/1costingfe-explorer/` (git init)
- Create: `vercel.json`, `api/index.py`, `requirements.txt`, `pyproject.toml`, `.gitignore`, `README.md`

- [ ] **Step 1: Init repo and write config files**

```bash
mkdir -p /mnt/c/Users/talru/1cfe/1costingfe-explorer
cd /mnt/c/Users/talru/1cfe/1costingfe-explorer
git init -b master
mkdir -p api backend/routes backend/services backend/data tests scripts
```

`vercel.json` (lifted from fusion-backcasting, proven pattern):

```json
{
  "framework": null,
  "buildCommand": "cd frontend && npm install && npm run build",
  "outputDirectory": "frontend/dist",
  "functions": {
    "api/index.py": {
      "excludeFiles": "frontend/**"
    }
  },
  "rewrites": [
    { "source": "/api/(.*)", "destination": "/api/index.py" }
  ]
}
```

`api/index.py`:

```python
"""Vercel serverless entrypoint — re-exports the FastAPI app."""

from backend.main import app  # noqa: F401
```

`requirements.txt` (what the Vercel function installs):

```
fastapi>=0.109.0
pydantic>=2.5.0
numpy>=1.26.0
pyyaml>=6.0
```

`pyproject.toml` (local dev env via uv; mirrors requirements.txt plus dev tools):

```toml
[project]
name = "costingfe-explorer"
version = "0.1.0"
description = "Live forward-costing explorer for 1costingFE (numpy-only port)"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "pydantic>=2.5.0",
    "numpy>=1.26.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "httpx>=0.27"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`.gitignore`:

```
__pycache__/
*.pyc
.venv/
node_modules/
frontend/dist/
.vercel/
```

`README.md`:

```markdown
# 1costingfe-explorer

Live forward-costing explorer for [1costingFE](https://github.com/1cfe/1costingfe),
frozen at tag `v0.1.0-alpha.1`. The embedded `costingfe/` package is a
numpy-only port of that tag (no JAX); sensitivities use central finite
differences. Parity with the JAX original is enforced by
`tests/test_parity.py` against `tests/parity_reference.json`.

## Local dev

Backend: `uv sync --all-extras && uv run uvicorn backend.main:app --reload`
Frontend: `cd frontend && npm install && npm run dev` (proxies /api to :8000)

## Deploy

Vercel project; `vercel --prod`. Build config in `vercel.json`.
```

- [ ] **Step 2: Commit**

```bash
git add -A && git commit -m "Scaffold explorer repo: Vercel function entrypoint, FastAPI deps, README"
```

### Task 3: Embed and strip costingfe (numpy-only port)

**Files:**
- Create: `costingfe/` (copied from freeze worktree `src/costingfe/`, then edited)
- Modify: `costingfe/model.py`, `costingfe/layers/tokamak.py`, `costingfe/layers/cas22.py`, `costingfe/layers/{physics,costs,radiation,economics}.py`, `costingfe/__init__.py`

The strip recipe is proven: fusion-backcasting's embedded copy used exactly this approach (alias `numpy as jnp`, FD sensitivity, Python loop for bisection).

- [ ] **Step 1: Copy the package from the freeze worktree**

```bash
cp -r /mnt/c/Users/talru/1cfe/1costingfe-freeze/src/costingfe /mnt/c/Users/talru/1cfe/1costingfe-explorer/costingfe
```

- [ ] **Step 2: Alias numpy as jnp in the five mechanical files**

In `costingfe/layers/physics.py`, `costingfe/layers/costs.py`, `costingfe/layers/radiation.py`, `costingfe/layers/economics.py` replace the line

```python
import jax.numpy as jnp
```

with

```python
import numpy as jnp  # numpy-only port: jnp is an alias for numpy
```

All `jnp.*` calls used in these files (`exp, sqrt, where, maximum, minimum, abs, ceil, logical_and, pi`) exist in numpy under the same names — no other edits.

- [ ] **Step 3: Port tokamak.py (fori_loop bisection)**

In `costingfe/layers/tokamak.py`: same `import jax.numpy as jnp` → `import numpy as jnp` swap, then delete this block (currently right after `_find_T_for_pfus`):

```python
def _import_fori_loop():
    import jax.lax

    return jax.lax.fori_loop


# Use jax.lax.fori_loop but import lazily to avoid issues
try:
    import jax.lax

    jax_fori_loop = jax.lax.fori_loop
except Exception:
    jax_fori_loop = None
```

and replace it with:

```python
def jax_fori_loop(lo, hi, body, state):
    """numpy-only port of jax.lax.fori_loop: plain Python loop."""
    for i in range(lo, hi):
        state = body(i, state)
    return state
```

The `_find_T_for_pfus` bisection body is purely functional (`jnp.where` on scalars works in numpy), so it needs no change.

- [ ] **Step 4: Port model.py**

Four edits:

1. Header: delete `import jax`, change `import jax.numpy as jnp` to `import numpy as jnp`.
2. Line ~556 (`_tracing` detection): replace

```python
        _tracing = any(isinstance(v, jax.core.Tracer) for v in params.values())
```

with

```python
        _tracing = False  # numpy-only port: values are always concrete
```

3. `sensitivity()`: keep `_build_lcoe_fn` untouched (it is jax-free except `jnp.array`, now numpy). Replace the gradient block

```python
        lcoe_fn, keys, base_vals = self._build_lcoe_fn(params, cost_overrides)
        base_lcoe = float(lcoe_fn(base_vals))

        grad_fn = jax.grad(lcoe_fn)
        grads = grad_fn(base_vals)

        engineering = {}
        financial = {}
        costing = {}
        for i, key in enumerate(keys):
            p = float(base_vals[i])
            dLCOE_dp = float(grads[i])
```

with central finite differences (lifted from fusion-backcasting's proven port) plus an `include_costing` switch the backend uses to skip the ~100 costing-constant keys per request:

```python
        lcoe_fn, keys, base_vals = self._build_lcoe_fn(params, cost_overrides)
        base_lcoe = float(lcoe_fn(base_vals))

        engineering = {}
        financial = {}
        costing = {}
        h = 1e-5  # relative perturbation for central differences
        for i, key in enumerate(keys):
            if not include_costing and key in self._COSTING_KEYS:
                continue
            p = float(base_vals[i])
            if abs(p) < 1e-12:
                continue
            dp = abs(p) * h
            x_plus = base_vals.copy()
            x_minus = base_vals.copy()
            x_plus[i] = p + dp
            x_minus[i] = p - dp
            dLCOE_dp = (float(lcoe_fn(x_plus)) - float(lcoe_fn(x_minus))) / (2 * dp)
```

and change the signature and docstring line accordingly:

```python
    def sensitivity(
        self,
        params: dict,
        cost_overrides: dict[str, float] | None = None,
        include_costing: bool = True,
    ) -> dict[str, dict[str, float]]:
```

Docstring: replace "Uses jax.grad for exact autodiff gradients." with "Uses central finite differences (numpy-only port)."

4. `batch_lcoe()`: replace the vmap block

```python
        n = len(next(iter(param_sets.values())))
        # Build matrix: each row is a param vector
        rows = []
        for _ in range(n):
            rows.append(base_vals)
        batch = jnp.stack(rows)

        # Override the varying params
        for param_name, values in param_sets.items():
            if param_name in keys:
                idx = keys.index(param_name)
                batch = batch.at[:, idx].set(jnp.array(values))

        vmapped = jax.vmap(lcoe_fn)
        results = vmapped(batch)
        return [float(r) for r in results]
```

with a plain loop:

```python
        n = len(next(iter(param_sets.values())))
        results = []
        for row_i in range(n):
            x = base_vals.copy()
            for param_name, values in param_sets.items():
                if param_name in keys:
                    x[keys.index(param_name)] = values[row_i]
            results.append(float(lcoe_fn(x)))
        return results
```

- [ ] **Step 5: Port cas22.py and __init__.py**

`costingfe/layers/cas22.py`: delete `import jax`, swap `import jax.numpy as jnp` → `import numpy as jnp`, and replace the Tracer guard near line 296 — find the `isinstance(..., jax.core.Tracer)` expression and replace the whole check with `False` (the surrounding `if`/`where` logic stays; with concrete numpy values the Python branch is always valid).

`costingfe/__init__.py`: delete the JAX_PLATFORMS block at the top:

```python
import os as _os

# Default to CPU — suppresses "NVIDIA GPU may be present" warning.
# Users with CUDA-enabled jaxlib can set JAX_PLATFORMS=cuda to override.
_os.environ.setdefault("JAX_PLATFORMS", "cpu")
```

- [ ] **Step 6: Verify no jax references remain, smoke-test**

```bash
cd /mnt/c/Users/talru/1cfe/1costingfe-explorer
grep -rn "jax" costingfe/ --include="*.py" | grep -v "numpy as jnp" | grep -v "numpy-only" | grep -v "jax_fori_loop"
```

Expected: only comment/docstring mentions, no imports. Then:

```bash
uv sync --all-extras
uv run python -c "
from costingfe import CostModel, ConfinementConcept, Fuel
m = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
r = m.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
print(round(float(r.costs.lcoe), 2))
s = m.sensitivity(r.params, include_costing=False)
print(len(s['engineering']), len(s['financial']), len(s['costing']))
"
```

Expected: an LCOE close to the Task 1 anchor (within ~0.5%, float32 vs float64), and `costing` count is 0 with `include_costing=False`.

- [ ] **Step 7: Commit**

```bash
git add costingfe && git commit -m "Embed numpy-only costingfe stripped from 1costingfe v0.1.0-alpha.1"
```

### Task 4: Parity tests against the JAX original

**Files:**
- Create: `scripts/dump_parity_reference.py`
- Create: `tests/parity_reference.json` (generated)
- Create: `tests/test_parity.py`

- [ ] **Step 1: Write the reference dump script**

`scripts/dump_parity_reference.py` — run with the FREEZE worktree env (JAX version). It enumerates all 17 concepts x 4 fuels at defaults plus slider-extreme points for three representative concepts:

```python
"""Dump JAX-version reference outputs for parity testing.

Run from the freeze worktree env:
    cd /mnt/c/Users/talru/1cfe/1costingfe-freeze
    uv run python /mnt/c/Users/talru/1cfe/1costingfe-explorer/scripts/dump_parity_reference.py \
        > /mnt/c/Users/talru/1cfe/1costingfe-explorer/tests/parity_reference.json
"""

import json

from costingfe import ConfinementConcept, CostModel, Fuel

BASE = dict(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30.0)
SLIDER_POINTS = [
    dict(net_electric_mw=100.0, availability=0.5, interest_rate=0.12,
         construction_time_yr=10.0, lifetime_yr=20.0),
    dict(net_electric_mw=2500.0, availability=0.95, interest_rate=0.03,
         construction_time_yr=3.0, lifetime_yr=60.0),
]
SLIDER_CONCEPTS = [
    (ConfinementConcept.TOKAMAK, Fuel.DT),
    (ConfinementConcept.PULSED_FRC, Fuel.DHE3),
    (ConfinementConcept.LASER_IFE, Fuel.DT),
]


def run_case(concept, fuel, kwargs):
    try:
        model = CostModel(concept=concept, fuel=fuel)
        r = model.forward(**kwargs)
        return {
            "lcoe": float(r.costs.lcoe),
            "overnight_cost": float(r.costs.overnight_cost),
            "cas22": float(r.costs.cas22),
            "cas22_detail": {k: float(v) for k, v in r.cas22_detail.items()},
        }
    except Exception as e:  # combo not supported in the model: record and skip
        return {"error": f"{type(e).__name__}: {e}"}


cases = []
for concept in ConfinementConcept:
    for fuel in Fuel:
        cases.append({
            "concept": concept.value, "fuel": fuel.value, "kwargs": BASE,
            "result": run_case(concept, fuel, BASE),
        })
for concept, fuel in SLIDER_CONCEPTS:
    for pt in SLIDER_POINTS:
        kwargs = {**BASE, **pt}
        cases.append({
            "concept": concept.value, "fuel": fuel.value, "kwargs": kwargs,
            "result": run_case(concept, fuel, kwargs),
        })

# Elasticity reference (engineering+financial) for two concepts
elasticities = []
for concept, fuel in SLIDER_CONCEPTS[:2]:
    model = CostModel(concept=concept, fuel=fuel)
    r = model.forward(**BASE)
    s = model.sensitivity(r.params)
    elasticities.append({
        "concept": concept.value, "fuel": fuel.value,
        "engineering": s["engineering"], "financial": s["financial"],
    })

print(json.dumps({"cases": cases, "elasticities": elasticities}, indent=1))
```

- [ ] **Step 2: Generate the reference**

```bash
cd /mnt/c/Users/talru/1cfe/1costingfe-freeze
uv run python /mnt/c/Users/talru/1cfe/1costingfe-explorer/scripts/dump_parity_reference.py \
    > /mnt/c/Users/talru/1cfe/1costingfe-explorer/tests/parity_reference.json
python3 -c "import json; d = json.load(open('/mnt/c/Users/talru/1cfe/1costingfe-explorer/tests/parity_reference.json')); errs = [c for c in d['cases'] if 'error' in c['result']]; print(len(d['cases']), 'cases,', len(errs), 'errors'); [print(c['concept'], c['fuel'], c['result']['error']) for c in errs]"
```

Expected: 74 cases (68 default + 6 slider points). Errored combos, if any, are listed — review them: they should be genuinely unsupported combos, not crashes in common ones (tokamak/DT etc. must succeed).

- [ ] **Step 3: Write the failing parity test**

`tests/test_parity.py`:

```python
"""Numpy port must match the JAX original (tests/parity_reference.json).

Tolerance note: the JAX reference ran in float32 (the freeze does not enable
x64); the numpy port runs float64. rtol=2e-3 absorbs that — far below
anything visible on a dashboard slider.
"""

import json
from pathlib import Path

import pytest

from costingfe import ConfinementConcept, CostModel, Fuel

REF = json.loads((Path(__file__).parent / "parity_reference.json").read_text())
RTOL = 2e-3
OK_CASES = [c for c in REF["cases"] if "error" not in c["result"]]


def rel(a, b):
    if abs(b) < 1e-9:
        return abs(a - b)
    return abs(a - b) / abs(b)


@pytest.mark.parametrize(
    "case", OK_CASES,
    ids=[f"{c['concept']}-{c['fuel']}-{int(c['kwargs']['net_electric_mw'])}MW"
         for c in OK_CASES],
)
def test_lcoe_parity(case):
    model = CostModel(
        concept=ConfinementConcept(case["concept"]), fuel=Fuel(case["fuel"])
    )
    r = model.forward(**case["kwargs"])
    ref = case["result"]
    assert rel(float(r.costs.lcoe), ref["lcoe"]) < RTOL
    assert rel(float(r.costs.overnight_cost), ref["overnight_cost"]) < RTOL
    assert rel(float(r.costs.cas22), ref["cas22"]) < RTOL


@pytest.mark.parametrize("ref", REF["elasticities"], ids=lambda r: r["concept"])
def test_elasticity_parity(ref):
    """FD elasticities match autodiff within tornado-relevant tolerance."""
    model = CostModel(
        concept=ConfinementConcept(ref["concept"]), fuel=Fuel(ref["fuel"])
    )
    r = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30.0)
    s = model.sensitivity(r.params, include_costing=False)
    got = {**s["engineering"], **s["financial"]}
    want = {**ref["engineering"], **ref["financial"]}
    for key, w in want.items():
        assert key in got, f"missing elasticity key {key}"
        # absolute 0.02 or 5% relative, whichever is looser: float32 autodiff
        # noise vs float64 FD; tornado ordering only needs this much
        assert abs(got[key] - w) < max(0.02, 0.05 * abs(w)), key
```

- [ ] **Step 4: Run the parity tests**

```bash
cd /mnt/c/Users/talru/1cfe/1costingfe-explorer
uv run pytest tests/test_parity.py -q
```

Expected: all pass. If a case fails with rel diff just above 2e-3, inspect whether the JAX reference was float32-noisy (re-check that case with `JAX_ENABLE_X64=1` in the freeze worktree); only relax the tolerance if the x64 reference confirms the numpy value.

- [ ] **Step 5: Commit**

```bash
git add scripts/ tests/ && git commit -m "Parity tests: numpy port matches JAX original at 74 reference points"
```

### Task 5: Labels and slider metadata

**Files:**
- Create: `backend/data/__init__.py` (empty), `backend/__init__.py` (empty), `backend/data/labels.py`, `backend/data/sliders.py`
- Create: `tests/test_labels.py`

- [ ] **Step 1: Write the failing test**

`tests/test_labels.py`:

```python
from backend.data.labels import CAS22_LABELS, CAS_LABELS, CONCEPT_LABELS, FUEL_LABELS
from costingfe import ConfinementConcept, CostModel, Fuel


def test_every_concept_and_fuel_has_label():
    for c in ConfinementConcept:
        assert c.value in CONCEPT_LABELS
    for f in Fuel:
        assert f.value in FUEL_LABELS


def test_every_cas22_detail_key_has_label():
    m = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    r = m.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30.0)
    for key in r.cas22_detail:
        if key == "C220000":
            continue  # total row, not charted
        assert key in CAS22_LABELS, key


def test_capital_accounts_have_labels():
    for code in ["cas10", "cas21", "cas22", "cas23", "cas24", "cas25", "cas26",
                 "cas27", "cas28", "cas29", "cas30", "cas40", "cas50", "cas60"]:
        assert code in CAS_LABELS
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_labels.py -q
```

Expected: FAIL with `ModuleNotFoundError: backend.data.labels`.

- [ ] **Step 3: Write labels.py**

English names verified against `cas22.py` section comments and `costs.py` docstrings at the freeze commit:

```python
"""Human-readable labels. English primary, account codes secondary."""

CONCEPT_LABELS = {
    "tokamak": "Tokamak",
    "stellarator": "Stellarator",
    "mirror": "Mirror",
    "steady_frc": "Steady-state FRC",
    "dipole": "Levitated dipole",
    "laser_ife": "Laser IFE",
    "zpinch": "Z-pinch",
    "heavy_ion": "Heavy-ion IFE",
    "mag_target": "Magnetized target",
    "plasma_jet": "Plasma-jet MIF",
    "pulsed_frc": "Pulsed FRC",
    "maglif": "MagLIF",
    "theta_pinch": "Theta pinch",
    "dense_plasma_focus": "Dense plasma focus",
    "staged_zpinch": "Staged Z-pinch",
    "orbitron": "Orbitron",
    "polywell": "Polywell",
}

FUEL_LABELS = {"dt": "D-T", "dd": "D-D", "dhe3": "D-He3", "pb11": "p-B11"}

CAS_LABELS = {
    "cas10": "Pre-construction",
    "cas21": "Buildings",
    "cas22": "Reactor plant equipment",
    "cas23": "Turbine plant equipment",
    "cas24": "Electric plant equipment",
    "cas25": "Miscellaneous plant equipment",
    "cas26": "Heat rejection",
    "cas27": "Special materials (blanket fill)",
    "cas28": "Digital twin",
    "cas29": "Contingency",
    "cas30": "Indirect service costs",
    "cas40": "Owner's costs",
    "cas50": "Supplementary costs",
    "cas60": "Interest during construction",
    "cas70": "Operations & maintenance (annual)",
    "cas80": "Fuel & targets (annual)",
    "cas90": "Annualized financial costs",
}

CAS22_LABELS = {
    "C220101": "Blanket & first wall",
    "C220102": "Shield",
    "C220103": "Magnets / confinement coils",
    "C220104": "Heating & current drive / driver",
    "C220105": "Primary structure & support",
    "C220106": "Reactor vacuum system",
    "C220106_vessel": "Vacuum vessel shell",
    "C220106_pump": "Vacuum pumping",
    "C220107": "Power supplies",
    "C220108": "Divertor / target factory",
    "C220109": "Direct energy conversion",
    "C220110": "Remote handling",
    "C220111": "Installation labor",
    "C220112": "Isotope separation (zeroed: market purchase)",
    "C220200": "Main & secondary coolant",
    "C220300": "Auxiliary cooling & cryoplant",
    "C220400": "Radioactive waste management",
    "C220500": "Fuel handling & storage",
    "C220600": "Other reactor plant equipment",
    "C220700": "Instrumentation & control",
}

PARAM_LABELS = {
    "net_electric_mw": "Net electric power",
    "availability": "Availability",
    "interest_rate": "Cost of capital (WACC)",
    "inflation_rate": "Inflation rate",
    "construction_time_yr": "Construction time",
    "lifetime_yr": "Plant lifetime",
    "eta_th": "Thermal cycle efficiency",
    "eta_pin": "Heating wall-plug efficiency",
    "eta_couple": "Heating coupling efficiency",
    "eta_de": "DEC efficiency",
    "b_center": "Peak field on coil",
    "r_bore": "Coil winding radius",
    "R0": "Major radius",
    "f_rep": "Repetition rate",
    "burn_fraction": "Burn fraction",
    "f_GW": "Greenwald density fraction",
    "q95": "Safety factor q95",
}
```

Note: if `test_every_cas22_detail_key_has_label` fails on a key not listed here, read that account's section comment in `costingfe/layers/cas22.py` and add the English name it states — do not invent one.

- [ ] **Step 4: Write sliders.py**

```python
"""Slider specs served to the frontend. Values are forward() kwargs."""

SLIDERS = [
    {"key": "net_electric_mw", "label": "Net electric power", "unit": "MW",
     "min": 100, "max": 2500, "step": 25, "default": 1000},
    {"key": "availability", "label": "Availability", "unit": "",
     "min": 0.40, "max": 0.95, "step": 0.01, "default": 0.85},
    {"key": "interest_rate", "label": "Cost of capital (WACC)", "unit": "",
     "min": 0.02, "max": 0.15, "step": 0.0025, "default": 0.07},
    {"key": "construction_time_yr", "label": "Construction time", "unit": "yr",
     "min": 2, "max": 12, "step": 0.5, "default": 6},
    {"key": "lifetime_yr", "label": "Plant lifetime", "unit": "yr",
     "min": 20, "max": 60, "step": 1, "default": 30},
    {"key": "eta_th", "label": "Thermal cycle efficiency", "unit": "",
     "min": 0.30, "max": 0.60, "step": 0.005, "default": 0.40},
]
```

- [ ] **Step 5: Generate the concept-fuel validity map from the parity reference**

The design requires not offering combos the model cannot compute. The parity dump already recorded which combos error. Generate `backend/data/valid_combos.py`:

```bash
uv run python - <<'EOF'
import json
from pathlib import Path

ref = json.loads(Path("tests/parity_reference.json").read_text())
valid = {}
for c in ref["cases"]:
    if c["kwargs"].get("net_electric_mw") != 1000.0:
        continue  # default-point cases only
    valid.setdefault(c["concept"], [])
    if "error" not in c["result"] and c["fuel"] not in valid[c["concept"]]:
        valid[c["concept"]].append(c["fuel"])
out = "# Generated from tests/parity_reference.json - do not edit by hand.\n"
out += f"VALID_FUELS = {json.dumps(valid, indent=4)}\n"
Path("backend/data/valid_combos.py").write_text(out)
print(out)
EOF
```

Add to `tests/test_labels.py`:

```python
def test_valid_combos_cover_all_concepts():
    from backend.data.valid_combos import VALID_FUELS

    for c in ConfinementConcept:
        assert c.value in VALID_FUELS
        assert len(VALID_FUELS[c.value]) >= 1
```

The meta endpoint (Task 6) serves this as `valid_fuels`, and the frontend fuel picker (Task 9) restricts options to `meta.valid_fuels[concept]`, resetting the fuel to the first valid one on concept switch if the current fuel is invalid.

- [ ] **Step 6: Run tests, commit**

```bash
uv run pytest tests/test_labels.py -q
git add backend tests/test_labels.py && git commit -m "Add English labels, slider specs, and concept-fuel validity map"
```

### Task 6: FastAPI backend

**Files:**
- Create: `backend/main.py`, `backend/routes/__init__.py` (empty), `backend/routes/costing.py`, `backend/services/__init__.py` (empty), `backend/services/costing_service.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write the failing API test**

`tests/test_api.py`:

```python
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_meta():
    r = client.get("/api/costing/meta")
    assert r.status_code == 200
    data = r.json()
    assert len(data["concepts"]) == 17
    assert len(data["fuels"]) == 4
    assert data["concept_labels"]["tokamak"] == "Tokamak"
    assert any(s["key"] == "availability" for s in data["sliders"])


def test_calculate_tokamak_dt():
    r = client.post("/api/costing/calculate", json={
        "concept": "tokamak", "fuel": "dt",
        "net_electric_mw": 1000.0, "availability": 0.85, "lifetime_yr": 30.0,
        "construction_time_yr": 6.0, "interest_rate": 0.07,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["lcoe"] > 0
    assert data["overnight_cost"] > 0
    codes = [a["code"] for a in data["accounts"]]
    assert "cas22" in codes
    labels = {a["code"]: a["label"] for a in data["accounts"]}
    assert labels["cas21"] == "Buildings"
    assert any(d["code"] == "C220103" for d in data["cas22_detail"])


def test_sensitivity_tokamak_dt():
    r = client.post("/api/costing/sensitivity", json={
        "concept": "tokamak", "fuel": "dt",
        "net_electric_mw": 1000.0, "availability": 0.85, "lifetime_yr": 30.0,
        "construction_time_yr": 6.0, "interest_rate": 0.07,
    })
    assert r.status_code == 200
    bars = r.json()["elasticities"]
    assert 0 < len(bars) <= 12
    assert abs(bars[0]["elasticity"]) >= abs(bars[-1]["elasticity"])
    assert "label" in bars[0]


def test_calculate_rejects_unknown_concept():
    r = client.post("/api/costing/calculate", json={
        "concept": "warp_drive", "fuel": "dt",
        "net_electric_mw": 1000.0, "availability": 0.85, "lifetime_yr": 30.0,
        "construction_time_yr": 6.0, "interest_rate": 0.07,
    })
    assert r.status_code == 422
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_api.py -q
```

Expected: FAIL with `ModuleNotFoundError: backend.main`.

- [ ] **Step 3: Implement the service**

`backend/services/costing_service.py`:

```python
"""Thin adapter: UI request -> CostModel.forward() -> JSON-able dict."""

from functools import lru_cache

from backend.data.labels import (
    CAS22_LABELS,
    CAS_LABELS,
    PARAM_LABELS,
)
from costingfe import ConfinementConcept, CostModel, Fuel

CAPITAL_ACCOUNTS = [
    "cas10", "cas21", "cas22", "cas23", "cas24", "cas25", "cas26",
    "cas27", "cas28", "cas29", "cas30", "cas40", "cas50", "cas60",
]
ANNUAL_ACCOUNTS = ["cas70", "cas80", "cas90"]
TORNADO_MAX_BARS = 12


@lru_cache(maxsize=128)
def _get_model(concept: str, fuel: str) -> CostModel:
    return CostModel(concept=ConfinementConcept(concept), fuel=Fuel(fuel))


def _forward(req: dict):
    model = _get_model(req["concept"], req["fuel"])
    kwargs = dict(
        net_electric_mw=req["net_electric_mw"],
        availability=req["availability"],
        lifetime_yr=req["lifetime_yr"],
        construction_time_yr=req["construction_time_yr"],
        interest_rate=req["interest_rate"],
    )
    if req.get("eta_th") is not None:
        kwargs["eta_th"] = req["eta_th"]
    return model, model.forward(**kwargs)


def calculate(req: dict) -> dict:
    _, result = _forward(req)
    accounts = [
        {"code": code, "label": CAS_LABELS[code],
         "value_musd": float(getattr(result.costs, code))}
        for code in CAPITAL_ACCOUNTS
    ]
    annual = [
        {"code": code, "label": CAS_LABELS[code],
         "value_musd": float(getattr(result.costs, code))}
        for code in ANNUAL_ACCOUNTS
    ]
    detail = [
        {"code": k, "label": CAS22_LABELS.get(k, k), "value_musd": float(v)}
        for k, v in result.cas22_detail.items()
        if k != "C220000" and not k.startswith("C220106_")
    ]
    return {
        "lcoe": float(result.costs.lcoe),
        "overnight_cost": float(result.costs.overnight_cost),
        "total_capital": float(result.costs.total_capital),
        "accounts": accounts,
        "annual": annual,
        "cas22_detail": detail,
    }


def sensitivity(req: dict) -> dict:
    model, result = _forward(req)
    s = model.sensitivity(result.params, include_costing=False)
    merged = {**s["engineering"], **s["financial"]}
    ranked = sorted(merged.items(), key=lambda kv: abs(kv[1]), reverse=True)
    bars = [
        {"param": k, "label": PARAM_LABELS.get(k, k), "elasticity": float(v)}
        for k, v in ranked[:TORNADO_MAX_BARS]
    ]
    return {"elasticities": bars}
```

- [ ] **Step 4: Implement routes and app**

`backend/routes/costing.py`:

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from backend.data.labels import CONCEPT_LABELS, FUEL_LABELS
from backend.data.sliders import SLIDERS
from backend.services import costing_service

router = APIRouter()


class CalculateRequest(BaseModel):
    concept: str
    fuel: str
    net_electric_mw: float
    availability: float
    lifetime_yr: float
    construction_time_yr: float
    interest_rate: float
    eta_th: float | None = None

    @field_validator("concept")
    @classmethod
    def concept_known(cls, v):
        if v not in CONCEPT_LABELS:
            raise ValueError(f"unknown concept: {v}")
        return v

    @field_validator("fuel")
    @classmethod
    def fuel_known(cls, v):
        if v not in FUEL_LABELS:
            raise ValueError(f"unknown fuel: {v}")
        return v


@router.get("/meta")
def meta():
    from backend.data.valid_combos import VALID_FUELS

    return {
        "concepts": list(CONCEPT_LABELS),
        "fuels": list(FUEL_LABELS),
        "concept_labels": CONCEPT_LABELS,
        "fuel_labels": FUEL_LABELS,
        "valid_fuels": VALID_FUELS,
        "sliders": SLIDERS,
    }


@router.post("/calculate")
def calculate(req: CalculateRequest):
    try:
        return costing_service.calculate(req.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sensitivity")
def sensitivity(req: CalculateRequest):
    try:
        return costing_service.sensitivity(req.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

`backend/main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.costing import router as costing_router

app = FastAPI(title="1costingfe-explorer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(costing_router, prefix="/api/costing")


@app.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 5: Run tests, measure latency, commit**

```bash
uv run pytest tests/ -q
uv run python -c "
import time
from backend.services import costing_service
req = dict(concept='tokamak', fuel='dt', net_electric_mw=1000.0,
           availability=0.85, lifetime_yr=30.0, construction_time_yr=6.0,
           interest_rate=0.07, eta_th=None)
costing_service.calculate(req)  # warm the model cache
t = time.perf_counter(); costing_service.calculate(req)
print('calculate ms:', round(1000 * (time.perf_counter() - t), 1))
t = time.perf_counter(); costing_service.sensitivity(req)
print('sensitivity ms:', round(1000 * (time.perf_counter() - t), 1))
"
git add backend tests/test_api.py && git commit -m "FastAPI backend: meta, calculate, sensitivity endpoints"
```

Expected: all tests pass; calculate well under 100 ms, sensitivity under ~2 s (it is 2 forward calls per engineering/financial key). If sensitivity exceeds ~3 s (Vercel function timeout risk is 10 s default), reduce TORNADO scope by computing FD only over the keys in `PARAM_LABELS` — add `keys_filter: set[str] | None` to `sensitivity()` in `costingfe/model.py`, pass `set(PARAM_LABELS)` from the service, and skip keys not in the filter inside the loop (same pattern as the `include_costing` guard).

### Task 7: Frontend scaffold

**Files:**
- Create: `frontend/` via Vite, plus `frontend/vite.config.ts`, `frontend/tailwind.config.js`, `frontend/postcss.config.js`, `frontend/src/index.css`

- [ ] **Step 1: Scaffold Vite + deps**

```bash
cd /mnt/c/Users/talru/1cfe/1costingfe-explorer
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm install zustand recharts
npm install -D tailwindcss@3 postcss autoprefixer
npx tailwindcss init -p
```

- [ ] **Step 2: Configure proxy and Tailwind**

`frontend/vite.config.ts`:

```ts
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { '/api': 'http://localhost:8000' },
  },
});
```

`frontend/tailwind.config.js`:

```js
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: { extend: {} },
  plugins: [],
};
```

`frontend/src/index.css`:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

Delete Vite boilerplate: `frontend/src/App.css`, `frontend/src/assets/react.svg`, `frontend/public/vite.svg`.

- [ ] **Step 3: Verify dev server boots, commit**

```bash
npm run build
```

Expected: builds without errors (App.tsx still boilerplate — fine for this task).

```bash
cd .. && git add frontend && git commit -m "Scaffold Vite/React/Tailwind frontend with /api proxy"
```

### Task 8: Frontend types, API client, store

**Files:**
- Create: `frontend/src/types.ts`, `frontend/src/api/client.ts`, `frontend/src/store/index.ts`

- [ ] **Step 1: types.ts**

```ts
export interface SliderSpec {
  key: string;
  label: string;
  unit: string;
  min: number;
  max: number;
  step: number;
  default: number;
}

export interface Meta {
  concepts: string[];
  fuels: string[];
  concept_labels: Record<string, string>;
  fuel_labels: Record<string, string>;
  valid_fuels: Record<string, string[]>;
  sliders: SliderSpec[];
}

export interface AccountRow {
  code: string;
  label: string;
  value_musd: number;
}

export interface CalculateResult {
  lcoe: number;
  overnight_cost: number;
  total_capital: number;
  accounts: AccountRow[];
  annual: AccountRow[];
  cas22_detail: AccountRow[];
}

export interface TornadoBar {
  param: string;
  label: string;
  elasticity: number;
}
```

- [ ] **Step 2: api/client.ts**

```ts
import type { CalculateResult, Meta, TornadoBar } from '../types';

const BASE = '/api/costing';

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchMeta(): Promise<Meta> {
  const res = await fetch(`${BASE}/meta`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export function calculate(params: Record<string, unknown>) {
  return post<CalculateResult>('/calculate', params);
}

export function sensitivity(params: Record<string, unknown>) {
  return post<{ elasticities: TornadoBar[] }>('/sensitivity', params);
}
```

- [ ] **Step 3: store/index.ts (Zustand, debounced recalc)**

```ts
import { create } from 'zustand';
import { calculate, fetchMeta, sensitivity } from '../api/client';
import type { CalculateResult, Meta, TornadoBar } from '../types';

interface State {
  meta: Meta | null;
  concept: string;
  fuel: string;
  params: Record<string, number>;
  result: CalculateResult | null;
  tornado: TornadoBar[] | null;
  loading: boolean;
  error: string | null;
  init: () => Promise<void>;
  setConcept: (c: string) => void;
  setFuel: (f: string) => void;
  setParam: (key: string, value: number) => void;
}

let calcTimer: ReturnType<typeof setTimeout> | undefined;
let sensTimer: ReturnType<typeof setTimeout> | undefined;
const CALC_DEBOUNCE_MS = 250;
const SENS_DEBOUNCE_MS = 1200;

function requestBody(s: State) {
  return { concept: s.concept, fuel: s.fuel, ...s.params };
}

function scheduleRecalc(get: () => State, set: (p: Partial<State>) => void) {
  clearTimeout(calcTimer);
  clearTimeout(sensTimer);
  set({ loading: true });
  calcTimer = setTimeout(async () => {
    try {
      const result = await calculate(requestBody(get()));
      set({ result, loading: false, error: null });
    } catch (e) {
      set({ error: String(e), loading: false });
    }
  }, CALC_DEBOUNCE_MS);
  sensTimer = setTimeout(async () => {
    try {
      const { elasticities } = await sensitivity(requestBody(get()));
      set({ tornado: elasticities });
    } catch {
      /* tornado is best-effort; keep the last one */
    }
  }, SENS_DEBOUNCE_MS);
}

export const useStore = create<State>((set, get) => ({
  meta: null,
  concept: 'tokamak',
  fuel: 'dt',
  params: {},
  result: null,
  tornado: null,
  loading: false,
  error: null,

  init: async () => {
    const meta = await fetchMeta();
    const params: Record<string, number> = {};
    for (const s of meta.sliders) params[s.key] = s.default;
    set({ meta, params });
    scheduleRecalc(get, set);
  },

  setConcept: (concept) => {
    const { meta, fuel } = get();
    const valid = meta?.valid_fuels[concept] ?? [];
    const nextFuel = valid.includes(fuel) ? fuel : valid[0] ?? fuel;
    set({ concept, fuel: nextFuel, tornado: null });
    scheduleRecalc(get, set);
  },

  setFuel: (fuel) => {
    set({ fuel, tornado: null });
    scheduleRecalc(get, set);
  },

  setParam: (key, value) => {
    set({ params: { ...get().params, [key]: value } });
    scheduleRecalc(get, set);
  },
}));
```

- [ ] **Step 4: Type-check, commit**

```bash
cd frontend && npx tsc --noEmit && cd ..
git add frontend/src && git commit -m "Frontend types, API client, debounced Zustand store"
```

### Task 9: Frontend components

**Files:**
- Create: `frontend/src/components/SliderInput.tsx`, `ConceptPicker.tsx`, `LCOEHeadline.tsx`, `CostBreakdownChart.tsx`, `TornadoChart.tsx`
- Modify: `frontend/src/App.tsx`, `frontend/src/main.tsx`, `frontend/index.html`

- [ ] **Step 1: SliderInput.tsx** (adapted from fusion-backcasting's proven component)

```tsx
interface Props {
  label: string;
  unit: string;
  value: number;
  defaultValue: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}

function fmt(v: number, step: number): string {
  const decimals = step >= 1 ? 0 : Math.min(4, -Math.floor(Math.log10(step)));
  return v.toFixed(decimals);
}

export function SliderInput({ label, unit, value, defaultValue, min, max, step, onChange }: Props) {
  const isAtDefault = Math.abs(value - defaultValue) < step * 0.5;
  return (
    <div className="mb-3">
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm font-medium text-gray-700">{label}</span>
        <span className="text-sm font-semibold text-sky-700">
          {fmt(value, step)} {unit}
          {!isAtDefault && (
            <button
              onClick={() => onChange(defaultValue)}
              className="ml-2 text-xs text-gray-400 hover:text-sky-600"
              title={`Reset to ${fmt(defaultValue, step)}`}
            >
              reset
            </button>
          )}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-sky-600"
      />
    </div>
  );
}
```

- [ ] **Step 2: ConceptPicker.tsx**

```tsx
import { useStore } from '../store';

export function ConceptPicker() {
  const { meta, concept, fuel, setConcept, setFuel } = useStore();
  if (!meta) return null;
  const fuels = meta.valid_fuels[concept] ?? meta.fuels;
  return (
    <div className="flex gap-4 mb-4">
      <label className="flex-1">
        <span className="block text-sm font-medium text-gray-700 mb-1">Confinement concept</span>
        <select
          value={concept}
          onChange={(e) => setConcept(e.target.value)}
          className="w-full border rounded px-2 py-1.5 text-sm"
        >
          {meta.concepts.map((c) => (
            <option key={c} value={c}>{meta.concept_labels[c]}</option>
          ))}
        </select>
      </label>
      <label className="w-36">
        <span className="block text-sm font-medium text-gray-700 mb-1">Fuel</span>
        <select
          value={fuel}
          onChange={(e) => setFuel(e.target.value)}
          className="w-full border rounded px-2 py-1.5 text-sm"
        >
          {fuels.map((f) => (
            <option key={f} value={f}>{meta.fuel_labels[f]}</option>
          ))}
        </select>
      </label>
    </div>
  );
}
```

- [ ] **Step 3: LCOEHeadline.tsx**

```tsx
import { useStore } from '../store';

export function LCOEHeadline() {
  const { result, loading, error } = useStore();
  if (error) return <div className="text-red-600 text-sm">{error}</div>;
  if (!result) return <div className="text-gray-400">computing…</div>;
  return (
    <div className={`flex gap-8 items-baseline ${loading ? 'opacity-50' : ''}`}>
      <div>
        <div className="text-4xl font-bold text-sky-800">
          ${result.lcoe.toFixed(1)}
          <span className="text-lg font-normal text-gray-500"> /MWh</span>
        </div>
        <div className="text-xs text-gray-500 uppercase tracking-wide">LCOE</div>
      </div>
      <div>
        <div className="text-2xl font-semibold text-gray-800">
          ${(result.overnight_cost / 1000).toFixed(2)}B
        </div>
        <div className="text-xs text-gray-500 uppercase tracking-wide">Overnight capital</div>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: CostBreakdownChart.tsx** (English labels primary, codes secondary; toggle to CAS22 detail)

```tsx
import { useState } from 'react';
import {
  Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { useStore } from '../store';
import type { AccountRow } from '../types';

function rows(data: AccountRow[]) {
  return data
    .filter((a) => a.value_musd > 0.5)
    .sort((a, b) => b.value_musd - a.value_musd)
    .map((a) => ({ ...a, name: `${a.label} (${a.code})` }));
}

export function CostBreakdownChart() {
  const { result } = useStore();
  const [detail, setDetail] = useState(false);
  if (!result) return null;
  const data = rows(detail ? result.cas22_detail : result.accounts);
  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-sm font-semibold text-gray-700">
          {detail ? 'Reactor plant equipment detail (CAS22)' : 'Capital cost accounts'}
        </h2>
        <button
          onClick={() => setDetail(!detail)}
          className="text-xs text-sky-600 hover:underline"
        >
          {detail ? 'show all accounts' : 'show CAS22 detail'}
        </button>
      </div>
      <ResponsiveContainer width="100%" height={Math.max(220, data.length * 26)}>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 16 }}>
          <XAxis type="number" tickFormatter={(v) => `$${v}M`} fontSize={11} />
          <YAxis type="category" dataKey="name" width={260} fontSize={11} />
          <Tooltip formatter={(v: number) => [`$${v.toFixed(0)}M`, 'cost']} />
          <Bar dataKey="value_musd" fill="#0284c7" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
```

- [ ] **Step 5: TornadoChart.tsx**

```tsx
import {
  Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { useStore } from '../store';

export function TornadoChart() {
  const { tornado } = useStore();
  if (!tornado || tornado.length === 0) {
    return <div className="text-gray-400 text-sm">computing sensitivities…</div>;
  }
  const data = tornado.map((b) => ({ ...b, name: b.label }));
  return (
    <div>
      <h2 className="text-sm font-semibold text-gray-700 mb-1">
        LCOE elasticities (%ΔLCOE per %Δparameter)
      </h2>
      <ResponsiveContainer width="100%" height={Math.max(200, data.length * 26)}>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 16 }}>
          <XAxis type="number" fontSize={11} />
          <YAxis type="category" dataKey="name" width={200} fontSize={11} />
          <Tooltip formatter={(v: number) => [v.toFixed(3), 'elasticity']} />
          <ReferenceLine x={0} stroke="#9ca3af" />
          <Bar dataKey="elasticity">
            {data.map((b) => (
              <Cell key={b.param} fill={b.elasticity > 0 ? '#dc2626' : '#16a34a'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="text-xs text-gray-400 mt-1">
        Red raises LCOE when increased; green lowers it. Central finite
        differences at the current design point.
      </p>
    </div>
  );
}
```

- [ ] **Step 6: App.tsx, main.tsx, index.html**

`frontend/src/App.tsx`:

```tsx
import { useEffect } from 'react';
import { ConceptPicker } from './components/ConceptPicker';
import { CostBreakdownChart } from './components/CostBreakdownChart';
import { LCOEHeadline } from './components/LCOEHeadline';
import { SliderInput } from './components/SliderInput';
import { TornadoChart } from './components/TornadoChart';
import { useStore } from './store';

export default function App() {
  const { meta, params, init, setParam } = useStore();
  useEffect(() => {
    init();
  }, [init]);

  return (
    <div className="max-w-6xl mx-auto p-6">
      <header className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">1costingFE explorer</h1>
        <p className="text-sm text-gray-500">
          Forward costing of 17 fusion concepts and 4 fuels, computed live by a
          numpy port of{' '}
          <a
            href="https://github.com/1cfe/1costingfe"
            className="text-sky-600 hover:underline"
          >
            1costingFE
          </a>{' '}
          at tag v0.1.0-alpha.1. Results carry the model's alpha-stage
          uncertainty; see the repo for assumptions.
        </p>
      </header>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div>
          <ConceptPicker />
          {meta?.sliders.map((s) => (
            <SliderInput
              key={s.key}
              label={s.label}
              unit={s.unit}
              value={params[s.key] ?? s.default}
              defaultValue={s.default}
              min={s.min}
              max={s.max}
              step={s.step}
              onChange={(v) => setParam(s.key, v)}
            />
          ))}
        </div>
        <div className="md:col-span-2 space-y-8">
          <LCOEHeadline />
          <CostBreakdownChart />
          <TornadoChart />
        </div>
      </div>
    </div>
  );
}
```

`frontend/src/main.tsx`:

```tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

`frontend/index.html`: set `<title>1costingFE explorer</title>`.

- [ ] **Step 7: Build, commit**

```bash
cd frontend && npx tsc --noEmit && npm run build && cd ..
git add frontend && git commit -m "Forward explorer UI: pickers, sliders, LCOE headline, CAS chart, tornado"
```

### Task 10: Local integration verification

- [ ] **Step 1: Run both servers**

```bash
cd /mnt/c/Users/talru/1cfe/1costingfe-explorer
uv run uvicorn backend.main:app --port 8000 &
cd frontend && npm run dev &
```

- [ ] **Step 2: API checks**

```bash
curl -s localhost:8000/api/costing/meta | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d['concepts']), 'concepts')"
curl -s -X POST localhost:8000/api/costing/calculate -H 'Content-Type: application/json' \
  -d '{"concept":"pulsed_frc","fuel":"dhe3","net_electric_mw":1000,"availability":0.85,"lifetime_yr":30,"construction_time_yr":6,"interest_rate":0.07}' \
  | python3 -m json.tool | head -20
```

Expected: 17 concepts; pulsed FRC D-He3 returns finite LCOE and a `cas22_detail` list with English labels.

- [ ] **Step 3: Browser check**

Open `http://localhost:5173`. Verify: concept switch recomputes; slider drag updates LCOE within ~quarter second; CAS22 detail toggle works; tornado appears about a second after slider settles; the LCOE for tokamak/DT at defaults matches the parity anchor from Task 1 within rounding. Take a screenshot for the blog post (save to `/mnt/c/Users/talru/1cfe/1costingfe/docs/blog/3 Intro/explorer_screenshot.png`).

- [ ] **Step 4: Kill servers, commit any fixes**

### Task 11: Vercel deploy (user-interactive)

- [ ] **Step 1: Create the GitHub repo and push** (needs user's org choice — 1cfe org assumed)

```bash
cd /mnt/c/Users/talru/1cfe/1costingfe-explorer
gh repo create 1cfe/1costingfe-explorer --private --source . --push
```

- [ ] **Step 2: Link and deploy on Vercel**

Vercel login is interactive — the user runs these (suggest `! vercel login` in the session if needed):

```bash
npx vercel link   # new project: 1costingfe-explorer
npx vercel --prod
```

- [ ] **Step 3: Verify production**

```bash
curl -s https://<deployment-url>/api/costing/meta | head -c 200
```

Expected: JSON with concepts. Browser-check the production URL exactly as in Task 10 Step 3. Record the final URL — the blog post plan (Task "Insert explorer link") consumes it.

- [ ] **Step 4: Cleanup**

```bash
git -C /mnt/c/Users/talru/1cfe/1costingfe worktree remove /mnt/c/Users/talru/1cfe/1costingfe-freeze
```

Only after BOTH plans are fully done (the blog plan also uses the freeze worktree).

---

## Addendum (2026-06-10, user request): advanced inputs + cost overrides

User direction: expose more inputs (port fusion-backcasting's EngineeringPanel pattern) and add CAS/C22 cost-account overrides; include the costing-constants (unit cost) knobs in an internal tab.

### Task 12: Backend — defaults endpoint + override plumbing

- `GET /api/costing/defaults?concept=&fuel=` returns `{defaults: {...}}`: the concept's merged YAML engineering defaults plus all CostingConstants float fields (plain names, single namespace exactly as `forward(**overrides)` accepts them; forward() already injects cc float fields into its params namespace).
- `CalculateRequest` gains `overrides: dict[str, float] = {}` (engineering + costing-constant keys, passed as `forward(**overrides)` after the slider kwargs; slider kwargs win on collision) and `cost_overrides: dict[str, float] = {}` (passed as `forward(cost_overrides=...)`).
- Pre-validate override keys against the defaults namespace and cost_overrides keys against CAS/C22 codes; unknown keys return 422 with the offending key named.
- `calculate` response gains `overridden: list[str]` from `result.overridden` plus the cost_overrides applied.
- Tests: defaults endpoint shape; an engineering override changes LCOE; a `C220103` override pins that account and appears in `overridden`; unknown key 422.

### Task 13: Frontend — advanced inputs card

- Store: `defaults` (fetched per concept/fuel switch, overrides reset on switch), `overrides`, `costOverrides`; request body includes both dicts; same debounce/stale-guard path.
- New `AdvancedInputs` card below the main sliders with three internal tabs:
  1. **Engineering** — port of backcasting's EngineeringPanel: collapsible sections (Power balance, Parasitic power, Geometry, Financial incl. inflation_rate), family-aware filtering, SliderInput widgets, only params present in `defaults`.
  2. **Unit costs** — port of CostingConstantsPanel's grouped sliders, filtered to keys present in `defaults`.
  3. **Account overrides** — port of CostOverridesPanel: M$ inputs per account with computed value alongside, blank = computed, highlighted when set.
- CostBreakdownChart marks overridden accounts (asterisk + tooltip note) using `overridden` from the response.
