# Numpy Backend for 1costingfe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make JAX an optional dependency of 1costingfe so the forward cost model and sensitivities run on plain numpy by default, eliminating the jaxlib OOM in fusion-tea's deployment.

**Architecture:** A single backend module (`_backend.py`) resolves jax-or-numpy once at import and exports a uniform surface (`xp`, `Tracer`, `grad`, `vmap`, `fori_loop`, `optimization_barrier`, `HAS_JAX`). The 9 layer files import their array namespace and jax helpers from it instead of from `jax` directly. In numpy mode, `sensitivity()` routes through the existing `_sensitivity_fd` and `batch()` loops instead of vmapping. JAX moves to a `[jax]` packaging extra.

**Tech Stack:** Python 3.10+, numpy, jax/jaxlib (optional extra), pydantic, pyyaml, pytest, hatchling.

## Global Constraints

- Backend selection is **auto-detect**: use jax if importable, else numpy. Env override `COSTINGFE_BACKEND=numpy` forces numpy; `COSTINGFE_BACKEND=jax` forces jax (errors if jax absent). Copy this logic verbatim from Task 1.
- Keep the local name `jnp` at every existing call site (import `xp as jnp`) to minimize diff.
- Numpy mode must never call `jax.grad` or `jax.vmap`.
- No Co-Authored-By line in commits. One-line commit messages.
- Version bump to `0.1.1`.
- Tests must pass in **both** backends. The numpy run is `COSTINGFE_BACKEND=numpy pytest`.
- Run tests from the worktree `/mnt/c/Users/talru/1cfe/1costingfe-numpy` on branch `feat/numpy-backend`.

---

## File Structure

- **Create** `src/costingfe/_backend.py` — backend resolver and uniform symbol surface.
- **Create** `tests/test_backend.py` — backend-shim unit tests.
- **Modify** `src/costingfe/layers/{economics,costs,radiation,physics}.py` — pure `jnp` import swap.
- **Modify** `src/costingfe/layers/reactivity.py` — `jnp` swap + `optimization_barrier`.
- **Modify** `src/costingfe/layers/tokamak.py` — `jnp` swap + `fori_loop` + `Tracer`.
- **Modify** `src/costingfe/layers/mirror.py` — `jnp` swap + `fori_loop` + `Tracer`.
- **Modify** `src/costingfe/layers/cas22.py` — `jnp` swap + `Tracer`.
- **Modify** `src/costingfe/model.py` — `jnp` swap + `Tracer` + `grad`/`vmap` + sensitivity/batch numpy paths.
- **Modify** `src/costingfe/__init__.py` — guard the `JAX_PLATFORMS` env default behind `HAS_JAX`.
- **Modify** `pyproject.toml` — move jax/jaxlib to a `[jax]` extra, add jax to dev, bump version.
- **Create** `tests/test_backend_parity.py` — cross-backend forward() parity smoke.

---

### Task 1: Backend resolver module

**Files:**
- Create: `src/costingfe/_backend.py`
- Test: `tests/test_backend.py`

**Interfaces:**
- Produces: module `costingfe._backend` exporting `xp` (numpy or jax.numpy module), `HAS_JAX: bool`, `Tracer` (type), `grad` (callable or None), `vmap` (callable or None), `fori_loop(lo:int, hi:int, body:Callable[[int,Any],Any], init:Any) -> Any`, `optimization_barrier(x) -> x`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backend.py
import os
import importlib


def _reload_backend():
    import costingfe._backend as b
    return importlib.reload(b)


def test_numpy_backend_surface(monkeypatch):
    monkeypatch.setenv("COSTINGFE_BACKEND", "numpy")
    b = _reload_backend()
    assert b.HAS_JAX is False
    # xp is numpy
    assert b.xp.array([1.0, 2.0]).sum() == 3.0
    # fori_loop accumulates like a python loop
    assert b.fori_loop(0, 5, lambda i, acc: acc + i, 0) == 10
    # optimization_barrier is identity
    assert b.optimization_barrier(7.0) == 7.0
    # nothing is a Tracer in numpy mode
    assert isinstance(3.0, b.Tracer) is False
    assert b.grad is None and b.vmap is None


def test_force_jax_when_present(monkeypatch):
    import importlib.util
    if importlib.util.find_spec("jax") is None:
        import pytest
        pytest.skip("jax not installed")
    monkeypatch.setenv("COSTINGFE_BACKEND", "jax")
    b = _reload_backend()
    assert b.HAS_JAX is True
    assert callable(b.grad) and callable(b.vmap)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'costingfe._backend'`.

- [ ] **Step 3: Write the module**

```python
# src/costingfe/_backend.py
"""Backend resolver: jax if available (or forced), else numpy.

Selecting numpy drops the jax/jaxlib dependency entirely for the forward
cost model and finite-difference sensitivities. See
docs/superpowers/specs/2026-06-29-numpy-backend-design.md.
"""
import os

_forced = os.environ.get("COSTINGFE_BACKEND", "").strip().lower()

if _forced == "numpy":
    HAS_JAX = False
elif _forced == "jax":
    HAS_JAX = True  # import below raises if jax is genuinely absent
else:
    try:
        import jax as _jax_probe  # noqa: F401
        HAS_JAX = True
    except ImportError:
        HAS_JAX = False

if HAS_JAX:
    import jax
    import jax.numpy as xp
    import jax.lax

    Tracer = jax.core.Tracer
    grad = jax.grad
    vmap = jax.vmap
    fori_loop = jax.lax.fori_loop
    optimization_barrier = jax.lax.optimization_barrier
else:
    import numpy as xp

    class Tracer:  # noqa: D401 - sentinel; nothing is ever an instance
        """Placeholder so isinstance(x, Tracer) is always False under numpy."""

    def fori_loop(lo, hi, body, init):
        val = init
        for i in range(int(lo), int(hi)):
            val = body(i, val)
        return val

    def optimization_barrier(x):
        return x

    grad = None
    vmap = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backend.py -v`
Expected: PASS (the `test_force_jax_when_present` may skip if jax is not in the env).

- [ ] **Step 5: Commit**

```bash
git add src/costingfe/_backend.py tests/test_backend.py
git commit -m "Add jax-or-numpy backend resolver module"
```

---

### Task 2: Wire the pure-jnp layer files

**Files:**
- Modify: `src/costingfe/layers/economics.py:3`, `costs.py:10`, `radiation.py:3`, `physics.py:3`
- Test: existing `tests/test_costs.py`, `tests/test_physics.py`

**Interfaces:**
- Consumes: `costingfe._backend.xp` (Task 1).
- Produces: these four modules import cleanly under numpy.

These files use only `jnp`; the change is one line each.

- [ ] **Step 1: Run the layer tests in numpy mode to confirm they currently fail to import**

Run: `COSTINGFE_BACKEND=numpy pytest tests/test_costs.py tests/test_physics.py -x -q`
Expected: collection/import error mentioning `jax` (jaxlib import or, if jax is installed, this may PASS — in that case proceed; the swap still removes the hard jax dependency).

- [ ] **Step 2: Swap the import in each file**

In each of `economics.py`, `costs.py`, `radiation.py`, `physics.py`, replace the line:

```python
import jax.numpy as jnp
```

with:

```python
from costingfe._backend import xp as jnp
```

- [ ] **Step 3: Run the layer tests in numpy mode**

Run: `COSTINGFE_BACKEND=numpy pytest tests/test_costs.py tests/test_physics.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/costingfe/layers/economics.py src/costingfe/layers/costs.py src/costingfe/layers/radiation.py src/costingfe/layers/physics.py
git commit -m "Route pure-jnp layers through the backend shim"
```

---

### Task 3: Wire reactivity.py

**Files:**
- Modify: `src/costingfe/layers/reactivity.py:21-22,37`
- Test: existing `tests/test_reactivity.py`

**Interfaces:**
- Consumes: `costingfe._backend.{xp, optimization_barrier}`.

- [ ] **Step 1: Swap imports**

Replace lines 21-22:

```python
import jax.lax
import jax.numpy as jnp
```

with:

```python
from costingfe._backend import optimization_barrier, xp as jnp
```

- [ ] **Step 2: Update the barrier call site (line 37)**

Replace:

```python
    return jax.lax.optimization_barrier(n_e * 1e-10)
```

with:

```python
    return optimization_barrier(n_e * 1e-10)
```

- [ ] **Step 3: Run reactivity tests in numpy mode**

Run: `COSTINGFE_BACKEND=numpy pytest tests/test_reactivity.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/costingfe/layers/reactivity.py
git commit -m "Route reactivity layer through the backend shim"
```

---

### Task 4: Wire tokamak.py

**Files:**
- Modify: `src/costingfe/layers/tokamak.py:11-12,400-411,586`
- Test: existing `tests/test_tokamak.py`

**Interfaces:**
- Consumes: `costingfe._backend.{xp, Tracer, fori_loop}`.

- [ ] **Step 1: Swap top imports (lines 11-12)**

Replace:

```python
import jax
import jax.numpy as jnp
```

with:

```python
from costingfe._backend import Tracer, fori_loop as jax_fori_loop, xp as jnp
```

- [ ] **Step 2: Remove the lazy fori_loop machinery (lines ~400-411)**

Delete the `_import_fori_loop` function and the `try/except` block that defines `jax_fori_loop`:

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

(The name `jax_fori_loop` is now provided by the Step 1 import. First confirm `_import_fori_loop` is unused: `grep -n "_import_fori_loop" src/costingfe/layers/tokamak.py` should show only its definition.)

- [ ] **Step 3: Update the Tracer check (line 586)**

Replace:

```python
    if enforce_plasma_limits and not isinstance(plasma_state.beta_N, jax.core.Tracer):
```

with:

```python
    if enforce_plasma_limits and not isinstance(plasma_state.beta_N, Tracer):
```

- [ ] **Step 4: Run tokamak tests in numpy mode**

Run: `COSTINGFE_BACKEND=numpy pytest tests/test_tokamak.py -q`
Expected: PASS (the bisection `jax_fori_loop` now runs as a python loop).

- [ ] **Step 5: Commit**

```bash
git add src/costingfe/layers/tokamak.py
git commit -m "Route tokamak layer through the backend shim"
```

---

### Task 5: Wire mirror.py

**Files:**
- Modify: `src/costingfe/layers/mirror.py:16-17,617,847`
- Test: existing `tests/test_mirror.py`

**Interfaces:**
- Consumes: `costingfe._backend.{xp, Tracer, fori_loop}`.

- [ ] **Step 1: Swap top imports (lines 16-17)**

Replace:

```python
import jax
import jax.numpy as jnp
```

with:

```python
from costingfe._backend import Tracer, fori_loop, xp as jnp
```

- [ ] **Step 2: Update the fori_loop call (line 617)**

Replace:

```python
    lo, hi = jax.lax.fori_loop(0, n_iter, body, (T_lo, T_hi))
```

with:

```python
    lo, hi = fori_loop(0, n_iter, body, (T_lo, T_hi))
```

- [ ] **Step 3: Update the Tracer check (line 847)**

Replace:

```python
    if enforce_plasma_limits and not isinstance(plasma_state.beta, jax.core.Tracer):
```

with:

```python
    if enforce_plasma_limits and not isinstance(plasma_state.beta, Tracer):
```

- [ ] **Step 4: Run mirror tests in numpy mode**

Run: `COSTINGFE_BACKEND=numpy pytest tests/test_mirror.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/costingfe/layers/mirror.py
git commit -m "Route mirror layer through the backend shim"
```

---

### Task 6: Wire cas22.py

**Files:**
- Modify: `src/costingfe/layers/cas22.py:18-19,419-420`
- Test: existing `tests/test_costs.py` (cas22 is exercised there)

**Interfaces:**
- Consumes: `costingfe._backend.{xp, Tracer}`.

- [ ] **Step 1: Swap top imports (lines 18-19)**

Replace:

```python
import jax
import jax.numpy as jnp
```

with:

```python
from costingfe._backend import Tracer, xp as jnp
```

- [ ] **Step 2: Update the Tracer checks (lines 419-420)**

Replace:

```python
                    isinstance(R0, jax.core.Tracer)
                    or isinstance(r_coil, jax.core.Tracer)
```

with:

```python
                    isinstance(R0, Tracer)
                    or isinstance(r_coil, Tracer)
```

- [ ] **Step 3: Run the cas22-exercising tests in numpy mode**

Run: `COSTINGFE_BACKEND=numpy pytest tests/test_costs.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/costingfe/layers/cas22.py
git commit -m "Route cas22 layer through the backend shim"
```

---

### Task 7: Wire model.py — imports, sensitivity, and batch

**Files:**
- Modify: `src/costingfe/model.py:7-8,924,1998-2003,2106-2123`
- Modify: `src/costingfe/__init__.py:5`
- Test: existing `tests/test_model.py`; new assertions added below

**Interfaces:**
- Consumes: `costingfe._backend.{xp, Tracer, grad, vmap, HAS_JAX}`.
- Produces: `CostModel.sensitivity()` returns FD elasticities in numpy mode; `CostModel.batch_lcoe()` returns a `list[float]` in numpy mode.

- [ ] **Step 1: Write a failing test for numpy-mode sensitivity + batch_lcoe**

```python
# append to tests/test_model.py
import os
from costingfe import CostModel, ConfinementConcept, Fuel


def test_sensitivity_and_batch_numpy_mode():
    """Under numpy, sensitivity uses FD and batch_lcoe loops — both must work."""
    if os.environ.get("COSTINGFE_BACKEND") != "numpy":
        import pytest
        pytest.skip("numpy-mode-only behavior check")
    from costingfe._backend import HAS_JAX
    assert HAS_JAX is False
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    result = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    params = result.params
    sens = model.sensitivity(params)
    assert set(sens) == {"engineering", "financial", "costing"}
    key = next(iter(sens["engineering"]))      # a varying continuous lever
    base = float(params[key])
    out = model.batch_lcoe({key: [base, base * 1.01]}, params)
    assert len(out) == 2 and all(isinstance(v, float) for v in out)
```

- [ ] **Step 2: Run it in numpy mode to verify it fails**

Run: `COSTINGFE_BACKEND=numpy pytest tests/test_model.py::test_sensitivity_and_batch_numpy_mode -v`
Expected: FAIL at import (`import jax` in model.py pulls jaxlib / or `jax` name errors after partial edits).

- [ ] **Step 3: Swap the top imports (lines 7-8)**

Replace:

```python
import jax
import jax.numpy as jnp
```

with:

```python
from costingfe._backend import HAS_JAX, Tracer, grad, vmap, xp as jnp
```

- [ ] **Step 4: Update the tracing guard (line 924)**

Replace:

```python
        _tracing = any(isinstance(v, jax.core.Tracer) for v in params.values())
```

with:

```python
        _tracing = any(isinstance(v, Tracer) for v in params.values())
```

- [ ] **Step 5: Route sensitivity to FD in numpy mode (line ~1998)**

Replace:

```python
        if params.get("use_0d_model", False) or params.get("size_from_power", False):
            return self._sensitivity_fd(params, cost_overrides)
```

with:

```python
        if (
            not HAS_JAX
            or params.get("use_0d_model", False)
            or params.get("size_from_power", False)
        ):
            return self._sensitivity_fd(params, cost_overrides)
```

Then on the autodiff line below it, replace `grad_fn = jax.grad(lcoe_fn)` with `grad_fn = grad(lcoe_fn)`.

- [ ] **Step 6: Give batch_lcoe() a numpy loop path (lines ~2106-2123)**

Replace the block:

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

with:

```python
        n = len(next(iter(param_sets.values())))

        if HAS_JAX:
            batch = jnp.stack([base_vals for _ in range(n)])
            for param_name, values in param_sets.items():
                if param_name in keys:
                    idx = keys.index(param_name)
                    batch = batch.at[:, idx].set(jnp.array(values))
            results = vmap(lcoe_fn)(batch)
            return [float(r) for r in results]

        # numpy: no vmap / no .at — loop the rows
        results = []
        for r in range(n):
            vec = list(base_vals)
            for param_name, values in param_sets.items():
                if param_name in keys:
                    vec[keys.index(param_name)] = values[r]
            results.append(float(lcoe_fn(jnp.array(vec))))
        return results
```

- [ ] **Step 7: Guard the JAX_PLATFORMS default in `__init__.py` (line 5)**

Replace:

```python
_os.environ.setdefault("JAX_PLATFORMS", "cpu")
```

with:

```python
from costingfe._backend import HAS_JAX as _HAS_JAX

if _HAS_JAX:
    _os.environ.setdefault("JAX_PLATFORMS", "cpu")
```

(Place the `from costingfe._backend import HAS_JAX` after the existing `import os as _os` line; keep it above the first `from costingfe...` model import.)

- [ ] **Step 8: Run the new test plus the full model suite in numpy mode**

Run: `COSTINGFE_BACKEND=numpy pytest tests/test_model.py -q`
Expected: PASS. If a pinned-value sensitivity assertion fails by a small margin, note it for Task 9 (tolerance loosening); do not loosen here unless it blocks the new test.

- [ ] **Step 9: Commit**

```bash
git add src/costingfe/model.py src/costingfe/__init__.py tests/test_model.py
git commit -m "Route model forward, sensitivity, and batch through the backend shim"
```

---

### Task 8: Packaging — jax as an optional extra

**Files:**
- Modify: `pyproject.toml:3,7-12,14-22`

**Interfaces:**
- Produces: `pip install 1costingfe` installs no jax; `pip install 1costingfe[jax]` adds jax+jaxlib.

- [ ] **Step 1: Edit dependencies and version**

Set `version = "0.1.1"`. Replace the base `dependencies` block:

```python
dependencies = [
    "jax>=0.4.0",
    "jaxlib>=0.4.0",
    "pydantic>=2.0",
    "pyyaml>=6.0",
]
```

with:

```python
dependencies = [
    "numpy>=1.24",
    "pydantic>=2.0",
    "pyyaml>=6.0",
]
```

In `[project.optional-dependencies]`, add a `jax` extra and append jax to `dev`:

```python
jax = [
    "jax>=0.4.0",
    "jaxlib>=0.4.0",
]
dev = [
    "jax>=0.4.0",
    "jaxlib>=0.4.0",
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "pytest-xdist>=3.5",
    "pre-commit>=3.0",
    "ruff==0.12.0",
    "matplotlib>=3.7",
]
```

Also update the project `description` to drop "JAX-native" — change to `"Fusion power plant costing framework (numpy core, optional JAX autodiff)"`.

- [ ] **Step 2: Verify the base install resolves without jax (dry run)**

Run: `python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); print(d['project']['dependencies']); print(list(d['project']['optional-dependencies']))"`
Expected: base deps list has no `jax`/`jaxlib`; optional-dependencies includes `jax` and `dev`.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "Make jax an optional extra; numpy is the base dependency"
```

---

### Task 9: Cross-backend parity + full-suite green in both modes

**Files:**
- Create: `tests/test_backend_parity.py`
- Modify: any test files with exact-value sensitivity assertions that need rtol-2e-3 tolerance (identified at Step 2)

**Interfaces:**
- Consumes: `costingfe.CostModel` forward LCOE across backends.

- [ ] **Step 1: Write the parity smoke test**

```python
# tests/test_backend_parity.py
"""Forward LCOE must be finite and positive under whichever backend runs.

CI runs the full suite once per backend (default jax, and
COSTINGFE_BACKEND=numpy). This file exercises a representative spread of
concepts so a backend-specific NaN/inf regression is caught in either run.
"""
import math
import pytest
from costingfe import CostModel, ConfinementConcept, Fuel

CASES = [
    (ConfinementConcept.TOKAMAK, Fuel.DT,
     dict(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)),
    (ConfinementConcept.MIRROR, Fuel.DT,
     dict(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)),
    (ConfinementConcept.ORBITRON, Fuel.PB11,
     dict(net_electric_mw=0.005, availability=0.85, lifetime_yr=30, n_mod=1)),
]


@pytest.mark.parametrize("concept,fuel,kw", CASES)
def test_forward_lcoe_finite_positive(concept, fuel, kw):
    lcoe = CostModel(concept=concept, fuel=fuel).forward(**kw).costs.lcoe
    assert math.isfinite(lcoe) and lcoe > 0
```

- [ ] **Step 2: Run the FULL suite in numpy mode and triage failures**

Run: `COSTINGFE_BACKEND=numpy pytest -q`
Expected: PASS. For any failure that is a small numeric mismatch on a *sensitivity/elasticity* assertion (FD vs the old jax.grad pin), loosen that single assertion to `pytest.approx(expected, rel=2e-3)`. Do **not** loosen forward-value or non-sensitivity assertions — investigate those instead.

- [ ] **Step 3: Run the FULL suite in jax mode**

Run: `pytest -q`
Expected: PASS (this is the default backend with jax installed).

- [ ] **Step 4: Commit**

```bash
git add tests/test_backend_parity.py
git add -u
git commit -m "Add cross-backend parity smoke; loosen FD-vs-grad sensitivity pins to rtol 2e-3"
```

---

## Out of scope (separate fusion-tea PR)

After this branch merges and `1costingfe==0.1.1` is published, fusion-tea gets its own PR (fusion-tea changes always go via PR):
- `requirements-serve.in`: bump `1costingfe==0.1.0` → `0.1.1` (no `[jax]` extra), recompile `requirements-serve.txt` so jax/jaxlib/scipy drop out.
- No change to `model_setup_helpers.py` (gates are backend-independent) or `server.py` (sensitivity auto-FDs).
- Stored served data left as-is (accept rtol-2e-3 float32→float64 drift, per the spec).
