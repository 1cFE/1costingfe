# Optional-JAX (numpy) backend for 1costingfe

**Date:** 2026-06-29
**Branch:** `feat/numpy-backend` (off `origin/master`)
**Status:** design approved, pending spec review

## Problem

The fusion-tea concept explorer (`exploration/concept_explorer/server.py`, deployed on
Railway) installs `1costingfe`, which pulls in `jax` + `jaxlib` + `scipy` as hard
dependencies. The container is OOM-killed (`Killed` in the Railway log, followed by a
uvicorn restart) — jaxlib's baseline RSS plus per-request tracing/compilation spikes tip
a small instance over its memory limit.

JAX exists in 1costingfe only to provide `jax.grad` autodiff for the
tornado/sensitivity charts and `jax.vmap` for batch evaluation. The forward cost model is
plain array arithmetic that numpy runs unchanged. A separate numpy-only re-port already
exists in `1costingfe-explorer/costingfe/`, but it is a maintained second codebase kept in
parity by hand and it omits the 0D / mirror / reactivity solvers that fusion-tea's tokamak
concepts depend on.

## Goal

A single 1costingfe codebase that runs on numpy by default, with JAX as an optional
extra. `pip install 1costingfe` installs no jaxlib; `pip install 1costingfe[jax]` adds the
autodiff path. fusion-tea installs the base build → no jaxlib → OOM resolved. No second
codebase, no parity-by-hand burden.

## Non-goals

- Not removing JAX. The autodiff path stays, gated behind the `[jax]` extra.
- Not retiring `1costingfe-explorer`. (It may later consume the slim build, but that is
  out of scope here.)
- Not changing the cost model, the 0D physics, or any account methodology.

## Decisions (locked with user 2026-06-29)

1. **Backend selection: auto-detect.** Base install = numpy. If `jax` is importable, use
   it; otherwise numpy. An env override `COSTINGFE_BACKEND=numpy` forces numpy even when
   jax is installed (for parity testing).
2. **fusion-tea served data: accept the rtol-2e-3 drift.** The JAX build runs float32
   (costingfe does not enable `jax_enable_x64`); the numpy build is float64, which is the
   more correct answer. Live compute is float64; stored data is left as-is. Drift is within
   the explorer's already-validated rtol 2e-3.

## Architecture

### 1. Backend shim — `src/costingfe/_backend.py` (new)

Resolves the backend once at import and exports a uniform surface:

```python
import os

_forced = os.environ.get("COSTINGFE_BACKEND", "").lower()
if _forced == "numpy":
    HAS_JAX = False
else:
    try:
        import jax  # noqa: F401
        HAS_JAX = _forced != "numpy"
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

    class Tracer:        # nothing is ever an instance → isinstance guards take
        pass             # the concrete branch, which is correct under numpy

    def fori_loop(lo, hi, body, init):
        val = init
        for i in range(lo, hi):
            val = body(i, val)
        return val

    def optimization_barrier(x):
        return x

    grad = None          # unused in numpy mode (sensitivity routes to FD)
    vmap = None          # unused in numpy mode (batch loops; see §3)
```

Exports: `xp`, `HAS_JAX`, `Tracer`, `grad`, `vmap`, `fori_loop`, `optimization_barrier`.

### 2. The 9 layer files

Files importing JAX: `model.py`, `layers/{economics,costs,mirror,cas22,radiation,
tokamak,reactivity,physics}.py`. Changes are mechanical:

- `import jax.numpy as jnp` → `from costingfe._backend import xp as jnp`
  (keeps every `jnp.` call site untouched — minimal diff).
- `jax.core.Tracer` → `from costingfe._backend import Tracer`.
- `jax.lax.fori_loop` → `from costingfe._backend import fori_loop`
  (mirror.py:617, tokamak.py:401-410).
- `jax.lax.optimization_barrier` → `from costingfe._backend import optimization_barrier`
  (reactivity.py:37).
- Bare `import jax` lines for `jax.grad` / `jax.vmap` in model.py → backend symbols
  (see §3).

### 3. Sensitivity and batch

`sensitivity()` already routes `use_0d_model` / `size_from_power` concepts to
`_sensitivity_fd` (concrete central differences on `forward()` with Python floats) — that
is every tokamak fusion-tea serves. The change:

- **Numpy mode:** `sensitivity()` routes **all** concepts to `_sensitivity_fd`; never
  calls `jax.grad`. FD-vs-grad parity is validated at rtol 2e-3 in the explorer, and FD is
  arguably more faithful (it captures the heating-mix renormalization that grad's tracing
  skips).
- **JAX mode:** unchanged (`jax.grad`, with FD only for the 0D paths as today).
- `batch()` (the only `jax.vmap` and `.at[:, idx].set()` user, model.py:2095-2121): numpy
  mode loops the rows through `forward()` instead of vmapping a stacked array. JAX mode
  unchanged.

Guarded with `from costingfe._backend import HAS_JAX`.

### 4. Packaging — `pyproject.toml`

- Base dependencies: drop `jax`, `jaxlib`. Keep `numpy`, `pydantic`, `pyyaml`, etc.
- Add extra: `[project.optional-dependencies] jax = ["jax", "jaxlib"]`.
- Dev/CI extra includes `jax` so the autodiff suite runs.
- Version bump `0.1.0` → `0.1.1`.

Result: `pip install 1costingfe` → numpy-only; `pip install 1costingfe[jax]` → autodiff.

### 5. fusion-tea integration (separate PR — fusion-tea changes always via PR)

- `requirements-serve.in`: bump `1costingfe==0.1.0` → `0.1.1`, no `[jax]` extra. Recompile
  `requirements-serve.txt` → `jax`/`jaxlib`/`scipy` drop out. Smaller image, OOM gone.
- `model_setup_helpers.py`: unchanged — still flips `MODELS_0D_ENABLED` /
  `SIZING_FEATURES_ENABLED` (backend-independent module-level gates).
- `server.py`: unchanged — `sensitivity()` auto-FDs in numpy mode.
- Stored served data: left as-is (accept rtol-2e-3 drift, per decision 2).

## Testing

- Full suite runs in **both** backends in CI: default (jax present) and
  `COSTINGFE_BACKEND=numpy`.
- Expected fixups: a few exact-value sensitivity assertions may need tolerance loosening to
  rtol 2e-3 where they currently pin `jax.grad` outputs; the explorer established FD ≈ grad
  at that tolerance.
- Parity smoke: compare `forward()` LCOE for all served concepts across backends; assert
  rtol ≤ 2e-3. Report the actual max drift.
- fusion-tea: run its existing explorer smoke / parity scripts
  (`scripts/{smoke_explorer,parity_explorer}.py`) against the slim install.

## Risks

- **Hidden JAX-only semantics.** A `jnp` call with behavior that diverges from numpy
  (dtype promotion, `.at`, NaN handling). Mitigation: the explorer already ran the full
  model on numpy, so the surface is known-portable; the both-backends test suite catches
  regressions.
- **float32 → float64 drift larger than expected somewhere.** Mitigation: parity smoke
  reports the real max drift; if any concept exceeds rtol 2e-3 we investigate before
  shipping.
- **Default flip surprises a JAX consumer.** Anyone who installed `1costingfe` and relied on
  jax.grad now silently gets numpy/FD. Mitigation: documented in the changelog; `[jax]`
  extra restores the old behavior.
