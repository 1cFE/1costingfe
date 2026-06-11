# Mirror 0D Model and Length Sizing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A 0D axisymmetric mirror physics model (confinement, end losses, ambipolar potential) with forward/inverse modes and a length-sizing solve, opt-in via `use_0d_model`/`size_from_power`, with all existing mirror results bit-identical at the defaults.

**Architecture:** New `layers/mirror.py` follows the `layers/tokamak.py` pattern exactly: pure JAX kernel functions, a frozen `MirrorPlasmaState`, `mirror_0d_forward`/`mirror_0d_inverse`, and `mirror_size_from_power`. Reactivity, dilution, and Z_eff come from `layers/reactivity.py` (already multi-fuel and float32-hardened). model.py gains a mirror 0D dispatch parallel to the tokamak's. Spec: `docs/superpowers/specs/2026-06-11-mirror-0d-sizing-design.md` — READ IT FIRST; every formula below is specified there with rationale.

**Tech Stack:** Python, JAX (float32, no x64), pydantic validation, pytest via `uv run pytest`, ruff (pre-commit hook). Branch: create `feat/mirror-0d-sizing` off master.

**Invariants for every task:**
- Full suite green; ruff clean. Baseline at branch time: 495 passed.
- All existing mirror concept results bit-identical with the default `use_0d_model: false` (no existing YAML value changes).
- No Python keyword defaults for new physics params — YAML only. Keyword defaults are allowed only for genuine optional overrides (None sentinels) and behavior flags, matching existing patterns.
- Every new kernel function: jit == eager and finite gradients for all four fuels (the XLA constant-gathering hazard documented in `reactivity.py` applies to `tau_ii`'s 2.09e13-class constant and the 1e-20-class density scalings; use `reactivity._density_1e10`-style barriers or fold constants so no partial product leaves float32 range).
- Commits: single-line messages, no body, NO Co-Authored-By.

---

### Task 0: Branch and shared exception relocation

**Files:**
- Modify: `src/costingfe/layers/physics.py`, `src/costingfe/layers/tokamak.py`

- [ ] Step 0.1: `git checkout -b feat/mirror-0d-sizing`
- [ ] Step 0.2: Move `OperatingPointInfeasible` from `tokamak.py` to `physics.py` (it is concept-agnostic; the mirror inverse needs it too). Keep a re-export in tokamak.py exactly like the `sigma_v_dt = sigv_dt` pattern: `from costingfe.layers.physics import OperatingPointInfeasible` plus a comment that it is re-exported for existing importers. Grep all importers (`grep -rn OperatingPointInfeasible src/ tests/ examples/`) and leave them working unchanged.
- [ ] Step 0.3: Run `uv run pytest tests/test_multifuel_0d.py tests/test_tokamak.py -q` (gate tests must still pass). Commit: `Move OperatingPointInfeasible to physics.py, re-export from tokamak`

---

### Task 1: Confinement kernel (tau functions + ambipolar potential)

**Files:**
- Create: `src/costingfe/layers/mirror.py`
- Create: `tests/test_mirror.py`

TDD. The formulas, constants, and validity notes are in the spec sections "Confinement time" and "Energy confinement time". Write the tests first.

- [ ] Step 1.1: Create `tests/test_mirror.py` with a `TestConfinement` class:

```python
"""Tests for the 0D mirror physics model."""

import jax
import jax.numpy as jnp
import pytest

from costingfe.layers.mirror import (
    compute_ambipolar_potential,
    compute_tau_classical,
    compute_tau_gas_dynamic,
    compute_tau_ii,
    compute_tau_pastukhov,
    compute_tau_radial,
)

_N = 1.0e20  # m^-3
_TI = 20.0  # keV
_TE = 20.0


class TestConfinement:
    def test_tau_ii_scaling(self):
        # tau_ii ~ T^1.5 / n
        t1 = float(compute_tau_ii(_N, _TI, 2.5))
        t2 = float(compute_tau_ii(_N, 2.0 * _TI, 2.5))
        t3 = float(compute_tau_ii(2.0 * _N, _TI, 2.5))
        assert t2 / t1 == pytest.approx(2.0**1.5, rel=1e-3)
        assert t3 / t1 == pytest.approx(0.5, rel=1e-3)

    def test_ambipolar_potential_magnitude(self):
        # e*phi = T_e * ln(sqrt(m_i/(2 pi m_e))) ~ 3-4 T_e for A = 2.5
        phi = float(compute_ambipolar_potential(_TE, 2.5))
        assert 2.5 * _TE < phi < 4.5 * _TE

    def test_pastukhov_beats_classical(self):
        # Electrostatic plugging is an exponential enhancement
        tii = compute_tau_ii(_N, _TI, 2.5)
        phi = compute_ambipolar_potential(_TE, 2.5)
        tc = float(compute_tau_classical(tii, R_m=10.0))
        tp = float(compute_tau_pastukhov(tii, R_m=10.0, phi_keV=phi, T_i=_TI))
        assert tp > 10.0 * tc

    def test_gas_dynamic_dominates_at_high_density(self):
        # High n, low T: mean free path < L -> tau_GD < tau_Pastukhov
        n_hi, t_lo = 5.0e20, 1.0
        tii = compute_tau_ii(n_hi, t_lo, 2.5)
        phi = compute_ambipolar_potential(t_lo, 2.5)
        tp = float(compute_tau_pastukhov(tii, R_m=30.0, phi_keV=phi, T_i=t_lo))
        tgd = float(compute_tau_gas_dynamic(R_m=30.0, L=20.0, T_i=t_lo, A=2.5))
        assert tgd < tp

    def test_radial_subdominant_at_reference(self):
        tii = compute_tau_ii(_N, _TI, 2.5)
        tr = float(compute_tau_radial(tii, a=0.5, T_i=_TI, A=2.5, B_min=3.0))
        phi = compute_ambipolar_potential(_TE, 2.5)
        tp = float(compute_tau_pastukhov(tii, R_m=10.0, phi_keV=phi, T_i=_TI))
        assert tr > tp  # radial losses subdominant in a well-confined mirror

    def test_jit_matches_eager(self):
        def chain(n, T):
            tii = compute_tau_ii(n, T, 2.5)
            phi = compute_ambipolar_potential(T, 2.5)
            return compute_tau_pastukhov(tii, R_m=10.0, phi_keV=phi, T_i=T)

        eager = float(chain(_N, _TI))
        jitted = float(jax.jit(chain)(_N, _TI))
        assert jnp.isfinite(jitted)
        assert jitted == pytest.approx(eager, rel=1e-4)

    def test_differentiable(self):
        g = float(jax.grad(lambda T: compute_tau_ii(_N, T, 2.5))(_TI))
        assert jnp.isfinite(g)
        assert g > 0.0
```

- [ ] Step 1.2: Run to verify ImportError failure.
- [ ] Step 1.3: Create `src/costingfe/layers/mirror.py` implementing exactly the spec formulas:

```python
"""Layer 2c: 0D Axisymmetric Mirror Plasma Model.

Confinement, end losses, and plasma state for axisymmetric magnetic
mirrors, following the tokamak 0D pattern. Pure JAX, float32-disciplined.
Physics per docs/superpowers/specs/2026-06-11-mirror-0d-sizing-design.md:
classical (Bing & Roberts 1961), Pastukhov electrostatic plugging
(Pastukhov 1974, Cohen et al. 1978), gas-dynamic (Mirnov & Ryutov 1979).
"""

import jax
import jax.numpy as jnp

from costingfe.layers.physics import OperatingPointInfeasible, ash_neutron_split
from costingfe.layers.reactivity import fusion_power, n_i_over_n_e
from costingfe.types import Fuel

_LN_LAMBDA = 17.0  # Coulomb logarithm, fusion-relevant plasmas
_M_P_OVER_M_E = 1836.15267  # proton-to-electron mass ratio


def compute_tau_ii(n_i, T_i_keV, A):
    """Ion-ion collision time [s]. tau_ii ~ T^1.5 sqrt(A) / n.

    n in 1e20-rescaled units internally so the 2.09e13 prefactor and the
    1e-20 density scale fold into one benign constant (XLA gathers
    multiplicative constants; see reactivity.py docstring).
    """
    n20 = n_i * 1e-20
    # 2.09e13 / 1e20 = 2.09e-7, folded in float64 before entering jnp
    return (2.09e13 * 1e-20) * T_i_keV**1.5 * jnp.sqrt(A) / (n20 * _LN_LAMBDA)


def compute_ambipolar_potential(T_e_keV, A):
    """Ambipolar potential e*phi [keV], Boltzmann relation for a simple
    mirror: e*phi = T_e * ln(sqrt(m_i / (2 pi m_e)))."""
    return T_e_keV * jnp.log(jnp.sqrt(A * _M_P_OVER_M_E / (2.0 * jnp.pi)))


def compute_tau_classical(tau_ii, R_m):
    """Classical mirror confinement (Bing & Roberts): 2.6 ln(R_m) tau_ii."""
    return 2.6 * jnp.log(R_m) * tau_ii


def compute_tau_pastukhov(tau_ii, R_m, phi_keV, T_i):
    """Pastukhov electrostatically plugged confinement [s]."""
    x = phi_keV / T_i
    return (
        tau_ii
        * (jnp.sqrt(jnp.pi) / 2.0)
        * ((R_m + 1.0) / R_m)
        * jnp.log(2.0 * R_m + 2.0)
        * x
        * jnp.exp(x)
    )


def compute_tau_gas_dynamic(R_m, L, T_i, A):
    """Gas-dynamic confinement (Mirnov & Ryutov): R_m L / v_thi."""
    # v_thi = sqrt(2 T_i / m_i); with T in keV and m_i = A m_p:
    # v_thi [m/s] = 4.377e5 * sqrt(T_keV / A)  (folded constant)
    v_thi = 4.377e5 * jnp.sqrt(T_i / A)
    return R_m * L / v_thi

# (verify the v_thi prefactor: sqrt(2 * 1.602e-16 J/keV / 1.6726e-27 kg)
#  = 4.377e5; recompute in the implementation and pin with a test before
#  trusting this transcription)


def compute_tau_radial(tau_ii, a, T_i, A, B_min):
    """Classical cross-field diffusion: (a/rho_i)^2 tau_ii."""
    # rho_i [m] = 4.57e-3 * sqrt(A * T_keV) / B  (deuteron-normalized
    # gyroradius; recompute the prefactor from first principles in the
    # implementation and pin with a test)
    rho_i = 4.57e-3 * jnp.sqrt(A * T_i) / B_min
    return (a / rho_i) ** 2 * tau_ii
```

CAUTION: the two transcribed prefactors (4.377e5, 4.57e-3) must be re-derived from constants in the implementation session and pinned with closed-form tests; do not trust them from this plan. Use exact 2018 CODATA values as in tokamak.py's constants block.

- [ ] Step 1.4: Run TestConfinement — all pass. `uv run ruff check`. Commit: `Add mirror confinement kernel (classical, Pastukhov, gas-dynamic, radial)`

---

### Task 2: Plasma state and forward model

**Files:**
- Modify: `src/costingfe/layers/mirror.py`, `tests/test_mirror.py`

- [ ] Step 2.1: Tests first (`TestForward`): forward returns a `MirrorPlasmaState` with positive p_fus/tau_p/tau_E; beta matches the closed-form dilution-aware expression; `tau_E < tau_p` (escaping particles are preferentially energetic); `f_axial_derived` in (0,1) and decreasing in R_m; energy bookkeeping `p_end ~ W_th / tau_E` to 1e-5; multi-fuel: DHE3 forward populates `dhe3_dd_frac_eff` and a pin overrides it (mirror the tokamak tests); jit==eager on the full forward for all four fuels.
- [ ] Step 2.2: Implement per spec sections "Energy confinement time", "End-loss power", "Outputs: MirrorPlasmaState":
  - `MirrorPlasmaState` frozen dataclass with the spec's field table (including `f_axial_derived` and `dhe3_dd_frac_eff: float = 0.0`).
  - `mirror_0d_forward(L, a, B_min, R_m, T_i, T_e, n_e, p_input, fuel, *, dhe3_dd_frac_pin, T_i_over_T_e is NOT used here — T_i and T_e are both explicit mirror inputs — plus the fuel-fraction kwargs and mix ratios exactly as `reactivity.fusion_power` requires)`. Volume `pi a^2 L`, fw_area `2 pi a L`, fusion power via `fusion_power(...)`, partition via `ash_neutron_split` with the effective fraction, combined tau per spec (`1/tau_axial = 1/tau_P + 1/tau_GD`, `1/tau_p = 1/tau_axial + 1/tau_radial`), `tau_E = tau_p * 1.5*(T_i+T_e) / (2*phi + T_i + T_e)`, `P_end = W_th / tau_E` with dilution-aware `W_th = 1.5 * n_e * (T_e + n_i_frac*T_i) * KEV_TO_J * V`, beta per the spec formula, `f_axial_derived = p_axial / (p_axial + p_radial)` strictly as a state field.
  - Keep the float32 discipline: density enters via a 1e-10 barrier exactly as `reactivity._density_1e10` (import it or replicate the documented pattern).
- [ ] Step 2.3: Tests green; commit: `Add MirrorPlasmaState and mirror 0D forward model`

---

### Task 3: Inverse mode with the beta gate

**Files:**
- Modify: `src/costingfe/layers/mirror.py`, `tests/test_mirror.py`

- [ ] Step 3.1: Tests first (`TestInverse`): DT inverse hits `pt.p_net == approx(target, rel=0.05)` at a GDT/WHAM-plausible machine; T_i lands inside the fuel bracket; implied beta over `beta_max` raises `OperatingPointInfeasible` with "beta" in the message; `enforce_plasma_limits=False` returns the state; DHE3 inverse with the `f_rad_fus=0.24` proxy converges.
- [ ] Step 3.2: Implement `mirror_0d_inverse` per the spec: single pass (f_dec is a fixed input), `mfe_inverse_power_balance` -> required p_fus -> bisect T_i via `jax.lax.fori_loop` over the fuel-keyed bracket `_T_BRACKET_MIRROR = {DT: (2.0, 80.0), DD: (5.0, 100.0), DHE3: (20.0, 100.0), PB11: (50.0, 300.0)}` -> `mirror_0d_forward` at the solution -> final `mfe_forward_power_balance` with the effective D-He3 fraction -> beta gate (error-severity is beta only; wall loading warns). Reuse the tokamak's gate structure verbatim, including the tracer-skip guard and the escape-hatch kwarg with default True (behavior flag).
- [ ] Step 3.3: Tests green; full suite; commit: `Add mirror 0D inverse mode with beta feasibility gate`

---

### Task 4: Model dispatch, YAML, validation

**Files:**
- Modify: `src/costingfe/model.py`, `src/costingfe/validation.py`, `src/costingfe/data/defaults/steady_state_mirror.yaml`
- Test: `tests/test_mirror.py` (model-integration class)

READ FIRST: model.py `_power_balance` (the `use_0d` branch near line 163), the multi-fuel gate in `forward()` (near line 785, currently `self.concept == ConfinementConcept.TOKAMAK`), and validation.py's `use_0d_model is only supported for TOKAMAK` check (near line 248).

- [ ] Step 4.1: Tests first: `CostModel(MIRROR, DT).forward(..., use_0d_model=True)` produces finite LCOE and a `MirrorPlasmaState` on the result; with `use_0d_model` absent (default false) the result is bit-identical to master for the existing mirror YAML (pin the current LCOE value in the test with a comment that it guards the opt-in default); DHE3 mirror 0D run uses the f_rad_fus proxy by default; explicit `dhe3_dd_frac` override pins.
- [ ] Step 4.2: Implement:
  - YAML: add `R_m: 10.0`, `T_i: 20.0`, `beta_max: 0.5`, `use_0d_model: false`, `enforce_plasma_limits: true` to `steady_state_mirror.yaml` (n_e, T_e, B, chamber_length, plasma_t already exist).
  - validation.py: allow `use_0d_model` for MIRROR as well as TOKAMAK; require the mirror-specific keys when enabled (R_m, beta_max).
  - model.py: extend the multi-fuel gate condition to `concept in (TOKAMAK, MIRROR)` for the Z_eff transform, `dhe3_dd_frac_pin`, and f_rad_fus resolution (T brackets stay concept-specific — mirror brackets live in mirror.py); add `_power_balance_mirror_0d` dispatched from `_power_balance` when `use_0d and concept == MIRROR`, mapping params (`chamber_length`->L, `plasma_t`->a, `B`->B_min) and storing the state on `self._plasma_state`.
  - Check `_plasma_state` consumers (disruption penalty!): the disruption penalty must NOT engage for mirror states — it keys off tokamak fields (beta_N, f_GW, q95). Guard it by concept, not by state presence; verify with a test that a mirror 0D run applies no disruption penalty.
- [ ] Step 4.3: Full suite green (the bit-identity pin is the load-bearing check); commit: `Dispatch mirror 0D model through CostModel with multi-fuel gate`

---

### Task 5: Length sizing and optimize mode

**Files:**
- Modify: `src/costingfe/layers/mirror.py`, `src/costingfe/model.py`, `steady_state_mirror.yaml`
- Test: `tests/test_mirror.py`

- [ ] Step 5.1: Tests first: sized L grows with net power target (200 vs 600 MWe); density equals the f_beta closed form at the solved point; pinning `chamber_length` in sizing mode raises; unreachable target raises `SizingInfeasible`; optimize mode returns f_beta within bounds; DHE3 sizes longer than DT at equal power (with the proxy and a fuel-appropriate no-blanket build, mirroring the tokamak example's treatment).
- [ ] Step 5.2: Implement per the spec's "Length sizing" section: `net_electric_at_L` (GSS over T_i under the fuel bracket; density from f_beta; reuse the tokamak's GSS constants), `mirror_size_from_power` (bisection on [L_min, L_max], `SizingInfeasible` contract), model.py gate (`size_from_power` for MIRROR routes here; forbid pinning `chamber_length`; optimize_lcoe wraps over f_beta via the existing `_optimize_fgw` golden-section helper, renamed or parameterized if needed). YAML: `f_beta: 0.85`, `L_min: 1.0`, `L_max: 200.0`, `f_beta_min: 0.3`, `f_beta_max: 1.0`.
- [ ] Step 5.3: Full suite; commit: `Add mirror length sizing and LCOE-over-f_beta optimize mode`

---

### Task 6: Validation anchors (research step)

**Files:**
- Create: `docs/account_justification/mirror_confinement.md`
- Modify: `tests/test_mirror.py`

- [ ] Step 6.1: Gather published parameters for GDT (measured: beta up to ~0.6, gas-dynamic confinement scaling, machine dimensions and fields) and WHAM (design: field, mirror ratio, predicted density/temperature/confinement) from primary sources via web search. Record every number with its citation in `mirror_confinement.md` (sources, methodology, validity caveats per the account-justification house style).
- [ ] Step 6.2: Add `TestAnchors`: forward-model confinement within a documented factor (start at 2x; tighten only with justification) of the anchor values for (a) GDT in the gas-dynamic regime, (b) WHAM in the Pastukhov regime. These tests pin the model to literature, not to itself.
- [ ] Step 6.3: Commit: `Validate mirror confinement against GDT and WHAM anchors`

---

### Task 7: Examples, docs, final review

- [ ] Step 7.1: New `examples/dt_mirror_0d.py` mirroring `dt_tokamak_0d.py` (forward, inverse, gate refusal, sizing, fuel comparison with D-He3 — the physically interesting case). Run it end to end.
- [ ] Step 7.2: Docs: spec status -> Implemented; paper appendix section for the mirror model (present-state, no history), parallel to the tokamak appendix; recompile the paper and commit the PDF (it is tracked now). Check `grep -rn "tokamak only\|TOKAMAK concept" docs/ src/` for claims falsified by the mirror dispatch.
- [ ] Step 7.3: Full suite + ruff; final review pass (spec compliance, then quality); merge decision per the user.

---

## Self-Review Notes

- f_dec: input-only per the settled decision; Task 2's diagnostic test asserts it never feeds the power balance.
- The two transcribed kernel prefactors are flagged for re-derivation (Task 1 caution) — do not skip.
- `use_0d_model: false` default plus the Task 4 bit-identity pin implements the opt-in decision.
- Anchor tolerances start loose (2x) deliberately: a 0D mirror model that matches GDT to 20% would be suspicious, not impressive.
