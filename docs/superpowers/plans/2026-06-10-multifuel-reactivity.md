# Multi-Fuel Reactivity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the tokamak 0D model and power-to-geometry sizing path fuel-aware (DT, DD, D-He3, p-B11): real per-fuel reactivity, quasineutrality dilution, fuel-aware Z_eff, hybrid derived channel fractions, T_i/T_e knob, and fuel-keyed temperature brackets — with DT results bit-identical.

**Architecture:** New concept-agnostic `layers/reactivity.py` holds the reactivity fits and mix algebra. `physics.py` gains an `event_energies` helper (extracted from `ash_neutron_split`) so per-event energy bookkeeping has one source of truth. `layers/tokamak.py`'s kernel calls `fusion_power_density` instead of the DT-only `compute_fusion_power`; `model.py` wires Z_eff, brackets, and the dhe3_dd_frac pin at `forward()` entry. Spec: `docs/superpowers/specs/2026-06-10-multifuel-reactivity-design.md`.

**Tech Stack:** Python, JAX (jnp, float32 — no x64), pydantic validation, pytest, ruff. Branch: `feat/multifuel-reactivity`.

**Run tests with:** `uv run pytest tests/ -x -q` (full) or `uv run pytest tests/test_reactivity.py -v` (targeted). Lint: `uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/`.

**Invariants that must hold at the end of every task:**
- Full suite green, ruff clean.
- DT results bit-identical: the existing LCOE pins (e.g. tokamak 226.89 in the eta_pin tests) and all sizing tests pass unmodified.
- No Python keyword defaults for the NEW params (`T_i_over_T_e`, `dhe3_fuel_ratio`, `pb11_fuel_ratio`): they are keyword-only required arguments fed from YAML-merged params.

---

### Task 0: Sync the spec (two small amendments)

**Files:**
- Modify: `docs/superpowers/specs/2026-06-10-multifuel-reactivity-design.md`

Implementation found two refinements; record them before coding.

- [ ] **Step 0.1: Replace the single `fuel_mix_ratio` knob with two fuel-specific keys**

In spec section 2, replace the `fuel_mix_ratio` sentences with:

> Mix knobs follow the existing fuel-specific key pattern (`dd_f_T`, `dhe3_dd_frac`, `pb11_f_alpha_n`): `dhe3_fuel_ratio` (n_He3/n_D, default 1.0) and `pb11_fuel_ratio` (n_B/n_p, default 0.15), declared in `steady_state_tokamak.yaml` only — no Python keyword default in any function signature.

Update the table column "Meaning of r" references accordingly (D-He3 row uses `dhe3_fuel_ratio`, p-B11 row uses `pb11_fuel_ratio`).

- [ ] **Step 0.2: Examples stay independent**

In spec section 1, replace the sentence about refactoring the example scripts with:

> The example scripts (`dhe3_mix_optimization.py`, `dhe3_burn_fractions.py`) keep their own numpy/float64 implementations as independent cross-checks (the package runs JAX float32; importing it would silently change example outputs). A unit test pins the package fits against float64 reference values computed from the example's coefficients.

- [ ] **Step 0.3: Commit**

```bash
git add docs/superpowers/specs/2026-06-10-multifuel-reactivity-design.md
git commit -m "Spec: fuel-specific mix keys, examples stay independent float64 cross-checks"
```

---

### Task 1: Reactivity fits module

**Files:**
- Create: `src/costingfe/layers/reactivity.py`
- Create: `tests/test_reactivity.py`

- [ ] **Step 1.1: Write the failing tests**

Create `tests/test_reactivity.py`:

```python
"""Tests for the fusion reactivity fits and fuel-mix algebra."""

import jax
import jax.numpy as jnp
import pytest

from costingfe.layers.reactivity import (
    sigv_dd_n,
    sigv_dd_p,
    sigv_dhe3,
    sigv_dt,
    sigv_pb11,
)

# Reference values computed in float64 from the Bosch-Hale (1992) coefficients
# as transcribed in examples/dhe3_mix_optimization.py (verified against the
# published tables there), and from the Nevins-Swain (2000) HT-branch fit
# (coefficients per Tentori & Belloni, Nucl. Fusion 63 (2023), table 2).
# Package runs float32, hence the 1e-3 relative tolerance.
_REF = {
    # T_keV: (sigv_dt, sigv_dhe3, sigv_dd_n, sigv_dd_p) [m^3/s]
    15.0: (2.7399e-22, 1.1754e-24, 1.4810e-24, 1.3900e-24),
    20.0: (4.3302e-22, 3.4821e-24, 2.6027e-24, 2.3990e-24),
    50.0: (8.6491e-22, 5.5539e-23, 1.1330e-23, 9.8383e-24),
    100.0: (8.4477e-22, 1.7185e-22, 2.6817e-23, 2.2439e-23),
}
_REF_PB11 = {
    # T_keV: sigv_pb11 [m^3/s], Nevins-Swain HT branch
    100.0: 6.1526e-23,
    200.0: 2.4274e-22,
    300.0: 3.3852e-22,
    400.0: 3.6982e-22,
}


class TestFitValues:
    @pytest.mark.parametrize("T", sorted(_REF))
    def test_bosch_hale_fits_match_float64_reference(self, T):
        ref_dt, ref_dhe3, ref_ddn, ref_ddp = _REF[T]
        assert float(sigv_dt(T)) == pytest.approx(ref_dt, rel=1e-3)
        assert float(sigv_dhe3(T)) == pytest.approx(ref_dhe3, rel=1e-3)
        assert float(sigv_dd_n(T)) == pytest.approx(ref_ddn, rel=1e-3)
        assert float(sigv_dd_p(T)) == pytest.approx(ref_ddp, rel=1e-3)

    @pytest.mark.parametrize("T", sorted(_REF_PB11))
    def test_pb11_nevins_swain_ht_branch(self, T):
        assert float(sigv_pb11(T)) == pytest.approx(_REF_PB11[T], rel=1e-3)


class TestFitPhysics:
    def test_dt_anchor_nrl(self):
        # NRL formulary: DT <sigma v> at 100 keV ~ 8.54e-16 cm^3/s
        assert float(sigv_dt(100.0)) == pytest.approx(8.54e-22, rel=0.15)

    def test_dt_peak_location(self):
        # Bosch-Hale DT reactivity peaks near 64-67 keV
        Ts = jnp.linspace(10.0, 100.0, 91)
        peak_T = float(Ts[jnp.argmax(jax.vmap(sigv_dt)(Ts))])
        assert 60.0 < peak_T < 72.0

    def test_advanced_fuels_far_below_dt_at_15kev(self):
        # The silent-wrong-fuel bug this feature fixes: at 15 keV D-He3 is
        # >2 orders of magnitude below DT.
        assert float(sigv_dhe3(15.0)) < 0.01 * float(sigv_dt(15.0))
        assert float(sigv_dd_n(15.0) + sigv_dd_p(15.0)) < 0.05 * float(sigv_dt(15.0))

    def test_pb11_peak_in_literature_range(self):
        # NS HT-branch peak: broad, ~3.7e-22 m^3/s near 400-500 keV
        Ts = jnp.linspace(50.0, 500.0, 451)
        vs = jax.vmap(sigv_pb11)(Ts)
        assert 3.0e-22 < float(jnp.max(vs)) < 4.5e-22
        assert float(Ts[jnp.argmax(vs)]) > 350.0

    def test_all_fits_differentiable_and_positive(self):
        for fn, T in [
            (sigv_dt, 15.0),
            (sigv_dhe3, 80.0),
            (sigv_dd_n, 40.0),
            (sigv_dd_p, 40.0),
            (sigv_pb11, 200.0),
        ]:
            g = float(jax.grad(fn)(T))
            assert jnp.isfinite(g)
            assert float(fn(T)) > 0.0
```

- [ ] **Step 1.2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_reactivity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'costingfe.layers.reactivity'`

- [ ] **Step 1.3: Create the module**

Create `src/costingfe/layers/reactivity.py`:

```python
"""Fusion reactivity fits and fuel-mix algebra (concept-agnostic).

Thermal reactivities <sigma*v>(T_i) for the four supported fuels, plus the
quasineutrality dilution and effective-charge algebra they imply. All
functions are pure and JAX-differentiable, so any concept layer (tokamak
today; mirror/FRC sizing later) can import them.

Sources:
- D-T, D-D (both branches), D-He3: Bosch & Hale, Nucl. Fusion 32 (1992) 611.
  Identical coefficients to the float64 verification in
  examples/dhe3_mix_optimization.py, which stays independent on purpose.
- p-B11: Nevins & Swain, Nucl. Fusion 40 (2000) 865, high-temperature branch
  (valid 50-500 keV; below ~50 keV the 148 keV resonance contribution is
  underestimated, acceptable because the p-B11 operating bracket starts at
  50 keV). Coefficients as tabulated by Tentori & Belloni, Nucl. Fusion 63
  (2023). Tentori & Belloni's own updated fit and Putvinski et al., Nucl.
  Fusion 59 (2019) 076018 give higher reactivity (up to +50% at the tail)
  and are the optimistic alternatives; Nevins-Swain is the default.
"""

import jax.numpy as jnp

from costingfe.types import Fuel


def _bosch_hale(T_keV, BG, mrc2, C1, C2, C3, C4, C5, C6, C7):
    """Bosch-Hale reactivity parameterization. T in keV, returns cm^3/s
    (callers convert). BG is the Gamow constant [keV^0.5], mrc2 the reduced
    mass energy [keV]."""
    theta = T_keV / (
        1.0
        - T_keV
        * (C2 + T_keV * (C4 + T_keV * C6))
        / (1.0 + T_keV * (C3 + T_keV * (C5 + T_keV * C7)))
    )
    xi = (BG**2 / (4.0 * theta)) ** (1.0 / 3.0)
    return C1 * theta * jnp.sqrt(xi / (mrc2 * T_keV**3)) * jnp.exp(-3.0 * xi)


def sigv_dt(T_keV):
    """D + T -> n + 4He reactivity [m^3/s], Bosch-Hale, valid 0.2-100 keV."""
    return (
        _bosch_hale(
            T_keV,
            34.3827,
            1124656.0,
            1.17302e-9,
            1.51361e-2,
            7.51886e-2,
            4.60643e-3,
            1.35000e-2,
            -1.06750e-4,
            1.36600e-5,
        )
        * 1e-6
    )


def sigv_dhe3(T_keV):
    """D + 3He -> p + 4He reactivity [m^3/s], Bosch-Hale, valid 0.5-190 keV."""
    return (
        _bosch_hale(
            T_keV,
            68.7508,
            1124572.0,
            5.51036e-10,
            6.41918e-3,
            -2.02896e-3,
            -1.91080e-5,
            1.35776e-4,
            0.0,
            0.0,
        )
        * 1e-6
    )


def sigv_dd_n(T_keV):
    """D + D -> n + 3He branch reactivity [m^3/s], Bosch-Hale."""
    return (
        _bosch_hale(
            T_keV,
            31.3970,
            937814.0,
            5.43360e-12,
            5.85778e-3,
            7.68222e-3,
            0.0,
            -2.96400e-6,
            0.0,
            0.0,
        )
        * 1e-6
    )


def sigv_dd_p(T_keV):
    """D + D -> p + T branch reactivity [m^3/s], Bosch-Hale."""
    return (
        _bosch_hale(
            T_keV,
            31.3970,
            937814.0,
            5.65718e-12,
            3.41267e-3,
            1.99167e-3,
            0.0,
            1.05060e-5,
            0.0,
            0.0,
        )
        * 1e-6
    )


# p-11B Gamow energy E_G = B_G^2 [keV] and reduced mass energy [keV]
# (Tentori & Belloni 2023, after Nevins & Swain 2000).
_PB11_EG = 22589.0
_PB11_MRC2 = 859526.0


def sigv_pb11(T_keV):
    """p + 11B -> 3 alpha reactivity [m^3/s], Nevins-Swain HT branch.

    Valid 50-500 keV. The C1 coefficient is in keV m^3/s (no cm^3 -> m^3
    conversion, unlike the Bosch-Hale fits above).
    """
    return _bosch_hale(
        T_keV,
        _PB11_EG**0.5,
        _PB11_MRC2,
        4.4467e-14,
        -5.9357e-2,
        2.0165e-1,
        1.0404e-3,
        2.7621e-3,
        -9.1653e-6,
        9.8305e-7,
    )
```

- [ ] **Step 1.4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_reactivity.py -v`
Expected: all PASS. If the pb11 values miss the reference by more than 1e-3 relative, the functional form is wrong — do NOT loosen the tolerance; re-check the coefficient transcription against the spec discussion.

- [ ] **Step 1.5: Commit**

```bash
git add src/costingfe/layers/reactivity.py tests/test_reactivity.py
git commit -m "Add multi-fuel reactivity fits (Bosch-Hale + Nevins-Swain pb11)"
```

---

### Task 2: Extract `event_energies` from `ash_neutron_split`

**Files:**
- Modify: `src/costingfe/layers/physics.py:60-137` (the `ash_neutron_split` region)
- Test: `tests/test_reactivity.py` (extend), existing suite guards the refactor

`ash_neutron_split` (physics.py:60-137) computes per-event `E_total` / `E_neutron` per fuel inline. `fusion_power_density` (Task 3) needs the same `E_total` to convert reaction rates to power consistently. Extract, don't duplicate.

- [ ] **Step 2.1: Write the failing test**

Add to `tests/test_reactivity.py`:

```python
from costingfe.layers.physics import ash_neutron_split, event_energies
from costingfe.types import Fuel

_FRACS = dict(
    dd_f_T=0.969,
    dd_f_He3=0.689,
    dhe3_dd_frac=0.131,
    dhe3_f_T=0.5,
    dhe3_f_He3=0.05,
    pb11_f_alpha_n=0.0,
    pb11_f_p_n=0.0,
)


class TestEventEnergies:
    @pytest.mark.parametrize("fuel", [Fuel.DT, Fuel.DD, Fuel.DHE3, Fuel.PB11])
    def test_consistent_with_ash_neutron_split(self, fuel):
        """ash_neutron_split's ash fraction must equal (E_total - E_neutron)/E_total."""
        E_total, E_neutron = event_energies(fuel, **_FRACS)
        p_ash, p_neutron = ash_neutron_split(100.0, fuel, **_FRACS)
        assert float(p_neutron) == pytest.approx(
            100.0 * float(E_neutron) / float(E_total), rel=1e-6
        )

    def test_dt_event_energy(self):
        E_total, E_neutron = event_energies(Fuel.DT, **_FRACS)
        assert float(E_total) == pytest.approx(17.58, rel=1e-6)
        assert float(E_neutron) == pytest.approx(14.06, rel=1e-6)
```

Note: check the actual YAML value of `dd_f_He3` in `src/costingfe/data/defaults/steady_state_tokamak.yaml` (line ~85) and use it in `_FRACS` — the value above is a placeholder for whatever the YAML says; read it before writing the test.

- [ ] **Step 2.2: Run to verify failure**

Run: `uv run pytest tests/test_reactivity.py::TestEventEnergies -v`
Expected: FAIL with `ImportError: cannot import name 'event_energies'`

- [ ] **Step 2.3: Extract the helper**

In `physics.py`, restructure `ash_neutron_split` (keep its signature and behavior exactly). The existing body computes, per fuel branch, the quantities that lead to `ash_frac`. Pull that into a new function placed directly above it:

```python
def event_energies(
    fuel: Fuel,
    dd_f_T: float,
    dd_f_He3: float,
    dhe3_dd_frac: float,
    dhe3_f_T: float,
    dhe3_f_He3: float,
    pb11_f_alpha_n: float,
    pb11_f_p_n: float,
):
    """Per-fusion-event total and neutron energies [MeV] for each fuel,
    including the secondary-burn channels. Single source of truth shared by
    ash_neutron_split (partition) and fusion_power_density (rate -> power).
    """
```

Move the per-fuel `E_total` / `E_neutron` algebra from `ash_neutron_split` into it verbatim (the DT branch returns `(Q_DT, E_N_DT)`; DD/DHE3/PB11 branches move their existing expressions; raise the same `ValueError` on unknown fuel). Then `ash_neutron_split` becomes:

```python
    E_total, E_neutron = event_energies(
        fuel, dd_f_T, dd_f_He3, dhe3_dd_frac, dhe3_f_T, dhe3_f_He3,
        pb11_f_alpha_n, pb11_f_p_n,
    )
    ash_frac = (E_total - E_neutron) / E_total
    p_ash = p_fus * ash_frac
    p_neutron = p_fus * (1.0 - ash_frac)
    return p_ash, p_neutron
```

CAREFUL: the current DT branch computes `ash_frac = E_ALPHA_DT / Q_DT` directly. `(Q_DT - E_N_DT)/Q_DT = 3.52/17.58 = E_ALPHA_DT/Q_DT` only if `Q_DT == E_ALPHA_DT + E_N_DT` (17.58 = 3.52 + 14.06 — true). Verify each branch's moved expression reproduces the same `ash_frac` algebraically; if any branch computed `ash_frac` without an explicit `E_neutron`, derive `E_neutron = E_total * (1 - ash_frac)` from its own quantities rather than inventing new physics.

- [ ] **Step 2.4: Run tests**

Run: `uv run pytest tests/test_reactivity.py::TestEventEnergies tests/test_power_balance.py -q`
Expected: PASS. Then the full suite: `uv run pytest tests/ -q` — PASS (pure refactor).

- [ ] **Step 2.5: Commit**

```bash
git add src/costingfe/layers/physics.py tests/test_reactivity.py
git commit -m "Extract event_energies helper from ash_neutron_split"
```

---

### Task 3: Mix algebra and `fusion_power_density`

**Files:**
- Modify: `src/costingfe/layers/reactivity.py`
- Test: `tests/test_reactivity.py` (extend)

- [ ] **Step 3.1: Write the failing tests**

Add to `tests/test_reactivity.py`:

```python
from costingfe.layers.reactivity import (
    fusion_power_density,
    n_i_over_n_e,
    z_eff_fuel,
)


class TestMixAlgebra:
    def test_dilution_factors(self):
        assert n_i_over_n_e(Fuel.DT, 1.0, 0.15) == pytest.approx(1.0)
        assert n_i_over_n_e(Fuel.DD, 1.0, 0.15) == pytest.approx(1.0)
        # D-He3 at r=1: (1+r)/(1+2r) = 2/3
        assert n_i_over_n_e(Fuel.DHE3, 1.0, 0.15) == pytest.approx(2.0 / 3.0)
        # p-B11 at r=0.15: (1+r)/(1+5r) = 1.15/1.75
        assert n_i_over_n_e(Fuel.PB11, 1.0, 0.15) == pytest.approx(1.15 / 1.75)

    def test_z_eff_fuel(self):
        assert z_eff_fuel(Fuel.DT, 1.0, 0.15) == pytest.approx(1.0)
        assert z_eff_fuel(Fuel.DD, 1.0, 0.15) == pytest.approx(1.0)
        # D-He3 at r=1: (1+4r)/(1+2r) = 5/3
        assert z_eff_fuel(Fuel.DHE3, 1.0, 0.15) == pytest.approx(5.0 / 3.0)
        # p-B11 at r=0.15: (1+25r)/(1+5r) = 4.75/1.75
        assert z_eff_fuel(Fuel.PB11, 0.5, 0.15) == pytest.approx(4.75 / 1.75)


class TestFusionPowerDensity:
    _KW = dict(
        dhe3_fuel_ratio=1.0,
        pb11_fuel_ratio=0.15,
        dhe3_dd_frac_pin=None,
        **_FRACS,
    )

    def test_dt_matches_legacy_formula(self):
        """DT must reproduce compute_fusion_power exactly (bit-identical path)."""
        from costingfe.layers.tokamak import compute_fusion_power

        kw = dict(self._KW)
        kw.pop("dhe3_dd_frac")  # _FRACS carries it; pin governs instead
        p, frac = fusion_power_density(Fuel.DT, 1.0e20, 13.0, 830.0, **kw)
        assert float(p) == float(compute_fusion_power(1.0e20, 13.0, 830.0))
        assert float(frac) == 0.0

    def test_dhe3_derives_side_channel_fraction(self):
        kw = dict(self._KW)
        kw.pop("dhe3_dd_frac")
        p, frac = fusion_power_density(Fuel.DHE3, 1.0e20, 70.0, 830.0, **kw)
        # r=1: n_D = n_e/3, n_He3 = n_e/3
        n_D = 1.0e20 / 3.0
        R_dhe3 = n_D * n_D * float(sigv_dhe3(70.0))
        R_dd = 0.5 * n_D * n_D * float(sigv_dd_n(70.0) + sigv_dd_p(70.0))
        assert float(frac) == pytest.approx(R_dd / (R_dd + R_dhe3), rel=1e-5)
        assert float(p) > 0.0

    def test_dhe3_pin_overrides_derived(self):
        kw = dict(self._KW)
        kw.pop("dhe3_dd_frac")
        kw["dhe3_dd_frac_pin"] = 0.25
        _, frac = fusion_power_density(Fuel.DHE3, 1.0e20, 70.0, 830.0, **kw)
        assert float(frac) == pytest.approx(0.25)

    def test_pb11_dilution_suppresses_power(self):
        kw = dict(self._KW)
        kw.pop("dhe3_dd_frac")
        p, _ = fusion_power_density(Fuel.PB11, 1.0e20, 300.0, 830.0, **kw)
        # Undiluted n_e^2/4 estimate must overshoot the quasineutral result:
        # n_p*n_B = 0.15*n_e^2/(1.75^2) ~ 0.049 n_e^2 < 0.25 n_e^2
        from costingfe.layers.physics import event_energies

        E_total, _ = event_energies(Fuel.PB11, **_FRACS)
        undiluted = (
            0.25 * 1.0e20 * float(sigv_pb11(300.0)) * 1.0e20
            * float(E_total) * 1.602176634e-13 * 830.0 * 1e-6
        )
        assert float(p) < 0.3 * undiluted
        assert float(p) > 0.0
```

- [ ] **Step 3.2: Run to verify failure**

Run: `uv run pytest tests/test_reactivity.py::TestMixAlgebra tests/test_reactivity.py::TestFusionPowerDensity -v`
Expected: FAIL with ImportError.

- [ ] **Step 3.3: Implement the algebra and dispatch**

Append to `src/costingfe/layers/reactivity.py`:

```python
# ---------------------------------------------------------------------------
# Quasineutrality mix algebra: n_e = sum(Z_j n_j)
# ---------------------------------------------------------------------------
def n_i_over_n_e(fuel, dhe3_fuel_ratio, pb11_fuel_ratio):
    """Total fuel-ion to electron density ratio. Plain arithmetic (works on
    floats and JAX tracers alike)."""
    if fuel == Fuel.DT or fuel == Fuel.DD:
        return 1.0
    if fuel == Fuel.DHE3:
        r = dhe3_fuel_ratio
        return (1.0 + r) / (1.0 + 2.0 * r)
    if fuel == Fuel.PB11:
        r = pb11_fuel_ratio
        return (1.0 + r) / (1.0 + 5.0 * r)
    raise ValueError(f"Unknown fuel type: {fuel}")


def z_eff_fuel(fuel, dhe3_fuel_ratio, pb11_fuel_ratio):
    """Fuel-ion contribution to Z_eff = sum(n_j Z_j^2)/n_e (fully stripped)."""
    if fuel == Fuel.DT or fuel == Fuel.DD:
        return 1.0
    if fuel == Fuel.DHE3:
        r = dhe3_fuel_ratio
        return (1.0 + 4.0 * r) / (1.0 + 2.0 * r)
    if fuel == Fuel.PB11:
        r = pb11_fuel_ratio
        return (1.0 + 25.0 * r) / (1.0 + 5.0 * r)
    raise ValueError(f"Unknown fuel type: {fuel}")


# ---------------------------------------------------------------------------
# Rate -> fusion power dispatch
# ---------------------------------------------------------------------------
_MEV_TO_J = 1.602176634e-13


def fusion_power_density(
    fuel,
    n_e,
    T_i,
    V_plasma,
    *,
    dhe3_fuel_ratio,
    pb11_fuel_ratio,
    dhe3_dd_frac_pin,
    dd_f_T,
    dd_f_He3,
    dhe3_f_T,
    dhe3_f_He3,
    pb11_f_alpha_n,
    pb11_f_p_n,
):
    """Fusion power [MW] and effective D-D side-channel fraction for a
    thermal plasma at ion temperature T_i [keV].

    Reactant densities follow quasineutrality with the fuel-mix ratio knobs.
    Per-event energies come from physics.event_energies, so the result is
    consistent with ash_neutron_split's partition. dhe3_dd_frac_pin, when not
    None, overrides the rate-derived side-channel fraction (and is what the
    partition will see). Returns (p_fus_MW, dhe3_dd_frac_eff); the fraction
    is 0.0 for fuels without a side channel.

    Multiplication order keeps intermediates in float32-safe range
    (n^2 ~ 1e40 would overflow), mirroring the original compute_fusion_power.
    """
    from costingfe.layers.physics import event_energies

    def _power(rate_density, E_total_mev):
        # rate_density already split as (n * sv) * n to stay in range
        return rate_density * E_total_mev * _MEV_TO_J * V_plasma * 1e-6

    if fuel == Fuel.DT:
        E_total, _ = event_energies(
            fuel, dd_f_T, dd_f_He3, 0.0, dhe3_f_T, dhe3_f_He3,
            pb11_f_alpha_n, pb11_f_p_n,
        )
        rate = 0.25 * (n_e * sigv_dt(T_i)) * n_e
        return _power(rate, E_total), 0.0

    if fuel == Fuel.DD:
        E_total, _ = event_energies(
            fuel, dd_f_T, dd_f_He3, 0.0, dhe3_f_T, dhe3_f_He3,
            pb11_f_alpha_n, pb11_f_p_n,
        )
        n_D = n_e
        rate = 0.5 * (n_D * (sigv_dd_n(T_i) + sigv_dd_p(T_i))) * n_D
        return _power(rate, E_total), 0.0

    if fuel == Fuel.DHE3:
        r = dhe3_fuel_ratio
        n_D = n_e / (1.0 + 2.0 * r)
        n_He3 = r * n_D
        R_dhe3 = (n_D * sigv_dhe3(T_i)) * n_He3
        R_dd = 0.5 * (n_D * (sigv_dd_n(T_i) + sigv_dd_p(T_i))) * n_D
        derived = R_dd / (R_dd + R_dhe3)
        frac = derived if dhe3_dd_frac_pin is None else dhe3_dd_frac_pin
        E_total, _ = event_energies(
            fuel, dd_f_T, dd_f_He3, frac, dhe3_f_T, dhe3_f_He3,
            pb11_f_alpha_n, pb11_f_p_n,
        )
        return _power(R_dhe3 + R_dd, E_total), frac

    if fuel == Fuel.PB11:
        r = pb11_fuel_ratio
        n_p = n_e / (1.0 + 5.0 * r)
        n_B = r * n_p
        E_total, _ = event_energies(
            fuel, dd_f_T, dd_f_He3, 0.0, dhe3_f_T, dhe3_f_He3,
            pb11_f_alpha_n, pb11_f_p_n,
        )
        rate = (n_p * sigv_pb11(T_i)) * n_B
        return _power(rate, E_total), 0.0

    raise ValueError(f"Unknown fuel type: {fuel}")
```

Note the import of `event_energies` is function-local to avoid a circular import (`physics.py` must not import `reactivity.py`; if it doesn't, hoist the import to module level instead — check first).

- [ ] **Step 3.4: Run tests**

Run: `uv run pytest tests/test_reactivity.py -v`
Expected: PASS. The `test_dt_matches_legacy_formula` exact-equality test is the load-bearing one: if it fails by float noise, make the DT branch's multiplication order literally identical to `compute_fusion_power` (tokamak.py:115-127): `rate = n_e * sv`, then `0.25 * rate * n_e * E_fus_J * V_plasma * 1e-6` with `E_fus_J = E_total * MEV_TO_J` matching constants.

- [ ] **Step 3.5: Commit**

```bash
git add src/costingfe/layers/reactivity.py tests/test_reactivity.py
git commit -m "Add quasineutrality mix algebra and multi-fuel fusion power dispatch"
```

---

### Task 4: Wire the kernel into `tokamak.py`

**Files:**
- Modify: `src/costingfe/layers/tokamak.py` (forward model, beta/W_th, aux heating, inverse mode, brackets)
- Modify: `tests/test_tokamak.py` (signature call sites)
- Test: new `tests/test_multifuel_0d.py`

- [ ] **Step 4.1: Write the failing tests**

Create `tests/test_multifuel_0d.py`:

```python
"""Multi-fuel 0D tokamak model tests."""

import pytest

from costingfe.layers.tokamak import (
    _T_BRACKET_DEFAULTS,
    tokamak_0d_forward,
    tokamak_0d_inverse,
)
from costingfe.types import Fuel

_GEOM = dict(R=3.3, a=1.13, kappa=1.84, B=9.2, q95=3.05, f_GW=0.85)
_FRACS = dict(
    dd_f_T=0.969,
    dd_f_He3=0.689,  # use the actual steady_state_tokamak.yaml value
    dhe3_f_T=0.5,
    dhe3_f_He3=0.05,
    pb11_f_alpha_n=0.0,
    pb11_f_p_n=0.0,
)
_NEW = dict(T_i_over_T_e=1.0, dhe3_fuel_ratio=1.0, pb11_fuel_ratio=0.15)


def _forward(fuel, T_e, dhe3_dd_frac=None, **kw):
    return tokamak_0d_forward(
        **_GEOM,
        T_e=T_e,
        p_input=20.0,
        fuel=fuel,
        dhe3_dd_frac=dhe3_dd_frac,
        **_FRACS,
        **_NEW,
        **kw,
    )


class TestMultiFuelForward:
    def test_dhe3_power_far_below_dt_at_dt_temperature(self):
        ps_dt = _forward(Fuel.DT, 15.0)
        ps_dhe3 = _forward(Fuel.DHE3, 15.0)
        assert float(ps_dhe3.p_fus) < 0.02 * float(ps_dt.p_fus)

    def test_dhe3_derived_fraction_populated(self):
        ps = _forward(Fuel.DHE3, 70.0)
        assert 0.0 < float(ps.dhe3_dd_frac_eff) < 1.0

    def test_dhe3_pin_respected(self):
        ps = _forward(Fuel.DHE3, 70.0, dhe3_dd_frac=0.25)
        assert float(ps.dhe3_dd_frac_eff) == pytest.approx(0.25)

    def test_hot_ion_mode_raises_power_and_beta(self):
        cold = _forward(Fuel.DHE3, 70.0)
        hot = _forward(Fuel.DHE3, 70.0, T_i_over_T_e=1.5)
        assert float(hot.p_fus) > float(cold.p_fus)
        assert float(hot.beta_N) > float(cold.beta_N)

    def test_pb11_solves_at_high_temperature(self):
        ps = _forward(Fuel.PB11, 300.0)
        assert float(ps.p_fus) > 0.0


class TestInverseBrackets:
    def test_bracket_table(self):
        assert _T_BRACKET_DEFAULTS[Fuel.DT] == (1.0, 100.0)
        assert _T_BRACKET_DEFAULTS[Fuel.DD] == (5.0, 100.0)
        assert _T_BRACKET_DEFAULTS[Fuel.DHE3] == (20.0, 200.0)
        assert _T_BRACKET_DEFAULTS[Fuel.PB11] == (50.0, 400.0)

    def test_dhe3_inverse_lands_in_bracket(self):
        ps, pt = tokamak_0d_inverse(
            p_net_target=50.0,
            **_GEOM,
            fuel=Fuel.DHE3,
            dhe3_dd_frac=0.131,
            dhe3_dd_frac_pin=None,
            **_FRACS,
            **_NEW,
        )
        assert 20.0 <= float(ps.T_e) <= 200.0
        assert float(ps.p_fus) > 0.0
```

(`tokamak_0d_inverse`'s other power-balance kwargs keep their existing keyword defaults, so the call above relies on them, matching existing test style at `tests/test_tokamak.py:176-230`. Read `_FRACS`'s `dd_f_He3` from the YAML as in Task 2.)

- [ ] **Step 4.2: Run to verify failure**

Run: `uv run pytest tests/test_multifuel_0d.py -v`
Expected: FAIL (unknown kwargs / missing `_T_BRACKET_DEFAULTS`).

- [ ] **Step 4.3: Implement the kernel changes**

In `src/costingfe/layers/tokamak.py`:

(a) Import the new module near the top (after the physics import):

```python
from costingfe.layers.reactivity import (
    fusion_power_density,
    n_i_over_n_e,
    sigv_dt,
)
```

(b) Replace the `sigma_v_dt` definition (lines 63-93) with a re-export so existing imports keep working, deleting the `_BH_*` constants:

```python
# Bosch-Hale DT reactivity now lives in costingfe.layers.reactivity;
# re-exported here because tests and downstream code import it from tokamak.
sigma_v_dt = sigv_dt
```

(c) Keep `compute_fusion_power` exactly as-is (DT legacy kernel, still used by the DT fast path and existing tests). Update only its docstring's first line to "DT fusion power [MW] (legacy DT-only kernel; multi-fuel callers use reactivity.fusion_power_density)."

(d) Generalize `compute_beta_N` (line 130) preserving the DT value:

```python
def compute_beta_N(n_e, T_e, T_i, n_i_frac, B, I_p_MA, a):
    """Normalized beta [%·m·T/MA] from electron + fuel-ion pressure.

    beta_t = mu_0 * n_e * (T_e + n_i_frac * T_i) [J] / B^2
    For DT/DD (n_i = n_e, T_i = T_e) this reduces to the historical
    2*mu_0*n_e*T/B^2 convention exactly.
    """
    p_J = (T_e + n_i_frac * T_i) * KEV_TO_J
    beta_t = MU_0 * n_e * p_J / B**2
    return beta_t * 100.0 * a * B / I_p_MA
```

(e) Add the bracket table next to `_RADIAL_BUILD_DEFAULTS`:

```python
# Operating-temperature solve brackets [keV] by fuel. DT preserves the
# historical values (inverse bisection 1-100; sizing reads YAML T_min/T_max).
_T_BRACKET_DEFAULTS = {
    Fuel.DT: (1.0, 100.0),
    Fuel.DD: (5.0, 100.0),
    Fuel.DHE3: (20.0, 200.0),
    Fuel.PB11: (50.0, 400.0),
}
```

(f) `tokamak_0d_forward` (line 247): add keyword-only required params `T_i_over_T_e`, `dhe3_fuel_ratio`, `pb11_fuel_ratio`; change `dhe3_dd_frac: float` to `dhe3_dd_frac: float | None` (None means derive). Replace steps 4, 9, 10 of the body:

```python
    # 4. Fusion power (T_i from the hot-ion ratio; fuel-aware reactivity)
    T_i = T_i_over_T_e * T_e
    p_fus, dhe3_dd_frac_eff = fusion_power_density(
        fuel,
        n_e,
        T_i,
        V_plasma,
        dhe3_fuel_ratio=dhe3_fuel_ratio,
        pb11_fuel_ratio=pb11_fuel_ratio,
        dhe3_dd_frac_pin=dhe3_dd_frac,
        dd_f_T=dd_f_T,
        dd_f_He3=dd_f_He3,
        dhe3_f_T=dhe3_f_T,
        dhe3_f_He3=dhe3_f_He3,
        pb11_f_alpha_n=pb11_f_alpha_n,
        pb11_f_p_n=pb11_f_p_n,
    )
```

Step 5 (`ash_neutron_split`): pass `dhe3_dd_frac=dhe3_dd_frac_eff`.

Step 9 (stored energy):

```python
    # 9. Actual confinement: W_th = 1.5 * (n_e T_e + n_i T_i) * V
    n_i_frac = n_i_over_n_e(fuel, dhe3_fuel_ratio, pb11_fuel_ratio)
    W_th_J = 1.5 * n_e * (T_e + n_i_frac * T_i) * KEV_TO_J * V_plasma
```

Step 10: `beta_N = compute_beta_N(n_e, T_e, T_i, n_i_frac, B, I_p, a)`.

Return: add `dhe3_dd_frac_eff=dhe3_dd_frac_eff` to the `PlasmaState(...)` constructor, and add the field to the dataclass after `disruption_rate`:

```python
    dhe3_dd_frac_eff: float = 0.0  # effective D-D side-channel fraction (D-He3)
```

(g) `aux_heating_from_confinement` (line 226): add keyword-only `T_i` and `n_i_frac` args and generalize the stored energy line:

```python
def aux_heating_from_confinement(
    H_factor, I_p, B0, n_e, T_e, V_plasma, p_alpha, R0, a, kappa, M_ion,
    *, T_i, n_i_frac,
):
    ...
    W_th_MW = 1.5 * n_e * (T_e + n_i_frac * T_i) * KEV_TO_J * V_plasma * 1e-6
```

(h) `_find_T_for_pfus` (line 365): make it fuel-aware. New signature and body:

```python
def _find_T_for_pfus(target_pfus, n_e, V_plasma, fuel, fpd_kwargs, T_lo, T_hi, n_iter=60):
    """Bisection for the T_e [keV] yielding the target fusion power.
    fpd_kwargs are the fusion_power_density keyword args (mix ratios, pin,
    burn fractions). T_i = T_i_over_T_e * T_e is applied inside via
    fpd_kwargs["T_i_over_T_e"].
    """
    ratio = fpd_kwargs["T_i_over_T_e"]
    kw = {k: v for k, v in fpd_kwargs.items() if k != "T_i_over_T_e"}

    def body(i, state):
        lo, hi = state
        mid = 0.5 * (lo + hi)
        p_mid, _ = fusion_power_density(fuel, n_e, ratio * mid, V_plasma, **kw)
        lo = jnp.where(p_mid < target_pfus, mid, lo)
        hi = jnp.where(p_mid >= target_pfus, mid, hi)
        return (lo, hi)

    lo, hi = jax_fori_loop(0, n_iter, body, (T_lo, T_hi))
    return 0.5 * (lo + hi)
```

(i) `tokamak_0d_inverse` (line 396): add keyword-only required `T_i_over_T_e`, `dhe3_fuel_ratio`, `pb11_fuel_ratio`, `dhe3_dd_frac_pin: float | None` (keep `dhe3_dd_frac: float` as the initial/YAML value for pass 1). Two-pass structure replacing steps 1-4:

```python
    fpd_kwargs = dict(
        T_i_over_T_e=T_i_over_T_e,
        dhe3_fuel_ratio=dhe3_fuel_ratio,
        pb11_fuel_ratio=pb11_fuel_ratio,
        dhe3_dd_frac_pin=dhe3_dd_frac_pin,
        dd_f_T=dd_f_T,
        dd_f_He3=dd_f_He3,
        dhe3_f_T=dhe3_f_T,
        dhe3_f_He3=dhe3_f_He3,
        pb11_f_alpha_n=pb11_f_alpha_n,
        pb11_f_p_n=pb11_f_p_n,
    )
    T_lo, T_hi = _T_BRACKET_DEFAULTS[fuel]

    # Two-pass fixed point: the required p_fus depends on the energy
    # partition, whose D-He3 side-channel fraction depends on the solved T.
    # Pass 1 uses the supplied dhe3_dd_frac; pass 2 re-solves with the
    # fraction derived at the pass-1 temperature. One refinement suffices at
    # costing fidelity (the fraction moves the partition by a few percent).
    frac = dhe3_dd_frac if dhe3_dd_frac_pin is None else dhe3_dd_frac_pin
    T_e = None
    for _ in range(2):
        p_fus_required = mfe_inverse_power_balance(
            ...existing kwargs unchanged, except dhe3_dd_frac=frac...
        )
        T_e = _find_T_for_pfus(
            p_fus_required, n_e, V_plasma, fuel, fpd_kwargs, T_lo, T_hi
        )
        ps_probe = tokamak_0d_forward(
            ...existing kwargs, T_e=T_e, dhe3_dd_frac=dhe3_dd_frac_pin,
            T_i_over_T_e=T_i_over_T_e, dhe3_fuel_ratio=dhe3_fuel_ratio,
            pb11_fuel_ratio=pb11_fuel_ratio, ...
        )
        frac = ps_probe.dhe3_dd_frac_eff if fuel == Fuel.DHE3 else frac
    plasma_state = ps_probe
```

Then step 4's `mfe_forward_power_balance` call passes `dhe3_dd_frac=plasma_state.dhe3_dd_frac_eff` for DHE3 (and the unchanged `dhe3_dd_frac=frac` otherwise — in practice `plasma_state.dhe3_dd_frac_eff` is 0.0 for non-DHE3 fuels, so pass `dhe3_dd_frac=plasma_state.dhe3_dd_frac_eff if fuel == Fuel.DHE3 else dhe3_dd_frac`).

DT check: pass 1 and pass 2 are identical for DT (frac unused), so the bisection result and downstream numbers are unchanged; the extra pass costs ~60 cheap evaluations. If the DT regression pins move, the loop introduced a difference — find it, don't re-pin.

- [ ] **Step 4.4: Update existing call sites in tokamak.py and tests**

- `_net_at_R0_T` and `tokamak_0d_inverse`'s internal `tokamak_0d_forward` calls: deferred to Task 5 for the params-driven ones; the inverse-internal one is covered in (i).
- `tests/test_tokamak.py`: imports at lines 10-20 are fine (`compute_beta_N`, `sigma_v_dt`, `compute_fusion_power`, `tokamak_0d_forward` all still exist). Update:
  - `compute_beta_N` call sites: change `compute_beta_N(n_e, T_i, B, I_p, a)` to `compute_beta_N(n_e, T_i, T_i, 1.0, B, I_p, a)` (same value by construction).
  - `tokamak_0d_forward` call sites (lines ~176, 195, 218): add `T_i_over_T_e=1.0, dhe3_fuel_ratio=1.0, pb11_fuel_ratio=0.15` to the kwargs.
  - Any direct `tokamak_0d_inverse` test calls: add the same three plus `dhe3_dd_frac_pin=None`.
- `_net_at_R0_T` (line 744) and `aux_heating_from_confinement` callers won't compile until Task 5 — to keep this task green, update `_net_at_R0_T` minimally now: in its `tokamak_0d_forward` call add `T_i_over_T_e=params["T_i_over_T_e"], dhe3_fuel_ratio=params["dhe3_fuel_ratio"], pb11_fuel_ratio=params["pb11_fuel_ratio"]` and change `fuel_frac_kw`'s `dhe3_dd_frac` entry to `params["dhe3_dd_frac_pin"]` for the forward call while the `mfe_forward_power_balance` call uses `dhe3_dd_frac=ps.dhe3_dd_frac_eff if fuel == Fuel.DHE3 else params["dhe3_dd_frac"]` (split `fuel_frac_kw` into the two variants). Its `aux_heating_from_confinement` call gains `T_i=params["T_i_over_T_e"] * ps.T_e, n_i_frac=n_i_over_n_e(fuel, params["dhe3_fuel_ratio"], params["pb11_fuel_ratio"])`.

These params keys don't exist yet in YAML — sizing tests will fail until Task 5/6 adds them. Run the sizing tests; if they fail on the missing keys, proceed to Task 5 and run the suite at the end of Task 6 instead (note it in the commit message).

- [ ] **Step 4.5: Run tests**

Run: `uv run pytest tests/test_multifuel_0d.py tests/test_tokamak.py -q`
Expected: multifuel + tokamak unit tests PASS; sizing tests may fail on missing YAML keys (acceptable only until Task 6).

- [ ] **Step 4.6: Commit**

```bash
git add src/costingfe/layers/tokamak.py tests/test_multifuel_0d.py tests/test_tokamak.py
git commit -m "Wire multi-fuel reactivity, dilution, and hot-ion ratio into the 0D tokamak kernel"
```

---

### Task 5: Model-layer plumbing

**Files:**
- Modify: `src/costingfe/model.py` (forward() entry block, `_power_balance_0d`, `_size_tokamak`)
- Modify: `src/costingfe/data/defaults/steady_state_tokamak.yaml`

- [ ] **Step 5.1: Add the YAML keys**

In `src/costingfe/data/defaults/steady_state_tokamak.yaml`, after the fuel-fraction block (lines 84-89):

```yaml
T_i_over_T_e: 1.0     # Ion-to-electron temperature ratio (hot-ion mode > 1; reactivity, W_th, beta)
dhe3_fuel_ratio: 1.0  # D-He3 mix n_He3/n_D (quasineutrality dilution; 1.0 = 50:50)
pb11_fuel_ratio: 0.15 # p-B11 mix n_B/n_p (lean boron; to be calibrated)
```

- [ ] **Step 5.2: forward() entry block**

In `model.py` `forward()`, directly after the 0D radial-build block (after line 751), add:

```python
        # Multi-fuel kernel inputs for the tokamak 0D/sizing paths: effective
        # Z_eff (fuel-ion contribution + impurity excess over hydrogenic), the
        # dhe3_dd_frac pin (explicit user override -> pinned; otherwise derived
        # at the operating point), and non-DT operating-temperature brackets.
        if (use_0d or params.get("size_from_power", False)) and (
            self.concept == ConfinementConcept.TOKAMAK
        ):
            params["Z_eff"] = z_eff_fuel(
                self.fuel, params["dhe3_fuel_ratio"], params["pb11_fuel_ratio"]
            ) + (params["Z_eff"] - 1.0)
            params["dhe3_dd_frac_pin"] = overrides.get("dhe3_dd_frac")
            if self.fuel != Fuel.DT:
                if "T_min" not in overrides:
                    params["T_min"] = _T_BRACKET_DEFAULTS[self.fuel][0]
                if "T_max" not in overrides:
                    params["T_max"] = _T_BRACKET_DEFAULTS[self.fuel][1]
```

Imports: add `z_eff_fuel` (from `costingfe.layers.reactivity`) and `_T_BRACKET_DEFAULTS` (from `costingfe.layers.tokamak`) to model.py's imports. For DT: `z_eff_fuel` is 1.0, so `Z_eff` is unchanged; the pin defaults to None only inside the tokamak-0D gate, and `_power_balance_0d` for DT ignores it (DT branch of `fusion_power_density` never reads it).

- [ ] **Step 5.3: `_power_balance_0d` forward and inverse branches**

In `_power_balance_0d` (model.py:337):

- `fuel_frac_kw` (line 360): change `dhe3_dd_frac=params["dhe3_dd_frac"]` to stay (this dict now feeds only the power-balance calls). Build a second dict for the plasma-model calls:

```python
        kernel_kw = dict(
            T_i_over_T_e=params["T_i_over_T_e"],
            dhe3_fuel_ratio=params["dhe3_fuel_ratio"],
            pb11_fuel_ratio=params["pb11_fuel_ratio"],
        )
```

- Forward branch: `tokamak_0d_forward(...)` gains `**kernel_kw` and its `dhe3_dd_frac` entry becomes `params["dhe3_dd_frac_pin"]` (pass it explicitly, removing `dhe3_dd_frac` from the `fuel_frac_kw` it receives — restructure so the forward-model call takes the pin and the `mfe_forward_power_balance` call takes the effective value:

```python
            pb_frac = (
                plasma_state.dhe3_dd_frac_eff
                if self.fuel == Fuel.DHE3
                else params["dhe3_dd_frac"]
            )
```

and `mfe_forward_power_balance(..., dhe3_dd_frac=pb_frac, ...)`).

- Inverse branch: `tokamak_0d_inverse(...)` gains `**kernel_kw` and `dhe3_dd_frac_pin=params["dhe3_dd_frac_pin"]` (keeping `dhe3_dd_frac=params["dhe3_dd_frac"]` as the pass-1 seed).

- [ ] **Step 5.4: `_size_tokamak`**

In `_size_tokamak` (model.py:450): the direct `tokamak_0d_forward` call (line 495) gains the same three kernel kwargs and `dhe3_dd_frac=params["dhe3_dd_frac_pin"]`. The `aux_heating_from_confinement` call (line 516) gains:

```python
                T_i=params["T_i_over_T_e"] * float(ps.T_e),
                n_i_frac=n_i_over_n_e(
                    self.fuel, params["dhe3_fuel_ratio"], params["pb11_fuel_ratio"]
                ),
```

(import `n_i_over_n_e` in model.py). `solve_params` is a copy of `params`, so the pin and kernel keys flow into `_net_at_R0_T` automatically.

- [ ] **Step 5.5: Run the full suite — DT regression gate**

Run: `uv run pytest tests/ -q`
Expected: ALL PASS, including the untouched LCOE pins and sizing tests. Any DT numeric drift is a bug in the wiring (most likely the beta/W_th generalization or a wrong multiplication order) — fix the wiring, never re-pin.

- [ ] **Step 5.6: Commit**

```bash
git add src/costingfe/model.py src/costingfe/data/defaults/steady_state_tokamak.yaml
git commit -m "Plumb multi-fuel kernel params, fuel-aware Z_eff, and T brackets through forward()"
```

---

### Task 6: Validation fields and the p-B11 radiation guard

**Files:**
- Modify: `src/costingfe/validation.py`
- Test: `tests/test_multifuel_0d.py` (extend)

- [ ] **Step 6.1: Write the failing tests**

Add to `tests/test_multifuel_0d.py`:

```python
import warnings

from costingfe.model import CostModel
from costingfe.types import ConfinementConcept


class TestPb11Guard:
    def test_pb11_0d_without_f_rad_fus_warns(self):
        m = CostModel(ConfinementConcept.TOKAMAK, Fuel.PB11)
        with pytest.warns(UserWarning, match="f_rad_fus"):
            m.forward(
                net_electric_mw=100.0,
                availability=0.85,
                lifetime_yr=30.0,
                use_0d_model=True,
            )

    def test_pb11_0d_with_f_rad_fus_no_warning(self):
        m = CostModel(ConfinementConcept.TOKAMAK, Fuel.PB11)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            m.forward(
                net_electric_mw=100.0,
                availability=0.85,
                lifetime_yr=30.0,
                use_0d_model=True,
                f_rad_fus=0.83,
            )
```

If other UserWarnings fire legitimately in the second test (e.g. physics-check warnings on the p-B11 point), narrow the filter to the message: `warnings.filterwarnings("error", message=".*f_rad_fus.*")`.

- [ ] **Step 6.2: Run to verify failure**

Run: `uv run pytest tests/test_multifuel_0d.py::TestPb11Guard -v`
Expected: first test FAILS (no warning raised).

- [ ] **Step 6.3: Implement**

In `src/costingfe/validation.py`:

(a) Add fields in the "0D plasma model (tokamak only)" block (after line 161):

```python
    T_i_over_T_e: float | None = Field(default=None, gt=0)
    dhe3_fuel_ratio: float | None = Field(default=None, gt=0)
    pb11_fuel_ratio: float | None = Field(default=None, gt=0)
    f_rad_fus: float | None = Field(default=None, ge=0, le=1)
    size_from_power: bool = False
```

(b) In `check_physics` (the Tier-3 validator at line ~261), add:

```python
        # p-B11 with the 0D/sizing radiation model: the default brems/sync
        # model is not meaningful at p-B11 temperatures with a boron-loaded
        # plasma; the supported proxy is f_rad_fus (~0.83, Putvinski-class
        # optimum). Warn rather than error so exploration stays possible.
        if (
            self.fuel == Fuel.PB11
            and (self.use_0d_model or self.size_from_power)
            and self.f_rad_fus is None
        ):
            warnings.warn(
                "PB11 with use_0d_model/size_from_power and no f_rad_fus set: "
                "bremsstrahlung at p-B11 conditions is not represented by the "
                "default radiation model; set f_rad_fus (~0.83).",
                stacklevel=2,
            )
```

(c) `model.py` passes params into `CostingInput` filtered by `model_fields`, so the new fields validate automatically. Check that `size_from_power` (a bool override, not in YAML) reaches the validator: it arrives via `params` only when the user passes it — confirm `params.get("size_from_power")` lands in the filtered kwargs (it's in `_OPTIONAL_OVERRIDE_KEYS`, and params contains it when overridden, so it does).

- [ ] **Step 6.4: Run tests**

Run: `uv run pytest tests/test_multifuel_0d.py tests/test_validation.py -q` (use the actual validation test filename — check `ls tests/`).
Expected: PASS.

- [ ] **Step 6.5: Commit**

```bash
git add src/costingfe/validation.py tests/test_multifuel_0d.py
git commit -m "Validate multi-fuel knobs and warn on pb11 0D runs without f_rad_fus"
```

---

### Task 7: Integration tests — sizing per fuel and override semantics

**Files:**
- Test: `tests/test_multifuel_0d.py` (extend)

- [ ] **Step 7.1: Write the tests**

```python
class TestMultiFuelSizing:
    _KW = dict(availability=0.85, lifetime_yr=30.0)

    def test_dhe3_sizes_larger_machine_than_dt(self):
        m_dt = CostModel(ConfinementConcept.TOKAMAK, Fuel.DT)
        m_dhe3 = CostModel(ConfinementConcept.TOKAMAK, Fuel.DHE3)
        m_dt.forward(net_electric_mw=200.0, size_from_power=True, **self._KW)
        r0_dt = m_dt._last_R0
        m_dhe3.forward(
            net_electric_mw=200.0,
            size_from_power=True,
            H_factor=1.8,
            **self._KW,
        )
        assert m_dhe3._last_R0 > r0_dt

    def test_dhe3_explicit_dhe3_dd_frac_override_wins(self):
        m = CostModel(ConfinementConcept.TOKAMAK, Fuel.DHE3)
        m.forward(
            net_electric_mw=200.0,
            use_0d_model=True,
            dhe3_dd_frac=0.25,
            **self._KW,
        )
        assert float(m._plasma_state.dhe3_dd_frac_eff) == pytest.approx(0.25)

    def test_dhe3_derives_frac_without_override(self):
        m = CostModel(ConfinementConcept.TOKAMAK, Fuel.DHE3)
        m.forward(net_electric_mw=200.0, use_0d_model=True, **self._KW)
        frac = float(m._plasma_state.dhe3_dd_frac_eff)
        assert 0.0 < frac < 1.0
        assert frac != pytest.approx(0.131, abs=1e-4)  # not the YAML seed
```

- [ ] **Step 7.2: Run and stabilize physically**

Run: `uv run pytest tests/test_multifuel_0d.py::TestMultiFuelSizing -v`

The D-He3 sizing case may raise `SizingInfeasible` (weak reactivity + brems-heavy plasma may not reach 200 MWe inside the YAML `R0_max`): if so, adjust the TEST inputs, not the model — lower `net_electric_mw` (100), raise `H_factor` (up to 2.5), raise `T_i_over_T_e` (1.5), or pass a larger `R0_max`. Pick the minimal change that makes the point feasible and add a comment stating the choice. The assertion `R0_dhe3 > R0_dt` must hold at EQUAL net power — if you lower the power for D-He3, re-run DT at the same power. The derived-frac inequality (`!= 0.131`) assumes the derived value differs visibly from the YAML seed; if it happens to land within 1e-4 of 0.131, widen to `abs=1e-5` after printing the actual value and sanity-checking it against the Task 3 rate formula.

- [ ] **Step 7.3: Run the full suite and lint**

Run: `uv run pytest tests/ -q && uv run ruff check src/ tests/ && uv run ruff format src/ tests/`
Expected: all green, no reformat diffs (or commit them).

- [ ] **Step 7.4: Commit**

```bash
git add tests/test_multifuel_0d.py
git commit -m "Add multi-fuel sizing and override-precedence integration tests"
```

---

### Task 8: Docs sync

**Files:**
- Modify: `docs/superpowers/specs/2026-06-10-multifuel-reactivity-design.md` (status line only)
- Modify: `docs/account_justification/CAS22_reactor_components.md` — ONLY if it states the 0D model is D-T-only; otherwise skip.

- [ ] **Step 8.1: Flip the spec status**

Change `**Status:** Approved design, pending implementation` to `**Status:** Implemented (feat/multifuel-reactivity)`.

- [ ] **Step 8.2: Check for stale D-T-only claims**

Run: `grep -rn "D-T only\|DT only\|DT-only" docs/ src/ --include="*.md" --include="*.py" | grep -iv "pb11\|dd_f\|test"`
Fix any hit that the feature falsified (e.g. a docstring saying the 0D model is DT-only). Do NOT add history notes to `paper.tex` (house rule: the paper never documents past limitations).

- [ ] **Step 8.3: Final full-suite run and commit**

Run: `uv run pytest tests/ -q && uv run ruff check src/ tests/`

```bash
git add -A
git commit -m "Sync docs with multi-fuel reactivity implementation"
```

---

## Self-Review Notes (already applied)

- Spec section 2 (dilution/dispatch) → Tasks 1+3. Section 3 (hybrid fractions) → Tasks 3-5 (pin plumbed via `dhe3_dd_frac_pin`, derived in kernel, YAML seed kept for the non-0D path, which is untouched). Section 4 (T_i/T_e + dilution in W_th/beta) → Task 4 (d)(f)(g). Section 5 (brackets) → Task 4 (e)(h)(i) + Task 5.2. Section 6 (Z_eff) → Task 3 (`z_eff_fuel`) + Task 5.2 (applied once at forward() entry, scoped to tokamak 0D/sizing — non-0D concepts keep their tuned YAML Z_eff, avoiding silent shifts in mirror/pb11 example outputs). Section 7 (pb11 proxy guard) → Task 6. Section 8 tests 1-7 → Tasks 1 (fits), 3 (dilution/Z_eff algebra), 4-5 (DT regression via full suite), 7 (sizing + override).
- The non-0D path keeps using YAML `dhe3_dd_frac`/`Z_eff` everywhere — verified no non-0D call site reads the new keys.
- Type consistency: `fusion_power_density` returns `(p_fus, frac)` everywhere; `PlasmaState.dhe3_dd_frac_eff` is the only new state field; `compute_beta_N` has 7 params in definition and all call sites.
