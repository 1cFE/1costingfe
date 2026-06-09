# Tokamak Power-to-Geometry Sizing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a sizing solve that maps target net electric power to tokamak geometry (R0, a, B0) so that cost scales with power, for D-T fuel as the reference concept.

**Architecture:** A new pure solver `tokamak_size_from_power` bisects the major radius R0 so that the net electric power at the constraint-boundary operating point matches the target. It reuses the existing `tokamak_0d_forward` plus `mfe_forward_power_balance` for the per-trial physics, derives on-axis field from a magnet-selection table, and feeds the solved geometry into the unchanged geometry-to-cost pipeline. An optional outer loop minimizes LCOE over the Greenwald fraction.

**Tech Stack:** Python, JAX (jax.numpy) for the existing physics functions, pytest. The solver itself runs in plain Python (calls the jnp physics, converts scalars with `float()`); it is not differentiated through.

**Scope:** D-T tokamak only. Multi-fuel reactivity (DD/DHe3/pB11) is a tracked follow-on (the solver threads `fuel` through but the 0D model's `compute_fusion_power` is D-T only today). Other concepts (stellarator, mirror, ICF) are deferred.

**Reference spec:** `docs/superpowers/specs/2026-06-09-tokamak-power-sizing-design.md`

---

## File Structure

- `src/costingfe/defaults.py` — add the magnet-selection property table and accessor.
- `src/costingfe/layers/tokamak.py` — add `b0_from_radial_build`, `net_electric_at_R0`, `tokamak_size_from_power`. These are the new physics units.
- `src/costingfe/data/defaults/steady_state_tokamak.yaml` — add design-knob inputs, gating flags, and solver bounds.
- `src/costingfe/model.py` — gate the solver into `forward()`, reject pinned outputs, source disruption params from YAML (remove inline fallbacks).
- `tests/test_tokamak_sizing.py` — new test module for the solver and integration.

---

## Task 1: Magnet selection property table

**Files:**
- Modify: `src/costingfe/defaults.py`
- Test: `tests/test_tokamak_sizing.py`

The `coil_material` field already exists in the tokamak YAML (value `rebco_hts`). This task makes its physical properties (peak field ceiling, recirculating-power factor) available through a lookup instead of loose numbers.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tokamak_sizing.py
import pytest
from costingfe.defaults import get_magnet_properties, MAGNET_TABLE


def test_magnet_table_has_expected_materials():
    for key in ("rebco_hts", "nb3sn", "nbti", "copper"):
        assert key in MAGNET_TABLE


def test_get_magnet_properties_rebco():
    props = get_magnet_properties("rebco_hts")
    assert props.b_max == pytest.approx(23.0)
    assert props.recirc_power_factor == 0.0


def test_get_magnet_properties_copper_has_recirc():
    props = get_magnet_properties("copper")
    assert props.recirc_power_factor > 0.0


def test_get_magnet_properties_unknown_raises():
    with pytest.raises(KeyError):
        get_magnet_properties("unobtanium")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tokamak_sizing.py -k magnet -v`
Expected: FAIL with `ImportError: cannot import name 'get_magnet_properties'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/costingfe/defaults.py  (add near the other module-level data tables)
from dataclasses import dataclass


@dataclass(frozen=True)
class MagnetProperties:
    """Physical properties carried by a coil-conductor selection."""

    b_max: float            # Peak field ceiling at the conductor [T]
    recirc_power_factor: float  # Recirculating power as MW per (T^2 * m^3) of coil
    # volume-field product; 0 for superconductors, nonzero for resistive copper.
    cryo_temp_k: float      # Coil operating temperature [K]


# Peak-field ceilings: REBCO HTS ~23 T, Nb3Sn ~13 T, NbTi ~9 T (superconductors,
# zero recirculating power). Copper is resistive: stress/cooling-limited field,
# continuous dissipation. Values are engineering ceilings, sourced here rather
# than inline in any solver.
MAGNET_TABLE = {
    "rebco_hts": MagnetProperties(b_max=23.0, recirc_power_factor=0.0, cryo_temp_k=20.0),
    "nb3sn": MagnetProperties(b_max=13.0, recirc_power_factor=0.0, cryo_temp_k=4.5),
    "nbti": MagnetProperties(b_max=9.0, recirc_power_factor=0.0, cryo_temp_k=4.5),
    "copper": MagnetProperties(b_max=8.0, recirc_power_factor=2.0e-4, cryo_temp_k=300.0),
}


def get_magnet_properties(coil_material: str) -> MagnetProperties:
    """Look up magnet properties by conductor selection. Raises KeyError on
    an unknown material rather than substituting a default."""
    return MAGNET_TABLE[coil_material]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tokamak_sizing.py -k magnet -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/costingfe/defaults.py tests/test_tokamak_sizing.py
git commit -m "Add magnet selection property table"
```

---

## Task 2: On-axis field from radial build

**Files:**
- Modify: `src/costingfe/layers/tokamak.py`
- Test: `tests/test_tokamak_sizing.py`

On-axis field falls off as 1/R from the inboard coil leg. A small machine loses more of the peak field to fixed-meter blanket and shield, which is the physical reason high-field magnets matter more at small size.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tokamak_sizing.py
from costingfe.layers.tokamak import b0_from_radial_build


def test_b0_below_bmax_and_grows_with_size():
    # Same magnet, two machine sizes; the larger machine keeps more of B_max.
    thick = dict(blanket_t=0.8, ht_shield_t=0.2, structure_t=0.2, vessel_t=0.2)
    b_small = b0_from_radial_build(R0=3.0, a=1.0, b_max=23.0, **thick)
    b_large = b0_from_radial_build(R0=6.0, a=2.0, b_max=23.0, **thick)
    assert 0.0 < b_small < 23.0
    assert b_small < b_large < 23.0


def test_b0_formula():
    # B0 = B_max * (R0 - a - sum_thick) / R0
    b = b0_from_radial_build(
        R0=4.0, a=1.0, b_max=20.0,
        blanket_t=0.5, ht_shield_t=0.2, structure_t=0.2, vessel_t=0.1,
    )
    expected = 20.0 * (4.0 - 1.0 - 1.0) / 4.0
    assert b == pytest.approx(expected)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tokamak_sizing.py -k b0 -v`
Expected: FAIL with `ImportError: cannot import name 'b0_from_radial_build'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/costingfe/layers/tokamak.py  (add after the geometry helpers)
def b0_from_radial_build(R0, a, b_max, blanket_t, ht_shield_t, structure_t, vessel_t):
    """On-axis toroidal field [T] from the peak-field ceiling and the inboard
    radial build.

    The toroidal field falls as 1/R from the inboard TF leg to the axis:
        B0 = B_max * R_coil_inner / R0
    where R_coil_inner = R0 - a - (blanket + ht_shield + structure + vessel) is
    the major radius of the inboard coil leg. Fixed-meter inboard layers penalize
    small machines on field.
    """
    inboard = blanket_t + ht_shield_t + structure_t + vessel_t
    r_coil_inner = R0 - a - inboard
    return b_max * r_coil_inner / R0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tokamak_sizing.py -k b0 -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/costingfe/layers/tokamak.py tests/test_tokamak_sizing.py
git commit -m "Add on-axis field from radial build"
```

---

## Task 3: Net electric power at a fixed R0 (inner operating point)

**Files:**
- Modify: `src/costingfe/layers/tokamak.py`
- Test: `tests/test_tokamak_sizing.py`

At a trial R0 the operating point is pinned at the constraint boundary: density from Greenwald, and temperature chosen to maximize net power within the per-fuel `[T_min, T_max]` while keeping beta at or below the limit. This wraps `tokamak_0d_forward` + `mfe_forward_power_balance` (the existing "forward" branch of `_power_balance_0d`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tokamak_sizing.py
from costingfe.layers.tokamak import net_electric_at_R0


def _base_sizing_params():
    # Minimal param dict the evaluator needs (D-T tokamak, rebco).
    return dict(
        aspect_ratio=3.0, elon=1.85, q95=3.5, f_GW=0.85,
        b_max=23.0, beta_N_max=3.5, T_min=5.0, T_max=60.0,
        blanket_t=0.8, ht_shield_t=0.2, structure_t=0.2, vessel_t=0.2,
        p_input=50.0, mn=1.1, eta_th=0.45, eta_p=0.5, eta_pin=0.7,
        eta_de=0.85, f_sub=0.03, f_dec=0.0, p_coils=2.0, p_cool=13.7,
        p_pump=1.0, p_trit=10.0, p_house=4.0, p_cryo=0.5, Z_eff=1.5,
        M_ion=2.5, lambda_q=0.002, R_w=0.6, wall_material="W",
        T_edge=0.05, tau_ratio=3.0, recirc_power_factor=0.0,
        dd_f_T=0.969, dd_f_He3=0.689, dhe3_dd_frac=0.131, dhe3_f_T=0.5,
        dhe3_f_He3=0.5, pb11_f_alpha_n=0.0, pb11_f_p_n=0.0,
    )


def test_net_power_increases_with_R0():
    from costingfe.types import Fuel
    p = _base_sizing_params()
    pn_small = net_electric_at_R0(3.0, p, Fuel.DT)
    pn_large = net_electric_at_R0(5.0, p, Fuel.DT)
    assert pn_large > pn_small


def test_net_power_positive_for_reactor_scale():
    from costingfe.types import Fuel
    p = _base_sizing_params()
    assert net_electric_at_R0(4.5, p, Fuel.DT) > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tokamak_sizing.py -k net_power -v`
Expected: FAIL with `ImportError: cannot import name 'net_electric_at_R0'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/costingfe/layers/tokamak.py
from costingfe.layers.physics import mfe_forward_power_balance


def _net_at_R0_T(R0, T_e, params, fuel):
    """Net electric power [MW] at fixed R0 and operating temperature T_e."""
    A = params["aspect_ratio"]
    a = R0 / A
    kappa = params["elon"]
    b_max = params["b_max"]
    B0 = b0_from_radial_build(
        R0, a, b_max,
        params["blanket_t"], params["ht_shield_t"],
        params["structure_t"], params["vessel_t"],
    )
    fuel_frac_kw = dict(
        dd_f_T=params["dd_f_T"], dd_f_He3=params["dd_f_He3"],
        dhe3_dd_frac=params["dhe3_dd_frac"], dhe3_f_T=params["dhe3_f_T"],
        dhe3_f_He3=params["dhe3_f_He3"],
        pb11_f_alpha_n=params["pb11_f_alpha_n"], pb11_f_p_n=params["pb11_f_p_n"],
    )
    ps = tokamak_0d_forward(
        R=R0, a=a, kappa=kappa, B=B0, q95=params["q95"], f_GW=params["f_GW"],
        T_e=T_e, p_input=params["p_input"], fuel=fuel,
        M_ion=params["M_ion"], Z_eff=params["Z_eff"], lambda_q=params["lambda_q"],
        **fuel_frac_kw,
    )
    # Resistive coil recirculation (0 for superconductors).
    p_coils = params["p_coils"] + params["recirc_power_factor"] * B0**2 * ps.V_plasma
    pt = mfe_forward_power_balance(
        p_fus=ps.p_fus, fuel=fuel, p_input=params["p_input"], mn=params["mn"],
        eta_th=params["eta_th"], eta_p=params["eta_p"], eta_pin=params["eta_pin"],
        eta_de=params["eta_de"], f_sub=params["f_sub"], f_dec=params["f_dec"],
        p_coils=p_coils, p_cool=params["p_cool"], p_pump=params["p_pump"],
        p_trit=params["p_trit"], p_house=params["p_house"], p_cryo=params["p_cryo"],
        n_e=ps.n_e, T_e=ps.T_e, Z_eff=params["Z_eff"], plasma_volume=ps.V_plasma,
        B=B0, R_major=R0, a_minor=a, kappa=kappa, R_w=params["R_w"],
        wall_material=WallMaterial(params["wall_material"]),
        seeded_impurities=params.get("seeded_impurities") or None,
        T_edge=params["T_edge"], tau_ratio=params["tau_ratio"], fw_area=ps.fw_area,
        **fuel_frac_kw,
    )
    return float(pt.p_net), ps.beta_N


def net_electric_at_R0(R0, params, fuel, return_state=False):
    """Net electric power [MW] at the constraint-boundary operating point for a
    fixed R0. Temperature is chosen to maximize net power within [T_min, T_max]
    subject to beta <= beta_N_max, via golden-section search.
    """
    T_lo, T_hi = params["T_min"], params["T_max"]
    beta_cap = params["beta_N_max"]
    invphi = (5**0.5 - 1) / 2  # 0.618...

    def feasible_net(T):
        pn, beta = _net_at_R0_T(R0, T, params, fuel)
        # Penalize beta-limit violation so the optimizer backs off T.
        if beta > beta_cap:
            return -1e9 * (beta - beta_cap)
        return pn

    lo, hi = T_lo, T_hi
    for _ in range(40):
        c = hi - invphi * (hi - lo)
        d = lo + invphi * (hi - lo)
        if feasible_net(c) < feasible_net(d):
            lo = c
        else:
            hi = d
    T_star = 0.5 * (lo + hi)
    pn, beta = _net_at_R0_T(R0, T_star, params, fuel)
    if return_state:
        return pn, T_star, beta
    return pn
```

Note: import `WallMaterial` at the top of `tokamak.py` from `costingfe.types` alongside `Fuel`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tokamak_sizing.py -k net_power -v`
Expected: PASS (2 tests). If `pt.p_net` is not the net-electric attribute name, inspect `PowerTable` in `physics.py` and use the correct field; the monotonicity test will confirm.

- [ ] **Step 5: Commit**

```bash
git add src/costingfe/layers/tokamak.py tests/test_tokamak_sizing.py
git commit -m "Add constraint-boundary net-power evaluation at fixed R0"
```

---

## Task 4: Outer R0 bisection solver

**Files:**
- Modify: `src/costingfe/layers/tokamak.py`
- Test: `tests/test_tokamak_sizing.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tokamak_sizing.py
from costingfe.layers.tokamak import tokamak_size_from_power, SizingInfeasible


def test_size_hits_target():
    from costingfe.types import Fuel
    p = _base_sizing_params()
    p.update(R0_min=1.0, R0_max=12.0, net_electric_mw=500.0)
    result = tokamak_size_from_power(p, Fuel.DT)
    pn = net_electric_at_R0(result.R0, p, Fuel.DT)
    assert pn == pytest.approx(500.0, rel=0.02)
    assert result.a == pytest.approx(result.R0 / p["aspect_ratio"])
    assert 0.0 < result.B0 < p["b_max"]


def test_size_scales_with_power():
    from costingfe.types import Fuel
    p = _base_sizing_params()
    p.update(R0_min=1.0, R0_max=15.0)
    r1 = tokamak_size_from_power({**p, "net_electric_mw": 250.0}, Fuel.DT)
    r2 = tokamak_size_from_power({**p, "net_electric_mw": 2000.0}, Fuel.DT)
    assert r2.R0 > r1.R0  # bigger machine for more power
    # Roughly R0 ~ P^(1/3): 8x power -> about 2x R0 (loose bound)
    assert 1.5 < (r2.R0 / r1.R0) < 3.0


def test_infeasible_raises():
    from costingfe.types import Fuel
    p = _base_sizing_params()
    p.update(R0_min=1.0, R0_max=3.0, net_electric_mw=5000.0)  # too much power for tiny cap
    with pytest.raises(SizingInfeasible):
        tokamak_size_from_power(p, Fuel.DT)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tokamak_sizing.py -k size -v`
Expected: FAIL with `ImportError: cannot import name 'tokamak_size_from_power'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/costingfe/layers/tokamak.py
from dataclasses import dataclass


class SizingInfeasible(Exception):
    """Raised when no machine in [R0_min, R0_max] meets the power target."""


@dataclass(frozen=True)
class SizingResult:
    R0: float
    a: float
    B0: float
    T_e: float


def tokamak_size_from_power(params, fuel):
    """Solve major radius R0 so net electric power equals the target.

    Net power is monotonic increasing in R0 at the boundary operating point, so
    bisection is well posed. Raises SizingInfeasible if the target exceeds what
    R0_max can deliver.
    """
    target = params["net_electric_mw"]
    lo, hi = params["R0_min"], params["R0_max"]

    pn_hi = net_electric_at_R0(hi, params, fuel)
    if pn_hi < target:
        raise SizingInfeasible(
            f"net power at R0_max={hi} m is {pn_hi:.1f} MW < target {target} MW; "
            "machine cannot reach the power with these physics inputs"
        )

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if net_electric_at_R0(mid, params, fuel) < target:
            lo = mid
        else:
            hi = mid
    R0 = 0.5 * (lo + hi)
    a = R0 / params["aspect_ratio"]
    _, T_star, _ = net_electric_at_R0(R0, params, fuel, return_state=True)
    B0 = b0_from_radial_build(
        R0, a, params["b_max"], params["blanket_t"], params["ht_shield_t"],
        params["structure_t"], params["vessel_t"],
    )
    return SizingResult(R0=R0, a=a, B0=B0, T_e=T_star)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tokamak_sizing.py -k size -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/costingfe/layers/tokamak.py tests/test_tokamak_sizing.py
git commit -m "Add R0 bisection sizing solver"
```

---

## Task 5: YAML inputs and override-key whitelist

**Files:**
- Modify: `src/costingfe/data/defaults/steady_state_tokamak.yaml`
- Modify: `src/costingfe/model.py:1153-1179` (`_OPTIONAL_OVERRIDE_KEYS`)
- Test: `tests/test_tokamak_sizing.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tokamak_sizing.py
from costingfe.model import CostingModel
from costingfe.types import ConfinementConcept, Fuel


def test_sizing_overrides_accepted():
    m = CostingModel(ConfinementConcept.TOKAMAK, Fuel.DT)
    # Must not raise "unknown parameter": these are sizing knobs.
    m.forward(
        net_electric_mw=500.0, availability=0.85, lifetime_yr=30.0,
        size_from_power=True, aspect_ratio=3.1, beta_N_max=3.5, H_factor=1.0,
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tokamak_sizing.py -k overrides_accepted -v`
Expected: FAIL with `ValueError: ... unknown parameter(s) ... aspect_ratio ...`

- [ ] **Step 3: Add YAML inputs**

Append to `src/costingfe/data/defaults/steady_state_tokamak.yaml`:

```yaml
# Power-to-geometry sizing (active when size_from_power=true). When enabled,
# R0/plasma_t/B/b_center become solved outputs and must not be pinned.
size_from_power: false   # Solve R0 from net_electric_mw instead of using R0
optimize_lcoe: false     # Wrap sizing with an outer LCOE minimization over f_GW
aspect_ratio: 3.0        # A = R0 / a (conventional ~3, spherical ~1.7)
beta_N_max: 3.5          # Troyon normalized-beta limit (binding constraint)
H_factor: 1.0            # Assumed confinement quality (IPB98y2 multiplier)
R0_min: 1.0              # Lower bound for the R0 search [m]
R0_max: 12.0             # Upper bound for the R0 search [m]
T_min: 5.0               # Lower bound for the operating-temperature search [keV]
T_max: 60.0              # Upper bound for the operating-temperature search [keV]
```

- [ ] **Step 4: Add override keys**

In `src/costingfe/model.py`, add to the `_OPTIONAL_OVERRIDE_KEYS` frozenset (after the `"0d_mode"` entry):

```python
            # Power-to-geometry sizing (TOKAMAK)
            "size_from_power",
            "optimize_lcoe",
            "aspect_ratio",
            "beta_N_max",
            "H_factor",
            "R0_min",
            "R0_max",
            "T_min",
            "T_max",
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_tokamak_sizing.py -k overrides_accepted -v`
Expected: PASS (the sizing solve is not wired yet, but the override validation now accepts the keys). If it fails later in `forward()` because the gate is not implemented, that is expected and handled in Task 6; narrow the assertion to the validation step if needed by catching only the unknown-parameter path.

- [ ] **Step 6: Commit**

```bash
git add src/costingfe/data/defaults/steady_state_tokamak.yaml src/costingfe/model.py tests/test_tokamak_sizing.py
git commit -m "Add sizing inputs to tokamak YAML and override whitelist"
```

---

## Task 6: Gate the solver into forward()

**Files:**
- Modify: `src/costingfe/model.py:606-607` (replace the `_power_balance` call site for the sizing path)
- Modify: `src/costingfe/model.py` (add a `_size_tokamak` helper near `_power_balance_0d`)
- Test: `tests/test_tokamak_sizing.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tokamak_sizing.py
def test_sizing_runs_end_to_end_and_sets_geometry():
    m = CostingModel(ConfinementConcept.TOKAMAK, Fuel.DT)
    r = m.forward(
        net_electric_mw=500.0, availability=0.85, lifetime_yr=30.0,
        size_from_power=True, aspect_ratio=3.1, beta_N_max=3.5,
        coil_material="rebco_hts",
    )
    assert r.lcoe > 0.0
    assert m._plasma_state is not None


def test_pinning_R0_in_sizing_mode_raises():
    m = CostingModel(ConfinementConcept.TOKAMAK, Fuel.DT)
    with pytest.raises(ValueError, match="cannot be pinned"):
        m.forward(
            net_electric_mw=500.0, availability=0.85, lifetime_yr=30.0,
            size_from_power=True, R0=3.0,
        )


def test_coil_cost_scales_with_power():
    # The original issue #4 assertion: C220103 must move with power.
    m = CostingModel(ConfinementConcept.TOKAMAK, Fuel.DT)
    common = dict(availability=0.85, lifetime_yr=30.0, size_from_power=True,
                  aspect_ratio=3.1, beta_N_max=3.5, coil_material="rebco_hts")
    lo = m.forward(net_electric_mw=200.0, **common)
    hi = m.forward(net_electric_mw=1500.0, **common)
    c_lo = lo.costs.detail["C220103"]
    c_hi = hi.costs.detail["C220103"]
    assert c_hi > c_lo * 1.2  # coils genuinely scale up with power
```

Note: confirm the cost-detail accessor (`r.costs.detail["C220103"]`) against `CostResult` in the codebase; adjust the key access to match the actual structure.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tokamak_sizing.py -k "end_to_end or pinning or scales_with_power" -v`
Expected: FAIL (sizing not gated; `size_from_power` currently ignored so R0 stays fixed and coil cost is constant).

- [ ] **Step 3: Add the sizing helper**

In `src/costingfe/model.py`, add after `_power_balance_0d`:

```python
    def _size_tokamak(self, params, n_mod):
        """Solve geometry from the power target, inject it into params, and
        return the power table. Mutates params with solved R0/a/B/b_center."""
        from costingfe.defaults import get_magnet_properties
        from costingfe.layers.tokamak import (
            tokamak_size_from_power,
            net_electric_at_R0,
            SizingResult,
        )

        props = get_magnet_properties(params["coil_material"])
        solve_params = dict(params)
        solve_params["b_max"] = props.b_max
        solve_params["recirc_power_factor"] = props.recirc_power_factor
        # Size one module to the per-module net power.
        solve_params["net_electric_mw"] = params["net_electric_mw"] / n_mod

        result = tokamak_size_from_power(solve_params, self.fuel)

        # Inject solved geometry so downstream geometry and coil cost use it.
        params["R0"] = result.R0
        params["plasma_t"] = result.a
        params["B"] = result.B0
        params["b_center"] = result.B0
        params["T_e"] = result.T_e

        # Re-run the forward 0D power balance at the solved point to produce the
        # full PowerTable and plasma state for the rest of the pipeline.
        params["0d_mode"] = "forward"
        return self._power_balance_0d(params, n_mod)
```

- [ ] **Step 4: Wire the gate at the power-balance call site**

Replace `src/costingfe/model.py:607` (`pt = self._power_balance(params, n_mod)`) with:

```python
        # Layer 2: Power balance (dispatched by family), or sizing solve.
        if params.get("size_from_power", False):
            if self.concept != ConfinementConcept.TOKAMAK:
                raise ValueError(
                    "size_from_power is implemented for TOKAMAK only"
                )
            pinned = [k for k in ("R0", "plasma_t", "b_center", "B")
                      if k in overrides]
            if pinned:
                raise ValueError(
                    f"{pinned} cannot be pinned in size_from_power mode; they "
                    "are solved. Remove them or set size_from_power=False."
                )
            pt = self._size_tokamak(params, n_mod)
        else:
            pt = self._power_balance(params, n_mod)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_tokamak_sizing.py -k "end_to_end or pinning or scales_with_power" -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add src/costingfe/model.py tests/test_tokamak_sizing.py
git commit -m "Gate power-to-geometry sizing into forward()"
```

---

## Task 7: Source disruption params from YAML (remove inline fallbacks)

**Files:**
- Modify: `src/costingfe/model.py:874-878`
- Test: `tests/test_tokamak_sizing.py`

The inline `params.get("disruption_rate_base", 0.1)` fallbacks at lines 874-878 are the `arg=default` pattern that placement discipline forbids. The four keys exist in the tokamak YAML, so the fallback is dead weight that also hides a missing key.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tokamak_sizing.py
def test_missing_disruption_key_errors_not_silent():
    m = CostingModel(ConfinementConcept.TOKAMAK, Fuel.DT)
    # Simulate a params dict missing a disruption key under the 0D path; the
    # model must read it as a required input, not silently default.
    import pytest
    m2 = CostingModel(ConfinementConcept.TOKAMAK, Fuel.DT)
    m2._eng_defaults = dict(m2._eng_defaults)
    m2._eng_defaults.pop("disruption_damage")
    with pytest.raises(KeyError):
        m2.forward(net_electric_mw=400.0, availability=0.85, lifetime_yr=30.0,
                   use_0d_model=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tokamak_sizing.py -k missing_disruption -v`
Expected: FAIL (currently the inline `0.1`/`15.0`/`0.02`/`72.0` fallbacks swallow the missing key, so no error is raised).

- [ ] **Step 3: Replace inline fallbacks with required reads**

In `src/costingfe/model.py`, change lines 874-878 from:

```python
                rate_base=params.get("disruption_rate_base", 0.1),
                steepness=params.get("disruption_steepness", 15.0),
                damage_per_disruption=params.get("disruption_damage", 0.02),
                downtime_per_disruption=params.get("disruption_downtime", 72.0),
```

to:

```python
                rate_base=params["disruption_rate_base"],
                steepness=params["disruption_steepness"],
                damage_per_disruption=params["disruption_damage"],
                downtime_per_disruption=params["disruption_downtime"],
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tokamak_sizing.py -k missing_disruption -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/costingfe/model.py tests/test_tokamak_sizing.py
git commit -m "Source disruption params from YAML, drop inline fallbacks"
```

---

## Task 8: Optimize mode (LCOE over f_GW)

**Files:**
- Modify: `src/costingfe/model.py` (add `_optimize_fgw` and branch in the sizing gate)
- Test: `tests/test_tokamak_sizing.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tokamak_sizing.py
def test_optimize_returns_valid_fgw_and_lcoe():
    m = CostingModel(ConfinementConcept.TOKAMAK, Fuel.DT)
    r = m.forward(
        net_electric_mw=500.0, availability=0.85, lifetime_yr=30.0,
        size_from_power=True, optimize_lcoe=True, aspect_ratio=3.1,
        beta_N_max=3.5, coil_material="rebco_hts",
        disruption_rate_base=1.0, disruption_damage=0.1,
    )
    assert r.lcoe > 0.0
    # The chosen f_GW is recorded on the result/params and within bounds.
    assert 0.0 < m._sizing_fgw <= 1.0


def test_optimize_lcoe_no_worse_than_default_fgw():
    m = CostingModel(ConfinementConcept.TOKAMAK, Fuel.DT)
    common = dict(net_electric_mw=500.0, availability=0.85, lifetime_yr=30.0,
                  size_from_power=True, aspect_ratio=3.1, beta_N_max=3.5,
                  coil_material="rebco_hts", disruption_rate_base=1.0,
                  disruption_damage=0.1)
    fixed = m.forward(f_GW=0.85, **common)
    opt = m.forward(optimize_lcoe=True, **common)
    assert opt.lcoe <= fixed.lcoe + 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tokamak_sizing.py -k optimize -v`
Expected: FAIL with `AttributeError: ... _sizing_fgw` (optimize path not implemented).

- [ ] **Step 3: Implement the outer minimization**

In `src/costingfe/model.py`, add a helper and branch. The optimizer re-runs the full sizing-plus-cost pipeline for trial f_GW values and keeps the minimum-LCOE one. Add to `_size_tokamak` (or a wrapper called from the gate):

```python
    def _optimize_fgw(self, params, n_mod, run_pipeline):
        """Golden-section minimize LCOE over f_GW in (0.3, 1.0]. run_pipeline is
        a callable(f_GW) -> lcoe that runs sizing + cost with that f_GW."""
        invphi = (5**0.5 - 1) / 2
        lo, hi = 0.3, 1.0
        for _ in range(25):
            c = hi - invphi * (hi - lo)
            d = lo + invphi * (hi - lo)
            if run_pipeline(c) < run_pipeline(d):
                hi = d
            else:
                lo = c
        return 0.5 * (lo + hi)
```

Wire it so that when `optimize_lcoe` is true the gate first finds the best f_GW (running the pipeline through to LCOE per trial), records it as `self._sizing_fgw`, sets `params["f_GW"]` to it, and then runs the final pipeline once. Because the optimizer runs the full pipeline per trial, structure it to call a thin closure that builds a fresh params copy, sets `f_GW`, runs `_size_tokamak` plus the downstream cost layers, and returns `lcoe`. Set `self._sizing_fgw = params["f_GW"]` in the non-optimize path too (so the attribute always exists).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tokamak_sizing.py -k optimize -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/costingfe/model.py tests/test_tokamak_sizing.py
git commit -m "Add LCOE-optimize-over-fGW sizing mode"
```

---

## Task 9: Integration tests and backward-compat guard

**Files:**
- Test: `tests/test_tokamak_sizing.py`

- [ ] **Step 1: Write the integration tests**

```python
# tests/test_tokamak_sizing.py
def test_backward_compat_sizing_off_unchanged():
    # With size_from_power absent/false, output must equal the pre-feature path.
    m = CostingModel(ConfinementConcept.TOKAMAK, Fuel.DT)
    r = m.forward(net_electric_mw=500.0, availability=0.85, lifetime_yr=30.0)
    assert r.lcoe > 0.0
    # R0 stays at the YAML default (not solved).
    # (Assert against a frozen reference value captured from master.)


def test_arc_validation_reproduces_size():
    # Inject ARC-like design knobs; solved R0 should land near 3.3 m.
    m = CostingModel(ConfinementConcept.TOKAMAK, Fuel.DT)
    m.forward(
        net_electric_mw=270.0, availability=0.85, lifetime_yr=30.0,
        size_from_power=True, coil_material="rebco_hts", aspect_ratio=3.0,
        elon=1.85, beta_N_max=3.0, q95=3.5, f_GW=0.85,
    )
    from costingfe.defaults import get_magnet_properties
    # Pull the solved R0 off the model's last params/plasma state.
    assert 2.5 < m._last_R0 < 4.5  # ARC R0 ~3.3 m, allow modeling spread


def test_magnet_differentiation_rebco_smaller_than_nb3sn():
    m = CostingModel(ConfinementConcept.TOKAMAK, Fuel.DT)
    common = dict(net_electric_mw=500.0, availability=0.85, lifetime_yr=30.0,
                  size_from_power=True, aspect_ratio=3.1, beta_N_max=3.5)
    m.forward(coil_material="rebco_hts", **common)
    r_rebco = m._last_R0
    m.forward(coil_material="nb3sn", **common)
    r_nb3sn = m._last_R0
    assert r_rebco < r_nb3sn  # higher field -> smaller machine


def test_scale_mode_grows_R0_with_power():
    m = CostingModel(ConfinementConcept.TOKAMAK, Fuel.DT)
    common = dict(availability=0.85, lifetime_yr=30.0, size_from_power=True,
                  coil_material="rebco_hts", aspect_ratio=3.0, beta_N_max=3.0)
    m.forward(net_electric_mw=270.0, **common)
    r_small = m._last_R0
    m.forward(net_electric_mw=1000.0, **common)
    r_large = m._last_R0
    assert r_large > r_small
```

Note: this task introduces `self._last_R0`, set in `_size_tokamak` after solving (`self._last_R0 = result.R0`). Add that one-line assignment when implementing.

- [ ] **Step 2: Run the tests**

Run: `pytest tests/test_tokamak_sizing.py -v`
Expected: PASS for all. For `test_backward_compat_sizing_off_unchanged`, first capture the reference LCOE on master (`git stash`, run forward, record value) and assert equality to that frozen number.

- [ ] **Step 3: Run the full suite for regressions**

Run: `pytest -q`
Expected: All previously passing tests still pass (sizing defaults off, so existing configs are untouched).

- [ ] **Step 4: Commit**

```bash
git add tests/test_tokamak_sizing.py src/costingfe/model.py
git commit -m "Add sizing integration tests (ARC, magnet differentiation, scale)"
```

---

## Task 10: Documentation

**Files:**
- Modify: `docs/account_justification/CAS22_reactor_components.md` (coil sizing note)
- Modify: `docs/papers/1costingfe_paper/1costingfe_paper.tex` (sizing method paragraph)

- [ ] **Step 1: Document the sizing method in the account justification**

Add a section describing the constraint-boundary solve, the magnet table, the B0-from-radial-build relation, and that R0 is solved rather than input. Bare `$`, no em dashes, no tildes.

- [ ] **Step 2: Add the paper paragraph**

Add a short paragraph to the tokamak/coil section of the paper describing power-to-geometry sizing and the four modes (pin, size, scale, optimize), with the B0 relation. Do not document past behavior or history (paper.tex carries no history).

- [ ] **Step 3: Build the paper to confirm no LaTeX errors**

Run the paper build command used in the repo (check `docs/papers/1costingfe_paper/` for the Makefile or latexmk invocation).
Expected: builds clean.

- [ ] **Step 4: Commit**

```bash
git add docs/account_justification/CAS22_reactor_components.md docs/papers/1costingfe_paper/1costingfe_paper.tex
git commit -m "Document power-to-geometry sizing in justification and paper"
```

---

## Self-Review Notes

- **Spec coverage:** pin mode (Task 6, default off), size mode (Tasks 3-6), scale mode (Task 9 scale test), optimize mode (Task 8); magnet table (Task 1); B0 from radial build (Task 2); fuel threaded through but D-T only (Tasks 3-4, scope note); placement discipline (magnet table in defaults.py Task 1, disruption inputs Task 7); fusion-tea injection via overrides (Task 5 whitelist); backward compat (Task 9). All spec sections map to a task.
- **Deferred per spec:** multi-fuel reactivity, other concepts, freeing more optimizer knobs, anchor-on-raw-geometry back-out. Not in this plan by design.
- **Known verification points for the executor:** confirm the `PowerTable` net-electric attribute name (assumed `p_net`); confirm the `CostResult` cost-detail accessor (assumed `costs.detail["C220103"]`); confirm `mfe_forward_power_balance` impurity kwargs match Task 3. Each surfaces immediately as a test failure if wrong.
- **Disruption values:** the lit-review base values (rate_base 1.0, damage 0.01, downtime 72, steepness 12) from `docs/account_justification/disruption_severity.md` should replace the current YAML defaults as a follow-up; Task 8 tests pass severe values explicitly so they do not depend on the YAML default.
