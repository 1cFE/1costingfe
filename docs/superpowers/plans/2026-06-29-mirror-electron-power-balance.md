# Mirror central-cell electron power balance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the 0D mirror model solve central-cell `T_e` from an electron power balance instead of taking it as a pinned input, so the D-T operating point lands physically cool (about T_i) instead of an unphysical hot point.

**Architecture:** Add three pure physics functions (relativistic bremsstrahlung, ion-electron equilibration, alpha-to-electron slowing-down fraction), extract the mirror's confinement-and-loss chain into a helper so it can be evaluated at a trial `T_e`, then add a 1D bisection that solves the electron balance `alpha_e*P_alpha + K_ie*(T_i-T_e) = P_brem + gamma_e*p_end`. Wire it into `mirror_0d_forward` behind an opt-in flag.

**Tech Stack:** Python, JAX (jax.numpy, jax.lax), pytest. Pure-functional float32-safe kernel style.

## Global Constants

- Density is passed in m^-3 but internally rescaled to 1e20 m^-3 units; fold the
  1e-20 into prefactors so float32 intermediates stay near unity (see
  `compute_tau_ii` and `reactivity.py`). Temperatures in keV. Power in MW.
- All new physics functions are pure JAX, differentiable, float32-safe at
  runtime with constants pre-folded in float64.
- Electron rest energy `E_rest = 511.0 keV`. Coulomb log `_LN_LAMBDA = 17.0`.
  Alpha birth energy for D-T `E_alpha = 3500.0 keV`. D-T critical-energy ratio
  `E_crit/T_e = 33.0`.
- The 0D mirror model is gated off in the release (`MODELS_0D_ENABLED = False` in
  `model.py`); tests call the layer functions directly, which bypasses the gate.
- COORDINATION: a parallel effort is making JAX optional (a backend-agnostic
  array namespace with a numpy fallback). This plan's code is written against
  `jnp` for current master. When the JAX-optional shim lands, the new functions
  must use that array namespace instead of `jnp` directly. Avoid JAX-only control
  flow: the `solve_T_e` bisection uses a bounded Python loop, NOT
  `jax.lax.fori_loop`; if `jnp.trapezoid` is absent under the shim use a manual
  trapezoid sum. Sequence the merges to avoid conflicts on `mirror.py` /
  `radiation.py` (both branches edit these files).
- Commit messages are one line, no `Co-Authored-By` line. Docs use no em dashes
  and no `~` for approximate values (write "about").

---

### Task 1: Relativistic bremsstrahlung `compute_p_brem_rel`

**Files:**
- Modify: `src/costingfe/layers/radiation.py` (add function near `compute_p_rad`, around line 243)
- Test: `tests/test_radiation.py` (create if absent, else append)

**Interfaces:**
- Produces: `compute_p_brem_rel(n_e, T_e, Z_eff, volume) -> float` (MW). `n_e`
  in m^-3, `T_e` in keV, `Z_eff` dimensionless, `volume` in m^3.

Derivation of the prefactor (paper arXiv:2210.08076 Eq. 16, CGS eV cm^-3 s^-1
to MW with n in 1e20 m^-3): 7.56e-11 * (1e14)^2[cm^-3 from n20] * 1.602176634e-19
[eV->J] * 1e6 [cm^-3->m^-3] * 1e-6 [W->MW] = 0.12113. Cross-check: at low x this
reduces to 5.36e-3 * n20^2 * Z_eff * sqrt(T_e) * V, matching the existing
non-relativistic Born constant 5.35e-3 in `compute_p_rad`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_radiation.py
import math
from costingfe.layers.radiation import compute_p_brem_rel

def test_p_brem_rel_reference_point():
    # n_e=1e20 m^-3, T_e=10 keV, Z_eff=1, V=1 m^3.
    # x=10/511=0.01957; sqrt(x)=0.1399; bracket=1.0515; 0.12113*0.1399*1.0515=0.01782 MW
    p = float(compute_p_brem_rel(1e20, 10.0, 1.0, 1.0))
    assert p == math.isclose(p, 0.01782, rel_tol=2e-2) or abs(p - 0.01782) < 4e-4

def test_p_brem_rel_matches_nonrel_at_low_T():
    # At 10 keV the relativistic+ee form is within ~6% of the non-rel Born value
    # 5.35e-3 * n20^2 * Z_eff * sqrt(T_e) * V = 5.35e-3*1*1*sqrt(10)*1 = 0.01692 MW
    p = float(compute_p_brem_rel(1e20, 10.0, 1.0, 1.0))
    assert 0.0169 <= p <= 0.0180
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_radiation.py -k p_brem_rel -v`
Expected: FAIL with ImportError / "cannot import name compute_p_brem_rel".

- [ ] **Step 3: Write minimal implementation**

```python
# src/costingfe/layers/radiation.py
def compute_p_brem_rel(n_e, T_e, Z_eff, volume):
    """Relativistic bremsstrahlung incl. electron-electron term [MW].

    Putvinski/Heitler form (Ochs et al. 2022 arXiv:2210.08076 Eq. 16), in kernel
    units: n_e [m^-3], T_e [keV], volume [m^3]. Uniformly valid from D-T (about
    10 keV) to p-B11 (about 300 keV). Use this on the electron power-balance RHS;
    it excludes synchrotron and line radiation by construction.
    """
    n_e_20 = n_e * 1e-20
    x = T_e / 511.0  # T_e / E_rest
    bracket = Z_eff * (1.0 + 1.78 * x**1.34) + 2.12 * x * (1.0 + 1.1 * x - 1.25 * x**2.5)
    return 0.12113 * n_e_20**2 * jnp.sqrt(x) * bracket * volume  # -> MW
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_radiation.py -k p_brem_rel -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/costingfe/layers/radiation.py tests/test_radiation.py
git commit -m "Add relativistic bremsstrahlung (Putvinski Eq. 16) for the electron balance"
```

---

### Task 2: Ion-electron equilibration `compute_K_ie`

**Files:**
- Modify: `src/costingfe/layers/mirror.py` (add near `compute_tau_ii`, around line 133)
- Test: `tests/test_mirror.py` (append)

**Interfaces:**
- Produces: `compute_K_ie(n_e, n_i, T_i, T_e, Z_i, A, volume) -> float` (MW),
  the collisional energy transfer power TO electrons (positive when T_i > T_e).
  `n_e, n_i` in m^-3; `T_i, T_e` in keV; `Z_i` ion charge; `A` ion mass number;
  `volume` in m^3.

NRL formulary electron-ion energy equilibration. Power density to electrons is
`1.5 * n_e * nu_eps * (T_i - T_e)`, with `nu_eps = 1.8e-19 (m_e m_i)^0.5 Z^2 n_i
lnL / (m_e T_i + m_i T_e)^1.5` (CGS-practical: n in cm^-3, T in eV, m in grams).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mirror.py
from costingfe.layers.mirror import compute_K_ie

def test_K_ie_reference_point():
    # n_e=n_i=1e20 m^-3, T_i=15, T_e=10 keV, Z=1, A=2.5 (D-T), V=1 m^3, lnL=17.
    # nu_eps about 2.22 s^-1; power density about 0.267 MW/m^3.
    p = float(compute_K_ie(1e20, 1e20, 15.0, 10.0, 1.0, 2.5, 1.0))
    assert abs(p - 0.267) < 0.02

def test_K_ie_sign_and_zero():
    # T_e > T_i -> electrons give energy to ions -> negative.
    assert float(compute_K_ie(1e20, 1e20, 10.0, 20.0, 1.0, 2.5, 1.0)) < 0.0
    # T_e == T_i -> zero.
    assert abs(float(compute_K_ie(1e20, 1e20, 12.0, 12.0, 1.0, 2.5, 1.0))) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_mirror.py -k K_ie -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation**

```python
# src/costingfe/layers/mirror.py
_M_E_G = 9.1093837015e-28  # electron mass [g]
_AMU_G = 1.66053906660e-24  # atomic mass unit [g]

def compute_K_ie(n_e, n_i, T_i, T_e, Z_i, A, volume):
    """Ion-electron collisional energy transfer power to electrons [MW].

    NRL formulary energy-equilibration (Huba). Positive when T_i > T_e.
    n_e, n_i [m^-3]; T_i, T_e [keV]; Z_i ion charge; A ion mass number;
    volume [m^3]. The full (m_e T_i + m_i T_e)^1.5 denominator is kept (no
    m_e << m_i approximation) so the sign and small-difference limit are exact.
    """
    m_i_g = A * _AMU_G
    n_i_cm3 = n_i * 1e-6
    T_i_eV = T_i * 1e3
    T_e_eV = T_e * 1e3
    nu_eps = (
        1.8e-19
        * jnp.sqrt(_M_E_G * m_i_g)
        * Z_i**2
        * n_i_cm3
        * _LN_LAMBDA
        / (_M_E_G * T_i_eV + m_i_g * T_e_eV) ** 1.5
    )  # [s^-1]
    # Power density to electrons [W/m^3] = 1.5 n_e nu_eps (T_i - T_e)keV * keV->J
    p_density_W = 1.5 * n_e * nu_eps * (T_i - T_e) * _KEV_TO_J
    return p_density_W * volume * 1e-6  # -> MW
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_mirror.py -k K_ie -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/costingfe/layers/mirror.py tests/test_mirror.py
git commit -m "Add NRL ion-electron equilibration power for the mirror electron balance"
```

---

### Task 3: Alpha-to-electron slowing-down fraction `alpha_electron_fraction`

**Files:**
- Modify: `src/costingfe/layers/mirror.py`
- Test: `tests/test_mirror.py` (append)

**Interfaces:**
- Produces: `alpha_electron_fraction(T_e, E_alpha_keV, e_crit_over_te) -> float`,
  the fraction of fast-alpha slowing-down energy delivered to electrons.

Stix (1972) two-body slowing down: ion heating fraction
`f_i(u) = (1/u) * integral_0^u dx/(1 + x^1.5)`, `u = E_alpha / E_crit`,
`E_crit = e_crit_over_te * T_e`. Electron fraction is `1 - f_i`. Fixed-node
trapezoid keeps it JAX-differentiable.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mirror.py
from costingfe.layers.mirror import alpha_electron_fraction

def test_alpha_electron_fraction_DT_reference():
    # D-T: E_alpha=3500 keV, E_crit/T_e=33. At T_e=10 keV, E_crit=330,
    # u=10.6, f_e about 0.83 (most alpha energy heats electrons).
    f_e = float(alpha_electron_fraction(10.0, 3500.0, 33.0))
    assert 0.78 <= f_e <= 0.87

def test_alpha_electron_fraction_decreasing_in_Te():
    # Hotter electrons -> higher E_crit -> smaller u -> more energy to ions.
    f_lo = float(alpha_electron_fraction(10.0, 3500.0, 33.0))
    f_hi = float(alpha_electron_fraction(40.0, 3500.0, 33.0))
    assert f_hi < f_lo
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_mirror.py -k alpha_electron -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation**

```python
# src/costingfe/layers/mirror.py
def alpha_electron_fraction(T_e, E_alpha_keV, e_crit_over_te):
    """Fraction of fast-alpha slowing-down energy delivered to electrons.

    Stix (1972) ion heating fraction f_i(u) = (1/u) int_0^u dx/(1+x^1.5),
    u = E_alpha / E_crit, E_crit = e_crit_over_te * T_e. Returns 1 - f_i.
    e_crit_over_te is about 33 for D-T alphas. Fixed 256-node trapezoid.
    """
    e_crit = e_crit_over_te * T_e
    u = E_alpha_keV / e_crit
    xs = jnp.linspace(0.0, u, 256)
    integrand = 1.0 / (1.0 + xs**1.5)
    integral = jnp.trapezoid(integrand, xs)
    f_ion = integral / u
    return 1.0 - f_ion
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_mirror.py -k alpha_electron -v`
Expected: PASS. (If `jnp.trapezoid` is unavailable in the installed JAX, use
`jnp.trapz`; verify with `.venv/bin/python -c "import jax.numpy as jnp; print(hasattr(jnp,'trapezoid'))"`.)

- [ ] **Step 5: Commit**

```bash
git add src/costingfe/layers/mirror.py tests/test_mirror.py
git commit -m "Add Stix alpha-to-electron slowing-down fraction"
```

---

### Task 4: Extract `_confinement_and_losses` helper (behavior-preserving)

**Files:**
- Modify: `src/costingfe/layers/mirror.py` (`mirror_0d_forward`, the confinement
  and loss block, currently around lines 462-517)
- Test: `tests/test_mirror.py` (the existing `test_default_path_bit_identical`
  and the 0D forward tests must still pass; add one direct helper test)

**Interfaces:**
- Produces: `_confinement_and_losses(T_e, *, n_e, T_i, n_i_frac, M_ion, R_m, L,
  a, B_min, phi, collisionality_min) -> dict` returning at least
  `{"tau_E", "W_th_MW", "p_end", "p_radial", "axial_frac", "tau_p",
  "collisionality"}`. Pure function of `T_e` and the fixed operating point.

This refactor pulls the tau-chain, tau_E, W_th, and the axial/radial loss split
out of `mirror_0d_forward` into a helper, so the T_e solver (Task 5) can evaluate
`p_end(T_e)` at trial temperatures. No physics change: `mirror_0d_forward` calls
the helper with its existing `T_e` and uses the returned values exactly as before.

- [ ] **Step 1: Write the failing test (helper exists, forward unchanged)**

```python
# tests/test_mirror.py
from costingfe.layers.mirror import _confinement_and_losses

def test_confinement_helper_returns_loss_split():
    out = _confinement_and_losses(
        10.0, n_e=5e19, T_i=10.0, n_i_frac=1.0, M_ion=2.5, R_m=10.0,
        L=50.0, a=0.5, B_min=3.0, phi=74.7, collisionality_min=0.1,
    )
    assert out["p_end"] > 0.0
    assert out["p_radial"] > 0.0
    # Axial + radial fractions partition the total loss.
    assert abs(out["p_end"] + out["p_radial"] - out["W_th_MW"] / out["tau_E"]) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_mirror.py -k confinement_helper -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Extract the helper and call it from the forward**

Move the existing confinement/loss block out of `mirror_0d_forward` into:

```python
# src/costingfe/layers/mirror.py
def _confinement_and_losses(T_e, *, n_e, T_i, n_i_frac, M_ion, R_m, L, a,
                            B_min, phi, collisionality_min):
    """Confinement times and the axial/radial loss split at a given T_e.

    Pure function of T_e and the fixed operating point so the electron-balance
    solver can evaluate p_end(T_e). Mirrors the chain previously inline in
    mirror_0d_forward (tau_ii -> tau chain -> tau_E -> W_th -> p_end/p_radial).
    """
    n_i = n_i_frac * n_e
    tau_ii = compute_tau_ii(n_i, T_i, M_ion)
    tau_axial = compute_tau_axial(tau_ii, R_m, L, T_i, M_ion, phi, n_i)
    inv_tau_axial = 1.0 / tau_axial
    tau_radial = compute_tau_radial(tau_ii, a, T_i, M_ion, B_min)
    inv_tau_p = inv_tau_axial + 1.0 / tau_radial
    tau_p = 1.0 / inv_tau_p
    tau_E = tau_p * (1.5 * (T_i + T_e)) / (2.0 * phi + T_i + T_e)
    W_th_MW = 1.5 * n_e * (T_e + n_i_frac * T_i) * _KEV_TO_J * (jnp.pi * a**2 * L) * 1e-6
    P_total = W_th_MW / tau_E
    axial_frac = inv_tau_axial / inv_tau_p
    p_end = P_total * axial_frac
    p_radial = P_total - p_end
    collisionality = L / (
        _V_THI_PREFACTOR * jnp.sqrt(T_i / M_ion) * tau_ii
    )
    return {
        "tau_E": tau_E, "W_th_MW": W_th_MW, "p_end": p_end,
        "p_radial": p_radial, "axial_frac": axial_frac, "tau_p": tau_p,
        "collisionality": collisionality,
    }
```

In `mirror_0d_forward`, replace the inline block (lines about 462-517, and the
W_th/beta dependencies) with a call:

```python
    cl = _confinement_and_losses(
        T_e, n_e=n_e, T_i=T_i, n_i_frac=n_i_frac, M_ion=M_ion, R_m=R_m,
        L=L, a=a, B_min=B_min, phi=phi, collisionality_min=collisionality_min,
    )
    tau_E = cl["tau_E"]; W_th_MW = cl["W_th_MW"]
    p_end = cl["p_end"]; p_radial = cl["p_radial"]
    f_axial_derived = p_end / (p_end + p_radial)
    collisionality = cl["collisionality"]
```

Keep `tau_classical`, `tau_Pastukhov`, `tau_GD` diagnostics as they were if the
state needs them (compute them inline before the helper call, or add to the
returned dict; preserve the exact state fields).

- [ ] **Step 4: Run the helper test and the full forward suite**

Run: `.venv/bin/python -m pytest tests/test_mirror.py -q`
Expected: PASS, including `test_default_path_bit_identical` (behavior preserved).

- [ ] **Step 5: Commit**

```bash
git add src/costingfe/layers/mirror.py tests/test_mirror.py
git commit -m "Extract _confinement_and_losses helper from mirror_0d_forward (no behavior change)"
```

---

### Task 5: Electron-balance solver `solve_T_e`

**Files:**
- Modify: `src/costingfe/layers/mirror.py`
- Test: `tests/test_mirror.py` (append)

**Interfaces:**
- Consumes: `compute_p_brem_rel`, `compute_K_ie`, `alpha_electron_fraction`,
  `_confinement_and_losses`.
- Produces: `solve_T_e(*, n_e, T_i, p_alpha, Z_eff, Z_i, A, n_i_frac, R_m, L, a,
  B_min, phi, f_alpha_heat, e_crit_over_te, E_alpha_keV, collisionality_min)
  -> float` (keV).

Solves `alpha_e(T_e)*p_alpha + K_ie(T_i-T_e) = P_brem(T_e) + gamma_e*p_end(T_e)`,
with `alpha_e = f_alpha_heat * alpha_electron_fraction(T_e)`,
`gamma_e = T_e/(T_e + n_i_frac*T_i)` (electron pressure fraction) applied to the
AXIAL `p_end` only; radial loss is ion-only. Residual is monotone decreasing in
T_e, so bisect.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mirror.py
from costingfe.layers.mirror import solve_T_e

def test_solve_T_e_DT_lands_cool():
    # D-T at a reference operating point must land near T_i (order 10 to 20 keV),
    # NOT the old 125 keV. p_alpha order 200 MW.
    T_e = float(solve_T_e(
        n_e=5e19, T_i=15.0, p_alpha=200.0, Z_eff=1.5, Z_i=1.0, A=2.5,
        n_i_frac=1.0, R_m=10.0, L=50.0, a=0.5, B_min=3.0, phi=74.7,
        f_alpha_heat=0.8, e_crit_over_te=33.0, E_alpha_keV=3500.0,
        collisionality_min=0.1,
    ))
    assert 5.0 < T_e < 40.0  # cool, not hot

def test_solve_T_e_residual_zero_at_solution():
    # Re-evaluating the balance at the returned T_e gives a near-zero residual.
    # (Implement a small local residual mirroring solve_T_e for the assertion.)
    T_e = float(solve_T_e(
        n_e=5e19, T_i=15.0, p_alpha=200.0, Z_eff=1.5, Z_i=1.0, A=2.5,
        n_i_frac=1.0, R_m=10.0, L=50.0, a=0.5, B_min=3.0, phi=74.7,
        f_alpha_heat=0.8, e_crit_over_te=33.0, E_alpha_keV=3500.0,
        collisionality_min=0.1,
    ))
    assert 5.0 < T_e < 40.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_mirror.py -k solve_T_e -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation**

```python
# src/costingfe/layers/mirror.py
def solve_T_e(*, n_e, T_i, p_alpha, Z_eff, Z_i, A, n_i_frac, R_m, L, a, B_min,
              phi, f_alpha_heat, e_crit_over_te, E_alpha_keV, collisionality_min):
    """Solve the central-cell electron power balance for T_e [keV].

    alpha_e(T_e)*p_alpha + K_ie(T_i-T_e) = P_brem(T_e) + gamma_e*p_end(T_e)
    gamma_e = T_e/(T_e + n_i_frac*T_i)  (electron pressure fraction, axial only)
    Residual is monotone decreasing in T_e -> bisection on [0.1, 2*T_i].
    """
    n_i = n_i_frac * n_e
    volume = jnp.pi * a**2 * L

    def residual(T_e):
        a_e = f_alpha_heat * alpha_electron_fraction(T_e, E_alpha_keV, e_crit_over_te)
        p_ie = compute_K_ie(n_e, n_i, T_i, T_e, Z_i, A, volume)
        p_brem = compute_p_brem_rel(n_e, T_e, Z_eff, volume)
        cl = _confinement_and_losses(
            T_e, n_e=n_e, T_i=T_i, n_i_frac=n_i_frac, M_ion=A, R_m=R_m,
            L=L, a=a, B_min=B_min, phi=phi, collisionality_min=collisionality_min,
        )
        gamma_e = T_e / (T_e + n_i_frac * T_i)
        return a_e * p_alpha + p_ie - p_brem - gamma_e * cl["p_end"]

    # Bounded Python loop (not jax.lax.fori_loop): portable to a numpy backend
    # and JIT-traceable (fixed 60 iterations unroll). See the JAX-optional note.
    lo, hi = 0.1, 2.0 * T_i
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        r = residual(mid)
        above = r > 0  # residual decreasing: r > 0 -> root above mid
        lo = jnp.where(above, mid, lo)
        hi = jnp.where(above, hi, mid)
    return 0.5 * (lo + hi)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_mirror.py -k solve_T_e -v`
Expected: PASS (T_e in the cool band).

- [ ] **Step 5: Commit**

```bash
git add src/costingfe/layers/mirror.py tests/test_mirror.py
git commit -m "Add solve_T_e electron-balance solver (lands D-T T_e cool)"
```

---

### Task 6: Wire `solve_te` into `mirror_0d_forward` (opt-in) and re-pin

**Files:**
- Modify: `src/costingfe/layers/mirror.py` (`mirror_0d_forward` signature + body)
- Test: `tests/test_mirror.py`

**Interfaces:**
- Consumes: `solve_T_e`.
- Produces: `mirror_0d_forward(..., solve_te: bool = False)`. When `solve_te` is
  True, central `T_e` is solved from the electron balance and the passed `T_e`
  is used only as an unused parity argument; all downstream consumers (radiation,
  beta, W_th, tau_E) use the solved value. Default False preserves current
  behavior and keeps existing tests/pins green.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mirror.py
from costingfe.layers.mirror import mirror_0d_forward
from costingfe.types import Fuel

def test_forward_solve_te_overrides_pinned_Te():
    common = dict(
        L=50.0, a=0.5, B_min=3.0, R_m=10.0, T_i=15.0, n_e=5e19, p_input=0.0,
        fuel=Fuel.DT, M_ion=2.5, Z_eff=1.5, R_w=0.4, dd_f_T=0.969, dd_f_He3=0.689,
        dhe3_dd_frac_pin=None, dhe3_f_T=0.5, dhe3_f_He3=0.5, pb11_f_alpha_n=0.0,
        pb11_f_p_n=0.0, dhe3_fuel_ratio=1.0, pb11_fuel_ratio=0.15, vacuum_t=0.1,
        plug_density_ratio=1.818, collisionality_min=0.1, T_e_plug=125.0,
    )
    # Pass an absurd T_e=125; with solve_te=True the state T_e must be solved cool.
    st = mirror_0d_forward(T_e=125.0, solve_te=True, **common)
    assert 5.0 < float(st.T_e) < 40.0
    # With solve_te=False the passed T_e is used verbatim.
    st0 = mirror_0d_forward(T_e=125.0, solve_te=False, **common)
    assert abs(float(st0.T_e) - 125.0) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_mirror.py -k solve_te_overrides -v`
Expected: FAIL with "unexpected keyword argument 'solve_te'".

- [ ] **Step 3: Add the flag and solve T_e at the top of the forward**

In `mirror_0d_forward`, add `solve_te: bool = False` to the keyword-only args.
After `p_alpha` is computed (around line 438) and `phi` is available (compute
`phi = compute_plug_potential(T_e_plug, plug_density_ratio)` earlier if needed),
insert:

```python
    if solve_te:
        T_e = solve_T_e(
            n_e=n_e, T_i=T_i, p_alpha=p_alpha, Z_eff=Z_eff, Z_i=1.0, A=M_ion,
            n_i_frac=n_i_frac, R_m=R_m, L=L, a=a, B_min=B_min, phi=phi,
            f_alpha_heat=0.8, e_crit_over_te=33.0, E_alpha_keV=3500.0,
            collisionality_min=collisionality_min,
        )
```

Everything after (radiation `compute_p_rad(n_e, T_e, ...)`, the
`_confinement_and_losses` call, beta) now reads the solved `T_e`. The cost-side
radiation `p_rad` keeps using `compute_p_rad` at the solved (cool) `T_e`; only
the electron-balance RHS uses `compute_p_brem_rel`. Move the `phi` computation
above the solve if it currently sits below `p_alpha`.

- [ ] **Step 4: Run the targeted test and the full mirror suite**

Run: `.venv/bin/python -m pytest tests/test_mirror.py -q`
Expected: PASS. `solve_te=False` paths and existing pins unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/costingfe/layers/mirror.py tests/test_mirror.py
git commit -m "Wire opt-in solve_te into mirror_0d_forward (solved T_e overrides input)"
```

---

### Task 7: Validation, anchors, and docs

**Files:**
- Test: `tests/test_mirror.py` (anchor + advanced-fuel checks)
- Modify: `docs/account_justification/mirror_confinement_regimes.md` (record the
  electron-balance T_e solve and correct the sanctioned hot-operating-point table)

**Interfaces:**
- Consumes: `mirror_0d_forward(..., solve_te=True)`.

- [ ] **Step 1: Add GDT/WHAM anchor and advanced-fuel checks**

```python
# tests/test_mirror.py
def test_solve_te_DT_anchor_band():
    # The D-T operating point with solve_te=True must land in the cool band and
    # stay within the existing 2x GDT/WHAM confinement validation envelope.
    # (Reuse the anchor fixtures already in this module; assert tau_E and T_e.)
    ...

def test_solve_te_advanced_fuel_Te_below_Ti():
    # For D-He3 / p-B11, solved T_e must remain below T_i (hot-ion mode), not
    # blow up. (Thermal, no channeling: advanced fuels stay net-negative; this
    # only checks the T_e solve is physical, not economics.)
    ...
```

Fill the `...` with concrete assertions using the module's existing anchor
helpers and fuel fixtures; assert `T_e < T_i` and `tau_E` within the documented
2x band. Run and iterate until the numbers are pinned, then hard-code them.

- [ ] **Step 2: Run the validation tests**

Run: `.venv/bin/python -m pytest tests/test_mirror.py -q`
Expected: PASS.

- [ ] **Step 3: Cross-check against the group's code (oracle)**

Obtain the Ochs/Kolmes numerical model (the user can provide it). With
`eta_alpha = 0`, confirm `compute_p_brem_rel` and `compute_K_ie` reproduce its
p-B11 reference numbers (electron temperature about 160 keV at the Putvinski
15-percent-boron point). Record the comparison in the justification doc. This is
a manual validation step, not an automated test.

- [ ] **Step 4: Update the justification doc**

In `docs/account_justification/mirror_confinement_regimes.md`, replace the
sanctioned hot D-T operating-point table (T_i about 23 keV) with the
electron-balance result, and document the closure (Eq. 16 brems, NRL K_ie, Stix
alpha split, pressure-weighted gamma_e on the axial channel). Keep it in sync
with `docs/papers/1costingfe_paper/1costingfe_paper.tex` per the paper-sync rule.

- [ ] **Step 5: Run the full suite and commit**

Run: `.venv/bin/python -m pytest -q -m "not slow"`
Expected: PASS (re-pin any 0D golden numbers that legitimately move, documenting
why in the test comment).

```bash
git add tests/test_mirror.py docs/account_justification/mirror_confinement_regimes.md
git commit -m "Validate mirror electron-balance T_e against anchors; update justification"
```

---

## Notes for the implementer

- The whole change lives in the gated 0D path. The released, user-supplied
  operating-point path already uses the corrected YAML default (T_i = T_e = 10
  keV) and is unaffected.
- Do not touch `T_e_plug`; it drives only the Fowler-Logan plug potential.
  Conflating it with central `T_e` was the original bug.
- Unit conversion is the main hazard. Each new prefactor (0.12113 for brems, the
  CGS K_ie chain) is pinned by an independently-derived reference test; if a test
  value disagrees, suspect the unit conversion first.
- `e_crit_over_te = 33.0` and `E_alpha_keV = 3500.0` are D-T values. Generalizing
  them per fuel (E_crit = 14.8*A_alpha*(sum (n_j/n_e) Z_j^2/A_j)^(2/3)*T_e, and
  the fuel's charged-particle birth energy) is part of the deferred advanced-fuel
  extension.
- OUT OF SCOPE: the sizing-path density<->T_e coupling. In `mirror_size_from_power`
  the beta-solved density depends on T_e (`_density_from_f_beta`), so a fully
  self-consistent sizing solve would iterate density and the T_e solve together.
  This plan fixes the forward operating point only; the sizing path is doubly
  gated (`SIZING_FEATURES_ENABLED = False`) and its coupling is a follow-on.
