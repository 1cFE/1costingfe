# Multi-Fuel Reactivity for the Tokamak 0D/Sizing Kernel

**Date:** 2026-06-10
**Status:** Approved design, pending implementation

## Problem

The tokamak 0D plasma model and the power-to-geometry sizing path compute fusion
power with `sigma_v_dt` (Bosch-Hale D-T) unconditionally (`compute_fusion_power`,
`tokamak.py`), regardless of the `fuel` argument threaded through them. The
downstream chain (energy partition in `physics.py`, fuel-keyed radial build,
power balance, costing) already handles all four fuels (DT, DD, D-He3, p-B11),
so running 0D or sizing with an advanced fuel today silently produces a machine
with D-T reactivity and advanced-fuel bookkeeping: no error, wrong answer.
D-He3 reactivity near 15 keV is roughly two orders of magnitude below D-T's,
and the advanced fuels want 50-100+ keV operating points, so the error is not
small.

Both model directions are needed and share the kernel: sizing
(geometry-from-power) is a root-find wrapped around the forward model
(power-from-geometry), and pin-mode benchmark validation (ARC, ARIES-AT) uses
the forward direction outright. Multi-fuel support therefore lands once, in the
shared kernel, and every mode (forward, inverse, sizing, optimize) inherits it.

## Design

### 1. New module `src/costingfe/layers/reactivity.py`

Pure JAX functions, one per channel:

- `sigv_dt(T)` — moves from `tokamak.py` (Bosch-Hale 1992, unchanged).
- `sigv_dd_n(T)`, `sigv_dd_p(T)` — both D-D branches, coefficients lifted from
  the verified `examples/dhe3_mix_optimization.py` (Bosch-Hale 1992).
- `sigv_dhe3(T)` — same source.
- `sigv_pb11(T)` — Nevins & Swain 2000 fit. Putvinski et al. 2019 (higher
  reactivity at the tail) documented in the module docstring as the optimistic
  alternative; not the default.

All take T_i in keV, return `<sigma*v>` in m^3/s, and are JAX-differentiable
(`jnp` ops only). The example scripts (`dhe3_mix_optimization.py`,
`dhe3_burn_fractions.py`) are refactored to import these from the package so
the coefficients exist in exactly one place.

### 2. Fuel dispatch and dilution: `fusion_power_density`

`fusion_power_density(fuel, n_e, T_i, mix_ratio, ...)` replaces the body of
`compute_fusion_power`. Dispatch is a Python branch on `fuel` (static per
model instance), so JAX tracing is unaffected.

Reactant densities from quasineutrality, with one new YAML knob
`fuel_mix_ratio` (`r`):

| Fuel  | Meaning of r        | Densities                                   | Rate(s)                                              |
|-------|---------------------|---------------------------------------------|------------------------------------------------------|
| DT    | ignored (50/50)     | n_D = n_T = n_e/2                            | (1/4) n_e^2 sigv_dt                                   |
| DD    | ignored (pure D)    | n_D = n_e                                    | (1/2) n_D^2 (sigv_dd_n + sigv_dd_p)                   |
| D-He3 | n_He3/n_D           | n_D = n_e/(1+2r), n_He3 = r*n_D              | n_D n_He3 sigv_dhe3 + (1/2) n_D^2 (sigv_dd_n+sigv_dd_p) |
| p-B11 | n_B/n_p             | n_p = n_e/(1+5r), n_B = r*n_p                | n_p n_B sigv_pb11                                     |

`fuel_mix_ratio` defaults: 1.0 for D-He3 (50:50), 0.15 for p-B11 (typical
lean-boron design point); to be calibrated. Declared in concept YAML only —
no Python keyword default in any function signature.

Per-event energies reuse the existing `physics.py` constants (Q_DT, Q_DD_PT,
Q_DD_NHE3, Q_DHE3, Q_PB11, and the secondary-burn terms), so total p_fus is
consistent with what `ash_neutron_split` assumes when it partitions it. The
function returns `(p_fus, derived_channel_fracs)`.

### 3. Hybrid channel fractions

In the 0D and sizing paths, `dhe3_dd_frac` is **derived** at the operating
point from the rate ratio

    dhe3_dd_frac = R_dd / (R_dd + R_dhe3),
    R_dd = (1/2) n_D^2 (sigv_dd_n + sigv_dd_p),  R_dhe3 = n_D n_He3 sigv_dhe3

and fed into the energy partition — unless the user explicitly overrides
`dhe3_dd_frac`, which wins (mirrors the existing derived-`dhe3_f_He3` pattern,
`model.py:_dhe3_f_He3_eff`). The non-0D path has no operating point and keeps
using the YAML value unchanged. Secondary-burn knobs `dd_f_T`, `dd_f_He3`,
`dhe3_f_T` (and the derived `dhe3_f_He3`) stay as they are: they encode
fast-ion burnup physics, not thermal rates.

### 4. T_i/T_e ratio knob

New param `T_i_over_T_e` with its default of 1.0 declared in the concept
YAMLs (all current results unchanged). No Python keyword default anywhere:
`fusion_power_density` and the 0D/sizing functions take it (and
`fuel_mix_ratio`) as required arguments read from the merged params dict,
same as every other physics param. T_i = T_i_over_T_e * T_e enters
reactivity, stored energy, and beta. Stored
energy and beta generalize with the dilution factor n_i/n_e so that DT
reproduces today's numbers exactly:

    W_th = 1.5 * n_e * (T_e + (n_i/n_e) * T_i) * V * KEV_TO_J
    beta_t ∝ n_e * (T_e + (n_i/n_e) * T_i) / B^2   (same normalization as today)

For DT/DD, n_i = n_e and the defaults give the current factor-2 pressure sum.
For D-He3, n_i/n_e = (1+r)/(1+2r); for p-B11, (1+r)/(1+5r).

### 5. Fuel-aware temperature brackets

Sizing already reads `T_min`/`T_max` from YAML (5-60 keV in
`steady_state_tokamak.yaml`). The inverse-mode bisection
(`_find_T_for_pfus`) hardcodes 1-100 keV. **DT keeps its current brackets in
both paths, untouched** (sizing 5-60 from YAML, inverse 1-100), so existing
behavior is bit-identical. Non-DT fuels get fuel-keyed bracket defaults in
code, following the existing `_RADIAL_BUILD_DEFAULTS` pattern, overridable
from YAML:

| Fuel  | T_min [keV] | T_max [keV] |
|-------|------------|-------------|
| DD    | 5          | 100         |
| D-He3 | 20         | 200         |
| p-B11 | 50         | 400         |

### 6. Fuel-aware Z_eff and photon radiation

Fuel ions are fully stripped at all relevant operating temperatures, so they
add no line radiation; their photon contribution is enhanced bremsstrahlung,
which the existing radiation module already parametrizes through Z_eff. A new
`z_eff_fuel(fuel, r)` in `reactivity.py` computes the fuel-ion contribution
from the same quasineutrality mix as the dilution math:

    Z_eff_fuel = sum(n_j Z_j^2) / n_e
    DT, DD: 1.0
    D-He3:  (1 + 4r) / (1 + 2r)   (~1.67 at r = 1)
    p-B11:  (1 + 25r) / (1 + 5r)  (~2.71 at r = 0.15)

The YAML `Z_eff` is reinterpreted as fuel + impurity excess:

    Z_eff_effective = Z_eff_fuel + (Z_eff_yaml - 1)

For DT with the current YAML 1.5 this gives exactly 1.5 (bit-identical), and
the impurity content (the 0.5 excess) carries over to all fuels.
`Z_eff_effective` feeds `compute_p_rad` and the power balance wherever Z_eff
is consumed today. Seeded impurities and wall material stay user inputs and
add their line radiation through the existing module, unchanged. The power
balance's radiation branches are already mutually exclusive, so when
`f_rad_fus` is set (the p-B11 proxy below) it replaces the computed p_rad —
no double count.

The module's bremsstrahlung is non-relativistic; at D-He3 operating
temperatures (50-100 keV) this underestimates brems by roughly 20-30%
(quantified by `brem_factor_rel` in `examples/dhe3_mix_optimization.py`).
Applying the correction selectively would make the fuels inconsistent and
applying it everywhere would perturb DT, so it is documented here as a known
limitation and left out of scope.

### 7. p-B11 radiation proxy

No new radiation physics. p-B11 configs running 0D/sizing set
`f_rad_fus: 0.83` (bremsstrahlung at 83% of fusion power, the Putvinski-class
optimum; the power balance already supports `f_rad_fus`). The validator warns
when fuel is PB11 with `use_0d_model` or `size_from_power` enabled and
`f_rad_fus` is unset.

### 8. Tests

1. **DT regression:** existing pins bit-identical (tokamak LCOE pin 226.89,
   full suite green). DT keeps the same fit, mix, and pressure factors.
2. **Fit accuracy:** each `sigv_*` checked against published Bosch-Hale /
   Nevins-Swain table values at reference temperatures.
3. **Dilution algebra:** density and n_i/n_e factors per fuel.
4. **Derived `dhe3_dd_frac`:** matches `examples/dhe3_mix_optimization.py`
   at its published operating points; explicit override wins.
5. **Sizing per fuel:** D-He3 sizes a larger machine than DT at equal net
   electric power; p-B11 solves with the brems proxy inside a 50-400 keV
   bracket.
6. **Guard:** PB11 + 0D/sizing without `f_rad_fus` produces the validator
   warning.
7. **Z_eff:** `z_eff_fuel` algebra per fuel; `Z_eff_effective` equals the
   YAML value exactly for DT (1.5 stays 1.5); D-He3/p-B11 brems rises with
   the mix-derived Z_eff through the existing radiation module.

## Out of Scope

- Synchrotron radiation, explicit ion-electron coupling, hot-ion power
  balance (the `f_rad_fus` proxy stands in for p-B11).
- Relativistic bremsstrahlung correction (`brem_factor_rel`): documented
  limitation at D-He3 temperatures, see section 6.
- Fast-ion burn chains (the existing secondary-burn fraction knobs stay).
- Non-tokamak concepts; `reactivity.py` is concept-agnostic so mirror/FRC
  sizing can import it later.
