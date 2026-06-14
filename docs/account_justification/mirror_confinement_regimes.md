# Mirror Confinement Regime Bridge: Collisional vs Collisionless

**Date:** 2026-06-14
**Status:** Implemented; bridge sourced to Rognlien and Cutler 1980, smoothing
constructed and documented honestly.

## Purpose

The 0D axisymmetric mirror model (`src/costingfe/layers/mirror.py`) computes the
axial confinement time by combining two kernels: the gas-dynamic (collisional)
time `tau_GD` (Mirnov and Ryutov 1979) and the Pastukhov electrostatically
plugged (collisionless) time `tau_Pastukhov` (Pastukhov 1974, Cohen et al. 1978).
These two limits apply in physically distinct collisionality regimes and must be
joined by a single bridge that selects the right limit. This document records the
published transition criterion, the bridge form the model uses, its validity
window, and the sanctioned mirror re-pin that follows from correcting it.

A `$` here is a plain dollar sign; there are no costs in this document.

## The bug this replaces

The original combination was an unconditional harmonic sum of loss rates:

```
inv_tau_axial = 1 / tau_Pastukhov + 1 / tau_GD
```

A harmonic sum of rates always selects the SHORTER of the two times. At a
deeply collisionless operating point (for example the D-T sizing optimum, where
the collisionality `L / mfp` is about 3e-5), the gas-dynamic flow-out time is of
order 5e-5 s while the Pastukhov loss-cone time is of order seconds. The
gas-dynamic channel does not physically apply in that regime (the loss cone is
not kept full by collisions), yet the harmonic sum let its short time dominate,
making `tau_E` about 1e4 too short. The fix gates the gas-dynamic channel by
collisionality so it contributes only when the plasma is actually collisional.

## The published transition criterion (primary sources)

The collisional-to-kinetic transition for mirror axial confinement is treated
directly by Rognlien and Cutler 1980. Their stated result:

- The Pastukhov formula for charged-particle confinement in a combined magnetic
  and electrostatic well becomes invalid when the ion mean free path, reduced by
  the mirror ratio, is of the order of the system length. Below that (longer mean
  free path) the loss-cone-limited Pastukhov time governs; above it (shorter mean
  free path) the loss cone is kept full and the collisional gas-dynamic outflow
  governs.

- They validate, against a Monte Carlo transport code, that a good composite
  model across the transition is to ADD the analytic Pastukhov and collisional
  confinement times, `tau = tau_collisional + tau_Pastukhov`, which gives a smooth
  transition between the two regimes.

The same picture is restated in the gas-dynamic-trap literature: the GDT central
cell is by design long compared with the Coulomb mean free path so the trapped
ions are collisional and the outflow is gas-dynamic (Ivanov and Prikhodko 2013),
while a high-temperature low-density axisymmetric mirror (WHAM-class) sits in the
collisionless loss-cone regime where the Pastukhov scaling applies (Endrizzi et
al. 2023). The boundary in both descriptions is the same: the ion mean free path,
reduced by the mirror ratio, crossing the system length.

### Transition variable used in the model

The model already carries a collisionality diagnostic,

```
collisionality  =  L / mfp  =  L / (v_thi * tau_ii),
```

with `v_thi = _V_THI_PREFACTOR * sqrt(T_i / A)` the ion thermal speed and
`tau_ii` the ion-ion collision time. `collisionality >> 1` is collisional (mean
free path much shorter than L, gas-dynamic applies); `collisionality << 1` is
collisionless (Pastukhov applies). The Rognlien and Cutler criterion "mean free
path reduced by the mirror ratio is of order the system length" is

```
mfp / R_m  ~  L      <=>      L / mfp  ~  1 / R_m      <=>      collisionality  ~  1 / R_m,
```

so the model uses the regime-boundary value

```
collisionality_crit  =  1 / R_m,
```

which is the published criterion expressed in the model's own collisionality
variable. The 1/R_m placement is not a free knob; it is the mirror-ratio
reduction of the mean free path stated by Rognlien and Cutler.

## The bridge form the model uses

A plain sum of times `tau = tau_GD + tau_Pastukhov` (the literal Rognlien and
Cutler composite) selects the LONGER time and therefore does NOT reproduce the
gas-dynamic limit at a collisional point where the Pastukhov time happens to be
longer than the gas-dynamic time (low T, where the electrostatic plug is weak).
The physically correct statement is stronger than the literal sum: the
gas-dynamic loss channel exists only when the plasma is collisional, and it must
be switched off in the collisionless regime so that Pastukhov governs there. The
model therefore gates the gas-dynamic LOSS RATE by a smooth function of
collisionality and combines via the loss-rate (harmonic) sum:

```
g(collisionality)  =  sigmoid( ( log10(collisionality) - log10(1/R_m) ) / w )
inv_tau_axial      =  1 / tau_Pastukhov  +  g(collisionality) * (1 / tau_GD)
tau_axial          =  1 / inv_tau_axial
```

`g -> 1` when `collisionality >> 1/R_m` (collisional): the gas-dynamic rate is
fully present and, being the faster loss, dominates, so `tau_axial -> tau_GD`.
`g -> 0` when `collisionality << 1/R_m` (collisionless): the gas-dynamic rate is
suppressed below the Pastukhov rate, so `tau_axial -> tau_Pastukhov`. The
crossover is centered on the published boundary `collisionality = 1/R_m`.

### The smoothing width is constructed, documented honestly

Rognlien and Cutler establish that the transition is smooth (their Monte Carlo
gives a continuous confinement time across the boundary) and locate the boundary
(mean free path reduced by the mirror ratio equals L), but they do not publish a
closed-form transition width. The model uses a logistic gate in `log10`
collisionality with a width

```
w = 0.13  decades  (_REGIME_GATE_WIDTH_DECADES)
```

This is a constructed smoothing parameter, not a sourced number. It is chosen so
that the gate is centered on the sourced boundary `1/R_m`, transitions smoothly
across it, and is differentiable everywhere (the golden-section sizing and the
jax.grad sensitivity vector both traverse it). The width is deliberately narrow:
because `tau_GD` can be up to 1e4 shorter than `tau_Pastukhov` in the
collisionless regime, even a small residual gate value leaks a meaningful loss
rate (the same failure mode as the original harmonic-sum bug, in miniature). A
narrow gate keeps the suppressed gas-dynamic rate well below the Pastukhov rate
at the collisionless anchors (WHAM and the high-mirror-ratio reference points)
while remaining fully on at the collisional anchor (GDT). With `w = 0.13` the
gate is essentially off about 0.4 decades below the boundary and essentially on
about 0.4 decades above it, and the branch selection is robust at both anchors,
which sit more than a decade from their `1/R_m` boundaries.

## Validity window and caveats

- The bridge inherits the validity caveats of both kernels (single thermal
  Maxwellian, no fast-ion or sloshing-ion population), documented in
  `mirror_confinement.md`. It does not add physics; it selects which existing
  kernel governs.
- `collisionality = 1/R_m` is an order-unity boundary, not a sharp threshold; the
  smoothing width reflects that. Operating points within about a decade of the
  boundary carry a genuine regime ambiguity, and the gate interpolates smoothly
  rather than asserting a hard branch.
- In the collisionless regime the Pastukhov formula itself overestimates
  confinement when the plasma is too collisionless to be Maxwellian; that is a
  separate validity flag (Part 3 of the parent spec), not handled by this bridge.
- The bridge is float32-safe: the gate argument is built from `log10` of an
  order-unity-to-1e-5 collisionality, the prefactors are folded in float64, and
  every intermediate stays in a benign range. The logistic argument is clamped to
  `[-30, 30]` before `exp`, so even far past any physical operating point the gate
  keeps a finite value and a finite reverse-mode gradient (an unclamped saturated
  logistic gives `inf * 0 = NaN` under `jax.grad`). At physical collisionalities
  the clamp is never active.

## Re-anchor result (guard)

The GDT and WHAM anchors in `tests/test_mirror.py::TestAnchors` test the kernels
directly and are unchanged by this bridge. The regime guard is that the bridge
selects the correct branch at each anchor's regime. The table below is generated
from the model kernels (`compute_tau_axial` and friends) at the stated anchor
parameters, so every value reproduces; it uses gate width `w = 0.13`.

Exact anchor parameters:

- GDT: `R_m = 35`, `L = 7` m, `T_i = T_e = 0.25` keV, `n = 2e19` m^-3 (warm-plasma
  density, Bagryansky et al. 2015). Collisionality scales with `n` through
  `tau_ii`; at this density GDT sits about 0.7 decades inside the collisional
  branch.
- WHAM: `R_m = 17/0.86 = 19.77`, `n = 3e19` m^-3, `T_i = 10` keV, `T_e = 1` keV,
  `L = 2` m (central-cell length scale, Endrizzi et al. 2023). WHAM sits more than
  3 decades inside the collisionless branch, so the gate result is insensitive to
  the precise `L`.

| Anchor | Regime | collisionality | 1/R_m | gate g | tau_axial / tau_GD | tau_axial / tau_Pastukhov | Branch selected |
|--------|--------|---------------|-------|--------|--------------------|---------------------------|-----------------|
| GDT | collisional | 0.132 | 0.029 | 0.99 | 0.99 | 0.015 | gas-dynamic (correct) |
| WHAM | collisionless | 3.5e-5 | 0.051 | 2.8e-11 | 2.2e3 | 1.00 | Pastukhov (correct) |

GDT lands on the gas-dynamic branch within 1 percent; WHAM lands on the Pastukhov
branch within 1 percent. Both kernels remain within their documented 2x of the
published confinement times (see `mirror_confinement.md`).

## Sanctioned mirror re-pin (before / after)

Correcting `tau_E` for collisionless points moves mirror forward/inverse/sizing
values. The coil calibration pin (513.375 M$, capital) and all non-mirror
(tokamak, stellarator, IFE/MIF) pins do NOT move. Mirror pins that shifted in
Task 1 are recorded here; the energy-balance closure (Task 2) and the final
re-pin (Task 6) extend this table.

| Concept | Fuel | Quantity | Old | New | Note |
|---------|------|----------|-----|-----|------|
| (Task 1: forward/inverse/sizing tau-driven pins recorded as they shift below) |

## Sources

- **Rognlien, T. D. and Cutler, T. A. (1980)**, "Transition from Pastukhov to
  collisional confinement in a magnetic and electrostatic well," Nucl. Fusion 20,
  1003. Primary source for the transition criterion (Pastukhov invalid when the
  mean free path reduced by the mirror ratio is of order the system length) and
  for the Monte-Carlo-validated smooth composite of the Pastukhov and collisional
  confinement times.
- **Pastukhov, V. P. (1974)**, "Collisional losses of electrons from an adiabatic
  trap in a plasma with a positive potential," Nucl. Fusion 14, 3. Collisionless
  loss-cone confinement kernel.
- **Cohen, R. H., Rensink, M. E., Cutler, T. A. and Mirin, A. A. (1978)**,
  "Collisional loss of electrostatically confined species in a magnetic mirror,"
  Nucl. Fusion 18, 1229. Refinement of the Pastukhov scaling used by the kernel.
- **Mirnov, V. V. and Ryutov, D. D. (1979)**, gas-dynamic confinement scaling
  `tau_GD = R_m * L / v_thi`. Collisional outflow kernel.
- **Ivanov, A. A. and Prikhodko, V. V. (2013)**, "Gas-dynamic trap: an overview
  of the concept and experimental results," Plasma Phys. Control. Fusion 55,
  063001. GDT central cell designed long compared with the Coulomb mean free path
  (collisional, gas-dynamic regime).
- **Endrizzi, D. et al. (2023)**, "Physics basis for the Wisconsin HTS
  Axisymmetric Mirror (WHAM)," J. Plasma Phys. 89, 975890501. Collisionless
  loss-cone (Pastukhov) regime for a high-field compact axisymmetric mirror.

No number in this document is sourced from or calibrated against any
cost-modeling tool; all come from the primary plasma-physics literature, except
the constructed smoothing width `w`, which is documented as constructed.
