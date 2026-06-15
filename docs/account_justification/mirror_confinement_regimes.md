# Mirror Confinement Regime Bridge: Collisional vs Collisionless

**Date:** 2026-06-14
**Status:** Implemented. Collisionality bridge sourced to Rognlien and Cutler
1980 (smoothing constructed, documented honestly); tandem plug-limited
central-cell confinement calibrated to the Realta Hammir Q>5 design point (Frank
et al. 2024).

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
| Mirror | D-T | sized p_input (eff) | 40.0 MW (fixed YAML) | 2.0 MW (aux floor) | energy-balance closed |
| Mirror | D-T | sized T_i (keV) | 59.77 (ignited) | 29.86 | tandem plug confinement (Task 2b) |
| Mirror | D-T | sized tau_E (s) | about 2.4 (over-credited) | 0.96 | bounded plug potential (Task 2b) |
| Mirror | D-T | sized LCOE ($/MWh) | 109.70 | 100.53 | energy-balance + plug confinement |

## Energy-balance closure (Task 2)

Task 2 adds `mirror_aux_heating` (the mirror analog of the tokamak's
`aux_heating_from_confinement`): in sizing mode the auxiliary sustainment power
is `P_aux = max(P_aux_floor, P_end + P_radial + P_rad - P_alpha)` and that value
(not the fixed YAML `p_input`) is fed to the shared `mfe_forward_power_balance`.
The forward/inverse audit path keeps the stated `p_input` and reports
`sustainment_ratio = p_input / P_aux` as a consistency diagnostic.

## Tandem plug-limited central-cell confinement (Task 2b)

After Tasks 1 and 2 the D-T mirror spuriously IGNITES: the corrected collisional
bridge correctly places the deeply collisionless central cell on the Pastukhov
branch, but the confining potential fed to the Pastukhov enhancement was the
unbounded simple-mirror Boltzmann ambipolar value
`e*phi = T_e * ln(sqrt(m_i / (2 pi m_e)))` (about 3 to 9 `T_e`, growing without
limit). At the operating point that over-credits `tau_E` by 15 to 24x and pins
the D-T optimum at about 60 keV.

The defect is a physics-scope mismatch. The CAS22 coil account already costs a
TANDEM (`n_plug_coils = 4`, Hammir class: two high-field HTS end plugs per end
plus the long central solenoid). The confinement therefore must be the tandem
central-cell confinement, where the ions are held by the electrostatic potential
drop to the end plugs, NOT the simple-mirror electron-loss potential. In a
classical tandem (Fowler and Logan 1977) the ion confining potential between the
central cell and the end plug is set by their density ratio:

```
e*phi_c = T_e * ln(n_p / n_c)
```

(Frank et al. 2024 eq. 3.4; Fowler and Logan 1977). Because `n_p / n_c` is an
order-unity-to-few ratio, `e*phi_c` is BOUNDED, unlike the simple-mirror value.
This is the right tandem mechanism; the model only needs the confining potential
calibrated to a real tandem design point.

### The Hammir anchor (load-bearing)

The Realta Hammir Q>5 pre-conceptual design point (Frank et al. 2024,
arXiv:2411.06644, sec. 3.5, the `am = 0.15` m POPCON case) is:

- central cell length `ell_c = 50` m
- mirror throat field `B_m = 25` T, central cell field `B0c` about 3 T,
  central cell mirror ratio `R_mc = 13.3` (finite beta, `beta_c` about 0.6)
- end-plug density `n_p = 1.5e20` m^-3, density ratio `n_c / n_p = 0.55`, so
  `n_c = 0.825e20` m^-3
- `T_i = 45` keV, `T_e = 125` keV
- fusion power `P_fus = 157.4` MW, plug neutral-beam power `P_NBI = 30` MW, so
  the system gain `Q = P_fus / P_NBI = 5.2` (the Q>5 headline; the Realta
  announcement quotes `Q_sci` about 5.3)

At this point the tandem confining potential is
`e*phi_c = 125 * ln(1 / 0.55) = 74.7` keV, i.e. the ratio

```
e*phi_c / T_i = 74.7 / 45 = 1.66.
```

The model evaluated at the Hammir central cell with this calibrated ratio
reproduces the design point: the Pastukhov axial confinement time is about 6.3 s
(the paper quotes a tandem central-cell `tau_c` about 5 s, sec. 3.5; within the
2x anchor band), the central cell is NOT ignited (the confinement-required
auxiliary power is about 27 MW, matching the published 30 MW plug NBI), and the
gain `Q = P_fus / P_aux` is about 5.8 (within 2x of the published 5.2). By
contrast the unbounded Boltzmann potential at `T_e = 125` keV gives
`e*phi = 412` keV (`e*phi/T_i = 9.2`), which floors the auxiliary power and
produces the spurious ignition.

### The calibrated plug-confinement form

The model replaces the unbounded Boltzmann potential in the confinement chain
with the bounded tandem value (`compute_plug_potential`):

```
e*phi = plug_phi_over_T_i * T_i,     plug_phi_over_T_i = 1.66 (YAML default),
```

calibrated to the Hammir `e*phi_c / T_i` ratio above. `compute_ambipolar_potential`
(the simple-mirror Boltzmann value) is retained as a diagnostic only. The
collisionality bridge of Task 1 is unchanged (the tandem central cell runs
collisionless and correctly sits on the plugged branch).

Capping the ratio rather than carrying `T_e * ln(n_p/n_c)` with a free density
ratio is a deliberate costing-fidelity choice: the 0D model does not solve the
end-plug density self-consistently, so the single calibrated ratio fixes the
central-cell confinement to the published tandem operating point. The honest
caveat is that this is one anchored operating point, not a swept `n_p/n_c`
response; the ratio is pinned to Hammir and carries the same 2x validation band
as the GDT and WHAM kernel anchors.

### DEC path (corrected end-loss-only routing)

The mirror direct converter sits at the end-plug expanders and recovers only the
AXIAL end-loss channel `P_end`; `P_radial` and `P_rad` strike the lateral first
wall and go to the thermal cycle. The shared balance instead routes DEC off its
own `p_transport = p_ash + p_input_eff - p_rad`. Two corrections make the routing
physical without modifying the shared function:

1. `p_rad_override = ps.p_rad` pins the shared function's radiation to the
   mirror forward's own `p_rad` (clamped to `p_alpha`, open-ended synchrotron
   geometry, no impurity line radiation). Without this the shared function
   recomputed a different, impurity-laden, unclamped `p_rad` (about 10x larger
   at the sized point), which broke both the `p_transport` identity and the DEC
   credit. This is the reviewer's dual-`p_rad` finding, now fixed.
2. With the radiation consistent, an effective `f_dec_eff = f_dec * P_end /
   p_transport_shared` makes the shared term recover exactly
   `f_dec * eta_de * P_end`, the recoverable axial end-loss, so the DEC credit
   tracks `P_end` (which falls as confinement improves with temperature).

With the plasma now plug-limited rather than ignited, the clean identity
`p_transport == P_end + P_radial` holds at the sized D-T optimum (the auxiliary
power sits at the ignition-threshold floor where `P_end + P_radial + P_rad -
P_alpha` equals the floor), so `test_p_transport_identity_in_sizing` passes and
its earlier xfail is removed.

### Settled D-T regime

With the tandem calibration the sized D-T optimum lands at `T_i` about 29.9 keV
(was about 60 keV ignited), `tau_E` about 0.96 s (physical; `P_end` well below
`P_fus`), and the energy balance closes with auxiliary sustainment at the control
floor at the optimum. The optimum is a tandem-realistic temperature: it sits
between the cool simple-mirror GDT/WHAM cells (about 10 keV) and the Hammir
central-cell design point (45 keV), as expected for a tandem central cell that
must be hot enough for alpha heating to nearly sustain it against the plug-limited
end loss. The neutron-wall cap is binding (density set by `q_wall_max`), and the
net-electric objective peaks near 30 keV because the end-loss-based DEC credit
falls with temperature while the central cell ignites just above it. The earlier
"ignited-plateau" finding (D-T parked at 60 keV) is resolved by the bounded plug
potential: confinement no longer runs away, so the optimizer settles in the
tandem regime on its own without an explicit stability bound.

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
- **Frank, S. J. et al. (2024)**, "Confinement performance predictions for a
  high field axisymmetric tandem mirror," arXiv:2411.06644 (under consideration,
  J. Plasma Phys.). Primary source for the Realta Hammir Q>5 tandem central-cell
  design point (eq. 3.4 tandem confining potential `e*phi = T_e ln(n_p/n_c)`;
  sec. 3.5 design point: `ell_c = 50` m, `n_c/n_p = 0.55`, `T_i = 45` keV,
  `T_e = 125` keV, `P_fus = 157.4` MW, `P_NBI = 30` MW, `tau_c` about 5 s).
- **Fowler, T. K. and Logan, B. G. (1977)**, "The tandem mirror reactor,"
  Comments Plasma Phys. Control. Fusion 2, 167. Classical tandem-mirror concept:
  the central-cell ions are confined by the electrostatic potential drop to the
  end plugs, set by the plug-to-central density ratio.

No number in this document is sourced from or calibrated against any
cost-modeling tool; all come from the primary plasma-physics literature, except
the constructed smoothing width `w`, which is documented as constructed.
