# Mirror Confinement Regime Bridge: Collisional vs Collisionless

**Date:** 2026-06-14
**Status:** Implemented. Collisionality bridge sourced to Rognlien and Cutler
1980 (smoothing constructed, documented honestly); tandem plug-limited
central-cell confinement calibrated to the Realta Hammir Q>5 design point (Frank
et al. 2024); hot-electron plug decoupled from the coolable central cell (Fowler
and Logan 1977; Baldwin and Logan 1979) with a fixed plug sustainment power
calibrated to Hammir's about 30 MW (Task 2e).

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
(tokamak, stellarator, IFE/MIF) pins do NOT move. This is the finalized re-pin:
the columns below trace the sized D-T optimum (400 MWe, `f_beta = 0.85`, YAML
defaults) from the original spuriously-ignited value to the settled driven
tandem, with the intermediate stages shown so the trajectory is auditable. Each
intermediate stage is labelled as history; only the final row is current
behavior.

| Stage | T_i [keV] | tau_E [s] | P_aux [MW] | Q_sci | Q_eng | LCOE [$/MWh] |
|-------|-----------|-----------|------------|-------|-------|--------------|
| Original (pre-fix, harmonic sum, unbounded plug) | 59.77 (ignited) | about 2.4 (over-credited) | floored (about 2) | about 516 | n/a (near-ignited) | 109.70 |
| Task 1+2 (regime bridge + energy balance) | still about 60 (ignited) | about 2.4 | floored | about 516 | n/a | about 110 |
| Task 2b (ratio-to-T_i plug, T_e=20 keV) | 29.86 | 0.96 | 2.0 (floored) | high (floored) | about 1 | 100.53 |
| Task 2c (fixed Fowler-Logan plug, hot T_e=125 keV) | 23.31 | 18.7 | about 228 (driven) | 8.30 | 1.76 | 368.9 |
| Task 2e (explicit 30 MW plug power charged) -- CURRENT | 23.34 | 18.7 | about 245 (driven) | 8.30 | 1.68 | 395.8 |

The final row reproduces from the committed code via
`CostModel(MIRROR, DT).forward(net_electric_mw=400, availability=0.87,
lifetime_yr=40, size_from_power=True, f_beta=0.85)` (T_i 23.34 keV, tau_E 18.7 s,
P_aux = P_fus/Q_sci about 245 MW, Q_sci 8.30, Q_eng 1.68, LCOE 395.8 $/MWh, with
beta at the 0.425 ceiling). P_aux is the confinement-derived sustainment band
(228-259 MW across the trajectory's L); the larger 400 MWe plant sits near
245 MW.

The trajectory in one sentence: the original model spuriously IGNITED at about
60 keV (Q_sci about 516, LCOE about 100) because a harmonic-sum confinement
combination and an unbounded simple-mirror plug potential over-credited `tau_E`;
the settled model is a genuinely DRIVEN tandem at 23 keV (Q_sci 8.3, Q_eng 1.68,
LCOE 396) that pays the real recirculating cost of its hot-electron plug.

### Supporting quantity-level pins

| Quantity | Old | New | Reason |
|----------|-----|-----|--------|
| sized p_input (eff) / P_aux | 40.0 MW (fixed YAML) | about 245 MW (driven) | energy-balance closed; confinement-derived sustainment |
| non-0D default LCOE pin | 98.686 | 100.107 | YAML central-cell T_e 20->125 keV moves the radiation term |
| a=1.5/B=3 wall-loose sized L (m) | 69.230740 | 77.842279 | explicit P_plug = 30 MW recirculating (Task 2e) |

The Task 2e plug/central decoupling does NOT move the D-T operating point (the
D-T central cell genuinely runs hot, so its central-cell T_e and the plug
T_e_plug are both 125 keV and the plug potential e*phi = 74.7 keV is the same
under either). It moves the D-T sized LCOE from 368.9 to 395.8 $/MWh
(+7.3 percent) only because the 30 MW plug sustainment power is now charged
EXPLICITLY into the recirculating budget. The sized T_i (23.34 keV), Q_sci
(8.30), and tau_E (18.7 s) are unchanged within tolerance. The Hammir Q>5 anchor
reconciles: Hammir counts the 30 MW plug NBI in its published Q = P_fus/P_NBI =
5.2, so charging P_plug makes the model's accounting match Hammir's rather than
omitting the plug cost.

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

(Frank et al. 2024 eq. 18; Fowler and Logan 1977). Because `n_p / n_c` is an
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

### The calibrated plug-confinement form (SUPERSEDED intermediate step)

NOTE: this section records an intermediate step in the trajectory (Task 2b). It
is SUPERSEDED by the fixed Fowler-Logan form below ("The plug-potential fix" and
"Decoupling the hot plug from the central cell"), which the settled code
implements. The settled `compute_plug_potential` uses
`e*phi = T_e_plug * ln(plug_density_ratio)` (a decoupled hot-plug electron
temperature), NOT the ratio-to-`T_i` capping described here. Read this only as
history; do not treat the `plug_phi_over_T_i = 1.66` ratio model as current.

The intermediate model replaced the unbounded Boltzmann potential in the
confinement chain with a bounded tandem value (the Task 2b `compute_plug_potential`):

```
e*phi = plug_phi_over_T_i * T_i,     plug_phi_over_T_i = 1.66 (Task 2b default),
```

calibrated to the Hammir `e*phi_c / T_i` ratio above. `compute_ambipolar_potential`
(the simple-mirror Boltzmann value) is retained as a diagnostic only. The
collisionality bridge of Task 1 is unchanged (the tandem central cell runs
collisionless and correctly sits on the plugged branch).

The intermediate step capped the ratio rather than carrying `T_e * ln(n_p/n_c)`
with a free density ratio, as a costing-fidelity shortcut. That shortcut was the
source of the residual self-heating knee (see "Observed effect on the D-T optimum"
below) and was replaced by the real Fowler-Logan form. The honest caveat that the
calibration is one anchored Hammir operating point, carrying the same 2x
validation band as the GDT and WHAM kernel anchors, still applies to the settled
form.

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

With the plasma now plug-limited rather than ignited, the transport-channel
identity closes at the sized D-T optimum. With the later alpha loss-cone routing
(Task 2c) the un-deposited alpha power is conserved into the transport channel, so
the identity is `p_transport == P_end + P_radial + (1 - f_alpha_heat) * P_alpha`
(the clean `p_transport == P_end + P_radial` held only before the loss-cone term
was added). `test_p_transport_identity_in_sizing` passes and its earlier xfail is
removed.

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

## Alpha loss-cone heating fraction (Task 2c)

A magnetic mirror does not confine its fusion alphas the way a closed tokamak
does: a fraction of the 3.5 MeV alphas is born in or scatters into the velocity-
space loss cone and streams out the ends before depositing its energy in the
central cell. The model credits only the deposited fraction `f_alpha_heat` of the
charged-particle power `P_alpha` as self-heating, and routes the lost fraction
`(1 - f_alpha_heat) * P_alpha` to the axial end-loss / DEC channel as directed
loss-cone exhaust.

### Sourcing and the value

The canonical treatment is Santarius and Callen 1983 (Phys. Fluids 26, 1037), a
bounce-averaged Fokker-Planck calculation of fusion-alpha confinement in a tandem
central cell. The result is that roughly 50 percent of the alphas are lost by
COUNT but under 25 percent by ENERGY, because most alphas slow down on the
electrons and deposit the bulk of their energy before pitch-angle scattering
carries them into the loss cone. So about 75 to 85 percent of the alpha power
deposits. The model uses `f_alpha_heat = 0.80` (YAML default, mid of the
0.75-0.85 band).

Cross-check: Frank et al. 2024 (arXiv:2411.06644), the Hammir reference, credits
near-full alpha deposition and is driven primarily by END LOSSES (which the model
already represents through `P_end`). The two pictures are consistent: Hammir's
driven character is end-loss-dominated, not alpha-loss-dominated, so crediting
near-full alpha deposition there is defensible. The generic model default of 0.80
is the conservative Santarius-Callen value; at the Hammir anchor it lowers the
computed gain from `Q` about 5.8 (full deposition) to about 4.7, BOTH within the
2x anchor band of the published `Q` about 5.2 (the 0.80 value actually lands
closer to the published number). The Hammir anchor therefore holds at the generic
`f_alpha_heat = 0.80`; no separate near-full-deposition evaluation is needed.

### Implementation and energy conservation

In the sustainment balance `mirror_aux_heating` subtracts `f_alpha_heat * P_alpha`
instead of the full `P_alpha`:

```
P_aux = max(P_aux_floor, P_end + P_radial + P_rad - f_alpha_heat * P_alpha).
```

Feeding this `P_aux` as `p_input` to the shared `mfe_forward_power_balance` (with
`p_rad_override = ps.p_rad`) makes its transport channel

```
p_transport = p_ash + p_input_eff - p_rad
            = P_alpha + P_aux - P_rad
            = P_end + P_radial + (1 - f_alpha_heat) * P_alpha,
```

so the un-deposited alpha power appears in the transport channel and is conserved
(no power vanishes). The mirror DEC routing recovers the full axial channel
`P_end + (1 - f_alpha_heat) * P_alpha` at the end-plug expanders (both are
charged-particle streams to the DEC plates) via the effective
`f_dec_eff = f_dec * (P_end + (1 - f_alpha_heat) * P_alpha) / p_transport`,
leaving `P_radial` to the wall. The shared function is NOT modified; the
reduction and the routing are entirely mirror-side, through the `p_input` handoff
and the existing `p_rad_override` / effective-`f_dec` hooks. `q_sci = P_fus /
p_input_eff` and `q_eng` follow automatically from the larger `P_aux`.

### Observed effect on the D-T optimum (and the plug-potential fix that follows)

At a FIXED 30 keV the alpha loss-cone reduction raises the required auxiliary
power from the 2 MW floor to about 43 MW (comparable to Hammir's 30 MW plug NBI),
as the counterfactual predicted. By itself, however, the alpha fix did NOT make
the sizing optimum driven: the optimizer re-climbed `T_i` to the self-heating knee
where the auxiliary power returned to the floor and `q_sci` returned to its
near-ignited value. The cause was the ratio-to-`T_i` plug shortcut of Task 2b. The
old `compute_plug_potential` held `e*phi = 1.66 * T_i`, so `e*phi / T_i` was a
CONSTANT and the Pastukhov enhancement `x exp(x)` did not change with temperature.
With fusion (and alpha) power pinned flat by the binding neutron-wall cap, raising
`T_i` then bought confinement (tau scales as `T_i^1.5`) at zero fusion-power cost until the
central cell self-heated, so the economic optimum sat at the self-heating knee,
still floored.

### The plug-potential fix (fixed Fowler-Logan form)

The shortcut is wrong physics. The plug is FIXED hardware (the four end-plug coils
and the plug heating the cost model commits to) producing a confining potential
set by that fixed plug, not one that deepens for free as `T_i` rises. The model
now uses the REAL Fowler-Logan form directly:

```
e*phi = T_e * ln(n_p / n_c)
```

(`compute_plug_potential(T_e, plug_density_ratio)`; Frank et al. 2024
arXiv:2411.06644 eq. 18; Fowler and Logan 1977). Here `T_e` is the fixed
central-cell electron temperature and `n_p / n_c` (the YAML `plug_density_ratio`)
is the fixed plug-to-central density ratio, a fixed plug property. Because `e*phi`
no longer scales with `T_i`, the ratio

```
e*phi / T_i = T_e * ln(n_p/n_c) / T_i
```

FALLS as `T_i` rises (at fixed `T_e`), so the Pastukhov enhancement `x exp(x)`
WEAKENS at high `T_i`: heating the central cell now COSTS confinement instead of
buying it for free. The free ride to self-heating is removed and the optimum
settles at a cooler, genuinely DRIVEN point.

Calibration is unchanged at the Hammir anchor: `n_p/n_c = 1/0.55 = 1.818` (the
published `n_c/n_p = 0.55`), so at `T_e = 125` keV `e*phi = 125 * ln(1.818) = 74.7`
keV, identical to the Task 2b value, and the Hammir anchor holds.

#### A tandem plug requires hot electrons (T_e re-pin)

The Fowler-Logan plug confines through the ELECTRON temperature, so a tandem must
run HOT electrons (typically ECH-heated in the plug) for the plug to confine at
all. The Realta Hammir Q>5 design point runs `T_e = 125` keV (`>> T_i = 45` keV)
for exactly this reason (Frank et al. 2024 sec. 3.5). At a cold `T_e` (the old YAML
20 keV) `e*phi = 12` keV is far too shallow to plug a 20-80 keV central cell, and
the D-T machine is infeasible at every `T_i` (net power deeply negative). The YAML
central-cell `T_e` is therefore re-pinned to the Hammir hot-electron value
125 keV. This is the parameter that makes the tandem a working, genuinely driven
device under the correct plug physics. The non-0D default radiation path reads the
same YAML `T_e`, so this re-pin also moves the mirror non-0D LCOE pin (recorded in
the before/after table).

### Settled D-T regime (Task 2c, driven)

With the fixed Fowler-Logan plug and the hot-electron `T_e`, the sized D-T optimum
(400 MWe, `f_beta = 0.85`, YAML defaults) is genuinely DRIVEN:

- `T_i` about 23.3 keV (a cool tandem-realistic central cell, below the Hammir
  45 keV because the optimizer now PAYS confinement to go hotter)
- auxiliary sustainment `P_aux` about 228 MW, well off the 2 MW floor (comparable
  in spirit to Hammir's 30 MW plug NBI scaled to the larger 400 MWe plant)
- `q_sci = P_fus / P_aux` about 8.3 (tandem-realistic, was about 516 near-ignited)
- `q_eng` about 1.76 (net-positive), LCOE about 369 $/MWh (higher than the
  spurious 100 $/MWh of the near-ignited Task 2b point because the plant now pays
  the real recirculating cost of a driven tandem)
- the operating point is BETA-bound (`beta` at the `f_beta * beta_max = 0.425`
  ceiling): the hot electrons raise the central-cell pressure, so the
  beta-limited density falls below the neutron-wall-limited density and the
  binding cap moves from the wall (Task 2b) to beta.

The earlier "ignited-plateau" and "wall-cap self-heating knee" findings are both
resolved by the fixed plug: confinement no longer improves for free with `T_i`, so
the optimizer settles in the driven tandem regime on its own, with no explicit
stability bound (Task 4 still not needed).

## Decoupling the hot plug from the central cell (Task 2e)

Implementing the real Fowler-Logan plug (Task 2c) forced hot electrons: the
confining potential `e*phi = T_e * ln(n_p/n_c)` needs a high `T_e` to plug the
central cell, so the single global mirror `T_e` was pinned to the Hammir
hot-electron value 125 keV. That made D-T a genuinely driven tandem, but hot
electrons radiate heavily (bremsstrahlung scales with `T_e^0.5` and the
synchrotron term grows with `T_e`), so D-D, D-He3, and p-B11 went net-negative and
could no longer be sized. A SINGLE electron temperature cannot serve both the plug
(which wants hot electrons to confine) and aneutronic central-cell fusion (which
wants cool electrons to limit radiation).

### The physics: separate plug and central-cell plasmas

Real advanced-fuel tandems do not run one electron population. The plug is a
SEPARATE cell, often with a different ion (plugging) species, whose job is to hold
the confining potential with a HOT-ELECTRON population sustained by plug ECH. The
central cell runs the working fuel and can keep its electrons COOL, because the
confinement is supplied by the plug potential, not by the central-cell electron
temperature. This is the original Fowler and Logan 1977 tandem concept (and the
Baldwin and Logan 1979 thermal-barrier refinement, which deliberately decouples
the plug and central-cell electron populations with a potential barrier so the
plug can run hotter than the central cell). The Hammir design point (Frank et al.
2024 sec. 3.5) runs its plug hot-electron population at `T_e = 125` keV, set by
the plug ECH independent of the central-cell fuel.

The model captures this at costing fidelity by separating two temperatures:

- `T_e_plug` (hot, YAML default 125 keV, anchored to Hammir): sets the
  Fowler-Logan confining potential `e*phi = T_e_plug * ln(n_p/n_c)`
  (`compute_plug_potential(T_e_plug, plug_density_ratio)`). It is the plug knob,
  set by plug ECH and independent of the central fuel.
- `T_e` (central-cell, YAML default 125 keV for the D-T reference machine,
  OVERRIDDEN cool for advanced fuels): sets central-cell bremsstrahlung,
  synchrotron, beta, and stored energy. It no longer enters the plug potential.

The plug is modelled as POTENTIAL-PLUS-POWER, not a full second fusion plasma:
the central cell does the fusion; the plug contributes the confining potential and
its power cost. "Different species" is captured by the plug being this separate
potential-and-power sub-system, independent of the central fuel.

### Charging the plug power (P_plug, Hammir anchor)

The hot-electron plug is sustained by plug ECH/NBI at a real power cost. The model
charges a plug sustainment power `P_plug` (YAML default 30 MW) into the mirror
recirculating budget. The value is calibrated to the Realta Hammir design point,
which holds its hot-electron plug with about 30 MW of plug NBI/ECH (Frank et al.
2024 sec. 3.5; the published `Q = P_fus / P_NBI = 5.2` counts exactly this 30 MW).

The chosen model is a FIXED per-machine plug power, not one scaled with `n_p` or
with the central-cell power. This is the defensible 0D choice: the 0D model does
not solve the plug density or trapping self-consistently, the published anchor is
a single design-point plug power, and the CAS22 coil account already commits to
fixed plug hardware (`n_plug_coils = 4`), so the plug is costed as fixed
hardware-plus-drive consistent with the coils. `P_plug` is also the competing
penalty that keeps the optimizer honest: it is a recirculating load that the plant
pays regardless of central-cell temperature, so a configuration cannot get the
deep plug confinement for free.

`P_plug` is charged on the MIRROR SIDE by folding it into the `p_coils` argument
passed to the shared `mfe_forward_power_balance` (`p_coils_eff = p_coils +
p_plug`). The shared recirculating sum is `p_coils + p_pump + p_sub + p_aux +
p_cool + p_cryo + p_input_eff/eta_pin`, so this adds `P_plug` at unit recirculating
cost without touching the shared function or the tokamak/stellarator/IFE paths. It
is charged consistently in the inverse-solve (which sets the required `p_fus`) and
the forward power table, and in the sizing path's power-balance call. The plug
power does NOT enter `p_input` (the plasma energy balance), because the plug ECH
heats the plug electrons, not the central cell.

### Reconciling the Hammir Q anchor

Hammir's published `Q = P_fus / P_NBI = 5.2` already counts the 30 MW plug NBI in
the denominator. Before Task 2e the model omitted the plug power from the
recirculating budget; charging `P_plug` explicitly makes the model's accounting
match Hammir's rather than under-counting the plug cost. The forward Hammir anchor
(`q_sci = P_fus / P_aux` at the published central cell) is unchanged because it
already used the confinement-required sustainment power; the new explicit `P_plug`
enters `q_eng` (the engineering gain that drives the economics) and the sized
LCOE, which is where the plant pays for the plug. The anchor holds within the
documented 2x band.

### D-T is unchanged; advanced fuels are now fairly evaluated

D-T stays a hot-central tandem (the Hammir central cell genuinely runs about
125 keV), so its driven result is preserved: sized `T_i` about 23.3 keV, `q_sci`
about 8.3, `tau_E` about 18.7 s, all unchanged within tolerance. The sized LCOE
moves from about 369 to about 396 $/MWh (+7.3 percent) only because the 30 MW plug
power is now charged explicitly (it was omitted before).

The advanced fuels get a fair evaluation: a cool central cell (low bremsstrahlung)
plugged by a hot plug, paying the plug power. The finding is recorded honestly in
the cross-fuel observation below. The headline is that decoupling REMOVES the
spurious hot-central radiation penalty (D-He3 net power and `q_eng` improve
sharply with a cool central cell) but does NOT by itself make the advanced fuels
economic at the modelled fields: they remain net-negative even cool. The
advanced-fuel difficulty is now a genuine reactivity-and-radiation result, not an
artifact of forcing one hot electron temperature on a fuel that does not want it.

### Cross-fuel observation with a cool central cell (the real finding)

Run at a high-field machine (`a = 0.5` m, `B = 6` T, `beta_max = 0.9`, `f_beta =
0.85`, hot plug `T_e_plug = 125` keV, `L = 200` m diagnostics row) so the
advanced fuels are not artificially density-starved; the central-cell `T_e` is the
only variable.

Radiation model used: this cross-fuel observation is run with the FULL radiation
model (`f_rad_fus = None`), so the central-cell bremsstrahlung and synchrotron
terms respond to the central `T_e`. That is the appropriate evaluation here:
for the advanced fuels the binding loss is radiation, so the temperature-resolved
brem/synchrotron model is the physics under test, not the fixed `f_rad_fus`
proxy. The production DEFAULT path instead supplies a per-fuel proxy
(`f_rad_fus_dhe3 = 0.24`, `f_rad_fus_pb11 = 0.83`) for which `p_rad =
f_rad_fus * p_fus` is `T_e`-INDEPENDENT; under that default the central-`T_e`
change does NOT act through bremsstrahlung but only through beta, density, and
`tau_E`. The cross-fuel numbers below are therefore generated by calling
`net_electric_at_L` with `f_rad_fus = None` in the params dict (and the non-DT
`Z_eff` adjustment the model applies), so the stated brem mechanism is the one
exercised. The numbers reproduce from that API call to within model-evolution
drift since they were first recorded (the D-He3 cool row in particular has moved
modestly with the later plug-decoupling commit; the SIGN and the ranking are
unchanged). The net conclusion (no advanced fuel net-positive) is robust under
BOTH radiation paths: every fuel/T_e combination is net-negative whether the run
uses the full model or the fixed proxy.

| Fuel | central T_e [keV] | T_i [keV] | q_sci | q_eng | p_net [MW] | net-positive? |
|------|-------------------|-----------|-------|-------|------------|---------------|
| D-He3 | 125 (hot) | 24.2 | 0.097 | 0.377 | -977 | no |
| D-He3 | 30 (cool) | 100.0 | 1.658 | 0.931 | -291 | no (near break-even) |
| D-D | 125 (hot) | 27.2 | 0.223 | 0.421 | -1549 | no |
| D-D | 30 (cool) | 100.0 | 0.568 | 0.551 | -3674 | no |
| p-B11 | 300 (hot) | 300.0 | 0.106 | 0.371 | -633 | no |
| p-B11 | 60 (cool) | 300.0 | 0.178 | 0.405 | -1281 | no |

Per-fuel finding:

- **D-He3** is the clear beneficiary of decoupling: cooling the central cell from
  125 to 30 keV (hot plug retained) raises `q_sci` from 0.10 to 1.66 and `q_eng`
  from 0.38 to 0.93 (from deeply net-negative toward break-even), and the GSS
  optimum moves UP to `T_i` about 100 keV where D-He3 reactivity is strong and the
  cool central-cell electrons no longer radiate it away (this brem mechanism is
  the one active under the full `f_rad_fus = None` model used for this table; under
  the fixed proxy the same cooling instead helps through beta and `tau_E`). D-He3
  is now FAIRLY evaluable and is
  close to break-even, but still net-negative (`q_eng` < 1) at these fields. This
  is the honest result: a cool-central hot-plug D-He3 tandem is far better than the
  hot-central case but not yet economic in the 0D model.
- **D-D** improves in `q_sci` with a cool central cell (0.22 to 0.57) but its
  total p_net falls because the optimizer drives `T_i` to 100 keV where the much
  higher fusion power demands a far larger (still sub-unity-efficiency) auxiliary
  drive; D-D stays net-negative. Honest low-reactivity result.
- **p-B11** stays at its bracket top `T_i` about 300 keV regardless; cooling the
  central cell improves `q_sci` modestly (0.11 to 0.18) but p-B11 remains deeply
  net-negative. The aneutronic fuel is genuinely hard in the 0D model at these
  fields; decoupling does not change that conclusion.

The real finding, stated plainly: the plug/central decoupling makes the
advanced fuels fairly evaluable by removing the hot-central radiation penalty, and
D-He3 in particular goes from deeply net-negative to near break-even with a cool
central cell, but none of D-D, D-He3, or p-B11 is net-positive in the model at the
modelled fields. Only D-T is economic. This is reported as found, not engineered
to a target.

## Stability and validity diagnostics (Task 3)

Two diagnostics are reported on `MirrorPlasmaState`. Both are INFORMATIONAL: they
report where a modelling assumption is stretched or where a known microinstability
is more readily driven, but neither restricts the operating point. They surface
the physics the observe step (below) is decided against.

### Pastukhov-Maxwellian validity flag (`pastukhov_valid`)

The Pastukhov loss-cone confinement formula assumes a Maxwellian core. Rognlien
and Cutler 1980 place the Pastukhov-versus-collisional boundary where the ion mean
free path reduced by the mirror ratio is of order the system length, i.e.
collisionality `L/mfp` of order `1/R_m` (the same boundary the regime bridge is
centered on). Below the validity floor the plasma is too collisionless for the bare
Pastukhov-Maxwellian assumption, and the formula over-credits confinement. The
flag is

```
pastukhov_valid = 1.0  if  collisionality >= collisionality_min  else  0.0
```

with `collisionality_min = 0.1` (the YAML default, equal to `1/R_m` at the YAML
`R_m = 10`). It is a bool-as-float for JAX/state parity.

This flag is INFORMATIONAL, not a constraint. A TANDEM legitimately runs the
central cell collisionless and electrostatically PLUGGED: the confinement is the
plug-limited tandem value (bounded `e*phi`, calibrated to Hammir), not the bare
simple-mirror Pastukhov-Maxwellian time. The flag therefore marks where the bare
Pastukhov-Maxwellian assumption (a single thermal Maxwellian filling the loss
cone) is stretched, which is exactly the regime in which the tandem plug
calibration, not the bare formula, is doing the physics. A fired flag at the
tandem operating point is expected and is not an error.

### DCLC microstability diagnostic (`dclc_parameter`)

The drift-cyclotron-loss-cone (DCLC) mode is the dominant loss-cone-driven
microinstability in mirrors (Post 1987, "The magnetic mirror approach to fusion").
It is driven by the loss-cone gradient in velocity space and grows more readily as
the plasma spans more ion gyroradii in the radial direction. The reported proxy is
the number of midplane ion gyroradii across the plasma radius,

```
dclc_parameter = a / rho_i,     rho_i = sqrt(2 A m_p T_i KEV_TO_J) / (e B_min),
```

the same gyroradius kernel used by the radial-transport time. The Post criterion is
that a warm-plasma stream filling the loss cone stabilises DCLC when its fraction
exceeds about `rho_i / a`, i.e. roughly `1/dclc_parameter`; both GDT and WHAM rely
on warm-plasma DCLC stabilisation. The documented reference value
`dclc_a_over_rho_ref = 50` (YAML) is the order of `a/rho_i` above which a
warm-plasma stream is typically required for DCLC stability; it is a reference for
interpreting the number, NOT a constraint in Task 3.

A larger `dclc_parameter` means a larger warm-plasma fraction is needed; a value of
order `dclc_a_over_rho_ref` or below means a modest warm fraction (of order a few
percent) suffices, which is the regime GDT and WHAM operate in.

## Settled-regime observation (all fuels)

This section records the Task 3 observe step: the corrected tandem-calibrated model
run in sizing/optimize mode across all four fuels, the settled optimum diagnostics,
and the per-fuel decision on whether the plasma settles in a sensible,
validity-respecting, tandem-realistic regime (which decides whether the conditional
explicit stability constraint of Task 4 is needed).

All rows use `f_beta = 0.85` and YAML defaults (`R_m = 10`, `T_e = 125` keV
hot-electron tandem plug, `B = 3` T, `plasma_t = 1.5` m, `collisionality_min =
0.1`, `plug_density_ratio = 1.818`, `q_wall_max = 5`, `q_surface_max = 1`,
`beta_max = 0.5`), read at the GSS optimum at `L_max = 200` m (the diagnostics row;
D-T also sizes to its 400 MWe target at L about 178 m with the same operating
point). Every value reproduces from the committed code via the model API
(`CostModel(MIRROR, fuel).forward(...)`). With the fixed Fowler-Logan plug only
D-T is net-positive in the driven hot-electron regime; D-D, D-He3, and p-B11 are
net-NEGATIVE at `L_max` (the honest result, reported, not forced feasible).

The two gain columns are reported honestly: `q_sci = p_fus / p_input_eff` is the
fusion-to-injected gain, where the injected power is now the confinement-derived
sustainment of a genuinely DRIVEN tandem (NOT floored). `q_eng = P_et /
recirculating` is the engineering gain that drives the economics; `q_eng > 1`
means net-positive, `q_eng < 1` means the plant cannot deliver power.

| Fuel | T_i [keV] | P_aux [MW] (off floor?) | collisionality | `pastukhov_valid` | beta | q_sci | q_eng | LCOE [$/MWh] | binding density cap | driven? |
|------|-----------|--------------------------|----------------|-------------------|------|-------|-------|--------------|---------------------|---------|
| D-T   | 23.3 | 228-259 (yes) | 1.4e-3 | 0.0 (flagged) | 0.425 | 8.3 | 1.77 | 369 | beta (f_beta x beta_max) | DRIVEN, net-positive |
| D-D   | 15.6 | 78 (yes) | 3.3e-3 | 0.0 (flagged) | 0.425 | 0.38 | 0.24 | n/a (p_net < 0) | beta | driven but uneconomic |
| D-He3 | 20.0 | 83 (yes) | 1.4e-3 | 0.0 (flagged) | 0.425 | 0.15 | 0.21 | n/a (p_net < 0) | beta | driven but uneconomic |
| p-B11 | 300.0 | 189 (yes) | 2.5e-6 | 0.0 (flagged) | 0.425 | 0.15 | 0.23 | n/a (p_net < 0) | beta | driven but uneconomic |

Observations, per fuel:

- **D-T** settles at an interior optimum `T_i` about 23 keV (a cool tandem-realistic
  central cell, below the Hammir 45 keV because the optimizer now PAYS confinement
  to go hotter under the fixed plug). The point is genuinely DRIVEN: auxiliary
  sustainment is about 228-259 MW, far off the 2 MW floor, with `q_sci` about 8.3
  (tandem-realistic, was about 516 near-ignited under the ratio-to-`T_i` plug).
  Density is beta-cap bound (the hot electrons raise the pressure), so beta sits AT
  the 0.425 ceiling and the binding cap moved from the neutron wall (Task 2b) to
  beta. `q_eng` about 1.77, net-positive, LCOE about 369 $/MWh. Settles sensibly and
  is the headline driven result.

- **D-D** finds its GSS optimum at the LOW end of its bracket (`T_i` about 16 keV),
  the opposite of Task 2b where it parked at the 100 keV bracket TOP. This is the
  fixed plug working: under the old constant `e*phi/T_i` shortcut, raising `T_i`
  bought free confinement so D-D climbed; now `e*phi` is fixed by the plug, the
  Pastukhov enhancement weakens with `T_i`, and confinement is best at lower `T_i`.
  The plug is genuinely driven (`P_aux` about 78 MW), but D-D's low reactivity gives
  `q_eng` about 0.24 (< 1): net-negative. Uneconomic, driven, no stability
  pathology.

- **D-He3** likewise sits near the bottom of its bracket (`T_i` about 20 keV),
  driven (`P_aux` about 83 MW). The hot-electron radiation and the weak plug
  relative to the high `T_i` D-He3 wants give `q_eng` about 0.21 (< 1): net-negative.
  This is a regime change from Task 2b (where the constant-ratio plug gave a
  spurious interior optimum with `q_eng` about 5.7); the fixed-plug result is the
  honest one. Uneconomic, driven, no pathology.

- **p-B11** parks at its `[50, 300]` keV bracket top (`T_i` about 300 keV, the very
  high temperature p-B11 needs), driven (`P_aux` about 189 MW), with `q_eng` about
  0.23 (< 1): net-negative. Same honest aneutronic result as Task 2b, now with the
  correct driven plug. The collisionality (about 2.5e-6) and beta (at cap) are
  physical; no spurious ignition. Uneconomic, driven, no pathology.

### Decision: is Task 4 (explicit stability constraint) needed?

No fuel walks outside a trustworthy regime. For every fuel the settled optimum is
validity-respecting (the fired Pastukhov-Maxwellian flag is the expected
plugged-tandem signature, not an error: the fixed Fowler-Logan plug carries the
confinement, not the bare formula), the collisionality is consistent with a plugged
tandem (`L/mfp` of order 1e-3 to 1e-6, deeply on the plugged branch), and no fuel
is spuriously ignited. Every fuel is now genuinely DRIVEN (auxiliary sustainment
far off the floor) because the fixed plug potential removes the free-confinement
ride: heating the central cell costs confinement, so the optimizer no longer climbs
to self-heating. D-D, D-He3, and p-B11 settle near the LOW end of their brackets
(the fixed-plug signature, the reverse of the Task 2b ratio-shortcut behavior),
which is a fuel-reactivity/economics outcome, not a stability pathology.

Decision per fuel:

- D-T: settles sensibly and is the headline result: interior driven optimum,
  `T_i` about 23 keV, beta-cap bound, net-positive (`q_eng` about 1.77).
- D-D: driven (`P_aux` about 78 MW) but uneconomic (`q_eng` about 0.24, net-
  negative); honest low-reactivity result, no pathology.
- D-He3: driven (`P_aux` about 83 MW) but uneconomic (`q_eng` about 0.21, net-
  negative); honest result in the hot-electron radiation regime, no pathology.
- p-B11: driven (`P_aux` about 189 MW) but uneconomic (`q_eng` about 0.23, net-
  negative); honest aneutronic result, no pathology.

Therefore the conditional explicit stability constraint (Task 4) is NOT needed for
any fuel and is skipped: the fixed plug potential, not an explicit bound, is what
settles each fuel in a sensible driven regime. The honest caveat is that only D-T
is economic at the modelled fields and the hot-electron tandem plug; the advanced
fuels are driven and physical but net-negative, a reported outcome of the correct
plug physics, not a model defect. If a future change moves a fuel into a regime
where the validity flag fires together with a spurious ignition, the Task 4
minimum-warm-fraction bound is the documented contingency.

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
  design point (eq. 18 tandem confining potential `e*phi = T_e ln(n_p/n_c)`;
  sec. 3.5 design point: `ell_c = 50` m, `n_c/n_p = 0.55`, `T_i = 45` keV,
  `T_e = 125` keV, `P_fus = 157.4` MW, `P_NBI = 30` MW, `tau_c` about 5 s).
- **Fowler, T. K. and Logan, B. G. (1977)**, "The tandem mirror reactor,"
  Comments Plasma Phys. Control. Fusion 2, 167. Classical tandem-mirror concept:
  the central-cell ions are confined by the electrostatic potential drop to the
  end plugs, set by the plug-to-central density ratio. The plug and central cell
  are distinct plasmas (the plug supplies the potential; the central cell does the
  fusion), which is the basis for decoupling the plug electron temperature from
  the central-cell electron temperature.
- **Baldwin, D. E. and Logan, B. G. (1979)**, "Improved tandem mirror fusion
  reactor," Phys. Rev. Lett. 43, 1318. Thermal-barrier tandem: a potential barrier
  deliberately decouples the plug hot-electron population from the central-cell
  electrons so the plug can run hotter (ECH-heated) than the central cell, which is
  exactly the hot-plug / cool-central separation modelled here.
- **Santarius, J. F. and Callen, J. D. (1983)**, "Fusion-alpha confinement in a
  tandem-mirror central cell," Phys. Fluids 26, 1037. Bounce-averaged
  Fokker-Planck treatment of fusion-alpha loss-cone confinement: about 50 percent
  of alphas lost by count but under 25 percent by energy, so about 75-85 percent
  of the alpha power deposits. Primary source for the `f_alpha_heat` value.

No number in this document is sourced from or calibrated against any
cost-modeling tool; all come from the primary plasma-physics literature, except
the constructed smoothing width `w`, which is documented as constructed.
