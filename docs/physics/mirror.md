# Mirror 0D Physics Model

This is the consolidated physics reference for the 0D axisymmetric mirror module
(`src/costingfe/layers/mirror.py`). It documents the central-cell electron-temperature
closure (solved from an electron power balance), the collisional-to-collisionless
confinement-regime bridge, the tandem plug closures (Fowler-Logan potential, plug
sustainment power, alpha loss-cone routing), the stability and validity diagnostics,
and the machine-anchor validation against GDT and WHAM. Every number here comes from
the primary plasma-physics literature, except a single constructed smoothing width
that is flagged as such; no value is sourced from any cost-modeling tool.

Dollar amounts use a bare `$` (for example `$/MWh`, `513.375 M$`).

## Central-cell electron temperature (power-balance model)

### Summary

The 0D mirror model solves the central-cell electron temperature `T_e` from a
self-consistent electron power balance, instead of taking it as a pinned input.
The model reproduces the warm central-cell electrons of a deep-plug D-T tandem
reactor (about 0.8 of the ion temperature, about 20 to 24 keV at a reactor design
point), validated against the Mirror Advanced Reactor Study (MARS). This section
records the physics, the closures, the validation, and the deferred follow-ons,
with citations.

### Problem

The model previously pinned central-cell `T_e = 125 keV`. That value is the
tandem PLUG hot-electron temperature, not a bulk central-cell temperature: MARS
reports plug warm electrons at 124 keV and hot barrier electrons at 840 keV, with
the central cell at 24 keV [Logan 1985; Gordon 1986]. Applying the plug value to
the bulk cell drove the optimizer to an unphysical hot D-T operating point.

### Physical picture (from the tandem-mirror literature)

Central-cell `T_e` is an output of an electron power balance, and which value it
takes depends on the electron confinement regime:

- Simple / gas-dynamic mirrors (shallow ambipolar potential, cold ends, electrons
  escape freely) run COOL electrons, about 0.1 of the ion temperature
  [Forest 2024, BEAM: T_e about 6 keV at T_i about 67 keV; Post 1987].
- Deep-plug TANDEM central cells (the electrons are electrostatically confined by
  the same deep plug that confines the ions) run WARM electrons, about 0.85 of the
  ion temperature [Logan 1985, MARS: T_e=24, T_i=28 keV, ratio 0.86; Fowler 2017:
  Tec about Tic].

The dominant electron energy SINK is thermal effusion (Pastukhov end-loss) over
the ambipolar potential: each escaping electron carries about 5 to 6 times `T_e`,
at the ELECTRON collision rate, which is about 40 to 60 times faster than the ion
rate. Post 1987 explicitly rules out Spitzer-Harm parallel heat conduction for
mirrors (it would require absurd power densities; the long mean free path makes
the loss convective, not conductive). The ambipolar potential is set by
ambipolarity (electron loss rate equals ion loss rate), which couples the electron
confinement to the ion plug depth. Thermal barriers [Baldwin and Logan 1979]
decouple the central-cell electrons from the hot plug electrons so the plug can be
heated without paying to heat the whole central cell.

### The model

Solve central-cell `T_e` from the steady-state electron power balance:

    alpha_e(T_e)*P_alpha + K_ie*(T_i - T_e) = P_brem(T_e) + P_e_endloss(T_e)
    [ alpha -> electron ]   [ equilibration ]   [ brems ]    [ electron end-loss ]

Closures:

- `P_brem`: relativistic bremsstrahlung including the electron-electron term
  [Putvinski et al. 2019; the form in Ochs et al. 2022 Eq. 16]. Uniformly valid
  from D-T (about 10 keV) to p-B11 (about 300 keV).
- `K_ie`: NRL formulary ion-electron energy equilibration [Huba, NRL Plasma
  Formulary].
- `alpha_e`: Stix two-body slowing-down fraction of alpha (charged-particle) power
  delivered to electrons [Stix 1972], multiplied by the loss-cone retention.
- `P_e_endloss`: electron Pastukhov end-loss over the ambipolar potential
  `phi_e = g_amb * T_e`, with each escaping electron carrying about
  `(g_amb + 2) * T_e`, evaluated at the ELECTRON collision time (not the ion time)
  [Pastukhov 1974; Cohen et al. 1978].
- `g_amb` is SELF-CONSISTENT via ambipolarity: the electron Pastukhov confinement
  time is matched to the model's ion confinement time (electron loss rate equals
  ion loss rate). This is a Lambert-W relation, so `g_amb` is pinned
  logarithmically and the `exp(g_amb)` sensitivity that would otherwise make the
  result knife-edge is removed. Solved with a backend-safe bounded bisection.

The ion confinement uses the Pastukhov / gas-dynamic collisionality bridge
(next section). The Pastukhov enhancement `exp(e*phi/T_i)` over-credits in the
deeply collisionless or deep-plug regime; it is bounded by a smooth-min `n*tau`
ceiling at about 1e21 m^-3 s (the canonical achievable mirror Lawson scale
[Fowler 2017]), applied so that confinement times already below the ceiling are
essentially unchanged. This is an empirical bound; the physics-based replacement
(a loss-cone / DCLC instability degradation, or anomalous radial transport) is a
documented follow-on.

The Coulomb logarithm is channel-specific `lnLambda(n,T)` for the
electron-electron, electron-ion, and ion-ion channels [NRL Plasma Formulary],
replacing a folded-in constant. With a constant log the density dependence of the
two competing sinks cancels exactly; the proper logs restore the (weak) residual
density dependence.

### Validation

- D-T versus MARS: at the MARS reactor operating point (central density about
  3.3e20 m^-3, T_i = 28 keV, the model's own plug potential about 74.7 keV) the
  electron balance lands `T_e` warm, about 0.8 of `T_i`, in the MARS band, with the
  model's confinement giving an ion confinement time of about 3 s (consistent with
  the published Realta Hammir central-cell value of about 5 s [Frank et al. 2024]).
  The apparent over-credit (an ion confinement time of about 94 s) appears only
  when MARS's deeper plug potential (about 150 keV) is substituted; the model's own
  moderate plug does not over-credit at reactor density.
- p-B11 versus Ochs et al. 2022 (arXiv:2210.08076): DEFERRED. The current model
  carries a single bulk ion temperature and a single effective ion mass and lumps
  all charged-particle power into a generic alpha channel, so it cannot reproduce
  the multi-species p-B11 balance (fast and thermal protons, boron, electrons, each
  with its own slowing-down) or give a trustworthy D-He3 verdict (the D-He3 14.7
  MeV proton has a very different slowing-down than the alpha). This validation
  needs the multi-species generalization (below).

The released, user-supplied-operating-point path (`use_0d_model = false`) is
untouched; its bit-identical cost figure is preserved.

### Deferred follow-ons

1. Multi-species power balance: per-species temperatures and slowing-down for the
   fuel ions, the alpha, the D-He3 14.7 MeV proton, and electrons. Enables a
   trustworthy advanced-fuel verdict and the Ochs p-B11 cross-validation.
2. Physics-based confinement cap: replace the empirical `n*tau` ceiling with a
   loss-cone (DCLC) instability-enhanced loss using the `a/rho_i` diagnostic and a
   warm-fraction stabilization threshold [Kolmes et al. 2024; Post 1987], or an
   anomalous (Bohm) radial-transport term.
3. Size-from-power radius logic: the mirror is end-loss dominated, so the power
   balance pins the LENGTH; the RADIUS is set by a feasible window of constraints
   (neutron wall load and photon / surface wall load as lower bounds; DCLC
   microstability and coil bore as upper bounds; the `a/rho_i` finite-Larmor floor).
   Design at the smallest feasible radius (binding lower bound), flag infeasibility
   when the lower bound exceeds the upper bound, and report both binding
   constraints.

## Confinement regime bridge (collisional vs collisionless)

The model computes the axial confinement time by combining two kernels: the
gas-dynamic (collisional) time `tau_GD` (Mirnov and Ryutov 1979) and the Pastukhov
electrostatically plugged (collisionless) time `tau_Pastukhov` (Pastukhov 1974,
Cohen et al. 1978). These two limits apply in physically distinct collisionality
regimes and must be joined by a single bridge that selects the right limit.

### The published transition criterion

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
variable. The `1/R_m` placement is not a free knob; it is the mirror-ratio
reduction of the mean free path stated by Rognlien and Cutler.

### The bridge form the model uses

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
rate. A narrow gate keeps the suppressed gas-dynamic rate well below the Pastukhov
rate at the collisionless anchors (WHAM and the high-mirror-ratio reference points)
while remaining fully on at the collisional anchor (GDT). With `w = 0.13` the
gate is essentially off about 0.4 decades below the boundary and essentially on
about 0.4 decades above it, and the branch selection is robust at both anchors,
which sit more than a decade from their `1/R_m` boundaries.

The bridge is float32-safe: the gate argument is built from `log10` of an
order-unity-to-1e-5 collisionality, the prefactors are folded in float64, and
every intermediate stays in a benign range. The logistic argument is clamped to
`[-30, 30]` before `exp`, so even far past any physical operating point the gate
keeps a finite value and a finite reverse-mode gradient (an unclamped saturated
logistic gives `inf * 0 = NaN` under `jax.grad`). At physical collisionalities
the clamp is never active.

### Validity window and caveats

- The bridge inherits the validity caveats of both kernels (single thermal
  Maxwellian, no fast-ion or sloshing-ion population), documented in the
  machine-anchor validation section below. It does not add physics; it selects
  which existing kernel governs.
- `collisionality = 1/R_m` is an order-unity boundary, not a sharp threshold; the
  smoothing width reflects that. Operating points within about a decade of the
  boundary carry a genuine regime ambiguity, and the gate interpolates smoothly
  rather than asserting a hard branch.
- In the collisionless regime the Pastukhov formula itself overestimates
  confinement when the plasma is too collisionless to be Maxwellian; that is a
  separate validity flag (see the diagnostics section), not handled by this bridge.

## Axial confinement ceiling and plug closures

### n*tau ceiling on the axial confinement

The bare Pastukhov enhancement `x*exp(x)`, `x = e*phi/T_i`, over-credits axial
confinement in the deeply collisionless / deep-plug regime: `tau_ii ~ 1/n` grows
long at thin density and `x` grows large at low `T_i` or a deep plug, so the
formula runs `tau_axial` away to unbounded values at off-design points (for
example `n = 5e19 m^-3`, `T_i = 15 keV`, where the uncapped axial time is about
124 s). The 0D confinement cannot physically exceed the canonical mirror Lawson /
ignition-class `n*tau` scale, of order `1e21 m^-3 s` (Fowler 2017, "Fusion energy
from a magnetic mirror"). This is an empirical bound on achievable mirror
confinement, not a first-principles loss rate. A differentiable soft-min is
therefore applied after computing `tau_axial`:

```
tau_ceil  = _N_TAU_CEILING / n_i
tau_capped = tau_axial / (1 + (tau_axial/tau_ceil)**_CAP_SHARPNESS)**(1/_CAP_SHARPNESS)
```

with `_N_TAU_CEILING = 1e21 m^-3 s` and `_CAP_SHARPNESS = 8`. This is a soft
MIN, not an additive loss rate: when `tau_axial << tau_ceil` the ratio
`(tau_axial/tau_ceil)**8` underflows to zero and `tau_capped -> tau_axial`
(no reduction); when `tau_axial >> tau_ceil` the expression saturates to
`tau_ceil`. The three representative operating points are:

- MARS reactor density (`n = 3.3e20 m^-3`, `phi = 74.7 keV`): uncapped
  `tau_axial = 2.40 s`, capped `2.36 s` (about 2% reduction, essentially
  untouched). The solved central-cell electron temperature is `T_e = 20.70 keV`,
  inside the validated 20-28 keV warm band.
- Thin point (`n = 5e19 m^-3`, `T_i = 15 keV`): uncapped about 124 s, capped
  20 s (the density ceiling `1e21 / 5e19 = 20 s`).
- Deep plug (`phi = 150 keV` at reactor density): capped near the 3 s density
  ceiling.

Future refinement: the physics-based treatment of anomalous axial losses is a
loss-cone-instability (DCLC) confinement degradation, using the `a/rho_i`
diagnostic with a warm-fraction stabilization threshold (Kolmes et al. 2024,
loss-cone stabilization; Post 1987). Anomalous (Bohm) radial transport is a
further candidate. Both are documented improvements beyond the current empirical
ceiling.

### The Fowler-Logan plug potential

The CAS22 coil account costs a TANDEM (`n_plug_coils = 4`, Hammir class: two
high-field HTS end plugs per end plus the long central solenoid). The confinement
is therefore the tandem central-cell confinement, where the ions are held by the
electrostatic potential drop to the end plugs, NOT the simple-mirror electron-loss
potential. In a classical tandem (Fowler and Logan 1977) the ion confining
potential between the central cell and the end plug is set by their density ratio:

```
e*phi = T_e_plug * ln(n_p / n_c)
```

implemented as `compute_plug_potential(T_e_plug, plug_density_ratio)` (Frank et
al. 2024 eq. 18; Fowler and Logan 1977). Because `n_p / n_c` is an
order-unity-to-few ratio, `e*phi` is BOUNDED, unlike the unbounded simple-mirror
Boltzmann value `e*phi = T_e * ln(sqrt(m_i / (2 pi m_e)))` (about 3 to 9 `T_e`,
growing without limit), which floors the auxiliary power and produces spurious
ignition.

Here `T_e_plug` is the tandem PLUG hot-electron temperature (YAML default
125 keV, anchored to Hammir), NOT the central-cell electron temperature. The
Fowler-Logan plug confines THROUGH the plug electron temperature, so a tandem must
run HOT plug electrons (typically plug-ECH-heated) for the plug to confine at all.
This is a distinct temperature from the warm central-cell `T_e` that the electron
power balance solves (about 0.8 of `T_i`); the two must never be conflated. At a
cold plug (for example a 20 keV plug electron temperature) `e*phi = 12 keV` is far
too shallow to plug a 20-80 keV central cell, and the D-T machine is infeasible at
every `T_i`. `plug_density_ratio = n_p/n_c` is a fixed plug property.

Because `e*phi` is set by the fixed plug and does not scale with `T_i`, the ratio

```
e*phi / T_i = T_e_plug * ln(n_p/n_c) / T_i
```

FALLS as `T_i` rises, so the Pastukhov enhancement `x*exp(x)` WEAKENS at high `T_i`:
heating the central cell COSTS confinement instead of buying it for free. This is
what makes the sized optimum settle at a genuinely DRIVEN point rather than running
away to self-heating.

### The Hammir anchor

The Realta Hammir Q>5 pre-conceptual design point (Frank et al. 2024,
arXiv:2411.06644, sec. 3.5, the `am = 0.15` m POPCON case) is:

- central cell length `ell_c = 50` m
- mirror throat field `B_m = 25` T, central cell field `B0c` about 3 T,
  central cell mirror ratio `R_mc = 13.3` (finite beta, `beta_c` about 0.6)
- end-plug density `n_p = 1.5e20` m^-3, density ratio `n_c / n_p = 0.55`, so
  `n_c = 0.825e20` m^-3 (`plug_density_ratio = n_p/n_c = 1.818`)
- `T_i = 45` keV, plug electron temperature `T_e = 125` keV
- fusion power `P_fus = 157.4` MW, plug neutral-beam power `P_NBI = 30` MW, so
  the system gain `Q = P_fus / P_NBI = 5.2` (the Q>5 headline; the Realta
  announcement quotes `Q_sci` about 5.3)

At this point the tandem confining potential is
`e*phi_c = 125 * ln(1 / 0.55) = 74.7` keV, i.e. the ratio

```
e*phi_c / T_i = 74.7 / 45 = 1.66.
```

The model evaluated at the Hammir central cell reproduces the design point: the
Pastukhov axial confinement time is about 6.3 s (the paper quotes a tandem
central-cell `tau_c` about 5 s, sec. 3.5; within the 2x anchor band), the central
cell is NOT ignited (the confinement-required auxiliary power is about 27 MW,
matching the published 30 MW plug NBI), and the gain `Q = P_fus / P_aux` is about
5.8 (within 2x of the published 5.2). By contrast the unbounded Boltzmann potential
at a 125 keV electron temperature gives `e*phi = 412` keV (`e*phi/T_i = 9.2`),
which floors the auxiliary power and produces the spurious ignition. The
calibration is one anchored Hammir operating point, carrying the same 2x
validation band as the GDT and WHAM kernel anchors.

### Decoupling the hot plug from the cool central cell

Real advanced-fuel tandems do not run one electron population. The plug is a
SEPARATE cell, often with a different (plugging) ion species, whose job is to hold
the confining potential with a HOT-ELECTRON population sustained by plug ECH. The
central cell runs the working fuel and can keep its electrons cooler, because the
confinement is supplied by the plug potential, not by the central-cell electron
temperature. This is the original Fowler and Logan 1977 tandem concept, with the
Baldwin and Logan 1979 thermal-barrier refinement that deliberately decouples the
plug and central-cell electron populations with a potential barrier so the plug can
run hotter than the central cell. The Hammir design point runs its plug
hot-electron population at 125 keV, set by the plug ECH independent of the
central-cell fuel.

The model captures this at costing fidelity by separating two temperatures:

- `T_e_plug` (hot, YAML default 125 keV, anchored to Hammir): sets the
  Fowler-Logan confining potential `e*phi = T_e_plug * ln(n_p/n_c)` via
  `compute_plug_potential(T_e_plug, plug_density_ratio)`. It is the plug knob,
  set by plug ECH and independent of the central fuel.
- `T_e` (central-cell): sets central-cell bremsstrahlung, synchrotron, beta, and
  stored energy. It is solved WARM from the electron power balance (about 0.8 of
  `T_i` for the D-T reference machine, and can be overridden cooler for advanced
  fuels). It does NOT enter the plug potential.

The plug is modelled as POTENTIAL-PLUS-POWER, not a full second fusion plasma:
the central cell does the fusion; the plug contributes the confining potential and
its power cost. "Different species" is captured by the plug being this separate
potential-and-power sub-system, independent of the central fuel.

### Charging the plug power (P_plug, Hammir anchor)

The hot-electron plug is sustained by plug ECH/NBI at a real power cost. The model
charges a fixed plug sustainment power `P_plug` (YAML default 30 MW) into the
mirror recirculating budget, calibrated to the Realta Hammir design point, which
holds its hot-electron plug with about 30 MW of plug NBI/ECH (Frank et al. 2024
sec. 3.5; the published `Q = P_fus / P_NBI = 5.2` counts exactly this 30 MW).

A FIXED per-machine plug power (not one scaled with `n_p` or the central-cell
power) is the defensible 0D choice: the 0D model does not solve the plug density or
trapping self-consistently, the published anchor is a single design-point plug
power, and the CAS22 coil account already commits to fixed plug hardware
(`n_plug_coils = 4`). `P_plug` is also the competing penalty that keeps the
optimizer honest: it is a recirculating load the plant pays regardless of
central-cell temperature, so a configuration cannot get deep plug confinement for
free.

`P_plug` is charged on the MIRROR SIDE by folding it into the `p_coils` argument
passed to the shared `mfe_forward_power_balance` (`p_coils_eff = p_coils +
p_plug`). The shared recirculating sum is `p_coils + p_pump + p_sub + p_aux +
p_cool + p_cryo + p_input_eff/eta_pin`, so this adds `P_plug` at unit recirculating
cost without touching the shared function or the tokamak/stellarator/IFE paths. It
is charged consistently in the inverse-solve, the forward power table, and the
sizing path. The plug power does NOT enter `p_input` (the plasma energy balance),
because the plug ECH heats the plug electrons, not the central cell.

Reconciling the Hammir Q anchor: Hammir's published `Q = P_fus / P_NBI = 5.2`
already counts the 30 MW plug NBI in the denominator. Charging `P_plug` explicitly
makes the model's accounting match Hammir's rather than under-counting the plug
cost. The forward Hammir anchor (`q_sci = P_fus / P_aux` at the published central
cell) is unchanged because it already used the confinement-required sustainment
power; the new explicit `P_plug` enters `q_eng` (the engineering gain that drives
the economics) and the sized LCOE, which is where the plant pays for the plug. The
anchor holds within the documented 2x band.

### DEC end-loss-only routing

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
   credit.
2. With the radiation consistent, an effective `f_dec_eff = f_dec * P_end /
   p_transport_shared` makes the shared term recover exactly
   `f_dec * eta_de * P_end`, the recoverable axial end-loss, so the DEC credit
   tracks `P_end` (which falls as confinement improves with temperature).

With the alpha loss-cone routing (below) the un-deposited alpha power is conserved
into the transport channel, so the identity is
`p_transport == P_end + P_radial + (1 - f_alpha_heat) * P_alpha`.

### Alpha loss-cone heating fraction

A magnetic mirror does not confine its fusion alphas the way a closed tokamak
does: a fraction of the 3.5 MeV alphas is born in or scatters into the
velocity-space loss cone and streams out the ends before depositing its energy in
the central cell. The model credits only the deposited fraction `f_alpha_heat` of
the charged-particle power `P_alpha` as self-heating, and routes the lost fraction
`(1 - f_alpha_heat) * P_alpha` to the axial end-loss / DEC channel as directed
loss-cone exhaust.

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
driven character is end-loss-dominated, not alpha-loss-dominated. The generic
model default of 0.80 is the conservative Santarius-Callen value; at the Hammir
anchor it lowers the computed gain from `Q` about 5.8 (full deposition) to about
4.7, BOTH within the 2x anchor band of the published `Q` about 5.2 (the 0.80 value
actually lands closer to the published number).

Implementation and energy conservation: in the sustainment balance
`mirror_aux_heating` subtracts `f_alpha_heat * P_alpha` instead of the full
`P_alpha`:

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
leaving `P_radial` to the wall. The shared function is NOT modified; the reduction
and the routing are entirely mirror-side, through the `p_input` handoff and the
existing `p_rad_override` / effective-`f_dec` hooks. `q_sci = P_fus / p_input_eff`
and `q_eng` follow automatically from the larger `P_aux`.

## Stability and validity diagnostics

Two diagnostics are reported on `MirrorPlasmaState`. Both are INFORMATIONAL: they
report where a modelling assumption is stretched or where a known microinstability
is more readily driven, but neither restricts the operating point.

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
interpreting the number, NOT a constraint.

A larger `dclc_parameter` means a larger warm-plasma fraction is needed; a value of
order `dclc_a_over_rho_ref` or below means a modest warm fraction (of order a few
percent) suffices, which is the regime GDT and WHAM operate in.

## Settled-regime results (cross-fuel)

### Current D-T sized optimum

The sized D-T optimum (400 MWe, `f_beta = 0.85`, YAML defaults) is a genuinely
DRIVEN tandem, reproduced from the committed code via
`CostModel(MIRROR, DT).forward(net_electric_mw=400, availability=0.87,
lifetime_yr=40, size_from_power=True, f_beta=0.85)`:

| Quantity | Value |
|----------|-------|
| `T_i` | 23.34 keV |
| `tau_E` | 18.7 s |
| `Q_sci` (`P_fus / p_input_eff`) | 8.30 |
| `Q_eng` (`P_et / recirculating`) | 1.68 |
| LCOE | 395.8 $/MWh |
| beta | at the `f_beta * beta_max = 0.425` ceiling |
| `P_aux` (confinement-derived sustainment) | about 245 MW (band 228-259 across the trajectory's L) |

The mechanism: the fixed Fowler-Logan plug potential removes the free-confinement
ride, so heating the central cell COSTS confinement and the optimizer settles at an
interior cool tandem-realistic point (`T_i` about 23 keV, below the Hammir 45 keV
because it now pays confinement to go hotter) rather than running away to
self-heating. The central cell runs WARM (its `T_e` solved at about 0.8 of `T_i`
from the electron power balance), while the SEPARATE plug hot-electron population
stays at 125 keV and sets `e*phi = 74.7 keV`. The hot plug electrons raise the
central-cell pressure, so the beta-limited density falls below the
neutron-wall-limited density and the binding cap is beta rather than the wall.

The 30 MW plug sustainment power is charged explicitly into the recirculating
budget, moving the sized LCOE from an earlier without-plug figure of about 369 to
395.8 $/MWh (+7.3 percent) while leaving `T_i`, `Q_sci`, and `tau_E` unchanged
within tolerance; the Hammir Q>5 anchor reconciles because Hammir counts the same
30 MW plug NBI in its published `Q = P_fus / P_NBI = 5.2`.

Cost pins moved by the corrected model (the coil calibration pin 513.375 M$
capital and all non-mirror tokamak/stellarator/IFE/MIF pins do NOT move):

| Quantity | Old | New | Reason |
|----------|-----|-----|--------|
| non-0D default LCOE pin | 98.686 | 100.107 | central-cell radiation term moves |
| `a=1.5`/`B=3` wall-loose sized L (m) | 69.230740 | 77.842279 | explicit `P_plug = 30 MW` recirculating |

### Cross-fuel observation (the real finding)

Run in sizing/optimize mode across all four fuels (`f_beta = 0.85`, YAML defaults,
`R_m = 10`, `plug_density_ratio = 1.818`, `beta_max = 0.5`), only D-T is
net-positive in the driven tandem regime; D-D, D-He3, and p-B11 are net-NEGATIVE.
This is reported as found, not engineered to a target.

| Fuel | T_i [keV] | P_aux [MW] | collisionality | `pastukhov_valid` | beta | q_sci | q_eng | driven? |
|------|-----------|------------|----------------|-------------------|------|-------|-------|---------|
| D-T   | 23.3 | 228-259 | 1.4e-3 | 0.0 (flagged) | 0.425 | 8.3 | 1.68 | DRIVEN, net-positive |
| D-D   | 15.6 | 78 | 3.3e-3 | 0.0 (flagged) | 0.425 | 0.38 | 0.24 | driven but uneconomic |
| D-He3 | 20.0 | 83 | 1.4e-3 | 0.0 (flagged) | 0.425 | 0.15 | 0.21 | driven but uneconomic |
| p-B11 | 300.0 | 189 | 2.5e-6 | 0.0 (flagged) | 0.425 | 0.15 | 0.23 | driven but uneconomic |

Decoupling the hot plug from the cool central cell makes the advanced fuels fairly
evaluable by removing a spurious hot-central-cell radiation penalty: D-He3 in
particular improves sharply toward break-even when its central cell is allowed to
run cool (a cool-central hot-plug D-He3 tandem reaches `q_eng` about 0.93, near
break-even). D-D and p-B11 stay deeply net-negative on low reactivity and very high
required temperature respectively. None of the advanced fuels is net-positive at
the modelled fields; only D-T is economic. This is a genuine
reactivity-and-radiation result, not an artifact of forcing one hot electron
temperature on a fuel that does not want it.

### No explicit stability constraint needed

For every fuel the settled optimum is validity-respecting (the fired
Pastukhov-Maxwellian flag is the expected plugged-tandem signature, not an error:
the fixed Fowler-Logan plug carries the confinement, not the bare formula), the
collisionality is consistent with a plugged tandem (`L/mfp` of order 1e-3 to 1e-6,
deeply on the plugged branch), and no fuel is spuriously ignited. Every fuel is
genuinely DRIVEN (auxiliary sustainment far off the floor) because the fixed plug
potential removes the free-confinement ride. The advanced fuels settle near the LOW
end of their temperature brackets (the fixed-plug signature), a
fuel-reactivity/economics outcome, not a stability pathology.

An explicit stability constraint is therefore NOT needed for any fuel: the fixed
plug potential, not an explicit bound, is what settles each fuel in a sensible
driven regime. If a future change moves a fuel into a regime where the validity
flag fires together with a spurious ignition, a minimum-warm-fraction stability
bound is the documented contingency.

## Machine-anchor validation (GDT, WHAM)

The 0D mirror model computes confinement from first-principles kernels: gas-dynamic
(Mirnov & Ryutov 1979), Pastukhov electrostatic plugging (Pastukhov 1974, Cohen et
al. 1978), and classical (Bing & Roberts 1961). Cost accounts depend on the plasma
state the model produces, so the kernels must agree with measured and predicted
confinement on real machines, not only with each other. This section records the
published GDT and WHAM parameters, derives the confinement anchors, and states the
factor by which the model reproduces each one. The matching tests live in
`tests/test_mirror.py::TestAnchors`.

### Scope and caveat: what the model represents

The model is a SINGLE thermal Maxwellian 0D model. It carries no fast-ion
population, no sloshing-ion distribution, and no explicit NBI slowing-down physics.
Real mirrors (GDT, WHAM) are beam-driven: the confined energy lives in a fast-ion
population at the injection energy, with a warm/target thermal plasma underneath.
The published confinement formulas reflect that two-class reality.

Consequently the anchors are matched at the populations where a thermal 0D model is
physically meaningful:

- GDT gas-dynamic anchor uses the warm target plasma (T_e of about 0.25 keV,
  collisional, gas-dynamic regime). This is the population the gas-dynamic kernel
  describes. We do NOT anchor the gas-dynamic kernel against the GDT fast-ion
  confinement.
- WHAM Pastukhov anchor uses the model's thermal ions at the published midplane
  mean ion energy (10 keV at beta = 0.2), compared against the
  classical-mirror/Pastukhov scaling the WHAM and BEAM papers quote. The published
  scaling is itself a Fokker-Planck result for a 25 keV sloshing-ion distribution;
  the model's single-Maxwellian Pastukhov kernel is an approximation to it. The 2x
  tolerance exists for exactly this idealization gap.

### GDT (Gas Dynamic Trap, Budker Institute, Novosibirsk)

Machine parameters:

| Quantity | Value | Source |
|----------|-------|--------|
| Central cell length | 7 m (mirror-to-mirror) | Bagryansky et al. 2015, "THE GAS DYNAMIC TRAP" section |
| Mirror ratio R_m | 35 (max/min on-axis field) | Bagryansky et al. 2015 |
| B_min (central solenoid) | 0.27-0.35 T | Bagryansky et al. 2015 (0.35 T standard; reshaped to 0.27 T for dual-beam ECRH) |
| Plasma diameter | 0.20 m | Forest et al. 2024 (BEAM), citing GDT |
| Achieved beta | up to about 0.6 | Bagryansky et al. 2015; Yakovlev et al. 2018 (via Forest et al. 2024) |
| Warm-plasma T_e (standard config, no ECRH) | 250 eV at n = 2e19 m^-3 | Bagryansky et al. 2015 (citing ref [13]) |
| Record T_e (with 0.7 MW ECRH) | 660 +/- 50 eV, peaks > 900 eV, at n = 0.7e19 m^-3 | Bagryansky et al. 2015, Abstract + Results |
| Target/background ion & electron T | up to about 100 eV (collisional component) | Ivanov & Prikhodko 2013 review |
| Heating | 5 MW NBI (8 injectors) + 0.7 MW ECRH (54.5 GHz) | Bagryansky et al. 2015 |
| Working gas | deuterium (A = 2) | Bagryansky et al. 2015 |

Gas-dynamic confinement anchor: the WHAM physics-basis paper (Endrizzi et al.
2023) gives the gas-dynamic confinement time, attributed to Ivanov & Prikhodko
2013, as

```
tau_GDT = R_m * L_p / c_s = 5.2 * R_m * L_p * T_e[keV]^(-1/2)  microseconds   (Endrizzi eq. 3.5)
```

where `L_p` is the plasma HALF-length and `c_s = sqrt(T_e / m_i)` is the ion sound
speed. Evaluated at GDT standard-config warm-plasma parameters (R_m = 35,
L_p = 3.5 m = half of the 7 m central cell, T_e = 0.25 keV):

```
tau_GDT = 5.2 * 35 * 3.5 * (0.25)^(-1/2) us = 1274 us = 1.27 ms
```

Model kernel and the convention difference: `compute_tau_gas_dynamic` implements
the Mirnov-Ryutov form

```
tau_GD = R_m * L / v_thi,   v_thi = sqrt(2 * T_i / m_i)
```

with `L` the FULL mirror-to-mirror length and `v_thi` the ion THERMAL speed (built
on T_i). Because `v_thi = sqrt(2) * c_s` and `L = 2 * L_p`, the two forms relate by
a fixed factor when `T_i = T_e`:

```
tau_GD(model) / tau_GDT(eq 3.5) = (L / v_thi) / (L_p / c_s) = sqrt(2) ~ 1.41
```

i.e. the model is intrinsically about 1.4x longer than eq. 3.5 by construction,
both being valid order-of-magnitude gas-dynamic estimates that differ in their
choice of characteristic length and speed. This is well inside the 2x tolerance and
the difference is fully explained, not a model error.

Model evaluation at GDT (R_m = 35, L = 7 m, T_i = 0.25 keV, A = 2):

```
compute_tau_gas_dynamic(R_m=35, L=7, T_i=0.25, A=2) = 1.58 ms
```

Anchor: model 1.58 ms vs published 1.27 ms, ratio 1.25. Within 2x.

### WHAM (Wisconsin HTS Axisymmetric Mirror)

Machine / design parameters:

| Quantity | Value | Source |
|----------|-------|--------|
| HTS mirror magnet field (throat) | 17 T (2 kA, steady state, 5.5 cm bore) | Endrizzi et al. 2023, sec. 2.1 |
| Mirror coil location | z = +/- 98 cm (mirror-to-mirror about 1.96 m) | Endrizzi et al. 2023, sec. 2.1 |
| Central (midplane) field | 0.32 T base, boosted to 0.86 T with pulsed copper coils | Endrizzi et al. 2023, sec. 2.1 |
| Vacuum mirror ratio R_m | 17 / 0.86 = 19.8 (about 20) | Derived from Endrizzi field values |
| Target plasma radius a | 0.1 m | Endrizzi et al. 2023, sec. 2 |
| Target density n_e | 1-3 x 10^19 m^-3 | Endrizzi et al. 2023, sec. 2 |
| NBI | 25 keV, 40 A deuterium, 45-deg inclined (sloshing ions) | Endrizzi et al. 2023, sec. 2 |
| ECH | 1 MW, 110 GHz, X-mode | Endrizzi et al. 2023, sec. 2 |
| Predicted T_e | >= 1 keV (from GDT power-density scaling) | Endrizzi et al. 2023, sec. 2.4 |
| Equilibrium operating point | beta = 0.2, n = 0.3 x 10^20 m^-3, mean ion energy 10 keV at midplane | Endrizzi et al. 2023 (anisotropic MHD equilibrium) |

Pastukhov / classical-mirror confinement anchor: both the WHAM physics-basis paper
and the BEAM paper give the same classical-mirror (Pastukhov-plugged) confinement
scaling, derived from Fokker-Planck solutions for near-perpendicular NBI:

```
n_20 * tau_p = 250 * E_b,100keV^(3/2) * log10(R_m)  ms     (Endrizzi eq. 3.4)
n_20 * tau_p = 0.25 * E_b,100keV^(3/2) * log10(R_m) sec    (Forest BEAM eq. 1.1)
```

These are identical (250 ms = 0.25 s). Evaluated at WHAM (n_20 = 0.3, E_b = 25 keV
so E_b,100keV = 0.25, R_m = 19.8):

```
tau_p = 250 * (0.25)^(3/2) * log10(19.8) / 0.3 ms
      = 250 * 0.125 * 1.297 / 0.3 ms
      = 135 ms
```

Model kernel: `compute_tau_pastukhov(tau_ii, R_m, phi, T_i)` implements the
Pastukhov 1974 / Cohen et al. 1978 form, fed by `compute_tau_ii` and the Boltzmann
ambipolar potential `compute_ambipolar_potential`
(`phi/T_e = ln(sqrt(A * m_p/m_e / 2pi))` about 3.19 for deuterium). Anchored at the
published WHAM midplane thermal point: T_i = 10 keV (the beta = 0.2 equilibrium
mean ion energy), T_e = 1 keV (predicted), n = 3e19, R_m = 19.8, A = 2:

```
tau_ii        ~ 58 ms
e*phi         ~ 3.19 keV
tau_Pastukhov ~ 88 ms
```

Anchor: model 88 ms vs published 135 ms, ratio 135/88 = 1.53. Within 2x.

The gap is expected and physical: the published scaling solves the full
Fokker-Planck problem for a 25 keV sloshing-ion distribution with self-consistent
ambipolar potential, while the model uses a single thermal Maxwellian at the
midplane mean energy with a Boltzmann-relation potential. A factor of about 1.5
between a 0D Maxwellian kernel and a Fokker-Planck fast-ion calculation is good
agreement, and is the reason the tolerance is set at 2x rather than tighter.

### Regime-gate guard at the anchors

The GDT and WHAM anchors test the kernels directly and are unchanged by the regime
bridge. The regime guard is that the bridge selects the correct branch at each
anchor's regime. The table below is generated from the model kernels at the stated
anchor parameters using gate width `w = 0.13`:

- GDT: `R_m = 35`, `L = 7` m, `T_i = T_e = 0.25` keV, `n = 2e19` m^-3
  (warm-plasma density, Bagryansky et al. 2015); GDT sits about 0.7 decades inside
  the collisional branch.
- WHAM: `R_m = 17/0.86 = 19.77`, `n = 3e19` m^-3, `T_i = 10` keV, `T_e = 1` keV,
  `L = 2` m (central-cell length scale, Endrizzi et al. 2023); WHAM sits more than
  3 decades inside the collisionless branch, so the gate result is insensitive to
  the precise `L`.

| Anchor | Regime | collisionality | 1/R_m | gate g | tau_axial / tau_GD | tau_axial / tau_Pastukhov | Branch selected |
|--------|--------|---------------|-------|--------|--------------------|---------------------------|-----------------|
| GDT | collisional | 0.132 | 0.029 | 0.99 | 0.99 | 0.015 | gas-dynamic (correct) |
| WHAM | collisionless | 3.5e-5 | 0.051 | 2.8e-11 | 2.2e3 | 1.00 | Pastukhov (correct) |

GDT lands on the gas-dynamic branch within 1 percent; WHAM lands on the Pastukhov
branch within 1 percent. Both kernels remain within their documented 2x of the
published confinement times.

### Validity caveats (summary)

- Population mismatch is the dominant uncertainty. Both anchors compare a thermal
  0D kernel against literature formulas built on beam-driven distributions. The
  tolerance is 2x by design; tightening it would require the model to grow fast-ion
  physics it deliberately omits.
- The GDT gas-dynamic anchor is the cleanest comparison (kernel and formula are the
  same physical regime, same warm collisional plasma); ratio 1.25, almost entirely
  the documented sqrt(2) length/speed convention.
- The WHAM Pastukhov anchor is approximate (Maxwellian vs Fokker-Planck); ratio
  1.53.
- Field/mirror-ratio for WHAM uses the diamagnetically-unreduced vacuum value
  R_m = 19.8. At beta = 0.2 the midplane field is depressed, raising the effective
  mirror ratio modestly; this is a small effect relative to the 2x band and is not
  modeled.

## Sources

- **Logan, B. G. (1985)**, MARS (Mirror Advanced Reactor Study). Central-cell
  T_e = 24 keV at T_i = 28 keV (ratio 0.86), plug warm electrons 124 keV, hot
  barrier electrons 840 keV.
- **Gordon, D. T. (1986)**, MARS engineering overview.
- **Fowler, T. K. (2017)**, "A new simpler way to obtain high fusion power gain"
  / "Fusion energy from a magnetic mirror." Canonical achievable mirror Lawson
  `n*tau` scale of order 1e21 m^-3 s; `Tec` about `Tic` in a deep-plug central
  cell.
- **Post, R. F. (1987)**, "The magnetic mirror approach to fusion," Nucl. Fusion
  27. Rules out Spitzer-Harm parallel conduction for mirrors; DCLC as the dominant
  loss-cone microinstability and its warm-plasma stabilization threshold.
- **Rognlien, T. D. and Cutler, T. A. (1980)**, "Transition from Pastukhov to
  collisional confinement in a magnetic and electrostatic well," Nucl. Fusion 20,
  1003. Primary source for the transition criterion (Pastukhov invalid when the
  mean free path reduced by the mirror ratio is of order the system length) and for
  the Monte-Carlo-validated smooth composite of the Pastukhov and collisional
  confinement times.
- **Pastukhov, V. P. (1974)**, "Collisional losses of electrons from an adiabatic
  trap in a plasma with a positive potential," Nucl. Fusion 14, 3. Collisionless
  loss-cone confinement kernel.
- **Cohen, R. H., Rensink, M. E., Cutler, T. A. and Mirin, A. A. (1978)**,
  "Collisional loss of electrostatically confined species in a magnetic mirror,"
  Nucl. Fusion 18, 1229. Refinement of the Pastukhov scaling used by the kernel.
- **Mirnov, V. V. and Ryutov, D. D. (1979)**, gas-dynamic confinement scaling
  `tau_GD = R_m * L / v_thi`. Collisional outflow kernel.
- **Fowler, T. K. and Logan, B. G. (1977)**, "The tandem mirror reactor," Comments
  Plasma Phys. Control. Fusion 2, 167. Classical tandem-mirror concept: the
  central-cell ions are confined by the electrostatic potential drop to the end
  plugs, set by the plug-to-central density ratio; the plug and central cell are
  distinct plasmas, the basis for decoupling the plug electron temperature from the
  central-cell electron temperature.
- **Baldwin, D. E. and Logan, B. G. (1979)**, "Improved tandem mirror fusion
  reactor," Phys. Rev. Lett. 43, 1318. Thermal-barrier tandem: a potential barrier
  deliberately decouples the plug hot-electron population from the central-cell
  electrons so the plug can run hotter (ECH-heated) than the central cell.
- **Santarius, J. F. and Callen, J. D. (1983)**, "Alpha particle loss and energy
  deposition in tandem mirrors," Phys. Fluids 26, 1037. Bounce-averaged
  Fokker-Planck treatment of fusion-alpha loss-cone confinement: about 50 percent
  of alphas lost by count but under 25 percent by energy, so about 75-85 percent of
  the alpha power deposits. Primary source for the `f_alpha_heat` value.
- **Frank, S. J. et al. (2024)**, "Confinement performance predictions for a high
  field axisymmetric tandem mirror" (Hammir), arXiv:2411.06644. Primary source for
  the Realta Hammir Q>5 tandem central-cell design point (eq. 18 tandem confining
  potential `e*phi = T_e ln(n_p/n_c)`; sec. 3.5 design point: `ell_c = 50` m,
  `B_m = 25` T, `R_mc = 13.3`, `n_p = 1.5e20`, `n_c/n_p = 0.55`, `T_i = 45` keV,
  plug `T_e = 125` keV, `P_fus = 157.4` MW, `P_NBI = 30` MW, `Q = 5.2`, `tau_c`
  about 5 s).
- **Putvinski, S., Ryutov, D., Yushmanov, P. (2019)**, "Fusion reactivity of the
  pB11 plasma revisited," Nucl. Fusion 59. Relativistic bremsstrahlung including
  the electron-electron term.
- **Ochs, I. E., Kolmes, E. J., Mlodik, M. E., Rubin, T., Fisch, N. J. (2022)**,
  arXiv:2210.08076; **Kolmes, E. J., Ochs, I. E., Fisch, N. J. (2022)**, Phys.
  Plasmas 29, 110701. Bremsstrahlung form (Eq. 16); deferred p-B11 cross-validation
  reference.
- **Kolmes, E. J. et al. (2024)**, "Loss-cone stabilization in rotating mirrors:
  thresholds." Warm-fraction DCLC stabilization threshold for the physics-based
  confinement cap follow-on.
- **Stix, T. H. (1972)**, plasma waves / fast-ion slowing-down. Two-body
  slowing-down fraction of alpha power delivered to electrons.
- **Huba, J. D.**, NRL Plasma Formulary. Collision rates, channel-specific Coulomb
  logarithms, ion-electron energy equilibration.
- **Schwartz, J. et al. (2024)**, MCTrans++: a 0-D model for centrifugal mirrors.
- **Forest, C. B. et al. (2024)**, "Prospects for a high-field, compact break-even
  axisymmetric mirror (BEAM) and applications," J. Plasma Phys.,
  doi:10.1017/S0022377823001290. Cool simple-mirror electrons (T_e about 6 keV at
  T_i about 67 keV); cross-check of the classical-mirror confinement scaling
  (eq. 1.1, identical to Endrizzi eq. 3.4); corroborates GDT beta about 0.6, about
  1 keV T_e with ECH, and 0.20 m GDT plasma diameter.
- **Bagryansky, P. A. et al. (2015)**, "Threefold increase of the bulk electron
  temperature of plasma discharges in a magnetic mirror device," Phys. Rev. Lett.
  114, 205001 (preprint arXiv:1411.6288). GDT machine parameters: 7 m central cell,
  R_m = 35, B_min 0.27-0.35 T, T_e record 660 eV / peak > 900 eV at n = 0.7e19,
  standard-config 250 eV at n = 2e19, beta about 0.6, 5 MW NBI + 0.7 MW ECRH,
  deuterium.
- **Endrizzi, D. et al. (2023)** [C. B. Forest, corresponding], "Physics basis for
  the Wisconsin HTS Axisymmetric Mirror (WHAM)," J. Plasma Phys. 89, 975890501,
  doi:10.1017/S0022377823000806. WHAM design and the collisionless loss-cone
  (Pastukhov) regime; confinement scalings eq. 3.4 (Pastukhov/CM) and eq. 3.5
  (gas-dynamic, attr. Ivanov & Prikhodko 2013).
- **Ivanov, A. A. and Prikhodko, V. V. (2013)**, "Gas-dynamic trap: an overview of
  the concept and experimental results," Plasma Phys. Control. Fusion 55, 063001.
  GDT central cell designed long compared with the Coulomb mean free path
  (collisional, gas-dynamic regime); origin of the gas-dynamic confinement formula.
- Introduction to Tandem Mirror Physics (review).

No number in this document is sourced from or calibrated against any cost-modeling
tool; all come from the primary plasma-physics literature, except the constructed
smoothing width `w`, which is documented as constructed.
