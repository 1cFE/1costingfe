# Tokamak Power-to-Geometry Sizing

## Problem

Across every concept the device geometry is a fixed input. The power balance
solves the operating point (temperature, driver energy) on top of frozen
geometry, but never sizes the machine. As a result the coil account (C220103)
and all geometry-driven accounts are constant across power: verified at $515.5M,
identical from 50 to 2000 MWe, because R0 is pinned at its YAML value. This is
the subject of the open coil-scaling issue, but it is systemic across concepts,
not tokamak-specific.

## Goal

Add a sizing solve that maps target net electric power to tokamak geometry
(R0, a, B0), feeding the existing geometry-to-cost pipeline so that cost scales
with power. Tokamak is the reference implementation. The layered pattern is
built to generalize to other concepts later, each with its own sizing law.

## Scope

In scope: tokamak, on the 0D physics path.

Deferred (framework generalizes, not built here): stellarator (R0/a, plus 3D
coil markup), mirror (fixed radius, length sets power), laser and heavy-ion ICF
(rep rate then yield-per-shot), pulsed-power and module-replication concepts.

## Modes: one solve, four uses

1. **Pin.** Existing behavior. Cost exactly the stated machine. Sizing off.
2. **Size.** Solve R0/a/B0 from the design knobs plus the power target. Used to
   check whether a vendor's claimed machine is consistent with its claimed
   physics, and to cost an honest version.
3. **Scale.** Hold a real machine's design knobs, change the power target,
   re-solve. The design knobs are the scale-invariant fingerprint; only R0
   grows. This is the direct coil-scaling fix in action.
4. **Optimize.** Minimize LCOE over the Greenwald fraction f_GW as a nested
   outer loop, with R0 slaved to the power target.

## Inputs and solved outputs

Design knobs (the scale-invariant fingerprint), all sourced from YAML or
defaults.py or passed as overrides:

| Knob | Type | Encodes |
| --- | --- | --- |
| coils | enum: rebco, nb3sn, nbti, copper | B_max, coil cost, recirculating power, cryo class |
| aspect_ratio (A) | float | conventional (about 3) vs spherical (about 1.7) |
| elongation | float | plasma shaping |
| beta_N | float | Troyon limit (physics aggressiveness) |
| H_factor | float | confinement multiplier |
| f_GW | float | Greenwald fraction (fixed in size mode, free in optimize mode) |
| q95 | float | current and kink stability floor |
| fuel | enum | reactivity, energy per reaction, neutron fraction, conversion path |
| net_electric_mw | float | the power target to size for |

Solved outputs: R0, a (= R0 / A), b_center (= B0, derived from B_max and the
radial build).

## Magnet selection table

The coils enum carries a bundle of physical properties, looked up from a table
keyed by magnet type:

- b_max: peak field ceiling at the conductor.
- coil_cost_per_kAm: conductor cost. HTS is costlier per kA-m but needs less
  because the machine is smaller.
- recirc_power_factor: continuous power to hold the field. Zero for
  superconductors, nonzero for copper (resistive dissipation reduces net
  electric and forces the machine larger to compensate).
- cryo_class: operating temperature and cryoplant class.

This table lives in YAML or defaults.py, keyed by magnet type. b_max never
appears as a loose number. Picking rebco brings the ceiling, conductor cost,
recirculating power, and cryo class together.

## Sizing physics: constraint-boundary, fuel-general

At a fixed magnet ceiling and fixed aspect ratio, the constraint-boundary
operating point determines the plasma from size alone:

- Plasma current from the q95 floor: I_p proportional to a^2 * B0 / (R0 * q95).
- Density from the Greenwald fraction: n = f_GW * n_GW(I_p).
- Temperature from bounded maximization of net power over a per-fuel
  [T_min, T_max], with the beta limit (set by beta_N) as a binding check.
- On-axis field from the magnet ceiling and the radial build:
  B0 = B_max * (R0 - a - sum of inboard thicknesses) / R0. Fixed-meter blanket
  and shield penalize small machines on field, which is physical and makes
  high-field magnets matter more at small size.
- Fusion and net power from the existing tokamak_0d_forward physics, including
  fuel reactivity, bremsstrahlung, and the conversion path. No D-T constants
  appear in the solver.

Outer solve: bisect R0 in [R0_min, R0_max] (bounds in YAML) until net power
equals the target. Net power is monotonic in R0 at the boundary operating point,
so bisection is well posed.

Temperature bounds [T_min, T_max] per fuel live in YAML or defaults.py. A global
upper cap of about 400 keV reflects the validity range of the reactivity fits
(beyond it the parameterizations break down and relativistic corrections matter).
Per-fuel lower floors keep the search in the burning regime.

Feasibility: if net power at R0_max is below the target, or the temperature
maximization sits at its upper bound with net power not positive, the solve
raises a clear infeasibility error rather than diverging. Feasibility couples to
both magnet and fuel: aneutronic fuels at conventional field and beta do not
close, and the model reports this honestly (for example, p-B11 on copper is
infeasible).

## LCOE optimization: optimize mode, f_GW only

The model carries a disruption rate that rises exponentially as the operating
point approaches each limit (Greenwald, beta, kink), feeding back as
component-life shortening and downtime, hence into replacement cost and LCOE.
This makes LCOE non-monotonic in f_GW: pushing toward the Greenwald limit raises
power density and shrinks the machine (lower capital) but raises the disruption
rate (higher replacement). The minimum can be interior.

Nested structure:

- Inner: R0 bisection at a given f_GW (the size solve above).
- Outer: minimize LCOE over f_GW in (0, 1], R0 slaved, the hard limits as
  constraints. The model is JAX-differentiable, so a gradient-based or bounded
  scalar minimization applies. Boundary-sizing is the degenerate case (optimum
  at the limit).

Only f_GW is freed in this version. Freeing q95, the beta fraction, or T finds a
better optimum but is harder to interpret and leans harder on the disruption
model across more channels. Those are deferred until the f_GW path is trusted.

Disruption severity as input: the four disruption knobs
(disruption_rate_base, disruption_steepness, disruption_damage,
disruption_downtime) are explicit input parameters, sourced from YAML or
defaults.py, read without any inline fallback. The current code also hardcodes
inline params.get fallbacks for these (model.py:875-878); those are removed, so
a missing key errors rather than silently taking a magic number, consistent with
the placement discipline.

Where the optimum lands is set entirely by these input values. With the current
placeholder values (disruption_damage = 0.02, downtime 72 h) the replacement
penalty is small (single-digit percent of life across the f_GW range) while the
capital savings from shrinking the machine are larger, so the optimum sits at
the Greenwald boundary and optimize mode reproduces boundary-sizing. The
deciding knob is disruption_damage: a physically severe major-disruption value
(roughly 0.1 to 0.2) produces a 30 to 40 percent life reduction and a genuine
interior optimum. The placeholder values are not literature-grounded, so the
location of the economic optimum is currently unjustified. Grounding these four
inputs with a literature review (tracked as a separate task feeding an
account_justification writeup) is what makes optimize mode quantitatively
trustworthy. Optimize mode ships regardless, since the values are inputs; its
output simply tracks whatever severity the inputs encode.

## Plumbing

Placement in forward(): after derive_radial_build, in place of the
_power_balance call for the tokamak in sizing mode. The pipeline order becomes:

```
validate overrides -> merge params -> derive_radial_build
  -> [sizing solve, if gated]
  -> geometry (RadialBuild now uses solved R0/a)
  -> cost accounts -> economics
```

The solve, tokamak_size_from_power(params), returns R0, a, B0, the plasma
state, and the power table. The returned power table becomes the pipeline's pt;
there is no separate power-balance call in sizing mode.

Gating flags in YAML, both default false:

- size_from_power: enables the sizing solve.
- optimize_lcoe: wraps sizing with the outer f_GW minimization (implies
  size_from_power).

When size_from_power is true and the concept is tokamak:

- Reject R0, plasma_t, and b_center if present in overrides. Pinning a solved
  output is contradictory, and the existing contract never silently swallows a
  parameter, so this raises a clear ValueError.
- Call the sizing solve, inject solved R0, plasma_t, and b_center into params,
  and use the returned power table.

n_mod: the solve sizes one module to net_electric_mw / n_mod, consistent with
the existing per-module power balance.

## Placement discipline

All numbers (the magnet table, temperature bounds, physics limits, R0 bounds,
disruption parameters) live in YAML or defaults.py, keyed by magnet type,
concept, or fuel as appropriate. None appear inside function bodies, and none
appear as arg=default values in function signatures. The solver is pure: a
params dict in, geometry and power out.

## fusion-tea interface

The injection channel is unchanged. Physics and design knobs arrive through
forward(**overrides), validated against the concept YAML plus the costing
constants plus the optional-override whitelist, then merged over the YAML
defaults (overrides win). Published costs arrive through cost_overrides with
override_reference_mw.

Sizing shifts which keys are injected: from R0, a, and b_center (pinned
geometry) to the design knobs (coils, aspect_ratio, beta_N, H_factor, f_GW,
q95). The new keys are added to mfe_tokamak.yaml and the override whitelist so
that overrides accept them. A published R0 becomes a validation target rather
than an input: inject a machine's design knobs and check that the solver
reproduces its published R0.

## Backward compatibility

Both gating flags default false. Every existing config and the full test suite
behave identically, because the sizing solve is never entered. The new design
knobs are inert when sizing is off.

## Testing

- Backward compatibility: with sizing off, outputs are identical to current.
- Monotonicity: R0 increases with the power target.
- Scaling: R0 proportional to P^(1/3), so 8x power gives about 2x R0.
- Coil-scaling fix: C220103 varies across 50 to 2000 MWe (the original failing
  assertion).
- Fuel generality: D-T solves; p-B11 on copper is flagged infeasible; p-B11 on
  rebco solves to a large R0.
- Magnet differentiation: rebco sizes smaller than nb3sn at the same power and
  physics.
- ARC validation: inject ARC design knobs and confirm R0 about 3.3 m.
- R0-pin rejection: size_from_power with an R0 override raises.
- Scale mode: 270 to 1000 MWe grows R0 about 1.5x and scales cost.
- Optimize mode: returns f_GW at the boundary under current disruption
  parameters, and an interior f_GW under a severe disruption_damage.

## Out of scope and follow-ups

- Other concepts (stellarator, mirror, ICF, pulsed): same nesting, different
  sizing law. Deferred.
- Disruption severity literature review: the four disruption inputs
  (rate_base, steepness, damage, downtime) currently carry placeholder values.
  A literature review grounds them and feeds an account_justification writeup.
  This sets whether the LCOE optimum is interior or at the boundary, so it is
  what makes optimize mode quantitatively trustworthy. Separate task.
- Freeing more optimizer knobs (q95, beta fraction, T): after the f_GW path is
  trusted.
- Anchor-on-raw-geometry back-out, needed for scale mode when only R0/a/B0 are
  given instead of the design knobs: minor, deferred.
