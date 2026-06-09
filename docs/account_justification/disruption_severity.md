# Disruption Severity Parameters

## Purpose

The tokamak disruption model (`src/costingfe/layers/tokamak.py`,
`compute_disruption_rate` and `apply_disruption_penalty`) converts proximity to
the Greenwald, beta, and kink stability limits into a disruption frequency, then
penalizes component life and availability. It needs four input parameters:

- `disruption_rate_base` [disruptions per full-power-year, FPY]: baseline rate
  far from the limits.
- `disruption_steepness` [dimensionless]: how sharply the rate rises as the
  operating point approaches a limit (exponent in the exponential margin law).
- `disruption_damage` [fraction of plasma-facing-component life per disruption].
- `disruption_downtime` [hours per disruption].

This writeup grounds those four values in reactor-relevant literature. They are
inputs (YAML or defaults.py), not hardcoded constants.

## Recommended values

| Parameter | Low | Base | High | Units | Confidence |
| --- | --- | --- | --- | --- | --- |
| disruption_rate_base | 0.3 | 1.0 | 5.0 | disruptions/FPY | Medium |
| disruption_steepness | 8 | 12 | 20 | dimensionless | Low |
| disruption_damage | 0.002 | 0.01 | 0.05 | fraction PFC life/disruption | Low-Medium |
| disruption_downtime | 12 | 72 | 720 | hours/disruption | Low-Medium |

Current YAML defaults (`steady_state_tokamak.yaml`) are rate_base 0.1,
steepness 15.0, damage 0.02, downtime 72.0. The review supports raising
rate_base by about a factor of 10 (to about 1.0); the other three current values
sit inside the recommended ranges.

## Rationale

### rate_base (base 1.0 disruptions/FPY)

The de Vries et al. JET disruption survey gives an irreducible, precursorless
disruption floor of about 0.4 percent of pulses, rising to about 1.0 to 1.6
percent of pulses once human error and system failures are included. ITER design
targets a disruptivity of about 1 percent of pulses with 95 to 99 percent
mitigation and a tolerance of about 1 unmitigated disruption in 10,000 pulses.
Converted to a per-FPY basis using reactor pulse-length and duty assumptions,
this lands at roughly 0.5 to 5 disruptions/FPY. The current default of 0.1/FPY
is below even the ITER unmitigated floor and should move up to about 1.0.
Confidence: medium (the pulse-to-FPY conversion carries assumptions, but the
pulse-fraction data are solid).

### steepness (base 12)

The DIII-D disruptivity database shows disruptivity above 60 percent at q95 about
3, falling to near zero by q95 about 6. JET density-limit data show the Greenwald
boundary behaving as a near-hard wall. Mapped onto the model's single exponential
margin law, these slopes correspond to a steepness of roughly 8 to 20, base 12.
Confidence: low. This is the least defensible parameter, for two reasons: the
exponential is a curve fit rather than a physics law, and a single steepness is
shared across three boundaries that are not equally sharp (the density limit is
harder than the beta and kink limits). A future refinement is a per-channel
steepness.

### damage (base 0.01 fraction of PFC life per disruption)

Thermal-quench energy densities reach tens of MJ/m2, above the melt thresholds
for tungsten and beryllium. The JET ITER-like-wall program recorded beryllium
melting and the dislocation of about 42 g of material in a single event. Runaway
electrons deposit about 20 to 100 kJ per panel with deep, localized melting that
can breach cooling channels. Translating these into fraction-of-life against the
EU-DEMO divertor replacement interval (about 2 FPY) and an approximately
3000-disruption qualification envelope gives about 0.001 to 0.005 for a minor or
mitigated event and about 0.02 to 0.2 for a major unmitigated event. The
distribution-weighted base is about 0.01. The current default of 0.02 is a
defensible mid-to-high value. Confidence: low-medium.

### downtime (base 72 hours per disruption)

Recovery ranges from hours (plasma reconditioning, wall reconditioning) for minor
events to weeks or months for events that require remote-handling component
replacement after melt or runaway-electron damage. A distribution-weighted base
of about 72 hours is reasonable, with the high tail (720 hours) representing
major in-vessel repairs. Confidence: low-medium.

## Mitigation

ITER requires its disruption mitigation system (massive gas injection, then
shattered pellet injection) to cut divertor heat loads by at least 90 percent and
electromagnetic forces by a factor of 2 to 3, at 95 to 99 percent reliability. A
reactor design therefore assumes a largely mitigated disruption population, which
is why the recommended base damage and downtime sit in the lower-middle of their
ranges rather than at the unmitigated severe end.

## Other concepts

Net-current-free stellarators are genuinely disruption-free, so a zero penalty is
physically justified rather than a modeling shortcut (the stellarator YAML already
disables it). This does not automatically extend to current-carrying or pulsed
concepts, which must be assessed individually.

## Implication for the sizing optimize mode

The LCOE-optimal Greenwald fraction depends on whether the disruption penalty is
strong enough to offset the capital saving from running closer to the density
limit. At the current default rate_base of 0.1/FPY the penalty is weak and the
optimum sits at the Greenwald boundary. At the literature-grounded base
(rate_base about 1.0, damage about 0.01) the penalty is about an order of
magnitude stronger: for a core life of about 30 FPY the effective-life reduction
near the limit is roughly 20 to 30 percent, comparable to the capital saving.
With grounded values the LCOE optimum is therefore likely interior, so the sizing
optimize mode is quantitatively meaningful rather than cosmetic.

## Sources and confidence note

Primary references identified in this review: the de Vries et al. JET disruption
characterization studies (Nuclear Fusion); the DIII-D disruptivity database
publications; the JET ITER-like-wall beryllium melt experiments; EU-DEMO divertor
lifetime and replacement studies; and the ITER disruption mitigation system and
shattered-pellet-injection requirement documents. Exact DOIs and URLs should be
appended before any of these numbers are cited in the paper. The value ranges and
confidence levels above reflect the state of that literature, which is strongest
for disruption frequency, weakest for the steepness of the frequency-versus-margin
relationship.
