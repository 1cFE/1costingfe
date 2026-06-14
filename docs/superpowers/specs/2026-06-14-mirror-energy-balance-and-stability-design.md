# Mirror 0D Energy-Balance Closure, Confinement-Regime Fix, and Stability Validity

**Date:** 2026-06-14
**Status:** Approved design, pending implementation
**Settled decisions (user, 2026-06-14):**
- Scope: full real-machine calibration goal, reached economics-first with a
  stability bound added only if needed (observe-first).
- Confinement regime: sourced smooth collisionality bridge (always-on kernel
  fix), not a hard switch.
- Energy balance: close it in sizing/optimize mode by computing required
  auxiliary power from the corrected confinement and reusing the existing
  recirculating + DEC machinery; the tokamak path is the template and stays
  unchanged.
- Stability: observe-first. Ship the regime + balance fixes plus
  collisionality/DCLC diagnostics; add an explicit stability constraint only
  if the corrected plasma does not settle into a sensible regime on its own.
- Branch off `feat/mirror-wall-fluence` (this builds on the wall-loading and
  sizing work; the two merge together).

**Depends on:** 2026-06-12-mirror-wall-loading-and-radial-build.md (Implemented,
on the held `feat/mirror-wall-fluence` branch), 2026-06-11-mirror-0d-sizing-design.md.

## Motivation

The mirror length-sizing and LCOE-optimize path drives D-T to about 60 keV
thermal. Real D-T mirrors (GDT, WHAM) run cool, near 10 keV. This is a model
bug, not a designed regime, with two compounding causes (both verified by
reading the code and running the model):

1. **Confinement regime is not gated by collisionality**
   (`src/costingfe/layers/mirror.py:286`):
   `inv_tau_axial = 1.0 / tau_Pastukhov + 1.0 / tau_GD` is an unconditional
   harmonic sum, which always selects the shorter time. At the operating point
   the plasma is deeply collisionless (L / mean-free-path about 3e-5), where
   confinement is physically loss-cone (Pastukhov) limited and of order
   seconds, but the gas-dynamic flow-out time (about 5e-5 s) does not apply
   there yet spuriously dominates the sum, making tau_E about 1e4 too short.
   The original spec claimed this sum "selects Pastukhov at low collisionality";
   it does the opposite. The `collisionality` diagnostic
   (`mirror.py:329`) that would catch this is computed and never used.

2. **The steady-state energy balance is never closed** in the mirror sizing
   path (`mfe_forward_power_balance` in `physics.py`, fed a fixed YAML
   `p_input`): the computed transport and end loss is never charged as the
   sustainment power. The optimizer gets confinement for free and climbs in
   temperature until the wall and beta caps box it in.

These interact perversely: cause 1 inflates p_end to about 6e6 MW (6 TW) at
60 keV; cause 2 ignores it. A 6 TW end loss against about 280 MW of available
heating (alpha plus p_input) means the reported operating point violates energy
conservation and cannot exist in steady state. A naive fix of only charging the
existing p_end would charge 6 TW and call every mirror infeasible; both causes
must be fixed together.

## The tokamak is the healthy reference (verified)

The tokamak 0D path has neither bug and needs no change:

- **Closed balance in sizing mode** (`tokamak.py:213-242`,
  `aux_heating_from_confinement`, used at `tokamak.py:904-943`): the required
  auxiliary power is derived from the confinement quality (H_factor) and the
  IPB98 scaling, `P_heat = (W_th / (H_factor * K))^(1/0.31)`, aux =
  `max(0, P_heat - p_alpha)`, and that value is fed as `p_input` to the power
  balance. The comment is explicit: "H_factor (not a fixed p_input) sets the
  recirculating power." This is exactly the closure the mirror lacks.
- **Single empirical confinement regime** (IPB98), so there is no two-regime
  bridge to get wrong, plus Greenwald, beta, and q95 limits that box the
  optimizer in.
- **Forward/audit mode** (`tokamak.py:328`) takes a stated `p_input` and reports
  the implied H_factor, which is correct for auditing a stated machine.

The mirror fix is therefore "give the mirror the analog of
`aux_heating_from_confinement`" plus the regime bridge. We copy a reviewed,
working pattern rather than invent one.

## Mode distinction

- The **confinement-regime bridge** is a kernel fix and applies ALWAYS
  (forward, inverse, sizing): tau_E is wrong everywhere it is computed.
- The **energy-balance closure** applies in **sizing/optimize mode** (compute
  the required auxiliary power from the corrected confinement and charge it),
  paralleling the tokamak. **Forward/inverse audit mode** keeps a stated
  `p_input` for auditing a given machine and additionally reports a
  sustainment-consistency diagnostic (the mirror analog of the tokamak's
  implied H_factor): the ratio of stated p_input to the
  confinement-required auxiliary power.

---

## Part 1: Confinement-regime collisionality bridge (always-on)

Replace the unconditional harmonic sum (`mirror.py:286`) with a smooth,
JAX-differentiable bridge between the gas-dynamic (collisional) and Pastukhov
(collisionless) limits, gated by collisionality so the gas-dynamic channel only
governs when the mean-free-path is at or below L.

**Sourcing.** Research the published collisional-to-kinetic mirror confinement
treatment (Ryutov gas-dynamic confinement and its validity boundary; Pastukhov
and Cohen et al. for the kinetic limit; the gas-dynamic-trap reviews of Ivanov
and Prikhodko). Implement the transition as a single smooth function of
collisionality. If the literature provides only the two limits plus a regime
boundary rather than a turnkey closed-form bridge, the implementation builds a
smooth gate on that boundary (for example, suppressing the gas-dynamic loss rate
by a smooth function that vanishes as mean-free-path over L grows) and documents
it as such. The formula, its validity window, and the sourcing go in a new
account-justification writeup
(`docs/account_justification/mirror_confinement_regimes.md`) in house style.

**Constraints.** Smooth and differentiable (the golden-section sizing and the
jax.grad sensitivity vector both traverse this). float32-disciplined per the
1e20 density convention. Re-anchor: GDT (collisional) must still land on the
gas-dynamic branch and WHAM (collisionless) on the Pastukhov branch, both within
the existing 2x anchor tolerance in `mirror_confinement.md`. This re-anchoring is
the guard that the new bridge did not break the literature-validated points.

## Part 2: Energy-balance closure in sizing mode

Add a mirror analog of `aux_heating_from_confinement`. From the forward
`MirrorPlasmaState` (all terms already known, no iteration):

    P_aux = max(P_aux_floor, P_end + P_radial + P_rad - P_alpha)

where P_end, P_radial, P_rad, P_alpha are the corrected forward powers (corrected
because Part 1 makes tau_E, hence P_end, physical) and `P_aux_floor` is a small
YAML control/startup floor so an ignited point still pays some control power. In
sizing/optimize mode, `P_aux` is passed as `p_input` to
`mfe_forward_power_balance`, exactly as the tokamak sizing passes its
confinement-derived aux. The existing recirculating term charges
`P_aux / eta_pin` (NBI wall-plug efficiency) and the existing DEC term recovers
`f_dec * p_transport` at `eta_de`, so beam cost nets against direct-conversion
recovery of the same end loss in code that already exists. `f_dec` remains the
YAML input per the settled decision in the parent spec. The shared
`mfe_forward_power_balance` and the tokamak path are not modified.

**Verification risk (carried into the plan).** Confirm the shared balance's
recirculating/DEC structure expresses "beam-in minus DEC-recovery-of-the-same-
end-loss" when fed `p_input = P_aux`. If it cannot express this cleanly, fall
back to an explicit mirror double-duty term
(`net recirc = P_aux/eta_pin - eta_de * recoverable_axial_fraction * P_end`)
rather than forcing the shared function. Decide and document at implementation.

**Sizing handoff.** This composes with the GSS-optimum handoff already fixed on
the branch (the sized state is built from the GSS optimum, not an inverse
re-solve). The GSS objective (net electric) now reflects the true recirculating
cost of sustainment, which is what pulls the optimum off the hot runaway.

## Part 3: Stability and validity (observe-first)

Ship diagnostics now; constrain only if needed.

**Diagnostics (now).** Compute and report on `MirrorPlasmaState`:
- `collisionality` (already computed; now actually surfaced and used as a
  validity flag): when L / mean-free-path is below the Pastukhov-Maxwellian
  validity threshold, the Pastukhov formula overestimates confinement; flag it.
- a DCLC-relevant diagnostic (candidate: a warm-plasma or sloshing-ion stream
  parameter, or the loss-cone-driven microstability proxy from the mirror
  microstability literature), reported as a number, not yet constraining.

**Constraint (only if needed).** After Parts 1 and 2 land, run the corrected
model across the fuel set and observe where the D-T plasma settles. If it lands
in a sensible, validity-respecting regime on its own (economics alone does the
work), no explicit stability constraint is added. If it still walks outside the
trustworthy regime, add a sourced operating bound (the leading candidate is a
minimum warm-plasma fraction required to fill the loss cone and stabilize DCLC,
which both GDT and WHAM rely on) as a sizing/optimize feasibility constraint that
raises `OperatingPointInfeasible` or restricts the T and n search window. Any
such limit is sourced in the same account-justification writeup, with new YAML
knobs. This decision is explicitly deferred to an observation step in the plan.

---

## Validation

1. **Mirror anchors re-pass.** GDT and WHAM confinement within the documented
   2x of `mirror_confinement.md`, on the new regime bridge.
2. **Energy-balance sanity.** At the corrected D-T sizing optimum: tau_E is
   physical (P_end is well below P_fus), the steady-state balance closes
   (P_aux equals losses minus alpha by construction), and Q_eng and the
   recirculating fraction are realistic.
3. **Realistic operating temperature.** The D-T sizing/optimize optimum lands
   in a realistic band (pin a sanity range, for example 8 to 25 keV, not a
   point), no longer near 60 keV. Sensible optima for the other fuels.
4. **Tokamak cross-check against literature.** Because the tokamak closure is
   the template we copy and the reference for "what a healthy closed-balance 0D
   result looks like," confirm the unchanged tokamak path reproduces published
   design points within a documented tolerance: operating temperature, Q, and
   recirculating/auxiliary-power fraction against a small set of designs
   (candidates: ITER, SPARC, ARC, ARIES-AT, EU-DEMO), sourced in an
   account-justification writeup. This is a sibling of the mirror GDT/WHAM
   anchors and validates the reference, not new tokamak behavior. If the
   tokamak deviates materially from literature, surface it (it would mean the
   reference pattern itself needs scrutiny), but do not change tokamak behavior
   under this spec without a separate decision.

## The re-pin (sanctioned)

Closing the balance and correcting tau_E moves essentially every mirror cost
number. Charging beam sustainment will likely raise mirror LCOE substantially
and may make some geometries uneconomic; the honest result may be that the
earlier mirror economics were optimistic. This is a large, sanctioned re-pin
with a before/after table in the regime writeup (concept, fuel, operating T,
tau_E, P_aux, Q_eng, recirculating fraction, LCOE: old then new). Discipline as
in the fluence task: the coil calibration pin (513.375 M$, capital) must NOT
move; non-mirror concepts (tokamak, stellarator, IFE/MIF) are untouched and
their pins must NOT move; a moved coil pin or a broken non-mirror result is a
regression to investigate, not a re-pin.

## Scope and phasing

One spec; the implementation plan phases it: (1) regime bridge plus re-anchor;
(2) energy-balance closure in sizing, with the verification-risk decision;
(3) diagnostics and the observe step; (4) the conditional stability constraint
if the observation requires it; (5) tokamak-vs-literature cross-check;
(6) the sanctioned re-pin, docs, paper, examples, final review.

## Out of scope

- Full hot-ion non-Maxwellian transport (beam-driven sloshing-ion distribution,
  fast-ion fusion, T_i decoupled from T_e as a solved quantity). The model keeps
  T_i and T_e as inputs. Explicitly declined; would turn the 0D costing layer
  into a design code.
- Any change to the tokamak, stellarator, or IFE/MIF physics or costs. The
  tokamak is validated as the reference only.
- The f_dec provenance writeup (still open, tracked separately).

## Open items and risks

- The "sourced smooth bridge" may rest on a constructed gate over a published
  boundary rather than a named closed-form; acceptable per the regime decision,
  documented honestly.
- Whether economics alone settles a sensible regime (Part 3) is unknown until
  observed; the explicit stability constraint is the contingency.
- Mirror economics may degrade enough that mirrors look much less favorable;
  this is a result to report, not a problem to engineer around.

## Test plan

- Regime bridge: jit==eager and finite gradients across all four fuels for the
  new bridge function; GDT and WHAM anchors within 2x; a collisional point uses
  the gas-dynamic branch and a collisionless point uses Pastukhov (pin the
  branch selection at two collisionality values).
- Energy balance: at a sizing point, P_aux equals (P_end + P_radial + P_rad -
  P_alpha) clamped at the floor (closed-form check); recirculating power tracks
  P_aux / eta_pin net of DEC recovery; the D-T sizing optimum T is in the
  sanity band; tau_E physical (P_end below P_fus).
- Mode distinction: forward/audit mode still accepts a stated p_input and reports
  the sustainment-consistency diagnostic; sizing mode ignores the stated p_input
  and uses the confinement-derived P_aux.
- Diagnostics: collisionality and the DCLC proxy are present on the state and
  finite across fuels; the validity flag fires when collisionless.
- Tokamak cross-check: the unchanged tokamak reproduces the chosen literature
  design points within the documented tolerance.
- Pins: coil 513.375 unmoved; non-mirror LCOE pins unmoved; mirror pins re-pinned
  with the before/after table.
