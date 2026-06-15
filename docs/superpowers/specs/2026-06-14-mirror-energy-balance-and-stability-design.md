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

## Revision 2026-06-14 (tandem reframe, user-approved)

During implementation (after Tasks 1 and 2 landed) verification found the
corrected model spuriously ignites: with the gas-dynamic/Pastukhov bridge in
place, the deeply collisionless D-T point (collisionality about 3e-5, 3.5
decades below the 1/R_m Pastukhov validity boundary) gives tau_E about 1350 ms
versus about 98 ms from the WHAM scaling, too long by 15-24x. Density is pinned
by the neutron-wall cap so P_fus is flat across 15-55 keV, the net-electric
objective is flat, and the energy-balance closure has no thermal lever; the
optimizer still parks at about 60 keV.

Root cause and decision: the cost model ALREADY commits to a TANDEM mirror. The
CAS22 coil account costs `n_plug_coils = 4` high-field end-plug HTS coils (2 per
end, Hammir class) at the throat field plus the long central solenoid string.
The parent spec's "simple axisymmetric mirror, no tandem plugs" physics scope is
therefore inconsistent with the hardware being costed, and Realta Hammir (the
reference machine) is a tandem. The electrostatic-plugging structure in
`compute_tau_pastukhov` (the `exp(e*phi/T_i)` enhancement) is the correct tandem
mechanism; the defect is that the confining potential is uncalibrated and
unbounded (the simple Boltzmann ambipolar value `phi = T_e*ln(sqrt(m_i/2 pi
m_e))` grows with T_e without limit), so confinement is over-credited and the
plasma ignites. Real tandems are plug-limited at Q about 1 to a few at roughly
10-30 keV.

Approved direction: model a tandem at costing fidelity. Keep Task 1's
collisionality bridge (a tandem central cell runs collisionless and correctly
sits on the plugged branch; gas-dynamic still governs collisional GDT-class
machines). Keep Task 2's energy-balance closure (P_aux = losses - alpha is
correct for any mirror). REPLACE the unbounded Boltzmann confining potential
with a tandem plug-limited confining potential, calibrated to a published tandem
design point (Realta Hammir Q>5 at the 50 m central cell) plus tandem-mirror
literature (MFTF-B, TMX-U, Fowler and Ryutov), so the operating point lands
where real tandems are, the spurious ignition disappears, and the energy-balance
closure regains its thermal lever. This supersedes the earlier "degrade tau
toward the WHAM simple-mirror anchor" idea (which would have credited
simple-mirror confinement to hardware we cost as a tandem) and the "port the
WHAM/Endrizzi simple-mirror scaling" idea (same incoherence). The sections below
are updated to this tandem framing; Part 1's "Pastukhov branch" now means the
calibrated tandem plug-limited confinement, and the WHAM simple-cell anchor
remains valid only for a single end-plug cell, not the central-cell confinement.
The buggy `f_dec_eff` fallback from Task 2 is expected to be removable once the
plasma is no longer ignited (the clean p_transport identity returns).

## Revision 2 2026-06-15 (alpha loss-cone heating, user-approved)

After Task 2b the D-T optimum moved to 29.9 keV, but the cross-fuel observation
(Task 3) showed it is still near-ignited: Q_sci about 516, with the required
auxiliary power pinned at the 2 MW control floor, because alpha heating
(206.8 MW) almost exactly cancels total losses (208.8 MW). The model credits
100 percent of the charged fusion power (`p_alpha` from `ash_neutron_split`) as
plasma self-heating, with NO loss-cone reduction. Real D-T mirrors lose fusion
alphas out the loss cone before they thermalize, which is why they run driven
(Q about 5, like Hammir) rather than ignited. The model being more ignited at
30 keV than the real Hammir machine is at 45 keV is physically backwards, and
the floored auxiliary power means the beam-drive lever is dead, so the optimum
is set by the wall cap and DEC credit rather than confinement-vs-drive
economics. The 30 keV optimum is therefore still partly an artifact.

Physics and decision: mirrors lose roughly 50 percent of alphas by count but
under 25 percent by energy (Santarius and Callen, Phys. Fluids 26, 1037, 1983,
the canonical bounce-averaged Fokker-Planck treatment for tandem central-cell
alphas), because most alphas slow down on the electrons before scattering into
the loss cone. So about 75-85 percent of alpha power deposits. Introduce a
sourced alpha-heating fraction `f_alpha_heat` (YAML, default about 0.80, range
0.75-0.85, calibrated/sourced to Santarius and Callen; no Python default per the
defaults-in-YAML rule). In the sustainment balance subtract
`f_alpha_heat * p_alpha` instead of the full `p_alpha`, and route the lost
fraction `(1 - f_alpha_heat) * p_alpha` to the axial end-loss / DEC channel (it
exits the loss cone as directed exhaust). Propagate the reduced self-heating
consistently into the q_sci / q_eng accounting and the P_aux handoff (sizing and
inverse). Counterfactual check: a 20 percent loss raises the required beam power
from 2 MW to about 43 MW (comparable to Hammir's 30 MW NBI), un-floors the lever,
and should land the optimum at a genuinely driven point. Honest nuance: Hammir
itself credits near-full alpha deposition and is driven primarily by end losses
(which the model already represents via P_end); the fix matters here because full
alpha heating happens to land within 1 percent of cancelling those end losses, so
the missing alpha-energy loss is what tips the model from spurious near-ignition
into the driven regime. After this fix, re-run the cross-fuel observation; the
settled point should be driven and Hammir-consistent. A 0D costing model retains
residual approximations by design; this is the last planned physics layer before
declaring the model sound within its stated band.

## Revision 3 2026-06-15 (decouple plug and central cell, user-approved)

Implementing the real Fowler-Logan plug potential (Revision 2) forced hot
electrons: e*phi = T_e*ln(n_p/n_c) needs a high T_e to plug the central cell, so
the global mirror default was re-pinned to T_e = 125 keV (Hammir). That made D-T
a genuinely driven tandem (T_i about 23 keV, P_aux about 230 MW, Q_eng about
1.77, LCOE about 369), but hot electrons produce large bremsstrahlung, so D-D,
D-He3, and p-B11 went net-negative and could no longer be sized. A single
electron temperature cannot serve both the plug (wants hot) and aneutronic
central-cell fusion (wants cool).

Decision (user): real advanced-fuel tandems use different plasmas, often
different ion species, in the plug versus the central cell. The plug builds the
confining potential with hot electrons (and a hydrogenic plugging species); the
central cell runs the working fuel and can keep its electrons cool to control
bremsstrahlung. Decouple them at costing fidelity:

- Add a separate PLUG electron temperature `T_e_plug` (hot, ECH-heated) that sets
  the Fowler-Logan potential: e*phi = T_e_plug * ln(n_p/n_c). The central-cell
  electron temperature (the existing `T_e`) keeps setting central-cell
  bremsstrahlung and is coolable for advanced fuels.
- Charge the plug's power: add a plug sustainment power `P_plug` to recirculating
  power (the ECH/NBI holding the hot-electron plug), calibrated to Hammir's about
  30 MW plug drive. This is also the competing penalty that keeps the optimizer
  honest (a deeper/hotter plug costs more).
- Model the plug as potential-plus-power (a separate cell whose role is the
  confining potential and its power cost), NOT a full second fusion plasma. The
  central cell does the fusion. "Different species" is captured by the plug being
  this separate potential-and-power sub-system, independent of the central fuel.

Consequences: D-T stays a hot-central tandem (Hammir's central cell genuinely
runs about 125 keV), so its driven result (about 369) is unchanged, now with the
plug power charged explicitly (reconcile the Hammir Q anchor, which already
counts the 30 MW plug NBI). Advanced fuels get a fair evaluation: a cool central
cell (low brem) plugged by a hot plug, paying the plug power. The outcome
(viable or still hard) is a real finding either way. New YAML: `T_e_plug` (hot,
default Hammir about 125 keV) and the plug-power calibration; the central `T_e`
default stays Hammir-hot for the D-T reference machine and is overridden cool for
advanced-fuel configs. This is the last planned physics layer; after it the model
is declared sound for D-T within its band and the advanced-fuel result is
reported as found.

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

1. **Mirror anchors re-pass.** GDT (gas-dynamic, collisional branch) and WHAM
   (single end-plug cell) confinement within the documented 2x of
   `mirror_confinement.md`, on the new regime bridge.
1b. **Tandem anchor (new, load-bearing).** The calibrated tandem plug-limited
   central-cell confinement reproduces the Realta Hammir Q>5 design point (50 m
   central cell, published fields/density/temperature) within a documented
   tolerance (start 2x). This is the anchor that pins the plug-confinement
   calibration; it is the tandem analog of the WHAM/GDT anchors. Sourced in
   `mirror_confinement_regimes.md` from primary tandem-mirror literature (Realta
   announcements/Forest et al.; MFTF-B, TMX-U, Fowler and Ryutov).
2. **Energy-balance sanity.** At the corrected D-T sizing optimum: tau_E is
   physical (P_end is well below P_fus), the steady-state balance closes
   (P_aux equals losses minus alpha by construction), the plasma is NOT
   spuriously ignited (Q is tandem-realistic, about 1 to a few), and Q_eng and
   the recirculating fraction are realistic.
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
