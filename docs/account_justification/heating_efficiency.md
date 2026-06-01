# Heating wall-plug efficiency (eta_pin) account justification

Decomposition of the heating wall-plug efficiency `eta_pin` into a per-method
source efficiency and a per-concept coupling efficiency, with sourcing and
provenance for each value.

See also: `docs/plans/2026-06-01-method-dependent-eta-pin.md` for the design
rationale, and the Steady-State Power Balance subsection of the paper
(`docs/papers/1costingfe_paper/1costingfe_paper.tex`) for the equation in context.

## Method

`eta_pin` is the wall-plug-to-plasma-absorbed efficiency of the auxiliary
heating system. It enters the steady-state power balance as the recirculating
heating draw `P_in / eta_pin`. It is not a single device constant: it is the
product of a method-level source efficiency and a device-level coupling
efficiency. For a heating mix it is the mix-weighted product

    eta_pin = sum_i p_i / sum_i ( p_i / (eta_source_i * eta_couple) )

where `i` runs over the heating methods present (NBI, ICRF, ECRH, LHCD),
`eta_source_i` is the wall-plug-to-delivered source efficiency of method `i`
(a global constant), and `eta_couple` is the concept's delivered-to-absorbed
coupling (a per-concept YAML parameter). For a single-method concept this
reduces to `eta_pin = eta_source(method) * eta_couple`.

Concepts whose input power is not delivered by an NBI/RF heating system
(electrostatic confinement such as orbitron and polywell; all pulsed drivers)
specify `eta_pin` directly instead of `eta_couple`. An explicit `eta_pin`
override always bypasses the derivation.

## eta_source (per method, wall-plug to delivered)

| Method | eta_source | Basis |
|---|---|---|
| NBI | 0.60 | Negative-ion NBI source wall-plug efficiency (OSTI 2441289; ITER NBI). |
| ICRF | 0.70 | RF tetrode transmitter final-amplifier efficiency (ITER ICRF, about 70%). |
| ECRH | 0.50 | Gyrotron wall-plug efficiency. |
| LHCD | 0.50 | Klystron wall-plug efficiency. |

These are device-independent properties of the heating source technology. They
live in `costing_constants.yaml` / `CostingConstants` as `eta_source_{nbi,icrf,ecrh,lhcd}`.

## eta_couple (per concept, delivered to plasma-absorbed)

| Concept | Method | eta_couple | eta_pin = source x couple | Provenance |
|---|---|---|---|---|
| Tokamak | NBI | 0.8333 | 0.60 x 0.8333 = 0.50 | Chosen to preserve the prior calibrated `eta_pin = 0.50`. |
| Mirror | NBI | 0.8333 | 0.50 | Preserves prior `eta_pin = 0.50`. |
| Stellarator | ECRH | 1.0 | 0.50 x 1.0 = 0.50 | Preserves prior `eta_pin = 0.50`; ECRH single-pass absorption is high. |
| Dipole | ICRF | 1.0 | 0.70 x 1.0 = 0.70 | Preserves prior `eta_pin = 0.70`. |
| Steady FRC | NBI | 0.43 | 0.60 x 0.43 = 0.26 | Sourced, not back-fit. TAE C-2W coupling: tangential injection through long ducts plus short plasma path give shine-through and duct losses (OSTI 2441289). |

The FRC coupling is the one value anchored directly to source data; it
reproduces the TAE two-step efficiency chain (source 0.60 times coupling 0.43).
The other concepts' coupling factors are chosen so the product reproduces each
concept's previously calibrated `eta_pin`, so the cross-concept benchmarks
(ARC, ARIES) are unchanged by the decomposition. The factorization is also more
physically faithful: it makes the recirculating heating load track the actual
driver rather than a single fudged scalar.

## Why this matters for the generic FRC

The FRC concept is driver-neutral: a TAE-style machine is NBI-driven and a
PFRC-style machine is RF/RMF-driven. With the decomposition, swapping the driver
changes the source efficiency automatically (NBI 0.60 versus RF 0.70). An
RF-driven FRC also couples better than a beam-driven one (RMF coupling about
0.60 versus 0.43 shine-through-limited NBI), which is captured by overriding
`eta_couple` alongside the RF heating split. So the PFRC variant correctly has a
lower recirculating fraction than the TAE variant.

## Known simplification

`eta_couple` is a single per-concept value, so within one device all methods
share the coupling and the method distinction comes through `eta_source` alone.
The within-device coupling difference (an FRC's NBI coupling 0.43 versus its RMF
coupling about 0.60) is handled by overriding `eta_couple` for the RF case, not
automatically. Per-(concept, method) coupling is a documented future refinement.

## Implementation

- Constants: `eta_source_{nbi,icrf,ecrh,lhcd}` in
  `src/costingfe/data/defaults/costing_constants.yaml` and the `CostingConstants`
  dataclass in `src/costingfe/defaults.py`.
- Derivation: `CostModel._effective_eta_pin` in `src/costingfe/model.py`,
  injected at the top of `_power_balance` so every power-balance path uses it.
- Per-concept `eta_couple`: the heated steady-state concept YAMLs in
  `src/costingfe/data/defaults/`. Electrostatic and pulsed concepts keep `eta_pin`.
- Tests: `tests/test_eta_pin_coupling.py` (benchmark preservation, FRC driver
  swap, explicit-override bypass).
