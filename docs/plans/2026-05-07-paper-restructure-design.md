# Paper restructure -- design

Date: 2026-05-07
Target file: `docs/papers/1costingfe_paper/1costingfe_paper.tex`

## Goal

Restructure the paper so it serves three roles cleanly:

1. **Introduce 1costingfe** to readers who have not seen it.
2. **Show the calculations** (economics, physics, cost accounts) in a human-readable way.
3. **Show how to use it**, so other papers (e.g. `cost_floor_dec.tex` and follow-on Astera work) can cite this paper as the canonical methodology + usage reference.

Goals 1 and 2 are largely covered by the current draft. Goal 3 is the dominant gap: the abstract claims "exact LCOE gradients via JAX autodiff" but no gradient is shown anywhere; the Code Availability section is one URL paragraph; there is no worked input-to-output example in the body.

This restructure adds a new Section 2 that walks a worked example from physics outputs to LCOE, includes the headline gradient demo, and tightens the surrounding sections so the new section can stand at the front without bloating the paper.

## Framing decisions (locked)

- **Headline use case**: external physics model -> 1costingfe -> LCOE. The 0D tokamak in Appendix A is reframed as an included default for users without their own physics, not the canonical entry point. The pitch is "everyone has a 0D tokamak; we provide the costing layer that consumes physics outputs."
- **Reference scenario for the worked tour (Section 2)**: synthetic 1 GWe D-${}^3$He steady-state mirror with venetian-blind DEC, NOAK defaults, $A=0.85$, $n=30$ yr, $T_c=6$ yr. Deliberately not a tokamak: the framing is "everyone has a 0D tokamak; we provide the costing layer that consumes physics outputs from any concept." The mirror+D-${}^3$He+DEC choice exercises (a) mirror coil geometry distinct from tokamak (16$\pi$ path factor vs. 4$\pi^2$), (b) the D-${}^3$He secondary-burn structure documented in current Section 3.1.3, and (c) the electrostatic-DEC cost basis in CAS22.01.09. Stays steady-state so the worked tour does not require Appendix C. Distinct from the D-T tokamak benchmarks in Section 6 to avoid double-coverage.
- **Non-tokamak physics layers**: a 0D mirror physics model is in flight (`docs/plans/mirror_physics_model.md`), paralleling `layers/tokamak.py`. The abstract is left as-is on coverage. Appendix D (0D mirror model) is anticipated in the structure but kept out of this restructure's scope; it slots in after the mirror physics work merges. The paper does not headline mirror as a contribution; the editorial line is "1costingfe is concept-agnostic, exercised here on a non-tokamak case study to show it." Magic-number defaults inherited from older sources (e.g. the former hardcoded $n_{\text{coils}}=4$ for mirror) are replaced with parameter-exposed defaults during related code work, not as part of this restructure.
- **Code style in body**: short verbatim listings (3-5 lines) for the API surfaces in Section 2. Adds the `listings` package to the preamble.
- **CAS chapter compression**: prose subsections kept only for accounts that 1costingfe extends or replaces relative to prior conventions. Inherited accounts collapse to one-line stubs in the overview table or short clusters of paragraphs.
- **No "earlier revisions said X" framing in `paper.tex`**: per project rule, the paper is forward-looking only. Restructure rationale lives in this design doc, not in the paper.

## Top-level structure (target)

```
Abstract                    [revised: add worked-tour + gradient-demo sentence]

1. Introduction             [largely kept; one new paragraph anchoring Sec. 2]

2. From customer requirements to LCOE: a worked tour     [NEW]
   2.1 Pipeline overview (figure)
   2.2 Worked forward call from physics outputs
   2.3 Sensitivity tornado from JAX autodiff
   2.4 Other API surfaces (overrides, sweeps, backcasts)

3. Economics module                  [kept; one-line NOAK note]
4. Physics module                    [tightened: pulsed + inverse balance to App. C]
5. Cost account structure            [restructured: 'Disposition' column;
                                      prose only for Extended/Replaced accounts;
                                      stubs for Inherited; CAS28 + CAS22.01.07
                                      sourcing fixes]
6. Benchmarking and cross-validation [TODO discussion paragraphs finished]
7. Conclusion                        [updated: gradient demo no longer "deferred"]

Code Availability                    [pointers to new examples + scripts]

Appendix A: 0D Tokamak model         [kept; one-paragraph reframing top]
Appendix B: Synchrotron model        [kept]
Appendix C: Pulsed and inverse balances   [NEW; moved from Sec. 3]
```

## Section 2 design (the new chapter)

### 2.1 Pipeline overview

One TikZ figure showing the dataflow from external physics model into 1costingfe and out to LCOE. Five conceptual columns:

1. **External physics model** (user-owned, possibly proprietary): produces $P_{\text{fus}}$, concept-specific geometry, plasma profiles $(n_e, T_e)$ or $(n_e, T_i)$, magnetic field configuration, thermal efficiency $\eta_{\text{th}}$, and (for DEC concepts) $\eta_{\text{dec}}$ and $f_{\text{dec}}$.
2. **1costingfe physics module**: consumes whatever subset the user provides, computes radiation, recirculating power, thermal/electric conversion (with optional DEC bypass), and the inverse balance for the target $P_{\text{net}}^*$.
3. **Engineering sizing** (radial build for tokamaks, axial build for mirrors, volumes), implied by Sections 3-4.
4. **CAS accounts** (CAS10-CAS90).
5. **LCOE** in $/MWh.

Lateral arrow: vendor quotes / known costs feed in as `cost_overrides` at the CAS-account level.

Two short paragraphs accompany the figure. First paragraph: which inputs are physics-output engineering parameters (geometry, $n_e$, $T_e$ or $T_i$, $B$, $\eta_{\text{th}}$, $\eta_{\text{dec}}$, $f_{\text{dec}}$, secondary burn fractions for catalyzed cycles) and which are costing parameters (WACC, $T_c$, NOAK switch, conductor $/kA-m, ${}^3$He market price, etc.). Second paragraph: 1costingfe is a thin physics layer over a thick costing layer; users supply whichever physics outputs they have, and the framework backs out the rest from defaults. The concept-specific 0D models in Appendix A and follow-on appendices are convenience layers for users without an external physics model in hand.

Render as TikZ to stay consistent with the existing cashflow figure (no PDF asset required).

### 2.2 Worked forward call from physics outputs

Synthetic reference: 1 GWe D-${}^3$He steady-state mirror with venetian-blind DEC. Physics outputs to hand off (illustrative):

| Quantity | Value | Notes |
|---|---|---|
| Concept | Mirror | linear, tandem-style |
| Fuel | D-${}^3$He | aneutronic primary; D-D side reactions |
| $P_{\text{fus}}$ | 2400 MW | from external physics |
| $L$ | 80 m | central cell length |
| $r_p$ | 0.4 m | plasma radius |
| $B_{\text{cc}}$ | 6 T | central cell field |
| $B_{\text{mirror}}$ | 12 T | mirror coil field |
| $T_i$ | 70 keV | per existing Section 3.1.3 example |
| $n_e$ | $3.3 \times 10^{19}$ m$^{-3}$ | per existing Section 3.1.3 example |
| $\eta_{\text{th}}$ | 0.40 | thermal cycle on residual neutrons + bremsstrahlung |
| $\eta_{\text{dec}}$ | 0.70 | venetian-blind direct conversion efficiency |
| $f_{\text{dec}}$ | 0.90 | charged-particle collection fraction (loss-cone capture) |
| $f_T^*$ | 0.5 | secondary D-T burn (steady-state mirror default) |
| $f_{{}^3\text{He}}^*$ | 0.1 | secondary D-${}^3$He burn (steady-state default) |
| $f_{\text{DD}}$ | 0.131 | per existing Section 3.1.3 example |

Numerical values are illustrative; the script that produces them (`examples/external_physics_handoff.py`) is the authoritative source. The point of the table is the handoff pattern, not the design.

Forward call (verbatim listing in body):

```python
from costingfe import CostModel, ConfinementConcept, Fuel
model  = CostModel(concept=ConfinementConcept.MIRROR, fuel=Fuel.DHE3)
result = model.forward(
    net_electric_mw=1000.0, availability=0.85, lifetime_yr=30,
    p_fus=2400.0,
    cell_length=80.0, plasma_radius=0.4,
    bt_cc=6.0, bt_mirror=12.0,
    t_i=70.0, n_e=3.3e19,
    eta_th=0.40, eta_dec=0.70, f_dec=0.90,
    f_T_secondary=0.5, f_He3_secondary=0.1,
)
```

(Exact kwarg names are validated against the framework API during implementation; this listing reflects the design intent and may need touch-ups to match `CostModel.forward` signature.)

Followed by a group-level rollup table:

| Group | M$ | Notes |
|---|---|---|
| CAS10 | ... | Pre-construction (D-${}^3$He: 0.75 yr licensing, low jurisdictional burden) |
| CAS21 | ... | Buildings (light enhanced industrial grade for D-${}^3$He) |
| CAS22 | ... | Reactor plant equipment |
| CAS23-26 | ... | Turbine, electric, misc, heat rejection |
| CAS27-30 | ... | Special materials, digital twin, contingency, indirect |
| CAS40 | ... | Owners costs |
| CAS50 | ... | Supplementary |
| CAS60 | ... | IDC |
| Total overnight | ... | $/kW: ... |
| CAS70 (annualized O&M) | ... | M$/yr |
| CAS80 (annualized fuel) | ... | M$/yr -- ${}^3$He market dominates |
| CAS90 (annualized financial) | ... | M$/yr |
| **LCOE** | ... | $/MWh |

Numbers populated by the new `examples/external_physics_handoff.py` script. Caption: "Canonical 1 GWe D-${}^3$He mirror with venetian-blind DEC, NOAK reference. Reproduced by `examples/external_physics_handoff.py`."

This table is the `\label`ed object that other papers cite as "see Table X for the canonical breakdown of a 1 GWe D-${}^3$He mirror NOAK reference."

### 2.3 Sensitivity tornado from JAX autodiff

Headline figure of the section. Tornado plot of LCOE elasticities $\epsilon_p = (\partial\text{LCOE}/\partial p)(p/\text{LCOE})$ over a deliberately mixed parameter set, visually grouped into three colored bands:

1. **Physics outputs**: $P_{\text{fus}}, T_i, n_e, B_{\text{cc}}, B_{\text{mirror}}, \eta_{\text{th}}, \eta_{\text{dec}}, f_{\text{dec}}, f_T^*, f_{{}^3\text{He}}^*$, plant availability $A$.
2. **Cost unit prices**: conductor $c_{\text{kAm}}$ (mirror coils, simpler 2.5x markup vs. tokamak's 8x), DEC collector $/m$^2$ or $/MW collected, building $/m$^2$, staffing $/FTE-yr, ${}^3$He market price (dominant for D-${}^3$He fuel cycle).
3. **Financial / methodology**: WACC $i$, construction time $T_c$, lifetime $n$, NOAK contingency, CAS30 indirect-cost fraction.

Caption gives the absolute $\Delta$LCOE in $/MWh from a +1% perturbation per parameter, so the example "magnets at $X$/kA-m, +1% -> +Y % LCOE" reads directly off the figure. The mirror choice naturally surfaces ${}^3$He price as a top entry, which is informative: it tells the reader that for an aneutronic mirror, fuel-market dynamics rival reactor-engineering parameters in LCOE leverage.

Body discussion: one paragraph identifying the top finding from each bucket. Reader takeaway: if the physics is trusted, here is the cost-side parameter that dominates the residual uncertainty; conversely, if the cost basis is trusted, here is the physics-output parameter where simulation effort is most valuable. The DEC-specific entries ($\eta_{\text{dec}}, f_{\text{dec}}$, DEC $/m$^2$) make explicit how much LCOE leverage the venetian-blind architecture exerts relative to the surrounding cost basis.

Sidebar: 5-row table of autodiff vs. centered finite-difference values for the top elasticities, demonstrating exactness ("autodiff is not an approximation").

Figure produced by a new script `scripts/make_tornado.py`. Output `figures/tornado.pdf`.

### 2.4 Other API surfaces

Three short verbatim listings (3-5 lines each), one motivating sentence per. Examples are mirror-flavored to stay consistent with Section 2.2.

**Cost override** (vendor knows the mirror coil cost):

```python
result = model.forward(
    net_electric_mw=1000.0, availability=0.85, lifetime_yr=30,
    cost_overrides={"C220103": 80.0},   # M$, mirror coil quote
)
```

Sentence: "Use when a procurement quote replaces a parametric estimate; downstream rollups (CAS22, total capital, LCOE) recompute automatically. Mirror coils are typically smaller absolute cost than tokamak TF/CS/PF systems, reflecting the simpler solenoid topology."

**Batch sweep** over a physics-output uncertainty band:

```python
lcoes = model.batch_lcoe(
    {"f_T_secondary": [0.3, 0.4, 0.5, 0.6, 0.7]},
    base_params=result.params,
)
```

Sentence: "Vectorised over JAX `vmap`; suitable for uncertainty-band propagation and Monte Carlo. Here the swept parameter is the secondary D-T burn fraction, which controls how much of the D-${}^3$He plant's neutron load comes from D-D-bred tritium."

**Backcasting** on a single lever:

```python
from costingfe.analysis.backcast import backcast_single
eta_dec_target = backcast_single(
    model, target_lcoe=60.0, param_name="eta_dec",
    param_range=(0.50, 0.85), base_params=result.params,
)
```

Sentence: "Solves the inverse problem: which value of a single parameter hits a target LCOE, given everything else fixed. Here the question is what venetian-blind efficiency would be required to bring the mirror plant to a 60 $/MWh LCOE."

## Section 3 (Economics) -- changes

Insert one sentence near the start clarifying that defaults are NOAK and that the financial knobs ($i$, $T_c$, $n$) appearing in the Section 2.3 tornado are defined here. No structural changes.

## Section 4 (Physics) -- moves

**Stays in main text**: fusion power split (4 fuels, all subsubsections), thermal and electric power, recirculating power and net electric output.

**Moves to new Appendix C**:

- Section 3.4 *Pulsed Power Balance* (per-pulse framework, pulsed thermal, pulsed inductive DEC).
- Section 3.5 *Inverse Power Balance* (Newton iteration with analytical Jacobian, pulsed-inverse closed form).

**Cuts**: the forward-looking paragraph "A more involved power plant thermodynamics model could be implemented in the future" goes; ambient text not needed for parsimony.

Net effect: Section 4 contracts to roughly 60-65% of current length, narrative is steady-state-only.

## Section 5 (CAS) -- restructure

The CAS overview table (currently Table 2) gains a **Disposition** column with three values:

- *Inherited*: uses prior conventions (pyFECONS, ARIES, Gen-IV EMWG) without modification.
- *Extended*: same backbone, additional sub-cases (e.g. fuel-dependent scoping).
- *Replaced*: re-derived from procurement / first principles.

Subsection prose is kept only for *Extended* and *Replaced* accounts. Inherited accounts collapse to one-line stub entries in the overview table. The CAS22.02-.07 cluster gets one short joint paragraph.

Disposition assignments:

| Account | Disposition | Prose? |
|---|---|---|
| CAS10 | Replaced | Yes (compact siting, Part 30) |
| CAS21 | Replaced | Yes (per-building, per-fuel) |
| CAS22.01.01 (FW + blanket) | Inherited | Stub |
| CAS22.01.02 (shield) | Inherited | Stub |
| CAS22.01.03 (coils) | Replaced | Yes |
| CAS22.01.04 (driver) | Extended | Yes |
| CAS22.01.05 (structure) | Inherited | Stub |
| CAS22.01.06 (vacuum) | Inherited | Stub |
| CAS22.01.07 (power supplies) | Replaced | Yes -- see vendor-source note below |
| CAS22.01.08 (divertor) | Inherited | Stub |
| CAS22.01.09 (DEC) | Replaced | Yes |
| CAS22.01.10 (RH&M) | Replaced | Yes |
| CAS22.01.11 (install) | Inherited | Stub |
| CAS22.01.12 (isotope sep) | Replaced | Yes |
| CAS22.02-.07 | Inherited | One short joint paragraph |
| CAS23-26 | Replaced (NETL-calibrated) | Yes |
| CAS27 | Extended | Short |
| CAS28 | Replaced (single source) | Short, with sourcing flag (see below) |
| CAS29 | Inherited | Stub |
| CAS30 | Replaced | Yes |
| CAS40 | Replaced | Yes |
| CAS50 | Replaced | Yes |
| CAS60 | Inherited | Stub |
| CAS70 | Replaced | Yes |
| CAS80 | Replaced | Yes |
| CAS90 | Inherited | Stub |

Two sourcing fixes the project rules require:

- **CAS28** (digital twin, $5M from a pyFECONS internal note): annotate as "lowest-information account; wide uncertainty band; the figure is a placeholder pending an independent benchmark." Per project rule, pyFECONS is the least-trusted source.
- **CAS22.01.07** (power supplies): the ARIES-CS-derived figure is borderline because ARIES is not a procurement source. Either substitute a procurement reference, or annotate explicitly that this sub-account is not yet on procurement footing and flag for future revision. Concretely: add one paragraph at the end of the CAS22.01.07 subsection acknowledging this and stating the substitute reference path. Do not invent a procurement number.

Net effect: roughly 15-20% reduction in CAS chapter length; the accounts that remain feel like the reason the contribution claim ("rebuilt from procurement") is true, instead of getting drowned among inherited accounts.

## Section 6 (Benchmarking) -- finish three TODO paragraphs

Three `% TODO` discussion paragraphs in the body (ARC delta, ARIES-AT cross-walk, LCOE composition). Drafts already in the file lay out the substance; the editing task is to lift the markers and tighten. Per the "paper.tex no history" rule, the prose is the current explanation, no "earlier draft said X."

Explicit granularity statement (already partially present, kept intact): Najmabadi 2006 publishes only EMWG top-level rollups, so per-CAS-account cross-walk against pyFECONS is left for future work.

## Section 7 (Conclusion) -- minor edits

- Remove the line "Gradient-enabled use cases (sensitivity tornados, target-driven inverse design, autodiff-versus-finite-difference comparisons) are supported by the framework but are not demonstrated in the present text." Now demonstrated in Section 2.
- Keep the limitation on non-tokamak physics layers and the future-work pointer to per-account pyFECONS cross-validation.

## Abstract -- one edit

**Add a worked-tour sentence** near the end (after the JAX/autodiff sentence): `"A worked tour from physics outputs to LCOE, including JAX-derived sensitivities across physics, cost, and financial parameters, is presented as the canonical use case."`

The existing coverage claim ("It spans the several confinement families across magnetic, inertial, and magneto-inertial confinement schemes") is left in place. The 0D mirror physics work in flight will catch the abstract up to that claim by the time the paper is submitted; this restructure does not gate on it.

## Appendix C (new) -- pulsed and inverse balances

New appendix collecting:

- Per-pulse energy framework.
- Pulsed thermal conversion.
- Pulsed inductive DEC conversion.
- Inverse power balance (steady-state and pulsed).
- Newton iteration with analytical Jacobian.

Equations and labels carry over verbatim from the current Section 3.4-3.5; section/subsection demotion only. Cross-references in Section 4, Section 5 (CAS22.01.07 capacitor-bank costing), and Section 2 update to point at App. C labels.

## Code Availability -- additions

Append two lines pointing at:

- `examples/external_physics_handoff.py`: produces the Section 2.2 reference numbers.
- `scripts/make_tornado.py`: produces the Section 2.3 figure and the autodiff-vs-finite-difference sidebar.

## Introduction -- one new paragraph

After the existing roadmap paragraph (`The remainder of this paper is organized as follows...`), one short paragraph anchoring Section 2 as the worked tour entry point and noting that the 0D tokamak in Appendix A is included for users without their own physics. Two sentences max.

## New artifacts to create

| Artifact | Path | Produced by |
|---|---|---|
| Pipeline figure (TikZ) | inline in `1costingfe_paper.tex` | n/a (TikZ in source) |
| Reference rollup table | inline | numbers from `examples/external_physics_handoff.py` |
| Tornado figure | `figures/tornado.pdf` | `scripts/make_tornado.py` |
| Autodiff vs. FD sidebar table | inline | same script |
| External physics handoff example | `examples/external_physics_handoff.py` | new |
| Tornado script | `docs/papers/1costingfe_paper/scripts/make_tornado.py` | new |

The example script lives in the package `examples/` directory (so users can run it from a clean install). The figure-generation script lives next to the paper, alongside the existing `benchmark_*.py` scripts.

## Out of scope

Explicitly **not** in this restructure:

- Stub 0D physics models for stellarator, mirror, FRC, IFE. Out per the framing decision; abstract is softened instead.
- Per-CAS-account cross-walk of ARIES-AT against pyFECONS. Belongs to a separate future work item.
- Substituting a real procurement reference for CAS22.01.07 power supplies. Restructure annotates the gap; closing it requires sourcing work.
- Substituting a non-pyFECONS reference for CAS28 digital twin. Restructure annotates uncertainty; closing it requires benchmark work.
- Decomposing Section 2 into a separate companion paper. Considered and rejected; single-citation simplicity wins.

## Implementation notes

This design will be turned into a step-by-step implementation plan in a follow-up document. High-level ordering is expected to be:

1. Add the `listings` package and TikZ pipeline figure to the preamble + Section 2 skeleton.
2. Write `examples/external_physics_handoff.py`; populate the Section 2.2 rollup table with the numbers it produces.
3. Write `scripts/make_tornado.py`; verify autodiff-vs-FD agreement; generate `figures/tornado.pdf`; populate Section 2.3.
4. Write Section 2.4 listings.
5. Move pulsed and inverse-balance content to the new Appendix C; update cross-refs.
6. Compress the CAS chapter against the disposition table; collapse inherited subsections; add CAS28 + CAS22.01.07 sourcing notes.
7. Finish the three benchmarking TODO paragraphs.
8. Apply abstract, conclusion, intro edits.
9. Build the paper, walk the diff, sanity-check that section numbering and `\cref{}` targets all resolve.
