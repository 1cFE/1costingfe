# Paper benchmarking section — design

Date: 2026-05-05
Target file: `docs/papers/1costingfe_paper/1costingfe_paper.tex`

## Goal

Two related changes to the paper:

1. Finish and tighten the abstract. The user started a revision and left an incomplete bridge sentence ("The \textsc{1costingfe} ...") between a new pyFECONS provenance sentence and the original "principal contribution" sentence.
2. Add a new section, **Benchmarking and Cross-Validation**, that runs 1costingfe on two reference designs (ARC and ARIES-AT) and reports the predictions against the published numbers as a calibration cross-check.

The existing `todo.md` next to the paper has a "Validation against prior tools" item targeting ARIES-AT; this design retargets and expands it.

## Framing decisions (locked)

- **Reactor targets**: ARC (Sorbom et al. 2015) as headline single-point comparison; ARIES-AT (Najmabadi et al. 2006) as per-account cross-walk.
- **Framing**: calibration cross-check, not validation. Divergence between 1costingfe and the published numbers is expected and informative; the framework's pitch is procurement-grounded re-derivation that replaces heritage scaling. Where divergences appear, they are explained by pointing at the specific accounts in which procurement data departs from the heritage estimate.
- **Scope (artifacts)**: two prose case studies + one per-account cross-walk table + one stacked-bar figure of LCOE composition. No uncertainty bands or gradient/tornado work in this section (those belong to the separate "Differentiability demonstration" todo item).
- **Base year**: 2025 USD.
- **Escalation index**: BLS Consumer Price Index for All Urban Consumers (CPI-U), matching the convention used by pyFECONS (`pyfecons/pyfecons/costing/calculations/conversions.py`, sourced from usinflationcalculator.com). ARC (2015 USD) and ARIES-AT (2002 USD) escalated to 2025 USD via CPI-U. The chosen factors and source are stated in the section intro; per-cell arithmetic is shown in `%` LaTeX comments above each escalated figure for traceability.
- **1cFE program provenance**: tool was developed within the 1cFE program at the Astera Institute Residency. Public landing page: <https://1cf.energy/>. Placed in a new Acknowledgments section (not in abstract or Introduction body — keeps the abstract academic in tone). The Code Availability section also cites <https://1cf.energy/> next to the existing GitHub URL.
- **NOAK basis**: defaults represent Nth-of-a-kind costs after learning-curve effects are applied. FOAK is supported (`examples/foak_vs_noak.py`) but is not the default. Stated in the abstract only; not duplicated in the Cost Account Structure intro.

## Abstract revision

Fix the broken bridge; replace the over-broad "account-by-account re-derivation" claim with a selective-rebuild claim that the account_justification folder actually supports; add NOAK sentence; trim the JAX/`vmap` listing; add benchmarking forward-pointer. No 1cFE-program naming, no URL — those go in Acknowledgments and Code Availability instead. Final form:

> We present \textsc{1costingfe}, a fusion power plant costing framework that computes the levelized cost of electricity (LCOE) from the ground up. The code uses the Code of Accounts Structure (CAS) approach, used in the ARIES program and later formalized by the Generation IV Economic Modeling Working Group, and adopted and developed under ARPA-E funding by Woodruff and colleagues in the pyFECONS code. \textsc{1costingfe}'s principal contribution is rebuilding the cost accounts most sensitive to fuel cycle and confinement family --- buildings, magnets, drivers, direct energy converters, isotope separation, and staffing --- from current procurement data, vendor pricing, and cross-industry benchmarks, in place of the heritage scaling that dominates these accounts in prior fusion cost models. Accounts that are largely fuel-and-concept-agnostic inherit prior conventions where appropriate. Defaults represent Nth-of-a-kind costs after learning-curve effects are applied; first-of-a-kind treatment is supported. The framework spans the four candidate fuel cycles (D-T, D-D, D-${}^3$He, p-${}^{11}$B) and the principal confinement families (tokamak, stellarator, mirror, field-reversed configuration, laser and Z-pinch inertial systems, and magneto-inertial concepts). It is implemented in JAX with reverse-mode automatic differentiation, exposing exact LCOE gradients to support sensitivity analysis, Monte Carlo uncertainty propagation, and identification of technological corridors compatible with a 1 cent/kWh LCOE target. The economics, physics, and cost-account methodology are described in turn, and the framework is exercised against two reference designs, ARC and ARIES-AT, as a calibration cross-check, with divergences attributed to specific accounts where procurement data departs from heritage scaling.

Why the contribution claim was softened: the `docs/account_justification/` folder shows that genuine procurement-and-benchmark re-derivation was done for CAS21 (buildings), CAS22 (reactor components, magnets, power supplies, heating drivers, DEC, isotope separation, remote handling), CAS40 (owners costs), and CAS70 (staffing). Other accounts (CAS28 digital twin, CAS50 supplementary, CAS80 fuel, CAS22.01.04 driver methodology) inherit pyFECONS / EMWG / ARIES conventions. The original claim "an account-by-account re-derivation" reads as 100% coverage and overstates the work.

Why the JAX listing was trimmed: removed the explicit `\texttt{vmap}` listing because vectorised sweeps and Monte Carlo are *uses* of autodiff infrastructure rather than separate contributions; folded into the single "to support sensitivity analysis, Monte Carlo uncertainty propagation, and identification of technological corridors" clause.

The orphaned phrase "principal contribution is an account-by-account re-derivation..." that currently floats below the broken sentence is folded into the abstract above and removed from below.

The Conclusion (currently lines 2265-2272) contains the sentence "Cross-validation of LCOE outputs against ARIES and pyFECONS reference cases is in progress." That sentence is replaced with a forward-pointing reference to the new section.

## Introduction addition

One sentence at the end of the existing paragraph on JAX-driven sensitivity (around current line 168-178) places 1costingfe in its software ecosystem (technical context, not program framing):

> \textsc{1costingfe} is the cost-accounting layer of a broader fusion techno-economic analysis pipeline; the upstream concept-ingestion, SysML, and code-generation stages are developed separately in the \texttt{fusion-tea} repository family.

This is the only Introduction-body addition. No NOAK sentence in the Cost Account Structure intro; no 1cFE program-target language anywhere in the body.

## New Acknowledgments section

A new `\section*{Acknowledgments}` (unnumbered) is inserted between the Conclusion and the Code Availability section. Content:

> This work was performed within the 1cFE program at the Astera Institute Residency. The 1cFE program seeks to identify plausible corridors to fusion electricity at $\le \$0.01$/kWh (2025 USD); see \url{https://1cf.energy/}. \textsc{1costingfe} is the cost-accounting layer of the program's broader techno-economic analysis stack, developed alongside the \texttt{fusion-tea} repository family.

Use `\section*{...}` so it is unnumbered (standard practice for Acknowledgments) and does not perturb the appendix section numbering.

## Code Availability addition

The existing Code Availability section (currently lines 2274-2283) is extended with one sentence pointing readers at the program landing page next to the GitHub URL, and one sentence pointing at the three new benchmark scripts:

> The 1cFE program landing page is available at \url{https://1cf.energy/}. The benchmarking results in \cref{sec:benchmark} are reproduced by the scripts in \texttt{docs/papers/1costingfe\_paper/scripts/}: \texttt{benchmark\_arc.py}, \texttt{benchmark\_aries\_at.py}, and \texttt{make\_benchmark\_bars.py}.

## New section structure

Inserted between the end of "Cost Account Structure" (current line 2250) and the start of "Conclusion" (current line 2252).

```
\section{Benchmarking and Cross-Validation}
\label{sec:benchmark}

  Intro paragraph: purpose of the section is calibration cross-check.
  Restate that procurement-grounded re-derivation is expected to diverge
  from heritage-scaling estimates; the question is whether divergences
  land in the accounts where we expect them. State base year (2025 USD)
  and the BLS CPI-U deflation factors used for ARIES-AT (2002 USD) and
  ARC (2015 USD), citing the pyFECONS convention as the choice of
  index.

\subsection{ARC: Headline Comparison}
\label{sec:benchmark-arc}

  One paragraph describing the Sorbom et al. 2015 ARC reactor.
  Inline parameter table of the inputs used (geometry, power balance,
  fuel, coil technology). Reported published overnight capital,
  escalated to 2025 USD. 1costingfe predicted overnight capital, $/kWe,
  and LCOE. One to two paragraphs reconciling the delta, pointing at
  specific accounts (likely candidates: CAS22.01.03 coils because
  REBCO procurement now is much cheaper than 2015 projection;
  CAS21 buildings; CAS50 supplementary).

\subsection{ARIES-AT: Per-Account Cross-Walk}
\label{sec:benchmark-aries}

  One paragraph describing the Najmabadi et al. 2006 ARIES-AT reactor.
  Inputs table.
  Cross-walk table: rows = CAS accounts (CAS21, CAS22.01.03,
  CAS22.01.04, CAS22.01.07, CAS23-26, CAS50, CAS70, CAS80, total
  overnight, LCOE); columns = ARIES-AT published (2025 USD),
  1costingfe predicted, delta percent, dominant driver. Discussion
  paragraphs: highlight two or three most informative agreements and
  two or three most informative divergences.

\subsection{LCOE Composition}
\label{sec:benchmark-bars}

  Stacked bar figure: two side-by-side stacks (ARC | ARIES-AT) showing
  LCOE in $/MWh broken out by major CAS account groups (CAS21
  buildings, CAS22 reactor, CAS23-26 BoP, CAS50 supplementary, CAS70
  O&M, CAS80 fuel, CAS90 financial). One paragraph reading off the key
  compositional differences.
```

## Inputs to 1costingfe

### ARC (Sorbom et al. 2015, *Fusion Engineering and Design*)

| Parameter | Value | Notes |
|-----|-----|-----|
| R0 | 3.3 m | major radius |
| a | 1.13 m | minor radius |
| kappa | 1.84 | elongation |
| B0 | 9.2 T | on-axis field |
| B_max | 23 T | peak at coil |
| P_fus | 525 MW | fusion power |
| P_net (electric) | 270 MWe | net electric output |
| eta_th | 0.40 | He-cooled FLiBe blanket |
| Fuel | D-T | |
| Coils | REBCO HTS, demountable | |

Reference cost: Sorbom 2015 Section 3 quote. Exact figure read from the paper at script-write time.

### ARIES-AT (Najmabadi et al. 2006, *Fusion Engineering and Design* 80:3-23)

| Parameter | Value | Notes |
|-----|-----|-----|
| R0 | 5.2 m | |
| a | 1.3 m | A = 4.0 |
| kappa | 2.2 | |
| B0 | 5.86 T | |
| B_max | 11.4 T | |
| P_fus | 1755 MW | |
| P_net (electric) | 1000 MWe | |
| eta_th | 0.59 | He-cooled SiC composite, Brayton |
| Fuel | D-T | |
| Coils | LTS Nb3Sn | |

Reference cost: per-account table in the paper (2002 USD).

Inputs that 1costingfe requires but the source papers do not publish are defaulted to the framework's existing tokamak defaults; each such default is annotated with a one-line comment in the example script noting it is filled in.

## Code and figure pipeline

Per user direction, paper-exclusive scripts and figures live next to the paper.

New directories (created):
- `docs/papers/1costingfe_paper/scripts/`
- `docs/papers/1costingfe_paper/figures/`

New scripts:

1. `docs/papers/1costingfe_paper/scripts/benchmark_arc.py`
   - Builds `CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)`, runs `forward()` with ARC-matched inputs.
   - Prints overnight capital, $/kWe, LCOE.
   - Writes per-account JSON to `docs/papers/1costingfe_paper/scripts/_outputs/arc.json` for the figure script to ingest.

2. `docs/papers/1costingfe_paper/scripts/benchmark_aries_at.py`
   - Same shape with ARIES-AT inputs.
   - Prints per-account breakdown matching the cross-walk table rows.
   - Writes JSON to `docs/papers/1costingfe_paper/scripts/_outputs/aries_at.json`.

3. `docs/papers/1costingfe_paper/scripts/make_benchmark_bars.py`
   - Reads both JSON outputs, builds the side-by-side stacked-bar figure with matplotlib (default light theme; figures are not affected by the paper's dark-mode toggle).
   - Saves to `docs/papers/1costingfe_paper/figures/benchmark_lcoe_stacks.pdf`.

The .tex includes the figure via `\includegraphics{figures/benchmark_lcoe_stacks.pdf}` (relative to the .tex source).

The 1costingfe-predicted numbers and the escalated published numbers are pasted into the .tex source once. CEPCI deflation arithmetic is shown in a LaTeX comment (`%`) above each escalated figure for traceability.

A sentence is added to the existing `\section{Code Availability}` pointing readers at the three scripts.

## Out of scope

Explicitly not in this section, deferred to existing `todo.md` items:

- Uncertainty bands / Monte Carlo on ARC inputs (belongs to "Differentiability demonstration" item).
- Tornado plots, gradient analysis (same).
- Validation against pyFECONS or NETL baselines beyond the two reactors above.
- Stub 0D models for non-tokamak families (ARC and ARIES-AT are both tokamaks; the existing 0D tokamak appendix covers the physics layer used).
- Pipeline / dataflow figure, Sankey (separate "Figures to add" item).

## Definition of done

- Abstract reads cleanly with no broken sentences.
- New section compiles (paper builds without errors).
- All three scripts run end-to-end and produce the JSON outputs and the PDF figure.
- Cross-walk table numbers in the .tex match the script outputs (paste discipline).
- Conclusion no longer says cross-validation is "in progress".
- `todo.md` "Validation against prior tools" item is updated or removed to reflect that this section now exists.
