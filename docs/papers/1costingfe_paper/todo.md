# 1costingfe paper — deferred items

Items flagged by peer review that require new analysis, code runs, or new figures rather than text edits. Deferred from the May 2026 review pass.

## Figures to add

- **Pipeline / dataflow figure**: customer requirements -> physics module -> CAS accounts -> LCOE. Currently the only figure is a schematic cash-flow diagram.
- **Sample LCOE breakdown bar chart** for a reference plant (e.g. ARIES-AT D-T tokamak), broken out by CAS account.
- **Power-balance Sankey** for a representative configuration.

## Differentiability demonstration

The headline contribution is "extraction of exact gradients" via JAX autodiff, but no gradient is shown anywhere in the paper.

- Pick a reference plant.
- Compute LCOE.
- Compute partial derivatives `dLCOE/dtheta` for `theta` in {B, T_e, n_e, c_kAm, eta_th, ...} via JAX.
- Cross-check against finite differences.
- Present as a tornado plot or Pareto chart.

Without this, the paper is a methodology document, not a JAX-differentiable contribution.

## Validation against prior tools

Done as of 2026-05-06: see Section 5 ("Benchmarking and Cross-Validation") in `1costingfe_paper.tex`, with reproducible scripts under `scripts/` and the figure produced by `make_benchmark_bars.py` in `figures/`. ARIES-AT cross-walk is at top-level rollups only (Direct/Owner's/Total/LCOE) because Najmabadi 2006 publishes that granularity, not per-CAS-account detail. Per-account cross-walk against pyFECONS is left as future work.

Three `% TODO` discussion paragraphs in Section 5 are preserved for author review before publication.

## Stub appendices for non-tokamak plasma models

The abstract claims coverage of "all major confinement families and fuel cycles", but only the tokamak has a 0D physics layer. Either:

- Add stub 0D models for stellarator, mirror, FRC, IFE.
- Or soften the abstract claim ("designed to support all major confinement families; a 0D tokamak model is included as a first instantiation").

## Stub paragraphs for skipped CAS22 sub-accounts

Section 4 jumps from CAS22.01.03 to .04, .07, .09, .10, .12, skipping .01, .02, .05, .06, .08, .11. Add a one-line stub for each skipped account, or a paragraph noting "these accounts use the methodology of Woodruff (2026) unmodified".

## CAS28 cost basis

Digital twin cost ($5M) is sourced from a pyFECONS internal note. Per project rules, pyFECONS is the least-trusted source. Either find an independent benchmark or annotate as "lowest-information account, ± wide uncertainty".

## CAS22 vendor-system audit

Per project rules, vendor systems should use procurement data (not material build-ups or pyFECONS internals). Audit each CAS22 sub-account against this rule:

- CAS22.01.04 uses ITER procurement contracts: correct.
- CAS22.01.07 uses an ARIES-CS-derived figure: borderline (ARIES is not procurement). Reconcile.
- Walk the rest.
