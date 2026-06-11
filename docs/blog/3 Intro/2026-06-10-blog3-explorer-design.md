# Design: Blog post #3 (1costingFE intro) + live forward-costing explorer

Date: 2026-06-10
Status: awaiting user approval

## Goal

Publish blog post #3 on 1cf.energy: an invitation to clone and mess around with
1costingFE, plus a description of what it does. Blog-post sized (about 1,500
words), not a report (the paper carries the full treatment). The post is backed
by two supporting artifacts: a freeze tag on the 1costingfe repo and a live,
interactive forward-costing explorer on Vercel.

## Scope boundaries

- The June 9-10 issue #4 power-to-geometry sizing work is out of scope
  everywhere: not mentioned in the post, not included in the freeze.
- A parallel session is actively developing issue #4 in the 1costingfe working
  tree. This session must not switch branches, modify tracked files, or commit
  in 1costingfe. All file work here happens in the untracked
  `docs/blog/3 Intro/` folder, in a separate `git worktree` checkout of the
  freeze commit, or in the new explorer repo.
- The old May 27 draft (`1costingFE intro.md`) is superseded; the new post is a
  rewrite from scratch, mining the old draft only for facts and links. The old
  files stay in the folder until the new post is approved.

## Deliverable 1: freeze tag

Tag commit `4c46468` ("Target factory: three-term capex...", 2026-06-09, parent
of the first issue-#4 commit) in 1costingfe. Suggested tag name: decided at
implementation time with the user (candidate: `v0.1.0-alpha.1`). Tagging does
not touch the working tree, so it is safe alongside the parallel session.

Facts at the freeze (verified against `4c46468`, must be used instead of the
old draft's stale numbers):

- 17 confinement concepts (`ConfinementConcept` enum), not 15
- 4 fuels: D-T, D-D, D-He3, p-B11
- 27 entries in `examples/`, 23 test files in `tests/`

## Deliverable 2: `1costingfe-explorer` (new repo, Vercel)

A forward-costing explorer: the real model computing live in the browser
session, not a precomputed lookup.

### Model layer: numpy-only costingfe

- Strip JAX from costingfe at the freeze tag. The existing `numpy-only` branch
  (April 10) is too stale to rebase; this is a fresh re-strip.
- Surface measured at the freeze: 8 of 19 source files import jax, almost all
  as `jnp` used as drop-in numpy; 8 call sites use `grad`/`vmap`/`jit`/`lax`
  (the `jax.lax` constructs are in `layers/tokamak.py`).
- Mechanics: `jnp` -> `np` swap; rewrite `jax.lax` control flow as plain
  Python; sensitivities via central finite differences instead of `jax.grad`.
- Validation: numerical agreement spot checks between the stripped copy and
  the frozen JAX version on a grid of (concept, fuel, slider-range) points;
  agreement tolerance decided during implementation (target: relative LCOE
  difference well below slider-visible resolution).

### Backend

- FastAPI on a Vercel Python serverless function, same pattern as
  fusion-backcasting's `api/index.py` + `vercel.json` rewrites.
- One compute endpoint: input (concept, fuel, slider values) -> output (LCOE,
  overnight capex, CAS account breakdown, finite-difference elasticities).
- Concept-specific defaults and valid concept-fuel combinations come from the
  frozen YAML configs, exposed via a metadata endpoint the frontend reads at
  load.

### Frontend

- Vite + React, lifting structure from fusion-backcasting's frontend where it
  helps.
- Controls: concept picker (17), fuel picker (valid fuels per concept),
  sliders for the big levers: net electric power, availability, WACC,
  construction time, plant lifetime, thermal-cycle / DEC efficiency.
- Outputs, all recomputed live on slider move:
  - LCOE + overnight capex headline
  - Ranked CAS cost-breakdown bar chart. Account labels in English
    ("Magnets", "Blanket & first wall"), with the CAS code (C220103) as a
    secondary label.
  - Elasticity tornado for the current design point (finite difference,
    computed server-side).

### Out of scope for the explorer (this round)

- Backcasting mode (fusion-backcasting stays as-is, untouched)
- Cost-account overrides UI, multi-module controls, Monte Carlo
- Hosted persistence, sharing links, comparison views

## Deliverable 3: the blog post

Rewrite from scratch. Model: the fusion-tea "Using the pipeline" post (short
mission framing, caveat aside, practical setup instructions, figures carrying
weight, invitation at the end). Target about 1,500 words.

### Outline

1. **Hook.** Three prior posts ran on one engine: the cost-floor post and the
   DEC post made concrete numerical claims with it, and the fusion-tea
   pipeline post ("From Papers to Plant Economics") used it as the
   deterministic costing layer under all 38 concept analyses. This post hands
   the reader that engine.
2. **What it is.** Three modules: economics (LCOE with CRF, IDC,
   growing-annuity O&M), physics (4-fuel power partitioning, radiation
   models, steady-state and pulsed power balances), cost accounts (full
   Schulte/ARIES Code of Accounts, 19 CAS22 sub-accounts, all overridable).
   17 concepts, 4 fuels, one `forward()` call. Kept short.
3. **Try it live.** Link to the Vercel explorer with one screenshot or GIF.
4. **Clone and run.** Install instructions verified against a fresh clone at
   the freeze tag; the one-call `CostModel.forward()` snippet; real terminal
   output (LCOE, ranked CAS accounts, power table). Then `jax.grad` /
   `model.sensitivity()` in a few lines with a regenerated elasticity table
   or tornado figure (one matplotlib figure, generated by a script in
   `examples/` at the freeze tag).
5. **Landscape.** One short paragraph: existing open tools are tokamak/D-T
   anchored (PROCESS, bluemira, FUSE.jl, FAROES) or parameterized cost
   walkers (pyFECONS); 1costingFE fills the cross-concept, cross-fuel,
   differentiable slot. Link to the paper for the full comparison. pyFECONS
   mentioned for account structure/landscape only, never as a source of cost
   values.
6. **What's shaky.** One condensed honest-limitations paragraph (stellarator/
   mirror sizing defaults, aneutronic BOP calibration, unvalidated pulsed
   inductive DEC). No roadmap section; the paper carries it.
7. **Invitation.** What feedback helps most, in order: reference cost data,
   account-level review of `docs/account_justification/`, validation cases,
   bug reports. Links: repo, tag, paper, explorer.

### Verification requirements before the post is publishable

- Every code snippet runs verbatim against a fresh clone at the freeze tag.
- Every number (LCOE values, elasticities, counts of concepts/fuels/examples/
  tests) regenerated at the freeze tag via the canonical example scripts, not
  inline python.
- Install instructions tested from scratch (the package is not on PyPI;
  instructions are clone + `pip install -e .` or
  `pip install git+...@<tag>`).
- All cross-links resolve (prior post URLs, paper path, explorer URL).
- Style: bare `$` in markdown, no em dashes, no tildes for approximations.

## Sequencing

1. Freeze tag (5 minutes, unblocks everything)
2. Explorer repo: numpy strip -> backend -> frontend -> deploy
3. Blog post draft in parallel with explorer development; the post's
   "Try it live" link and screenshot land last, once the deployment is up.

## Open items (resolved during implementation, with user)

- Tag name
- Explorer URL / Vercel project name
- Post title (old draft's title is a candidate, but rewrite may suggest
  a better one)
- Whether the GitHub org's repo is public at publication time and the final
  repo URL for links
