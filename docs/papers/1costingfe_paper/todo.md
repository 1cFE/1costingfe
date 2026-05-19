# 1costingfe paper — deferred items

Items flagged by peer review that require new analysis, code runs, or new figures rather than text edits. Deferred from the May 2026 review pass.

## Figures to add

- **Power-balance Sankey** for a representative configuration.

## Stub appendices for non-tokamak plasma models

The abstract claims coverage of "all major confinement families and fuel cycles", but only the tokamak has a 0D physics layer. Either:

- Add stub 0D models for stellarator, mirror, FRC, IFE.
- Or soften the abstract claim ("designed to support all major confinement families; a 0D tokamak model is included as a first instantiation").

## Stub paragraphs for skipped CAS22 sub-accounts

Section 4 still skips CAS22.01.02 (shield), .05 (primary structure), .06 (vacuum system), and .08 (divertor). These are bundled into the "Inherited Sub-Accounts" paragraph as pyFECONS-unmodified. Add a one-line stub for each, or expand the inherited paragraph to call out each by name.

## CAS22 vendor-system audit

Per project rules, vendor systems should use procurement data (not material build-ups or pyFECONS internals). Audit each CAS22 sub-account against this rule:

- CAS22.01.04 uses ITER procurement contracts: correct.
- CAS22.01.07 uses an ARIES-CS-derived figure: borderline (ARIES is not procurement). Reconcile.
- Walk the rest.
