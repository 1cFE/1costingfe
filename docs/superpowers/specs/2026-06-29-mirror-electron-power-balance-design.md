# Mirror central-cell electron power balance (solve T_e)

Date: 2026-06-29 (revised 2026-06-30 after literature validation)
Status: D-T model IMPLEMENTED and VALIDATED. The forward 0D mirror model solves
the central-cell T_e from the self-consistent ambipolar electron power balance
for D-T (solve_te=True), reproducing the MARS warm central cell (T_e about 0.8
T_i, about 24 keV at the reactor design point). The model dispatch wires
solve_te on for D-T and off for other fuels; the layer guards the deferred
boundary by raising when solve_te is requested for a non-DT fuel. The released
default path (use_0d_model=False) is untouched and its bit-identical LCOE pin
holds. Advanced fuels (multi-species alpha slowing-down constants) and the
deeply collisionless Pastukhov / DCLC physics-cap are documented follow-ons.

## Problem

The 0D mirror model settled on an unphysical hot D-T operating point because the
central-cell electron temperature `T_e` was a pinned input, set to 125 keV. That
125 keV is the tandem PLUG hot-electron temperature (MARS/Logan 1985 plug warm
electrons = 124 keV), wrongly applied to the bulk central cell. The shipped YAML
default was already corrected to 10 keV (master c34a2a8); this spec replaces the
pinned value with a solved central-cell `T_e` from a self-consistent electron
power balance.

## What the literature says (the validation anchors)

A deep-plug D-T tandem central cell runs WARM electrons, T_e ~ 0.85 T_i, set by a
power balance (MARS/Logan 1985: T_e=24, T_i=28 keV, ratio 0.86; Fowler 2017:
Tec ~ Tic). The dominant electron energy SINK is thermal effusion / Pastukhov
END-LOSS over the AMBIPOLAR electron-confining potential (each escaping electron
carries ~5-6 T_e, at the electron collision rate ~40-60x faster than ions). Post
1987 explicitly rules out Spitzer-Harm parallel conduction. The electron-confining
potential is set by ambipolarity (electron loss rate = ion loss rate), which
couples it to the ion confinement. Simple mirrors (shallow ambipolar potential,
cold ends) run cool electrons (T_e ~ 0.1 T_i, Forest 2024 BEAM); tandems with deep
plugs run warm. Thermal barriers (Baldwin-Logan 1979) decouple the central cell
from the hot plug electrons.

## Validated model (empirically reproduces MARS)

Solve central-cell `T_e` from the steady-state electron power balance:

    alpha_e(T_e)*P_alpha + K_ie*(T_i - T_e) = P_brem(T_e) + P_e_endloss(T_e)
    [ alpha->electron ]   [ equilibration ]   [ brems ]    [ electron end-loss ]

with the electron end-loss being a dedicated electron Pastukhov term over the
SELF-CONSISTENT ambipolar potential, NOT the ion convective loss gamma_e*p_end
(that earlier attempt gave ~0.2 MW and T_e ~ 350 keV).

Closures:
- `P_brem`: relativistic + e-e bremsstrahlung, Putvinski Eq. 16 (Task 1, done).
- `K_ie`: NRL ion-electron equilibration power (Task 2, done).
- `alpha_e`: Stix alpha-to-electron slowing-down fraction * f_alpha_heat retention
  (Task 3, done).
- `P_e_endloss`: electron Pastukhov end-loss over phi_e = g_amb*T_e. Each escaping
  electron carries ~(g_amb + 2)*T_e; rate is the electron Pastukhov time using the
  ELECTRON collision time tau_ee (NOT the ion compute_tau_ii). Reuse
  `compute_tau_pastukhov(tau_ee, R_m, g_amb*T_e, T_e)`.
- `g_amb` SELF-CONSISTENT via ambipolarity: electron loss rate = ion loss rate,
  i.e. tau_e_electron(g_amb, T_e) = tau_i_ion (the model's compute_tau_axial loss
  time). LambertW pins g_amb logarithmically, so the exp(g_amb) knife-edge is GONE
  (robust). The system is a coupled solve for (T_e, g_amb).

VALIDATION RESULT (empirical, scratchpad): at the MARS reactor design point
(n_e=3.3e20, T_i=28, model's own plug phi=74.7, realistic p_alpha 520 MW) this
lands T_e = 24.4 keV = 0.87 T_i, matching MARS. The model's own confinement gives
tau_p ~ 3 s there (matching the published Hammir tau_c ~ 5 s), NOT the 94.5 s that
a substituted deep MARS plug (phi=150) produced.

## Additional scope (decided 2026-06-30)

1. Proper Coulomb logarithm. Replace the folded-in constant `_LN_LAMBDA = 17.0`
   with channel-specific lnLambda(n, T) for ee / ei / ii (NRL forms). With a
   constant lnLambda the n^2 in K_ie and the end-loss cancel EXACTLY, making T_e
   spuriously density-independent; the real lnLambda(n,T) and the distinct
   ee/ei/ii logs break that cancellation and restore the weak density dependence.
   Touches compute_tau_ii, compute_K_ie, and the new tau_ee.

2. Operating-point fix. The released 0D default operating point (n_e=5e19,
   T_i=15) is thin and low-T, where the Pastukhov exp(ephi/T_i) over-credits
   (tau_p ~ 88 s) and T_e lands hot (1.59 T_i). Move the 0D default toward a
   realistic reactor density and/or cap/correct Pastukhov in the deeply
   collisionless regime (pastukhov_valid = 0), so the default operating point
   gives the warm, MARS-consistent T_e.

## Out of scope / known separate over-credits

- Deep-plug (phi >~ 150 keV, MARS-class) Pastukhov over-credit. The model's own
  phi=74.7 avoids it; deeper plugs need a Pastukhov cap (separate work).
- Flat-profile `fusion_power` gives ~1.8x too much p_alpha (948 vs 520 MW at the
  MARS point) - a pre-existing model issue, separate from this fix.
- Alpha channeling (eta, chi) and the fast-ion population (advanced-fuel extension).
- Re-baselining the T_i objective (keep T_i fixed; solve only T_e).

## Components

New functions (mirror.py):
- `compute_tau_ee(...)` - NRL electron collision time, float32-safe pre-folded
  prefactor (ratio tau_ii/tau_ee = 60.8*sqrt(A)).
- `compute_p_e_endloss(T_e, g_amb, ...)` - electron Pastukhov end-loss [MW].
- `solve_g_amb_ambipolar(T_e, ...)` - LambertW solve of tau_e_electron = tau_i_ion.
- `solve_T_e(...)` - coupled (T_e, g_amb) solve. REPLACES the broken d64b1ee
  solve_T_e (gamma_e*p_end).
- Coulomb-log helper(s) lnLambda_ee/ei/ii(n, T).

Reuse: compute_p_brem_rel, compute_K_ie, alpha_electron_fraction (done),
compute_tau_pastukhov, compute_tau_axial, _confinement_and_losses (Task 4, done).

## Validation

- D-T vs MARS: at the reactor design point, T_e ~ 0.87 T_i ~ 24 keV.
- p-B11 vs Ochs et al. arXiv:2210.08076: single-region (electron end-loss off),
  reproduce T_e0 ~ 160 keV at the Putvinski 15%-boron breakeven. Validates
  brems + K_ie + alpha-split independent of the tandem end-loss.
- Robustness: LambertW g_amb is logarithmic (no knife-edge); confirm.

## Citations

MARS: Logan 1985 (Nucl. Fusion / Fusion Technol.), Gordon 1986. Fowler 2017.
Post 1987 "The Magnetic Mirror Approach to Fusion". "Introduction to Tandem
Mirror Physics". Schwartz 2024 (MCTrans++). Forest 2024 (BEAM). Baldwin & Logan
1979 (thermal barriers). Pastukhov 1974; Cohen et al. 1978. Ochs/Kolmes/Mlodik/
Rubin/Fisch arXiv:2210.08076; Kolmes/Ochs/Fisch 2022 PoP 29 110701. All in Zotero.
