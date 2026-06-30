# Mirror central-cell electron temperature: power-balance model and tandem-regime validation

## Summary

The 0D mirror model now solves the central-cell electron temperature `T_e` from a
self-consistent electron power balance, instead of taking it as a pinned input.
The model reproduces the warm central-cell electrons of a deep-plug D-T tandem
reactor (about 0.8 of the ion temperature, about 20 to 24 keV at a reactor design
point), validated against the Mirror Advanced Reactor Study (MARS). This document
records the physics, the closures, the validation, and the deferred follow-ons,
with citations.

## Problem

The model previously pinned central-cell `T_e = 125 keV`. That value is the
tandem PLUG hot-electron temperature, not a bulk central-cell temperature: MARS
reports plug warm electrons at 124 keV and hot barrier electrons at 840 keV, with
the central cell at 24 keV [Logan 1985; Gordon 1986]. Applying the plug value to
the bulk cell drove the optimizer to an unphysical hot D-T operating point.

## Physical picture (from the tandem-mirror literature)

Central-cell `T_e` is an output of an electron power balance, and which value it
takes depends on the electron confinement regime:

- Simple / gas-dynamic mirrors (shallow ambipolar potential, cold ends, electrons
  escape freely) run COOL electrons, about 0.1 of the ion temperature
  [Forest 2024, BEAM: T_e about 6 keV at T_i about 67 keV; Post 1987].
- Deep-plug TANDEM central cells (the electrons are electrostatically confined by
  the same deep plug that confines the ions) run WARM electrons, about 0.85 of the
  ion temperature [Logan 1985, MARS: T_e=24, T_i=28 keV, ratio 0.86; Fowler 2017:
  Tec about Tic].

The dominant electron energy SINK is thermal effusion (Pastukhov end-loss) over
the ambipolar potential: each escaping electron carries about 5 to 6 times `T_e`,
at the ELECTRON collision rate, which is about 40 to 60 times faster than the ion
rate. Post 1987 explicitly rules out Spitzer-Harm parallel heat conduction for
mirrors (it would require absurd power densities; the long mean free path makes
the loss convective, not conductive). The ambipolar potential is set by
ambipolarity (electron loss rate equals ion loss rate), which couples the electron
confinement to the ion plug depth. Thermal barriers [Baldwin and Logan 1979]
decouple the central-cell electrons from the hot plug electrons so the plug can be
heated without paying to heat the whole central cell.

## The model

Solve central-cell `T_e` from the steady-state electron power balance:

    alpha_e(T_e)*P_alpha + K_ie*(T_i - T_e) = P_brem(T_e) + P_e_endloss(T_e)
    [ alpha -> electron ]   [ equilibration ]   [ brems ]    [ electron end-loss ]

Closures:

- `P_brem`: relativistic bremsstrahlung including the electron-electron term
  [Putvinski et al. 2019; the form in Ochs et al. 2022 Eq. 16]. Uniformly valid
  from D-T (about 10 keV) to p-B11 (about 300 keV).
- `K_ie`: NRL formulary ion-electron energy equilibration [Huba, NRL Plasma
  Formulary].
- `alpha_e`: Stix two-body slowing-down fraction of alpha (charged-particle) power
  delivered to electrons [Stix 1972], multiplied by the loss-cone retention.
- `P_e_endloss`: electron Pastukhov end-loss over the ambipolar potential
  `phi_e = g_amb * T_e`, with each escaping electron carrying about
  `(g_amb + 2) * T_e`, evaluated at the ELECTRON collision time (not the ion time)
  [Pastukhov 1974; Cohen et al. 1978].
- `g_amb` is SELF-CONSISTENT via ambipolarity: the electron Pastukhov confinement
  time is matched to the model's ion confinement time (electron loss rate equals
  ion loss rate). This is a Lambert-W relation, so `g_amb` is pinned
  logarithmically and the `exp(g_amb)` sensitivity that would otherwise make the
  result knife-edge is removed. Solved with a backend-safe bounded bisection.

The ion confinement uses the existing Pastukhov / gas-dynamic collisionality
bridge. The Pastukhov enhancement `exp(e*phi/T_i)` over-credits in the deeply
collisionless or deep-plug regime; it is bounded by a smooth-min `n*tau` ceiling
at about 1e21 m^-3 s (the canonical achievable mirror Lawson scale [Fowler 2017]),
applied so that confinement times already below the ceiling are essentially
unchanged. This is an empirical bound; the physics-based replacement (a
loss-cone / DCLC instability degradation, or anomalous radial transport) is a
documented follow-on.

The Coulomb logarithm is channel-specific `lnLambda(n,T)` for the
electron-electron, electron-ion, and ion-ion channels [NRL Plasma Formulary],
replacing a folded-in constant. With a constant log the density dependence of the
two competing sinks cancels exactly; the proper logs restore the (weak) residual
density dependence.

## Validation

- D-T versus MARS: at the MARS reactor operating point (central density about
  3.3e20 m^-3, T_i = 28 keV, the model's own plug potential about 74.7 keV) the
  electron balance lands `T_e` warm, about 0.8 of `T_i`, in the MARS band, with the
  model's confinement giving an ion confinement time of about 3 s (consistent with
  the published Realta Hammir central-cell value of about 5 s [Frank et al. 2024]).
  The apparent over-credit (an ion confinement time of about 94 s) appears only
  when MARS's deeper plug potential (about 150 keV) is substituted; the model's own
  moderate plug does not over-credit at reactor density.
- p-B11 versus Ochs et al. 2022 (arXiv:2210.08076): DEFERRED. The current model
  carries a single bulk ion temperature and a single effective ion mass and lumps
  all charged-particle power into a generic alpha channel, so it cannot reproduce
  the multi-species p-B11 balance (fast and thermal protons, boron, electrons, each
  with its own slowing-down) or give a trustworthy D-He3 verdict (the D-He3 14.7
  MeV proton has a very different slowing-down than the alpha). This validation
  needs the multi-species generalization (below).

The released, user-supplied-operating-point path (use_0d_model = false) is
untouched; its bit-identical cost figure is preserved.

## Deferred follow-ons

1. Multi-species power balance: per-species temperatures and slowing-down for the
   fuel ions, the alpha, the D-He3 14.7 MeV proton, and electrons. Enables a
   trustworthy advanced-fuel verdict and the Ochs p-B11 cross-validation.
2. Physics-based confinement cap: replace the empirical `n*tau` ceiling with a
   loss-cone (DCLC) instability-enhanced loss using the `a/rho_i` diagnostic and a
   warm-fraction stabilization threshold [Kolmes et al. 2024; Post 1987], or an
   anomalous (Bohm) radial-transport term.
3. Size-from-power radius logic: the mirror is end-loss dominated, so the power
   balance pins the LENGTH; the RADIUS is set by a feasible window of constraints
   (neutron wall load and photon / surface wall load as lower bounds; DCLC
   microstability and coil bore as upper bounds; the `a/rho_i` finite-Larmor floor).
   Design at the smallest feasible radius (binding lower bound), flag infeasibility
   when the lower bound exceeds the upper bound, and report both binding
   constraints.

## Citations

- B. G. Logan, MARS (Mirror Advanced Reactor Study), 1985.
- D. T. Gordon, MARS engineering overview, 1986.
- T. K. Fowler et al., A new simpler way to obtain high fusion power gain, 2017.
- R. F. Post, The Magnetic Mirror Approach to Fusion, Nucl. Fusion 27, 1987.
- Introduction to Tandem Mirror Physics (review).
- J. Schwartz et al., MCTrans++: a 0-D model for centrifugal mirrors, 2024.
- C. B. Forest et al., Prospects for a compact break-even axisymmetric mirror
  (BEAM), 2024.
- D. E. Baldwin and B. G. Logan, thermal barrier, Phys. Rev. Lett. 43, 1979.
- V. P. Pastukhov, Nucl. Fusion 14, 1974; R. H. Cohen et al., Nucl. Fusion 20, 1978.
- S. Putvinski, D. Ryutov, P. Yushmanov, Fusion reactivity of the pB11 plasma
  revisited, Nucl. Fusion 59, 2019.
- I. E. Ochs, E. J. Kolmes, M. E. Mlodik, T. Rubin, N. J. Fisch, arXiv:2210.08076,
  2022; E. J. Kolmes, I. E. Ochs, N. J. Fisch, Phys. Plasmas 29, 110701, 2022.
- E. J. Kolmes et al., Loss-cone stabilization in rotating mirrors: thresholds, 2024.
- S. J. Frank et al., high-field axisymmetric tandem mirror (Hammir),
  arXiv:2411.06644, 2024.
- T. H. Stix, Plasma waves / fast-ion slowing-down, 1972.
- J. D. Huba, NRL Plasma Formulary (collision rates, Coulomb logarithms).
