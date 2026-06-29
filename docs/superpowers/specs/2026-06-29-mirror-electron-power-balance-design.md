# Mirror central-cell electron power balance (solve T_e)

Date: 2026-06-29
Status: design, pending review

## Problem

The 0D mirror model self-consistently settles on a physically implausible HOT
operating point for a D-T central cell (interior T_i around 23 keV, beta-cap
bound, q_eng about 1.7). The root cause is that the central-cell electron
temperature `T_e` is a pinned input. It had been set to 125 keV, which is the
tandem PLUG hot-electron temperature read onto the bulk central cell. At the
beta-limited density `n_e proportional to 1/(T_e + n_i*T_i)`, a large `T_e`
swamps the denominator, defeats the "D-T power density peaks near 14 keV" brake,
and lets raw reactivity drag the optimum hot.

The shipped YAML default has already been corrected to a near-Maxwellian
Hammir/WHAM value (T_i = T_e = 10 keV; master c34a2a8). That removes the symptom
from the released, user-supplied-operating-point path (the released path takes
density as input, so T_e only touches cost through radiation; the LCOE shift was
about 2.6 percent). It does NOT fix the gated 0D physics model, where T_e is
load-bearing through the beta-solved density. This spec fixes the physics: make
central-cell `T_e` an OUTPUT solved from an electron power balance, so the model
cannot be configured into an unphysical operating point.

## Goal and scope

Solve central-cell `T_e` from a steady-state electron power balance, following
the Fisch-group framework: Ochs, Kolmes, Mlodik, Rubin, Fisch, arXiv:2210.08076
(the user is a co-author), and the companion Kolmes, Ochs, Fisch 2022,
Phys. Plasmas 29, 110701.

In scope (this pass):
- Core THERMAL case only: no alpha channeling (eta_alpha = 0), no fast-ion
  population. The per-species balance reduces to standard collisional physics.
- Solve `T_e` for all fuels via the same closure; validate D-T first.
- Keep the existing mirror confinement physics (tau_E from Pastukhov/gas-dynamic)
  as the conduction-loss channel `P_L = U_K / tau_E`.

Out of scope (deferred):
- Alpha channeling parameters (eta_alpha, chi) and the fast/thermal-proton split.
- The full energy-integrated alpha-to-species interpolator (arXiv Appendix C);
  the D-T monoenergetic analytic split is its correct limit and is enough here.
- A more detailed electron end-loss model (ambipolar-potential-resolved, or a
  rho_e/rho_i-scaled radial electron term). The core uses the pressure-weighted
  gamma_e on p_end with zero radial electron loss.
- The `f_dec = 0.3` provenance writeup (separate task).
- Re-baselining the T_i objective. Per the agreed plan we fix T_e first, then
  reassess where T_i lands empirically before changing the T_i path.

## Architecture

Graft a per-species electron power balance onto the existing mirror 0D forward.
`T_e` becomes an inner unknown solved at each operating point; everything that
currently reads central `T_e` (radiation, beta, stored energy, tau_E numerator,
beta-solved density) then consumes the solved value.

Steady-state electron balance (eta_alpha = 0, thermal), solved for `T_e`:

    alpha_e0(T_e)*P_alpha,ret + K_ie(T_e)*(T_i - T_e) = P_B(T_e) + gamma_e*p_end(T_e)
    [ alpha -> electron ]    [ ion-electron equilib ]   [ brems ]   [ axial end-loss ]

CORRECTION to the paper's gamma_e = 0. The paper (Eq. 15) lets electrons lose
energy only through bremsstrahlung, which is valid for p-B11 (brems is the
dominant electron sink) but NOT for D-T: at D-T temperatures brems is tiny
(order 1 MW at 15 keV), so with no electron transport loss the alpha-to-electron
power (order 100 MW) would have nowhere to go but heating the electrons until
they dump it back to ions, forcing T_e to about T_i + 130 keV. That resurrects
the hot-electron bug. So electrons must carry a conduction-loss share.

The share is CHANNEL-SPECIFIC, because a mirror's P_L is two channels:
- Axial end-loss `p_end`: electrons and ions stream out the ends together
  (ambipolar), so electrons carry a real share. Pressure-weighted:
  `gamma_e = n_e*T_e/(n_e*T_e + sum n_i*T_i) = T_e/(T_e + n_i_frac*T_i)`.
- Radial cross-field transport `p_radial`: scales with gyroradius, and
  `rho_e/rho_i ~ sqrt(m_e/m_i) ~ 1/60`, so electron cross-field transport is
  negligible. This channel is ION-ONLY; electron radial loss is approximated as
  zero (a rho_e/rho_i refinement is possible later).

The mirror code already separates `p_end` (axial tau-channel) from `p_radial`
(radial tau-channel), so the split is clean. This extends the paper's own
pressure-weighting of P_L among ions (Eq. 17) to the electron axial channel.

Synchrotron / ECR stays OFF the right-hand side; it is device-specific and
largely reabsorbed (paper Sec. VIII.B), handled separately in the existing
radiation accounting.

## Closures (transcribed from arXiv:2210.08076)

All formulae are CGS as written in the paper (n in cm^-3, power density in
eV cm^-3 s^-1). The JAX kernel rescales density to 1e20 m^-3 and works in MW, so
each prefactor must fold in the unit conversion; do this once per term and unit-
test the conversion against a hand value.

1. Bremsstrahlung `P_B` (relativistic + electron-electron, paper Eq. 16):

       P_B = 7.56e-11 * n_e^2 * x^(1/2)
             * [ Z_eff*(1 + 1.78*x^1.34) + 2.12*x*(1 + 1.1*x - 1.25*x^2.5) ]
       x = T_e / 511 keV,   Z_eff = sum_i n_i Z_i^2 / sum_i n_i Z_i

   Uniformly valid D-T (x about 0.02, reduces to non-rel within a few percent)
   through p-B11 (x about 0.3). Replaces the non-relativistic Born brems on the
   electron RHS; reuse `z_eff_fuel` / `n_i_over_n_e` for the composition.

2. Ion-electron equilibration `K_ie` (paper Eq. A6, with relativistic factor R):

       K_ie = 4.8e-9 * Z_i^2 * lambda_ie * n_i * n_e * R / (m_i * T_e^(3/2))
       R from Eq. A7 (Putvinski relativistic correction; R -> 1 at low T_e)

   Power to electrons is `K_ie*(T_i - T_e)`. Fuel-weighted over ion species
   (sum over Z_j^2 n_j). Coulomb log lambda_ie: standard NRL e-i form. New code;
   the existing `compute_tau_ii` is ion-ion only and not reusable directly.

3. Alpha -> electron fraction `alpha_e0` - D-T core: the analytic monoenergetic
   (3.5 MeV) slowing-down ion/electron partition via the critical energy
   `E_crit proportional to T_e * <Z^2/A>^(2/3)`. This is the monoenergetic limit
   of the paper's energy-integrated alpha_s0 (Eq. 20, Appendix C). Compose with
   the existing mirror loss-cone RETENTION fraction `f_alpha_heat`:

       alpha_e,effective = f_alpha_heat * f_e(E_0 / E_crit(T_e))

   The paper keeps loss-cone loss as a separate term (Sec. IV.C, Eq. 28); the
   mirror retention is the multiplicative layer on top of the generic split.

4. `P_alpha,ret`: the retained alpha (charged-particle) power, already
   `state.p_alpha` in the mirror forward. For D-T the neutrons (80 percent) do
   not heat.

5. Electron conduction share: `gamma_e = T_e/(T_e + n_i_frac*T_i)` (electron
   pressure fraction) applied to the AXIAL end-loss `p_end` only. Radial loss is
   ion-only (rho_e << rho_i). NOT the paper's gamma_e = 0, which lands T_e hot
   for D-T (see the balance section above).

## The solve and its coupling

`T_e` appears in K_ie (proportional to T_e^-3/2 * R), in P_B (rising with T_e),
and in alpha_e0 via E_crit(T_e). It also feeds back through the operating point:
the beta-limited density `n_e proportional to 1/(T_e + n_i*T_i)`, beta, stored
energy W_th, and the tau_E numerator `1.5*(T_i + T_e)`. So the electron balance is
coupled to the density/beta solve, not independent.

Approach: nested root-solve. Given (T_i, B, geometry, fuel), solve `T_e` from the
electron balance with density taken from the current beta solve, iterating the
T_e <-> n_e coupling to convergence (or solve the joint 2-equation system).
Bracket `T_e` in a physical range (for example 0.1 keV to T_i) and bisect; the
balance is monotone in T_e over the bracket (heating falls, brems rises), so a
single root exists. Keep `T_e_plug` entirely separate; it drives only the
Fowler-Logan plug potential `e*phi = T_e_plug*ln(n_p/n_c)` and must not be touched.

Touch points in `src/costingfe/layers/mirror.py` (verify line numbers before
editing): radiation call (`compute_p_rad`), tau_E (`tau_E = tau_p*1.5*(T_i+T_e)/
(2*phi+T_i+T_e)`), stored energy `W_th_MW`, beta, and `_density_from_f_beta`
(`T_sum = T_e + n_i_frac*T_i`). The inverse/audit path currently bisects T_i with
T_e pinned; the T_e solve nests inside that.

## Components

New (in `mirror.py`, or a small `mirror_electron_balance` helper):
- `compute_p_brem_rel(n_e, T_e, Z_eff)` - paper Eq. 16. (Or extend
  `radiation.py` with an isolated relativistic brems that the bundled
  `compute_p_rad` can also adopt; keep it callable WITHOUT synchrotron/line.)
- `compute_K_ie(n_e, n_i, T_e, Z_i, m_i)` - paper Eq. A6 plus R (Eq. A7).
- `alpha_electron_fraction(T_e, fuel, ...)` - analytic slowing-down split.
- `solve_T_e(operating_point, fuel)` - the nested root-solve returning T_e.

Reuse:
- `state.p_alpha` (alpha charged-particle power), `ash_neutron_split`.
- `z_eff_fuel`, `n_i_over_n_e` (`reactivity.py`).
- `_density_from_f_beta` (fed the solved T_e).
- Existing synchrotron `compute_p_sync_albajar` stays in the cost-side radiation
  accounting, OUT of the electron balance.

## Validation

- Oracle (Option 1): the group's numerical model (the user can obtain it).
  Reproduce its p-B11 Fig. 3 / Fig. 4 numbers (T_e0 about 160 keV at the
  Putvinski 15 percent-boron point) with our transcribed closures, eta_alpha = 0.
- D-T sanity: with the new solve, central-cell T_e must land cool (about T_i,
  order 10 to 15 keV at the 10 keV operating point), and the self-consistent D-T
  optimum must fall back from the 23 keV hot point into a physical band.
- Anchors: GDT / WHAM operating points stay within their existing 2x validation
  band; the corrected model must not break them.
- Unit conversion: hand-check each CGS-to-(1e20 m^-3, MW) prefactor against a
  worked value.

## Testing

- `compute_p_brem_rel`: matches Eq. 16 at a tabulated (n_e, T_e, Z_eff) point;
  reduces to within a few percent of the non-rel Born value at T_e = 10 keV.
- `compute_K_ie`: matches Eq. A6 at a tabulated point; R -> 1 as T_e -> 0.
- `alpha_electron_fraction`: limits correct (cold electrons -> most alpha energy
  to electrons; hot electrons -> more to ions); monotone in T_e.
- `solve_T_e`: converges, single root in the bracket, T_e < T_i for D-T at the
  reference point.
- Integration: D-T mirror 0D forward lands at a physical operating point;
  advanced-fuel verdict re-checked (it was net-negative partly because of the
  wrong T_e and no channeling; the thermal re-check is informative, not a claim
  that advanced fuels become economic without channeling).
- Re-pin any 0D golden numbers that move, documenting why.

## Risks and notes

- Unit conversion is the most error-prone step (CGS eV cm^-3 s^-1 to MW, density
  to 1e20 m^-3). Isolate and unit-test each prefactor.
- The T_e <-> n_e coupling could be stiff near the beta cap; if naive iteration
  oscillates, solve the joint 2-equation system or damp the iteration.
- This fix changes only the gated 0D path. The released path already uses the
  corrected YAML default; no released-number change beyond what already shipped.
- Keep `T_e_plug` independent throughout; conflating it again is the original bug.
