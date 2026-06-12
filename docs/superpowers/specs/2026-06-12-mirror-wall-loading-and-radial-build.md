# Mirror Wall-Loading Constraint and Radial-Build Consistency

**Date:** 2026-06-12
**Status:** Approved design, pending implementation
**Settled decisions (user, 2026-06-12):** (a) throat-derived plug bore;
(b) SizingInfeasible when the wall cap defeats the target; (c) surface
heat flux is a full constraint with a sourced q_surface_max; (d1)
fluence-based core lifetime applies to all MFE concepts at once.
**Depends on:** 2026-06-11-mirror-0d-sizing-design.md (Implemented)

## Motivation

Two findings from the post-merge audit of the mirror 0D sizing workline:

1. **Radial-build contradiction in the coil account.** The mirror YAML's
   radial build stacks to a vessel outer radius of 3.20 m (plasma 1.5 +
   vacuum 0.10 + first wall 0.05 + blanket 0.80 + reflector 0.20 + HT
   shield 0.20 + structure 0.15 + gap 0.10 + vessel 0.10), but C220103
   costs both coil classes at `r_bore: 1.85` m. A 1.85 m bore cannot
   enclose a 3.20 m vessel. The paper states the coil bore is taken from
   the radial build; that is true for the tokamak calibration but not for
   the mirror value, which is an independent untraced constant (same
   provenance family as the parked r_bore recalibration issue).

2. **Unconstrained wall loading in sizing mode.** At the beta-boundary
   density both P_fus and A_fw scale linearly in L, so neutron wall
   loading is L-independent: q_wall is proportional to a * n^2, set by
   radius and f_beta alone. The LCOE-over-f_beta optimizer therefore
   rides f_beta to the upper bound (LCOE is monotone decreasing in
   f_beta under the current cost structure) and drags wall loading
   through the 5 MW/m^2 advisory threshold (13.3 MW/m^2 at f_beta 0.95
   on the default machine). Nothing in the solve pushes back.

A third, minor item: the 0D diagnostic computes wall loading on the bare
plasma surface (2 pi a L) while the geometry layer that feeds every CAS22
volume puts the first wall at vacuum_or = a + vacuum_t (1.6 m vs 1.5 m at
defaults, a 6.7% difference). Once wall loading becomes a constraint
rather than a diagnostic, the two must agree.

## Part 1: Coil bore from the radial build

### Central-cell solenoids

The central-cell coil bore is derived from the radial build instead of
the YAML `r_bore`:

    r_bore_central = vessel_or(radial build) + coil_standoff

with `coil_standoff` a new YAML key (assembly gap between vessel outer
surface and winding pack; default 0.10 m, matching the gap2_t convention).
At YAML defaults: r_bore_central = 3.20 + 0.10 = 3.30 m.

The mirror coil markup is recalibrated ONCE so C220103 at the YAML
default machine still equals 513.375 M$ (calibration-neutrality
construction, same procedure as the two-class introduction; the markup
absorbs the bore change by design). Existing pins (LCOE 93.643616,
C220103 513.375) must hold bit-identically. What changes is the
sensitivity structure: coil cost now responds to blanket_t and the rest
of the radial build, which is physically correct (thicker blanket means
larger, costlier magnets).

### Plug coils

The plug coils sit at the throat where the machine necks down and the
blanket does not extend. Flux conservation gives the throat plasma
radius:

    a_throat = a / sqrt(R_m)        (1.5 / sqrt(10) = 0.47 m at defaults)

**DECISION (a), settled 2026-06-12 (user):** throat-derived plug bore.

    r_bore_plug = a_throat + plug_standoff

with `plug_standoff` a new YAML key covering vacuum gap, throat
structure, and cryostat (no blanket at the throat). Default 0.30 m,
giving r_bore_plug of about 0.77 m (WHAM-like proportions; their 17 T
HTS coils have winding-pack bores well under 1 m).

The structural picture this encodes, unlike a tokamak's single coil
class: the plug/throat coils sit at SMALLER radius than the central-cell
plasma (flux conservation necks the plasma down; no blanket to clear),
while the central-cell solenoids sit at LARGER radius than the plasma
(outside the full radial build). Small-bore/high-field plug coils versus
large-bore/low-field solenoids is the cost-relevant distinction.

The plug-coil kAm drops sharply versus today (bore enters squared:
(0.77/1.85)^2 is about 0.17), which shifts cost weight toward the
central cell. The one-time markup recalibration absorbs the total, but
the plug/central split and therefore the L-scaling slope change. Tests
pin the new split.

## Part 2: Wall-loading constraint in sizing mode

### Constraint formulation

`q_wall_max` becomes a YAML key (default 5.0 MW/m^2, replacing the
hardcoded `_WALL_LOADING_MAX`; the non-sizing inverse keeps
warning-severity semantics, now reading the same key).

In sizing mode the density ceases to be set by beta alone. Inside the
per-T_i evaluation of `net_electric_at_L`:

    n_beta(T)  = f_beta * beta_max * B^2 / (2 mu_0 (T_e + (n_i/n_e) T_i) KEV_TO_J)
    n_wall(T)  = sqrt( q_wall_max * 2 / (f_n * C_fus(T) * a_fw_ratio * a) )
    n_e(T)     = min(n_beta(T), n_wall(T))

where C_fus(T) is the per-n_e^2 fusion power density coefficient at
temperature T (from the existing reactivity kernel), f_n the neutron
fraction from ash_neutron_split, and the closed form follows from
q_wall = f_n * C_fus * n^2 * V / (2 pi a_fw L) with V = pi a^2 L (the L
cancellation that makes the constraint well-posed). Both branches are
smooth in T and the min() is JAX-compatible (jnp.minimum); the GSS over
T_i and the L bisection proceed unchanged.

Consequences:

- The f_beta optimizer stops riding to the bound once the wall-load
  branch binds; the LCOE minimum becomes interior or saturates exactly
  at the constraint, which is the honest answer.
- For machines where the wall-load branch binds at all T, beta drops
  below f_beta * beta_max; the state reports which constraint bound
  (diagnostic flag, e.g. `density_limited_by: "beta" | "wall"`,
  reported like confinement_regime, not stored in the JAX dataclass).
- Forward and inverse (audit) modes are unchanged except the warning
  threshold reads `q_wall_max` from YAML.

**DECISION (b), settled 2026-06-12 (user):** when even n_wall yields an
infeasible target (cannot reach P_net at L_max under the cap), raise
SizingInfeasible with the wall-load cap named in the message. No silent
fallback to beta-limited density.

### First-wall radius basis

The wall-loading area moves from the plasma surface to the costed first
wall:

    A_fw = 2 pi (a + vacuum_t) L

consistently in MirrorPlasmaState (`fw_area`, `wall_loading`) and in the
n_wall closed form (the `a_fw_ratio = (a + vacuum_t)/a` factor above).
The 0D dispatch passes vacuum_t from the radial build params. This is a
6.7% relaxation at defaults and changes 0D-path test values (updated
deliberately; the non-0D path and both load-bearing pins are untouched).

### Surface heat flux constraint

**DECISION (c), settled 2026-06-12 (user):** surface heat flux is a full
constraint, symmetric with the neutron cap.

The first wall takes two physically distinct loads. Neutrons pass
through and deposit volumetrically in the blanket (a fluence/lifetime
problem: Part 2 cap, Part 3 economics). Photons (p_rad) and cross-field
charged transport (p_radial) deposit in the first microns of the surface
(a real-time cooling problem). Large-area actively cooled first walls
manage about 1 MW/m^2; the axial end loss p_end exits through the
throats to the expander/DEC plates and never touches the lateral wall.

    q_surface = (p_rad + p_radial) / A_fw
    q_surface <= q_surface_max        (YAML key, default 1.0 MW/m^2,
                                       pinned by citation in the same
                                       account-justification writeup as
                                       the Part 3 fluence limits)

For advanced fuels this is THE binding wall constraint (D-He3 radiates
24% of fusion power as photons at the proxy, p-B11 83%, with few or no
neutrons), while the neutron cap barely binds; for DT it lands the same
order as the neutron concern.

Sizing-mode mechanism: a third branch of the density resolution,
n_e(T) = min(n_beta, n_wall, n_surf). For advanced fuels n_surf(T) has
the same closed form as n_wall (proxy radiation scales as P_fus, i.e.
n^2). For DT/DD the full radiation model is not a pure power of n, so
the n_surf branch is found by a short monotone bisection on n at fixed
T (radiation is monotone increasing in density). Infeasibility under the
cap raises SizingInfeasible naming q_surface_max, per decision (b).
MirrorPlasmaState gains a `q_surface` field; forward/inverse audit modes
warn at the same threshold.

## Part 3: Fluence-based core lifetime (CAS72)

Today CAS72 scheduled replacement uses a per-fuel `core_lifetime`
constant (DT 5 FPY, DD 10, D-He3 30, p-B11 50) regardless of wall
loading: a machine at 13 MW/m^2 replaces its first wall and blanket on
the same schedule as one at 5 MW/m^2. This is the missing economic
pushback behind the optimizer's boundary-riding.

Replace the per-fuel lifetime constants with per-fuel neutron fluence
limits:

    Phi_max[fuel]  [MW yr / m^2]      (YAML, sourced)
    core_lifetime_FPY = Phi_max / q_n_wall

Consistency check on the existing constants: DT 5 FPY at a 3-4 MW/m^2
class wall loading implies Phi_max of 15-20 MW yr/m^2, which is the
commonly cited RAFM-steel FW/blanket fluence window, so the current
defaults are recoverable as Phi_max divided by a reference wall loading.
Per-fuel limits stay separate (the dpa per MW yr/m^2 differs between
14.1 MeV, 2.45 MeV, and aneutronic side-channel spectra); each value
needs a citation in an account-justification writeup before the default
flips on.

The lifetime enters CAS72 through `levelized_replacement_cost`, which is
smooth in q_n, so the optimizer sees a continuous LCOE penalty for
pushing f_beta (higher density -> higher q_n -> shorter life -> larger
annualized replacement). The hard cap of Part 2 remains as the
engineering backstop; Part 3 makes the region below the cap correctly
non-free. Guard rails: lifetime clamped to [a small floor, plant life]
so the gradient stays finite at extreme q_n.

**DECISION (d), settled 2026-06-12 (user): (d1)** — the fluence basis
applies to ALL MFE concepts at once. Existing tokamak/stellarator CAS72
results move wherever their q_n differs from the implied reference; the
basis change is documented with citations in the account-justification
writeup, and the result shifts are quantified in the implementation
(before/after table for the reference concepts). IFE/MIF stay out of
scope (their chamber-wall fluence story is per-shot and lives in the
existing per-concept treatments).

## Out of scope

- Solving the plasma radius a from power or wall loading. The constraint
  above makes the a-trade visible (smaller a lowers q_wall and lengthens
  L); promoting a to a solved variable is a separate workline needing
  its own constraint set (NBI penetration, a/rho_i floor).
- f_dec provenance (still open, tracked separately).
- Tandem-mirror end-plug physics (unchanged scope from the parent spec).

## Test plan

- Radial build: pin r_bore_central = 3.30 m at YAML defaults; coil cost
  responds to blanket_t (differentiability test); calibration-neutrality
  pins hold bit-identically; plug/central split pinned at the new basis.
- Wall constraint: sizing at default machine with q_wall_max = 5.0 gives
  q_wall <= 5.0 + epsilon and beta < f_beta * beta_max (wall-bound
  regime); raising q_wall_max to 20 recovers today's beta-bound solution
  (regression continuity); optimizer f_beta result becomes interior or
  cap-saturated with q_wall at the cap; jit==eager and finite gradients
  for n_wall(T) all four fuels.
- A_fw basis: state fw_area equals geometry firstwall_area for the same
  build (cross-layer consistency test, closing the audit finding).
- Surface flux: p-B11 sizing is q_surface-bound (n_surf < n_beta and
  n_surf < n_wall at the solution) with q_surface at the cap; DT default
  machine reports q_surface on the state and warns above q_surface_max
  in audit modes; jit==eager for the advanced-fuel closed form; the
  DT/DD inner bisection converges (monotonicity test of p_rad in n).
- Fluence lifetime: DT at the implied reference wall loading reproduces
  5 FPY (continuity); CAS72 grows smoothly with f_beta in sizing mode
  (finite positive gradient of LCOE w.r.t. f_beta through the
  replacement term); before/after CAS72 table for tokamak and
  stellarator reference configs recorded in the account doc.
