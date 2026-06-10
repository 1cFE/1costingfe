# Tokamak Power-to-Geometry Sizing

## Purpose

Geometry-driven cost accounts (C220103 coils, C220101 blanket, C220106 vessel)
must scale with the power target. Without a sizing solve the geometry is a
fixed YAML input, so those accounts are constant across the entire power range.
The sizing solve maps a target net electric power to tokamak geometry (R0, a,
B0), feeding the existing geometry-to-cost pipeline so that capital costs
scale with power.

## Constraint-Boundary Method

At fixed aspect ratio and magnet conductor ceiling, the operating point at the
constraint boundary is determined by size alone.

**Plasma current.** From MHD force balance at q95:
```
I_p = 2 pi a^2 kappa B0 / (mu0 R0 q95)
```

**Density.** Set to the Greenwald fraction:
```
n_e = f_GW * I_p [MA] / (pi a^2) * 1e20   [m^-3]
```

**Temperature.** Maximized over a per-fuel interval [T_min, T_max] via
golden-section search on net power, with the Troyon beta limit
(beta_N <= beta_N_max) as a binding check. When beta is the binding
constraint the optimum sits on the cap. A small tolerance (1e-4 * beta_N_max)
prevents floating-point artifacts from penalizing genuinely feasible points.

**On-axis field.** Derived from the magnet peak-field ceiling and the inboard
radial build (see next section).

**Net power.** Evaluated with the existing `mfe_forward_power_balance`, which
accounts for auxiliary heating, recirculating power (including resistive
losses for copper coils), thermal conversion, and balance-of-plant loads.

**R0 bisection.** Net power is monotonic increasing in R0 at the boundary
operating point, so R0 is found by bisection in [R0_min, R0_max] (bounds in
the concept YAML). Convergence is guaranteed by monotonicity; 60 iterations
give sub-millimetre precision.

**Feasibility.** If net power at R0_max is below the target, or the
temperature maximization is at its upper bound with negative net power, the
solver raises `SizingInfeasible` rather than diverging silently. Feasibility
couples to both magnet and fuel: aneutronic fuels at conventional field and
beta do not close (e.g., p-B11 on copper is infeasible).

**D-T scope.** The 0D model's reactivity is currently D-T only (Bosch-Hale
parameterization). The solver is written fuel-generally (no D-T constants in
the bisection logic), but extending to other fuel reactivities is a tracked
follow-on task.

## On-Axis Field Relation

The toroidal field falls as 1/R from the inboard TF leg to the plasma axis.
The inboard coil inner radius is:
```
R_coil_inner = R0 - a - (blanket_t + ht_shield_t + structure_t + vessel_t)
```
and the on-axis field is:
```
B0 = B_max * R_coil_inner / R0
```

This is implemented in `b0_from_radial_build` in
`src/costingfe/layers/tokamak.py`. Fixed-meter inboard layers (blanket,
shield, structure, vessel) penalize small machines: as R0 shrinks,
R_coil_inner shrinks faster than R0, so B0/B_max drops. This makes
high-field conductor (rebco) matter more at small size and is physically
correct. A floor at R_coil_inner = 1e-3 m keeps the 0D physics real-valued
when R0 is geometrically infeasible; such points return near-zero net power
and the bisection moves away from them.

## Magnet Selection

The `coil_material` parameter selects conductor type. Properties are looked
up from `MAGNET_TABLE` in `src/costingfe/defaults.py` via
`get_magnet_properties()`. The table carries:

| Conductor | B_max (T) | recirc_power_factor | cryo_temp_k |
|-----------|-----------|---------------------|-------------|
| rebco_hts | 23.0      | 0.0                 | 20.0        |
| nb3sn     | 13.0      | 0.0                 | 4.5         |
| nbti      | 9.0       | 0.0                 | 4.5         |
| copper    | 8.0       | 2.0e-4              | 300.0       |

Superconductors (rebco, nb3sn, nbti) carry zero recirculating power.
Copper's resistive dissipation adds to the p_coils load:
```
P_recirc = recirc_power_factor * B0^2 * V_plasma   [MW]
```
implemented in `resistive_recirc_power`. The additional load reduces net
electric output, forcing the bisection to a larger R0 to compensate;
copper therefore sizes larger than a superconducting machine at equal power.

Picking `coil_material` also determines which CAS22.01.03 costing path is
used: $/kAm for superconductors, mass build-up for copper.

## Four Usage Modes

The sizing solve is gated by two YAML flags, both defaulting to false, and
implements four distinct usage patterns.

**Pin.** Default mode. Both flags false. Geometry is a YAML input; the
sizing solve is bypassed. Suitable for pricing a stated reference machine.

**Size.** Set `size_from_power: true`. The solver computes R0, a, and B0
from the design knobs (aspect_ratio, elon, beta_N, H_factor, f_GW, q95,
coil_material) and the power target. Useful for checking whether a vendor's
claimed design is self-consistent, or for producing an honest bottom-up cost.
In this mode any R0, plasma_t, b_center, or B override is rejected with a
ValueError.

**Scale.** Same as size mode, but the design knobs are extracted from a
published reference machine (aspect ratio, elongation, q95, etc.) while the
power target is changed. R0 grows to reach the new target, everything else
stays anchored to the reference design fingerprint.

**Optimize.** Set `optimize_lcoe: true` (implies size_from_power). A nested
outer golden-section search minimizes LCOE over f_GW, with R0 slaved to the
power target at each trial f_GW. LCOE is non-monotonic in f_GW: higher f_GW
raises power density and shrinks the machine (lower capital) but raises the
disruption rate (higher replacement cost). The optimum may be interior or at
the Greenwald boundary, depending on the disruption input parameters. See
[[disruption_severity]] for the literature grounding of those parameters.
With current default disruption values the optimum sits at the boundary.

## Pipeline Integration

The sizing solve runs in `_size_tokamak` (model.py) after the radial build is
derived and before the geometry pass. It mutates the params dict in place,
injecting solved R0, plasma_t, B, b_center, and T_e. The returned power table
replaces the normal `_power_balance` call; there is no separate power-balance
call in sizing mode. The full pipeline order in sizing mode is:

```
validate overrides -> merge params -> derive_radial_build
  -> _size_tokamak (bisection; injects R0, a, B0, T_e into params)
  -> geometry (RadialBuild uses solved R0/a)
  -> cost accounts -> economics
```

Multi-module plants: the solver sizes one module to
net_electric_mw / n_mod, consistent with the existing per-module power
balance.

## Representative Result

At 270 MWe with rebco and ARC-like knobs (aspect_ratio 3.3, elon 1.8,
beta_N 2.8, q95 3.5, f_GW 0.85, H_factor 1.1), the solver returns R0
approximately 3.6 m (the ARC design point is approximately 3.3 m at
slightly different knobs). Switching to nb3sn at equal power and equal
physics knobs yields a larger machine, confirming that rebco sizes smaller
due to its higher B_max.

## Sources and Methodology Notes

- All physics constants (magnet table, temperature bounds, R0 bounds)
  live in YAML or defaults.py. No numbers appear in function bodies or
  as Python argument defaults, consistent with the placement discipline.
- The solver is pure: a params dict in, a SizingResult out.
- Backward compatibility: with both flags false, every existing config
  and test behaves identically.
