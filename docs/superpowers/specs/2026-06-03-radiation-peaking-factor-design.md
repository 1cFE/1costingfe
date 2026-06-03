# Radiation peaking factor for highly-peaked plasmas (issue #24)

## Problem

The MFE radiation model in `compute_p_rad` (`src/costingfe/layers/radiation.py`)
integrates local emissivity over a single uniform `plasma_volume`:

    P_brem = 5.35e3 * n_e_20^2 * Z_eff * sqrt(T_e) * V * 1e-6   [MW]
    P_line = sum_z(f_z * L_z(T_e)) * n_e_20^2 * V * 1e34         [MW]

`n_e` and `T_e` are supplied as peak/central values. For flat tokamak
profiles peak ~= volume-average and this is fine. For a levitated dipole the
Hasegawa-Mauel profile is extreme (n proportional to R^-4, T proportional to
R^-8/3): the hot, dense radiating core is a few percent of the geometric
plasma volume. Multiplying peak n^2 * sqrt(T) by the full volume therefore
over-counts bremsstrahlung by more than an order of magnitude.

### Concrete failure (OpenStar Reactor A, Simpson et al. 2026, arXiv:2602.20564)

With the real geometric plasma volume of 13,600 m^3 (Simpson Table 6):

| Quantity | Library | Simpson Reactor A |
|---|---:|---:|
| P_brems | 1,370 MW | (radiation is a small fraction of fusion) |
| p_fus | ~2,775 MW | ~667 MW |
| recirc fraction | ~86% (unphysical) | ~25-35% |

The bremsstrahlung overhead forces the inverse balance to manufacture ~4x the
fusion power Simpson reports, inflating every p_th-scaled CAS22 account by
~2.5x.

### Current workaround (to be removed)

`steady_state_dipole.yaml` sets `plasma_volume: 200`, an undocumented
calibration value (effective 0.0147 of the geometric volume) that is fragile
and non-obvious.

## Verification of radiation channels (lit + empirical)

Simpson et al. Eq 10 models radiation as `P_rad = P_brem + f_gcr * f_alpha *
P_fus`: bremsstrahlung (impurities folded into `Z_eff = 1.5`) plus a
recycled-alpha term. There is no synchrotron term and no separate coronal
line-radiation term.

Empirical breakdown for the dipole defaults (n_e=1.95e20, T_e=10.9 keV,
Z_eff=1.5, B=2 T), using the library's own radiation functions:

| Channel | Dipole value today | Reason |
|---|---:|---|
| Bremsstrahlung @ V=13,600 | 1,370 MW | the term that explodes with volume |
| Synchrotron | 0 MW | `R_major=0` gates Albajar off (`radiation.py:271`) |
| Line / impurity | 0 MW | `fw_area` defaults to 0 -> wall-derived f_z=0; no seeded impurities |

- Synchrotron is negligible and structurally unsuited to a volume multiplier:
  it scales as B^2 (dipole plasma-region field 2 T is far below tokamak
  reactor fields), the Albajar fit has no volume term, and it already carries
  profile peaking via `alpha_n`/`alpha_T`. It is left untouched (and remains
  zero for the dipole).
- Line radiation shares the n^2 * V emission-measure structure of brems, so it
  takes the same correction, but it is zero for the dipole today.
- Bremsstrahlung is 100% of dipole radiation and the only term that must be
  corrected.

## Design

### 1. Library: emission-measure correction on the collisional volume terms

Add a `radiation_peaking_factor` parameter to `compute_p_rad`. It multiplies
the two terms that integrate local emissivity over volume (bremsstrahlung and
line), and leaves synchrotron untouched:

    P_rad = peaking * (P_brem + P_line) + P_sync

The factor is the emission-measure ratio
`integral(n^2 sqrt(T) dV) / (n_peak^2 sqrt(T_peak) * V)`. For a uniform profile
it is 1.0; for the dipole's peaked profile it is small.

`compute_p_rad` is called from `mfe_forward_power_balance` and
`mfe_inverse_power_balance` (`src/costingfe/layers/physics.py`). Both gain a
`radiation_peaking_factor` parameter, threaded from `model.py`'s `rad_kw`.

The factor only affects the `compute_p_rad` branch. The `f_rad_fus` and
`p_rad_override` branches are unchanged (a peaked profile is already implicit
when radiation is specified as a fraction of P_fus or as a fixed value).

### 2. YAML defaults

`radiation_peaking_factor` is read as a required parameter (no Python-side
default, per the project rule that defaults live in YAML). It is added to all
seven steady-state defaults that carry `plasma_volume`:

| File | Value |
|---|---|
| steady_state_tokamak.yaml | 1.0 |
| steady_state_stellarator.yaml | 1.0 |
| steady_state_mirror.yaml | 1.0 |
| steady_state_steady_frc.yaml | 1.0 |
| steady_state_orbitron.yaml | 1.0 |
| steady_state_polywell.yaml | 1.0 |
| steady_state_dipole.yaml | 0.05 |

For the dipole the same change also replaces the calibration volume:

    plasma_volume: 13600          # was 200; Simpson Table 6 geometric plasma volume
    radiation_peaking_factor: 0.05 # Hasegawa-Mauel core fraction, Simpson 2.1.1

Net dipole bremsstrahlung becomes 0.10 * 13,600 * 0.05 = 68 MW, which sits
below p_ash (~133 MW at 667 MW fusion), so it is fully absorbed and does not
inflate the heating power.

Pulsed concepts do not carry `plasma_volume` and do not reach `compute_p_rad`
(they use the f_rad fraction path), so they are not touched.

### 3. Backward compatibility

`radiation_peaking_factor = 1.0` for every non-dipole concept reproduces the
current bremsstrahlung exactly. Only the dipole's combined volume + factor
change alters any output.

## Testing

- Unit test in `tests/`: `compute_p_rad` scales the bremsstrahlung+line
  contribution linearly with `radiation_peaking_factor` while leaving the
  synchrotron contribution unchanged (use a geometry that enables Albajar to
  confirm synchrotron is invariant to the factor).
- Concept test: the DIPOLE default model runs end-to-end with a sane
  recirculating fraction (well under 50%) and p_fus in the right ballpark
  (order 700 MW, not thousands) at the geometric 13,600 m^3 with factor 0.05.
- Regression: existing non-dipole steady-state tests are unchanged because
  their factor is 1.0.

## Out of scope

- The OpenStar per-instance concept spec and override-side validators live in
  the segregated fusion-tea repo (PR #37), not here.
- No change to synchrotron or line-radiation physics beyond the multiplier
  placement.
