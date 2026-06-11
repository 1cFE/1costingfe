# Mirror 0D Physics Model and Length Sizing

**Date:** 2026-06-11
**Status:** Approved design, pending implementation

Settled decisions: f_dec stays a YAML input (derived axial share is a
diagnostic only; the 0.3 value is untraced and owes a provenance writeup);
validation anchors are GDT (gas-dynamic branch, measured) and WHAM
(Pastukhov branch, published design), with Realta as the first evaluation;
0D model and length sizing ship in one pass, opt-in
(use_0d_model: false) until the anchors validate.


## Objective

Add a reduced-order (0D) physics model for axisymmetric magnetic mirrors to
1costingfe, following the same pattern as the tokamak 0D model
(`layers/tokamak.py`). The model derives fusion power and plasma state from
machine-level parameters, enabling inverse-mode operation (target P_net
determines required machine parameters) and gradient-based sensitivity
analysis via JAX.

## Design philosophy

This is a costing model with a physics layer, not a mirror design code. The
user specifies a handful of machine parameters; the model returns fusion
power and a complete power table. Detailed field geometry (coil positions,
field curvature, flux surface shape) is deliberately excluded. Where a
detailed design code would solve for field profiles, this model uses the
mirror ratio R_m as a single lumped parameter.

---

## Inputs

| Parameter | Symbol | Units | Typical range | Notes |
|-----------|--------|-------|---------------|-------|
| Central cell length | L | m | 3 - 200 | Axial extent of confinement region. Realta's Hammir tandem mirror baselines a 50 m center cell for Q > 5 and explicitly proposes longer cells for Q > 10+ (Forest et al., arXiv:2411.06644, table 1; Realta 2025 announcements), so the model must remain valid well past 50 m |
| Plasma radius | a | m | 0.1 - 1.5 | At midplane |
| Mirror ratio | R_m | - | 2 - 100 | B_max / B_min |
| Ion temperature | T_i | keV | 5 - 80 | Volume-averaged |
| Electron temperature | T_e | keV | 5 - 80 | Volume-averaged; defaults to T_i |
| Electron density | n_e | m^-3 | 1e19 - 1e21 | Volume-averaged |
| Auxiliary heating | P_input | MW | 0 - 200 | NBI-dominated, treated as steady-state sustainment power |
| Fuel type | fuel | enum | DT, DD, ... | From existing Fuel enum |
| Beta limit | beta_max | - | 0.3 - 0.6 | User constraint (default 0.5) |

### Parameter mapping to existing YAML

The mirror YAML (`steady_state_mirror.yaml`) already defines several
parameters. The mapping to mirror model inputs:

| YAML key | Mirror model symbol | Notes |
|----------|-------------------|-------|
| `chamber_length` | L | Direct mapping |
| `plasma_t` | a | Plasma radius at midplane |
| `b_max` | - | **Peak field on conductor** (retained for coil costing CAS22) |
| `B` | B_min | Central (midplane) magnetic field |
| `n_e` | n_e | Direct mapping |
| `T_e` | T_e | Direct mapping |

New YAML keys to add:
- `R_m: 10.0` (mirror ratio; B_max_throat = R_m * B derived from this)
- `T_i: 20.0` (ion temperature; defaults to T_e if absent)
- `beta_max: 0.5` (MHD stability limit)
- `use_0d_model: false` (opt-in, as for the tokamak; flip the default only
  after validation against the anchor machines)

The existing `b_max` is the peak field on the superconductor for coil
costing. The mirror throat field B_max_throat = R_m * B is derived, not
input directly. These are physically distinct: b_max >= B_max_throat due
to coil geometry. For costing purposes, `b_max` may be overridden to equal
B_max_throat as a conservative default.

### Derived quantities (not user inputs)

| Quantity | Formula | Notes |
|----------|---------|-------|
| Throat field | B_max_throat = R_m * B | Mirror throat field |
| Plasma volume | V = pi * a^2 * L | Cylinder |
| First-wall area | A_fw = 2 * pi * a * L | Lateral surface only |
| Beta | beta = mu_0 * n_e * (T_e + (n_i/n_e) * T_i) * 2 / B^2 | At midplane (B = B_min); ion term diluted via n_i/n_e for advanced fuels |

---

## Physics model

### Fusion power

Computed by `reactivity.fusion_power(fuel, n_e, T_i, V_plasma, ...)` with
cylindrical volume V_plasma = pi * a^2 * L. The shared kernel provides
per-fuel reactivity fits (Bosch-Hale for D-T/D-D/D-He3, Nevins-Swain for
p-B11), quasineutrality dilution from the fuel-mix knobs, the derived
D-He3 side-channel fraction (pin-overridable), and the hot-ion T_i/T_e
treatment - all of which matter more for mirrors than tokamaks, since
hot-ion D-He3 is the flagship mirror use case. The kernel is float32/jit
hardened (optimization barrier on density, 1e-22 reactivity units); new
mirror physics functions must follow the same discipline and carry a
jit-equals-eager regression test, since the T_i bisection runs inside
jax.lax.fori_loop. Per-fuel ion dilution (n_i/n_e from
reactivity.n_i_over_n_e) also enters the stored energy and beta below.

### Confinement time

Mirror confinement is governed by axial particle losses through the loss
cone. Three regimes exist; the model computes all three and combines them.

**Ion-ion collision time** (base timescale):

    tau_ii = 2.09e13 * T_i[keV]^1.5 * sqrt(A) / (n_i[m^-3] * Z^4 * ln_Lambda)

where A is the ion mass number (2.5 for DT), Z = 1, and ln_Lambda is the
Coulomb logarithm (taken as 17 for fusion-relevant plasmas).

**Classical mirror confinement** (Bing and Roberts, 1961):

    tau_classical = 2.6 * ln(R_m) * tau_ii

This is the baseline: ions scatter into the loss cone and escape axially.

**Pastukhov confinement** (electrostatic plugging, Pastukhov 1974, Cohen
et al. 1978):

    tau_Pastukhov = tau_ii * (sqrt(pi)/2) * ((R_m + 1) / R_m) * ln(2*R_m + 2) * (e*phi / T_i) * exp(e*phi / T_i)

The ambipolar potential phi is set self-consistently. For a simple mirror
the Boltzmann relation gives:

    e * phi = T_e * ln(sqrt(m_i / (2 * pi * m_e)))

which gives e*phi of 3 - 4 T_e. This exponential enhancement over classical
confinement is the primary benefit of electrostatic plugging and the
dominant physics for modern mirror concepts (Realta, WHAM).

When T_e defaults to T_i, the ratio e*phi / T_i = (T_e/T_i) * ln(...) is
constant, and exp(e*phi/T_i) is a constant multiplier. The Pastukhov time
then scales as tau_ii, which is monotonically increasing in T_i, so
P_fus(T_i) remains monotonically increasing (from the reactivity curve)
and the inverse bisection converges. If T_e and T_i are decoupled (T_e
held fixed while T_i varies), the exp(phi/T_i) factor decreases with
increasing T_i, reducing tau_Pastukhov. In this regime P_fus(T_i) is still
monotonically increasing because the reactivity <sigma*v> dominates, but
the confinement quality degrades. The inverse solver should verify the
operating point is self-consistent after convergence.

**Gas-dynamic confinement** (Mirnov and Ryutov, 1979):

    tau_GD = R_m * L / v_thi

where v_thi = sqrt(2 * T_i / m_i). This regime applies when the mean free
path is shorter than the device length (high density, low temperature). It
is relevant for GDT-class devices.

**Combined axial confinement time**:

    1/tau_axial = 1/tau_Pastukhov + 1/tau_GD

This interpolation naturally selects the dominant regime: Pastukhov at low
collisionality, gas-dynamic at high collisionality.

**Radial confinement time** (classical cross-field diffusion):

    tau_radial = (a / rho_i)^2 * tau_ii

where rho_i = sqrt(2 * m_i * T_i) / (e * B) is the midplane ion
gyroradius at B = B_min. In well-confined mirrors, tau_radial >> tau_axial
and radial losses are subdominant. The model includes this term for
completeness and to flag regimes where radial transport becomes competitive.

**Total particle confinement time**:

    1/tau_p = 1/tau_axial + 1/tau_radial

### Energy confinement time

The energy confinement time differs from the particle confinement time
because escaping particles carry energy. Each escaping ion carries on
average phi + T_i of energy (potential energy plus thermal), and each
electron carries phi + T_e. The energy confinement time is:

    tau_E = tau_p * (3/2 * (T_i + T_e)) / (phi + T_i + phi + T_e)
          = tau_p * (3/2 * (T_i + T_e)) / (2*phi + T_i + T_e)

This ratio is less than 1 (energy is lost faster than particles) because
escaping particles are preferentially energetic.

### End-loss power

The dominant loss channel in a mirror. The total end-loss power:

    P_end = W_th / tau_E

where W_th = (3/2) * n_e * (T_i + T_e) * V_plasma is the stored thermal
energy. Equivalently:

    P_end = n_e * V_plasma * (2*phi + T_i + T_e) / tau_p

The axial end losses carry directed kinetic energy (ions accelerated
through the ambipolar potential) and are suitable for direct energy
conversion. Radial losses deposit on the first wall as heat and enter the
thermal cycle.

### Power balance integration with existing MFE path

**This is the key integration point.** The existing `mfe_forward_power_balance()`
computes `p_transport = p_ash + p_input_eff - p_rad`, which is the generic
MFE transport loss (deposited on plasma-facing surfaces). In a mirror, the
transport loss has two physically distinct channels:

1. **Axial end losses (P_end_axial):** directed plasma exhaust through the
   mirror throats, recoverable via DEC
2. **Radial losses (P_radial):** classical cross-field transport deposited
   on the first wall, entering the thermal cycle

The mirror model does NOT replace `mfe_forward_power_balance()`. Instead,
it operates as follows:

**Forward path:**
1. Mirror model computes P_fus, P_end, P_radial, P_rad from its own physics
2. The existing `mfe_forward_power_balance()` is called with P_fus as input
3. The existing function computes `p_transport` from its own energy balance
4. The existing `f_dec` parameter controls what fraction of `p_transport`
   goes to DEC vs. thermal

The existing power balance already routes `f_dec * p_transport` through DEC
at efficiency `eta_de`. For a mirror, `f_dec` represents the fraction of
total transport power that exits axially (vs. radially). This is
physically justified: `p_transport` in the existing model is
`p_ash + p_input_eff - p_rad`, which is exactly the total non-radiative
loss, and in a mirror this splits between axial and radial channels.

`f_dec` REMAINS A YAML INPUT under the 0D model. The physically derived
axial share

    f_axial_derived = P_end_axial / (P_end_axial + P_radial)

is reported on MirrorPlasmaState as a DIAGNOSTIC only, never load-bearing:
P_radial comes from classical cross-field diffusion, the least-trusted and
likely underestimated term in the model, so deriving f_dec from this ratio
would yield ~0.95+ and silently replace an engineering judgment with an
optimistic idealization. f_dec also bundles DEC-convertibility physics
(expander geometry, charge exchange, electron-channel recovery) that the
tau ratio does not capture. Open item: the current YAML value f_dec = 0.3
is NOT traced to any source and needs a provenance writeup (expander and
end-converter studies) in docs/account_justification/ before mirror
results are published; carry it as a range until then.

**Inverse path:** with `f_dec` a fixed input, the inverse is single-pass
(no outer self-consistency loop):
1. Call `mfe_inverse_power_balance()` with the YAML `f_dec` to get the
   required P_fus
2. Bisect on T_i to match that P_fus
3. Recompute the full mirror state and power table at the solved T_i,
   including the f_axial_derived diagnostic

### Radiation

Radiation follows the same per-fuel resolution as the tokamak 0D path:
D-T/D-D use the full `compute_p_rad()` model; D-He3 and p-B11 default to
the CostingConstants `f_rad_fus` proxies (0.24 / 0.83), with an explicit
`f_rad_fus` override winning - consistent with the non-0D path. The fuel-ion
charge enters bremsstrahlung via `z_eff_fuel` + impurity excess, as in the
tokamak. For the full model, The
synchrotron geometry mapping for mirrors is already handled in `model.py`
(line 112-115): when `R0 = 0`, the effective major radius is set to
`L / (2*pi)`, mapping the cylinder to an equivalent torus for the Albajar
formula. Wall reflectivity `R_w` defaults to 0.4 (reduced from 0.6 for
tokamaks) to account for radiation escaping through the open ends.

### Beta limit

    beta = 2 * mu_0 * n_e * (T_i + T_e) / B^2

where B = B_min (midplane field). Mirror machines can operate at high beta
(0.3 - 0.6 demonstrated in GDT). The MHD stability limit for an
axisymmetric mirror depends on the field geometry (minimum-B vs. simple
mirror with vortex stabilization). The model treats `beta_max` as a user
input (default 0.5). Following the tokamak inverse-mode precedent
(`OperatingPointInfeasible`): in inverse mode, an implied operating point
with beta > beta_max raises an error carrying the diagnosis (the beta
limit is the mirror's only error-severity stability limit - there is no
Greenwald or kink channel), with `enforce_plasma_limits: false` as the
explicit escape hatch. Forward mode reports beta as a diagnostic with a
warning. Wall loading remains warning-severity in both modes.

### Wall loading

    q_wall = P_neutron / A_fw

where A_fw = 2 * pi * a * L (lateral first-wall area). End plates are not
included in the neutron wall loading because the end regions are outside the
blanket.

---

## Outputs: MirrorPlasmaState

Following the pattern of `PlasmaState` in `tokamak.py`:

| Field | Type | Units | Description |
|-------|------|-------|-------------|
| n_e | float | m^-3 | Operating density |
| T_i | float | keV | Ion temperature |
| T_e | float | keV | Electron temperature |
| beta | float | - | Midplane beta |
| tau_p | float | s | Particle confinement time |
| tau_E | float | s | Energy confinement time |
| tau_classical | float | s | Classical mirror confinement |
| tau_Pastukhov | float | s | Pastukhov confinement |
| tau_GD | float | s | Gas-dynamic confinement |
| phi | float | keV | Ambipolar potential (e*phi in keV) |
| p_fus | float | MW | Fusion power |
| p_alpha | float | MW | Alpha heating |
| p_end | float | MW | End-loss power (axial) |
| p_radial | float | MW | Radial transport power |
| p_rad | float | MW | Radiation power |
| f_axial_derived | float | - | Diagnostic: axial share of transport losses (NOT used as f_dec) |
| V_plasma | float | m^3 | Plasma volume |
| fw_area | float | m^2 | First-wall area |
| wall_loading | float | MW/m^2 | Neutron wall loading |
| R_m | float | - | Mirror ratio |
| collisionality | float | - | L / mean_free_path diagnostic |

The `confinement_regime` diagnostic (Pastukhov-dominated, gas-dynamic, or
radial) is reported as a string annotation but not stored in the dataclass
(not JAX-traceable). The `collisionality` diagnostic flags whether the
Maxwellian assumption underlying the Pastukhov formula is valid
(collisionality > 1 means collisional, Pastukhov valid; < 0.1 means
nearly collisionless, Pastukhov may overestimate confinement).

---

## Operating modes

### Forward mode

`mirror_0d_forward(L, a, B_min, R_m, T_i, n_e, p_input, fuel, ...)`

Given machine geometry and plasma parameters, compute the complete plasma
state including P_fus, confinement times, beta, end losses, and wall loading.
Returns MirrorPlasmaState.

### Inverse mode

`mirror_0d_inverse(p_net_target, L, a, B_min, R_m, n_e, fuel, ...)`

Given a target net electric power, find the ion temperature T_i that
produces the required fusion power.

The inverse solver:
1. Take f_dec from the YAML (fixed input)
2. Call `mfe_inverse_power_balance()` with current f_dec to get required P_fus
3. Bisect on T_i to match that P_fus (via `reactivity.fusion_power` with
   cylindrical volume), inside a fuel-keyed bracket (mirror operating
   ranges differ from the tokamak's `_T_BRACKET_DEFAULTS`; proposed:
   DT 2-80 keV, DD 5-100, D-He3 20-100, p-B11 50-300, bounded by the
   fit validity ranges)
4. Run `mirror_0d_forward()` at the solved T_i to get the full plasma
   state (including the f_axial_derived diagnostic)
5. Return (MirrorPlasmaState, PowerTable)

---

## Integration with 1costingfe

### New file

`src/costingfe/layers/mirror.py` containing:
- `MirrorPlasmaState` frozen dataclass
- Pure JAX functions: `compute_tau_ii`, `compute_tau_classical`,
  `compute_tau_pastukhov`, `compute_ambipolar_potential`,
  `compute_tau_gas_dynamic`, `compute_tau_radial`,
  `compute_combined_tau`, `compute_end_loss_power`,
  `compute_axial_loss_fraction` (diagnostic)
- `mirror_0d_forward()` and `mirror_0d_inverse()`
- Reuses `fusion_power()`, `n_i_over_n_e()`, `z_eff_fuel()` from
  `layers/reactivity.py` (the concept-agnostic multi-fuel kernel)

### Model dispatch (model.py)

Add a mirror branch parallel to the tokamak branch at line 94:

```python
if use_0d and self.concept == ConfinementConcept.MIRROR:
    return self._power_balance_mirror_0d(params, n_mod)
```

The `_power_balance_mirror_0d` method:
1. Extracts mirror-specific params (L, a, B, R_m) from the params dict
   (mapped from `chamber_length`, `plasma_t`, `B`, `R_m`)
2. Calls `mirror_0d_inverse()` to get (MirrorPlasmaState, PowerTable)
3. Stores MirrorPlasmaState on the result (parallel to tokamak PlasmaState)
4. Returns the PowerTable (same interface as the tokamak branch)

### YAML defaults update

Update `steady_state_mirror.yaml` to add:
- `R_m: 10.0` (mirror ratio)
- `T_i: 20.0` (ion temperature; defaults to T_e if absent)
- `beta_max: 0.5` (MHD stability constraint)
- `use_0d_model: false` (opt-in until validated)

Existing `b_max: 12.0` is retained unchanged (peak field on conductor for
coil costing). Existing `B: 3.0` is the midplane field (B_min).

### Reuse

The following existing components are reused without modification:
- `fusion_power()` and the mix algebra from `layers/reactivity.py`
  (per-fuel reactivity, dilution, derived D-He3 fraction)
- `ash_neutron_split()` from `physics.py` (fuel-dependent power split)
- `compute_p_rad()` from `radiation.py` (bremsstrahlung + synchrotron + line)
- `mfe_inverse_power_balance()` and `mfe_forward_power_balance()` from
  `physics.py` (with the YAML `f_dec`, unchanged)
- All CAS costing layers (mirror-aware via existing concept dispatch)
- Geometry module (already has mirror cylindrical volume)
- Synchrotron geometry mapping in `model.py` (R_eff = L / 2*pi)

### Tests

New file: `tests/test_mirror.py`
- `test_mirror_forward_dt`: Forward mode produces valid plasma state
- `test_mirror_inverse_dt`: Inverse mode finds T_i for target P_net
- `test_confinement_regimes`: Verify Pastukhov > classical for R_m > 2
- `test_gas_dynamic_limit`: At high density, tau_GD < tau_Pastukhov
- `test_beta_diagnostic`: Beta computed correctly and flagged if > beta_max
- `test_end_loss_power`: P_end consistent with tau_p and stored energy
- `test_f_axial_diagnostic`: higher R_m improves axial confinement, so
  the axial share of total losses falls; pin that direction at two R_m
  values, and assert the diagnostic does NOT feed the power balance
  (PowerTable identical under different derived values with f_dec fixed)
- `test_model_integration`: Full CostModel forward pass with
  use_0d_model=True for mirror concept
- `test_jit_matches_eager`: every new physics function finite and equal
  to eager under jax.jit, for all four fuels (the XLA constant-gathering
  hazard documented in reactivity.py applies to any chain with 1e13-class
  or 1e-20-class constants, e.g. tau_ii)
- `test_beta_gate`: inverse mode raises on implied beta > beta_max; the
  enforce_plasma_limits=False escape hatch returns the implied point
- `test_existing_mirror_runs_unchanged`: with use_0d_model=False (the
  default), all existing mirror concept results are bit-identical

### Paper

Add appendix section (parallel to the tokamak appendix) documenting the
mirror model equations and assumptions.

---

## Assumptions and limitations

1. **Axisymmetric geometry only.** No tandem mirror end plugs, no
   field-reversed mirrors. Tandem mirror physics (thermal barriers,
   sloshing ion distributions) would require a separate model.
2. **No detailed field geometry.** The mirror ratio R_m is a single number;
   actual field profiles, coil spacing, and flux surface shapes are not
   modeled.
3. **MHD stability assumed.** The model does not check interchange or flute
   stability; beta_max is an input constraint. The user is responsible for
   ensuring the chosen beta_max is achievable for the assumed stabilization
   method (minimum-B geometry, vortex flow, or expander stabilization).
4. **Maxwellian distribution.** The Pastukhov formula assumes a
   near-Maxwellian ion distribution, which breaks down at very low
   collisionality. The model reports a `collisionality` diagnostic
   (L / mean_free_path) to flag this regime.
5. **No DCLC instabilities.** Drift-cyclotron loss-cone microinstabilities
   are not modeled; they can enhance losses beyond Pastukhov in simple
   mirrors but are suppressed by warm-plasma stabilization or sloshing
   ions. The model assumes adequate stabilization.
6. **Collision physics is hydrogenic.** Reactivity, dilution, and the
   energy partition are fully fuel-aware via `layers/reactivity.py`, but
   tau_ii (and hence the Pastukhov and classical confinement times) is
   evaluated with Z = 1 and a single average ion mass; the Z^4 scaling
   means He-3 and boron admixtures shorten collision times faster than
   the average-ion approximation captures. Acceptable at costing fidelity
   for D-T and lean advanced-fuel mixes; a mix-averaged Z_eff^4 correction
   is a cheap refinement if D-He3 mirror results prove sensitive to it.
7. **T_i = T_e by default.** Decoupled electron and ion temperatures can
   be specified but require external justification. In NBI-heated mirrors,
   T_i > T_e is common; the model supports this.
8. **Steady-state sustainment.** The `p_input` parameter represents
   steady-state NBI power required to maintain the plasma (fueling,
   current drive equivalent for mirrors). No distinction is made between
   startup transient and steady-state power; startup costs are handled
   elsewhere in the costing model (CAS55 startup fuel inventory).


---

## Length sizing (size_from_power)

The mirror analog of the tokamak power-to-geometry solve: chamber length L
becomes a solved output, mapping a net electric target to machine size so
the geometry-driven accounts scale with power. Radius a, midplane field B,
and mirror ratio R_m stay inputs (radius is set by physics and engineering
constraints that do not scale simply with power; length does).

**Operating point at the beta boundary.** In sizing mode the density is not
a free input: it is pinned to a beta fraction, the mirror analog of the
tokamak's Greenwald fraction:

    n_e = f_beta * beta_max * B^2 / (2 * mu_0 * (T_e + (n_i/n_e) * T_i) * KEV_TO_J)

with f_beta a YAML knob (default 0.85) and (n_i/n_e) the pure mix function
from reactivity.n_i_over_n_e (no circularity: it depends only on the fuel
and mix ratio). The operating temperature T_i is chosen by golden-section
maximization of net power over the fuel-keyed bracket; because density
falls as temperature rises at fixed beta, the trade between reactivity and
pressure is explicit, exactly as in the tokamak solve.

**L bisection.** Net power is monotonic increasing in L in both confinement
regimes (P_fus scales with volume, i.e. linearly in L; tau_GD improves with
L, and Pastukhov confinement is L-independent, so end losses grow no faster
than linearly), so L is located by bisection on [L_min, L_max]. If L_max
cannot deliver the target, the solve raises SizingInfeasible, same contract
as the tokamak. L (chamber_length) cannot be pinned in sizing mode; pinning
it raises the same error the tokamak raises for R0.

**Optimize mode.** optimize_lcoe wraps the sizing solve with an LCOE
minimization over f_beta in [f_beta_min, f_beta_max], the analog of the
tokamak's f_GW optimizer.

**Mode asymmetry, by design.** Forward and inverse modes take n_e directly
(auditing a stated machine at its stated density); sizing mode derives
density from f_beta (designing a machine at the beta boundary).

**New YAML keys** (steady_state_mirror.yaml): f_beta: 0.85, L_min: 1.0,
L_max: 200.0, f_beta_min: 0.3, f_beta_max: 1.0, plus size_from_power and
optimize_lcoe flags (default false).

## Coil cost must scale with length (C220103)

The current mirror coil costing uses a FIXED n_coils = 10 in cas22.py,
calibrated to a Hammir-class 50 m central cell. Under length sizing that
fixed count would freeze the coil account while the machine grows - the
same defect length sizing exists to fix. Replace the single count with a
two-class structure matching the Realta architecture (4 high-field HTS
end-plug magnets - 25 T throats in Hammir, 17 T CFS-built in WHAM - plus
central-cell solenoid coils at 2.4-5.0 T, "significantly cheaper", whose
number grows with the cell):

    n_central = L / coil_spacing          (continuous, JAX-differentiable;
                                           a costing aggregate, not a
                                           physical config knob)
    n_plug    = n_plug_coils              (YAML; 4 for a tandem mirror,
                                           2 for a simple mirror)

Central-cell coils carry ampere-meters at b_center (the midplane-class
field); plug coils at b_plug = R_m * B (the throat field), with their
smaller bore. New YAML keys: coil_spacing: 5.0 [m], n_plug_coils: 4.
CALIBRATION NEUTRALITY REQUIRED: at the YAML reference point (L = 50 m,
spacing 5 m -> n_central = 10, matching today's n_coils = 10 plus the
plug-coil contribution), the new structure must reproduce the current
C220103 mirror cost by construction (recalibrate the markup once,
documented in the account justification), so existing mirror results are
unchanged at the defaults while sized machines scale.
