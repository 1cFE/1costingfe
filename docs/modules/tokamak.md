# Tokamak 0D Plasma Model

`src/costingfe/layers/tokamak.py` 

## Purpose

Layer 2b of the costing model. Derives fusion power, plasma current, density, confinement time, stability margins, and disruption risk from machine parameters (R, a, kappa, B, q95, f_GW) using standard tokamak scaling laws. Activated by `use_0d_model=True` on a tokamak `CostModel`.

Without this layer, the costing model treats fusion power as a free variable solved from the energy balance (Layer 2a). With this layer, fusion power is derived from plasma physics -- the model can tell you whether a given machine can physically produce the required power, and what the stability consequences are.

All core functions are pure and JAX-differentiable.

## PlasmaState

The module produces a `PlasmaState` dataclass containing all derived quantities:

| Field | Units | Description |
|-------|-------|-------------|
| `I_p` | MA | Plasma current |
| `n_GW` | 10^20 m^-3 | Greenwald density limit |
| `n_e` | m^-3 | Operating electron density |
| `T_e` | keV | Electron temperature |
| `beta_N` | %*m*T/MA | Normalized beta |
| `tau_E` | s | Energy confinement time |
| `p_fus` | MW | Fusion power |
| `p_alpha` | MW | Alpha heating power |
| `p_rad` | MW | Radiation power |
| `V_plasma` | m^3 | Plasma volume |
| `fw_area` | m^2 | First wall surface area |
| `q95` | -- | Safety factor at 95% flux |
| `f_GW` | -- | Greenwald fraction |
| `wall_loading` | MW/m^2 | Neutron wall loading |
| `div_heat_flux` | MW/m^2 | Divertor heat flux estimate |
| `H_factor` | -- | Confinement quality (tau_E / tau_E_scaling) |
| `disruption_rate` | /FPY | Disruption frequency from stability margins |

## Operating Modes

### Inverse mode (default)

`tokamak_0d_inverse`: Given a net electric power target, finds the electron temperature T_e that produces the required fusion power.

1. Compute I_p, n_e from machine geometry (R, a, kappa, B, q95, f_GW)
2. Use `mfe_inverse_power_balance` to get required p_fus
3. Bisect on T_e to match target p_fus (JAX-compatible `fori_loop`)
4. Build full PlasmaState and PowerTable

### Forward mode

`tokamak_0d_forward`: Given all machine params including T_e, compute the resulting plasma state directly. No iteration needed.

Useful for parameter scans where you want to control T_e explicitly.

## Physics Functions

### DT fusion reactivity -- `sigma_v_dt(T_keV)`

Bosch-Hale analytic parameterization of <sigma*v>(T) for DT fusion. Valid 0.2--100 keV. Returns m^3/s.

### Plasma current -- `compute_plasma_current(a, kappa, B, R, q95)`

Cylindrical safety factor approximation:

```
I_p = 2*pi * a^2 * kappa * B / (mu_0 * R * q95)
```

Large-aspect-ratio limit. Shaped equilibria (triangularity, Shafranov shift) give 10--30% higher current at the same q95.

### Greenwald density -- `compute_greenwald_density(I_p_MA, a)`

Empirical density limit:

```
n_GW = I_p [MA] / (pi * a^2)    [10^20 m^-3]
```

Operating density is set by `f_GW`: n_e = f_GW * n_GW * 1e20.

### Fusion power -- `compute_fusion_power(n_e, T_keV, V_plasma)`

0D volume-averaged fusion power:

```
P_fus = (1/4) * n_e^2 * <sigma*v>(T) * E_fus * V
```

Factor 1/4 assumes 50/50 DT mix (n_D = n_T = n_e/2). E_fus = 17.58 MeV per reaction. Multiplication order avoids float32 overflow.

### Normalized beta -- `compute_beta_N(n_e, T_keV, B, I_p_MA, a)`

Troyon normalization:

```
beta_t = 2 * mu_0 * n_e * T [J] / B^2
beta_N = beta_t [%] * a * B / I_p [MA]
```

Ideal MHD stability requires beta_N < 3.5 (no-wall Troyon limit).

### Energy confinement -- `compute_tau_E_ipb98y2(...)`

IPB98(y,2) ELMy H-mode scaling:

```
tau_E = 0.0562 * I_p^0.93 * B^0.15 * n_e19^0.41 * P^-0.69
        * R^1.97 * epsilon^0.58 * kappa^0.78 * M^0.19
```

H-factor = tau_E_actual / tau_E_scaling. H > 1 means better-than-scaling confinement.

### Wall loading -- `compute_wall_loading(p_neutron_MW, fw_area)`

Neutron wall loading = P_neutron / A_fw [MW/m^2]. Design limit typically 3--5 MW/m^2.

### Divertor heat flux -- `compute_div_heat_flux(p_transport_MW, R, a, kappa, lambda_q)`

Simplified SOL flux-tube model:

```
q_div = P_transport / (2*pi*R * 2*lambda_q * f_expansion)
```

Default lambda_q = 2 mm (H-mode SOL power width), f_expansion = 5 (flux expansion to divertor).

### Geometry -- `_plasma_volume`, `_first_wall_area`

Elongated torus approximations:

```
V = 2 * pi^2 * R * a^2 * kappa
A = 4 * pi^2 * R * a * kappa
```

## Physics Limits

`PlasmaLimits` dataclass and `check_plasma_limits(state, limits)` check the plasma state against hard limits and produce warnings/errors:

| Limit | Default | Severity |
|-------|---------|----------|
| f_GW > 1.0 | -- | error |
| beta_N > beta_N_max | 3.5 | error |
| q95 < q95_min | 2.0 | error |
| wall_loading > max | 5.0 MW/m^2 | warning |
| div_heat_flux > max | 10.0 MW/m^2 | warning |

These are discrete checks on concrete values (not JAX-traced).

## Disruption Penalty Model

Converts proximity to stability limits into economic penalties on component lifetime and plant availability.

### Stability margins to disruption frequency

Each stability boundary contributes a partial disruption rate that increases exponentially near the limit:

```
margin_GW   = 1 - f_GW
margin_beta = 1 - beta_N / beta_N_max
margin_kink = 1 - q95_min / q95

rate_i = rate_base * exp(-steepness * margin_i)
disruption_rate = rate_GW + rate_beta + rate_kink
```

Channels are independent -- any one can trigger a disruption.

### Parameters

| Parameter | Default | Units | Physical basis |
|-----------|---------|-------|----------------|
| `rate_base` | 0.1 | disrupt/FPY | Baseline rate far from all limits. ITER targets <5% disruption rate; steady-state power plants expected lower. |
| `steepness` | 15.0 | -- | Exponential steepness. At margin=0.15: ~0.01/FPY. At margin=0.05: ~0.047/FPY. At margin=0: rate_base per channel. |
| `damage_per_disruption` | 0.02 | fraction | Each disruption consumes 2% of component life. |
| `downtime_per_disruption` | 72.0 | hours | Recovery time per event. Range: 24h (fast) to 168h (inspection). |

### Effective lifetime and availability

```
effective_core_lifetime = core_lifetime / (1 + disruption_rate * damage * core_lifetime)
effective_availability  = availability * (1 - disruption_rate * downtime / 8760)
```

These replace the raw values in CAS70/LCOE calculations when the 0D model is active.

### Example calculations

| Operating point | f_GW | beta_N | q95 | Rate (/FPY) | Eff. lifetime | Eff. avail |
|----------------|------|--------|-----|-------------|---------------|------------|
| Safe | 0.70 | 2.0 | 4.0 | 0.001 | 4.999 | 0.8500 |
| Moderate | 0.85 | 2.8 | 3.5 | 0.016 | 4.99 | 0.8499 |
| Aggressive | 0.95 | 3.3 | 2.5 | 0.094 | 4.91 | 0.849 |
| At limits | 1.00 | 3.5 | 2.0 | 0.300 | 4.85 | 0.847 |

The model is intentionally smooth -- no cliff at the limit, just a steepening gradient.

### Configuration

Defaults in `src/costingfe/data/defaults/mfe_tokamak.yaml`:

```yaml
disruption_rate_base: 0.1
disruption_steepness: 15.0
disruption_damage: 0.02
disruption_downtime: 72.0
```

Override via `CostModel.forward(**overrides)`. To disable while keeping the 0D model, set `disruption_damage=0.0` and `disruption_downtime=0.0`.

### Integration into CostModel

In `model.py forward()`, applied between capital cost computation and CAS70/LCOE when `_plasma_state is not None`:

```python
# Steady-state MFE (tokamak, stellarator, mirror, steady FRC) derives the
# core lifetime from the neutron wall loading: core_lt =
# clamp(fluence_limit(fuel) / q_n, 0.5 FPY, plant life). IFE/MIF keep the
# fixed cc.core_lifetime(self.fuel) constant.
q_n = pt.p_neutron / max(geo.firstwall_area, 1e-6)
core_lt = _core_lifetime_fpy(cc, self.fuel, q_n, lifetime_yr, availability)
avail_eff = availability
if self._plasma_state is not None:
    dm = DisruptionModel(
        rate_base=params.get("disruption_rate_base", 0.1),
        steepness=params.get("disruption_steepness", 15.0),
        damage_per_disruption=params.get("disruption_damage", 0.02),
        downtime_per_disruption=params.get("disruption_downtime", 72.0),
    )
    core_lt, avail_eff = apply_disruption_penalty(
        core_lt, availability, self._plasma_state.disruption_rate, dm,
    )
```

When `use_0d_model=False`, `_plasma_state` is `None` -- no penalty, full backward compatibility.

## Radial Build Derivation

`derive_radial_build(fuel, **overrides)` provides physics-based default thicknesses for blanket, shield, structure, and vessel by fuel type:

| Fuel | Blanket | HT Shield | Structure | Vessel |
|------|---------|-----------|-----------|--------|
| DT | 1.0 m | 0.5 m | 0.20 m | 0.20 m |
| DD | 0.5 m | 0.3 m | 0.18 m | 0.15 m |
| DHe3 | 0.0 m | 0.1 m | 0.15 m | 0.10 m |
| pB11 | 0.0 m | 0.02 m | 0.15 m | 0.10 m |

Aneutronic fuels (DHe3, pB11) need no breeding blanket. User overrides take precedence.

## Caveats

- **Inverse mode interaction**: Changing f_GW forces the solver to adjust T_e to match the target p_net, which changes beta_N. Lowering f_GW doesn't necessarily reduce disruption risk -- the temperature increase may push beta_N above the Troyon limit. Forward mode is better for isolating the effect of individual stability margins.
- **Cylindrical approximation**: The plasma current formula uses the large-aspect-ratio cylindrical q. Real shaped equilibria produce 10--30% higher current at the same q95. The model is internally consistent but I_p values should not be compared directly to experimental tokamak data without correction.
- **Flat profiles**: The 0D fusion power formula uses volume-averaged quantities. Peaked density/temperature profiles produce lower fusion power than the 0D estimate by factors of 0.3--0.5 (profile correction not applied).
- **Simplified divertor model**: The SOL heat flux model is a single-channel flux-tube estimate. Real divertor geometries, detachment, and radiation distribute heat more favorably. The model gives an upper bound.
- **Smooth beyond limits**: The disruption model produces finite rates even when margins are negative. It does not impose a hard cutoff. `check_plasma_limits` still provides discrete warnings/errors.
- **JAX differentiable**: All functions use `jnp` only, so gradients of LCOE w.r.t. plasma parameters naturally include all physics contributions.

## Tests

`tests/test_tokamak.py`:

| Test class | Coverage |
|------------|----------|
| `TestBoschHale` | Monotonicity, known values at 15/65 keV, JAX grad |
| `TestPlasmaCurrentDensity` | ITER-like I_p, Greenwald density formula |
| `TestFusionPower` | ITER-like fusion power range |
| `TestIPB98` | ITER confinement time range |
| `TestWallAndDivertor` | Physical ranges for wall loading and divertor heat flux |
| `TestForwardMode` | Returns PlasmaState, CATF-like parameters |
| `TestInverseMode` | Forward-inverse roundtrip recovers T_e |
| `TestPhysicsLimits` | Greenwald, Troyon, q95 violations; wall loading warning; clean state |
| `TestRadialBuild` | DT thicker than pB11, overrides, aneutronic no blanket |
| `TestEndToEnd` | CostModel with 0D inverse and forward modes produce valid LCOE |
| `TestBackwardCompat` | use_0d_model=False identical, non-tokamak unaffected, IFE unaffected |
| `TestJAXAutodiff` | Finite gradients through sigma_v, fusion power, beta_N |
| `TestDisruptionRate` | Safe negligible, aggressive significant, at-limit = 3*rate_base, monotonicity, JAX grad |
| `TestDisruptionPenalty` | Negligible penalty at safe point, visible penalty at aggressive, end-to-end LCOE increase, backward compat |

## References

### DT fusion cross section (`sigma_v_dt`)

- H.-S. Bosch and G.M. Hale, "Improved formulas for fusion cross-sections and thermal reactivities," *Nuclear Fusion* 32(4), 611 (1992). doi:10.1088/0029-5515/32/4/I07. Coefficients `_BH_C1`--`_BH_C7`, Gamow constant `_BH_BG`, and reduced mass `_BH_MRC2` from Table IV. Valid 0.2--100 keV. DT fusion energy E_fus = 17.58 MeV (3.52 MeV alpha + 14.06 MeV neutron) from the same source.

### Plasma current and safety factor (`compute_plasma_current`)

- J.P. Freidberg, *Plasma Physics and Fusion Energy*, Cambridge University Press (2007), Ch. 11. Cylindrical safety factor approximation q = 2*pi*a^2*kappa*B / (mu_0*R*I_p) for an elongated cross-section. Large-aspect-ratio limit.

- J. Wesson, *Tokamaks*, 4th edition, Oxford University Press (2011), Ch. 3. Safety factor q(r) from MHD equilibrium; relationship between q95, q_edge, and kink stability.

### Greenwald density limit (`compute_greenwald_density`)

- M. Greenwald, "Density limits in toroidal plasmas," *Plasma Physics and Controlled Fusion* 44(8), R27 (2002). doi:10.1088/0741-3335/44/8/201. Empirical scaling n_GW = I_p [MA] / (pi * a^2) in units of 10^20 m^-3. Proposed mechanisms include edge radiation collapse and MHD island overlap.

### Fusion power (`compute_fusion_power`)

- J.P. Freidberg, *Plasma Physics and Fusion Energy*, Cambridge University Press (2007), Ch. 3. 0D formula P_fus = (1/4) * n_e^2 * <sigma*v> * E_fus * V; factor 1/4 for 50/50 DT mix. Profile correction factors discussed but not applied in this implementation.

### Normalized beta (`compute_beta_N`)

- F. Troyon et al., "MHD-limits to plasma confinement," *Plasma Physics and Controlled Fusion* 26(1A), 209 (1984). doi:10.1088/0741-3335/26/1A/319. Defines beta_N = beta_t [%] * a*B / I_p [MA]. Ideal MHD stability boundary at beta_N ~ 2.8--3.5 depending on profiles and wall distance.

- F. Troyon and R. Gruber, "A semi-empirical scaling law for the beta limit in tokamaks," *Physics Letters A* 110(1), 29 (1985). doi:10.1016/0375-9601(85)90227-5. Confirms scaling across wider parameter range; normalization collapses stability boundary across machine sizes.

### Energy confinement scaling (`compute_tau_E_ipb98y2`)

- ITER Physics Expert Group on Confinement and Transport, "Chapter 2: Plasma confinement and transport," *Nuclear Fusion* 39(12), 2175 (1999). doi:10.1088/0029-5515/39/12/302. IPB98(y,2) ELMy H-mode scaling. Coefficients from Table 5. Fitted to multi-machine database.

### SOL power width and divertor heat flux (`compute_div_heat_flux`)

- T. Eich et al., "Scaling of the tokamak near the scrape-off layer H-mode power width and implications for ITER," *Nuclear Fusion* 53(9), 093031 (2013). doi:10.1088/0029-5515/53/9/093031. Multi-machine scaling lambda_q ~ 1--3 mm for H-mode. Default lambda_q = 2 mm. Flux-tube heat flux model with f_expansion ~ 5.

### Elongated torus geometry (`_plasma_volume`, `_first_wall_area`)

- J.P. Freidberg, *Plasma Physics and Fusion Energy*, Cambridge University Press (2007), Ch. 11. V = 2*pi^2*R*a^2*kappa, A = 4*pi^2*R*a*kappa. Zeroth-order forms; triangularity corrections O(delta^2) neglected.

### Stability limits and kink mode (disruption model)

- J. Wesson, *Tokamaks*, 4th edition, Oxford University Press (2011), Ch. 6. External kink mode unstable for q_edge < 2.

### Disruption statistics and rates

- P.C. de Vries et al., "Survey of disruption causes at JET," *Nuclear Fusion* 51(5), 053018 (2011). doi:10.1088/0029-5515/51/5/053018. Analysis of ~2500 JET disruptions; density limit, beta limit, and locked modes as dominant causes. Disruption rates 5--20% per pulse. Basis for per-channel rate model.

- T.C. Hender et al., "Chapter 3: MHD stability, operational limits and disruptions," *Nuclear Fusion* 47(6), S128 (2007). doi:10.1088/0029-5515/47/6/S03. ITER Physics Basis chapter; <5% disruption rate target for ITER. Basis for rate_base = 0.1 disrupt/FPY.

- J.W. Berkery et al., "A reduced resistive wall mode kinetic stability model for disruption forecasting," *Physics of Plasmas* 24, 056103 (2017). doi:10.1063/1.4977464. Disruption probability increases exponentially near stability boundaries; supports the exponential rate model.

### Disruption loads and damage

- M. Lehnen et al., "Disruptions in ITER and strategies for their control and mitigation," *Journal of Nuclear Materials* 463, 39--48 (2015). doi:10.1016/j.jnucmat.2014.10.075. Thermal loads up to 60 MJ/m^2 unmitigated; ITER first wall designed for ~3000 mitigated disruptions. Basis for damage_per_disruption = 0.02.

- V. Riccardo and A. Loarte, "Timescale and magnitude of plasma thermal energy loss before and during disruptions in JET," *Nuclear Fusion* 45(11), 1427 (2005). doi:10.1088/0029-5515/45/11/019. Thermal quench energy deposition measurements; component fatigue life basis.

- S.N. Gerasimov et al., "JET disruption studies in support of ITER," *Nuclear Fusion* 55(11), 113006 (2015). doi:10.1088/0029-5515/55/11/113006. Halo current and sideways force measurements during VDEs; cumulative structural fatigue basis.

### Disruption recovery and downtime

- ITER Organization, "ITER Research Plan within the Staged Approach," ITR-18-003 (2018). Plasma re-establishment 1--4h after controlled shutdown, 24--168h after disruption. Default 72h is mid-range for mitigated disruptions in a power plant.

### Power plant availability

- T. Ihli et al., "Review of blanket designs for advanced fusion reactors," *Fusion Engineering and Design* 83(7-9), 912--919 (2008). doi:10.1016/j.fusengdes.2008.07.039. Blanket/first-wall lifetime as dominant driver of scheduled maintenance downtime.
