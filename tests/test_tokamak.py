"""Tests for the 0D tokamak plasma model."""

import jax
import jax.numpy as jnp

from costingfe import ConfinementConcept, CostModel, Fuel, PlasmaState
from costingfe.layers.tokamak import (
    apply_disruption_penalty,
    check_plasma_limits,
    compute_beta_N,
    compute_disruption_rate,
    compute_div_heat_flux,
    compute_fusion_power,
    compute_greenwald_density,
    compute_plasma_current,
    compute_tau_E_ipb98y2,
    compute_wall_loading,
    derive_radial_build,
    sigma_v_dt,
    tokamak_0d_forward,
    tokamak_0d_inverse,
)

# Default fuel fraction params for DT tests
_DT_FUEL_FRAC = dict(
    dd_f_T=0.969,
    dd_f_He3=0.689,
    dhe3_f_T=0.5,
    dhe3_f_He3=0.1,
    pb11_f_alpha_n=0.0,
    pb11_f_p_n=0.0,
    T_i_over_T_e=1.0,
    dhe3_fuel_ratio=1.0,
    pb11_fuel_ratio=0.15,
)

# Default physics params for power balance tests
_PB_PHYSICS = dict(
    n_e=1.0e20,
    T_e=15.0,
    Z_eff=1.5,
    plasma_volume=500.0,
    B=5.0,
    dd_f_T=0.969,
    dd_f_He3=0.689,
    dhe3_dd_frac=0.07,
    dhe3_f_T=0.5,
    dhe3_f_He3=0.1,
    pb11_f_alpha_n=0.0,
    pb11_f_p_n=0.0,
    wall_material=None,
    T_edge=0.05,
    tau_ratio=3.0,
    fw_area=0.0,
    R_major=0.0,
    a_minor=0.0,
    kappa=1.7,
    R_w=0.6,
)


# ---------------------------------------------------------------------------
# 1. Bosch-Hale DT reactivity
# ---------------------------------------------------------------------------
class TestBoschHale:
    def test_monotonic_below_peak(self):
        """<sigma*v> should increase monotonically from 1 to ~65 keV."""
        temps = jnp.array([1.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 65.0])
        svs = jnp.array([sigma_v_dt(t) for t in temps])
        for i in range(len(svs) - 1):
            assert svs[i] < svs[i + 1], f"Not monotonic at T={temps[i + 1]}"

    def test_known_value_15keV(self):
        """At 15 keV, <sigma*v> ~ 2-3e-22 m^3/s."""
        sv = float(sigma_v_dt(15.0))
        assert 1e-22 < sv < 5e-22, f"sigma_v(15 keV) = {sv}"

    def test_known_value_65keV(self):
        """At 65 keV (near peak), <sigma*v> ~ 8-9e-22 m^3/s."""
        sv = float(sigma_v_dt(65.0))
        assert 5e-22 < sv < 1.2e-21, f"sigma_v(65 keV) = {sv}"

    def test_jax_differentiable(self):
        """Should be differentiable with jax.grad."""
        grad_fn = jax.grad(lambda T: sigma_v_dt(T))
        g = grad_fn(15.0)
        assert jnp.isfinite(g)
        assert g > 0  # increasing at 15 keV


# ---------------------------------------------------------------------------
# 2. Plasma current and density
# ---------------------------------------------------------------------------
class TestPlasmaCurrentDensity:
    def test_iter_plasma_current(self):
        """ITER-like: R=6.2, a=2.0, kappa=1.7, B=5.3, q95=3.0.

        Cylindrical approximation gives ~10 MA (real ITER is 15 MA due to
        triangularity and shaping not captured by the 0D formula).
        """
        I_p = float(compute_plasma_current(a=2.0, kappa=1.7, B=5.3, R=6.2, q95=3.0))
        assert 8.0 < I_p < 12.0, f"ITER I_p = {I_p} MA"

    def test_greenwald_density(self):
        """n_GW = I_p / (pi * a^2) in 10^20 m^-3."""
        n_GW = float(compute_greenwald_density(I_p_MA=15.0, a=2.0))
        expected = 15.0 / (jnp.pi * 2.0**2)
        assert abs(n_GW - expected) < 0.01


# ---------------------------------------------------------------------------
# 3. Fusion power
# ---------------------------------------------------------------------------
class TestFusionPower:
    def test_iter_like_fusion_power(self):
        """ITER-like parameters should give hundreds of MW fusion.

        With n_e=1e20, T=13 keV, V=830 m^3, the 0D Bosch-Hale model
        gives ~1000+ MW (higher than real ITER's ~500 MW because the 0D
        model uses a flat profile rather than peaked profiles with
        profile correction factors).
        """
        p_fus = float(compute_fusion_power(n_e=1.0e20, T_i=13.0, V_plasma=830.0))
        assert 100 < p_fus < 3000, f"ITER-like P_fus = {p_fus} MW"


# ---------------------------------------------------------------------------
# 4. IPB98(y,2) confinement time
# ---------------------------------------------------------------------------
class TestIPB98:
    def test_iter_confinement(self):
        """ITER: tau_E ~ 3-4 s from IPB98(y,2)."""
        tau_E = float(
            compute_tau_E_ipb98y2(
                I_p_MA=15.0,
                B=5.3,
                n_e19=10.0,
                P_heat_MW=100.0,
                R=6.2,
                a=2.0,
                kappa=1.7,
                M=2.5,
            )
        )
        assert 2.0 < tau_E < 6.0, f"ITER tau_E = {tau_E} s"


# ---------------------------------------------------------------------------
# 5. Wall loading and divertor heat flux
# ---------------------------------------------------------------------------
class TestWallAndDivertor:
    def test_wall_loading_physical(self):
        """Wall loading should be in reasonable range for tokamaks."""
        wl = float(compute_wall_loading(p_neutron_MW=400.0, fw_area=700.0))
        assert 0.1 < wl < 5.0, f"wall loading = {wl} MW/m^2"

    def test_div_heat_flux_physical(self):
        """Divertor heat flux should be in plausible range."""
        dhf = float(
            compute_div_heat_flux(
                p_transport_MW=100.0,
                R=6.2,
                a=2.0,
                kappa=1.7,
                lambda_q=0.002,
            )
        )
        assert dhf > 0
        assert dhf < 200  # MW/m^2, very high but SOL model is simplified


# ---------------------------------------------------------------------------
# 6. Forward mode
# ---------------------------------------------------------------------------
class TestForwardMode:
    def test_returns_plasma_state(self):
        """tokamak_0d_forward should return a PlasmaState."""
        ps = tokamak_0d_forward(
            R=3.0,
            a=1.1,
            kappa=3.0,
            B=5.0,
            q95=3.5,
            f_GW=0.85,
            T_e=15.0,
            p_input=50.0,
            dhe3_dd_frac_pin=0.07,
            **_DT_FUEL_FRAC,
        )
        assert isinstance(ps, PlasmaState)
        assert ps.p_fus > 0
        assert ps.I_p > 0
        assert ps.n_e > 0
        assert ps.V_plasma > 0

    def test_catf_like_params(self):
        """CATF-like spherical tokamak should give reasonable fusion power."""
        ps = tokamak_0d_forward(
            R=3.0,
            a=1.1,
            kappa=3.0,
            B=5.0,
            q95=3.5,
            f_GW=0.85,
            T_e=15.0,
            p_input=50.0,
            dhe3_dd_frac_pin=0.07,
            **_DT_FUEL_FRAC,
        )
        assert ps.p_fus > 0
        assert ps.wall_loading > 0
        assert 0 < ps.f_GW <= 1.0


# ---------------------------------------------------------------------------
# 7. Inverse mode
# ---------------------------------------------------------------------------
class TestInverseMode:
    def test_roundtrip(self):
        """Forward -> get p_net -> inverse with that target -> recover T_e."""
        # Forward pass
        ps_fwd = tokamak_0d_forward(
            R=3.0,
            a=1.1,
            kappa=3.0,
            B=5.0,
            q95=3.5,
            f_GW=0.85,
            T_e=15.0,
            p_input=50.0,
            dhe3_dd_frac_pin=0.07,
            **_DT_FUEL_FRAC,
        )
        # Use the forward p_net as target for inverse
        from costingfe.layers.physics import mfe_forward_power_balance

        pb_physics = dict(_PB_PHYSICS)
        pb_physics["n_e"] = ps_fwd.n_e
        pb_physics["T_e"] = ps_fwd.T_e
        pb_physics["plasma_volume"] = ps_fwd.V_plasma
        pt_fwd = mfe_forward_power_balance(
            p_fus=ps_fwd.p_fus,
            fuel=Fuel.DT,
            p_input=50.0,
            mn=1.1,
            eta_th=0.46,
            eta_p=0.5,
            eta_pin=0.5,
            eta_de=0.85,
            f_sub=0.03,
            f_dec=0.0,
            p_coils=2.0,
            p_cool=13.7,
            p_pump=1.0,
            p_trit=10.0,
            p_house=4.0,
            p_cryo=0.5,
            **pb_physics,
        )
        p_net = float(pt_fwd.p_net)

        # Inverse pass
        ps_inv, pt_inv = tokamak_0d_inverse(
            p_net_target=p_net,
            R=3.0,
            a=1.1,
            kappa=3.0,
            B=5.0,
            q95=3.5,
            f_GW=0.85,
            dhe3_dd_frac=0.07,
            dhe3_dd_frac_pin=None,
            **_DT_FUEL_FRAC,
        )
        # T_e should be recovered approximately (radiation coupling
        # means the inverse path sees slightly different p_rad, so
        # allow 2 keV tolerance)
        assert abs(float(ps_inv.T_e) - 15.0) < 2.0, (
            f"Recovered T_e = {float(ps_inv.T_e):.2f}, expected ~15.0"
        )


# ---------------------------------------------------------------------------
# 8. Physics limits
# ---------------------------------------------------------------------------
class TestPhysicsLimits:
    def test_detect_greenwald_violation(self):
        """f_GW > 1.0 should produce an error."""
        ps = PlasmaState(
            I_p=15.0,
            n_GW=1.0,
            n_e=1.2e20,
            T_e=15.0,
            beta_N=2.0,
            tau_E=3.0,
            p_fus=500.0,
            p_alpha=100.0,
            p_rad=20.0,
            V_plasma=830.0,
            fw_area=700.0,
            q95=3.0,
            f_GW=1.1,
            wall_loading=0.5,
            div_heat_flux=5.0,
            H_factor=1.0,
        )
        issues = check_plasma_limits(ps)
        errors = [msg for sev, msg in issues if sev == "error"]
        assert any("Greenwald" in msg for msg in errors)

    def test_detect_troyon_violation(self):
        """beta_N > 3.5 should produce an error."""
        ps = PlasmaState(
            I_p=15.0,
            n_GW=1.0,
            n_e=1.0e20,
            T_e=15.0,
            beta_N=4.0,
            tau_E=3.0,
            p_fus=500.0,
            p_alpha=100.0,
            p_rad=20.0,
            V_plasma=830.0,
            fw_area=700.0,
            q95=3.0,
            f_GW=0.85,
            wall_loading=0.5,
            div_heat_flux=5.0,
            H_factor=1.0,
        )
        issues = check_plasma_limits(ps)
        errors = [msg for sev, msg in issues if sev == "error"]
        assert any("beta_N" in msg for msg in errors)

    def test_detect_low_q95(self):
        """q95 < 2.0 should produce an error."""
        ps = PlasmaState(
            I_p=15.0,
            n_GW=1.0,
            n_e=1.0e20,
            T_e=15.0,
            beta_N=2.0,
            tau_E=3.0,
            p_fus=500.0,
            p_alpha=100.0,
            p_rad=20.0,
            V_plasma=830.0,
            fw_area=700.0,
            q95=1.5,
            f_GW=0.85,
            wall_loading=0.5,
            div_heat_flux=5.0,
            H_factor=1.0,
        )
        issues = check_plasma_limits(ps)
        errors = [msg for sev, msg in issues if sev == "error"]
        assert any("q95" in msg for msg in errors)

    def test_wall_loading_warning(self):
        """High wall loading should produce a warning, not error."""
        ps = PlasmaState(
            I_p=15.0,
            n_GW=1.0,
            n_e=1.0e20,
            T_e=15.0,
            beta_N=2.0,
            tau_E=3.0,
            p_fus=500.0,
            p_alpha=100.0,
            p_rad=20.0,
            V_plasma=830.0,
            fw_area=700.0,
            q95=3.0,
            f_GW=0.85,
            wall_loading=6.0,
            div_heat_flux=5.0,
            H_factor=1.0,
        )
        issues = check_plasma_limits(ps)
        warnings_ = [msg for sev, msg in issues if sev == "warning"]
        assert any("wall loading" in msg for msg in warnings_)

    def test_clean_state_no_issues(self):
        """A well-behaved plasma should have no issues."""
        ps = PlasmaState(
            I_p=15.0,
            n_GW=1.0,
            n_e=1.0e20,
            T_e=15.0,
            beta_N=2.0,
            tau_E=3.0,
            p_fus=500.0,
            p_alpha=100.0,
            p_rad=20.0,
            V_plasma=830.0,
            fw_area=700.0,
            q95=3.0,
            f_GW=0.85,
            wall_loading=0.5,
            div_heat_flux=5.0,
            H_factor=1.0,
        )
        issues = check_plasma_limits(ps)
        assert len(issues) == 0


# ---------------------------------------------------------------------------
# 9. Radial build derivation
# ---------------------------------------------------------------------------
class TestRadialBuild:
    def test_dt_thicker_than_pb11(self):
        """DT should have thicker blanket and shield than pB11."""
        dt = derive_radial_build(Fuel.DT)
        pb11 = derive_radial_build(Fuel.PB11)
        assert dt["blanket_t"] > pb11["blanket_t"]
        assert dt["ht_shield_t"] > pb11["ht_shield_t"]

    def test_overrides_respected(self):
        """User overrides should take precedence."""
        rb = derive_radial_build(Fuel.DT, blanket_t=2.0)
        assert rb["blanket_t"] == 2.0

    def test_dd_intermediate(self):
        """DD should be between DT and DHe3."""
        dt = derive_radial_build(Fuel.DT)
        dd = derive_radial_build(Fuel.DD)
        dhe3 = derive_radial_build(Fuel.DHE3)
        assert dt["blanket_t"] > dd["blanket_t"] > dhe3["blanket_t"]

    def test_aneutronic_no_blanket(self):
        """DHe3 and pB11 should have zero blanket."""
        dhe3 = derive_radial_build(Fuel.DHE3)
        pb11 = derive_radial_build(Fuel.PB11)
        assert dhe3["blanket_t"] == 0.0
        assert pb11["blanket_t"] == 0.0


# ---------------------------------------------------------------------------
# 10. End-to-end: CostModel with 0D model
# ---------------------------------------------------------------------------
class TestEndToEnd:
    def test_forward_with_0d_produces_lcoe(self):
        """CostModel.forward(use_0d_model=True) should produce valid LCOE."""
        m = CostModel(ConfinementConcept.TOKAMAK, Fuel.DT)
        r = m.forward(
            1000,
            0.85,
            30,
            use_0d_model=True,
            q95=3.5,
            f_GW=0.85,
        )
        assert r.costs.lcoe > 0
        assert r.plasma_state is not None
        assert r.plasma_state.p_fus > 0
        assert r.plasma_state.I_p > 0

    def test_forward_0d_forward_mode(self):
        """Forward mode should also work end-to-end."""
        m = CostModel(ConfinementConcept.TOKAMAK, Fuel.DT)
        r = m.forward(
            1000,
            0.85,
            30,
            use_0d_model=True,
            **{"0d_mode": "forward"},
            q95=3.5,
            f_GW=0.85,
            T_e=15.0,
        )
        assert r.costs.lcoe > 0
        assert r.plasma_state is not None


# ---------------------------------------------------------------------------
# 11. Backward compatibility
# ---------------------------------------------------------------------------
class TestBackwardCompat:
    def test_0d_false_identical(self):
        """use_0d_model=False should give identical results to no 0D."""
        m = CostModel(ConfinementConcept.TOKAMAK, Fuel.DT)
        r_base = m.forward(1000, 0.85, 30)
        r_off = m.forward(1000, 0.85, 30, use_0d_model=False)
        assert r_base.costs.lcoe == r_off.costs.lcoe
        assert r_base.plasma_state is None
        assert r_off.plasma_state is None

    def test_non_tokamak_unaffected(self):
        """Stellarator should not be affected by 0D model flag."""
        m = CostModel(ConfinementConcept.STELLARATOR, Fuel.DT)
        r = m.forward(1000, 0.85, 30)
        assert r.plasma_state is None
        assert r.costs.lcoe > 0

    def test_ife_unaffected(self):
        """IFE should not be affected by 0D model."""
        m = CostModel(ConfinementConcept.LASER_IFE, Fuel.DT)
        r = m.forward(1000, 0.85, 30)
        assert r.plasma_state is None
        assert r.costs.lcoe > 0


# ---------------------------------------------------------------------------
# 12. JAX autodiff through 0D pipeline
# ---------------------------------------------------------------------------
class TestJAXAutodiff:
    def test_sigma_v_grad(self):
        """sigma_v_dt should have finite positive gradient at 15 keV."""
        g = jax.grad(sigma_v_dt)(15.0)
        assert jnp.isfinite(g)
        assert g > 0

    def test_fusion_power_grad(self):
        """Fusion power should have finite gradient w.r.t. T_e."""

        def p_fus_fn(T):
            return compute_fusion_power(n_e=1e20, T_i=T, V_plasma=500.0)

        g = jax.grad(p_fus_fn)(15.0)
        assert jnp.isfinite(g)

    def test_beta_N_grad(self):
        """beta_N should have finite gradient w.r.t. T_e."""

        def beta_fn(T):
            return compute_beta_N(
                n_e=1e20, T_e=T, T_i=T, n_i_frac=1.0, B=5.0, I_p_MA=15.0, a=2.0
            )

        g = jax.grad(beta_fn)(15.0)
        assert jnp.isfinite(g)


# ---------------------------------------------------------------------------
# 13. Disruption rate model
# ---------------------------------------------------------------------------
class TestDisruptionRate:
    def test_safe_point_negligible(self):
        """Safe operating point (far from limits) has negligible rate."""
        rate = float(compute_disruption_rate(f_GW=0.70, beta_N=2.0, q95=4.0))
        assert rate < 0.01, f"Safe point rate = {rate}"

    def test_aggressive_point_significant(self):
        """Aggressive operating point has significant rate."""
        rate = float(compute_disruption_rate(f_GW=0.95, beta_N=3.3, q95=2.5))
        assert rate > 0.05, f"Aggressive point rate = {rate}"

    def test_at_limit_rate(self):
        """All margins = 0 -> rate = 3 * rate_base."""
        rate = float(compute_disruption_rate(f_GW=1.0, beta_N=3.5, q95=2.0))
        assert abs(rate - 0.3) < 0.001, f"At-limit rate = {rate}, expected 0.3"

    def test_monotonicity_f_GW(self):
        """Higher f_GW -> higher disruption rate (other params fixed)."""
        r1 = float(compute_disruption_rate(f_GW=0.7, beta_N=2.5, q95=3.5))
        r2 = float(compute_disruption_rate(f_GW=0.85, beta_N=2.5, q95=3.5))
        r3 = float(compute_disruption_rate(f_GW=0.95, beta_N=2.5, q95=3.5))
        assert r1 < r2 < r3

    def test_jax_differentiable(self):
        """disruption_rate should be differentiable w.r.t. f_GW."""

        def rate_fn(f_GW):
            return compute_disruption_rate(f_GW=f_GW, beta_N=2.5, q95=3.5)

        g = jax.grad(rate_fn)(0.85)
        assert jnp.isfinite(g)
        assert g > 0  # rate increases with f_GW


class TestDisruptionPenalty:
    def test_safe_point_negligible_penalty(self):
        """Safe point: lifetime and availability barely change."""
        rate = float(compute_disruption_rate(f_GW=0.70, beta_N=2.0, q95=4.0))
        lt, av = apply_disruption_penalty(5.0, 0.85, rate)
        lt, av = float(lt), float(av)
        assert lt > 4.99, f"Effective lifetime = {lt}"
        assert av > 0.849, f"Effective availability = {av}"

    def test_aggressive_point_visible_penalty(self):
        """Aggressive point: visible reduction in lifetime."""
        rate = float(compute_disruption_rate(f_GW=0.95, beta_N=3.3, q95=2.5))
        lt, av = apply_disruption_penalty(5.0, 0.85, rate)
        lt, av = float(lt), float(av)
        assert lt < 5.0, f"Effective lifetime = {lt}"
        assert av < 0.85, f"Effective availability = {av}"

    def test_end_to_end_disruption_increases_lcoe(self):
        """Disruption penalty should increase LCOE vs zero-penalty baseline."""
        m = CostModel(ConfinementConcept.TOKAMAK, Fuel.DT)
        # With disruption penalty (default params)
        r_with = m.forward(
            1000,
            0.85,
            30,
            use_0d_model=True,
            q95=3.5,
            f_GW=0.85,
        )
        # Without disruption penalty (zero damage and downtime)
        r_without = m.forward(
            1000,
            0.85,
            30,
            use_0d_model=True,
            q95=3.5,
            f_GW=0.85,
            disruption_damage=0.0,
            disruption_downtime=0.0,
        )
        assert r_with.costs.lcoe > r_without.costs.lcoe, (
            f"Penalized LCOE {r_with.costs.lcoe:.2f} should be > "
            f"unpenalized LCOE {r_without.costs.lcoe:.2f}"
        )

    def test_backward_compat_no_0d(self):
        """use_0d_model=False -> no disruption penalty applied."""
        m = CostModel(ConfinementConcept.TOKAMAK, Fuel.DT)
        r1 = m.forward(1000, 0.85, 30)
        r2 = m.forward(1000, 0.85, 30, use_0d_model=False)
        assert r1.costs.lcoe == r2.costs.lcoe
        assert r1.plasma_state is None
