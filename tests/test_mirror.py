"""Tests for the 0D mirror physics model."""

import math

import jax
import jax.numpy as jnp
import pytest

from costingfe.defaults import load_costing_constants
from costingfe.layers.mirror import (
    _KEV_TO_J,
    _MU_0,
    _RHO_I_PREFACTOR,
    _V_THI_PREFACTOR,
    MirrorPlasmaState,
    SizingInfeasible,
    _density_from_f_beta,
    _density_from_surface_cap,
    _density_from_wall_cap,
    compute_ambipolar_potential,
    compute_tau_axial,
    compute_tau_classical,
    compute_tau_gas_dynamic,
    compute_tau_ii,
    compute_tau_pastukhov,
    compute_tau_radial,
    mirror_0d_forward,
    mirror_0d_inverse,
    mirror_aux_heating,
    mirror_size_from_power,
    net_electric_at_L,
)
from costingfe.layers.physics import OperatingPointInfeasible
from costingfe.layers.reactivity import n_i_over_n_e
from costingfe.model import CostModel, _core_lifetime_fpy
from costingfe.types import ConfinementConcept, Fuel

_N = 1.0e20  # m^-3
_TI = 20.0  # keV
_TE = 20.0

# Reference machine geometry (WHAM-class)
_L = 30.0  # m
_A = 0.3  # m
_B_MIN = 3.0  # T
_R_M = 10.0  # mirror ratio

# Fuel-mix kwargs (same defaults as multifuel tests)
_FRACS = dict(
    dd_f_T=0.969,
    dd_f_He3=0.689,
    dhe3_f_T=0.5,
    dhe3_f_He3=0.05,
    pb11_f_alpha_n=0.0,
    pb11_f_p_n=0.0,
)
_MIX = dict(dhe3_fuel_ratio=1.0, pb11_fuel_ratio=0.15)


def _forward(fuel=Fuel.DT, dhe3_dd_frac_pin=None, **kw):
    """Convenience wrapper for mirror_0d_forward at reference geometry."""
    args = {**_FRACS, **_MIX, **kw}
    return mirror_0d_forward(
        L=args.pop("L", _L),
        a=args.pop("a", _A),
        B_min=args.pop("B_min", _B_MIN),
        R_m=args.pop("R_m", _R_M),
        T_i=args.pop("T_i", _TI),
        T_e=args.pop("T_e", _TE),
        n_e=args.pop("n_e", _N),
        p_input=args.pop("p_input", 20.0),
        fuel=fuel,
        dhe3_dd_frac_pin=dhe3_dd_frac_pin,
        vacuum_t=args.pop("vacuum_t", 0.10),
        **args,
    )


class TestConfinement:
    def test_tau_ii_scaling(self):
        # tau_ii ~ T^1.5 / n
        t1 = float(compute_tau_ii(_N, _TI, 2.5))
        t2 = float(compute_tau_ii(_N, 2.0 * _TI, 2.5))
        t3 = float(compute_tau_ii(2.0 * _N, _TI, 2.5))
        assert t2 / t1 == pytest.approx(2.0**1.5, rel=1e-3)
        assert t3 / t1 == pytest.approx(0.5, rel=1e-3)

    def test_ambipolar_potential_magnitude(self):
        # e*phi = T_e * ln(sqrt(m_i/(2 pi m_e))) ~ 3-4 T_e for A = 2.5
        phi = float(compute_ambipolar_potential(_TE, 2.5))
        assert 2.5 * _TE < phi < 4.5 * _TE

    def test_pastukhov_beats_classical(self):
        # Electrostatic plugging is an exponential enhancement
        tii = compute_tau_ii(_N, _TI, 2.5)
        phi = compute_ambipolar_potential(_TE, 2.5)
        tc = float(compute_tau_classical(tii, R_m=10.0))
        tp = float(compute_tau_pastukhov(tii, R_m=10.0, phi_keV=phi, T_i=_TI))
        assert tp > 10.0 * tc

    def test_gas_dynamic_dominates_at_high_density(self):
        # High n, low T: mean free path < L -> tau_GD < tau_Pastukhov
        n_hi, t_lo = 5.0e20, 1.0
        tii = compute_tau_ii(n_hi, t_lo, 2.5)
        phi = compute_ambipolar_potential(t_lo, 2.5)
        tp = float(compute_tau_pastukhov(tii, R_m=30.0, phi_keV=phi, T_i=t_lo))
        tgd = float(compute_tau_gas_dynamic(R_m=30.0, L=20.0, T_i=t_lo, A=2.5))
        assert tgd < tp

    def test_radial_subdominant_at_reference(self):
        tii = compute_tau_ii(_N, _TI, 2.5)
        tr = float(compute_tau_radial(tii, a=0.5, T_i=_TI, A=2.5, B_min=3.0))
        phi = compute_ambipolar_potential(_TE, 2.5)
        tp = float(compute_tau_pastukhov(tii, R_m=10.0, phi_keV=phi, T_i=_TI))
        assert tr > tp  # radial losses subdominant in a well-confined mirror

    def test_jit_matches_eager(self):
        def chain(n, T):
            tii = compute_tau_ii(n, T, 2.5)
            phi = compute_ambipolar_potential(T, 2.5)
            return compute_tau_pastukhov(tii, R_m=10.0, phi_keV=phi, T_i=T)

        eager = float(chain(_N, _TI))
        jitted = float(jax.jit(chain)(_N, _TI))
        assert jnp.isfinite(jitted)
        assert jitted == pytest.approx(eager, rel=1e-4)

    def test_differentiable(self):
        g = float(jax.grad(lambda T: compute_tau_ii(_N, T, 2.5))(_TI))
        assert jnp.isfinite(g)
        assert g > 0.0

    def test_v_thi_prefactor_matches_constants(self):
        # Pin _V_THI_PREFACTOR against an independent float64 derivation.
        # v_thi = sqrt(2 * T_keV * KEV_TO_J / (A * m_p))
        # = _V_THI_PREFACTOR * sqrt(T_keV / A)
        _EV = 1.602176634e-19
        _M_P = 1.67262192369e-27
        expected = math.sqrt(2.0 * _EV * 1e3 / _M_P)
        assert float(_V_THI_PREFACTOR) == pytest.approx(expected, rel=1e-3)
        # Cross-check: compute_tau_gas_dynamic agrees with the from-scratch formula.
        T_i, A, R_m, L = 10.0, 2.5, 15.0, 30.0
        v_thi_ref = math.sqrt(2.0 * T_i * _EV * 1e3 / (A * _M_P))
        tau_ref = R_m * L / v_thi_ref
        tau_fn = float(compute_tau_gas_dynamic(R_m=R_m, L=L, T_i=T_i, A=A))
        assert tau_fn == pytest.approx(tau_ref, rel=1e-3)

    def test_rho_i_prefactor_matches_constants(self):
        # Pin _RHO_I_PREFACTOR against an independent float64 derivation.
        # rho_i = sqrt(2 * A * m_p * T_keV * KEV_TO_J) / (e * B)
        # = _RHO_I_PREFACTOR * sqrt(A * T_keV) / B
        _EV = 1.602176634e-19
        _M_P = 1.67262192369e-27
        expected = math.sqrt(2.0 * _M_P * _EV * 1e3) / _EV
        assert float(_RHO_I_PREFACTOR) == pytest.approx(expected, rel=1e-3)
        # Cross-check: compute_tau_radial agrees with the from-scratch formula.
        n, T_i, A, B, a = _N, _TI, 2.5, 3.0, 0.5
        tii = float(compute_tau_ii(n, T_i, A))
        rho_ref = math.sqrt(2.0 * A * _M_P * T_i * _EV * 1e3) / (_EV * B)
        tau_ref = (a / rho_ref) ** 2 * tii
        tau_fn = float(compute_tau_radial(tii, a=a, T_i=T_i, A=A, B_min=B))
        assert tau_fn == pytest.approx(tau_ref, rel=1e-3)


class TestForward:
    def test_returns_plasma_state(self):
        ps = _forward()
        assert isinstance(ps, MirrorPlasmaState)

    def test_positive_p_fus_tau_p_tau_E(self):
        ps = _forward()
        assert float(ps.p_fus) > 0.0
        assert float(ps.tau_p) > 0.0
        assert float(ps.tau_E) > 0.0

    def test_tau_E_less_than_tau_p(self):
        # Escaping particles are preferentially energetic, so tau_E < tau_p
        ps = _forward()
        assert float(ps.tau_E) < float(ps.tau_p)

    def test_beta_closed_form(self):
        # beta = 2 mu_0 n_e (T_e + n_i_frac * T_i) KEV_TO_J / B_min^2
        ps = _forward()
        MU_0 = 1.25663706127e-06
        KEV_TO_J = 1.602176634e-19 * 1e3
        n_i_frac = 1.0  # DT: n_i/n_e = 1
        beta_ref = 2.0 * MU_0 * _N * (_TE + n_i_frac * _TI) * KEV_TO_J / _B_MIN**2
        assert float(ps.beta) == pytest.approx(beta_ref, rel=1e-4)

    def test_f_axial_derived_in_0_1(self):
        ps = _forward()
        f = float(ps.f_axial_derived)
        assert 0.0 < f < 1.0

    def test_f_axial_derived_decreases_with_R_m(self):
        # Higher R_m increases tau_axial while tau_radial is unchanged,
        # so the axial loss fraction decreases.
        ps_lo = _forward(R_m=5.0)
        ps_hi = _forward(R_m=30.0)
        assert float(ps_hi.f_axial_derived) < float(ps_lo.f_axial_derived)

    def test_energy_bookkeeping(self):
        # P_end + P_radial = W_th / tau_E to 1e-5 relative tolerance
        ps = _forward()
        p_total = float(ps.p_end) + float(ps.p_radial)
        KEV_TO_J = 1.602176634e-19 * 1e3
        V = math.pi * _A**2 * _L
        n_i_frac = 1.0  # DT
        W_th = 1.5 * _N * (_TE + n_i_frac * _TI) * KEV_TO_J * V * 1e-6  # MW
        p_ref = W_th / float(ps.tau_E)
        assert p_total == pytest.approx(p_ref, rel=1e-5)

    def test_dhe3_forward_populates_frac_eff(self):
        ps = _forward(fuel=Fuel.DHE3, T_i=70.0, T_e=70.0)
        assert 0.0 < float(ps.dhe3_dd_frac_eff) < 1.0

    def test_dhe3_pin_overrides_frac_eff(self):
        ps = _forward(fuel=Fuel.DHE3, T_i=70.0, T_e=70.0, dhe3_dd_frac_pin=0.25)
        assert float(ps.dhe3_dd_frac_eff) == pytest.approx(0.25)

    def test_dt_frac_eff_is_zero(self):
        ps = _forward(fuel=Fuel.DT)
        assert float(ps.dhe3_dd_frac_eff) == 0.0

    @pytest.mark.parametrize(
        "fuel,T_i,T_e,extra",
        [
            (Fuel.DT, 20.0, 20.0, {}),
            (Fuel.DD, 30.0, 30.0, {}),
            (Fuel.DHE3, 70.0, 70.0, {}),
            (Fuel.PB11, 200.0, 200.0, {}),
        ],
    )
    def test_jit_equals_eager_all_fuels(self, fuel, T_i, T_e, extra):
        def run(n_e):
            return mirror_0d_forward(
                L=_L,
                a=_A,
                B_min=_B_MIN,
                R_m=_R_M,
                T_i=T_i,
                T_e=T_e,
                n_e=n_e,
                p_input=20.0,
                fuel=fuel,
                dhe3_dd_frac_pin=None,
                vacuum_t=0.10,
                **_FRACS,
                **_MIX,
                **extra,
            ).p_fus

        eager_val = float(run(jnp.array(_N)))
        jit_val = float(jax.jit(run)(jnp.array(_N)))
        assert jnp.isfinite(jit_val)
        assert jit_val == pytest.approx(eager_val, rel=1e-4)

    def test_wall_loading_positive(self):
        ps = _forward()
        assert float(ps.wall_loading) >= 0.0

    def test_collisionality_positive(self):
        ps = _forward()
        assert float(ps.collisionality) > 0.0

    def test_volumes_correct(self):
        ps = _forward()
        V_ref = math.pi * _A**2 * _L
        # fw_area basis moved to a + vacuum_t (default 0.10 in _forward helper)
        fw_ref = 2.0 * math.pi * (_A + 0.10) * _L
        assert float(ps.V_plasma) == pytest.approx(V_ref, rel=1e-5)
        assert float(ps.fw_area) == pytest.approx(fw_ref, rel=1e-5)

    def test_all_confinement_times_positive(self):
        ps = _forward()
        assert float(ps.tau_classical) > 0.0
        assert float(ps.tau_Pastukhov) > 0.0
        assert float(ps.tau_GD) > 0.0
        assert float(ps.tau_p) > 0.0
        assert float(ps.tau_E) > 0.0

    def test_p_alpha_positive_for_dt(self):
        ps = _forward()
        assert float(ps.p_alpha) > 0.0

    def test_phi_positive(self):
        ps = _forward()
        assert float(ps.phi) > 0.0

    def test_f_rad_fus_proxy(self):
        # p_rad = f_rad_fus * p_fus (unclamped), then clamped to p_alpha.
        # Use f_rad_fus=0.1 for DT (p_alpha ~ 0.2 * p_fus) so clamp is inactive.
        f = 0.1
        ps = _forward(f_rad_fus=f)
        assert float(ps.p_rad) == pytest.approx(f * float(ps.p_fus), rel=1e-5)
        # Confirm clamp was not active (p_rad < p_alpha)
        assert float(ps.p_rad) < float(ps.p_alpha)

    def test_fw_area_matches_geometry_layer(self):
        # Cross-layer consistency: the 0D state's first-wall area must equal
        # the geometry layer's (2 pi (a + vacuum_t) L), closing the audit
        # finding that the 0D diagnostic sat on the bare plasma surface.
        from costingfe.layers.geometry import RadialBuild, compute_geometry

        rb = RadialBuild(plasma_t=1.5, chamber_length=20.0, vacuum_t=0.10)
        geo = compute_geometry(rb, ConfinementConcept.MIRROR)
        ps = _forward(L=20.0, a=1.5, vacuum_t=0.10)
        assert float(ps.fw_area) == pytest.approx(geo.firstwall_area, rel=1e-6)

    def test_state_reports_q_surface(self):
        # q_surface = (p_rad + p_radial) / fw_area, computed from state fields.
        ps = _forward()
        p_rad = float(ps.p_rad)
        p_radial = float(ps.p_radial)
        fw_area = float(ps.fw_area)
        expected = (p_rad + p_radial) / fw_area
        assert float(ps.q_surface) == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Shared inverse kwargs
# ---------------------------------------------------------------------------
# GDT/WHAM-plausible test machine for inverse mode:
#   L=50 m central cell, a=0.4 m, B_min=3 T (midplane), R_m=20, n_e=1e20
#   m^-3, T_e=10 keV. Forward scan confirms p_fus ~ 77-120 MW at T_i=20-30
#   keV; the required p_fus for p_net=20 MW is ~86 MW, well inside the DT
#   bracket [2, 80] keV (solved T_i ~ 22 keV). Beta at solution ~ 0.14,
#   comfortably below the default beta_max=0.5.
_INV_L = 50.0
_INV_A = 0.4
_INV_B = 3.0
_INV_R_M = 20.0
_INV_N = 1.0e20
_INV_TE = 10.0
_INV_BETA_MAX = 0.5

# Power-balance kwargs: lighter plant loads typical for a mirror concept
# (superconducting coils, no divertor, simpler tritium system). eta_pin=0.7
# reflects better NBI wall-plug efficiency than the tokamak default.
_PB_KWARGS = dict(
    p_input=10.0,
    mn=1.1,
    eta_th=0.40,
    eta_p=0.5,
    eta_pin=0.7,
    eta_de=0.85,
    f_sub=0.05,
    f_dec=0.3,
    p_coils=1.0,
    p_cool=2.0,
    p_pump=1.0,
    p_trit=2.0,
    p_house=2.0,
    p_cryo=0.5,
)


def _inverse(
    fuel=Fuel.DT,
    dhe3_dd_frac_pin=None,
    f_rad_fus=None,
    enforce_plasma_limits=True,
    **kw,
):
    """Convenience wrapper for mirror_0d_inverse at the reference inverse machine."""
    return mirror_0d_inverse(
        p_net_target=kw.pop("p_net_target", 100.0),
        L=kw.pop("L", _INV_L),
        a=kw.pop("a", _INV_A),
        B_min=kw.pop("B_min", _INV_B),
        R_m=kw.pop("R_m", _INV_R_M),
        n_e=kw.pop("n_e", _INV_N),
        T_e=kw.pop("T_e", _INV_TE),
        fuel=fuel,
        beta_max=kw.pop("beta_max", _INV_BETA_MAX),
        # high defaults; inverse tests are not wall- or surface-capped
        q_wall_max=kw.pop("q_wall_max", 50.0),
        q_surface_max=kw.pop("q_surface_max", 50.0),
        p_aux_floor=kw.pop("p_aux_floor", 2.0),
        enforce_plasma_limits=enforce_plasma_limits,
        dhe3_dd_frac=0.131,
        dhe3_dd_frac_pin=dhe3_dd_frac_pin,
        vacuum_t=kw.pop("vacuum_t", 0.10),
        f_rad_fus=f_rad_fus,
        **_FRACS,
        **_MIX,
        **{**_PB_KWARGS, **kw},
    )


class TestInverse:
    def test_dt_inverse_p_net_converges(self):
        # DT inverse at a GDT/WHAM-plausible machine:
        #   L=50 m, a=0.4 m, B=3 T, R_m=20, n_e=1e20, T_e=10 keV.
        # Forward scan: p_fus ~ 86 MW at T_i ~ 22 keV for p_net=20 MW.
        # The solved state should deliver p_net within 5% of the target.
        state, pt = _inverse(fuel=Fuel.DT, p_net_target=20.0)
        assert isinstance(state, MirrorPlasmaState)
        assert float(pt.p_net) == pytest.approx(20.0, rel=0.05)
        assert state.beta < 0.5  # beta_max for this test machine

    def test_dt_inverse_T_i_in_bracket(self):
        # The solved T_i must lie inside the DT fuel bracket [2, 80] keV.
        state, pt = _inverse(fuel=Fuel.DT, p_net_target=20.0)
        T_i = float(state.T_i)
        assert 2.0 < T_i < 80.0

    def test_beta_over_max_raises(self):
        # Force implied beta > beta_max by using a very tight beta_max.
        # At the reference machine T_i ~ 22 keV -> beta ~ 0.14; beta_max=0.001
        # is well below the implied value, so OperatingPointInfeasible must fire.
        with pytest.raises(OperatingPointInfeasible, match=r"beta = "):
            _inverse(fuel=Fuel.DT, p_net_target=20.0, beta_max=0.001)

    def test_enforce_plasma_limits_false_returns_state(self):
        # With the tight beta_max that would raise, enforce_plasma_limits=False
        # should return the implied (infeasible) point without raising.
        state, pt = _inverse(
            fuel=Fuel.DT,
            p_net_target=20.0,
            beta_max=0.001,
            enforce_plasma_limits=False,
        )
        assert isinstance(state, MirrorPlasmaState)
        # beta > beta_max (that was the whole point)
        assert float(state.beta) > 0.001

    def test_dhe3_inverse_converges(self):
        # D-He3 inverse with f_rad_fus=0.24 proxy (standard advanced-fuel proxy).
        # Machine: L=50 m, a=0.4 m, B=5 T, R_m=30, n_e=3e20, T_e=30 keV.
        # Forward scan: T_i ~ 60 keV for p_net=20 MW (inside [20, 100] bracket).
        # Separate power balance to reflect near-aneutronic plant (mn=1.02, f_dec=0.6).
        _pb = {**_PB_KWARGS, "mn": 1.02, "f_dec": 0.6}  # near-aneutronic deltas
        state, pt = mirror_0d_inverse(
            p_net_target=20.0,
            L=50.0,
            a=0.4,
            B_min=5.0,
            R_m=30.0,
            n_e=3.0e20,
            T_e=30.0,
            fuel=Fuel.DHE3,
            beta_max=0.5,
            q_wall_max=50.0,  # high cap; this test is not wall-bound
            q_surface_max=50.0,  # high cap; this test is not surface-bound
            p_aux_floor=2.0,
            dhe3_dd_frac=0.131,
            dhe3_dd_frac_pin=None,
            vacuum_t=0.10,
            f_rad_fus=0.24,
            **_FRACS,
            **_MIX,
            **_pb,
        )
        assert isinstance(state, MirrorPlasmaState)
        assert float(pt.p_net) == pytest.approx(20.0, rel=0.05)


# ---------------------------------------------------------------------------
# Model-integration tests (CostModel dispatch)
# ---------------------------------------------------------------------------

# Pinned LCOE for the mirror default path (use_0d_model=False).
# Captured from branch tip aa7eef8 before any model.py changes; guards that
# the 0D opt-in default does NOT alter the existing non-0D path.
# re-pinned 2026-06-13: fluence-based CAS72 basis change, see
# wall_limits_and_fluence.md (was 93.643616).
# $/MWh at 500 MW, avail=0.87, lifetime=40 yr
_MIRROR_DT_PINNED_LCOE = 98.68644714355469


class TestModelIntegration:
    def test_default_path_bit_identical(self):
        """use_0d_model=False (default) must produce the pinned LCOE exactly."""
        m = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
        r = m.forward(net_electric_mw=500.0, availability=0.87, lifetime_yr=40.0)
        # Exact equality: same deterministic float path, no solver involved.
        # Pinned value is float32; rel=1e-12 is tighter than float32 epsilon,
        # so this is bit-identity in practice.
        assert r.costs.lcoe == pytest.approx(_MIRROR_DT_PINNED_LCOE, rel=1e-12)

    def test_0d_mirror_produces_finite_lcoe(self):
        """use_0d_model=True produces a finite LCOE for DT mirror."""
        m = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
        r = m.forward(
            net_electric_mw=500.0,
            availability=0.87,
            lifetime_yr=40.0,
            use_0d_model=True,
            # Override geometry to match a machine that can reach 500 MW net:
            # large central cell, adequate density, tight-enough n_e for p_fus.
            chamber_length=80.0,
            plasma_t=0.8,
            B=3.0,
            R_m=10.0,
            n_e=2.0e20,
            T_e=20.0,
            T_i=20.0,
            beta_max=0.5,
        )
        assert math.isfinite(r.costs.lcoe)
        assert r.costs.lcoe > 0.0
        assert isinstance(r.plasma_state, MirrorPlasmaState)

    def test_0d_mirror_plasma_state_attached(self):
        """ForwardResult.plasma_state is a MirrorPlasmaState when use_0d_model=True."""
        m = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
        r = m.forward(
            net_electric_mw=500.0,
            availability=0.87,
            lifetime_yr=40.0,
            use_0d_model=True,
            chamber_length=80.0,
            plasma_t=0.8,
            B=3.0,
            R_m=10.0,
            n_e=2.0e20,
            T_e=20.0,
            T_i=20.0,
            beta_max=0.5,
        )
        ps = r.plasma_state
        assert isinstance(ps, MirrorPlasmaState)
        assert math.isfinite(float(ps.T_i))
        assert float(ps.T_i) > 0.0

    def test_0d_mirror_no_disruption_penalty(self):
        """MirrorPlasmaState carries no disruption_rate field, and the model's
        disruption block is concept-gated, so a mirror 0D run completes without
        AttributeError and r.plasma_state has no disruption_rate attribute.
        """
        m = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
        r = m.forward(
            net_electric_mw=500.0,
            availability=0.87,
            lifetime_yr=40.0,
            use_0d_model=True,
            chamber_length=80.0,
            plasma_t=0.8,
            B=3.0,
            R_m=10.0,
            n_e=2.0e20,
            T_e=20.0,
            T_i=20.0,
            beta_max=0.5,
        )
        # Forward completed without AttributeError -> disruption gate is guarded.
        # Also: plasma_state carries no disruption_rate field.
        assert not hasattr(r.plasma_state, "disruption_rate")

    def test_0d_mirror_dhe3_uses_f_rad_fus_proxy(self):
        """D-He3 mirror 0D run uses the f_rad_fus proxy by default."""
        m = CostModel(ConfinementConcept.MIRROR, Fuel.DHE3)
        r = m.forward(
            net_electric_mw=100.0,
            availability=0.87,
            lifetime_yr=40.0,
            use_0d_model=True,
            chamber_length=80.0,
            plasma_t=0.8,
            B=5.0,
            R_m=30.0,
            n_e=5.0e20,
            T_e=50.0,
            T_i=50.0,
            beta_max=0.6,
            enforce_plasma_limits=False,
        )
        assert isinstance(r.plasma_state, MirrorPlasmaState)
        assert math.isfinite(r.costs.lcoe)

    def test_0d_mirror_dhe3_explicit_pin(self):
        """Explicit dhe3_dd_frac override pins the side-channel fraction."""
        m = CostModel(ConfinementConcept.MIRROR, Fuel.DHE3)
        r = m.forward(
            net_electric_mw=100.0,
            availability=0.87,
            lifetime_yr=40.0,
            use_0d_model=True,
            chamber_length=80.0,
            plasma_t=0.8,
            B=5.0,
            R_m=30.0,
            n_e=5.0e20,
            T_e=50.0,
            T_i=50.0,
            beta_max=0.6,
            enforce_plasma_limits=False,
            dhe3_dd_frac=0.25,
        )
        assert isinstance(r.plasma_state, MirrorPlasmaState)
        assert float(r.plasma_state.dhe3_dd_frac_eff) == pytest.approx(0.25, rel=1e-6)


# ---------------------------------------------------------------------------
# Mirror coil length-scaling tests (Task 5)
# ---------------------------------------------------------------------------


class TestMirrorCoilLengthScaling:
    """C220103 for the mirror must scale with central-cell length.

    Two-class structure:
      n_central = chamber_length / coil_spacing  (continuous aggregate)
      n_plug_coils  at throat field b_plug = R_m * B

    Markup recalibrated so that at the YAML defaults (L=20, coil_spacing=5,
    n_plug=4, R_m=10, B=3 -> b_plug=30 T) the cost reproduces the
    doc-validated 513.375 M$ exactly (calibration-neutrality invariant).
    """

    def _base(self, **overrides):
        """Mirror forward() at YAML defaults; override any subset of params."""
        m = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
        kw = dict(net_electric_mw=500.0, availability=0.87, lifetime_yr=40.0)
        kw.update(overrides)
        return m.forward(**kw)

    def test_calibration_neutrality_pin(self):
        """At YAML defaults (L=20, coil_spacing=5, n_plug=4) C220103 == 513.375.

        Calibration-neutrality pin: the two-class markup is solved so that the
        YAML default machine reproduces the doc-validated 513.375 M$ exactly.
        Any change to this number means the calibration algebra is wrong: STOP.
        """
        r = self._base()
        assert r.cas22_detail["C220103"] == pytest.approx(513.375, rel=1e-6)

    def test_lcoe_pin_unchanged(self):
        """The LCOE pinned value must be preserved exactly after the coil refactor.

        If this fails the calibration is wrong and nothing else can be trusted.
        """
        r = self._base()
        assert r.costs.lcoe == pytest.approx(_MIRROR_DT_PINNED_LCOE, rel=1e-12)

    def test_doubling_length_doubles_central_contribution(self):
        """Doubling chamber_length doubles the central-coil kA*m (and cost share).

        The plug contribution (fixed n_plug_coils, fixed b_plug) must be
        UNCHANGED.  We test the contributions, not just the total.
        """
        from costingfe.types import CoilMaterial

        MU0 = 4 * math.pi * 1e-7
        b_center = 12.0  # from steady_state_mirror.yaml
        # Central bore from radial build: vessel_or(3.20) + coil_standoff(0.10)
        r_bore_central = (
            1.5 + 0.10 + 0.05 + 0.80 + 0.20 + 0.20 + 0.15 + 0.10 + 0.10 + 0.10
        )
        # Plug bore from flux conservation: a/sqrt(R_m) + plug_standoff
        r_bore_plug = 1.5 / math.sqrt(10.0) + 0.30
        B = 3.0
        R_m = 10.0
        b_plug = R_m * B  # 30.0 T
        n_plug = 4
        coil_spacing = 5.0
        cost_per_kAm = CoilMaterial.REBCO_HTS.default_cost_per_kAm  # 50.0

        def _central_cost(L):
            n_c = L / coil_spacing
            G = n_c * 4 * math.pi
            kAm = G * b_center * r_bore_central**2 / (MU0 * 1000)
            return kAm * cost_per_kAm / 1e6

        def _plug_cost():
            G = n_plug * 4 * math.pi
            kAm = G * b_plug * r_bore_plug**2 / (MU0 * 1000)
            return kAm * cost_per_kAm / 1e6

        central_20 = _central_cost(20.0)
        central_40 = _central_cost(40.0)
        plug = _plug_cost()

        # Central contribution doubles exactly
        assert abs(central_40 / central_20 - 2.0) < 1e-10, (
            f"central contribution ratio={central_40 / central_20}, expected 2.0"
        )
        # Plug contribution is fixed (same n_plug, same b_plug)
        assert abs(plug) > 0  # sanity: plug cost is non-zero
        # Verify the contributions are additive in the full model
        r_base = self._base(chamber_length=20.0)
        r_double = self._base(chamber_length=40.0)
        # Delta from L=20 to L=40 equals one extra central_20 worth (times markup).
        # (C220103_L40 - C220103_L20) == central_20 * markup
        from costingfe.defaults import load_costing_constants

        cc = load_costing_constants()
        markup = cc.coil_markup["mirror"]

        delta_model = float(r_double.cas22_detail["C220103"]) - float(
            r_base.cas22_detail["C220103"]
        )
        delta_expected = central_20 * markup  # one extra n_central's worth
        assert delta_model == pytest.approx(delta_expected, rel=1e-6), (
            f"delta={delta_model:.4f} M$, expected {delta_expected:.4f} M$"
        )

    def test_jax_grad_chamber_length_finite_positive(self):
        """jax.grad of C220103 w.r.t. chamber_length must be finite and positive.

        Tests via model.sensitivity (reverse-mode AD through the full cost graph)
        AND via a finite-difference check to confirm direction.
        """
        m = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
        r = m.forward(net_electric_mw=500.0, availability=0.87, lifetime_yr=40.0)
        # JAX sensitivity: dLCOE/d(chamber_length) should be finite.
        # chamber_length enters C220103 linearly so the gradient is well-defined.
        sens = m.sensitivity(r.params)
        grad_L = float(sens["engineering"]["chamber_length"])
        assert math.isfinite(grad_L), f"dLCOE/d(chamber_length) is not finite: {grad_L}"
        assert grad_L > 0.0, f"dLCOE/d(chamber_length) must be positive, got {grad_L}"

        # Finite-difference check: longer machine -> higher coil cost -> higher LCOE
        r_lo = self._base(chamber_length=20.0)
        r_hi = self._base(chamber_length=21.0)
        c_lo = float(r_lo.cas22_detail["C220103"])
        c_hi = float(r_hi.cas22_detail["C220103"])
        dC_dL = c_hi - c_lo
        assert math.isfinite(dC_dL), "C220103 finite-diff gradient is not finite"
        assert dC_dL > 0, (
            f"C220103 must increase with chamber_length, got dC/dL={dC_dL}"
        )

    def test_central_bore_from_radial_build(self):
        """r_bore_central = vessel_or + coil_standoff, r_bore_plug from throat flux.

        At YAML defaults the build stacks to vessel_or = 3.20 m (1.5 plasma +
        0.10 vacuum + 0.05 FW + 0.80 blanket + 0.20 reflector + 0.20 HT shield +
        0.15 structure + 0.10 gap1 + 0.10 vessel) and coil_standoff = 0.10,
        so r_bore_central = 3.30 m. The plug bore comes from flux conservation:
        a_throat = plasma_t / sqrt(R_m) = 1.5/sqrt(10); r_bore_plug = a_throat + 0.30.
        """
        m = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
        r = m.forward(net_electric_mw=500.0, availability=0.87, lifetime_yr=40.0)
        assert r.cas22_detail["r_bore_central"] == pytest.approx(3.30, abs=1e-6)
        assert r.cas22_detail["r_bore_plug"] == pytest.approx(
            1.5 / math.sqrt(10.0) + 0.30, abs=1e-6
        )

    def test_coil_cost_responds_to_blanket_t(self):
        """Thicker blanket -> larger central bore -> costlier coils."""
        lo = self._base()
        hi = self._base(blanket_t=1.2)
        assert hi.cas22_detail["C220103"] > lo.cas22_detail["C220103"]

    def test_plug_central_split_pinned(self):
        """At YAML defaults the conductor split is pinned at the derived ratio.

        central kAm / plug kAm = (b_center * r_bore_central^2) /
                                  (b_plug * r_bore_plug^2) * n_central / n_plug
        Exact float64 ratio derived analytically at implementation time.
        """
        m = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
        r = m.forward(net_electric_mw=500.0, availability=0.87, lifetime_yr=40.0)
        split = r.cas22_detail["C220103_central"] / r.cas22_detail["C220103_plug"]
        assert split == pytest.approx(7.2647827768, rel=1e-6)


# ---------------------------------------------------------------------------
# Mirror length sizing tests (Task 6)
# ---------------------------------------------------------------------------

# Shared sizing params: WHAM/Realta-class mirror geometry inputs (fixed).
# The sizing solve only finds L; a, B, R_m stay as-is.
_SZ_A = 0.4  # plasma radius [m]
_SZ_B = 3.0  # midplane field [T]
_SZ_R_M = 10.0  # mirror ratio
_SZ_T_E = 20.0  # electron temperature [keV] (held fixed; GSS scans T_i)
_SZ_F_BETA = 0.85
_SZ_BETA_MAX = 0.5
_SZ_L_MIN = 1.0
_SZ_L_MAX = 200.0


def _sz_params(
    net_mw=200.0,
    f_beta=_SZ_F_BETA,
    fuel=Fuel.DT,
    f_rad_fus=None,
    q_wall_max=50.0,  # high default so existing tests are never wall-bound
    q_surface_max=50.0,  # high default so existing tests are never surface-bound
):
    """Build a minimal sizing params dict that mirror_size_from_power accepts."""
    return dict(
        net_electric_mw=net_mw,
        a=_SZ_A,
        B_min=_SZ_B,
        R_m=_SZ_R_M,
        T_e=_SZ_T_E,
        f_beta=f_beta,
        beta_max=_SZ_BETA_MAX,
        L_min=_SZ_L_MIN,
        L_max=_SZ_L_MAX,
        M_ion=2.5,
        Z_eff=1.2,
        R_w=0.4,
        p_input=40.0,
        mn=1.1,
        eta_th=0.40,
        eta_p=0.5,
        eta_pin=0.7,
        eta_de=0.85,
        f_sub=0.03,
        f_dec=0.3,
        p_coils=5.0,
        p_cool=20.0,
        p_pump=1.5,
        p_trit=10.0,
        p_house=4.0,
        p_cryo=1.0,
        p_aux_floor=2.0,
        dd_f_T=0.969,
        dd_f_He3=0.689,
        dhe3_dd_frac=0.131,
        dhe3_f_T=0.5,
        dhe3_f_He3=0.05,
        pb11_f_alpha_n=0.0,
        pb11_f_p_n=0.0,
        dhe3_fuel_ratio=1.0,
        pb11_fuel_ratio=0.15,
        dhe3_dd_frac_pin=None,
        wall_material=None,
        T_edge=0.2,
        tau_ratio=3.0,
        enforce_plasma_limits=True,
        vacuum_t=0.10,
        f_rad_fus=f_rad_fus,
        q_wall_max=q_wall_max,
        q_surface_max=q_surface_max,
    )


@pytest.mark.slow
class TestMirrorSizing:
    def test_sized_L_grows_with_net_power(self):
        """A higher net power target requires a longer chamber."""
        p_lo = dict(_sz_params(200.0), n_mod=1)
        p_hi = dict(_sz_params(600.0), n_mod=1)
        L_lo = mirror_size_from_power(p_lo, Fuel.DT)
        L_hi = mirror_size_from_power(p_hi, Fuel.DT)
        assert L_hi > L_lo, (
            f"Larger target should need longer L, got L_lo={L_lo:.1f} L_hi={L_hi:.1f}"
        )

    def test_sized_density_equals_f_beta_closed_form(self):
        """At the solved L, density matches the f_beta closed form at the GSS T_i."""
        params = dict(_sz_params(300.0), n_mod=1)
        L_solved = mirror_size_from_power(params, Fuel.DT)
        # net_electric_at_L returns p_net; to recover T_i call with return_state=True
        pn, T_i_star, n_e_star = net_electric_at_L(
            L_solved, params, Fuel.DT, return_state=True
        )
        # Verify density matches the closed form.
        # _density_from_f_beta returns [10^20 m^-3], converted at the
        # _net_at_L_T boundary; n_e_star is already in m^-3 (post-conversion).
        f_beta = params["f_beta"]
        beta_max = params["beta_max"]
        B = params["B_min"]
        T_e = params["T_e"]
        n_i_frac = n_i_over_n_e(
            Fuel.DT, params["dhe3_fuel_ratio"], params["pb11_fuel_ratio"]
        )
        T_sum = T_e + n_i_frac * T_i_star
        n20_expected = f_beta * beta_max * B**2 / (2 * _MU_0 * _KEV_TO_J * 1e20) / T_sum
        n_e_expected = n20_expected * 1e20  # convert n20 -> m^-3 for comparison
        assert n_e_star == pytest.approx(n_e_expected, rel=1e-6)

    def test_sized_lcoe_reflects_coil_account(self):
        """Larger sized machine has higher coil cost in its CAS22 account."""
        m_lo = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
        m_hi = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
        r_lo = m_lo.forward(
            net_electric_mw=200.0,
            availability=0.87,
            lifetime_yr=40.0,
            size_from_power=True,
        )
        r_hi = m_hi.forward(
            net_electric_mw=600.0,
            availability=0.87,
            lifetime_yr=40.0,
            size_from_power=True,
        )
        # Larger machine -> longer chamber -> higher C220103
        assert float(r_hi.cas22_detail["C220103"]) > float(r_lo.cas22_detail["C220103"])

    def test_pinning_chamber_length_in_sizing_mode_raises(self):
        """Pinning chamber_length in size_from_power mode must raise ValueError."""
        m = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
        with pytest.raises(ValueError, match=r"chamber_length"):
            m.forward(
                net_electric_mw=300.0,
                availability=0.87,
                lifetime_yr=40.0,
                size_from_power=True,
                chamber_length=50.0,  # pinned — not allowed
            )

    def test_unreachable_target_raises_sizing_infeasible(self):
        """A target beyond what L_max can deliver raises SizingInfeasible."""
        params = dict(_sz_params(net_mw=1e8), n_mod=1)  # absurdly large
        with pytest.raises(SizingInfeasible):
            mirror_size_from_power(params, Fuel.DT)

    def test_optimize_mode_returns_f_beta_in_bounds(self):
        """optimize_lcoe mode returns a result; stored f_beta is within bounds."""
        m = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
        r = m.forward(
            net_electric_mw=200.0,
            availability=0.87,
            lifetime_yr=40.0,
            size_from_power=True,
            optimize_lcoe=True,
        )
        assert math.isfinite(r.costs.lcoe)
        # _sizing_fbeta is set by the optimizer
        best = m._sizing_fbeta
        f_beta_min = m._eng_defaults["f_beta_min"]
        f_beta_max = m._eng_defaults["f_beta_max"]
        assert f_beta_min <= best <= f_beta_max

    def test_dhe3_sizes_longer_than_dt(self):
        """D-He3 mirror requires a longer chamber than DT at equal net power.

        D-He3 reactivity is lower than DT at any temperature, and dilution
        reduces n_i/n_e, so at equal f_beta density the machine needs more
        volume (larger L) to produce the same net electric power.

        Both fuels use the same geometry (B=5 T, a=0.6 m, R_m=20) to isolate
        the fuel dependence. DHE3 uses f_rad_fus=0.24 proxy and an aneutronic
        power balance (mn=1.02, f_dec=0.6, no tritium processing);
        L_max is extended to 500 m to accommodate the longer DHE3 machine.
        """
        common = dict(
            B_min=5.0,
            a=0.6,
            R_m=20.0,
            T_e=50.0,
            f_beta=0.85,
            beta_max=0.5,
            L_min=1.0,
            L_max=500.0,
            M_ion=2.5,
            Z_eff=1.2,
            R_w=0.4,
            p_input=40.0,
            eta_th=0.40,
            eta_p=0.5,
            eta_pin=0.7,
            eta_de=0.85,
            f_sub=0.03,
            p_coils=5.0,
            p_cool=20.0,
            p_pump=1.5,
            p_house=4.0,
            p_cryo=1.0,
            p_aux_floor=2.0,
            dd_f_T=0.969,
            dd_f_He3=0.689,
            dhe3_dd_frac=0.131,
            dhe3_f_T=0.5,
            dhe3_f_He3=0.05,
            pb11_f_alpha_n=0.0,
            pb11_f_p_n=0.0,
            dhe3_fuel_ratio=1.0,
            pb11_fuel_ratio=0.15,
            dhe3_dd_frac_pin=None,
            wall_material=None,
            T_edge=0.2,
            tau_ratio=3.0,
            vacuum_t=0.10,
            net_electric_mw=200.0,
            n_mod=1,
            q_wall_max=50.0,  # high cap so this test is never wall-bound
            q_surface_max=50.0,  # high cap so this test is never surface-bound
        )
        p_dt = dict(
            common,
            mn=1.1,
            f_dec=0.3,
            p_trit=10.0,
            f_rad_fus=None,
        )
        p_dhe3 = dict(
            common,
            mn=1.02,
            f_dec=0.6,
            p_trit=0.0,
            f_rad_fus=0.24,
        )
        L_dt = mirror_size_from_power(p_dt, Fuel.DT)
        L_dhe3 = mirror_size_from_power(p_dhe3, Fuel.DHE3)
        assert L_dhe3 > L_dt, (
            f"D-He3 should need longer chamber than DT, "
            f"got L_DT={L_dt:.1f} m, L_DHE3={L_dhe3:.1f} m"
        )

    def test_net_electric_at_L_return_full_state(self):
        """return_full surfaces the GSS-optimum forward (state, table).

        p_net and n_e must match the float-returning contract, and the
        returned state's beta must equal the closed-form beta at (T_star,
        n_e_star) -- i.e. it is the GSS operating point, not a re-solve.
        """
        params = dict(_sz_params(300.0), n_mod=1)
        L = mirror_size_from_power(params, Fuel.DT)
        pn_state, T_star, n_e_star = net_electric_at_L(
            L, params, Fuel.DT, return_state=True
        )
        pn_full, ps, pt = net_electric_at_L(L, params, Fuel.DT, return_full=True)
        assert pn_full == pytest.approx(pn_state, rel=1e-9)
        assert float(ps.n_e) == pytest.approx(n_e_star, rel=1e-9)
        assert float(ps.T_i) == pytest.approx(T_star, rel=1e-9)
        assert float(pt.p_net) == pytest.approx(pn_full, rel=1e-9)

    def test_high_f_beta_sizes_feasibly(self):
        """Regression: f_beta=0.95 sizes (no OperatingPointInfeasible).

        On the 400 MWe example machine the neutron wall cap (q_wall_max) binds
        at f_beta=0.95, so the operating density is wall-capped and beta lands
        just below 0.95 * beta_max but well under beta_max. The pre-fix path
        re-solved T_i through the inverse at the fixed wall-capped n_e_star and
        reported beta = 0.5315 > beta_max = 0.5, tripping
        OperatingPointInfeasible. Building the state directly from the GSS
        optimum keeps beta bounded by construction.
        """
        m = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
        r = m.forward(
            net_electric_mw=400.0,
            availability=0.85,
            lifetime_yr=30.0,
            size_from_power=True,
            f_beta=0.95,
            L_min=1.0,
            L_max=400.0,
        )
        assert math.isfinite(r.costs.lcoe)
        ps = m._plasma_state
        beta_max = 0.5  # YAML default for the mirror machine
        # Wall-bound here: beta is below the f_beta ceiling and below beta_max.
        assert float(ps.beta) <= 0.95 * beta_max * (1 + 1e-6)
        assert float(ps.beta) < beta_max
        # Confirm the wall cap is the binding constraint (q_wall at its ceiling).
        assert float(ps.wall_loading) == pytest.approx(5.0, rel=1e-3)

    def test_sized_state_beta_within_f_beta_cap(self):
        """Invariant the bug violated: sized-state beta <= f_beta * beta_max.

        For a beta-bound sizing case the operating beta must never exceed the
        f_beta-scaled ceiling (with float tolerance). This locks the
        correctness-by-construction property of the GSS-optimum handoff.
        """
        m = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
        f_beta = 0.85
        beta_max = 0.5
        m.forward(
            net_electric_mw=400.0,
            availability=0.85,
            lifetime_yr=30.0,
            size_from_power=True,
            f_beta=f_beta,
            L_min=1.0,
            L_max=400.0,
        )
        ps = m._plasma_state
        assert float(ps.beta) <= f_beta * beta_max * (1 + 1e-6)

    @pytest.mark.parametrize(
        "fuel,T_i,T_e",
        [
            (Fuel.DT, 20.0, 20.0),
            (Fuel.DD, 30.0, 30.0),
            (Fuel.DHE3, 70.0, 70.0),
            (Fuel.PB11, 200.0, 200.0),
        ],
    )
    def test_density_from_f_beta_jit_matches_eager_all_fuels(self, fuel, T_i, T_e):
        """_density_from_f_beta: jit matches eager (rel 1e-4) and gradient is finite.

        _density_from_f_beta returns [10^20 m^-3],
        converted at the _net_at_L_T boundary.
        """
        f_beta = 0.85
        beta_max = 0.5
        B_min = 3.0
        dhe3_fuel_ratio = 1.0
        pb11_fuel_ratio = 0.15

        def kernel(T_i_arr):
            # returns [10^20 m^-3], converted at the _net_at_L_T boundary
            return _density_from_f_beta(
                T_i_arr,
                T_e,
                f_beta,
                beta_max,
                B_min,
                fuel,
                dhe3_fuel_ratio,
                pb11_fuel_ratio,
            )

        T_i_arr = jnp.array(T_i)
        eager_val = float(kernel(T_i_arr))
        jit_val = float(jax.jit(kernel)(T_i_arr))
        assert jnp.isfinite(jit_val)
        assert jit_val == pytest.approx(eager_val, rel=1e-4)

        grad_val = float(jax.grad(kernel)(T_i_arr))
        assert jnp.isfinite(grad_val)


# ---------------------------------------------------------------------------
# Wall-load constraint tests (Task 3)
# ---------------------------------------------------------------------------

# Shared sizing params for wall-constraint tests: a=1.5, B=3.0 machine,
# which IS wall-bound at q_wall_max=5.0 (n_wall < n_beta at f_beta=0.85).
# (The standard _sz_params machine a=0.4, B=3.0 is NOT wall-bound; the 1.5 m
# radius machine reproduces the YAML-defaults geometry for this test class.)
_SZ_A_WALL = 1.5  # plasma radius [m] — YAML defaults machine
_SZ_B_WALL = 3.0  # midplane field [T]
_SIZING_PARAMS = dict(
    net_electric_mw=200.0,
    a=_SZ_A_WALL,
    B_min=_SZ_B_WALL,
    q_wall_max=50.0,  # high default; cap non-binding unless a test overrides it
    q_surface_max=50.0,  # high default; cap non-binding unless a test overrides it
    R_m=10.0,
    T_e=20.0,
    f_beta=0.85,
    beta_max=0.5,
    L_min=1.0,
    L_max=200.0,
    M_ion=2.5,
    Z_eff=1.2,
    R_w=0.4,
    p_input=40.0,
    mn=1.1,
    eta_th=0.40,
    eta_p=0.5,
    eta_pin=0.7,
    eta_de=0.85,
    f_sub=0.03,
    f_dec=0.3,
    p_coils=5.0,
    p_cool=20.0,
    p_pump=1.5,
    p_trit=10.0,
    p_house=4.0,
    p_cryo=1.0,
    p_aux_floor=2.0,
    dd_f_T=0.969,
    dd_f_He3=0.689,
    dhe3_dd_frac=0.131,
    dhe3_f_T=0.5,
    dhe3_f_He3=0.05,
    pb11_f_alpha_n=0.0,
    pb11_f_p_n=0.0,
    dhe3_fuel_ratio=1.0,
    pb11_fuel_ratio=0.15,
    dhe3_dd_frac_pin=None,
    wall_material=None,
    T_edge=0.2,
    tau_ratio=3.0,
    enforce_plasma_limits=True,
    vacuum_t=0.10,
    f_rad_fus=None,
    n_mod=1,
)

# Per-fuel mix kwargs dict for _density_from_wall_cap parametrized test.
# Mirrors the kwargs pattern fusion_power and ash_neutron_split expect.
_MIX_KWARGS_FOR = {
    fuel: dict(
        dd_f_T=0.969,
        dd_f_He3=0.689,
        dhe3_dd_frac_pin=None,
        dhe3_f_T=0.5,
        dhe3_f_He3=0.05,
        pb11_f_alpha_n=0.0,
        pb11_f_p_n=0.0,
        dhe3_fuel_ratio=1.0,
        pb11_fuel_ratio=0.15,
    )
    for fuel in [Fuel.DT, Fuel.DD, Fuel.DHE3, Fuel.PB11]
}


def _size(params, fuel=Fuel.DT):
    """Size a mirror and return (L_solved, p_net, MirrorPlasmaState) at the solution."""
    L = mirror_size_from_power(params, fuel)
    pn, T_star, n_e_star = net_electric_at_L(L, params, fuel, return_state=True)
    # Evaluate the state at the solved operating point.
    # n_e_star is in m^-3 (converted at the _net_at_L_T boundary).
    ps = mirror_0d_forward(
        L=L,
        a=params["a"],
        B_min=params["B_min"],
        R_m=params["R_m"],
        T_i=T_star,
        T_e=params["T_e"],
        n_e=float(n_e_star),
        p_input=params["p_input"],
        fuel=fuel,
        M_ion=params.get("M_ion", 2.5),
        Z_eff=params.get("Z_eff", 1.2),
        R_w=params.get("R_w", 0.4),
        dd_f_T=params["dd_f_T"],
        dd_f_He3=params["dd_f_He3"],
        dhe3_dd_frac_pin=params.get("dhe3_dd_frac_pin"),
        dhe3_f_T=params["dhe3_f_T"],
        dhe3_f_He3=params["dhe3_f_He3"],
        pb11_f_alpha_n=params["pb11_f_alpha_n"],
        pb11_f_p_n=params["pb11_f_p_n"],
        dhe3_fuel_ratio=params["dhe3_fuel_ratio"],
        pb11_fuel_ratio=params["pb11_fuel_ratio"],
        vacuum_t=params["vacuum_t"],
        f_rad_fus=params.get("f_rad_fus"),
    )
    return L, pn, ps


@pytest.mark.slow
class TestWallConstraint:
    def test_sizing_respects_q_wall_max(self):
        # Default machine (a=1.5, B=3.0) is wall-bound at q_wall_max=5.0:
        # n_wall(T) < n_beta(T) across the GSS T_i bracket at these params.
        params = dict(_SIZING_PARAMS, q_wall_max=5.0)
        L, pnet, state = _size(params)
        assert float(state.wall_loading) <= 5.0 * 1.001
        # wall cap (5.0) is the active constraint:
        assert float(state.wall_loading) >= 4.5
        # energy-balance closed: sustainment charged from confinement -> the GSS
        # optimum shifts to the hot end of the ignited plateau (T_i ~ 60 keV),
        # where the f_beta cap co-binds with the wall cap, so beta sits AT the
        # f_beta * beta_max ceiling rather than below it. The wall cap remains
        # binding (wall_loading = 5.0); both caps are active at this T.
        assert float(state.beta) <= 0.85 * 0.5 * (1 + 1e-3)

    def test_loose_cap_recovers_beta_bound_solution(self):
        # q_wall_max=50 must reproduce the (beta-bound) solution at the closed
        # energy balance.
        # energy-balance closed: sustainment charged from confinement (was the
        # legacy beta-bound L = 4.9773251338 m at tip 5530ec8; the closure
        # changed the recirculating power, hence the L that meets the target).
        params = dict(_SIZING_PARAMS, q_wall_max=50.0)
        L, _, _ = _size(params)
        assert L == pytest.approx(4.509277938, rel=1e-4)

    def test_infeasible_under_cap_raises_naming_cap(self):
        with pytest.raises(SizingInfeasible, match=r"q_wall_max"):
            _size(dict(_SIZING_PARAMS, q_wall_max=0.5, net_electric_mw=600.0))

    def test_optimizer_survives_tight_cap_returns_finite_lcoe(self):
        """optimize_lcoe completes when the wall cap causes SizingInfeasible
        at some f_beta values in the GSS sweep.

        q_wall_max=5.0 with the a=1.5 m machine is wall-bound: some f_beta
        values in [f_beta_min, f_beta_max] trigger SizingInfeasible inside
        mirror_size_from_power.  The _lcoe_at_fb handler must catch both
        OperatingPointInfeasible and SizingInfeasible and return the 1e8
        sentinel so the optimizer can find the feasible region and return a
        finite LCOE.
        """
        m = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
        r = m.forward(
            net_electric_mw=200.0,
            availability=0.87,
            lifetime_yr=40.0,
            size_from_power=True,
            optimize_lcoe=True,
            q_wall_max=5.0,  # tight cap; some f_beta values are SizingInfeasible
        )
        assert math.isfinite(r.costs.lcoe)
        assert r.costs.lcoe < 1e7  # not the sentinel; a real LCOE

    @pytest.mark.parametrize("fuel", [Fuel.DT, Fuel.DD, Fuel.DHE3, Fuel.PB11])
    def test_n_wall_jit_matches_eager_all_fuels(self, fuel):
        # Same harness as test_density_from_f_beta_jit_matches_eager_all_fuels:
        # copy its per-fuel mix kwargs verbatim.
        mix = _MIX_KWARGS_FOR[fuel]

        def f(T):
            return _density_from_wall_cap(T, 20.0, 5.0, 1.5, 0.10, fuel, mix)

        eager = float(f(20.0))
        jitted = float(jax.jit(f)(20.0))
        assert jnp.isfinite(jitted)
        assert jitted == pytest.approx(eager, rel=1e-4)
        g = float(jax.grad(f)(20.0))
        assert jnp.isfinite(g)


# ---------------------------------------------------------------------------
# Surface heat-flux constraint tests (Task 4)
# ---------------------------------------------------------------------------

# p-B11 sizing params: aneutronic, f_rad_fus=0.83 (83% of fusion power radiated).
# The surface cap (q_surface_max=1.0) must bind before the neutron and beta caps.
# Machine: high-field compact (B=12T, a=1.0m) where at T_i~290 keV the
# beta-limited n20=4.44 exceeds the surface-limited n20=3.41, so the surface
# cap is the binding density constraint. At q_surface_max=1.0 and L~27 m
# this machine produces ~50 MWe (verified at implementation).
_PB11_SIZING_PARAMS = dict(
    net_electric_mw=50.0,  # achievable at B=12T, a=1.0 m with surface cap
    a=1.0,  # compact radius; high B pushes surface cap above beta cap
    B_min=12.0,  # high field so beta-limited density exceeds surface-limited density
    q_wall_max=50.0,  # high cap; neutron cap does not bind for aneutronic p-B11
    q_surface_max=1.0,  # surface cap binding for p-B11 (83% radiation fraction)
    R_m=10.0,
    T_e=150.0,  # hot electron temperature for p-B11 hot-ion regime
    f_beta=0.85,
    beta_max=0.5,
    L_min=1.0,
    L_max=200.0,
    M_ion=2.5,
    Z_eff=1.2,
    R_w=0.4,
    p_input=40.0,
    mn=1.02,  # near-aneutronic: no blanket multiplier
    eta_th=0.40,
    eta_p=0.5,
    eta_pin=0.7,
    eta_de=0.85,
    f_sub=0.03,
    f_dec=0.6,  # high DEC fraction: charged-particle dominant
    p_coils=5.0,
    p_cool=5.0,
    p_pump=1.5,
    p_trit=0.0,  # no tritium processing for p-B11
    p_house=4.0,
    p_cryo=1.0,
    p_aux_floor=2.0,
    dd_f_T=0.969,
    dd_f_He3=0.689,
    dhe3_dd_frac=0.131,
    dhe3_f_T=0.5,
    dhe3_f_He3=0.05,
    pb11_f_alpha_n=0.0,
    pb11_f_p_n=0.0,
    dhe3_fuel_ratio=1.0,
    pb11_fuel_ratio=0.15,
    dhe3_dd_frac_pin=None,
    wall_material=None,
    T_edge=0.2,
    tau_ratio=3.0,
    enforce_plasma_limits=True,
    vacuum_t=0.10,
    f_rad_fus=0.83,  # p-B11: 83% of fusion power radiated as photons
    n_mod=1,
)


@pytest.mark.slow
class TestSurfaceConstraint:
    def test_pb11_sizing_is_surface_bound(self):
        """p-B11 sized with q_surface_max=1.0: surface cap must bind.

        p-B11 radiates 83% of fusion power; the surface cap binds before
        both the beta boundary and the neutron cap (which barely applies
        for an aneutronic fuel).
        """
        L, pnet, state = _size(_PB11_SIZING_PARAMS, fuel=Fuel.PB11)
        # Surface cap must be at or below the threshold (with 0.1% tolerance).
        assert float(state.q_surface) <= 1.0 * 1.001
        # Neutron cap must not be the binding constraint (p-B11 is aneutronic).
        assert float(state.wall_loading) < 5.0  # neutron cap slack
        # Beta cap must not be the binding constraint either.
        assert float(state.beta) < 0.85 * 0.5  # below f_beta * beta_max

    def test_dt_audit_mode_warns_above_q_surface_max(self):
        """Forcing a small q_surface_max triggers a UserWarning in audit mode.

        The reference inverse machine at n_e=1e20, T_i~22 keV produces
        q_surface ~ 0.008 MW/m^2. Setting q_surface_max=0.001 (below the
        achieved value) triggers the 'surface' warning.
        """
        with pytest.warns(UserWarning, match=r"surface"):
            _inverse(p_net_target=20.0, q_surface_max=0.001)

    def test_surface_cap_infeasible_raises_naming_q_surface_max(self):
        """Infeasibility under the surface cap names q_surface_max in the message."""
        # Force a ridiculously tight surface cap so the machine cannot reach 50 MWe.
        params = dict(_PB11_SIZING_PARAMS, q_surface_max=1e-6)
        with pytest.raises(SizingInfeasible, match=r"q_surface_max"):
            _size(params, fuel=Fuel.PB11)

    def test_loose_surface_cap_recovers_wall_bound_or_beta_bound(self):
        """With q_surface_max=50, the p-B11 solution is NOT surface-bound.

        Raising the cap to 50 MW/m^2 allows the solver to run free; the
        active constraint should be either the beta boundary or the neutron
        wall cap, not the surface cap.
        """
        params = dict(_PB11_SIZING_PARAMS, q_surface_max=50.0, q_wall_max=50.0)
        L, pnet, state = _size(params, fuel=Fuel.PB11)
        # With loose caps the surface heat flux is above 1 MW/m^2 (otherwise
        # the tight-cap test would be trivially passing).
        assert float(state.q_surface) > 1.0

    def test_density_from_surface_cap_returns_finite(self):
        """_density_from_surface_cap returns a finite, positive n20 for each fuel.

        Uses a tightly capped q_surface_max=0.01 MW/m^2 at the _SIZING_PARAMS
        machine; the bisection must converge to a finite density for all fuels.
        """
        mix = _MIX_KWARGS_FOR[Fuel.DT]
        fwd_kwargs = dict(
            B_min=_SZ_B_WALL,
            R_m=10.0,
            M_ion=2.5,
            Z_eff=1.2,
            R_w=0.4,
            f_rad_fus=None,  # DT: full radiation model
        )
        n_surf_20 = _density_from_surface_cap(
            20.0,  # T_i
            20.0,  # T_e
            0.01,  # q_surface_max [MW/m^2]: tight cap
            _SZ_A_WALL,
            0.10,  # vacuum_t
            Fuel.DT,
            mix,
            fwd_kwargs,
        )
        assert n_surf_20 > 0.0
        assert n_surf_20 < 1e3  # not the sentinel (cap is binding at 0.01 MW/m^2)


class TestRegimeBridge:
    """Collisionality-gated bridge between gas-dynamic and Pastukhov confinement.

    See docs/account_justification/mirror_confinement_regimes.md (Rognlien and
    Cutler 1980): gas-dynamic governs when collisional, Pastukhov when
    collisionless, transition at collisionality = 1/R_m.
    """

    def test_collisionless_uses_pastukhov_branch(self):
        # Deeply collisionless (low n, high T): tau_axial must approach the
        # Pastukhov time, NOT the much shorter gas-dynamic time. This is the
        # bug guard: the old harmonic sum gave ~tau_GD here.
        n, T, A, R_m, L = 1.0e20, 60.0, 2.5, 10.0, 20.0
        tii = float(compute_tau_ii(n, T, A))
        phi = float(compute_ambipolar_potential(T, A))
        tau_p = float(compute_tau_pastukhov(tii, R_m, phi, T))
        tau_gd = float(compute_tau_gas_dynamic(R_m, L, T, A))
        # sanity: this point is collisionless (tau_gd << tau_p)
        assert tau_gd < 0.01 * tau_p
        tau_axial = float(compute_tau_axial(tii, R_m, L, T, A, phi, n))
        # bridge must pick (near) Pastukhov, within a small factor, not tau_GD
        assert tau_axial > 0.5 * tau_p

    def test_collisional_uses_gas_dynamic_branch(self):
        # High n, low T: collisional, gas-dynamic governs (tau_axial ~ tau_GD).
        n, T, A, R_m, L = 5.0e20, 1.0, 2.5, 30.0, 7.0
        tii = float(compute_tau_ii(n, T, A))
        phi = float(compute_ambipolar_potential(T, A))
        tau_gd = float(compute_tau_gas_dynamic(R_m, L, T, A))
        tau_axial = float(compute_tau_axial(tii, R_m, L, T, A, phi, n))
        assert tau_axial == pytest.approx(tau_gd, rel=0.5)

    @pytest.mark.parametrize("fuel", [Fuel.DT, Fuel.DD, Fuel.DHE3, Fuel.PB11])
    def test_bridge_jit_matches_eager_and_differentiable(self, fuel):
        n, A, R_m, L = 1.0e20, 2.5, 10.0, 20.0

        def f(T):
            tii = compute_tau_ii(n, T, A)
            phi = compute_ambipolar_potential(T, A)
            return compute_tau_axial(tii, R_m, L, T, A, phi, n)

        eager = float(f(30.0))
        jitted = float(jax.jit(f)(30.0))
        assert jnp.isfinite(jitted)
        assert jitted == pytest.approx(eager, rel=1e-4)
        g = float(jax.grad(f)(30.0))
        assert jnp.isfinite(g)

    def test_grad_finite_at_extreme_collisionless_point(self):
        # Far past any physical operating point (n -> ~1e12, T -> 2000 keV) the
        # logistic argument saturates beyond the float32 exp() overflow band, so
        # the unclamped gate yields inf*0 = NaN in reverse mode. The clamp keeps
        # the gradient finite. Golden-section sizing and the jax.grad sensitivity
        # vector both traverse this function, so this tail must stay finite.
        # Verified: without the [-30, 30] clamp this grad is NaN here.
        n, A, R_m, L = 1.0e12, 2.5, 10.0, 20.0

        def f(T):
            tii = compute_tau_ii(n, T, A)
            phi = compute_ambipolar_potential(T, A)
            return compute_tau_axial(tii, R_m, L, T, A, phi, n)

        g = float(jax.grad(f)(2000.0))
        assert jnp.isfinite(g)


class TestEnergyBalanceClosure:
    """Sizing-mode steady-state energy balance: sustainment is charged from the
    confinement-derived auxiliary power, paralleling the tokamak closure."""

    def test_mirror_aux_heating_closes_balance(self):
        # P_aux = max(floor, P_end + P_radial + P_rad - P_alpha) from the state.
        ps = _forward(T_i=15.0)
        p_aux = float(mirror_aux_heating(ps, p_aux_floor=2.0))
        expected = max(
            2.0,
            float(ps.p_end) + float(ps.p_radial) + float(ps.p_rad) - float(ps.p_alpha),
        )
        assert p_aux == pytest.approx(expected, rel=1e-6)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Identity p_transport == P_end + P_radial is a sub-ignition property. "
            "Post-Task-1 the D-T sizing optimum is ignited (aux floors), so "
            "p_transport inflates toward p_ash and the identity does not hold. "
            "See mirror_confinement_regimes.md (ignited-plateau finding). Flips "
            "when a stability bound (Task 4) moves the optimum sub-ignition."
        ),
    )
    def test_p_transport_identity_in_sizing(self):
        # When sizing feeds p_input=P_aux, the shared balance's p_transport
        # equals P_end + P_radial to a small tolerance. p_transport is not a
        # PowerTable field; reconstruct it from its definition in
        # mfe_forward_power_balance: p_transport = p_ash + p_input_eff - p_rad.
        m = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
        r = m.forward(
            net_electric_mw=400.0,
            availability=0.87,
            lifetime_yr=40.0,
            size_from_power=True,
            f_beta=0.85,
        )
        ps = r.plasma_state
        pt = r.power_table
        p_transport = float(pt.p_ash) + float(pt.p_input) - float(pt.p_rad)
        p_transport_expected = float(ps.p_end) + float(ps.p_radial)
        assert p_transport == pytest.approx(p_transport_expected, rel=0.05)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Energy-balance closure alone does NOT pull the D-T optimum below "
            "60 keV: post-Task-1 the mirror is ignited and the neutron wall-load "
            "cap pins fusion power independent of T, so the net-electric "
            "objective is flat across the hot band and aux floors throughout. "
            "The lever to a realistic 10 keV is the conditional Task 4 stability "
            "bound. See mirror_confinement_regimes.md (ignited-plateau finding)."
        ),
    )
    def test_sized_dt_optimum_is_realistic_temperature(self):
        # THE headline regression: with the balance closed and tau fixed, the
        # D-T sizing optimum lands in a realistic band, not ~60 keV.
        m = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
        r = m.forward(
            net_electric_mw=400.0,
            availability=0.87,
            lifetime_yr=40.0,
            size_from_power=True,
            f_beta=0.85,
        )
        assert 8.0 <= float(r.plasma_state.T_i) <= 25.0

    def test_tau_E_physical_p_end_below_p_fus(self):
        m = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
        r = m.forward(
            net_electric_mw=400.0,
            availability=0.87,
            lifetime_yr=40.0,
            size_from_power=True,
            f_beta=0.85,
        )
        assert float(r.plasma_state.p_end) < float(r.plasma_state.p_fus)

    def test_audit_mode_reports_sustainment_ratio(self):
        # Forward/audit mode (mirror_0d_inverse) keeps the stated p_input and
        # reports the sustainment-consistency diagnostic: stated p_input divided
        # by the confinement-required auxiliary power.
        ps, _pt = _inverse(p_input=50.0)
        aux = float(mirror_aux_heating(ps, p_aux_floor=2.0))
        assert float(ps.sustainment_ratio) == pytest.approx(50.0 / aux, rel=1e-4)


class TestAnchors:
    """Pin the confinement kernels to published GDT and WHAM literature.

    Provenance and the full derivation of every number below live in
    docs/account_justification/mirror_confinement.md. Each anchor carries its
    primary-source citation inline. The 2x tolerance is documented there: the
    model is a single thermal Maxwellian, while the published formulas are
    built on beam-driven distributions, so a factor-of-2 band is the
    appropriate validation gate (these tests pin the model to the literature,
    not to itself).
    """

    # Deuterium mass number for both machines.
    _A_D = 2.0

    def test_gdt_gas_dynamic_anchor(self):
        """GDT warm-plasma gas-dynamic time vs Endrizzi 2023 eq. 3.5.

        GDT machine (Bagryansky et al. 2015, PRL 114, 205001): central cell
        7 m mirror-to-mirror, R_m = 35, deuterium. Warm/target plasma in the
        standard config: T_e = 0.25 keV (Bagryansky 2015), gas-dynamic regime.

        Published value: Endrizzi et al. 2023 (J. Plasma Phys. 89, 975890501)
        eq. 3.5, tau_GDT = 5.2 * R_m * L_p * T_e^-0.5 us with L_p the plasma
        half-length (3.5 m). At R_m=35, L_p=3.5, T_e=0.25: 1.27 ms.

        The model kernel uses full length L and ion thermal speed v_thi, so it
        is intrinsically sqrt(2) larger than eq. 3.5 (see the doc); ratio 1.25.
        See docs/account_justification/mirror_confinement.md.
        """
        gdt_R_m = 35.0  # Bagryansky et al. 2015
        gdt_L = 7.0  # m, central cell mirror-to-mirror, Bagryansky et al. 2015
        gdt_T_warm = 0.25  # keV, standard-config warm plasma, Bagryansky 2015

        tau_model = float(
            compute_tau_gas_dynamic(R_m=gdt_R_m, L=gdt_L, T_i=gdt_T_warm, A=self._A_D)
        )

        # Published gas-dynamic time, Endrizzi 2023 eq. 3.5 [s].
        # 5.2 * R_m * L_p * T_e^-0.5 microseconds, L_p = half of 7 m = 3.5 m.
        gdt_L_p = 3.5
        tau_published = 5.2e-6 * gdt_R_m * gdt_L_p * gdt_T_warm**-0.5

        ratio = tau_model / tau_published
        assert 0.5 < ratio < 2.0, (
            f"GDT gas-dynamic anchor: model {tau_model * 1e3:.2f} ms vs "
            f"published {tau_published * 1e3:.2f} ms, ratio {ratio:.2f} "
            "outside 2x; see docs/account_justification/mirror_confinement.md"
        )

    def test_wham_pastukhov_anchor(self):
        """WHAM Pastukhov/CM confinement vs Endrizzi 2023 eq. 3.4.

        WHAM design (Endrizzi et al. 2023, J. Plasma Phys. 89, 975890501):
        17 T HTS mirrors, 0.86 T midplane -> vacuum R_m = 19.8; 25 keV NBI;
        beta=0.2 equilibrium at n = 0.3e20 m^-3 and 10 keV mean ion energy at
        the midplane; predicted T_e >= 1 keV.

        Published value: eq. 3.4 (= Forest BEAM 2024 eq. 1.1), classical-mirror
        scaling n_20 * tau_p = 250 * E_b,100keV^1.5 * log10(R_m) ms. At
        n_20=0.3, E_b=25 keV, R_m=19.8: 135 ms.

        Model uses a single thermal Maxwellian Pastukhov kernel at the midplane
        thermal point (T_i=10 keV, T_e=1 keV); ratio 135/88 = 1.53, within 2x.
        See docs/account_justification/mirror_confinement.md.
        """
        wham_R_m = 17.0 / 0.86  # vacuum mirror ratio, Endrizzi 2023 sec. 2.1
        wham_n = 3.0e19  # m^-3, beta=0.2 equilibrium density, Endrizzi 2023
        wham_T_i = 10.0  # keV, midplane mean ion energy, Endrizzi 2023
        wham_T_e = 1.0  # keV, predicted electron temperature, Endrizzi 2023

        tau_ii = compute_tau_ii(wham_n, wham_T_i, self._A_D)
        phi = compute_ambipolar_potential(wham_T_e, self._A_D)
        tau_model = float(compute_tau_pastukhov(tau_ii, wham_R_m, phi, wham_T_i))

        # Published Pastukhov/CM time, Endrizzi 2023 eq. 3.4 [s].
        # n_20 * tau_p = 250 * E_b,100keV^1.5 * log10(R_m) ms.
        E_b_100 = 25.0 / 100.0  # 25 keV NBI normalized to 100 keV
        n_20 = wham_n / 1e20
        tau_published = (250.0 * E_b_100**1.5 * math.log10(wham_R_m) / n_20) * 1e-3

        ratio = tau_published / tau_model
        assert 0.5 < ratio < 2.0, (
            f"WHAM Pastukhov anchor: model {tau_model * 1e3:.1f} ms vs "
            f"published {tau_published * 1e3:.1f} ms, ratio {ratio:.2f} "
            "outside 2x; see docs/account_justification/mirror_confinement.md"
        )


# ---------------------------------------------------------------------------
# Fluence-based CAS72 core lifetime (Task 6, decision d1)
# ---------------------------------------------------------------------------


class TestFluenceLifetime:
    """Steady-state MFE core lifetime is Phi_max / q_n, clamped to plant life.

    See docs/account_justification/wall_limits_and_fluence.md.
    """

    def test_fluence_lifetime_continuity_at_reference(self):
        # DT at exactly q_n = Phi_max / 5 FPY reproduces the legacy 5 FPY.
        cc = load_costing_constants()
        q_ref = cc.fluence_limit(Fuel.DT) / 5.0
        lifetime = float(
            _core_lifetime_fpy(
                cc, Fuel.DT, q_n=q_ref, lifetime_yr=40.0, availability=0.87
            )
        )
        assert lifetime == pytest.approx(5.0, rel=1e-9)

    def test_cas72_grows_with_wall_loading(self):
        m = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
        lo = m.forward(net_electric_mw=300.0, availability=0.87, lifetime_yr=40.0)
        hi = m.forward(net_electric_mw=700.0, availability=0.87, lifetime_yr=40.0)
        # Higher power at fixed geometry -> higher q_n -> shorter life -> larger CAS72
        assert hi.costs.cas72 > lo.costs.cas72

    def test_lifetime_clamped(self):
        cc = load_costing_constants()
        # Floor: extreme wall loading is clamped at 0.5 FPY.
        floor = float(
            _core_lifetime_fpy(cc, Fuel.DT, q_n=1e6, lifetime_yr=40.0, availability=1.0)
        )
        assert floor >= 0.5
        # Plant-life cap: aneutronic fuel at vanishing q_n clamps at
        # lifetime_yr * availability = 40.0 (availability=1.0).
        cap = float(
            _core_lifetime_fpy(
                cc, Fuel.PB11, q_n=1e-9, lifetime_yr=40.0, availability=1.0
            )
        )
        assert cap <= 40.0
