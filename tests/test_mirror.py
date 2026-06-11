"""Tests for the 0D mirror physics model."""

import math

import jax
import jax.numpy as jnp
import pytest

from costingfe.layers.mirror import (
    _RHO_I_PREFACTOR,
    _V_THI_PREFACTOR,
    MirrorPlasmaState,
    compute_ambipolar_potential,
    compute_tau_classical,
    compute_tau_gas_dynamic,
    compute_tau_ii,
    compute_tau_pastukhov,
    compute_tau_radial,
    mirror_0d_forward,
)
from costingfe.types import Fuel

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
        fw_ref = 2.0 * math.pi * _A * _L
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
