"""Tests for the 0D mirror physics model."""

import math

import jax
import jax.numpy as jnp
import pytest

from costingfe.layers.mirror import (
    _RHO_I_PREFACTOR,
    _V_THI_PREFACTOR,
    compute_ambipolar_potential,
    compute_tau_classical,
    compute_tau_gas_dynamic,
    compute_tau_ii,
    compute_tau_pastukhov,
    compute_tau_radial,
)

_N = 1.0e20  # m^-3
_TI = 20.0  # keV
_TE = 20.0


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
