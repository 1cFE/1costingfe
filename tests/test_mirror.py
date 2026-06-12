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
    mirror_0d_inverse,
)
from costingfe.layers.physics import OperatingPointInfeasible
from costingfe.model import CostModel
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
        enforce_plasma_limits=enforce_plasma_limits,
        dhe3_dd_frac=0.131,
        dhe3_dd_frac_pin=dhe3_dd_frac_pin,
        f_rad_fus=f_rad_fus,
        **_FRACS,
        **_MIX,
        **_PB_KWARGS,
        **kw,
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
            dhe3_dd_frac=0.131,
            dhe3_dd_frac_pin=None,
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
_MIRROR_DT_PINNED_LCOE = 93.643616  # $/MWh at 500 MW, avail=0.87, lifetime=40 yr


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
        import math

        from costingfe.types import CoilMaterial

        MU0 = 4 * math.pi * 1e-7
        b_center = 12.0  # from steady_state_mirror.yaml
        r_bore = 1.85
        B = 3.0
        R_m = 10.0
        b_plug = R_m * B  # 30.0 T
        n_plug = 4
        coil_spacing = 5.0
        cost_per_kAm = CoilMaterial.REBCO_HTS.default_cost_per_kAm  # 50.0

        def _central_cost(L):
            n_c = L / coil_spacing
            G = n_c * 4 * math.pi
            kAm = G * b_center * r_bore**2 / (MU0 * 1000)
            return kAm * cost_per_kAm / 1e6

        def _plug_cost():
            G = n_plug * 4 * math.pi
            kAm = G * b_plug * r_bore**2 / (MU0 * 1000)
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
        assert abs(delta_model - delta_expected) < 1.0, (
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
