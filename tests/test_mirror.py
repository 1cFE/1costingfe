"""Tests for the 0D mirror physics model."""

import math

import pytest

from costingfe._backend import HAS_JAX

if HAS_JAX:
    import jax
    import jax.numpy as jnp
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

# Calibrated tandem plug-to-central density ratio n_p/n_c = 1/0.55 (Frank et al.
# 2024 eq. 18 at the Hammir n_c/n_p = 0.55 design point). Matches the YAML default.
# e*phi = T_e * ln(n_p/n_c) is FIXED by the plug hardware and T_e, independent of
# T_i, so the Pastukhov enhancement weakens as T_i rises (drives the optimum cool).
_PLUG_DENSITY_RATIO = 1.818

# Pastukhov-Maxwellian validity floor on collisionality (matches YAML default
# collisionality_min = 1/R_m at R_m=10). Diagnostic-only threshold (Task 3).
_COLLISIONALITY_MIN = 0.1

# Alpha loss-cone heating fraction: fraction of fusion-alpha power that deposits
# in the central cell before scattering into the loss cone (Santarius & Callen
# 1983). Matches the YAML default. Task 2c.
_F_ALPHA_HEAT = 0.80

# Plug hot-electron temperature [keV] (Fowler-Logan potential; Hammir anchor).
# Decoupled from the central-cell T_e: the plug builds the confining potential
# e*phi = T_e_plug * ln(n_p/n_c) with hot electrons while the central cell sets
# radiation at its own (coolable) T_e. Matches the YAML default. Task 2e.
_TE_PLUG = 125.0

# Plug sustainment power [MW] charged into the mirror recirculating budget (the
# ECH/NBI holding the hot-electron plug, calibrated to Hammir's about 30 MW).
# Matches the YAML default. Task 2e.
_P_PLUG = 30.0


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
        plug_density_ratio=args.pop("plug_density_ratio", _PLUG_DENSITY_RATIO),
        collisionality_min=args.pop("collisionality_min", _COLLISIONALITY_MIN),
        T_e_plug=args.pop("T_e_plug", _TE_PLUG),
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

    @pytest.mark.skipif(
        not HAS_JAX,
        reason="exercises jax.grad/jit directly; numpy mode uses finite differences",
    )
    def test_jit_matches_eager(self):
        def chain(n, T):
            tii = compute_tau_ii(n, T, 2.5)
            phi = compute_ambipolar_potential(T, 2.5)
            return compute_tau_pastukhov(tii, R_m=10.0, phi_keV=phi, T_i=T)

        eager = float(chain(_N, _TI))
        jitted = float(jax.jit(chain)(_N, _TI))
        assert jnp.isfinite(jitted)
        assert jitted == pytest.approx(eager, rel=1e-4)

    @pytest.mark.skipif(
        not HAS_JAX,
        reason="exercises jax.grad/jit directly; numpy mode uses finite differences",
    )
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
    @pytest.mark.skipif(
        not HAS_JAX,
        reason="exercises jax.grad/jit directly; numpy mode uses finite differences",
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
                plug_density_ratio=_PLUG_DENSITY_RATIO,
                collisionality_min=_COLLISIONALITY_MIN,
                T_e_plug=_TE_PLUG,
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
        plug_density_ratio=kw.pop("plug_density_ratio", _PLUG_DENSITY_RATIO),
        collisionality_min=kw.pop("collisionality_min", _COLLISIONALITY_MIN),
        f_alpha_heat=kw.pop("f_alpha_heat", _F_ALPHA_HEAT),
        T_e_plug=kw.pop("T_e_plug", _TE_PLUG),
        p_plug=kw.pop("p_plug", _P_PLUG),
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
            plug_density_ratio=_PLUG_DENSITY_RATIO,
            collisionality_min=_COLLISIONALITY_MIN,
            f_alpha_heat=_F_ALPHA_HEAT,
            T_e_plug=_TE_PLUG,
            p_plug=_P_PLUG,
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
# $/MWh at 500 MW, avail=0.87, lifetime=40 yr (non-0D default path).
# Re-pinned 2026-06-15 at the master+mirror merge; this value reflects BOTH
# independent changes landing on it: (1) construction_time_yr now reads the mirror
# YAML value (5.0 yr) instead of a generic 6.0 signature default, lowering
# IDC/indirect; (2) the fluence-based CAS72 basis change (wall_limits_and_fluence.md)
# plus the central-cell electron temperature the non-0D path reads from YAML for
# its radiation term. The coil calibration pin (513.375 M$) is unaffected (T_e
# does not enter coils).
# re-pinned: CAS10 land changed to sqrt(plant-total power) scaling (CAS50/40/70
# n_mod fixes are no-ops at n_mod=1), then gross-electric reference unified to
# 1100 MWe (ref_gross_power_mwe); LCOE moved 100.5444 -> 100.5561 -> 100.4065.
# re-pinned: central-cell T_i and T_e corrected to the near-Maxwellian Hammir/WHAM
# value (10 keV), from the prior 125 keV plug hot-electron value that had been read
# onto the bulk cell. This removes the spurious high-T_e synchrotron term, so the
# default-path LCOE moved 100.4065 -> 98.9895 (Q_eng 3.5 -> 4.6, recirc 28% -> 22%).
_MIRROR_DT_PINNED_LCOE = 98.98950958251953


class TestModelIntegration:
    def test_default_path_bit_identical(self):
        """use_0d_model=False (default) must produce the pinned LCOE exactly."""
        m = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
        r = m.forward(net_electric_mw=500.0, availability=0.87, lifetime_yr=40.0)
        # Exact equality: same deterministic float path, no solver involved.
        # Pinned value is float32; rel=1e-12 is tighter than float32 epsilon,
        # so this is bit-identity in practice.
        assert r.costs.lcoe == pytest.approx(
            _MIRROR_DT_PINNED_LCOE, rel=1e-12 if HAS_JAX else 1e-6
        )

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
        assert r.costs.lcoe == pytest.approx(
            _MIRROR_DT_PINNED_LCOE, rel=1e-12 if HAS_JAX else 1e-6
        )

    def test_doubling_length_doubles_central_contribution(self):
        """Doubling chamber_length doubles the central-coil kA*m (and cost share).

        The plug contribution (fixed n_plug_coils, fixed b_plug) must be
        UNCHANGED.  We test the contributions, not just the total.
        """

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
        cost_per_kAm = load_costing_constants().conductor_cost_rebco  # 50.0

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
# Plasma radius [m]. Raised from 0.4 to 1.5 with the fixed Fowler-Logan plug:
# the tandem hot-electron central cell (T_e = 125 keV) holds density at the beta
# cap, so a small-radius machine cannot reach the test power targets in the
# genuinely DRIVEN regime; the 1.5 m radius (the YAML-defaults machine) keeps the
# 200-600 MW sizing targets feasible with headroom.
_SZ_A = 1.5  # plasma radius [m]
_SZ_B = 3.0  # midplane field [T]
_SZ_R_M = 10.0  # mirror ratio
# Tandem hot-electron central cell (Hammir anchor, matches YAML). The Fowler-
# Logan plug e*phi = T_e*ln(n_p/n_c) needs hot electrons to confine; held fixed
# while the GSS scans T_i so e*phi/T_i falls with T_i (drives the optimum cool).
_SZ_T_E = 125.0  # electron temperature [keV] (held fixed; GSS scans T_i)
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
        plug_density_ratio=_PLUG_DENSITY_RATIO,
        collisionality_min=_COLLISIONALITY_MIN,
        f_alpha_heat=_F_ALPHA_HEAT,
        T_e_plug=_TE_PLUG,
        p_plug=_P_PLUG,
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
            net_electric_mw=400.0,
            availability=0.87,
            lifetime_yr=40.0,
            size_from_power=True,
        )
        # 400 MWe (not 600) is the high target: in the driven hot-electron
        # regime the YAML-default a=1.5,B=3 machine tops out near 455 MWe at
        # L_max, so 600 MWe is infeasible via the CostModel default path.
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

    def test_dhe3_is_infeasible_where_dt_sizes(self):
        """D-He3 is infeasible in the tandem hot-electron plug regime.

        The Fowler-Logan plug e*phi = T_e*ln(n_p/n_c) requires hot electrons
        (T_e raised to confine the plug). Hot electrons make D-He3
        radiation/aux-dominated: at every probed geometry the net electric
        power is large and negative (about -2900 MW at L_max here), so D-He3
        sizing raises SizingInfeasible. This is the honest result of the
        density-ratio plug change; D-He3 is not forced feasible.

        At the same geometry (B=5 T, a=0.6 m, R_m=20, T_e=50) D-T still sizes
        to a finite, positive chamber length, so the failure is fuel-specific,
        not a geometry artifact.
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
            plug_density_ratio=_PLUG_DENSITY_RATIO,
            collisionality_min=_COLLISIONALITY_MIN,
            f_alpha_heat=_F_ALPHA_HEAT,
            T_e_plug=_TE_PLUG,
            p_plug=_P_PLUG,
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
        assert isinstance(L_dt, float) and L_dt > 0.0, (
            f"D-T should size to a finite positive L, got {L_dt}"
        )
        with pytest.raises(SizingInfeasible):
            mirror_size_from_power(p_dhe3, Fuel.DHE3)

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

        In the tandem hot-electron plug regime (T_e raised to confine the
        Fowler-Logan plug) the 400 MWe example machine is beta-bound, not
        wall-bound: hot electrons raise the pressure, so at fixed beta the
        density is lower and the neutron wall load stays well under q_wall_max.
        Beta therefore sits exactly at the f_beta ceiling (f_beta * beta_max)
        and below beta_max. The regression this guards (a re-solve reporting
        beta > beta_max and tripping OperatingPointInfeasible) is still
        exercised: building the state directly from the GSS optimum keeps beta
        bounded by construction.
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
        q_wall_max = 5.0  # YAML default neutron wall-load cap
        # Beta-bound: beta sits at the f_beta ceiling and below beta_max.
        assert float(ps.beta) == pytest.approx(0.95 * beta_max, rel=1e-4)
        assert float(ps.beta) < beta_max
        # The wall cap is NOT binding here (wall_loading about 1.06, well
        # below q_wall_max=5.0): the f_beta ceiling is what binds.
        assert float(ps.wall_loading) < q_wall_max

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
    @pytest.mark.skipif(
        not HAS_JAX,
        reason="exercises jax.grad/jit directly; numpy mode uses finite differences",
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
    # Tandem hot-electron central cell (Hammir anchor). The Fowler-Logan plug
    # potential e*phi = T_e*ln(n_p/n_c) requires hot electrons to confine the
    # central cell; a cold T_e (20 keV) gives e*phi = 12 keV, too shallow to plug,
    # and the machine is infeasible. Matches the YAML default T_e = 125 keV.
    T_e=125.0,
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
    plug_density_ratio=_PLUG_DENSITY_RATIO,
    collisionality_min=_COLLISIONALITY_MIN,
    f_alpha_heat=_F_ALPHA_HEAT,
    T_e_plug=_TE_PLUG,
    p_plug=_P_PLUG,
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
        plug_density_ratio=params["plug_density_ratio"],
        collisionality_min=params["collisionality_min"],
        T_e_plug=params["T_e_plug"],
        f_rad_fus=params.get("f_rad_fus"),
    )
    return L, pn, ps


@pytest.mark.slow
class TestWallConstraint:
    def test_sizing_respects_q_wall_max(self):
        # In the tandem hot-electron plug regime (T_e=125), the YAML-default
        # a=1.5,B=3 machine is BETA-bound even at q_wall_max=5.0: hot electrons
        # raise pressure, so at fixed beta the density (and hence neutron wall
        # load) is low and the f_beta cap binds first. To keep a genuine
        # WALL-bound test, this case uses a higher-field, smaller-radius machine
        # (a=0.5, B_min=6.0) with a tighter wall cap (q_wall_max=2.0), where
        # n_wall(T) < n_beta(T) across the GSS T_i bracket.
        params = dict(_SIZING_PARAMS, a=0.5, B_min=6.0, q_wall_max=2.0)
        L, pnet, state = _size(params)
        # wall cap (2.0) is the active, binding constraint:
        assert float(state.wall_loading) <= 2.0 * 1.001
        assert float(state.wall_loading) >= 1.9
        # beta sits BELOW the f_beta * beta_max ceiling: the wall cap, not the
        # f_beta cap, is what binds here.
        assert float(state.beta) < 0.85 * 0.5

    def test_loose_cap_recovers_beta_bound_solution(self):
        # q_wall_max=50 must reproduce the (beta-bound) solution at the closed
        # energy balance.
        # re-pinned for the driven hot-electron regime: the Fowler-Logan plug
        # e*phi = T_e*ln(n_p/n_c) with T_e=125 raises pressure, so at fixed beta
        # the density is lower and a longer chamber is needed to meet the target.
        # re-pinned 2026-06-15: explicit plug power (P_plug = 30 MW charged into
        # recirculating, Task 2e) raises the recirculating cost, so more fusion
        # power and a longer chamber are needed: L = 77.842279425 m here
        # (was 69.230740222 m before the plug power was charged explicitly). See
        # docs/account_justification/mirror_confinement_regimes.md.
        params = dict(_SIZING_PARAMS, q_wall_max=50.0)
        L, _, _ = _size(params)
        assert L == pytest.approx(77.842279425, rel=1e-4)

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
    @pytest.mark.skipif(
        not HAS_JAX,
        reason="exercises jax.grad/jit directly; numpy mode uses finite differences",
    )
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
    plug_density_ratio=_PLUG_DENSITY_RATIO,
    collisionality_min=_COLLISIONALITY_MIN,
    f_alpha_heat=_F_ALPHA_HEAT,
    T_e_plug=_TE_PLUG,
    p_plug=_P_PLUG,
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
    def test_pb11_sizing_is_infeasible_naming_q_surface_max(self):
        """p-B11 is infeasible in the driven hot-electron plug regime.

        p-B11 radiates 83% of fusion power; with the tandem hot-electron plug
        (T_e=150 to confine the Fowler-Logan plug e*phi = T_e*ln(n_p/n_c))
        the machine is radiation/aux-dominated. Net electric power is negative
        at every L and grows MORE negative with volume (probed about -577 MW at
        L=10 m down to -11240 MW at L_max=200 m with loose caps), so there is
        no feasible surface-bound target at this geometry. This is the honest
        result of the density-ratio plug change; p-B11 is not forced feasible.

        At q_surface_max=1.0 the surface cap is the lowest-density (binding) cap
        at L_max, so sizing raises SizingInfeasible naming q_surface_max.
        """
        with pytest.raises(SizingInfeasible, match=r"q_surface_max"):
            _size(_PB11_SIZING_PARAMS, fuel=Fuel.PB11)

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

    def test_loose_surface_cap_still_infeasible_not_from_surface(self):
        """Loosening the surface cap does NOT recover a p-B11 solution.

        In the driven hot-electron plug regime p-B11 is radiation/aux-dominated:
        net electric power is negative at every L even with loose caps
        (q_surface_max=50, q_wall_max=50 -> about -11240 MW at L_max). So sizing
        still raises SizingInfeasible. Crucially, with the caps loose the
        message does NOT name q_surface_max (no density cap binds; the power
        balance itself is negative). This proves the infeasibility is the
        hot-electron regime, not the surface heat-flux cap -- the tight-cap
        sibling test that names q_surface_max is therefore not trivially
        passing.
        """
        params = dict(_PB11_SIZING_PARAMS, q_surface_max=50.0, q_wall_max=50.0)
        with pytest.raises(SizingInfeasible) as exc:
            _size(params, fuel=Fuel.PB11)
        assert "q_surface_max" not in str(exc.value)

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
            plug_density_ratio=_PLUG_DENSITY_RATIO,
            collisionality_min=_COLLISIONALITY_MIN,
            T_e_plug=_TE_PLUG,
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
    @pytest.mark.skipif(
        not HAS_JAX,
        reason="exercises jax.grad/jit directly; numpy mode uses finite differences",
    )
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

    @pytest.mark.skipif(
        not HAS_JAX,
        reason="exercises jax.grad/jit directly; numpy mode uses finite differences",
    )
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
        p_aux = float(
            mirror_aux_heating(ps, p_aux_floor=2.0, f_alpha_heat=_F_ALPHA_HEAT)
        )
        expected = max(
            2.0,
            float(ps.p_end)
            + float(ps.p_radial)
            + float(ps.p_rad)
            - _F_ALPHA_HEAT * float(ps.p_alpha),
        )
        assert p_aux == pytest.approx(expected, rel=1e-6)

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
        # The shared transport channel carries the real axial/radial loss PLUS
        # the loss-cone alpha exhaust (1 - f_alpha_heat) * p_alpha routed here as
        # directed exhaust (alpha loss-cone heating, Task 2c).
        lost_alpha = (1.0 - _F_ALPHA_HEAT) * float(ps.p_alpha)
        p_transport_expected = float(ps.p_end) + float(ps.p_radial) + lost_alpha
        assert p_transport == pytest.approx(p_transport_expected, rel=0.05)

    def test_sized_dt_optimum_is_realistic_temperature(self):
        # THE headline regression: with the FIXED Fowler-Logan plug potential
        # e*phi = T_e*ln(n_p/n_c) (independent of T_i, hot-electron tandem T_e =
        # 125 keV), the D-T sizing optimum lands in a realistic TANDEM band, not the
        # spurious ~60 keV ignition the unbounded Boltzmann potential produced nor
        # the wall-cap-pinned near-ignited ~35 keV the ratio-to-T_i shortcut left.
        #
        # The band is tandem-appropriate, NOT the simple-mirror GDT/WHAM band
        # (those single-cell devices run cool, ~10 keV). A tandem CENTRAL CELL
        # runs hotter: the Realta Hammir Q>5 design point is T_i = 45 keV (Frank
        # et al. 2024). With the fixed plug the optimum settles at about 23 keV: the
        # Pastukhov enhancement exp(e*phi/T_i) WEAKENS as T_i rises (e*phi is fixed
        # by the plug, not bought by heating), so the central cell no longer rides
        # to ignition and the point is genuinely DRIVEN (P_aux ~ 230 MW, well off the
        # floor). See docs/account_justification/mirror_confinement_regimes.md.
        m = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
        r = m.forward(
            net_electric_mw=400.0,
            availability=0.87,
            lifetime_yr=40.0,
            size_from_power=True,
            f_beta=0.85,
        )
        assert 15.0 <= float(r.plasma_state.T_i) <= 35.0

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
        aux = float(mirror_aux_heating(ps, p_aux_floor=2.0, f_alpha_heat=_F_ALPHA_HEAT))
        assert float(ps.sustainment_ratio) == pytest.approx(50.0 / aux, rel=1e-4)


class TestAlphaHeating:
    """Alpha loss-cone heating fraction (Santarius & Callen 1983).

    A magnetic mirror loses fusion alphas out the loss cone before they fully
    thermalise: about 50 percent by count but under 25 percent by energy, so
    about 75-85 percent of the alpha power deposits as central-cell self-heating.
    The model credits f_alpha_heat * p_alpha (default 0.80) instead of the full
    p_alpha, which un-floors the auxiliary drive and lands the D-T optimum at a
    genuinely DRIVEN tandem operating point. The lost fraction
    (1 - f_alpha_heat) * p_alpha exits the loss cone as directed exhaust and is
    accounted in the axial end-loss / DEC channel (energy bookkeeping closes).
    See docs/account_justification/mirror_confinement_regimes.md.
    """

    def test_aux_heating_uses_alpha_fraction(self):
        # mirror_aux_heating subtracts f_alpha_heat * p_alpha, NOT the full
        # p_alpha, so a sub-unity f_alpha_heat raises the required auxiliary
        # power. Closed-form check at a fixed forward state. Use a shallow plug
        # (T_e_plug=20) so the point is genuinely driven (aux above the floor),
        # which is required for the f_alpha_heat < 1 vs = 1 comparison to bite;
        # at the hot default plug (T_e_plug=125) confinement is deep and aux
        # floors at both fractions (plug decoupled from central cell, Task 2e).
        ps = _forward(T_i=30.0, T_e_plug=20.0)
        f_alpha = 0.80
        p_aux = float(mirror_aux_heating(ps, p_aux_floor=2.0, f_alpha_heat=f_alpha))
        expected = max(
            2.0,
            float(ps.p_end)
            + float(ps.p_radial)
            + float(ps.p_rad)
            - f_alpha * float(ps.p_alpha),
        )
        assert p_aux == pytest.approx(expected, rel=1e-6)
        # Full-deposition (f_alpha_heat=1.0) requires strictly less aux power.
        p_aux_full = float(mirror_aux_heating(ps, p_aux_floor=2.0, f_alpha_heat=1.0))
        assert p_aux > p_aux_full

    def test_dt_optimum_is_driven(self):
        # THE headline regression: with the alpha loss-cone reduction the D-T
        # sizing optimum is genuinely DRIVEN -- the auxiliary power is well above
        # the 2 MW control floor and the scientific gain drops from the spurious
        # near-ignited ~516 to a tandem-realistic value.
        m = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
        r = m.forward(
            net_electric_mw=400.0,
            availability=0.87,
            lifetime_yr=40.0,
            size_from_power=True,
            f_beta=0.85,
        )
        pt = r.power_table
        assert float(pt.p_input) > 10.0, (
            f"P_aux = {float(pt.p_input):.1f} MW still near the floor; "
            "alpha loss-cone reduction did not un-floor the drive"
        )
        assert float(pt.q_sci) < 50.0, (
            f"q_sci = {float(pt.q_sci):.1f} still near-ignited (was ~516)"
        )

    def test_alpha_loss_routed_to_end_channel(self):
        # The lost alpha fraction (1 - f_alpha_heat) * p_alpha must appear in the
        # shared balance's transport channel so no power vanishes. With the
        # mirror feeding p_input = P_aux and p_rad_override = ps.p_rad, the shared
        # p_transport = p_ash + p_input_eff - p_rad collapses to
        # P_end + P_radial + (1 - f_alpha_heat) * p_alpha. Energy bookkeeping
        # closes: the un-deposited alpha power is conserved as directed exhaust.
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
        lost_alpha = (1.0 - _F_ALPHA_HEAT) * float(ps.p_alpha)
        p_transport_expected = float(ps.p_end) + float(ps.p_radial) + lost_alpha
        assert p_transport == pytest.approx(p_transport_expected, rel=0.05)
        # The lost alpha power is a material part of the channel, not noise.
        assert lost_alpha > 1.0

    @pytest.mark.parametrize("fuel", [Fuel.DT, Fuel.DD, Fuel.DHE3, Fuel.PB11])
    @pytest.mark.skipif(
        not HAS_JAX,
        reason="exercises jax.grad/jit directly; numpy mode uses finite differences",
    )
    def test_alpha_heating_jit_and_grad(self, fuel):
        # jit == eager and finite grad through mirror_aux_heating across fuels.
        from costingfe.layers.mirror import mirror_aux_heating as _aux

        def g(f_alpha):
            ps = _forward(fuel=fuel, T_i=40.0)
            return _aux(ps, p_aux_floor=2.0, f_alpha_heat=f_alpha)

        eager = float(g(0.80))
        jitted = float(jax.jit(g)(0.80))
        assert jnp.isfinite(jitted)
        assert jitted == pytest.approx(eager, rel=1e-4)
        grad = float(jax.grad(g)(0.80))
        assert jnp.isfinite(grad)


class TestTandemConfinement:
    """Tandem plug-limited central-cell confinement calibrated to Realta Hammir.

    The confining potential is the Fowler & Logan / Frank et al. tandem value
    e*phi = T_e * ln(n_p/n_c) (bounded by the plug/central density ratio), NOT
    the unbounded simple-mirror Boltzmann value. Calibrated to e*phi/T_i = 1.66
    at the Hammir nc/np = 0.55 design point so the central cell reproduces the
    Q > 5 design point and is not spuriously ignited. See
    docs/account_justification/mirror_confinement_regimes.md (Frank et al. 2024,
    arXiv 2411.06644).
    """

    def test_hammir_anchor_reproduces_Q(self):
        """At the published Hammir central cell the model's Q matches Q > 5.

        Hammir Q > 5 design point (Frank et al. 2024 sec. 3.5): L_c = 50 m,
        central cell B0c = 3 T, central cell mirror ratio Rmc = 13.3, n_c =
        0.825e20 m^-3 (nc/np = 0.55, np = 1.5e20), T_i = 45 keV, T_e = 125 keV,
        central cell radius a_c = 0.5 m (flux conservation from am = 0.15 m).
        Published: P_fus = 157.4 MW, P_NBI(plug) = 30 MW, Q = P_fus/P_NBI = 5.2.

        The model reports gain as q_sci = p_fus / p_input_eff, with p_input the
        confinement-required sustainment power (the mirror analog of the plug
        NBI). Assert it lands within 2x of the published Q = 5.2.
        """
        ps, _pt = mirror_0d_inverse(
            p_net_target=50.0,
            L=50.0,
            a=0.5,
            B_min=3.0,
            R_m=13.3,
            n_e=0.825e20,
            T_e=125.0,
            fuel=Fuel.DT,
            beta_max=0.9,
            q_wall_max=50.0,
            q_surface_max=50.0,
            p_aux_floor=2.0,
            plug_density_ratio=_PLUG_DENSITY_RATIO,
            collisionality_min=_COLLISIONALITY_MIN,
            f_alpha_heat=_F_ALPHA_HEAT,
            T_e_plug=_TE_PLUG,
            p_plug=_P_PLUG,
            enforce_plasma_limits=False,
            dhe3_dd_frac=0.131,
            dhe3_dd_frac_pin=None,
            vacuum_t=0.10,
            **_FRACS,
            **_MIX,
            **_PB_KWARGS,
        )
        # Build the forward state directly at the published central-cell point
        # (T_i fixed at 45 keV, not solved) to read the central-cell gain.
        psf = _forward(
            L=50.0,
            a=0.5,
            B_min=3.0,
            R_m=13.3,
            T_i=45.0,
            T_e=125.0,
            n_e=0.825e20,
        )
        p_aux = float(
            mirror_aux_heating(psf, p_aux_floor=2.0, f_alpha_heat=_F_ALPHA_HEAT)
        )
        q_model = float(psf.p_fus) / p_aux
        assert 0.5 * 5.2 <= q_model <= 2.0 * 5.2, (
            f"Hammir anchor: model Q = {q_model:.2f} (P_fus={float(psf.p_fus):.1f} "
            f"MW, P_aux={p_aux:.1f} MW) vs published 5.2, outside 2x"
        )
        # The inverse path, driven only by the net-electric target at the same
        # geometry/density, must independently recover the Hammir design point:
        # its bisected ion temperature should land near the published 45 keV
        # (within 2x), confirming forward and inverse agree at the anchor.
        assert 0.5 * 45.0 <= float(ps.T_i) <= 2.0 * 45.0, (
            f"Hammir anchor: inverse-solved T_i = {float(ps.T_i):.1f} keV vs "
            f"published 45 keV, outside 2x"
        )
        # Central-cell axial confinement should be of order the published tau_c
        # ~ 5 s (Frank et al. 2024 sec. 3.5), within 2x.
        assert 0.5 * 5.0 <= float(psf.tau_E) <= 2.0 * 5.0

    def test_not_spuriously_ignited(self):
        """At the published Hammir central cell the plasma is NOT ignited.

        Real tandems are plug-limited: external sustainment power is genuinely
        required (P_aux above the floor), giving a finite tandem-realistic Q of
        order a few, not the spurious ignition the unbounded Boltzmann potential
        produced (which floored aux and gave Q in the tens-to-hundreds).
        """
        psf = _forward(
            L=50.0, a=0.5, B_min=3.0, R_m=13.3, T_i=45.0, T_e=125.0, n_e=0.825e20
        )
        p_aux = float(
            mirror_aux_heating(psf, p_aux_floor=2.0, f_alpha_heat=_F_ALPHA_HEAT)
        )
        # Genuinely externally driven: aux is well above the 2 MW control floor.
        assert p_aux > 5.0, f"aux floored at {p_aux:.1f} MW -> spuriously ignited"
        q_model = float(psf.p_fus) / p_aux
        assert q_model < 15.0, f"Q = {q_model:.1f} too high; tandem is plug-limited"

    @pytest.mark.parametrize("fuel", [Fuel.DT, Fuel.DD, Fuel.DHE3, Fuel.PB11])
    @pytest.mark.skipif(
        not HAS_JAX,
        reason="exercises jax.grad/jit directly; numpy mode uses finite differences",
    )
    def test_plug_potential_jit_and_grad(self, fuel):
        # e*phi = T_e * ln(n_p/n_c): differentiate w.r.t. the central-cell electron
        # temperature T_e (the plug potential is set by T_e and the fixed density
        # ratio, NOT by T_i). grad should be ln(n_p/n_c), finite and constant.
        from costingfe.layers.mirror import compute_plug_potential

        def f(T_e):
            return compute_plug_potential(T_e, _PLUG_DENSITY_RATIO)

        eager = float(f(125.0))
        jitted = float(jax.jit(f)(125.0))
        assert jnp.isfinite(jitted)
        assert jitted == pytest.approx(eager, rel=1e-4)
        # Hammir anchor: at T_e = 125 keV, e*phi = 125 * ln(1.818) = 74.7 keV.
        assert eager == pytest.approx(125.0 * math.log(_PLUG_DENSITY_RATIO), rel=1e-4)
        g = float(jax.grad(f)(125.0))
        assert jnp.isfinite(g)
        assert g == pytest.approx(math.log(_PLUG_DENSITY_RATIO), rel=1e-4)


class TestPlugDecoupling:
    """Decouple the hot-electron plug from the coolable central cell (Task 2e).

    Real advanced-fuel tandems run a HOT-ELECTRON plug (set by plug ECH,
    independent of the central fuel) that builds the Fowler-Logan confining
    potential e*phi = T_e_plug * ln(n_p/n_c), DISTINCT from a central cell that
    keeps its electrons cool to limit bremsstrahlung. A single electron
    temperature cannot serve both (hot plug vs cool aneutronic central cell), so
    the plug temperature T_e_plug is separated from the central-cell T_e. The
    plug's sustainment power P_plug (ECH/NBI, calibrated to Hammir's about 30 MW)
    is charged into the mirror recirculating power. See
    docs/account_justification/mirror_confinement_regimes.md.
    """

    def test_plug_potential_uses_plug_temperature(self):
        # e*phi = T_e_plug * ln(n_p/n_c) is set by the PLUG temperature, NOT the
        # central-cell T_e. The forward's phi must track T_e_plug and ignore the
        # central T_e: hold T_e_plug fixed and vary central T_e -> phi unchanged;
        # vary T_e_plug -> phi scales with it.
        ps_a = _forward(T_i=20.0, T_e=20.0, T_e_plug=125.0)
        ps_b = _forward(T_i=20.0, T_e=60.0, T_e_plug=125.0)
        # Central T_e changed 20 -> 60 keV but the plug potential is unchanged.
        assert float(ps_a.phi) == pytest.approx(float(ps_b.phi), rel=1e-5)
        assert float(ps_a.phi) == pytest.approx(
            125.0 * math.log(_PLUG_DENSITY_RATIO), rel=1e-4
        )
        # Changing the PLUG temperature DOES move phi (it is the plug knob).
        ps_c = _forward(T_i=20.0, T_e=20.0, T_e_plug=60.0)
        assert float(ps_c.phi) == pytest.approx(
            60.0 * math.log(_PLUG_DENSITY_RATIO), rel=1e-4
        )
        assert float(ps_c.phi) < float(ps_a.phi)

    def test_cool_central_helps_advanced_fuel(self):
        # At a FIXED hot plug, lowering the central-cell T_e sharply cuts
        # bremsstrahlung, so a cool-central advanced-fuel run has far higher net
        # electric than the hot-central case. Evaluate D-He3 net power as an
        # OUTPUT at a fixed chamber length (net_electric_at_L runs the sizing-mode
        # operating point, GSS over T_i with the energy-balance closure), holding
        # the plug hot (T_e_plug = 125) and varying only the central-cell T_e.
        # Hot central radiates heavily (net deeply negative); cool central
        # radiates far less and p_net rises sharply -- the fair-evaluation
        # regression the decoupling enables.
        base = dict(
            _SIZING_PARAMS,
            a=0.5,
            B_min=6.0,
            q_wall_max=50.0,
            q_surface_max=50.0,
            beta_max=0.9,
            T_e_plug=125.0,  # hot plug, fixed across both runs
            f_rad_fus=None,
            n_mod=1,
        )
        p_hot = dict(base, T_e=125.0)  # hot central cell: heavy bremsstrahlung
        p_cool = dict(base, T_e=30.0)  # cool central cell: plug stays hot
        L = 60.0
        pn_hot = net_electric_at_L(L, p_hot, Fuel.DHE3)
        pn_cool = net_electric_at_L(L, p_cool, Fuel.DHE3)
        # Cooling the central cell raises net electric by a large margin (brem
        # drops) at the same machine and the same hot plug.
        assert pn_cool > pn_hot, (
            f"cool central p_net={pn_cool:.1f} not above hot "
            f"central p_net={pn_hot:.1f}; decoupling did not help D-He3"
        )
        # The improvement is sizeable, not marginal.
        assert pn_cool - pn_hot > 50.0

    def test_plug_power_in_recirculating(self):
        # P_plug appears in the recirculating power: charging a larger plug power
        # raises recirculating and lowers q_eng (the competing penalty that keeps
        # the optimizer honest). Audit two inverse runs differing only in p_plug.
        _ps0, pt0 = _inverse(fuel=Fuel.DT, p_plug=0.0)
        _ps1, pt1 = _inverse(fuel=Fuel.DT, p_plug=60.0)
        # More plug power -> lower engineering gain (more recirculating load).
        assert float(pt1.q_eng) < float(pt0.q_eng), (
            f"q_eng did not fall with plug power: "
            f"p_plug=0 -> {float(pt0.q_eng):.3f}, "
            f"p_plug=60 -> {float(pt1.q_eng):.3f}"
        )
        # The recirculating term carries the extra plug power: p_coils enters the
        # recirculating sum at unit cost, so q_eng's recirculating denominator
        # rises by the 60 MW difference.
        p_net0 = float(pt0.p_net)
        p_net1 = float(pt1.p_net)
        assert p_net1 < p_net0

    def test_dt_unchanged_within_tolerance(self):
        # D-T (hot central cell, Hammir-consistent) still drives at about 23 keV
        # and about 369 LCOE, now with the plug power charged EXPLICITLY. The
        # explicit 30 MW plug power adds a modest recirculating cost on top of the
        # already-driven point, so LCOE may shift slightly; re-pinned here with a
        # 10 percent tolerance to absorb the explicit-plug-power increment while
        # confirming the driven D-T regime is preserved.
        m = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
        r = m.forward(
            net_electric_mw=400.0,
            availability=0.87,
            lifetime_yr=40.0,
            size_from_power=True,
            f_beta=0.85,
        )
        T_i = float(r.plasma_state.T_i)
        assert 18.0 <= T_i <= 28.0, f"D-T T_i={T_i:.1f} keV outside the driven band"
        lcoe = float(r.costs.lcoe)
        # Within 10 percent of the 369 $/MWh driven D-T result (now with explicit
        # plug power). A larger shift would mean the plug-power charge broke the
        # D-T regime, not a tolerable re-pin.
        assert 330.0 <= lcoe <= 410.0, f"D-T LCOE={lcoe:.1f} outside tolerance of 369"

    @pytest.mark.parametrize("fuel", [Fuel.DT, Fuel.DD, Fuel.DHE3, Fuel.PB11])
    @pytest.mark.skipif(
        not HAS_JAX,
        reason="exercises jax.grad/jit directly; numpy mode uses finite differences",
    )
    def test_plug_decoupling_jit_and_grad(self, fuel):
        # jit == eager and finite grad of the forward phi w.r.t. T_e_plug across
        # fuels (the plug potential is the decoupled knob).
        def f(T_e_plug):
            ps = _forward(fuel=fuel, T_i=40.0, T_e=40.0, T_e_plug=T_e_plug)
            return ps.phi

        eager = float(f(125.0))
        jitted = float(jax.jit(f)(125.0))
        assert jnp.isfinite(jitted)
        assert jitted == pytest.approx(eager, rel=1e-4)
        g = float(jax.grad(f)(125.0))
        assert jnp.isfinite(g)
        assert g == pytest.approx(math.log(_PLUG_DENSITY_RATIO), rel=1e-4)


class TestStabilityDiagnostics:
    """Collisionality-validity flag and DCLC microstability diagnostic.

    Both are DIAGNOSTICS (Task 3), not constraints. The validity flag reports
    where the bare Pastukhov-Maxwellian assumption is stretched (deeply
    collisionless); a tandem legitimately runs there and plugged, so it is
    informational. The DCLC parameter is the number of ion gyroradii across the
    plasma (Post loss-cone microstability criterion). See
    docs/account_justification/mirror_confinement_regimes.md.
    """

    def test_collisionality_validity_flag_fires_when_collisionless(self):
        # collisionality = L/mfp; below the Pastukhov-Maxwellian validity
        # threshold (collisionality_min) the diagnostic flags overestimated
        # confinement. Hot + low density is collisionless; cold + high density
        # is collisional.
        ps_hot = _forward(T_i=60.0, T_e=60.0, n_e=1.0e20)  # collisionless
        ps_cold = _forward(T_i=2.0, T_e=2.0, n_e=5.0e20)  # collisional
        # The flag is a bool-as-float: 0.0 when collisionless (invalid), 1.0 when
        # collisional enough for the Maxwellian Pastukhov assumption to hold.
        assert float(ps_hot.pastukhov_valid) == 0.0
        assert float(ps_cold.pastukhov_valid) == 1.0
        assert float(ps_cold.collisionality) > float(ps_hot.collisionality)

    def test_dclc_diagnostic_present_and_finite_all_fuels(self):
        for fuel in (Fuel.DT, Fuel.DD, Fuel.DHE3, Fuel.PB11):
            ps = _forward(fuel=fuel, T_i=40.0, T_e=40.0)
            assert math.isfinite(float(ps.dclc_parameter))
            # Number of ion gyroradii across the plasma is strictly positive.
            assert float(ps.dclc_parameter) > 0.0

    def test_dclc_parameter_equals_a_over_rho_i(self):
        # dclc_parameter = a / rho_i, the number of ion gyroradii across the
        # plasma radius (Post loss-cone microstability criterion). rho_i is the
        # midplane gyroradius at B_min, identical to the radial-transport kernel.
        a, B_min, T_i = _A, _B_MIN, 30.0
        ps = _forward(a=a, B_min=B_min, T_i=T_i, T_e=T_i)
        rho_i = _RHO_I_PREFACTOR * math.sqrt(2.5 * T_i) / B_min
        assert float(ps.dclc_parameter) == pytest.approx(a / rho_i, rel=1e-4)

    @pytest.mark.parametrize("fuel", [Fuel.DT, Fuel.DD, Fuel.DHE3, Fuel.PB11])
    @pytest.mark.skipif(
        not HAS_JAX,
        reason="exercises jax.grad/jit directly; numpy mode uses finite differences",
    )
    def test_dclc_kernel_jit_matches_eager_and_differentiable(self, fuel):
        from costingfe.layers.mirror import compute_dclc_parameter

        def f(T_i):
            return compute_dclc_parameter(a=0.5, T_i=T_i, A=2.5, B_min=3.0)

        eager = float(f(30.0))
        jitted = float(jax.jit(f)(30.0))
        assert jnp.isfinite(jitted)
        assert jitted == pytest.approx(eager, rel=1e-4)
        g = float(jax.grad(f)(30.0))
        assert jnp.isfinite(g)

    def test_state_field_count(self):
        # Pin the MirrorPlasmaState field set so a stray add/remove is caught.
        # Task 3 adds pastukhov_valid and dclc_parameter (26 fields total).
        import dataclasses

        names = {f.name for f in dataclasses.fields(MirrorPlasmaState)}
        assert "pastukhov_valid" in names
        assert "dclc_parameter" in names
        assert len(names) == 26


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


# ---------------------------------------------------------------------------
# compute_K_ie tests
# ---------------------------------------------------------------------------


def test_K_ie_reference_point():
    # n_e=n_i=1e20 m^-3, T_i=15, T_e=10 keV, Z=1, A=2.5 (D-T), V=1 m^3, lnL=17.
    # nu_eps about 2.22 s^-1; power density about 0.267 MW/m^3.
    from costingfe.layers.mirror import compute_K_ie

    p = float(compute_K_ie(1e20, 1e20, 15.0, 10.0, 1.0, 2.5, 1.0))
    assert abs(p - 0.267) < 0.02


def test_K_ie_sign_and_zero():
    from costingfe.layers.mirror import compute_K_ie

    # T_e > T_i -> electrons give energy to ions -> negative.
    assert float(compute_K_ie(1e20, 1e20, 10.0, 20.0, 1.0, 2.5, 1.0)) < 0.0
    # T_e == T_i -> zero.
    assert abs(float(compute_K_ie(1e20, 1e20, 12.0, 12.0, 1.0, 2.5, 1.0))) < 1e-9
