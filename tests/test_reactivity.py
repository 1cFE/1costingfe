"""Tests for the fusion reactivity fits and fuel-mix algebra."""

import math

import pytest

from costingfe._backend import HAS_JAX

if HAS_JAX:
    import jax
    import jax.numpy as jnp
from costingfe.layers.physics import ash_neutron_split, event_energies
from costingfe.layers.reactivity import (
    fusion_power,
    n_i_over_n_e,
    sigv_dd_n,
    sigv_dd_p,
    sigv_dhe3,
    sigv_dt,
    sigv_pb11,
    z_eff_fuel,
)
from costingfe.types import Fuel

# Reference values computed in float64 from the Bosch-Hale (1992) coefficients
# as transcribed in examples/dhe3_mix_optimization.py (verified against the
# published tables there), and from the Nevins-Swain (2000) HT-branch fit
# (coefficients per Tentori & Belloni, Nucl. Fusion 63 (2023), table 2).
# Package runs float32, hence the 1e-3 relative tolerance.
_REF = {
    # T_keV: (sigv_dt, sigv_dhe3, sigv_dd_n, sigv_dd_p) [m^3/s]
    15.0: (2.7399e-22, 1.1754e-24, 1.4810e-24, 1.3900e-24),
    20.0: (4.3302e-22, 3.4821e-24, 2.6027e-24, 2.3990e-24),
    50.0: (8.6491e-22, 5.5539e-23, 1.1330e-23, 9.8383e-24),
    100.0: (8.4477e-22, 1.7185e-22, 2.6817e-23, 2.2439e-23),
}
_REF_PB11 = {
    # T_keV: sigv_pb11 [m^3/s], Nevins-Swain HT branch
    100.0: 6.1526e-23,
    200.0: 2.4274e-22,
    300.0: 3.3852e-22,
    400.0: 3.6982e-22,
}


class TestFitValues:
    @pytest.mark.parametrize("T", sorted(_REF))
    def test_bosch_hale_fits_match_float64_reference(self, T):
        ref_dt, ref_dhe3, ref_ddn, ref_ddp = _REF[T]
        assert float(sigv_dt(T)) == pytest.approx(ref_dt, rel=1e-3)
        assert float(sigv_dhe3(T)) == pytest.approx(ref_dhe3, rel=1e-3)
        assert float(sigv_dd_n(T)) == pytest.approx(ref_ddn, rel=1e-3)
        assert float(sigv_dd_p(T)) == pytest.approx(ref_ddp, rel=1e-3)

    @pytest.mark.parametrize("T", sorted(_REF_PB11))
    def test_pb11_nevins_swain_ht_branch(self, T):
        assert float(sigv_pb11(T)) == pytest.approx(_REF_PB11[T], rel=1e-3)


class TestFitPhysics:
    def test_dt_anchor_nrl(self):
        # NRL formulary: DT <sigma v> at 100 keV ~ 8.54e-16 cm^3/s
        assert float(sigv_dt(100.0)) == pytest.approx(8.54e-22, rel=0.15)

    @pytest.mark.skipif(
        not HAS_JAX,
        reason="skipped in numpy (uses finite differences)",
    )
    def test_dt_peak_location(self):
        # Bosch-Hale DT reactivity peaks near 64-67 keV
        Ts = jnp.linspace(10.0, 100.0, 91)
        peak_T = float(Ts[jnp.argmax(jax.vmap(sigv_dt)(Ts))])
        assert 60.0 < peak_T < 72.0

    def test_advanced_fuels_far_below_dt_at_15kev(self):
        # The silent-wrong-fuel bug this feature fixes: at 15 keV D-He3 is
        # >2 orders of magnitude below DT.
        assert float(sigv_dhe3(15.0)) < 0.01 * float(sigv_dt(15.0))
        assert float(sigv_dd_n(15.0) + sigv_dd_p(15.0)) < 0.05 * float(sigv_dt(15.0))

    @pytest.mark.skipif(
        not HAS_JAX,
        reason="skipped in numpy (uses finite differences)",
    )
    def test_pb11_peak_in_literature_range(self):
        # NS HT-branch peak: broad, ~3.7e-22 m^3/s near 400-500 keV
        Ts = jnp.linspace(50.0, 500.0, 451)
        vs = jax.vmap(sigv_pb11)(Ts)
        assert 3.0e-22 < float(jnp.max(vs)) < 4.5e-22
        assert float(Ts[jnp.argmax(vs)]) > 350.0

    @pytest.mark.skipif(
        not HAS_JAX,
        reason="skipped in numpy (uses finite differences)",
    )
    def test_all_fits_differentiable_and_positive(self):
        for fn, T in [
            (sigv_dt, 15.0),
            (sigv_dhe3, 80.0),
            (sigv_dd_n, 40.0),
            (sigv_dd_p, 40.0),
            (sigv_pb11, 200.0),
        ]:
            g = float(jax.grad(fn)(T))
            assert jnp.isfinite(g)
            assert float(fn(T)) > 0.0

    def test_fits_finite_positive_at_bracket_edges(self):
        for fn, T in [
            (sigv_dt, 1.0),
            (sigv_dt, 100.0),
            (sigv_dd_n, 5.0),
            (sigv_dd_p, 100.0),
            (sigv_dhe3, 20.0),
            (sigv_dhe3, 200.0),
            (sigv_pb11, 50.0),
            (sigv_pb11, 400.0),
        ]:
            v = float(fn(T))
            assert math.isfinite(v)
            assert v > 0.0


_FRACS = dict(
    dd_f_T=0.969,
    dd_f_He3=0.689,
    dhe3_dd_frac=0.131,
    dhe3_f_T=0.5,
    dhe3_f_He3=0.05,
    pb11_f_alpha_n=0.0,
    pb11_f_p_n=0.0,
)


class TestEventEnergies:
    @pytest.mark.parametrize("fuel", [Fuel.DT, Fuel.DD, Fuel.DHE3, Fuel.PB11])
    def test_consistent_with_ash_neutron_split(self, fuel):
        """ash_neutron_split's split must equal the event_energies ratio."""
        E_total, E_neutron = event_energies(fuel, **_FRACS)
        p_ash, p_neutron = ash_neutron_split(100.0, fuel, **_FRACS)
        assert float(p_neutron) == pytest.approx(
            100.0 * float(E_neutron) / float(E_total), rel=1e-6
        )

    def test_dt_event_energy(self):
        E_total, E_neutron = event_energies(Fuel.DT, **_FRACS)
        assert float(E_total) == pytest.approx(17.58, rel=1e-6)
        assert float(E_neutron) == pytest.approx(14.06, rel=1e-6)


class TestMixAlgebra:
    def test_dilution_factors(self):
        assert n_i_over_n_e(Fuel.DT, 1.0, 0.15) == pytest.approx(1.0)
        assert n_i_over_n_e(Fuel.DD, 1.0, 0.15) == pytest.approx(1.0)
        # D-He3 at r=1: (1+r)/(1+2r) = 2/3
        assert n_i_over_n_e(Fuel.DHE3, 1.0, 0.15) == pytest.approx(2.0 / 3.0)
        # p-B11 at r=0.15: (1+r)/(1+5r) = 1.15/1.75
        assert n_i_over_n_e(Fuel.PB11, 1.0, 0.15) == pytest.approx(1.15 / 1.75)

    def test_z_eff_fuel(self):
        assert z_eff_fuel(Fuel.DT, 1.0, 0.15) == pytest.approx(1.0)
        assert z_eff_fuel(Fuel.DD, 1.0, 0.15) == pytest.approx(1.0)
        # D-He3 at r=1: (1+4r)/(1+2r) = 5/3
        assert z_eff_fuel(Fuel.DHE3, 1.0, 0.15) == pytest.approx(5.0 / 3.0)
        # p-B11 at r=0.15: (1+25r)/(1+5r) = 4.75/1.75
        assert z_eff_fuel(Fuel.PB11, 0.5, 0.15) == pytest.approx(4.75 / 1.75)


class TestFusionPowerDensity:
    _KW = dict(
        dhe3_fuel_ratio=1.0,
        pb11_fuel_ratio=0.15,
        dhe3_dd_frac_pin=None,
        dd_f_T=_FRACS["dd_f_T"],
        dd_f_He3=_FRACS["dd_f_He3"],
        dhe3_f_T=_FRACS["dhe3_f_T"],
        dhe3_f_He3=_FRACS["dhe3_f_He3"],
        pb11_f_alpha_n=_FRACS["pb11_f_alpha_n"],
        pb11_f_p_n=_FRACS["pb11_f_p_n"],
    )

    def test_dt_matches_legacy_formula(self):
        """DT must reproduce compute_fusion_power exactly (bit-identical path)."""
        from costingfe.layers.tokamak import compute_fusion_power

        p, frac = fusion_power(Fuel.DT, 1.0e20, 13.0, 830.0, **self._KW)
        assert float(p) == float(compute_fusion_power(1.0e20, 13.0, 830.0))
        assert float(frac) == 0.0

    def test_dhe3_derives_side_channel_fraction(self):
        p, frac = fusion_power(Fuel.DHE3, 1.0e20, 70.0, 830.0, **self._KW)
        # r=1: n_D = n_e/3, n_He3 = n_e/3
        n_D = 1.0e20 / 3.0
        R_dhe3 = n_D * n_D * float(sigv_dhe3(70.0))
        R_dd = 0.5 * n_D * n_D * float(sigv_dd_n(70.0) + sigv_dd_p(70.0))
        assert float(frac) == pytest.approx(R_dd / (R_dd + R_dhe3), rel=1e-5)
        assert float(p) > 0.0

    def test_dhe3_pin_overrides_derived(self):
        kw = dict(self._KW)
        kw["dhe3_dd_frac_pin"] = 0.25
        _, frac = fusion_power(Fuel.DHE3, 1.0e20, 70.0, 830.0, **kw)
        assert float(frac) == pytest.approx(0.25)

    def test_pb11_dilution_suppresses_power(self):
        p, _ = fusion_power(Fuel.PB11, 1.0e20, 300.0, 830.0, **self._KW)
        # Undiluted n_e^2/4 estimate must overshoot the quasineutral result:
        # n_p*n_B = 0.15*n_e^2/(1.75^2) ~ 0.049 n_e^2 < 0.25 n_e^2
        E_total, _ = event_energies(Fuel.PB11, **_FRACS)
        undiluted = (
            0.25
            * 1.0e20
            * float(sigv_pb11(300.0))
            * 1.0e20
            * float(E_total)
            * 1.602176634e-13
            * 830.0
            * 1e-6
        )
        assert float(p) < 0.3 * undiluted
        assert float(p) > 0.0

    def test_dd_rate_and_zero_fraction(self):
        from costingfe.layers.physics import event_energies

        p, frac = fusion_power(Fuel.DD, 1.0e20, 40.0, 830.0, **self._KW)
        E_total, _ = event_energies(Fuel.DD, **_FRACS)
        expected = (
            0.5
            * 1.0e20
            * float(sigv_dd_n(40.0) + sigv_dd_p(40.0))
            * 1.0e20
            * float(E_total)
            * 1.602176634e-13
            * 830.0
            * 1e-6
        )
        assert float(p) == pytest.approx(expected, rel=1e-5)
        assert float(frac) == 0.0

    @pytest.mark.skipif(
        not HAS_JAX,
        reason="skipped in numpy (uses finite differences)",
    )
    def test_dhe3_dispatch_differentiable(self):
        import jax

        def p_of_T(T):
            p, _ = fusion_power(Fuel.DHE3, 1.0e20, T, 830.0, **self._KW)
            return p

        g = float(jax.grad(p_of_T)(70.0))
        assert jnp.isfinite(g)
        assert g > 0.0  # below the D-He3 peak, power rises with T_i

    @pytest.mark.skipif(
        not HAS_JAX,
        reason="skipped in numpy (uses finite differences)",
    )
    def test_jit_matches_eager_all_fuels(self):
        import jax

        for fuel in [Fuel.DT, Fuel.DD, Fuel.DHE3, Fuel.PB11]:
            T = {Fuel.DT: 15.0, Fuel.DD: 40.0, Fuel.DHE3: 70.0, Fuel.PB11: 300.0}[fuel]

            def p_of(n_e, T_i, V):
                p, _ = fusion_power(fuel, n_e, T_i, V, **self._KW)
                return p

            eager = float(p_of(1.0e20, T, 830.0))
            jitted = float(jax.jit(p_of)(1.0e20, T, 830.0))
            assert jnp.isfinite(jitted)
            assert jitted == pytest.approx(eager, rel=1e-4)
