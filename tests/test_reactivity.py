"""Tests for the fusion reactivity fits and fuel-mix algebra."""

import jax
import jax.numpy as jnp
import pytest

from costingfe.layers.reactivity import (
    sigv_dd_n,
    sigv_dd_p,
    sigv_dhe3,
    sigv_dt,
    sigv_pb11,
)

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

    def test_pb11_peak_in_literature_range(self):
        # NS HT-branch peak: broad, ~3.7e-22 m^3/s near 400-500 keV
        Ts = jnp.linspace(50.0, 500.0, 451)
        vs = jax.vmap(sigv_pb11)(Ts)
        assert 3.0e-22 < float(jnp.max(vs)) < 4.5e-22
        assert float(Ts[jnp.argmax(vs)]) > 350.0

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
