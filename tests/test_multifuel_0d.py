"""Multi-fuel 0D tokamak model tests."""

import pytest

from costingfe.layers.tokamak import (
    _T_BRACKET_DEFAULTS,
    tokamak_0d_forward,
    tokamak_0d_inverse,
)
from costingfe.types import Fuel

_GEOM = dict(R=3.3, a=1.13, kappa=1.84, B=9.2, q95=3.05, f_GW=0.85)
_FRACS = dict(
    dd_f_T=0.969,
    dd_f_He3=0.689,
    dhe3_f_T=0.5,
    dhe3_f_He3=0.05,
    pb11_f_alpha_n=0.0,
    pb11_f_p_n=0.0,
)
_NEW = dict(T_i_over_T_e=1.0, dhe3_fuel_ratio=1.0, pb11_fuel_ratio=0.15)


def _forward(fuel, T_e, dhe3_dd_frac=None, **kw):
    args = {**_FRACS, **_NEW, **kw}
    return tokamak_0d_forward(
        **_GEOM,
        T_e=T_e,
        p_input=20.0,
        fuel=fuel,
        dhe3_dd_frac=dhe3_dd_frac,
        **args,
    )


class TestMultiFuelForward:
    def test_dhe3_power_far_below_dt_at_dt_temperature(self):
        ps_dt = _forward(Fuel.DT, 15.0)
        ps_dhe3 = _forward(Fuel.DHE3, 15.0)
        assert float(ps_dhe3.p_fus) < 0.02 * float(ps_dt.p_fus)

    def test_dhe3_derived_fraction_populated(self):
        ps = _forward(Fuel.DHE3, 70.0)
        assert 0.0 < float(ps.dhe3_dd_frac_eff) < 1.0

    def test_dhe3_pin_respected(self):
        ps = _forward(Fuel.DHE3, 70.0, dhe3_dd_frac=0.25)
        assert float(ps.dhe3_dd_frac_eff) == pytest.approx(0.25)

    def test_hot_ion_mode_raises_power_and_beta(self):
        cold = _forward(Fuel.DHE3, 70.0)
        hot = _forward(Fuel.DHE3, 70.0, T_i_over_T_e=1.5)
        assert float(hot.p_fus) > float(cold.p_fus)
        assert float(hot.beta_N) > float(cold.beta_N)

    def test_pb11_solves_at_high_temperature(self):
        ps = _forward(Fuel.PB11, 300.0)
        assert float(ps.p_fus) > 0.0


class TestInverseBrackets:
    def test_bracket_table(self):
        assert _T_BRACKET_DEFAULTS[Fuel.DT] == (1.0, 100.0)
        assert _T_BRACKET_DEFAULTS[Fuel.DD] == (5.0, 100.0)
        assert _T_BRACKET_DEFAULTS[Fuel.DHE3] == (20.0, 200.0)
        assert _T_BRACKET_DEFAULTS[Fuel.PB11] == (50.0, 400.0)

    def test_dhe3_inverse_lands_in_bracket(self):
        ps, pt = tokamak_0d_inverse(
            p_net_target=50.0,
            **_GEOM,
            fuel=Fuel.DHE3,
            dhe3_dd_frac=0.131,
            dhe3_dd_frac_pin=None,
            **_FRACS,
            **_NEW,
        )
        assert 20.0 <= float(ps.T_e) <= 200.0
        assert float(ps.p_fus) > 0.0
