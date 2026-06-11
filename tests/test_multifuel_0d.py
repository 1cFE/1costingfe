"""Multi-fuel 0D tokamak model tests."""

import pytest

from costingfe.layers.tokamak import (
    _T_BRACKET_DEFAULTS,
    tokamak_0d_forward,
    tokamak_0d_inverse,
)
from costingfe.model import CostModel
from costingfe.types import ConfinementConcept, Fuel

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


def _forward(fuel, T_e, dhe3_dd_frac_pin=None, **kw):
    args = {**_FRACS, **_NEW, **kw}
    return tokamak_0d_forward(
        **_GEOM,
        T_e=T_e,
        p_input=20.0,
        fuel=fuel,
        dhe3_dd_frac_pin=dhe3_dd_frac_pin,
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
        ps = _forward(Fuel.DHE3, 70.0, dhe3_dd_frac_pin=0.25)
        assert float(ps.dhe3_dd_frac_eff) == pytest.approx(0.25)

    def test_hot_ion_mode_raises_power_and_beta(self):
        cold = _forward(Fuel.DHE3, 70.0)
        hot = _forward(Fuel.DHE3, 70.0, T_i_over_T_e=1.5)
        assert float(hot.p_fus) > float(cold.p_fus)
        assert float(hot.beta_N) > float(cold.beta_N)

    def test_pb11_solves_at_high_temperature(self):
        ps = _forward(Fuel.PB11, 300.0)
        assert float(ps.p_fus) > 0.0

    def test_dhe3_dilution_lowers_beta_by_closed_form(self):
        # Same geometry and f_GW -> same I_p, n_GW, n_e for both fuels, so
        # beta_N ratio reduces to the pressure ratio
        # (T_e + n_i_frac*T_i)/(2*T_e) with n_i_frac = (1+r)/(1+2r) = 2/3 at r=1.
        ps_dt = _forward(Fuel.DT, 70.0)
        ps_dhe3 = _forward(Fuel.DHE3, 70.0)
        ratio = float(ps_dhe3.beta_N) / float(ps_dt.beta_N)
        assert ratio == pytest.approx((1.0 + 2.0 / 3.0) / 2.0, rel=1e-5)


class TestInverseBrackets:
    def test_bracket_table(self):
        assert _T_BRACKET_DEFAULTS[Fuel.DT] == (1.0, 100.0)
        assert _T_BRACKET_DEFAULTS[Fuel.DD] == (5.0, 100.0)
        assert _T_BRACKET_DEFAULTS[Fuel.DHE3] == (20.0, 190.0)
        assert _T_BRACKET_DEFAULTS[Fuel.PB11] == (50.0, 400.0)

    def test_dhe3_inverse_hits_net_power_target(self):
        # f_rad_fus proxies the radiation model (the model layer defaults it
        # from cc.f_rad_fus for D-He3/p-B11). Without a proxy the full
        # brems+synchrotron model is radiation-dominated for thermal D-He3 at
        # any temperature and no positive-net solution exists.
        ps, pt = tokamak_0d_inverse(
            p_net_target=50.0,
            **_GEOM,
            fuel=Fuel.DHE3,
            dhe3_dd_frac=0.131,
            dhe3_dd_frac_pin=None,
            f_rad_fus=0.24,
            **_FRACS,
            **_NEW,
        )
        assert float(pt.p_net) == pytest.approx(50.0, rel=0.05)
        assert 20.0 <= float(ps.T_e) <= 190.0


class TestMultiFuelModel:
    _KW = dict(availability=0.85, lifetime_yr=30.0)

    def test_dhe3_sizes_larger_machine_than_dt_at_equal_power(self):
        m_dt = CostModel(ConfinementConcept.TOKAMAK, Fuel.DT)
        m_dt.forward(net_electric_mw=200.0, size_from_power=True, **self._KW)
        m_dhe3 = CostModel(ConfinementConcept.TOKAMAK, Fuel.DHE3)
        m_dhe3.forward(
            net_electric_mw=200.0, size_from_power=True, H_factor=1.8, **self._KW
        )
        assert m_dhe3._last_R0 > m_dt._last_R0

    def test_explicit_dhe3_dd_frac_override_pins_partition(self):
        m = CostModel(ConfinementConcept.TOKAMAK, Fuel.DHE3)
        m.forward(
            net_electric_mw=200.0, use_0d_model=True, dhe3_dd_frac=0.25, **self._KW
        )
        assert float(m._plasma_state.dhe3_dd_frac_eff) == pytest.approx(0.25)

    def test_dhe3_derives_fraction_without_override(self):
        m = CostModel(ConfinementConcept.TOKAMAK, Fuel.DHE3)
        m.forward(net_electric_mw=200.0, use_0d_model=True, **self._KW)
        frac = float(m._plasma_state.dhe3_dd_frac_eff)
        assert 0.0 < frac < 1.0
        assert abs(frac - 0.131) > 1e-3  # not the YAML seed

    def test_fuel_mix_moves_fuel_cost_with_partition(self):
        # The CAS80 fuel bill must follow the derived side-channel fraction,
        # which moves with the He3:D mix ratio. Sizing path: it enforces the
        # beta cap, so the operating point (and its disruption-penalized
        # availability) is physical.
        fracs, costs = [], []
        for ratio in (0.5, 2.0):
            m = CostModel(ConfinementConcept.TOKAMAK, Fuel.DHE3)
            r = m.forward(
                net_electric_mw=200.0,
                size_from_power=True,
                H_factor=1.8,
                dhe3_fuel_ratio=ratio,
                **self._KW,
            )
            fracs.append(float(m._plasma_state.dhe3_dd_frac_eff))
            costs.append(float(r.costs.cas80))
        # More He3 (higher ratio) -> fewer D-D side events
        assert fracs[1] < fracs[0]
        assert costs[0] != pytest.approx(costs[1], rel=1e-3)

    def test_dt_results_unchanged_by_multifuel_knobs(self):
        m = CostModel(ConfinementConcept.TOKAMAK, Fuel.DT)
        base = m.forward(net_electric_mw=200.0, use_0d_model=True, **self._KW)
        knob = m.forward(
            net_electric_mw=200.0,
            use_0d_model=True,
            dhe3_fuel_ratio=3.0,
            pb11_fuel_ratio=0.5,
            **self._KW,
        )
        assert float(base.costs.lcoe) == float(knob.costs.lcoe)

    def test_pb11_sizing_fails_loudly_not_silently(self):
        # Under the current physics (Nevins-Swain reactivity, f_rad_fus=0.83
        # proxy, recirc structure) no tokamak up to R0_max nets positive
        # p-B11 power; the contract is a loud SizingInfeasible, not silent
        # garbage costs.
        from costingfe.layers.tokamak import SizingInfeasible

        m = CostModel(ConfinementConcept.TOKAMAK, Fuel.PB11)
        with pytest.raises(SizingInfeasible):
            m.forward(
                net_electric_mw=100.0,
                size_from_power=True,
                H_factor=2.0,
                **self._KW,
            )
