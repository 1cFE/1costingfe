import pytest

from costingfe.defaults import MAGNET_TABLE, get_magnet_properties
from costingfe.layers.tokamak import (
    SizingInfeasible,
    b0_from_radial_build,
    net_electric_at_R0,
    tokamak_size_from_power,
)
from costingfe.model import CostModel
from costingfe.types import ConfinementConcept, Fuel


def test_magnet_table_has_expected_materials():
    for key in ("rebco_hts", "nb3sn", "nbti", "copper"):
        assert key in MAGNET_TABLE
    for sc_key in ("rebco_hts", "nb3sn", "nbti"):
        assert MAGNET_TABLE[sc_key].recirc_power_factor == 0.0


def test_get_magnet_properties_rebco():
    props = get_magnet_properties("rebco_hts")
    assert props.b_max == pytest.approx(23.0)
    assert props.recirc_power_factor == 0.0
    assert props.cryo_temp_k == pytest.approx(20.0)


def test_get_magnet_properties_copper_has_recirc():
    props = get_magnet_properties("copper")
    assert props.recirc_power_factor > 0.0


def test_get_magnet_properties_unknown_raises():
    with pytest.raises(KeyError):
        get_magnet_properties("unobtanium")


def test_b0_below_bmax_and_grows_with_size():
    # Same magnet, two machine sizes; the larger machine keeps more of B_max.
    thick = dict(blanket_t=0.8, ht_shield_t=0.2, structure_t=0.2, vessel_t=0.2)
    b_small = b0_from_radial_build(R0=3.0, a=1.0, b_max=23.0, **thick)
    b_large = b0_from_radial_build(R0=6.0, a=2.0, b_max=23.0, **thick)
    assert 0.0 < b_small < 23.0
    assert b_small < b_large < 23.0


def test_b0_formula():
    # B0 = B_max * (R0 - a - sum_thick) / R0
    b = b0_from_radial_build(
        R0=4.0,
        a=1.0,
        b_max=20.0,
        blanket_t=0.5,
        ht_shield_t=0.2,
        structure_t=0.2,
        vessel_t=0.1,
    )
    expected = 20.0 * (4.0 - 1.0 - 1.0) / 4.0
    assert b == pytest.approx(expected)


def _base_sizing_params():
    return dict(
        aspect_ratio=3.0,
        elon=1.85,
        q95=3.5,
        f_GW=0.85,
        b_max=23.0,
        beta_N_max=3.5,
        H_factor=1.0,
        T_min=5.0,
        T_max=60.0,
        blanket_t=0.8,
        ht_shield_t=0.2,
        structure_t=0.2,
        vessel_t=0.2,
        p_input=50.0,
        mn=1.1,
        eta_th=0.45,
        eta_p=0.5,
        eta_pin=0.7,
        eta_de=0.85,
        f_sub=0.03,
        f_dec=0.0,
        p_coils=2.0,
        p_cool=13.7,
        p_pump=1.0,
        p_trit=10.0,
        p_house=4.0,
        p_cryo=0.5,
        Z_eff=1.5,
        M_ion=2.5,
        lambda_q=0.002,
        R_w=0.6,
        wall_material="W",
        T_edge=0.05,
        tau_ratio=3.0,
        recirc_power_factor=0.0,
        dd_f_T=0.969,
        dd_f_He3=0.689,
        dhe3_dd_frac=0.131,
        dhe3_f_T=0.5,
        dhe3_f_He3=0.5,
        pb11_f_alpha_n=0.0,
        pb11_f_p_n=0.0,
    )


def test_size_hits_target():
    from costingfe.types import Fuel

    p = _base_sizing_params()
    p.update(R0_min=1.0, R0_max=12.0, net_electric_mw=500.0)
    result = tokamak_size_from_power(p, Fuel.DT)
    pn = net_electric_at_R0(result.R0, p, Fuel.DT)
    assert pn == pytest.approx(500.0, rel=0.02)
    assert result.a == pytest.approx(result.R0 / p["aspect_ratio"])
    assert 0.0 < result.B0 < p["b_max"]


def test_size_scales_with_power():
    from costingfe.types import Fuel

    p = _base_sizing_params()
    p.update(R0_min=1.0, R0_max=15.0)
    r1 = tokamak_size_from_power({**p, "net_electric_mw": 250.0}, Fuel.DT)
    r2 = tokamak_size_from_power({**p, "net_electric_mw": 2000.0}, Fuel.DT)
    assert r2.R0 > r1.R0  # bigger machine for more power
    # 8x power grows R0 by well under the 2x that fixed-operating-point P ~ R0^3
    # would imply: on-axis field rises with R0 (fixed-meter blanket is a smaller
    # fraction of a larger machine), which relaxes the beta limit and lets the
    # bigger machine run hotter and more reactive. Power therefore scales steeper
    # than R0^3, so R0 grows sub-cubically with power (strong economy of scale).
    assert 1.1 < (r2.R0 / r1.R0) < 2.0


def test_infeasible_raises():
    from costingfe.types import Fuel

    p = _base_sizing_params()
    p.update(R0_min=1.0, R0_max=3.0, net_electric_mw=5000.0)
    with pytest.raises(SizingInfeasible):
        tokamak_size_from_power(p, Fuel.DT)


def test_net_power_increases_with_R0():
    from costingfe.types import Fuel

    p = _base_sizing_params()
    pn_small = net_electric_at_R0(3.0, p, Fuel.DT)
    pn_large = net_electric_at_R0(5.0, p, Fuel.DT)
    assert pn_large > pn_small


def test_net_power_positive_for_reactor_scale():
    from costingfe.types import Fuel

    p = _base_sizing_params()
    assert net_electric_at_R0(4.5, p, Fuel.DT) > 0.0


def test_missing_disruption_key_errors_not_silent():
    # When the 0D model is active, a missing disruption input must error rather
    # than silently fall back to a magic default.
    m = CostModel(ConfinementConcept.TOKAMAK, Fuel.DT)
    m._eng_defaults = dict(m._eng_defaults)
    m._eng_defaults.pop("disruption_damage")
    with pytest.raises(KeyError):
        m.forward(
            net_electric_mw=400.0,
            availability=0.85,
            lifetime_yr=30.0,
            use_0d_model=True,
        )


def test_recirc_reduces_net_power():
    from costingfe.types import Fuel

    p_zero = _base_sizing_params()
    p_zero["recirc_power_factor"] = 0.0

    p_recirc = _base_sizing_params()
    p_recirc["recirc_power_factor"] = 1.0e-3

    pn_zero = net_electric_at_R0(4.5, p_zero, Fuel.DT)
    pn_recirc = net_electric_at_R0(4.5, p_recirc, Fuel.DT)
    # Resistive coils draw continuous power, so net electric must drop.
    assert pn_recirc < pn_zero


def test_sizing_overrides_accepted():
    m = CostModel(ConfinementConcept.TOKAMAK, Fuel.DT)
    # Must not raise "unknown parameter": these are sizing knobs. (size_from_power
    # is not yet gated, so forward() runs the normal path; we only assert that
    # override validation accepts the keys.)
    m.forward(
        net_electric_mw=500.0,
        availability=0.85,
        lifetime_yr=30.0,
        size_from_power=True,
        aspect_ratio=3.1,
        beta_N_max=3.5,
        H_factor=1.0,
    )


def test_sizing_runs_end_to_end_and_sets_geometry():
    m = CostModel(ConfinementConcept.TOKAMAK, Fuel.DT)
    r = m.forward(
        net_electric_mw=500.0,
        availability=0.85,
        lifetime_yr=30.0,
        size_from_power=True,
        aspect_ratio=3.1,
        beta_N_max=3.5,
        coil_material="rebco_hts",
    )
    assert r.costs.lcoe > 0.0
    assert m._plasma_state is not None


def test_pinning_R0_in_sizing_mode_raises():
    m = CostModel(ConfinementConcept.TOKAMAK, Fuel.DT)
    with pytest.raises(ValueError, match="cannot be pinned"):
        m.forward(
            net_electric_mw=500.0,
            availability=0.85,
            lifetime_yr=30.0,
            size_from_power=True,
            R0=3.0,
        )


def test_coil_cost_scales_with_power():
    # The core issue: C220103 (coils) must move with power.
    m = CostModel(ConfinementConcept.TOKAMAK, Fuel.DT)
    common = dict(
        availability=0.85,
        lifetime_yr=30.0,
        size_from_power=True,
        aspect_ratio=3.1,
        beta_N_max=3.5,
        coil_material="rebco_hts",
    )
    lo = m.forward(net_electric_mw=200.0, **common)
    hi = m.forward(net_electric_mw=1500.0, **common)
    c_lo = lo.cas22_detail["C220103"]
    c_hi = hi.cas22_detail["C220103"]
    assert c_hi > c_lo * 1.2


def test_optimize_returns_valid_fgw_and_lcoe():
    m = CostModel(ConfinementConcept.TOKAMAK, Fuel.DT)
    r = m.forward(
        net_electric_mw=500.0,
        availability=0.85,
        lifetime_yr=30.0,
        size_from_power=True,
        optimize_lcoe=True,
        aspect_ratio=3.1,
        beta_N_max=3.5,
        coil_material="rebco_hts",
        disruption_rate_base=1.0,
        disruption_damage=0.1,
    )
    assert r.costs.lcoe > 0.0
    assert 0.0 < m._sizing_fgw <= 1.0


def test_optimize_implies_size_from_power():
    # Passing optimize_lcoe without size_from_power must still size the machine,
    # not silently fall back to the fixed-geometry power balance.
    m = CostModel(ConfinementConcept.TOKAMAK, Fuel.DT)
    m.forward(
        net_electric_mw=500.0,
        availability=0.85,
        lifetime_yr=30.0,
        optimize_lcoe=True,
        aspect_ratio=3.1,
        beta_N_max=3.5,
        coil_material="rebco_hts",
        disruption_rate_base=1.0,
        disruption_damage=0.1,
    )
    assert m._last_R0 > 0.0  # sizing actually happened
    assert 0.0 < m._sizing_fgw <= 1.0


def test_optimize_lcoe_no_worse_than_default_fgw():
    m = CostModel(ConfinementConcept.TOKAMAK, Fuel.DT)
    common = dict(
        net_electric_mw=500.0,
        availability=0.85,
        lifetime_yr=30.0,
        size_from_power=True,
        aspect_ratio=3.1,
        beta_N_max=3.5,
        coil_material="rebco_hts",
        disruption_rate_base=1.0,
        disruption_damage=0.1,
    )
    fixed = m.forward(f_GW=0.85, **common)
    opt = m.forward(optimize_lcoe=True, **common)
    assert opt.costs.lcoe <= fixed.costs.lcoe + 1e-6


def test_backward_compat_sizing_off_unchanged():
    # With size_from_power absent/false, the default tokamak path is unchanged.
    m = CostModel(ConfinementConcept.TOKAMAK, Fuel.DT)
    r = m.forward(net_electric_mw=500.0, availability=0.85, lifetime_yr=30.0)
    assert r.costs.lcoe > 0.0
    # FREEZE the reference: run this once, read the printed LCOE, and replace
    # REF below with that exact value, so this test guards against drift.
    REF = 124.343605
    if REF is not None:
        assert r.costs.lcoe == pytest.approx(REF, rel=1e-9)


def test_arc_validation_reproduces_size():
    # Inject ARC-like design knobs; solved R0 should be compact (ARC R0 ~3.3 m).
    m = CostModel(ConfinementConcept.TOKAMAK, Fuel.DT)
    m.forward(
        net_electric_mw=270.0,
        availability=0.85,
        lifetime_yr=30.0,
        size_from_power=True,
        coil_material="rebco_hts",
        aspect_ratio=3.0,
        elon=1.85,
        beta_N_max=3.0,
        q95=3.5,
        f_GW=0.85,
        H_factor=1.8,  # ARC operates at high confinement (H ~ 1.8)
    )
    assert 2.5 < m._last_R0 < 4.5  # ARC R0 ~3.3 m, allow modeling spread


def test_magnet_differentiation_rebco_smaller_than_nb3sn():
    m = CostModel(ConfinementConcept.TOKAMAK, Fuel.DT)
    common = dict(
        net_electric_mw=500.0,
        availability=0.85,
        lifetime_yr=30.0,
        size_from_power=True,
        aspect_ratio=3.1,
        beta_N_max=3.5,
    )
    m.forward(coil_material="rebco_hts", **common)
    r_rebco = m._last_R0
    m.forward(coil_material="nb3sn", **common)
    r_nb3sn = m._last_R0
    assert r_rebco < r_nb3sn  # higher field -> smaller machine


def test_scale_mode_grows_R0_with_power():
    m = CostModel(ConfinementConcept.TOKAMAK, Fuel.DT)
    common = dict(
        availability=0.85,
        lifetime_yr=30.0,
        size_from_power=True,
        coil_material="rebco_hts",
        aspect_ratio=3.0,
        beta_N_max=3.0,
    )
    m.forward(net_electric_mw=270.0, **common)
    r_small = m._last_R0
    m.forward(net_electric_mw=1000.0, **common)
    r_large = m._last_R0
    assert r_large > r_small


def test_higher_H_factor_gives_smaller_machine():
    # Better confinement (higher H) needs less aux heating -> less recirculating
    # power -> more net power -> a smaller machine for the same target.
    from costingfe.types import Fuel

    p_lo = _base_sizing_params()
    p_lo.update(R0_min=1.0, R0_max=15.0, net_electric_mw=500.0, H_factor=1.0)
    p_hi = _base_sizing_params()
    p_hi.update(R0_min=1.0, R0_max=15.0, net_electric_mw=500.0, H_factor=1.6)
    r_lo = tokamak_size_from_power(p_lo, Fuel.DT)
    r_hi = tokamak_size_from_power(p_hi, Fuel.DT)
    assert r_hi.R0 < r_lo.R0  # higher confinement -> smaller machine


def test_H_factor_affects_lcoe_end_to_end():
    # H_factor must change the costed result (it is no longer a dead knob).
    m = CostModel(ConfinementConcept.TOKAMAK, Fuel.DT)
    common = dict(
        net_electric_mw=500.0,
        availability=0.85,
        lifetime_yr=30.0,
        size_from_power=True,
        aspect_ratio=3.1,
        beta_N_max=3.5,
        coil_material="rebco_hts",
    )
    r1 = m.forward(H_factor=1.0, **common)
    r1_R0 = m._last_R0
    r2 = m.forward(H_factor=1.6, **common)
    r2_R0 = m._last_R0
    assert r2_R0 < r1_R0
    assert r1.costs.lcoe != r2.costs.lcoe
