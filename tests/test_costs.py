from costingfe.defaults import load_costing_constants
from costingfe.layers.costs import (
    cas10_preconstruction,
    cas21_buildings,
    cas23_turbine,
    cas24_electrical,
    cas25_misc,
    cas26_heat_rejection,
    cas30_indirect,
    cas70_om,
    cas80_fuel,
    cas90_financial,
)
from costingfe.types import Fuel

CC = load_costing_constants()


# ---- CAS30 indirect service cost tests ----


def test_cas30_is_fraction_of_direct_cost():
    """CAS30 = indirect_fraction * CAS20 * (t_con / t_ref)."""
    cas20 = 4000.0
    t_con = 6.0
    result = cas30_indirect(CC, cas20, t_con)
    # Default: 0.20 * 4000 * (6/6) = 800
    assert abs(result - 800.0) < 0.01


def test_cas30_scales_with_construction_time():
    """Longer construction should increase indirect costs."""
    cas20 = 4000.0
    c30_6yr = cas30_indirect(CC, cas20, 6.0)
    c30_9yr = cas30_indirect(CC, cas20, 9.0)
    # 9/6 = 1.5x
    assert abs(c30_9yr / c30_6yr - 1.5) < 0.01


def test_cas30_scales_with_direct_cost():
    """Higher direct cost should increase indirect costs."""
    c30_low = cas30_indirect(CC, 2000.0, 6.0)
    c30_high = cas30_indirect(CC, 4000.0, 6.0)
    assert abs(c30_high / c30_low - 2.0) < 0.01


def test_cas30_custom_fraction():
    """indirect_fraction override via costing constants."""
    cc_custom = CC.replace(indirect_fraction=0.30)
    result = cas30_indirect(cc_custom, 4000.0, 6.0)
    # 0.30 * 4000 * (6/6) = 1200
    assert abs(result - 1200.0) < 0.01


def test_cas30_custom_reference_time():
    """reference_construction_time override changes scaling."""
    cc_custom = CC.replace(reference_construction_time=8.0)
    result = cas30_indirect(cc_custom, 4000.0, 8.0)
    # 0.20 * 4000 * (8/8) = 800
    assert abs(result - 800.0) < 0.01
    result_short = cas30_indirect(cc_custom, 4000.0, 4.0)
    # 0.20 * 4000 * (4/8) = 400
    assert abs(result_short - 400.0) < 0.01


def test_cas30_1gwe_reference_case():
    """At 1 GWe with ~$4B direct cost, CAS30 should be ~$800M (20% of TDC)."""
    result = cas30_indirect(CC, 4000.0, 6.0)
    assert 700 < result < 900


def test_cas10_dt_licensing():
    """DT licensing should be $5M."""
    cost = cas10_preconstruction(CC, p_net=1000.0, n_mod=1, fuel=Fuel.DT, noak=True)
    assert cost > 0


def test_cas10_pb11_cheaper_licensing():
    """pB11 licensing should be cheaper than DT."""
    cost_dt = cas10_preconstruction(CC, p_net=1000.0, n_mod=1, fuel=Fuel.DT, noak=True)
    cost_pb11 = cas10_preconstruction(
        CC, p_net=1000.0, n_mod=1, fuel=Fuel.PB11, noak=True
    )
    assert cost_pb11 < cost_dt


def test_cas21_scales_with_power():
    """Building costs should scale with gross electric power."""
    cost_low = cas21_buildings(CC, p_et=500.0, fuel=Fuel.DT, noak=True)
    cost_high = cas21_buildings(CC, p_et=1000.0, fuel=Fuel.DT, noak=True)
    assert cost_high > cost_low


def test_cas23_to_26_scale_with_power():
    """BOP equipment scales with gross electric power."""
    c23 = cas23_turbine(CC, p_et=1000.0, n_mod=1)
    c24 = cas24_electrical(CC, p_et=1000.0, n_mod=1)
    c25 = cas25_misc(CC, p_et=1000.0, n_mod=1)
    c26 = cas26_heat_rejection(CC, p_et=1000.0, n_mod=1)
    for c in [c23, c24, c25, c26]:
        assert c > 0


def test_cas90_annualizes_capital():
    """CAS90 should be CRF * total capital."""
    c90 = cas90_financial(
        CC,
        total_capital=5000.0,
        interest_rate=0.07,
        plant_lifetime=30,
        construction_time=6,
        fuel=Fuel.DT,
        noak=True,
    )
    assert c90 > 0
    assert c90 < 5000  # annualized should be less than total


def test_lcoe_end_to_end_sanity():
    """Full cost stack should produce LCOE in reasonable range."""
    from costingfe.layers.economics import compute_lcoe

    # Minimal cas22_detail for cas70_om (only replaceable accounts needed)
    cas22_detail = {"C220101": 100.0, "C220108": 30.0}

    c90 = cas90_financial(CC, 5000.0, 0.07, 30, 6, Fuel.DT, True)
    c70, c71, c72 = cas70_om(
        CC,
        cas22_detail=cas22_detail,
        replaceable_accounts=CC.replaceable_accounts,
        n_mod=1,
        p_net=1000.0,
        availability=0.85,
        inflation_rate=0.02,
        interest_rate=0.07,
        lifetime_yr=30,
        core_lifetime=CC.core_lifetime(Fuel.DT),
        construction_time=6,
        fuel=Fuel.DT,
        noak=True,
    )
    c80 = cas80_fuel(
        CC,
        p_fus=2600.0,
        n_mod=1,
        availability=0.85,
        inflation_rate=0.02,
        construction_time=6,
        fuel=Fuel.DT,
        noak=True,
    )
    lcoe = compute_lcoe(c90, c70, c80, p_net=1000.0, n_mod=1, availability=0.85)
    assert 10 < lcoe < 500, f"LCOE {lcoe} $/MWh out of expected range"


# ---- CAS70 sub-account tests (CAS71 + CAS72) ----


def test_cas70_returns_tuple():
    """cas70_om should return (total, cas71, cas72) tuple."""
    cas22_detail = {"C220101": 100.0, "C220108": 30.0}
    result = cas70_om(
        CC,
        cas22_detail=cas22_detail,
        replaceable_accounts=CC.replaceable_accounts,
        n_mod=1,
        p_net=1000.0,
        availability=0.85,
        inflation_rate=0.02,
        interest_rate=0.07,
        lifetime_yr=30,
        core_lifetime=CC.core_lifetime(Fuel.DT),
        construction_time=6,
        fuel=Fuel.DT,
        noak=True,
    )
    assert isinstance(result, tuple)
    assert len(result) == 3
    total, cas71, cas72 = result
    assert abs(total - (cas71 + cas72)) < 0.001


def test_cas72_dt_30yr_has_replacements():
    """DT with 5 FPY core life, 30yr plant, 85% avail -> replacements."""
    cas22_detail = {"C220101": 100.0, "C220108": 30.0}
    _, _, cas72 = cas70_om(
        CC,
        cas22_detail=cas22_detail,
        replaceable_accounts=CC.replaceable_accounts,
        n_mod=1,
        p_net=1000.0,
        availability=0.85,
        inflation_rate=0.02,
        interest_rate=0.07,
        lifetime_yr=30,
        core_lifetime=CC.core_lifetime(Fuel.DT),
        construction_time=6,
        fuel=Fuel.DT,
        noak=True,
    )
    assert cas72 > 0, "DT 30yr should have replacement costs"


def test_cas72_pb11_30yr_no_replacements():
    """pB11 with 50 FPY core life, 30yr plant -> no replacements needed."""
    cas22_detail = {"C220101": 20.0, "C220108": 10.0}
    _, _, cas72 = cas70_om(
        CC,
        cas22_detail=cas22_detail,
        replaceable_accounts=CC.replaceable_accounts,
        n_mod=1,
        p_net=1000.0,
        availability=0.85,
        inflation_rate=0.02,
        interest_rate=0.07,
        lifetime_yr=30,
        core_lifetime=CC.core_lifetime(Fuel.PB11),
        construction_time=6,
        fuel=Fuel.PB11,
        noak=True,
    )
    assert cas72 == 0.0, "pB11 30yr should have zero replacement costs"


def test_cas72_longer_lifetime_more_cost():
    """Longer plant life should increase replacement costs."""
    cas22_detail = {"C220101": 100.0, "C220108": 30.0}
    kwargs = dict(
        cas22_detail=cas22_detail,
        replaceable_accounts=CC.replaceable_accounts,
        n_mod=1,
        p_net=1000.0,
        availability=0.85,
        inflation_rate=0.02,
        interest_rate=0.07,
        core_lifetime=CC.core_lifetime(Fuel.DT),
        construction_time=6,
        fuel=Fuel.DT,
        noak=True,
    )
    _, _, cas72_20 = cas70_om(CC, lifetime_yr=20, **kwargs)
    _, _, cas72_40 = cas70_om(CC, lifetime_yr=40, **kwargs)
    assert cas72_40 > cas72_20


def test_cas72_uses_overridden_component_costs():
    """CAS72 should use overridden component values from cas22_detail."""
    base_detail = {"C220101": 100.0, "C220108": 30.0}
    expensive_detail = {"C220101": 500.0, "C220108": 30.0}
    kwargs = dict(
        replaceable_accounts=CC.replaceable_accounts,
        n_mod=1,
        p_net=1000.0,
        availability=0.85,
        inflation_rate=0.02,
        interest_rate=0.07,
        lifetime_yr=30,
        core_lifetime=CC.core_lifetime(Fuel.DT),
        construction_time=6,
        fuel=Fuel.DT,
        noak=True,
    )
    _, _, cas72_base = cas70_om(CC, cas22_detail=base_detail, **kwargs)
    _, _, cas72_exp = cas70_om(CC, cas22_detail=expensive_detail, **kwargs)
    assert cas72_exp > cas72_base


# ---- CAS80 fuel-specific tests ----

_CAS80_KWARGS = dict(
    p_fus=2600.0,
    n_mod=1,
    availability=0.85,
    inflation_rate=0.02,
    construction_time=6,
    noak=True,
)


def test_cas80_scales_with_p_fus():
    """Higher fusion power should produce higher fuel cost."""
    low = cas80_fuel(
        CC,
        p_fus=1000.0,
        n_mod=1,
        availability=0.85,
        inflation_rate=0.02,
        construction_time=6,
        fuel=Fuel.DT,
        noak=True,
    )
    high = cas80_fuel(
        CC,
        p_fus=3000.0,
        n_mod=1,
        availability=0.85,
        inflation_rate=0.02,
        construction_time=6,
        fuel=Fuel.DT,
        noak=True,
    )
    assert high > low


def test_cas80_linear_in_p_fus():
    """Doubling fusion power should double fuel cost (linear relationship)."""
    kwargs = {k: v for k, v in _CAS80_KWARGS.items() if k != "p_fus"}
    for fuel in Fuel:
        c1 = cas80_fuel(CC, p_fus=1000.0, fuel=fuel, **kwargs)
        c2 = cas80_fuel(CC, p_fus=2000.0, fuel=fuel, **kwargs)
        ratio = float(c2) / float(c1)
        assert abs(ratio - 2.0) < 1e-10, f"{fuel}: expected ratio 2.0, got {ratio}"


def test_cas80_dt_includes_li6():
    """DT cost should include Li-6 (higher than D-only)."""
    dt_cost = cas80_fuel(CC, fuel=Fuel.DT, **_CAS80_KWARGS)
    # D-only cost: temporarily zero out Li-6
    cc_no_li6 = CC.replace(u_li6=0.0)
    d_only = cas80_fuel(cc_no_li6, fuel=Fuel.DT, **_CAS80_KWARGS)
    assert dt_cost > d_only, "DT should include Li-6 cost beyond deuterium"


def test_cas80_dd_deuterium_only():
    """DD fuel cost should depend only on deuterium price."""
    dd_cost = cas80_fuel(CC, fuel=Fuel.DD, **_CAS80_KWARGS)
    assert dd_cost > 0
    # Changing Li-6 price should not affect DD
    cc_expensive_li6 = CC.replace(u_li6=1e6)
    dd_cost2 = cas80_fuel(cc_expensive_li6, fuel=Fuel.DD, **_CAS80_KWARGS)
    assert abs(dd_cost - dd_cost2) < 1e-10


def test_cas80_pb11_b11_dominates():
    """pB11 feedstock should be more expensive than DT feedstock."""
    dt_cost = cas80_fuel(CC, fuel=Fuel.DT, **_CAS80_KWARGS)
    pb11_cost = cas80_fuel(CC, fuel=Fuel.PB11, **_CAS80_KWARGS)
    assert pb11_cost > dt_cost


def test_cas80_dhe3_most_expensive():
    """DHe3 should have highest fuel cost (He-3 at $2M/kg)."""
    dhe3_cost = cas80_fuel(CC, fuel=Fuel.DHE3, **_CAS80_KWARGS)
    for other_fuel in [Fuel.DT, Fuel.DD, Fuel.PB11]:
        other = cas80_fuel(CC, fuel=other_fuel, **_CAS80_KWARGS)
        assert dhe3_cost > other, f"DHe3 should be more expensive than {other_fuel}"


def test_cas80_fuel_cost_order():
    """Fuel cost ordering: DHe3 > pB11 > DD > DT."""
    costs = {f: float(cas80_fuel(CC, fuel=f, **_CAS80_KWARGS)) for f in Fuel}
    assert costs[Fuel.DHE3] > costs[Fuel.PB11] > costs[Fuel.DD] > costs[Fuel.DT]
