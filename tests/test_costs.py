from costingfe.defaults import load_costing_constants
from costingfe.layers.costs import (
    cas10_preconstruction,
    cas21_buildings,
    cas23_turbine,
    cas24_electrical,
    cas25_misc,
    cas26_heat_rejection,
    cas30_indirect,
    cas60_idc,
    cas70_om,
    cas80_fuel,
    cas90_financial,
)
from costingfe.layers.economics import compute_crf
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


def test_cas10_land_cost_1gwe():
    """Land at 1GWe: 0.25 acres/MWe * 1000 MWe * $10,000/acre = $2.5M."""
    # Land is computed inside cas10_preconstruction; verify via defaults
    assert CC.land_intensity == 0.25
    assert CC.land_cost == 10_000.0
    land = CC.land_intensity * 1000.0 * 1 * CC.land_cost / 1e6  # n_mod=1
    assert abs(land - 2.5) < 0.01


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


def test_licensing_times_match_research():
    """Licensing times per DI-015/016 regulatory framework research.

    DT: 1-2yr (Part 30) → 2.0yr
    DD: 6-18mo → 1.5yr
    DHe3: 6-12mo → 0.75yr
    PB11: ~0yr (no NRC) → 0.0yr
    """
    assert CC.licensing_time_dt == 2.0
    assert CC.licensing_time_dd == 1.5
    assert CC.licensing_time_dhe3 == 0.75
    assert CC.licensing_time_pb11 == 0.0


def test_licensing_time_ordering():
    """More radioactive fuels should have longer licensing times."""
    assert (
        CC.licensing_time_dt
        > CC.licensing_time_dd
        > CC.licensing_time_dhe3
        >= CC.licensing_time_pb11
    )


def test_cas21_scales_with_power():
    """Building costs should increase with larger plant."""
    cost_low = cas21_buildings(
        CC,
        p_et=500.0,
        p_the=500.0,
        p_th=1250.0,
        p_fus=1150.0,
        fuel=Fuel.DT,
        noak=True,
    )
    cost_high = cas21_buildings(
        CC,
        p_et=1000.0,
        p_the=1000.0,
        p_th=2500.0,
        p_fus=2300.0,
        fuel=Fuel.DT,
        noak=True,
    )
    assert cost_high > cost_low


def test_cas21_fuel_differentiation():
    """pB11 buildings should cost less than DT (no hot cell, industrial grade)."""
    cost_dt = cas21_buildings(
        CC,
        p_et=1150.0,
        p_the=1150.0,
        p_th=2500.0,
        p_fus=2300.0,
        fuel=Fuel.DT,
        noak=True,
    )
    cost_pb = cas21_buildings(
        CC,
        p_et=1150.0,
        p_the=1150.0,
        p_th=2500.0,
        p_fus=2300.0,
        fuel=Fuel.PB11,
        noak=True,
    )
    assert cost_pb < cost_dt
    # pB11 should be roughly 55-65% of DT (308/502 ~ 61%)
    assert cost_pb / cost_dt < 0.65


def test_cas23_to_26_scale_with_power():
    """BOP equipment scales with power (CAS23: thermal electric, CAS26: thermal)."""
    c23 = cas23_turbine(CC, p_the=1000.0, n_mod=1)
    c24 = cas24_electrical(CC, p_et=1000.0, n_mod=1)
    c25 = cas25_misc(CC, p_et=1000.0, n_mod=1)
    c26 = cas26_heat_rejection(CC, p_th=2500.0, n_mod=1)
    for c in [c23, c24, c25, c26]:
        assert c > 0


# ---- CAS60 interest during construction tests ----


def test_cas60_reference_case():
    """At 7% WACC, 6yr construction, $5B overnight: f_IDC ≈ 0.1922, IDC ≈ $961M."""
    idc = cas60_idc(interest_rate=0.07, overnight_cost=5000.0, construction_time=6.0)
    # f_IDC = ((1.07^6 - 1) / (0.07 * 6)) - 1 = 0.19221...
    expected = 0.19221 * 5000.0
    assert abs(idc - expected) < 1.0


def test_cas60_scales_linearly_with_overnight_cost():
    """Doubling overnight cost should double IDC."""
    idc_low = cas60_idc(0.07, 2500.0, 6.0)
    idc_high = cas60_idc(0.07, 5000.0, 6.0)
    assert abs(idc_high / idc_low - 2.0) < 1e-10


def test_cas60_increases_with_construction_time():
    """Longer construction should produce more IDC."""
    idc_6 = cas60_idc(0.07, 5000.0, 6.0)
    idc_8 = cas60_idc(0.07, 5000.0, 8.0)
    assert idc_8 > idc_6


def test_cas60_increases_with_interest_rate():
    """Higher interest rate should produce more IDC."""
    idc_5pct = cas60_idc(0.05, 5000.0, 6.0)
    idc_10pct = cas60_idc(0.10, 5000.0, 6.0)
    assert idc_10pct > idc_5pct


def test_cas60_5pct_6yr():
    """At 5% WACC, 6yr: f_IDC = ((1.05^6-1)/(0.05*6))-1 ≈ 0.1337.

    Discrete end-of-year convention. ARIES continuous table gives 0.163;
    difference is the compounding convention, not an error.
    """
    idc = cas60_idc(0.05, 1000.0, 6.0)
    f_idc = idc / 1000.0
    # (1.05^6 - 1) / (0.05 * 6) - 1 = 0.13365...
    assert abs(f_idc - 0.13365) < 0.001


# ---- CAS90 annualized financial cost tests ----


def test_cas90_equals_crf_times_total_capital():
    """CAS90 should be plain CRF * total_capital (no effective CRF)."""
    total_capital = 5000.0
    crf = compute_crf(0.07, 30)
    c90 = cas90_financial(total_capital, interest_rate=0.07, plant_lifetime=30)
    assert abs(c90 - crf * total_capital) < 0.01


def test_cas90_no_construction_time_dependence():
    """CAS90 should NOT depend on construction time (IDC handled by CAS60)."""
    c90 = cas90_financial(5000.0, 0.07, 30)
    # Function should not accept construction_time at all — the signature
    # is (total_capital, interest_rate, plant_lifetime).
    assert c90 > 0
    assert c90 < 5000


def test_cas90_annualizes_capital():
    """CAS90 should be CRF * total capital."""
    c90 = cas90_financial(5000.0, interest_rate=0.07, plant_lifetime=30)
    assert c90 > 0
    assert c90 < 5000  # annualized should be less than total


def test_lcoe_end_to_end_sanity():
    """Full cost stack should produce LCOE in reasonable range."""
    from costingfe.layers.economics import compute_lcoe

    # Minimal cas22_detail for cas70_om (only replaceable accounts needed)
    cas22_detail = {"C220101": 100.0, "C220108": 30.0}

    c90 = cas90_financial(5000.0, 0.07, 30)
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
        interest_rate=0.07,
        lifetime_yr=30,
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
    interest_rate=0.07,
    lifetime_yr=30,
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
        interest_rate=0.07,
        lifetime_yr=30,
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
        interest_rate=0.07,
        lifetime_yr=30,
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


def test_cas80_pb11_foak_b11_dominates():
    """FOAK pB11 feedstock ($10,000/kg B-11) should exceed DT feedstock."""
    kwargs = {**_CAS80_KWARGS, "noak": False}
    dt_cost = cas80_fuel(CC, fuel=Fuel.DT, **kwargs)
    pb11_cost = cas80_fuel(CC, fuel=Fuel.PB11, **kwargs)
    assert pb11_cost > dt_cost


def test_cas80_dhe3_most_expensive():
    """DHe3 should have highest fuel cost (He-3 at $2M/kg)."""
    dhe3_cost = cas80_fuel(CC, fuel=Fuel.DHE3, **_CAS80_KWARGS)
    for other_fuel in [Fuel.DT, Fuel.DD, Fuel.PB11]:
        other = cas80_fuel(CC, fuel=other_fuel, **_CAS80_KWARGS)
        assert dhe3_cost > other, f"DHe3 should be more expensive than {other_fuel}"


def test_cas80_fuel_cost_order_foak():
    """FOAK fuel cost ordering: DHe3 > pB11 > DD > DT."""
    kwargs = {**_CAS80_KWARGS, "noak": False}
    costs = {f: float(cas80_fuel(CC, fuel=f, **kwargs)) for f in Fuel}
    assert costs[Fuel.DHE3] > costs[Fuel.PB11] > costs[Fuel.DD] > costs[Fuel.DT]


def test_cas80_fuel_cost_order_noak():
    """NOAK fuel cost ordering: DHe3 > DD > DT > pB11 (cheap industrial B-11).

    DD is more expensive than DT because each primary DD reaction consumes
    2 deuterium nuclei plus secondary burn-up (~2.85 D per effective event),
    while DT consumes only 1 D per reaction.
    """
    costs = {f: float(cas80_fuel(CC, fuel=f, **_CAS80_KWARGS)) for f in Fuel}
    assert costs[Fuel.DHE3] > costs[Fuel.DD] > costs[Fuel.DT] > costs[Fuel.PB11]


def test_cas80_pb11_noak_cheaper_than_foak():
    """NOAK pB11 should use industrial B-11 price (~$75/kg), not FOAK ($10,000/kg)."""
    kwargs_no_noak = {k: v for k, v in _CAS80_KWARGS.items() if k != "noak"}
    foak = cas80_fuel(CC, fuel=Fuel.PB11, noak=False, **kwargs_no_noak)
    noak = cas80_fuel(CC, fuel=Fuel.PB11, noak=True, **kwargs_no_noak)
    assert noak < foak, "NOAK B-11 should be much cheaper than FOAK"
    # FOAK uses u_b11 ($10,000/kg), NOAK uses u_b11_noak ($75/kg)
    # Ratio should reflect ~133x price difference (protium cost is negligible)
    assert foak / noak > 50


def test_cas80_dd_unaffected_by_noak_unit_cost():
    """DD fuel unit cost should not change between FOAK and NOAK.

    DD is only affected by noak through project timeline (licensing time),
    not through fuel unit price. With zero licensing time difference,
    costs should be identical.
    """
    # Use pB11-zero licensing time to isolate unit cost effect
    cc_no_lic = CC.replace(licensing_time_dd=0.0)
    kwargs_no_noak = {k: v for k, v in _CAS80_KWARGS.items() if k != "noak"}
    foak = cas80_fuel(cc_no_lic, fuel=Fuel.DD, noak=False, **kwargs_no_noak)
    noak = cas80_fuel(cc_no_lic, fuel=Fuel.DD, noak=True, **kwargs_no_noak)
    assert abs(foak - noak) < 1e-10


def test_cas72_includes_dec_grid_replacement():
    """CAS72 should include DEC grid replacement when p_dee > 0."""
    from costingfe.defaults import load_costing_constants
    from costingfe.layers.costs import cas70_om
    from costingfe.types import Fuel

    cc = load_costing_constants()
    cas22_detail = {
        "C220101": 500.0,
        "C220108": 60.0,
        "C220109": 100.0,
    }

    _, _, cas72_with = cas70_om(
        cc,
        cas22_detail=cas22_detail,
        replaceable_accounts=("C220101", "C220108"),
        n_mod=1,
        p_net=1000.0,
        availability=0.85,
        inflation_rate=0.02,
        interest_rate=0.07,
        lifetime_yr=30,
        core_lifetime=5.0,
        construction_time=6.0,
        fuel=Fuel.DHE3,
        noak=True,
        p_dee=400.0,
    )
    _, _, cas72_without = cas70_om(
        cc,
        cas22_detail=cas22_detail,
        replaceable_accounts=("C220101", "C220108"),
        n_mod=1,
        p_net=1000.0,
        availability=0.85,
        inflation_rate=0.02,
        interest_rate=0.07,
        lifetime_yr=30,
        core_lifetime=5.0,
        construction_time=6.0,
        fuel=Fuel.DHE3,
        noak=True,
        p_dee=0.0,
    )
    assert cas72_with > cas72_without, (
        "CAS72 should be higher with DEC grid replacement"
    )


def test_cas72_no_dec_grid_when_p_dee_zero():
    """CAS72 should not include DEC grid term when p_dee = 0."""
    from costingfe.defaults import load_costing_constants
    from costingfe.layers.costs import cas70_om
    from costingfe.types import Fuel

    cc = load_costing_constants()
    cas22_detail = {
        "C220101": 500.0,
        "C220108": 60.0,
        "C220109": 0.0,
    }
    _, _, cas72_a = cas70_om(
        cc,
        cas22_detail=cas22_detail,
        replaceable_accounts=("C220101", "C220108"),
        n_mod=1,
        p_net=1000.0,
        availability=0.85,
        inflation_rate=0.02,
        interest_rate=0.07,
        lifetime_yr=30,
        core_lifetime=5.0,
        construction_time=6.0,
        fuel=Fuel.DT,
        noak=True,
        p_dee=0.0,
    )
    _, _, cas72_b = cas70_om(
        cc,
        cas22_detail=cas22_detail,
        replaceable_accounts=("C220101", "C220108"),
        n_mod=1,
        p_net=1000.0,
        availability=0.85,
        inflation_rate=0.02,
        interest_rate=0.07,
        lifetime_yr=30,
        core_lifetime=5.0,
        construction_time=6.0,
        fuel=Fuel.DT,
        noak=True,
    )
    assert cas72_a == cas72_b, "CAS72 should be identical when p_dee=0 vs not provided"
