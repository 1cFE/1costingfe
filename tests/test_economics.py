import math

import pytest

from costingfe.layers.economics import (
    compute_crf,
    compute_lcoe,
    levelized_annual_cost,
    levelized_replacement_cost,
)


def test_crf_basic():
    """CRF at 7% for 30 years should be ~0.0806."""
    crf = compute_crf(0.07, 30)
    assert abs(crf - 0.0806) < 0.001


def test_crf_high_rate():
    """CRF at 10% for 20 years should be ~0.1175."""
    crf = compute_crf(0.10, 20)
    assert abs(crf - 0.1175) < 0.001


# ---- levelized_annual_cost tests ----


def test_levelized_zero_inflation_equals_annual_cost():
    """With zero inflation, levelized cost equals the annual cost.

    PV of flat annuity at rate i = A/CRF, so CRF * PV = A.
    """
    result = levelized_annual_cost(
        annual_cost=10.0,
        interest_rate=0.07,
        inflation_rate=0.0,
        plant_lifetime=30,
        construction_time=6,
    )
    assert abs(result - 10.0) < 0.01


def test_levelized_reference_case():
    """At 7% nominal, 2% inflation, 30yr life, 6yr construction.

    A_1 = 100 * 1.02^6 = 112.616
    PV = 112.616 * (1 - (1.02/1.07)^30) / (0.07 - 0.02) = 1716.5
    CRF(0.07, 30) = 0.08059
    levelized = 0.08059 * 1716.5 = 138.35
    """
    result = levelized_annual_cost(
        annual_cost=100.0,
        interest_rate=0.07,
        inflation_rate=0.02,
        plant_lifetime=30,
        construction_time=6,
    )
    assert abs(result - 138.35) < 1.0


def test_levelized_higher_than_simple_inflation():
    """Proper levelization should exceed the old cost * (1+g)^Tc approach.

    The growing annuity over 30 years adds ~23% vs just inflating to year 1.
    """
    simple = 100.0 * (1.02**6)  # old approach: ~112.6
    levelized = levelized_annual_cost(100.0, 0.07, 0.02, 30, 6)
    assert levelized > simple


def test_levelized_scales_linearly():
    """Doubling annual cost should double the levelized result."""
    low = levelized_annual_cost(50.0, 0.07, 0.02, 30, 6)
    high = levelized_annual_cost(100.0, 0.07, 0.02, 30, 6)
    assert abs(high / low - 2.0) < 1e-10


def test_levelized_inflation_increases_cost():
    """Higher inflation should produce higher levelized cost."""
    low = levelized_annual_cost(100.0, 0.07, 0.01, 30, 6)
    high = levelized_annual_cost(100.0, 0.07, 0.04, 30, 6)
    assert high > low


def test_levelized_i_equals_g():
    """Edge case: when interest rate equals inflation rate.

    PV = A_1 * n / (1+i) (L'Hopital limit). Should not crash.
    """
    result = levelized_annual_cost(100.0, 0.05, 0.05, 30, 6)
    assert result > 0
    assert result > 100.0  # inflation over 6yr + 30yr effect


def test_lcoe_sanity():
    """LCOE should be in reasonable range for fusion (10-200 $/MWh)."""
    lcoe = compute_lcoe(
        cas90=500.0,
        cas70=50.0,
        cas80=5.0,
        p_net=1000.0,
        n_mod=1,
        availability=0.85,
    )
    assert 10 < lcoe < 200


# ---- levelized_replacement_cost tests ----


def _manual_repl(event_cost, t_replace, i, n):
    s = (1.0 + i) ** (-t_replace)
    n_rep = max(0, math.ceil(n / t_replace) - 1)
    pv = sum(event_cost * s**k for k in range(1, n_rep + 1))
    crf = (i * (1 + i) ** n) / ((1 + i) ** n - 1)
    return pv * crf


def test_repl_matches_geometric_sum_multi_year():
    # interval 5 yr over 30 yr plant -> 5 replacements
    got = levelized_replacement_cost(100.0, 5.0, 0.07, 30)
    assert got == pytest.approx(_manual_repl(100.0, 5.0, 0.07, 30), rel=1e-6)


def test_repl_no_cap_at_many_events():
    # sub-annual interval -> ~80 events; closed form must NOT truncate at 20
    t = 1.0e8 / (
        10.0 * 8760.0 * 3600.0 * 0.85
    )  # cap_shot_lifetime / shots_per_yr at 10 Hz
    got = levelized_replacement_cost(100.0, t, 0.07, 30)
    assert got == pytest.approx(_manual_repl(100.0, t, 0.07, 30), rel=1e-6)


def test_repl_zero_when_item_outlives_plant():
    # interval 40 yr > 30 yr plant -> 0 replacements beyond the initial set
    assert levelized_replacement_cost(100.0, 40.0, 0.07, 30) == pytest.approx(0.0)
