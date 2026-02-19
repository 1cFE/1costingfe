from costingfe.layers.economics import (
    compute_crf,
    compute_effective_crf,
    levelized_annual_cost,
    compute_lcoe,
)


def test_crf_basic():
    """CRF at 7% for 30 years should be ~0.0806."""
    crf = compute_crf(0.07, 30)
    assert abs(crf - 0.0806) < 0.001


def test_crf_high_rate():
    """CRF at 10% for 20 years should be ~0.1175."""
    crf = compute_crf(0.10, 20)
    assert abs(crf - 0.1175) < 0.001


def test_effective_crf():
    """Effective CRF accounts for construction time."""
    crf = compute_crf(0.07, 30)
    eff_crf = compute_effective_crf(0.07, 30, 6)
    assert eff_crf > crf  # construction time increases effective CRF
    expected = crf * (1.07**6)
    assert abs(eff_crf - expected) < 0.0001


def test_levelized_annual_cost_zero_inflation():
    """With zero inflation, cost is unchanged."""
    result = levelized_annual_cost(
        annual_cost=10.0,
        inflation_rate=0.0,
        construction_time=6,
    )
    assert abs(result - 10.0) < 0.001


def test_levelized_annual_cost_inflation_scales():
    """Higher inflation should increase levelized cost."""
    low = levelized_annual_cost(
        annual_cost=10.0,
        inflation_rate=0.02,
        construction_time=6,
    )
    high = levelized_annual_cost(
        annual_cost=10.0,
        inflation_rate=0.05,
        construction_time=6,
    )
    assert high > low


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
