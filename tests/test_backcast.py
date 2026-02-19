import pytest

from costingfe import CostModel, ConfinementConcept, Fuel
from costingfe.analysis.backcast import backcast_single, backcast_multi


MODEL = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
BASE = MODEL.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
BASE_LCOE = float(BASE.costs.lcoe)


def test_backcast_eta_th_lower_lcoe():
    """Higher eta_th should achieve a lower LCOE target."""
    target = BASE_LCOE * 0.998  # 0.2% cheaper (eta_th elasticity is small)
    eta_th = backcast_single(
        MODEL, target, "eta_th", (0.30, 0.70), base_params=BASE.params,
    )
    assert eta_th > BASE.params["eta_th"]  # Need better efficiency


def test_backcast_interest_rate():
    """Lower interest rate should achieve a lower LCOE target."""
    target = BASE_LCOE * 0.8  # 20% cheaper
    rate = backcast_single(
        MODEL, target, "interest_rate", (0.01, 0.15), base_params=BASE.params,
    )
    assert rate < BASE.params["interest_rate"]


def test_backcast_roundtrip():
    """Backcasting to current LCOE should return current parameter value."""
    eta_th = backcast_single(
        MODEL, BASE_LCOE, "eta_th", (0.30, 0.70), base_params=BASE.params,
    )
    assert abs(eta_th - BASE.params["eta_th"]) < 0.01


def test_backcast_unreachable_raises():
    """Impossible target should raise ValueError."""
    with pytest.raises(ValueError, match="not achievable"):
        backcast_single(
            MODEL, 1.0, "eta_th", (0.30, 0.70), base_params=BASE.params,
        )  # 1 $/MWh is impossible


def test_backcast_multi_returns_dict():
    """Multi-backcast should return results for achievable parameters."""
    target = BASE_LCOE * 0.80  # 20% cheaper — achievable via interest_rate
    results = backcast_multi(
        MODEL, target,
        {"eta_th": (0.30, 0.70), "interest_rate": (0.01, 0.15)},
        base_params=BASE.params,
    )
    assert "eta_th" in results
    assert "interest_rate" in results
    # interest_rate has high elasticity — should be achievable
    assert results["interest_rate"] is not None
    # eta_th has low elasticity — may not reach 20% reduction
    # (just check the dict structure, not achievability)


def test_backcast_multi_unreachable_is_none():
    """Unreachable params should return None in multi-backcast."""
    results = backcast_multi(
        MODEL, 1.0,  # impossible
        {"eta_th": (0.30, 0.70)},
        base_params=BASE.params,
    )
    assert results["eta_th"] is None
