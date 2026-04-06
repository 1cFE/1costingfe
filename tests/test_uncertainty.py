"""Tests for the probabilistic uncertainty analysis layer."""

import numpy as np
import pytest

from costingfe import ConfinementConcept, CostModel, Fuel
from costingfe.analysis.uncertainty import (
    LogNormal,
    Normal,
    Triangular,
    Uniform,
    run_uncertainty,
    run_uncertainty_full,
)

# Shared fixtures -----------------------------------------------------------


def _make_model_and_params():
    """Build a tokamak model and run forward to get base params."""
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    result = model.forward(
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
    )
    return model, result.params


MODEL, BASE_PARAMS = _make_model_and_params()


# Basic functionality -------------------------------------------------------


def test_basic_run_produces_positive_lcoes():
    res = run_uncertainty(
        MODEL,
        BASE_PARAMS,
        param_distributions={"eta_th": Normal(mean=0.40, sigma=0.03)},
        n_samples=50,
        seed=42,
    )
    assert res.n_samples == 50
    assert np.all(res.lcoe_samples > 0)


def test_deterministic_seed():
    kwargs = dict(
        model=MODEL,
        base_params=BASE_PARAMS,
        param_distributions={"eta_th": Uniform(0.35, 0.50)},
        n_samples=50,
        seed=123,
    )
    r1 = run_uncertainty(**kwargs)
    r2 = run_uncertainty(**kwargs)
    np.testing.assert_array_equal(r1.lcoe_samples, r2.lcoe_samples)


def test_wider_distribution_yields_larger_std():
    narrow = run_uncertainty(
        MODEL,
        BASE_PARAMS,
        {"eta_th": Normal(mean=0.40, sigma=0.005)},
        n_samples=200,
        seed=1,
    )
    wide = run_uncertainty(
        MODEL,
        BASE_PARAMS,
        {"eta_th": Normal(mean=0.40, sigma=0.05)},
        n_samples=200,
        seed=1,
    )
    assert wide.std > narrow.std


def test_single_uncertain_parameter():
    res = run_uncertainty(
        MODEL,
        BASE_PARAMS,
        {"availability": Triangular(0.75, 0.85, 0.95)},
        n_samples=30,
        seed=7,
    )
    assert res.n_samples == 30
    assert "availability" in res.param_samples


def test_p10_lt_p50_lt_p90():
    res = run_uncertainty(
        MODEL,
        BASE_PARAMS,
        {"eta_th": Normal(mean=0.40, sigma=0.03)},
        n_samples=200,
        seed=99,
    )
    assert res.p10 <= res.p50 <= res.p90


def test_ci_properties():
    res = run_uncertainty(
        MODEL,
        BASE_PARAMS,
        {"eta_th": Normal(mean=0.40, sigma=0.03)},
        n_samples=200,
        seed=99,
    )
    assert res.ci_80 == (res.p10, res.p90)
    assert res.ci_90 == (res.p5, res.p95)
    assert res.p5 <= res.p10
    assert res.p90 <= res.p95


# Validation -----------------------------------------------------------------


def test_invalid_parameter_name_raises():
    with pytest.raises(ValueError, match="not_a_real_param"):
        run_uncertainty(
            MODEL,
            BASE_PARAMS,
            {"not_a_real_param": Normal(0, 1)},
            n_samples=10,
            seed=0,
        )


def test_correlation_shape_mismatch_raises():
    with pytest.raises(ValueError, match="correlation_matrix shape"):
        run_uncertainty(
            MODEL,
            BASE_PARAMS,
            {
                "eta_th": Normal(0.40, 0.03),
                "availability": Normal(0.85, 0.03),
            },
            correlation_matrix=np.eye(3),  # wrong: 3x3 for 2 params
            n_samples=10,
            seed=0,
        )


# Correlation ----------------------------------------------------------------


def test_correlation_changes_joint_distribution():
    shared = dict(
        model=MODEL,
        base_params=BASE_PARAMS,
        param_distributions={
            "eta_th": Normal(0.40, 0.03),
            "availability": Normal(0.85, 0.03),
        },
        n_samples=500,
        seed=42,
    )
    r_ind = run_uncertainty(**shared)
    corr = np.array([[1.0, -0.8], [-0.8, 1.0]])
    r_cor = run_uncertainty(**shared, correlation_matrix=corr)

    # Rank correlation of samples should differ noticeably
    def _spearman_rho(x, y):
        """Spearman rank correlation (no scipy dependency)."""
        n = len(x)
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        d = rx - ry
        return 1.0 - 6.0 * np.sum(d**2) / (n * (n**2 - 1))

    rho_ind = _spearman_rho(
        r_ind.param_samples["eta_th"], r_ind.param_samples["availability"]
    )
    rho_cor = _spearman_rho(
        r_cor.param_samples["eta_th"], r_cor.param_samples["availability"]
    )
    # Independent should be near 0; correlated should be strongly negative
    assert abs(rho_ind) < 0.3
    assert rho_cor < -0.5


# Distribution types ---------------------------------------------------------


def test_lognormal_distribution():
    res = run_uncertainty(
        MODEL,
        BASE_PARAMS,
        {"eta_th": LogNormal(mu=np.log(0.40), sigma_log=0.05)},
        n_samples=50,
        seed=11,
    )
    assert res.n_samples == 50
    assert np.all(res.lcoe_samples > 0)


# run_uncertainty_full -------------------------------------------------------


def test_full_no_cc_falls_back_to_fast_path():
    r_fast = run_uncertainty(
        MODEL,
        BASE_PARAMS,
        {"eta_th": Normal(0.40, 0.03)},
        n_samples=50,
        seed=42,
    )
    r_full = run_uncertainty_full(
        ConfinementConcept.TOKAMAK,
        Fuel.DT,
        BASE_PARAMS,
        {"eta_th": Normal(0.40, 0.03)},
        cc_distributions=None,
        n_samples=50,
        seed=42,
    )
    np.testing.assert_array_almost_equal(r_fast.lcoe_samples, r_full.lcoe_samples)


def test_full_with_cc_changes_result():
    r_no_cc = run_uncertainty_full(
        ConfinementConcept.TOKAMAK,
        Fuel.DT,
        BASE_PARAMS,
        {"eta_th": Normal(0.40, 0.03)},
        cc_distributions=None,
        n_samples=50,
        seed=42,
    )
    r_with_cc = run_uncertainty_full(
        ConfinementConcept.TOKAMAK,
        Fuel.DT,
        BASE_PARAMS,
        {"eta_th": Normal(0.40, 0.03)},
        cc_distributions={"blanket_unit_cost_dt": Uniform(0.5, 2.0)},
        n_samples=50,
        n_cc_bins=3,
        seed=42,
    )
    # Adding CC uncertainty should change the spread
    assert not np.allclose(r_no_cc.lcoe_samples, r_with_cc.lcoe_samples)
    assert "blanket_unit_cost_dt" in r_with_cc.param_samples
