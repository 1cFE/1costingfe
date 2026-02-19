from costingfe import CostModel, ConfinementConcept, Fuel


def test_forward_basic():
    """Basic forward costing should produce an LCOE."""
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    result = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    assert result.costs.lcoe > 0
    assert result.power_table.p_net > 0
    assert result.power_table.p_fus > 0


def test_forward_lcoe_range():
    """LCOE for a tokamak DT plant should be in reasonable range."""
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    result = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    assert 10 < result.costs.lcoe < 500, f"LCOE {result.costs.lcoe} $/MWh unexpected"


def test_forward_pb11_cheaper_licensing():
    """pB11 plant should have lower licensing cost than DT."""
    model_dt = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    model_pb11 = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.PB11)
    result_dt = model_dt.forward(
        net_electric_mw=1000.0, availability=0.85, lifetime_yr=30
    )
    result_pb11 = model_pb11.forward(
        net_electric_mw=1000.0, availability=0.85, lifetime_yr=30
    )
    assert result_pb11.costs.cas10 < result_dt.costs.cas10


def test_sensitivity_returns_categorized():
    """Sensitivity should separate engineering from financial parameters."""
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    result = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    sens = model.sensitivity(result.params)
    assert "engineering" in sens
    assert "financial" in sens
    assert "eta_th" in sens["engineering"]
    assert sens["engineering"]["eta_th"] != 0
    assert "interest_rate" in sens["financial"]
    assert "interest_rate" not in sens["engineering"]


def test_forward_ife_laser():
    """IFE laser fusion should produce a valid LCOE."""
    model = CostModel(concept=ConfinementConcept.LASER_IFE, fuel=Fuel.DT)
    result = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    assert result.costs.lcoe > 0
    assert result.power_table.p_net > 0
    assert result.power_table.p_coils == 0.0  # No magnets in IFE
    assert result.power_table.p_target > 0    # Has target factory


def test_forward_mif_mag_target():
    """MIF magnetized target fusion should produce a valid LCOE."""
    model = CostModel(concept=ConfinementConcept.MAG_TARGET, fuel=Fuel.DT)
    result = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    assert result.costs.lcoe > 0
    assert result.power_table.p_net > 0
    assert result.power_table.p_target > 0  # Has liner/target factory


def test_sensitivity_ife():
    """IFE sensitivity should include driver-specific parameters."""
    model = CostModel(concept=ConfinementConcept.LASER_IFE, fuel=Fuel.DT)
    result = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    sens = model.sensitivity(result.params)
    assert "eta_pin1" in sens["engineering"]  # IFE-specific param
    assert "p_input" not in sens["engineering"]  # MFE-specific param


def test_sensitivity_jax_grad_matches_finite_diff():
    """JAX grad elasticities should be close to finite-difference estimates."""
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    result = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)

    # JAX grad
    sens = model.sensitivity(result.params)

    # Manual finite-difference for eta_th
    base_lcoe = float(result.costs.lcoe)
    eta_th = result.params["eta_th"]
    delta = eta_th * 0.01
    r2 = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30,
                        eta_th=eta_th + delta)
    fd_elasticity = ((float(r2.costs.lcoe) - base_lcoe) / delta) * eta_th / base_lcoe

    assert abs(sens["engineering"]["eta_th"] - fd_elasticity) < 0.01, (
        f"JAX grad {sens['engineering']['eta_th']:.4f} vs FD {fd_elasticity:.4f}"
    )


def test_batch_lcoe_vmap():
    """batch_lcoe should evaluate many parameter sets via vmap."""
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    result = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)

    # Sweep eta_th from 0.35 to 0.55
    eta_values = [0.35, 0.40, 0.45, 0.50, 0.55]
    lcoes = model.batch_lcoe({"eta_th": eta_values}, result.params)

    assert len(lcoes) == 5
    # Higher eta_th should give lower LCOE
    assert lcoes[0] > lcoes[-1]
    # All should be positive
    assert all(l > 0 for l in lcoes)


def test_compare_all_returns_ranking():
    """Cross-concept comparison should return sorted results."""
    from costingfe import compare_all

    results = compare_all(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    assert len(results) > 0
    # Should be sorted by LCOE ascending
    lcoes = [r.lcoe for r in results]
    assert lcoes == sorted(lcoes)


# ---- Cost override tests ----


def test_cost_override_cas21_propagates():
    """Overriding CAS21 should propagate to CAS20, total_capital, and LCOE."""
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    base = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    overridden = model.forward(
        net_electric_mw=1000.0, availability=0.85, lifetime_yr=30,
        cost_overrides={"CAS21": 50.0},
    )
    assert overridden.costs.cas21 == 50.0
    # CAS20 should differ from base (uses overridden CAS21)
    assert overridden.costs.cas20 != base.costs.cas20
    # total_capital and LCOE should also change
    assert overridden.costs.total_capital != base.costs.total_capital
    assert overridden.costs.lcoe != base.costs.lcoe
    # Overridden list tracks it
    assert "CAS21" in overridden.overridden


def test_cost_override_cas22_subaccount():
    """Overriding a CAS22 sub-account should recompute CAS22 total."""
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    base = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    overridden = model.forward(
        net_electric_mw=1000.0, availability=0.85, lifetime_yr=30,
        cost_overrides={"C220103": 300.0},
    )
    # CAS22 total should change
    assert overridden.costs.cas22 != base.costs.cas22
    # The sub-account should be the overridden value
    assert overridden.cas22_detail["C220103"] == 300.0
    # Tracked in overridden list
    assert "C220103" in overridden.overridden


def test_cost_override_no_overrides_unchanged():
    """No cost_overrides should produce identical results to default."""
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    base = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    with_empty = model.forward(
        net_electric_mw=1000.0, availability=0.85, lifetime_yr=30,
        cost_overrides={},
    )
    assert base.costs.lcoe == with_empty.costs.lcoe
    assert with_empty.overridden == []


def test_cost_override_overridden_list():
    """Overridden list should contain exactly the applied keys."""
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    result = model.forward(
        net_electric_mw=1000.0, availability=0.85, lifetime_yr=30,
        cost_overrides={"CAS10": 5.0, "CAS21": 50.0, "C220103": 300.0},
    )
    assert set(result.overridden) == {"CAS10", "CAS21", "C220103"}


def test_cas22_detail_in_result():
    """ForwardResult should include CAS22 sub-account detail."""
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    result = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    assert "C220101" in result.cas22_detail
    assert "C220103" in result.cas22_detail
    assert "C220000" in result.cas22_detail
    assert result.cas22_detail["C220000"] == result.costs.cas22
