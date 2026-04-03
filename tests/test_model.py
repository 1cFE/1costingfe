from costingfe import ConfinementConcept, CostModel, Fuel


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
    assert result.power_table.p_target > 0  # Has target factory


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
    r2 = model.forward(
        net_electric_mw=1000.0, availability=0.85, lifetime_yr=30, eta_th=eta_th + delta
    )
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
    assert all(v > 0 for v in lcoes)


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
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
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
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
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
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
        cost_overrides={},
    )
    assert base.costs.lcoe == with_empty.costs.lcoe
    assert with_empty.overridden == []


def test_cost_override_overridden_list():
    """Overridden list should contain exactly the applied keys."""
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    result = model.forward(
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
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


# ---- CAS71/CAS72 sub-account tests ----


def test_cas70_has_subaccounts():
    """CostResult should have cas71 and cas72, summing to cas70."""
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    result = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    assert result.costs.cas71 > 0, "CAS71 (O&M) should be positive"
    assert result.costs.cas72 > 0, "CAS72 (replacement) should be positive for DT"
    assert abs(result.costs.cas70 - (result.costs.cas71 + result.costs.cas72)) < 0.001


def test_cas72_zero_for_pb11_30yr():
    """pB11 with 50 FPY core life, 30yr plant -> CAS72 = 0."""
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.PB11)
    result = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    assert result.costs.cas72 == 0.0


def test_cas72_increases_with_lifetime():
    """Longer plant life -> more replacement events -> higher CAS72."""
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    r20 = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=20)
    r40 = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=40)
    assert r40.costs.cas72 > r20.costs.cas72


def test_cas22_no_c220119():
    """CAS22 detail should not contain C220119 (moved to CAS72)."""
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    result = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    assert "C220119" not in result.cas22_detail


def test_cas72_uses_cost_overrides():
    """Overriding C220101 should affect CAS72 replacement cost."""
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    base = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    expensive = model.forward(
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
        cost_overrides={"C220101": base.cas22_detail["C220101"] * 5},
    )
    assert expensive.costs.cas72 > base.costs.cas72


# ---- DEC (Direct Energy Conversion) wiring tests ----


def test_mirror_dec_populates_c220109():
    """Mirror with f_dec > 0 should have nonzero C220109."""
    from costingfe import ConfinementConcept, CostModel, Fuel

    model = CostModel(concept=ConfinementConcept.MIRROR, fuel=Fuel.DHE3)
    result = model.forward(net_electric_mw=500.0, availability=0.85, lifetime_yr=30)
    # Mirror defaults: f_dec=0.3, eta_de=0.60 → p_dee > 0
    assert result.cas22_detail["C220109"] > 0, (
        "DHe3 mirror with f_dec=0.3 should have nonzero DEC cost"
    )


def test_tokamak_no_dec():
    """Tokamak with f_dec=0 should have zero C220109."""
    from costingfe import ConfinementConcept, CostModel, Fuel

    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    result = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    assert result.cas22_detail["C220109"] == 0.0


def test_mirror_dhe3_dec_full_integration():
    """Full DHe3 mirror plant should have DEC costs in capital and O&M."""
    from costingfe import ConfinementConcept, CostModel, Fuel

    model = CostModel(concept=ConfinementConcept.MIRROR, fuel=Fuel.DHE3)
    result = model.forward(
        net_electric_mw=500.0,
        availability=0.85,
        lifetime_yr=30,
    )
    pt = result.power_table
    c = result.costs

    # Physics: DEC should produce electric power
    assert pt.p_dee > 0, "DHe3 mirror should have DEC electric output"

    # Capital: C220109 should be nonzero
    c220109 = result.cas22_detail["C220109"]
    assert c220109 > 0, "C220109 should be nonzero for DHe3 mirror"

    # C220109 should be included in CAS22 total
    assert c.cas22 > c220109, "CAS22 should include C220109"

    # O&M: CAS72 should include DEC grid replacement
    # Run same plant without DEC to compare
    model_no_dec = CostModel(concept=ConfinementConcept.MIRROR, fuel=Fuel.DHE3)
    result_no_dec = model_no_dec.forward(
        net_electric_mw=500.0,
        availability=0.85,
        lifetime_yr=30,
        f_dec=0.0,
    )
    assert c.cas72 > result_no_dec.costs.cas72, (
        "CAS72 should be higher with DEC grid replacement"
    )


def test_mirror_dt_no_dec_by_default():
    """DT mirror with default f_dec=0.3 should still compute DEC costs.

    The model does not gate on fuel — f_dec > 0 is sufficient.
    Mirror defaults have f_dec=0.3 for all fuels.
    """
    from costingfe import ConfinementConcept, CostModel, Fuel

    model = CostModel(concept=ConfinementConcept.MIRROR, fuel=Fuel.DT)
    result = model.forward(net_electric_mw=500.0, availability=0.85, lifetime_yr=30)
    # Mirror default f_dec=0.3 applies even for DT
    assert result.cas22_detail["C220109"] > 0
