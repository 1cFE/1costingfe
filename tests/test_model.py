import os

import pytest

from costingfe import ConfinementConcept, CostModel, Fuel
from costingfe.layers.physics import SizingInfeasible
from costingfe.model import _n_mod_for_target


def test_forward_rejects_unknown_kwarg():
    """An unknown override kwarg must raise, not be silently ignored.

    Silently dropping a stale/misspelled magnet kwarg (e.g. r_coil/b_max)
    leaves the YAML default in force and produces a wrong cost with no
    signal. The error should name the offending key and suggest the
    intended one.
    """
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    with pytest.raises(ValueError, match="r_coil"):
        model.forward(
            net_electric_mw=1000.0,
            availability=0.85,
            lifetime_yr=30,
            r_coil=5.0,  # not a real parameter; the live key is r_bore
        )


def test_forward_accepts_known_kwargs():
    """Legitimate engineering and costing-constant overrides still work."""
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    result = model.forward(
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
        B=11.0,
        eta_th=0.45,
    )
    assert result.costs.lcoe > 0


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


def test_capital_fields_have_correct_units():
    """overnight_cost is M$ true overnight (CAS10-50, excl IDC); capital_per_kw is $/kW.

    Regression for issue #34: overnight_cost used to hold the $/kW specific cost
    (and silently included IDC). It must now be the M$ overnight sum, distinct
    from total_capital (which adds CAS60 IDC), with the $/kW figure living in
    capital_per_kw.
    """
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    r = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    c = r.costs

    expected_overnight = c.cas10 + c.cas20 + c.cas30 + c.cas40 + c.cas50
    assert c.overnight_cost == pytest.approx(expected_overnight)
    # Overnight excludes IDC, so it must be strictly below total_capital.
    assert c.overnight_cost < c.total_capital
    assert c.total_capital == pytest.approx(c.overnight_cost + c.cas60)

    # capital_per_kw is the $/kW specific cost off total_capital and total net power.
    n_mod = r.params["n_mod"]
    expected_per_kw = c.total_capital * 1e6 / (r.power_table.p_net * n_mod * 1e3)
    assert c.capital_per_kw == pytest.approx(expected_per_kw)
    # The two are different magnitudes; the old field conflated them.
    assert c.capital_per_kw != pytest.approx(c.overnight_cost)


def test_construction_time_comes_from_yaml():
    """A concept's YAML construction_time_yr must be used, not silently
    replaced by a generic signature default. Orbitron's YAML says 3.0."""
    m = CostModel(concept=ConfinementConcept.ORBITRON, fuel=Fuel.PB11)
    r = m.forward(net_electric_mw=0.005, availability=0.85, lifetime_yr=30, n_mod=1)
    assert float(r.params["construction_time_yr"]) == 3.0


def test_construction_time_explicit_override_wins():
    """An explicit construction_time_yr from the caller overrides the YAML."""
    m = CostModel(concept=ConfinementConcept.ORBITRON, fuel=Fuel.PB11)
    r = m.forward(
        net_electric_mw=0.005,
        availability=0.85,
        lifetime_yr=30,
        n_mod=1,
        construction_time_yr=5.0,
    )
    assert float(r.params["construction_time_yr"]) == 5.0


def test_orbitron_runs_at_kwe_design_point():
    """ORBITRON defaults are kWe-class (Avalanche's 5 kWe Orbitron module).

    A single module at its 5 kWe design point must produce a valid forward()
    result with no analyst overrides. This requires (a) kWe-class YAML defaults
    and (b) the feasibility check evaluating the concept's actual low-radiation
    plasma rather than representative dense-thermal values.
    """
    model = CostModel(concept=ConfinementConcept.ORBITRON, fuel=Fuel.PB11)
    result = model.forward(
        net_electric_mw=0.005, availability=0.85, lifetime_yr=30, n_mod=1
    )
    assert result.power_table.p_net > 0
    assert float(result.power_table.rec_frac) < 0.95
    assert result.costs.lcoe > 0


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
    # Rep-rate concept: a 1 GWe plant exceeds one chamber's cited-shot ceiling,
    # so it sizes to multiple chambers via size_from_power (the cited shot is
    # now authoritative; the normal path solves f_rep and raises above one
    # chamber's ceiling).
    model = CostModel(concept=ConfinementConcept.LASER_IFE, fuel=Fuel.DT)
    result = model.forward(
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
        size_from_power=True,
    )
    assert result.costs.lcoe > 0
    assert result.power_table.p_net > 0
    assert result.power_table.p_coils == 0.0  # No magnets in IFE
    assert result.power_table.p_target > 0  # Has target factory


def test_forward_mif_mag_target():
    """MIF magnetized target fusion should produce a valid LCOE."""
    # Rep-rate concept: 1 GWe exceeds one chamber's ceiling, so size to multiple
    # chambers (see test_forward_ife_laser).
    model = CostModel(concept=ConfinementConcept.MAG_TARGET, fuel=Fuel.DT)
    result = model.forward(
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
        size_from_power=True,
    )
    assert result.costs.lcoe > 0
    assert result.power_table.p_net > 0
    # The generic MTF concept (General Fusion / Helion-class) forms the
    # magnetized target in-situ and recycles its liquid-metal liner, so there
    # is no target factory by default: p_target and the C220108 factory are 0.
    # A pellet-fed MTF (e.g. NearStar) sets these in its own concept config.
    assert float(result.power_table.p_target) == 0.0
    assert float(result.cas22_detail["C220108"]) == 0.0


def test_sensitivity_ife():
    """IFE sensitivity should include driver-specific parameters."""
    # Rep-rate concept at 1 GWe sizes to multiple chambers; sensitivity uses the
    # finite-difference path (the f_rep solve is not jax-differentiable).
    model = CostModel(concept=ConfinementConcept.LASER_IFE, fuel=Fuel.DT)
    result = model.forward(
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
        size_from_power=True,
    )
    sens = model.sensitivity(result.params)
    assert "eta_pin" in sens["engineering"]  # Pulsed driver efficiency
    assert "p_input" not in sens["engineering"]  # MFE-specific param


def test_sensitivity_drops_broken_heating_mix_sliders():
    """p_nbi/p_ecrh/p_icrf/p_lhcd are not valid sensitivity sliders: production
    renormalizes the heating mix to p_input, but the JAX-traced sensitivity path
    skips that renorm, so their reported elasticity does not match forward().
    They must not appear in any sensitivity category."""
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    result = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    sens = model.sensitivity(result.params)
    for category in sens.values():
        for k in ("p_nbi", "p_ecrh", "p_icrf", "p_lhcd"):
            assert k not in category, f"{k} should not be a sensitivity slider"


def test_sensitivity_includes_lifetime_yr():
    """Plant economic lifetime is an LCOE driver and must be a sensitivity slider."""
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    result = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    sens = model.sensitivity(result.params)
    assert "lifetime_yr" in sens["engineering"]
    assert sens["engineering"]["lifetime_yr"] != 0


def test_sensitivity_ife_includes_target_unit_cost():
    """Per-shot target cost (CAS80) is a dominant IFE/MIF LCOE driver and must be
    a sensitivity slider for pulsed concepts that consume targets."""
    model = CostModel(concept=ConfinementConcept.LASER_IFE, fuel=Fuel.DT)
    result = model.forward(
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
        size_from_power=True,  # rep-rate 1 GWe sizes to multiple chambers
    )
    sens = model.sensitivity(result.params)
    assert "target_unit_cost" in sens["engineering"]
    assert sens["engineering"]["target_unit_cost"] != 0


def test_sensitivity_jax_grad_matches_finite_diff():
    """JAX grad elasticities should be close to finite-difference estimates."""
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    result = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)

    # JAX grad
    sens = model.sensitivity(result.params)

    # Centered finite-difference for eta_th. The fluence-based CAS72 basis
    # (re-pinned 2026-06-13, see wall_limits_and_fluence.md) makes core
    # lifetime depend on eta_th through q_n, adding curvature near eta_th=0.4;
    # a small centered step is required to match the analytic derivative.
    base_lcoe = float(result.costs.lcoe)
    eta_th = result.params["eta_th"]
    delta = eta_th * 1e-3
    r_hi = model.forward(
        net_electric_mw=1000.0, availability=0.85, lifetime_yr=30, eta_th=eta_th + delta
    )
    r_lo = model.forward(
        net_electric_mw=1000.0, availability=0.85, lifetime_yr=30, eta_th=eta_th - delta
    )
    fd_elasticity = (
        ((float(r_hi.costs.lcoe) - float(r_lo.costs.lcoe)) / (2 * delta))
        * eta_th
        / base_lcoe
    )

    assert abs(sens["engineering"]["eta_th"] - fd_elasticity) < 0.01, (
        f"JAX grad {sens['engineering']['eta_th']:.4f} vs FD {fd_elasticity:.4f}"
    )


def test_forward_accepts_dhe3_dd_frac_pin_override():
    """forward() must accept dhe3_dd_frac_pin so a 0D forward's params round-trip.

    Regression for the issue #36 crash: the 0D path injects dhe3_dd_frac_pin
    into the returned params, but it was not an accepted override key, so
    replaying those params through forward() (as sensitivity() does) raised
    ValueError. params out of forward() must round-trip back into forward().
    """
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    r = model.forward(
        net_electric_mw=1000.0, availability=0.85, lifetime_yr=30, use_0d_model=True
    )
    assert "dhe3_dd_frac_pin" in r.params
    # Replaying every returned param as an override must not raise.
    named = {
        "net_electric_mw",
        "availability",
        "lifetime_yr",
        "n_mod",
        "interest_rate",
        "inflation_rate",
        "noak",
    }
    eng = {k: v for k, v in r.params.items() if k not in named | {"fuel", "concept"}}
    r2 = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30, **eng)
    assert float(r2.costs.lcoe) > 0


def test_sensitivity_0d_tokamak_does_not_crash():
    """sensitivity() on a 0D tokamak must return categorized elasticities.

    Regression for issue #36: this used to raise ValueError before reaching
    any gradient computation (dhe3_dd_frac_pin not round-trippable).
    """
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    r = model.forward(
        net_electric_mw=1000.0, availability=0.85, lifetime_yr=30, use_0d_model=True
    )
    sens = model.sensitivity(r.params)
    assert "engineering" in sens and "financial" in sens and "costing" in sens
    assert "B" in sens["engineering"]


def test_sensitivity_0d_matches_forward_finite_diff():
    """0D sensitivity must reproduce forward() (the slider answer), not jax.grad.

    Issue #36: jax.grad cannot see through the tokamak_0d_inverse bisection on
    T_e -- every parameter enters via the non-differentiable comparison, so
    autodiff returns gradients that disagree with, and even flip the sign of,
    the concrete forward(). The 0D path must fall back to finite differences on
    forward(). We verify the returned elasticities for physics levers match an
    independent central-difference on forward() (which jax.grad would fail).
    """
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    r = model.forward(
        net_electric_mw=1000.0, availability=0.85, lifetime_yr=30, use_0d_model=True
    )
    sens = model.sensitivity(r.params)
    base = float(r.costs.lcoe)

    def fd(key, h=1e-3):
        p0 = float(r.params[key])
        hi = model.forward(
            net_electric_mw=1000.0,
            availability=0.85,
            lifetime_yr=30,
            use_0d_model=True,
            **{key: p0 * (1 + h)},
        )
        lo = model.forward(
            net_electric_mw=1000.0,
            availability=0.85,
            lifetime_yr=30,
            use_0d_model=True,
            **{key: p0 * (1 - h)},
        )
        return (float(hi.costs.lcoe) - float(lo.costs.lcoe)) / (2 * p0 * h) * p0 / base

    for key in ("B", "elon", "R0", "f_GW"):
        want = fd(key)
        got = sens["engineering"][key]
        assert abs(got - want) < max(0.02, 0.05 * abs(want)), (
            f"{key}: sensitivity {got:.4f} vs forward-FD {want:.4f}"
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


def test_cas21_grows_with_module_count():
    """CAS21 (buildings/site) must not shrink when a plant is split into more
    (smaller) modules. A 1 GWe plant of 20 modules houses more installed
    equipment than a single 1 GWe machine, so its buildings/site cost is
    higher, not pinned to one small module's value."""
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    one = model.forward(
        net_electric_mw=1000.0, n_mod=1, availability=0.85, lifetime_yr=30
    )
    many = model.forward(
        net_electric_mw=1000.0, n_mod=20, availability=0.85, lifetime_yr=30
    )
    assert many.costs.cas21 > one.costs.cas21


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


def test_override_to_attr_keys_all_take_effect():
    """Every key in _OVERRIDE_TO_ATTR must actually be applied by forward().

    The map is consumed only by _scale_overrides(), which scales an override
    by the account's value ratio across plant sizes. A key that forward() never
    reads from cost_overrides is dead: its override is silently scaled yet never
    affects output. Overriding any mapped key must set costs.<attr> to that value.
    """
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    sentinel = 1234.5
    for key, attr in model._OVERRIDE_TO_ATTR.items():
        result = model.forward(
            net_electric_mw=1000.0,
            availability=0.85,
            lifetime_yr=30,
            cost_overrides={key: sentinel},
        )
        assert getattr(result.costs, attr) == sentinel, (
            f"{key} is in _OVERRIDE_TO_ATTR but forward() does not apply it "
            f"(costs.{attr}={getattr(result.costs, attr)}, expected {sentinel})"
        )
        assert key in result.overridden, f"{key} applied but not tracked in overridden"


def test_dipole_concept_runs():
    """DIPOLE (levitated dipole) is a steady-state concept that runs end-to-end,
    with the floating superconducting coil as a nonzero signature cost."""
    model = CostModel(concept=ConfinementConcept.DIPOLE, fuel=Fuel.DT)
    result = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    assert result.costs.lcoe > 0
    assert result.power_table.p_net > 0
    # The floating superconducting dipole coil dominates dipole capital — it must
    # not cost zero (the failure mode if DIPOLE is absent from _COIL_DEFAULTS).
    assert result.cas22_detail["C220103"] > 0


def test_dipole_radiation_peaking_keeps_recirc_sane():
    """With the peaking factor, the dipole's bremsstrahlung no longer explodes,
    so the recirculating fraction stays physical (issue #24). Without it, the
    13,600 m^3 geometric volume drives p_rad above the fusion power and pushes
    recirc toward 86%."""
    model = CostModel(concept=ConfinementConcept.DIPOLE, fuel=Fuel.DT)
    result = model.forward(net_electric_mw=208.0, availability=0.85, lifetime_yr=30)
    assert float(result.power_table.rec_frac) < 0.5
    # p_fus must be the right order of magnitude (hundreds of MW, not thousands).
    assert 300.0 < float(result.power_table.p_fus) < 1500.0


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


def test_laser_driver_type_defaults_to_dpssl_from_yaml():
    from costingfe.types import LaserDriverType

    model = CostModel(concept=ConfinementConcept.LASER_IFE, fuel=Fuel.DT)
    assert model.laser_driver_type == LaserDriverType.DPSSL


def test_laser_driver_type_override_changes_capital_and_om():
    from costingfe.types import LaserDriverType

    # Rep-rate concept at 1 GWe sizes to multiple chambers (same n_mod for both
    # driver types, so the C220104 ratio comparison is unaffected).
    dpssl = CostModel(
        ConfinementConcept.LASER_IFE, Fuel.DT, laser_driver_type=LaserDriverType.DPSSL
    ).forward(1000.0, 0.85, 30, size_from_power=True)
    krf = CostModel(
        ConfinementConcept.LASER_IFE, Fuel.DT, laser_driver_type=LaserDriverType.KRF
    ).forward(1000.0, 0.85, 30, size_from_power=True)
    assert krf.cas22_detail["C220104"] < dpssl.cas22_detail["C220104"]  # 40 < 205 $/MJ
    assert krf.costs.lcoe != dpssl.costs.lcoe


# ──────────────────────────────────────────────────────────────────────
# Regression tests for costingfe-library-preconditions spec (FR-1, FR-3)
#
# Two-knob projection call shape:
#   forward(net_electric_mw=1000, n_mod=1000/P_native,
#           override_reference_mw=P_native, cost_overrides={...})
#
# Expected semantics (FR-2):
#   - The reference-side forward inside _scale_overrides runs at n_mod=1, so
#     per-module reactor-island overrides pass through unchanged per module
#     (scaling ratio = 1.0 for power-dependent and power-independent accounts
#     alike, since both run at the same per-module power).
#   - The target-side forward uses the caller's n_mod, so plant-aggregate
#     accounts like CAS22 scale by approximately n_mod (close to but not
#     exactly equal due to the multi-unit labor learning factor).
#
# Each test exercises the two-knob call shape and asserts the implied scaling
# ratio matches the spec's semantic for one specific account family.
# ──────────────────────────────────────────────────────────────────────

# Shared two-knob fixture for the FR-3 regression tests
_TWO_KNOB_P_NATIVE = 261.0
_TWO_KNOB_BASE = dict(
    availability=0.85,
    lifetime_yr=30,
    construction_time_yr=5.0,
    R0=3.3,
    plasma_t=1.13,
    elon=1.84,
    blanket_t=0.80,
    ht_shield_t=0.20,
    structure_t=0.20,
    vessel_t=0.20,
    p_input=38.6,
    mn=1.1,
    eta_p=0.5,
    eta_couple=0.8,
    f_sub=0.03,
    p_coils=2.0,
    p_cool=13.7,
    p_pump=1.0,
    p_trit=10.0,
    p_house=4.0,
    p_cryo=0.5,
)


def test_n_mod_accepts_float():
    """FR-1: n_mod must accept positive real values for the two-knob projection.

    The rework's two-knob mechanism computes n_mod = 1000 / P_native, which is
    almost always non-integer (e.g. 1000/261 ≈ 3.83 for ARC). Passing a float
    must not raise a validation error.
    """
    from costingfe.validation import CostingInput

    valid = CostingInput(
        concept=ConfinementConcept.TOKAMAK,
        fuel=Fuel.DT,
        net_electric_mw=1000.0,
        n_mod=1000.0 / _TWO_KNOB_P_NATIVE,
    )
    assert valid.n_mod == 1000.0 / _TWO_KNOB_P_NATIVE


def test_two_knob_per_module_power_dependent_passthrough():
    """FR-3(a): per-module power-dependent override passes through unchanged.

    Spec example account: C220101 (first wall + blanket). The reference-side
    forward at n_mod=1 produces a per-module cost; the target-side forward at
    n_mod = 1000/P_native produces the same per-module cost (since per-module
    power is unchanged). The override's user-supplied value must arrive at the
    target plant with scaling ratio exactly 1.0.
    """
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    n_mod_run = 1000.0 / _TWO_KNOB_P_NATIVE
    override_val = 349.0

    result = model.forward(
        net_electric_mw=1000.0,
        n_mod=n_mod_run,
        override_reference_mw=_TWO_KNOB_P_NATIVE,
        cost_overrides={"C220101": override_val},
        **_TWO_KNOB_BASE,
    )
    # Per-module override flows through at the target per-module value (not
    # multiplied by n_mod, because the user's frame is one module).
    assert abs(result.cas22_detail["C220101"] - override_val) < 0.01, (
        f"C220101 override expected to pass through at {override_val} M$ per module; "
        f"got {result.cas22_detail['C220101']:.2f}"
    )


def test_two_knob_per_module_no_power_term_passthrough():
    """FR-3(a) variant: a per-module account with no power scaling term also
    passes through unchanged.

    Spec example account: C220103 (coils). The coil cost formula depends only
    on geometry (b_max, r_coil, n_coils), not on power. Under FR-2 this account
    arrives at ratio 1.0 in both the buggy and fixed reference frames — but the
    test still locks down the expected behavior.
    """
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    n_mod_run = 1000.0 / _TWO_KNOB_P_NATIVE
    override_val = 6901.0

    result = model.forward(
        net_electric_mw=1000.0,
        n_mod=n_mod_run,
        override_reference_mw=_TWO_KNOB_P_NATIVE,
        cost_overrides={"C220103": override_val},
        **_TWO_KNOB_BASE,
    )
    assert abs(result.cas22_detail["C220103"] - override_val) < 0.01, (
        f"C220103 override expected to pass through at {override_val} M$ per module; "
        f"got {result.cas22_detail['C220103']:.2f}"
    )


def test_two_knob_plant_aggregate_scales_with_n_mod():
    """FR-3(b): a plant-aggregate override scales by approximately n_mod.

    Spec example account: CAS22 (Reactor Plant Equipment total). Its default
    computation is per_module_equipment * n_mod + labor + plant_wide, so it
    already sums over modules. The user-supplied override (in plant-aggregate
    frame at the reference power) must scale toward the target plant total —
    approximately n_mod, but slightly less due to the multi-unit labor
    learning factor encoded in the plant-aggregate formula.

    This assertion guards against an over-eager FR-2 fix that also forces
    target-side n_mod=1 (which would incorrectly keep the override at the
    reference plant value instead of scaling to the target plant value).
    """
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    n_mod_run = 1000.0 / _TWO_KNOB_P_NATIVE
    override_val = 7940.0

    result = model.forward(
        net_electric_mw=1000.0,
        n_mod=n_mod_run,
        override_reference_mw=_TWO_KNOB_P_NATIVE,
        cost_overrides={"CAS22": override_val},
        **_TWO_KNOB_BASE,
    )
    ratio = result.costs.cas22 / override_val
    # Should be approximately n_mod (3.83), allowing for multi-unit labor
    # learning savings; not 1.0 (which would indicate the fix went too far).
    assert 0.8 * n_mod_run < ratio < n_mod_run, (
        f"CAS22 plant-aggregate override expected to scale by roughly n_mod "
        f"({n_mod_run:.2f}) minus multi-unit labor savings; got ratio {ratio:.3f}"
    )


def test_scale_overrides_cas22_zero_account_scales_linearly():
    """Issue #37: an absolute override on a CAS22 sub-account the library
    computes as $0 for this config must scale linearly with plant power, not
    freeze at the reference-power dollars.

    PULSED_FRC has no per-MJ driver coefficient for C220104, so the library
    value is $0 at every power; the analyst's override is the cost basis and,
    being reactor-island hardware, should grow with the plant.
    """
    # PULSED_FRC is a rep-rate concept: a 1 GWe target exceeds one chamber's
    # cited-shot ceiling, so the target-side forward sizes to multiple chambers
    # (size_from_power). The library-zero C220104 override still scales linearly
    # with plant net power regardless of chamber count.
    model = CostModel(concept=ConfinementConcept.PULSED_FRC, fuel=Fuel.DHE3)
    scaled = model._scale_overrides(
        {"C220104": 25.0},
        reference_mw=50.0,
        target_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
        size_from_power=True,
    )
    # 25 * (1000 / 50) = 500, not the frozen 25.
    assert scaled["C220104"] == pytest.approx(500.0, rel=1e-6)


def test_scale_overrides_cas22_nonzero_account_uses_library_ratio():
    """A CAS22 sub-account the library DOES compute must keep scaling by the
    library's own per-account ratio (tgt/ref), not the linear power ratio.

    Guards against the #37 fix bleeding into the working path.
    """
    # PULSED_FRC is a rep-rate concept: its per-module reactor-island hardware
    # (C220107) is fixed by the cited shot, so a 1 GWe plant scales that account
    # by the solved chamber count (size_from_power), not by per-module power.
    # The override must follow the library's own tgt/ref ratio, which is that
    # chamber count, not the naive linear power ratio.
    model = CostModel(concept=ConfinementConcept.PULSED_FRC, fuel=Fuel.DHE3)
    ref = model.forward(
        net_electric_mw=50.0, n_mod=1, availability=0.85, lifetime_yr=30
    )
    tgt = model.forward(
        net_electric_mw=1000.0, availability=0.85, lifetime_yr=30, size_from_power=True
    )
    lib_ratio = float(tgt.cas22_detail["C220107"]) / float(ref.cas22_detail["C220107"])
    scaled = model._scale_overrides(
        {"C220107": 100.0},
        reference_mw=50.0,
        target_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
        size_from_power=True,
    )
    assert scaled["C220107"] == pytest.approx(100.0 * lib_ratio, rel=1e-6)
    # The library ratio must differ from the naive linear power ratio, proving
    # the nonzero path still uses the per-account law (here the solved chamber
    # count, about 15x, not the 20x linear ratio).
    assert lib_ratio != pytest.approx(1000.0 / 50.0, rel=1e-3)


def test_scale_overrides_toplevel_zero_account_stays_frozen():
    """Scope boundary: top-level accounts the library computes as $0 are NOT
    linearly scaled (only CAS22 sub-account hardware is). Overriding such a
    top-level zero account passes through frozen, so genuinely fixed/absent
    costs (e.g. CAS28 digital twin) are not silently inflated with plant size.
    """
    model = CostModel(concept=ConfinementConcept.PULSED_FRC, fuel=Fuel.DHE3)
    ref = model.forward(
        net_electric_mw=50.0, n_mod=1, availability=0.85, lifetime_yr=30
    )
    assert float(ref.costs.cas23) == 0.0  # precondition: library-zero top-level
    scaled = model._scale_overrides(
        {"CAS23": 10.0},
        reference_mw=50.0,
        target_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
        size_from_power=True,  # rep-rate 1 GWe target sizes to multiple chambers
    )
    assert scaled["CAS23"] == 10.0  # frozen, not scaled


def test_n_mod_does_not_scale_per_module_core_lifetime():
    """Per-module q_n and core lifetime must be invariant to n_mod.

    The fluence-based core lifetime uses pt.p_neutron and geo.firstwall_area,
    both of which are per-module quantities. If someone erroneously multiplied
    q_n by n_mod, core lifetime would fall with n_mod, making CAS72 grow
    super-linearly rather than linearly. This test locks that invariant.

    Construction: run TOKAMAK DT at n_mod=1 / 500 MWe and again at n_mod=2 /
    1000 MWe (identical per-module operating point). Because the replacement
    cost per event is sum(cas22_detail[k]) * n_mod and core_lifetime is
    per-module, CAS72 must scale exactly linearly with n_mod. A doubling of
    n_mod at the same per-module power must therefore produce CAS72 = 2 *
    CAS72_single within floating-point tolerance.
    """
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    one = model.forward(
        net_electric_mw=500.0, n_mod=1, availability=0.85, lifetime_yr=30
    )
    two = model.forward(
        net_electric_mw=1000.0, n_mod=2, availability=0.85, lifetime_yr=30
    )
    cas72_one = float(one.costs.cas72)
    cas72_two = float(two.costs.cas72)
    assert cas72_one > 0, "CAS72 should be positive for a DT tokamak"
    # Tolerance of 0.1% allows for rounding in intermediate float ops but
    # would catch any n_mod multiplication of q_n (which would roughly halve
    # core lifetime and roughly double CAS72 per module, yielding ~4x total).
    assert abs(cas72_two - 2.0 * cas72_one) / cas72_one < 1e-3, (
        f"CAS72 at n_mod=2 ({cas72_two:.4f}) expected to be 2x CAS72 at "
        f"n_mod=1 ({cas72_one:.4f}); ratio = {cas72_two / cas72_one:.4f}"
    )


def test_sensitivity_and_batch_numpy_mode():
    """Under numpy, sensitivity uses FD and batch_lcoe loops — both must work."""
    if os.environ.get("COSTINGFE_BACKEND") != "numpy":
        import pytest

        pytest.skip("numpy-mode-only behavior check")
    from costingfe._backend import HAS_JAX

    assert HAS_JAX is False
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    result = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    params = result.params
    sens = model.sensitivity(params)
    assert set(sens) == {"engineering", "financial", "costing"}
    key = next(iter(sens["engineering"]))  # a varying continuous lever
    base = float(params[key])
    out = model.batch_lcoe({key: [base, base * 1.01]}, params)
    assert len(out) == 2 and all(isinstance(v, float) for v in out)


def test_mirror_sizing_still_gated():
    m = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
    with pytest.raises(NotImplementedError, match="are not available in this release"):
        m.forward(
            net_electric_mw=400, availability=0.87, lifetime_yr=40, size_from_power=True
        )


def test_tokamak_sizing_no_longer_gated():
    # Tokamak size_from_power must reach the solver (no NotImplementedError).
    # It may still raise SizingInfeasible or succeed, but NOT the release gate.
    m = CostModel(ConfinementConcept.TOKAMAK, Fuel.DT)
    try:
        m.forward(
            net_electric_mw=500, availability=0.87, lifetime_yr=40, size_from_power=True
        )
    except NotImplementedError:
        pytest.fail("tokamak size_from_power must not hit the release gate")
    except Exception:
        pass  # SizingInfeasible / other solver outcomes are acceptable here


def test_n_mod_for_target_basic():
    assert _n_mod_for_target(100.0, 250.0) == 1  # fits one unit
    assert _n_mod_for_target(500.0, 250.0) == 2  # ceil(500/250)
    assert _n_mod_for_target(501.0, 250.0) == 3  # ceil(501/250)


def test_n_mod_for_target_infeasible_unit():
    with pytest.raises(SizingInfeasible):
        _n_mod_for_target(100.0, 0.0)  # a unit that delivers no net power


@pytest.mark.slow
def test_tokamak_big_target_bumps_n_mod_instead_of_raising():
    # A target above one unit's R0_max capacity (about 6326 MW for the DT
    # default at R0_max=12 m, see test_tokamak_sizing.py) returns
    # solved_n_mod > 1, not SizingInfeasible.
    m = CostModel(ConfinementConcept.TOKAMAK, Fuel.DT)
    r = m.forward(
        net_electric_mw=8000,
        availability=0.87,
        lifetime_yr=40,
        size_from_power=True,
    )
    assert r.solved_n_mod is not None and r.solved_n_mod > 1


@pytest.mark.slow
def test_tokamak_pinned_n_mod_rejected_in_sizing_mode():
    m = CostModel(ConfinementConcept.TOKAMAK, Fuel.DT)
    with pytest.raises(ValueError, match="n_mod cannot be pinned"):
        m.forward(
            net_electric_mw=500,
            availability=0.87,
            lifetime_yr=40,
            size_from_power=True,
            n_mod=2,
        )
