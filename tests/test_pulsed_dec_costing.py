from costingfe import CostModel, Fuel
from costingfe.types import ConfinementConcept, PulsedConversion


def test_dec_c220107_uses_joule_basis():
    model = CostModel(
        concept=ConfinementConcept.MAG_TARGET,
        fuel=Fuel.DHE3,
        pulsed_conversion=PulsedConversion.INDUCTIVE_DEC,
    )
    result = model.forward(
        net_electric_mw=50.0,
        availability=0.85,
        lifetime_yr=30,
        e_driver_mj=12.0,
        f_rep=1.0,
        eta_pin=0.95,
        eta_dec=0.85,
        eta_th=0.0,
        mn=1.0,
        p_cryo=0.0,
        p_target=0.0,
    )
    c220107 = result.cas22_detail["C220107"]
    expected = 2.0 * 12.0 / 0.95  # c_cap * e_stored_mj
    assert abs(c220107 - expected) < 0.5, f"Expected ~{expected:.1f}, got {c220107:.1f}"


def test_dec_c220109_populated():
    model = CostModel(
        concept=ConfinementConcept.MAG_TARGET,
        fuel=Fuel.DHE3,
        pulsed_conversion=PulsedConversion.INDUCTIVE_DEC,
    )
    result = model.forward(
        net_electric_mw=50.0,
        availability=0.85,
        lifetime_yr=30,
        e_driver_mj=12.0,
        f_rep=1.0,
        eta_pin=0.95,
        eta_dec=0.85,
        eta_th=0.0,
        mn=1.0,
        p_cryo=0.0,
        p_target=0.0,
    )
    assert result.cas22_detail["C220109"] > 0


def test_dec_cas23_zero_when_no_thermal():
    model = CostModel(
        concept=ConfinementConcept.MAG_TARGET,
        fuel=Fuel.DHE3,
        pulsed_conversion=PulsedConversion.INDUCTIVE_DEC,
    )
    result = model.forward(
        net_electric_mw=50.0,
        availability=0.85,
        lifetime_yr=30,
        e_driver_mj=12.0,
        f_rep=1.0,
        eta_pin=0.95,
        eta_dec=0.85,
        eta_th=0.0,
        mn=1.0,
        p_cryo=0.0,
        p_target=0.0,
    )
    assert result.costs.cas23 == 0.0


def test_thermal_pulsed_cas23_nonzero():
    model = CostModel(concept=ConfinementConcept.ZPINCH, fuel=Fuel.DT)
    result = model.forward(
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
        e_driver_mj=20.0,
        f_rep=0.1,
        eta_pin=0.15,
    )
    assert result.costs.cas23 > 0


def test_dec_no_cost_overrides_needed():
    model = CostModel(
        concept=ConfinementConcept.MAG_TARGET,
        fuel=Fuel.DHE3,
        pulsed_conversion=PulsedConversion.INDUCTIVE_DEC,
    )
    result = model.forward(
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
        n_mod=20,
        e_driver_mj=12.0,
        f_rep=1.0,
        eta_pin=0.95,
        eta_dec=0.85,
        eta_th=0.0,
        mn=1.0,
        p_cryo=0.0,
        p_target=0.0,
    )
    assert len(result.overridden) == 0
    assert result.costs.cas23 == 0.0
    assert result.cas22_detail["C220107"] > 0
    assert result.cas22_detail["C220109"] > 0
