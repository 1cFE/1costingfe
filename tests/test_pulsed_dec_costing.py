import warnings

import pytest

from costingfe import CostModel, Fuel
from costingfe.types import ConfinementConcept, PulsedConversion


def test_dec_c220107_uses_joule_basis():
    """C220107 should equal c_cap_allin * e_stored_mj for INDUCTIVE_DEC."""
    model = CostModel(
        concept=ConfinementConcept.MAG_TARGET,
        fuel=Fuel.DHE3,
        pulsed_conversion=PulsedConversion.INDUCTIVE_DEC,
    )
    result = model.forward(
        net_electric_mw=50.0,
        availability=0.85,
        lifetime_yr=30,
        q_eng=5.0,
        f_rep=1.0,
        eta_pin=0.95,
        eta_dec=0.85,
        eta_th=0.0,
        mn=1.0,
        p_cryo=0.0,
        p_target=0.0,
    )
    c220107 = float(result.cas22_detail["C220107"])
    e_stored = float(result.power_table.e_stored_mj)
    # $/J basis: c220107 = c_cap_allin_per_joule * e_stored_mj
    expected = 0.5 * e_stored
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
        q_eng=5.0,
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
        q_eng=5.0,
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
    # ZPINCH runs thermal conversion and now sizes on the target-yield axis;
    # a thermal (non-DEC) pulsed plant still carries a nonzero CAS23.
    model = CostModel(concept=ConfinementConcept.ZPINCH, fuel=Fuel.DT)
    result = model.forward(
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
        eta_pin=0.15,
        size_from_power=True,
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
        q_eng=5.0,
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


def _inductive_dec_result(**overrides):
    """Faithful INDUCTIVE_DEC run: MAG_TARGET at its own pulsed parameters."""
    model = CostModel(
        concept=ConfinementConcept.MAG_TARGET,
        fuel=Fuel.DHE3,
        pulsed_conversion=PulsedConversion.INDUCTIVE_DEC,
    )
    kw = dict(
        net_electric_mw=50.0,
        availability=0.85,
        lifetime_yr=30,
        q_eng=5.0,
        f_rep=1.0,
        eta_pin=0.95,
        eta_dec=0.85,
        eta_th=0.0,
        mn=1.0,
        p_cryo=0.0,
        p_target=0.0,
    )
    kw.update(overrides)
    return model.forward(**kw)


def test_dec_staged_recovery_factor_scales_the_capacitance_markup():
    """The staged-storage credit applies to delta_cap only.

    delta_cap is the sole gain-dependent term in C220109; the switch,
    controls, and inverter markups are untouched, so halving the factor must
    reduce C220109 by less than half.
    """
    full = float(_inductive_dec_result().cas22_detail["C220109"])
    half = float(
        _inductive_dec_result(dec_staged_recovery_factor=0.5).cas22_detail["C220109"]
    )
    assert half < full
    assert half > 0.5 * full


def test_dec_staged_recovery_factor_defaults_to_no_credit():
    """Default 1.0 means no reduction is assumed without a sourced figure."""
    default = float(_inductive_dec_result().cas22_detail["C220109"])
    explicit = float(
        _inductive_dec_result(dec_staged_recovery_factor=1.0).cas22_detail["C220109"]
    )
    assert default == pytest.approx(explicit, rel=1e-12)


def test_short_cap_life_warns_with_replacement_count():
    """A bank that cannot survive the plant must not be silently levelized."""
    with pytest.warns(UserWarning, match="capacitor bank is replaced"):
        _inductive_dec_result(cap_shot_lifetime=1.0e7)


def test_cap_life_beyond_plant_shot_count_does_not_warn():
    """No warning once the bank outlives the plant."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _inductive_dec_result(cap_shot_lifetime=1.0e12)
    assert not [w for w in caught if "capacitor bank is replaced" in str(w.message)]
