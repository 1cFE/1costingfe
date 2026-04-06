from pathlib import Path

import yaml

from costingfe.adapter import FusionTeaInput, run_costing
from costingfe.defaults import POWER_CYCLE_DEFAULTS
from costingfe.types import PowerCycle


def test_power_cycle_defaults_has_all_cycles():
    """Every PowerCycle member has a defaults entry."""
    for cycle in PowerCycle:
        assert cycle in POWER_CYCLE_DEFAULTS, f"Missing defaults for {cycle}"


def test_power_cycle_defaults_keys():
    """Each preset contains exactly eta_th, turbine_per_mw, heat_rej_per_mw."""
    expected_keys = {"eta_th", "turbine_per_mw", "heat_rej_per_mw"}
    for cycle in PowerCycle:
        assert set(POWER_CYCLE_DEFAULTS[cycle].keys()) == expected_keys


def test_rankine_defaults_match_current_values():
    """Rankine preset should match the current CostingConstants defaults."""
    preset = POWER_CYCLE_DEFAULTS[PowerCycle.RANKINE]
    assert preset["eta_th"] == 0.40
    assert preset["turbine_per_mw"] == 0.19764
    assert preset["heat_rej_per_mw"] == 0.03416


def test_brayton_sco2_defaults():
    """sCO2 Brayton preset values."""
    preset = POWER_CYCLE_DEFAULTS[PowerCycle.BRAYTON_SCO2]
    assert preset["eta_th"] == 0.47
    assert preset["turbine_per_mw"] == 0.155
    assert preset["heat_rej_per_mw"] == 0.022


def test_combined_defaults():
    """Combined cycle preset values."""
    preset = POWER_CYCLE_DEFAULTS[PowerCycle.COMBINED]
    assert preset["eta_th"] == 0.53
    assert preset["turbine_per_mw"] == 0.235
    assert preset["heat_rej_per_mw"] == 0.018


def test_power_cycle_enum_members():
    """PowerCycle enum has three members with correct string values."""
    assert PowerCycle.RANKINE.value == "rankine"
    assert PowerCycle.BRAYTON_SCO2.value == "brayton_sco2"
    assert PowerCycle.COMBINED.value == "combined"
    assert len(PowerCycle) == 3


def test_power_cycle_from_string():
    """PowerCycle can be constructed from string value."""
    assert PowerCycle("rankine") == PowerCycle.RANKINE
    assert PowerCycle("brayton_sco2") == PowerCycle.BRAYTON_SCO2
    assert PowerCycle("combined") == PowerCycle.COMBINED


def test_eta_th_not_in_concept_yamls():
    """eta_th should not be in any concept YAML — cycle preset is source of truth."""
    defaults_dir = (
        Path(__file__).parent.parent / "src" / "costingfe" / "data" / "defaults"
    )
    concept_files = [
        "steady_state_tokamak.yaml",
        "steady_state_stellarator.yaml",
        "steady_state_mirror.yaml",
        "pulsed_laser_ife.yaml",
        "pulsed_zpinch.yaml",
        "pulsed_heavy_ion.yaml",
        "pulsed_mag_target.yaml",
        "pulsed_plasma_jet.yaml",
    ]
    for fname in concept_files:
        with open(defaults_dir / fname) as f:
            data = yaml.safe_load(f)
        assert "eta_th" not in data, f"eta_th still in {fname}"


def test_default_cycle_is_rankine():
    """Omitting power_cycle should default to Rankine."""
    from costingfe.model import CostModel
    from costingfe.types import ConfinementConcept, Fuel, PowerCycle

    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    assert model.power_cycle == PowerCycle.RANKINE


def test_rankine_matches_existing_behavior():
    """Rankine cycle should produce identical results to no-cycle baseline."""
    from costingfe.model import CostModel
    from costingfe.types import ConfinementConcept, Fuel, PowerCycle

    model_default = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    model_rankine = CostModel(
        concept=ConfinementConcept.TOKAMAK,
        fuel=Fuel.DT,
        power_cycle=PowerCycle.RANKINE,
    )
    r_default = model_default.forward(
        net_electric_mw=1000.0, availability=0.85, lifetime_yr=30
    )
    r_rankine = model_rankine.forward(
        net_electric_mw=1000.0, availability=0.85, lifetime_yr=30
    )
    assert abs(r_default.costs.lcoe - r_rankine.costs.lcoe) < 0.01


def test_sco2_differs_from_rankine():
    """sCO2 Brayton should produce different (lower) LCOE than Rankine."""
    from costingfe.model import CostModel
    from costingfe.types import ConfinementConcept, Fuel, PowerCycle

    model_rankine = CostModel(
        concept=ConfinementConcept.TOKAMAK,
        fuel=Fuel.DT,
        power_cycle=PowerCycle.RANKINE,
    )
    model_sco2 = CostModel(
        concept=ConfinementConcept.TOKAMAK,
        fuel=Fuel.DT,
        power_cycle=PowerCycle.BRAYTON_SCO2,
    )
    r_rankine = model_rankine.forward(
        net_electric_mw=1000.0, availability=0.85, lifetime_yr=30
    )
    r_sco2 = model_sco2.forward(
        net_electric_mw=1000.0, availability=0.85, lifetime_yr=30
    )
    assert r_sco2.costs.lcoe < r_rankine.costs.lcoe


def test_combined_differs_from_sco2():
    """Combined cycle should produce different (lower) LCOE than sCO2."""
    from costingfe.model import CostModel
    from costingfe.types import ConfinementConcept, Fuel, PowerCycle

    model_sco2 = CostModel(
        concept=ConfinementConcept.TOKAMAK,
        fuel=Fuel.DT,
        power_cycle=PowerCycle.BRAYTON_SCO2,
    )
    model_combined = CostModel(
        concept=ConfinementConcept.TOKAMAK,
        fuel=Fuel.DT,
        power_cycle=PowerCycle.COMBINED,
    )
    r_sco2 = model_sco2.forward(
        net_electric_mw=1000.0, availability=0.85, lifetime_yr=30
    )
    r_combined = model_combined.forward(
        net_electric_mw=1000.0, availability=0.85, lifetime_yr=30
    )
    assert r_combined.costs.lcoe < r_sco2.costs.lcoe


def test_eta_th_override_wins_over_preset():
    """User-supplied eta_th should override the cycle preset."""
    from costingfe.model import CostModel
    from costingfe.types import ConfinementConcept, Fuel, PowerCycle

    model = CostModel(
        concept=ConfinementConcept.TOKAMAK,
        fuel=Fuel.DT,
        power_cycle=PowerCycle.BRAYTON_SCO2,
    )
    result = model.forward(
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
        eta_th=0.40,
    )
    assert abs(result.params["eta_th"] - 0.40) < 1e-6


def test_costing_override_wins_over_preset():
    """User costing_overrides should override the cycle preset coefficients."""
    from costingfe.defaults import load_costing_constants
    from costingfe.model import CostModel
    from costingfe.types import ConfinementConcept, Fuel, PowerCycle

    cc = load_costing_constants()
    custom_turbine = 0.25
    cc = cc.replace(turbine_per_mw=custom_turbine)
    model = CostModel(
        concept=ConfinementConcept.TOKAMAK,
        fuel=Fuel.DT,
        power_cycle=PowerCycle.BRAYTON_SCO2,
        costing_constants=cc,
    )
    result = model.forward(
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
    )
    expected_cas23 = result.power_table.p_et * custom_turbine
    assert abs(result.costs.cas23 - expected_cas23) < 0.1


def test_bop_coefficients_not_in_costing_yaml():
    """turbine_per_mw and heat_rej_per_mw should not be in costing_constants.yaml."""
    defaults_dir = (
        Path(__file__).parent.parent / "src" / "costingfe" / "data" / "defaults"
    )
    with open(defaults_dir / "costing_constants.yaml") as f:
        data = yaml.safe_load(f)
    assert "turbine_per_mw" not in data, (
        "turbine_per_mw still in costing_constants.yaml"
    )
    assert "heat_rej_per_mw" not in data, (
        "heat_rej_per_mw still in costing_constants.yaml"
    )


def test_adapter_with_power_cycle():
    """Adapter should accept power_cycle string and pass to CostModel."""
    inp_rankine = FusionTeaInput(
        concept="tokamak",
        fuel="dt",
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
        power_cycle="rankine",
    )
    inp_sco2 = FusionTeaInput(
        concept="tokamak",
        fuel="dt",
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
        power_cycle="brayton_sco2",
    )
    out_rankine = run_costing(inp_rankine)
    out_sco2 = run_costing(inp_sco2)
    assert out_sco2.lcoe < out_rankine.lcoe


def test_adapter_default_power_cycle():
    """Adapter should default to rankine when power_cycle not provided."""
    inp = FusionTeaInput(
        concept="tokamak",
        fuel="dt",
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
    )
    out = run_costing(inp)
    assert out.lcoe > 0


def test_sensitivity_with_sco2():
    """Sensitivity analysis should work with non-Rankine cycle."""
    from costingfe.model import CostModel
    from costingfe.types import ConfinementConcept, Fuel, PowerCycle

    model = CostModel(
        concept=ConfinementConcept.TOKAMAK,
        fuel=Fuel.DT,
        power_cycle=PowerCycle.BRAYTON_SCO2,
    )
    result = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    sens = model.sensitivity(result.params)
    assert "engineering" in sens
    assert "eta_th" in sens["engineering"]
    # eta_th elasticity should be negative (higher efficiency -> lower LCOE)
    assert sens["engineering"]["eta_th"] < 0
