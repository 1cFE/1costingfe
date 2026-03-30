from pathlib import Path

import yaml

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
        "mfe_tokamak.yaml",
        "mfe_stellarator.yaml",
        "mfe_mirror.yaml",
        "ife_laser_ife.yaml",
        "ife_zpinch.yaml",
        "ife_heavy_ion.yaml",
        "mif_mag_target.yaml",
        "mif_plasma_jet.yaml",
    ]
    for fname in concept_files:
        with open(defaults_dir / fname) as f:
            data = yaml.safe_load(f)
        assert "eta_th" not in data, f"eta_th still in {fname}"


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
