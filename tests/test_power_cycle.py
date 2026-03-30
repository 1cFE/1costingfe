from costingfe.types import PowerCycle


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
