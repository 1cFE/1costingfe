import pytest

from costingfe.defaults import MAGNET_TABLE, get_magnet_properties


def test_magnet_table_has_expected_materials():
    for key in ("rebco_hts", "nb3sn", "nbti", "copper"):
        assert key in MAGNET_TABLE
    for sc_key in ("rebco_hts", "nb3sn", "nbti"):
        assert MAGNET_TABLE[sc_key].recirc_power_factor == 0.0


def test_get_magnet_properties_rebco():
    props = get_magnet_properties("rebco_hts")
    assert props.b_max == pytest.approx(23.0)
    assert props.recirc_power_factor == 0.0
    assert props.cryo_temp_k == pytest.approx(20.0)


def test_get_magnet_properties_copper_has_recirc():
    props = get_magnet_properties("copper")
    assert props.recirc_power_factor > 0.0


def test_get_magnet_properties_unknown_raises():
    with pytest.raises(KeyError):
        get_magnet_properties("unobtanium")
