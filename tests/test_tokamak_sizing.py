import pytest

from costingfe.defaults import MAGNET_TABLE, get_magnet_properties
from costingfe.layers.tokamak import b0_from_radial_build


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


def test_b0_below_bmax_and_grows_with_size():
    # Same magnet, two machine sizes; the larger machine keeps more of B_max.
    thick = dict(blanket_t=0.8, ht_shield_t=0.2, structure_t=0.2, vessel_t=0.2)
    b_small = b0_from_radial_build(R0=3.0, a=1.0, b_max=23.0, **thick)
    b_large = b0_from_radial_build(R0=6.0, a=2.0, b_max=23.0, **thick)
    assert 0.0 < b_small < 23.0
    assert b_small < b_large < 23.0


def test_b0_formula():
    # B0 = B_max * (R0 - a - sum_thick) / R0
    b = b0_from_radial_build(
        R0=4.0,
        a=1.0,
        b_max=20.0,
        blanket_t=0.5,
        ht_shield_t=0.2,
        structure_t=0.2,
        vessel_t=0.1,
    )
    expected = 20.0 * (4.0 - 1.0 - 1.0) / 4.0
    assert b == pytest.approx(expected)
