"""Tests for BlanketForm and BlanketFill enums and their cost wiring."""

from costingfe.types import (
    _BLANKET_FORM_DEFAULT_FILL,
    _BLANKET_FORM_VALID_FILLS,
    BlanketFill,
    BlanketForm,
)


def test_blanket_form_structure_factors():
    """Pin structure_factor values; drift here changes costs across the board."""
    assert BlanketForm.LIQUID_METAL.structure_factor == 1.0
    assert BlanketForm.MOLTEN_SALT.structure_factor == 1.3
    assert BlanketForm.SOLID_BREEDER.structure_factor == 1.2
    assert BlanketForm.NONE.structure_factor == 0.0


def test_blanket_fill_factors():
    """Pin fill_factor values; drift here changes CAS27 across the board."""
    assert BlanketFill.PBLI.fill_factor == 1.0
    assert BlanketFill.LI.fill_factor == 2.0
    assert BlanketFill.FLIBE.fill_factor == 5.0
    assert BlanketFill.BE_CERAMIC.fill_factor == 13.0
    assert BlanketFill.CERAMIC_ONLY.fill_factor == 3.0
    assert BlanketFill.NONE.fill_factor == 0.0


def test_blanket_form_valid_fills():
    """Compatibility table: only physical pairs allowed."""
    assert _BLANKET_FORM_VALID_FILLS[BlanketForm.LIQUID_METAL] == {
        BlanketFill.PBLI,
        BlanketFill.LI,
    }
    assert _BLANKET_FORM_VALID_FILLS[BlanketForm.MOLTEN_SALT] == {BlanketFill.FLIBE}
    assert _BLANKET_FORM_VALID_FILLS[BlanketForm.SOLID_BREEDER] == {
        BlanketFill.BE_CERAMIC,
        BlanketFill.CERAMIC_ONLY,
    }
    assert _BLANKET_FORM_VALID_FILLS[BlanketForm.NONE] == {BlanketFill.NONE}


def test_blanket_form_default_fills():
    """Each form has exactly one default fill."""
    assert _BLANKET_FORM_DEFAULT_FILL[BlanketForm.LIQUID_METAL] == BlanketFill.PBLI
    assert _BLANKET_FORM_DEFAULT_FILL[BlanketForm.MOLTEN_SALT] == BlanketFill.FLIBE
    assert (
        _BLANKET_FORM_DEFAULT_FILL[BlanketForm.SOLID_BREEDER] == BlanketFill.BE_CERAMIC
    )
    assert _BLANKET_FORM_DEFAULT_FILL[BlanketForm.NONE] == BlanketFill.NONE
