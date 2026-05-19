"""Tests for BlanketForm and BlanketFill enums and their cost wiring."""

from costingfe.types import (
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
    assert BlanketForm.LIQUID_METAL.valid_fills == {
        BlanketFill.PBLI,
        BlanketFill.LI,
    }
    assert BlanketForm.MOLTEN_SALT.valid_fills == {BlanketFill.FLIBE}
    assert BlanketForm.SOLID_BREEDER.valid_fills == {
        BlanketFill.BE_CERAMIC,
        BlanketFill.CERAMIC_ONLY,
    }
    assert BlanketForm.NONE.valid_fills == {BlanketFill.NONE}


def test_blanket_form_default_fills():
    """Each form has exactly one default fill."""
    assert BlanketForm.LIQUID_METAL.default_fill == BlanketFill.PBLI
    assert BlanketForm.MOLTEN_SALT.default_fill == BlanketFill.FLIBE
    assert BlanketForm.SOLID_BREEDER.default_fill == BlanketFill.BE_CERAMIC
    assert BlanketForm.NONE.default_fill == BlanketFill.NONE
