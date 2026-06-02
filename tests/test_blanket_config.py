"""Tests for BlanketForm and BlanketFill enums and their cost wiring."""

import pytest

from costingfe import ConfinementConcept, CostModel, Fuel
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
        BlanketFill.LI2O,
    }
    assert BlanketForm.NONE.valid_fills == {BlanketFill.NONE}


def test_blanket_form_default_fills():
    """Each form has exactly one default fill."""
    assert BlanketForm.LIQUID_METAL.default_fill == BlanketFill.PBLI
    assert BlanketForm.MOLTEN_SALT.default_fill == BlanketFill.FLIBE
    assert BlanketForm.SOLID_BREEDER.default_fill == BlanketFill.BE_CERAMIC
    assert BlanketForm.NONE.default_fill == BlanketFill.NONE


@pytest.mark.parametrize(
    "form, fill, exp_structure_factor, exp_fill_factor",
    [
        ("liquid_metal", "pbli", 1.0, 1.0),
        ("liquid_metal", "li", 1.0, 2.0),
        ("molten_salt", "flibe", 1.3, 5.0),
        ("solid_breeder", "be_ceramic", 1.2, 13.0),
        ("solid_breeder", "ceramic_only", 1.2, 3.0),
    ],
)
def test_dt_tokamak_blanket_cost_scaling(
    form, fill, exp_structure_factor, exp_fill_factor
):
    """Picking a non-default blanket scales CAS22.01 and CAS27 by the
    documented multipliers vs the (liquid_metal, pbli) baseline."""
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    kw = dict(
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
    )
    baseline = model.forward(**kw, blanket_form="liquid_metal", blanket_fill="pbli")
    result = model.forward(**kw, blanket_form=form, blanket_fill=fill)

    c220101_base = float(baseline.cas22_detail["C220101"])
    c220101_new = float(result.cas22_detail["C220101"])
    assert c220101_new == pytest.approx(
        c220101_base * exp_structure_factor, rel=1e-6
    ), f"C220101 multiplier wrong for {form}/{fill}"

    cas27_base = float(baseline.costs.cas27)
    cas27_new = float(result.costs.cas27)
    assert cas27_new == pytest.approx(cas27_base * exp_fill_factor, rel=1e-6), (
        f"CAS27 multiplier wrong for {form}/{fill}"
    )
