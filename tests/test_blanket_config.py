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
    "form, fill, exp_structure_factor",
    [
        ("liquid_metal", "pbli", 1.0),
        ("liquid_metal", "li", 1.0),
        ("molten_salt", "flibe", 1.3),
        ("solid_breeder", "be_ceramic", 1.2),
        ("solid_breeder", "ceramic_only", 1.2),
    ],
)
def test_dt_tokamak_blanket_structure_scaling(form, fill, exp_structure_factor):
    """Picking a non-default blanket scales CAS22.01 (structure) by the
    documented structure_factor vs the (liquid_metal, pbli) baseline. CAS27
    (the fill inventory) is volume-based — see test_cas27_volumetric_all_fills."""
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    kw = dict(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    baseline = model.forward(**kw, blanket_form="liquid_metal", blanket_fill="pbli")
    result = model.forward(**kw, blanket_form=form, blanket_fill=fill)

    c220101_base = float(baseline.cas22_detail["C220101"])
    c220101_new = float(result.cas22_detail["C220101"])
    assert c220101_new == pytest.approx(
        c220101_base * exp_structure_factor, rel=1e-6
    ), f"C220101 multiplier wrong for {form}/{fill}"


@pytest.mark.parametrize(
    "fill", ["pbli", "li", "flibe", "be_ceramic", "ceramic_only", "li2o", "none"]
)
def test_cas27_volumetric_all_fills(fill):
    """Every fill is costed volumetrically: blanket_vol x vol_frac x density x
    price. Independent of net power; 'none' is zero."""
    from costingfe.defaults import load_costing_constants
    from costingfe.layers.costs import cas27_special_materials

    cc = load_costing_constants()
    bv = 643.5
    m = cc.cas27_fill_materials[fill]
    expected = bv * m["vol_frac"] * m["density"] * m["price"] / 1e6
    got = float(cas27_special_materials(cc, BlanketFill(fill), bv))
    assert got == pytest.approx(expected, rel=1e-9)
    if fill == "none":
        assert got == 0.0
    else:
        assert got > 0.0
        # doubles with blanket volume; no dependence on net power at all.
        got2 = float(cas27_special_materials(cc, BlanketFill(fill), 2 * bv))
        assert got2 == pytest.approx(2 * got, rel=1e-9)


def test_cas27_flibe_exceeds_pbli_and_scales_with_blanket():
    """Through the full model: a thicker FLiBe blanket costs more CAS27, and the
    volumetric FLiBe inventory far exceeds the PbLi baseline."""
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    kw = dict(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    thin = model.forward(
        **kw, blanket_form="molten_salt", blanket_fill="flibe", blanket_t=0.5
    )
    thick = model.forward(
        **kw, blanket_form="molten_salt", blanket_fill="flibe", blanket_t=1.0
    )
    assert float(thick.costs.cas27) > float(thin.costs.cas27)
    pbli = model.forward(**kw)  # liquid_metal/pbli default
    assert float(thin.costs.cas27) > 3 * float(pbli.costs.cas27)
