"""Aneutronic fuels (D-He3, p-B11) need no breeding blanket or neutron
multiplier. A concept whose YAML defaults are D-T-flavored (e.g. the mirror's
PbLi breeding blanket, mn=1.1) must not silently carry that hardware when run
with an aneutronic fuel: the blanket form/fill normalize to `none` and mn to
1.0, so CAS27 (breeder inventory) is zero. This mirrors the existing p_trit
auto-zeroing for non-DT fuels, and respects explicit user overrides.
"""

import pytest

from costingfe import ConfinementConcept, CostModel, Fuel

_KW = dict(net_electric_mw=200.0, availability=0.85, lifetime_yr=30)


@pytest.mark.parametrize("fuel", [Fuel.DHE3, Fuel.PB11])
def test_aneutronic_normalizes_dt_config_blanket(fuel):
    """MIRROR's YAML default is a PbLi breeding blanket (mn=1.1). Run with an
    aneutronic fuel, it must drop to no blanket and mn=1.0, zeroing CAS27."""
    r = CostModel(concept=ConfinementConcept.MIRROR, fuel=fuel).forward(**_KW)
    assert r.params["blanket_form"] == "none"
    assert r.params["blanket_fill"] == "none"
    assert float(r.params["mn"]) == pytest.approx(1.0)
    assert float(r.costs.cas27) == 0.0


@pytest.mark.parametrize("fuel", [Fuel.DHE3, Fuel.PB11])
def test_explicit_blanket_override_respected(fuel):
    """An explicit blanket override is honored (the normalization only touches
    unspecified concept defaults, like the p_trit guard)."""
    r = CostModel(concept=ConfinementConcept.MIRROR, fuel=fuel).forward(
        blanket_form="liquid_metal", blanket_fill="pbli", **_KW
    )
    assert r.params["blanket_fill"] == "pbli"
    assert float(r.costs.cas27) > 0.0


def test_dd_uses_water_moderator_not_breeder():
    """D-D breeds its own tritium (D+D->T+p) so it needs no lithium breeder — but
    it IS neutronic, so it needs a moderating energy-capture blanket, not an empty
    shell. It normalizes to a water-cooled blanket: the steel energy-capture
    structure stays (C220101 > 0), mn drops to 1.0 (no Be multiplier), and the
    breeder inventory is replaced by cheap water so CAS27 is near zero."""
    pbli = CostModel(concept=ConfinementConcept.MIRROR, fuel=Fuel.DD).forward(
        blanket_form="liquid_metal", blanket_fill="pbli", **_KW
    )
    r = CostModel(concept=ConfinementConcept.MIRROR, fuel=Fuel.DD).forward(**_KW)
    assert r.params["blanket_form"] == "water_cooled"
    assert r.params["blanket_fill"] == "water"
    assert float(r.params["mn"]) == pytest.approx(1.0)
    assert float(r.cas22_detail["C220101"]) > 0.0  # energy-capture structure kept
    # water inventory is far cheaper than the PbLi breeder it replaces
    assert float(r.costs.cas27) < float(pbli.costs.cas27)


def test_dt_keeps_breeding_blanket():
    """D-T is unaffected: the breeding blanket and multiplier stay."""
    r = CostModel(concept=ConfinementConcept.MIRROR, fuel=Fuel.DT).forward(**_KW)
    assert r.params["blanket_fill"] == "pbli"
    assert float(r.params["mn"]) > 1.0
    assert float(r.costs.cas27) > 0.0
