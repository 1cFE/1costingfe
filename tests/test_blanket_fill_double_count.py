"""CAS22.01.01 (blanket structure) must not double-count the breeder/multiplier
fill that CAS27 charges separately.

The DT blanket unit cost is a *structure-only* anchor (RAFM steel + W first-wall
armor + fabrication). The breeder and neutron-multiplier material inventory is
priced once, in CAS27, as blanket_vol x vol_frac x density x price. The old 0.60
value folded the PbLi fill mass into the structure account, so it was charged a
second time in CAS27 for every breeding blanket (PbLi, Li, FLiBe, Be-ceramic).
"""

import pytest

from costingfe import ConfinementConcept, CostModel, Fuel
from costingfe.defaults import load_costing_constants


def test_blanket_unit_cost_dt_is_structure_only():
    """0.60 baked in the PbLi fill mass; the structure-only anchor is 0.35."""
    cc = load_costing_constants()
    assert cc.blanket_unit_cost_dt == pytest.approx(0.35)


def test_dt_structure_close_to_dd_structure():
    """DD is defined as 'RAFM steel + coolant, no breeder, no multiplier' — i.e.
    structure only. Once the breeder/multiplier fill is removed from DT and moved
    to CAS27, the DT structure unit cost sits near DD's (a modest W-armor and
    tritium-barrier premium apart), not at the old 2x gap that *was* the fill."""
    cc = load_costing_constants()
    ratio = cc.blanket_unit_cost_dt / cc.blanket_unit_cost_dd
    assert ratio < 1.5, f"DT/DD structure ratio {ratio:.2f} still carries fill premium"


def test_cas27_still_charges_fill_on_top():
    """The fix must not delete CAS27: the fill inventory is still priced there,
    additively, on top of the structure account."""
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    kw = dict(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    pbli = model.forward(**kw, blanket_form="liquid_metal", blanket_fill="pbli")
    assert float(pbli.costs.cas27) > 0.0  # PbLi fill still charged once, in CAS27


def test_non_breeding_unit_costs_unchanged():
    """DD/DHe3/pB11 have no breeder/multiplier fill baked in and no CAS27 fill
    charge, so re-anchoring DT must not touch them (a proportional cut would
    under-count them)."""
    cc = load_costing_constants()
    assert cc.blanket_unit_cost_dd == pytest.approx(0.30)
    assert cc.blanket_unit_cost_dhe3 == pytest.approx(0.08)
    assert cc.blanket_unit_cost_pb11 == pytest.approx(0.05)
