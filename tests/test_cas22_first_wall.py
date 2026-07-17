"""Tests for the aneutronic first-wall pricing in C220101 (blanket_form NONE).

With no blanket, the plasma-facing wall is priced as surface hardware:
C220101 = firstwall_area x fw_unit_cost[fw_class], with two discrete hardware
classes (panel / hhf) and a q_surface_max-vs-class audit warning.
"""

import math
import warnings

import pytest

from costingfe import ConfinementConcept, CostModel, Fuel
from costingfe.defaults import load_costing_constants

_KW = dict(
    net_electric_mw=400.0,
    availability=0.85,
    lifetime_yr=40,
    n_mod=1,
    construction_time_yr=5.0,
    interest_rate=0.07,
    inflation_rate=0.02,
    noak=True,
)


def test_pb11_mirror_wall_priced_by_area():
    """p-B11 mirror (blanket_form none): C220101 = cylinder wall area x panel $/m^2."""
    m = CostModel(ConfinementConcept.MIRROR, Fuel.PB11)
    r = m.forward(**_KW)
    cc = load_costing_constants()
    # Cylinder first wall at vacuum_or = plasma_t + vacuum_t, YAML defaults
    # (1.5 + 0.10) over chamber_length = 20 m.
    area = 2 * math.pi * 1.6 * 20.0
    expected = area * cc.fw_unit_cost["panel"]
    assert float(r.cas22_detail["C220101"]) == pytest.approx(expected, rel=1e-6)


def test_hhf_class_prices_at_class_ratio():
    """Switching fw_class panel -> hhf rescales C220101 by the unit-cost ratio."""
    m = CostModel(ConfinementConcept.MIRROR, Fuel.PB11)
    r_panel = m.forward(**_KW)
    r_hhf = m.forward(fw_class="hhf", **_KW)
    cc = load_costing_constants()
    ratio = cc.fw_unit_cost["hhf"] / cc.fw_unit_cost["panel"]
    assert float(r_hhf.cas22_detail["C220101"]) == pytest.approx(
        float(r_panel.cas22_detail["C220101"]) * ratio, rel=1e-6
    )


def test_blanketed_machine_ignores_fw_class():
    """D-T mirror keeps the blanket-structure pricing; fw_class is inert."""
    m = CostModel(ConfinementConcept.MIRROR, Fuel.DT)
    r_a = m.forward(**_KW)
    r_b = m.forward(fw_class="hhf", **_KW)
    assert float(r_a.cas22_detail["C220101"]) == float(r_b.cas22_detail["C220101"])
    assert float(r_a.cas22_detail["C220101"]) > 0.0


def test_q_cap_above_class_limit_warns():
    """q_surface_max beyond the declared class qualification limit warns."""
    m = CostModel(ConfinementConcept.MIRROR, Fuel.PB11)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.forward(q_surface_max=10.0, **_KW)
    assert any("first-wall class qualification limit" in str(w.message) for w in caught)


def test_q_cap_within_class_limit_no_warning():
    """hhf class at q_surface_max = 10 is within qualification: no audit warning."""
    m = CostModel(ConfinementConcept.MIRROR, Fuel.PB11)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.forward(q_surface_max=10.0, fw_class="hhf", **_KW)
    assert not any(
        "first-wall class qualification limit" in str(w.message) for w in caught
    )


def test_dhe3_wall_priced_too():
    """D-He3 (also blanket_form none via fuel normalization) pays for its wall."""
    m = CostModel(ConfinementConcept.MIRROR, Fuel.DHE3)
    r = m.forward(**_KW)
    assert float(r.cas22_detail["C220101"]) > 0.0
