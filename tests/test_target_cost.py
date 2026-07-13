"""Two-mode per-shot target cost, size-scaled by assembled FUEL MASS (design
doc D4). m_fuel = yield/(burn_fraction*e_fuel); ratio = m_fuel / archetype
reference mass (universal costing constant, not a per-concept knob).

METAL_LINER (MagLIF/Z-pinch): liner metal (~m_fuel) * machining + RTL (flat).
CAPSULE_FAB (laser/heavy-ion): cryo (~m_fuel) + coat (~m_fuel^2/3) + assembly.
"""

import pytest

from costingfe.layers.costs import target_shot_cost
from costingfe.types import TargetCostMode

E_DT = 3.4e5  # MJ/g

# Default calibration (defaults.py): MagLIF $9 @ 46.2 mg, laser $0.50 @ 2.75 mg.
METAL = dict(
    target_liner_cost_ref=3.0,
    target_machining_markup=1.3,
    target_rtl_cost=5.1,
    target_fuel_mass_ref_liner_mg=46.2,
)
CAPS = dict(
    target_cryo_cost_ref=0.25,
    target_coat_cost_ref=0.15,
    target_assembly_cost=0.10,
    target_material_floor=0.0,
    target_fuel_mass_ref_capsule_mg=2.75,
)


def yield_for_mass(m_mg, bf):
    """Yield [MJ] whose assembled fuel mass is m_mg at burn fraction bf."""
    return m_mg * bf * E_DT / 1.0e3


def test_capsule_reproduces_laser_anchor():
    # Reference capsule mass (2.75 mg) -> ratio 1 -> the $0.50 anchor.
    y = yield_for_mass(2.75, 0.25)
    cost, frac = target_shot_cost(TargetCostMode.CAPSULE_FAB, y, 0.25, E_DT, CAPS)
    assert cost == pytest.approx(0.5)  # 0.25 + 0.15 + 0.10
    assert frac == pytest.approx(0.0)  # fabrication-dominated


def test_metal_liner_reproduces_maglif_anchor():
    # Reference liner mass (46.2 mg) -> ratio 1 -> the $9 anchor.
    y = yield_for_mass(46.2, 0.15)
    cost, frac = target_shot_cost(TargetCostMode.METAL_LINER, y, 0.15, E_DT, METAL)
    assert cost == pytest.approx(9.0)  # 3.0*1.3 + 5.1
    assert frac == pytest.approx(3.9 / 9.0)  # material-dominated, ~43%


def test_capsule_scales_with_fuel_mass():
    # Twice the reference fuel mass -> cryo term doubles, coat ~2^(2/3).
    y = yield_for_mass(5.5, 0.25)  # 2x ref capsule mass
    cost, _ = target_shot_cost(TargetCostMode.CAPSULE_FAB, y, 0.25, E_DT, CAPS)
    assert cost == pytest.approx(0.25 * 2 + 0.15 * 2 ** (2 / 3) + 0.10)


def test_lower_burn_fraction_costs_more_at_equal_yield():
    # The reason to size by fuel MASS not yield: same yield, lower burn-up ->
    # more fuel assembled -> bigger/costlier target.
    y = 250.0
    hi_bf, _ = target_shot_cost(TargetCostMode.CAPSULE_FAB, y, 0.25, E_DT, CAPS)
    lo_bf, _ = target_shot_cost(TargetCostMode.CAPSULE_FAB, y, 0.10, E_DT, CAPS)
    assert lo_bf > hi_bf  # 0.10 burn-up needs 2.5x the fuel of 0.25


def test_hohlraum_floor_adds_size_scaled_indirect_premium():
    # Indirect drive: a hohlraum material floor (0.20 at the ref mass) adds a
    # ~2x premium AND scales with fuel mass, unlike the old flat override.
    caps_ind = dict(CAPS, target_material_floor=0.20)
    y_ref = yield_for_mass(2.75, 0.25)  # ref capsule mass -> ratio 1
    direct, _ = target_shot_cost("capsule_fab", y_ref, 0.25, E_DT, CAPS)
    indirect, frac = target_shot_cost("capsule_fab", y_ref, 0.25, E_DT, caps_ind)
    assert direct == pytest.approx(0.50)
    assert indirect == pytest.approx(0.70)  # 0.50 + 0.20 hohlraum at ref
    assert frac == pytest.approx(0.20 / 0.70)  # hohlraum share
    # The hohlraum term scales linearly with fuel mass (2x mass -> 2x floor):
    y2 = yield_for_mass(5.5, 0.25)  # ratio 2
    big_ind, _ = target_shot_cost("capsule_fab", y2, 0.25, E_DT, caps_ind)
    big_dir, _ = target_shot_cost("capsule_fab", y2, 0.25, E_DT, CAPS)
    assert big_ind - big_dir == pytest.approx(0.20 * 2.0)


def test_string_mode_accepted():
    y = yield_for_mass(46.2, 0.15)
    cost, _ = target_shot_cost("metal_liner", y, 0.15, E_DT, METAL)
    assert cost == pytest.approx(9.0)


def test_guard_flags_when_material_dominates():
    # A large material floor (e.g. a gold hohlraum) drives material_frac high.
    caps = dict(CAPS, target_material_floor=10.0)
    y = yield_for_mass(2.75, 0.25)
    _, frac = target_shot_cost(TargetCostMode.CAPSULE_FAB, y, 0.25, E_DT, caps)
    assert frac > 0.8
