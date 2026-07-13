"""IFE/MIF chamber sizing: R_fw = r_ref * sqrt(yield / (yield_ref * f_wall)).

Verifies the GEM/HAPL dry-wall carry-over and the wall_improvement_factor
behaviour committed in docs/plans/2026-07-07-target-yield-sizing-design.md (D2):
dry wall reproduces GEM, R scales as sqrt(yield), and a thick-liquid factor
pulls the naive ~22 m dry-wall chamber at 1.8 GJ down to a HYLIFE-class ~3 m.
"""

import math

import pytest

from costingfe.layers.geometry import chamber_radius_m
from costingfe.types import WallType

R_REF = 6.5
Y_REF = 150.0


def test_reproduces_gem_reference_point():
    # 150 MJ, dry wall -> exactly the GEM/HAPL 6.5 m reference.
    assert chamber_radius_m(150.0, R_REF, Y_REF, 1.0) == pytest.approx(6.5)


def test_radius_scales_as_sqrt_yield():
    r1 = chamber_radius_m(150.0, R_REF, Y_REF, 1.0)
    r4 = chamber_radius_m(600.0, R_REF, Y_REF, 1.0)  # 4x yield -> 2x radius
    assert r4 / r1 == pytest.approx(2.0)


def test_dry_wall_gj_chamber_is_huge():
    # 1.8 GJ dry wall -> ~22 m first-wall radius (why high-yield needs a liquid wall).
    r = chamber_radius_m(1800.0, R_REF, Y_REF, 1.0)
    assert r == pytest.approx(6.5 * math.sqrt(1800.0 / 150.0), rel=1e-9)
    assert 21.0 < r < 24.0


def test_thick_liquid_factor_shrinks_to_hylife_class():
    # f_wall=50 pulls the 1.8 GJ chamber down to a HYLIFE-II-class ~3 m.
    r = chamber_radius_m(1800.0, R_REF, Y_REF, 50.0)
    assert 2.5 < r < 3.5


def test_improvement_factor_is_inverse_sqrt():
    r1 = chamber_radius_m(600.0, R_REF, Y_REF, 1.0)
    r4 = chamber_radius_m(600.0, R_REF, Y_REF, 4.0)  # 4x tolerance -> half radius
    assert r4 / r1 == pytest.approx(0.5)


def test_walltype_enum_values():
    assert {w.value for w in WallType} == {"dry", "advanced_dry", "thick_liquid"}


# -- neutron wall-loading floor: R = max(R_fluence, R_power) ------------------


def test_floor_disabled_by_default():
    # No p_neutron/limit -> pure fluence behaviour (legacy callers unchanged).
    r = chamber_radius_m(150.0, R_REF, Y_REF, 1.0)
    assert r == chamber_radius_m(150.0, R_REF, Y_REF, 1.0, 0.0, 0.0)


def test_power_density_radius_formula():
    # Fluence radius tiny (huge f_wall) so the power floor binds; check R_power.
    p_n, gamma = 1600.0, 20.0
    r = chamber_radius_m(200.0, R_REF, Y_REF, 1e6, p_n, gamma)
    assert r == pytest.approx(math.sqrt(p_n / (4 * math.pi * gamma)))


def test_floor_binds_for_low_yield_high_rep():
    # Low per-shot yield (small fluence radius) but 1600 MW of neutrons: the
    # fluence term alone would give an unphysically tiny chamber; the floor lifts
    # it. This is the high-rep/low-yield free pass the floor is meant to close.
    y, p_n, gamma = 200.0, 1600.0, 20.0
    r_fluence = chamber_radius_m(y, R_REF, Y_REF, 50.0)
    r = chamber_radius_m(y, R_REF, Y_REF, 50.0, p_n, gamma)
    assert r > r_fluence
    assert r == pytest.approx(math.sqrt(p_n / (4 * math.pi * gamma)))


def test_fluence_still_binds_for_high_yield_low_rep():
    # Big single-shot yield: the fluence (survivability) radius dominates and the
    # power floor is slack, so high-yield concepts keep their fluence penalty.
    y, p_n, gamma = 2857.0, 1600.0, 20.0
    r_fluence = chamber_radius_m(y, R_REF, Y_REF, 50.0)
    r = chamber_radius_m(y, R_REF, Y_REF, 50.0, p_n, gamma)
    assert r == pytest.approx(r_fluence)
    assert r > math.sqrt(p_n / (4 * math.pi * gamma))
