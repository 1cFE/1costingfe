"""IFE/MIF chamber sizing: R_fw = r_ref * sqrt(yield / (yield_ref * f_wall)).

Verifies the GEM/HAPL dry-wall carry-over and the wall_improvement_factor
behaviour committed in docs/plans/2026-07-07-target-yield-sizing-design.md (D2):
dry wall reproduces GEM, R scales as sqrt(yield), and a thick-liquid factor
pulls the naive ~22 m dry-wall chamber at 1.8 GJ down to a HYLIFE-class ~3 m.
"""

import math
import warnings

import pytest

from costingfe import CostModel, Fuel
from costingfe.layers.geometry import chamber_radius_m
from costingfe.types import ConfinementConcept, PowerCycle, WallType

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


# ---- Linear pulsed concepts: cylindrical chamber geometry ----


def _geom(concept, chamber_length=0.0, plasma_t=0.5):
    from costingfe.layers.geometry import RadialBuild, compute_geometry

    rb = RadialBuild(
        R0=0.0,
        plasma_t=plasma_t,
        vacuum_t=0.0,
        blanket_t=0.05,
        ht_shield_t=0.05,
        structure_t=0.10,
        vessel_t=0.10,
        chamber_length=chamber_length,
    )
    return compute_geometry(rb, concept)


def test_linear_pulsed_concepts_get_a_cylinder_not_a_sphere():
    """A 10 m linear FRC module has a 10 m chamber, not a 0.5 m ball.

    The concept YAML describes each module as "a linear machine ~10m long
    with ~0.5m plasma radius"; the sphere branch gives it 3.1 m^2 of wall
    where the cylinder gives 31.4 m^2.
    """
    g = _geom(ConfinementConcept.PULSED_FRC, chamber_length=10.0)
    assert g.firstwall_area == pytest.approx(2 * math.pi * 0.5 * 10.0, rel=1e-9)


def test_length_equal_to_diameter_reproduces_the_sphere_area():
    """The area-preserving default: a cylinder of L = 2r has the same lateral
    area as the sphere of radius r it replaces, so concepts with no sourced
    length join the cylinder branch without their wall area moving."""
    r = 1.0
    sphere = 4 * math.pi * r**2
    g = _geom(ConfinementConcept.DENSE_PLASMA_FOCUS, chamber_length=2 * r, plasma_t=r)
    assert g.firstwall_area == pytest.approx(sphere, rel=1e-9)


def test_spherical_pulsed_concepts_stay_spherical():
    """Chamber-class IFE concepts keep the sphere: their vessel really is one."""
    r = 4.0
    g = _geom(ConfinementConcept.LASER_IFE, chamber_length=99.0, plasma_t=r)
    assert g.firstwall_area == pytest.approx(4 * math.pi * r**2, rel=1e-9)


def test_pulsed_concept_accepts_chamber_length():
    """chamber_length must reach the pulsed family, not just steady-state."""
    m = CostModel(
        concept=ConfinementConcept.PULSED_FRC,
        fuel=Fuel.DHE3,
        power_cycle=PowerCycle.BRAYTON_SCO2,
    )
    r = m.forward(
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
        blanket_form="none",
        blanket_fill="none",
        mn=1.0,
        chamber_length=20.0,
        size_from_power=True,
    )
    assert float(r.costs.lcoe) > 0.0


def _frc(**ovr):
    m = CostModel(
        concept=ConfinementConcept.PULSED_FRC,
        fuel=Fuel.DHE3,
        power_cycle=PowerCycle.BRAYTON_SCO2,
    )
    kw = dict(
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
        blanket_form="none",
        blanket_fill="none",
        mn=1.0,
        dhe3_dd_frac=0.314,
        f_rad=0.163,
        size_from_power=True,
    )
    kw.update(ovr)
    return m.forward(**kw)


def test_wall_flux_warning_fires_when_rep_rate_overruns_the_wall_class():
    """Rep rate must not be free: at 10 Hz the panel wall is 5x over limit.

    The declared-value audit cannot catch this, because nothing about the
    declaration changes when the machine is asked to run harder.
    """
    with pytest.warns(UserWarning, match="first-wall surface flux"):
        _frc(max_f_rep=10.0)


def test_wall_flux_warning_silent_when_the_wall_class_can_take_it():
    """Declaring the divertor-grade class clears the warning, and pays for it."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hhf = _frc(max_f_rep=10.0, fw_class="hhf")
    assert not [w for w in caught if "first-wall surface flux" in str(w.message)]
    panel = _frc(max_f_rep=10.0, fw_class="panel")
    assert float(hhf.cas22_detail["C220101"]) > 5 * float(panel.cas22_detail["C220101"])


def test_wall_flux_quiet_at_the_baseline_shot():
    """The design baseline sits under its declared wall class."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _frc()
    assert not [w for w in caught if "first-wall surface flux" in str(w.message)]
