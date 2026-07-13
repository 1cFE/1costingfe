"""First-principles pulsed-IFE yield from burn physics: Y = phi * m * e_DT, with
phi = the concept's sourced burn_fraction, m = fuel_mass_mg_per_mj * e_driver.
Company sets the driver; the (literature-ordered) burn-up fraction sets the gain.
"""

import pytest

from costingfe.layers.physics import physics_yield_mj

LOADING, E_DT = 1.1, 3.4e5  # mg DT / MJ delivered; DT Q-value MJ/g


def test_yield_equals_phi_m_edt():
    # MagLIF: burn_fraction 0.15 at 8.3 MJ.
    y = physics_yield_mj(8.3, 0.15, LOADING, E_DT)
    assert y == pytest.approx(0.15 * (LOADING * 8.3 * 1e-3) * E_DT)


def test_gain_is_phi_times_loading_times_edt():
    # gain = Y/e_driver = phi * loading * e_DT (constant per concept).
    e = 8.0
    for phi, expect in [
        (0.25, 0.25 * LOADING * E_DT * 1e-3),  # laser ~94x
        (0.15, 0.15 * LOADING * E_DT * 1e-3),  # MagLIF ~56x
        (0.10, 0.10 * LOADING * E_DT * 1e-3),
    ]:  # Z-pinch ~37x
        assert physics_yield_mj(e, phi, LOADING, E_DT) / e == pytest.approx(expect)


def test_burn_fraction_ordering_gives_gain_ordering():
    # Higher burn-up (laser) -> higher gain per driver joule than magnetic drive.
    las = physics_yield_mj(8.0, 0.25, LOADING, E_DT)
    mag = physics_yield_mj(8.0, 0.15, LOADING, E_DT)
    zp = physics_yield_mj(8.0, 0.10, LOADING, E_DT)
    assert las > mag > zp
    assert las / mag == pytest.approx(0.25 / 0.15)


def test_laser_gain_is_viable():
    # The bug we fixed: laser gain must exceed the ~28x needed to be net-positive.
    gain = physics_yield_mj(8.0, 0.25, LOADING, E_DT) / 8.0
    assert gain > 28.0  # ~94x -> viable


def test_scales_linearly_with_driver_energy():
    lo = physics_yield_mj(10.0, 0.15, LOADING, E_DT)
    hi = physics_yield_mj(20.0, 0.15, LOADING, E_DT)
    assert hi / lo == pytest.approx(2.0)  # m ~ e_driver -> Y ~ e_driver


def test_coupling_scales_yield_linearly():
    # Drive-mode coupling multiplies assembled fuel mass -> yield/gain linear in it.
    full = physics_yield_mj(8.0, 0.25, LOADING, E_DT, 1.0)
    half = physics_yield_mj(8.0, 0.25, LOADING, E_DT, 0.5)
    assert half == pytest.approx(0.5 * full)


def test_drive_mode_gain_ordering_direct_gt_indirect():
    # Same burn_fraction (same compressed-fuel burn); coupling separates the
    # drive modes: direct (1.0) > hybrid (0.75) > indirect (0.5).
    e = 8.0
    direct = physics_yield_mj(e, 0.25, LOADING, E_DT, 1.0) / e
    hybrid = physics_yield_mj(e, 0.25, LOADING, E_DT, 0.75) / e
    indirect = physics_yield_mj(e, 0.25, LOADING, E_DT, 0.5) / e
    assert direct > hybrid > indirect
    assert indirect / direct == pytest.approx(0.5)  # ~2x penalty for hohlraum


RHO_REF, E_RHO_REF, HB = 2.0, 2.5, 6.0  # laser-direct DT anchor


def test_gain_curve_reproduces_anchor():
    # At the anchor energy phi = rhoR/(rhoR+H_B) = 2/8 = 0.25 = laser burn_fraction,
    # so gain matches the flat model exactly there (burn_fraction arg ignored).
    y = physics_yield_mj(2.5, 0.99, LOADING, E_DT, 1.0, RHO_REF, E_RHO_REF, HB)
    assert y / 2.5 == pytest.approx(0.25 * LOADING * E_DT * 1e-3, rel=1e-3)


def test_gain_curve_rises_then_saturates():
    def g(e):
        return physics_yield_mj(e, 0.25, LOADING, E_DT, 1.0, RHO_REF, E_RHO_REF, HB) / e

    assert g(0.5) < g(2.5) < g(10) < g(73)  # gain rises with E
    ceiling = LOADING * E_DT * 1e-3  # phi -> 1 hard cap
    assert 0.95 * ceiling < g(1e6) < ceiling  # saturates below the ceiling


def test_gain_curve_off_uses_fixed_burn_fraction():
    # rhoR_ref = 0 -> flat burn_fraction, identical to the pre-curve behaviour.
    y = physics_yield_mj(8.0, 0.25, LOADING, E_DT, 1.0, 0.0, 0.0, HB)
    assert y == pytest.approx(0.25 * (LOADING * 8.0 * 1e-3) * E_DT)


def test_gain_curve_keeps_sub_mj_laser_viable():
    # The bug the old robust-burn law caused: sub-100 MJ laser must stay viable.
    g = physics_yield_mj(0.5, 0.25, LOADING, E_DT, 1.0, RHO_REF, E_RHO_REF, HB) / 0.5
    assert g > 28.0  # ~61x -> net-positive


def test_yield_is_unbounded_no_driver_ceiling():
    # No engineering ceiling: yield stays linear far past any nominal e_max.
    # Driver capital is linear $/J, so a bigger driver just costs proportionally
    # more -- the scale penalty lives in the chamber + target cost, not a clip.
    y_small = physics_yield_mj(10.0, 0.15, LOADING, E_DT)
    y_big = physics_yield_mj(500.0, 0.15, LOADING, E_DT)
    assert y_big / y_small == pytest.approx(50.0)
