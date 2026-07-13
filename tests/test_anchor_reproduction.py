"""Target-yield axis must pass through each concept's sourced design point.

The gain formula is Y = phi * m * e_DT with m = fuel_mass_mg_per_mj * e_driver.
`fuel_mass_mg_per_mj` (fuel assembled per delivered MJ) is the per-concept
calibration knob: it differs between laser ablation and magnetic-liner drive, so
each concept sets it so the physics yield at its cited driver energy reproduces
its cited/sourced yield. These are the assertions that catch a concept borrowing
a wrong-drive loading constant (the failure that let MagLIF overshoot by 22%).
"""

import pytest

from costingfe.defaults import load_engineering_defaults
from costingfe.layers.physics import physics_yield_mj

E_DT_DEFAULT = 3.4e5  # MJ/g DT Q-value (CostingConstants default)
LOADING_DEFAULT = 1.1  # mg/MJ (CostingConstants default, laser-fit)


def _physics_yield_at_cited_driver(yaml_name):
    """Physics yield evaluated at the concept's own cited driver energy, using
    only that concept's YAML params (faithful inputs)."""
    p = load_engineering_defaults(yaml_name)
    return (
        physics_yield_mj(
            p["e_driver_mj"],
            p["burn_fraction"],
            p.get("fuel_mass_mg_per_mj", LOADING_DEFAULT),
            p.get("e_fuel_mj_per_g", E_DT_DEFAULT),
            p.get("coupling_frac", 1.0),
            p.get("rhoR_ref_g_cm2", 0.0),
            p.get("e_rhoR_ref_mj", 0.0),
            p.get("gain_hb_g_cm2", 6.0),
        ),
        p,
    )


def test_maglif_target_yield_reproduces_cited_shot():
    # Pacific Fusion AMPS Al-DS: 8.4 MJ delivered -> 380 MJ at phi = 0.10.
    y, p = _physics_yield_at_cited_driver("pulsed_maglif")
    assert y == pytest.approx(p["yield_per_shot_mj"], rel=0.02)


def test_zpinch_target_yield_reproduces_cited_shot():
    # Sandia Z-IFE: ~40 MJ to the pinch -> 3 GJ at the dynamic-hohlraum capsule.
    y, p = _physics_yield_at_cited_driver("pulsed_zpinch")
    assert y == pytest.approx(p["yield_per_shot_mj"], rel=0.02)


def test_laser_gain_curve_meets_physics_anchor_not_vendor_claim():
    # Laser is the exception: the rhoR curve sets gain from physics, so at the
    # anchor energy it reproduces phi = burn_fraction (~94x), deliberately below
    # the vendor's cited ~100x -- physics sets the gain, not the datasheet.
    p = load_engineering_defaults("pulsed_laser_ife")
    e_ref = p["e_rhoR_ref_mj"]
    gain = (
        physics_yield_mj(
            e_ref,
            p["burn_fraction"],
            p.get("fuel_mass_mg_per_mj", LOADING_DEFAULT),
            p.get("e_fuel_mj_per_g", E_DT_DEFAULT),
            1.0,
            p["rhoR_ref_g_cm2"],
            p["e_rhoR_ref_mj"],
            p.get("gain_hb_g_cm2", 6.0),
        )
        / e_ref
    )
    phi_ref = p["rhoR_ref_g_cm2"] / (p["rhoR_ref_g_cm2"] + p.get("gain_hb_g_cm2", 6.0))
    expected = (
        phi_ref * p.get("fuel_mass_mg_per_mj", LOADING_DEFAULT) * E_DT_DEFAULT * 1e-3
    )
    assert gain == pytest.approx(expected, rel=1e-3)
