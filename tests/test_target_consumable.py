"""Per-shot target consumable (CAS80) and target-factory gating (C220108).

Concepts that manufacture a fabricated target/liner each shot (laser IFE,
heavy-ion, MagLIF, Z-pinch) carry both a target factory (C220108 capital) and a
per-shot consumable in CAS80. In-situ-formation concepts (plasma-jet, FRC,
theta-pinch, dense-plasma-focus, staged-Z, liquid-liner MTF) carry neither --
by default, without any cost override. Both are driven by the single YAML knob
target_unit_cost.
"""

import pytest

from costingfe import CostModel, Fuel
from costingfe.defaults import load_costing_constants
from costingfe.layers.costs import cas80_fuel
from costingfe.types import ConfinementConcept

CC = load_costing_constants()

_FWD = dict(net_electric_mw=500.0, availability=0.85, lifetime_yr=30)

# Concepts that consume a fabricated target -> target factory + CAS80 consumable
TARGET_CONCEPTS = [
    ConfinementConcept.LASER_IFE,
    ConfinementConcept.HEAVY_ION,
    ConfinementConcept.MAGLIF,
    ConfinementConcept.ZPINCH,
]
# Pulsed concepts that form the target/liner in-situ -> no factory, no
# consumable. Paired with a fuel each concept actually supports: the aneutronic
# devices (PULSED_FRC is Helion-class D-He3; theta-pinch and dense-plasma-focus
# run aneutronic) have no breeding blanket and reject DT.
IN_SITU_CONCEPTS = [
    (ConfinementConcept.PLASMA_JET, Fuel.DT),
    (ConfinementConcept.MAG_TARGET, Fuel.DT),
    (ConfinementConcept.PULSED_FRC, Fuel.DHE3),
    (ConfinementConcept.THETA_PINCH, Fuel.DHE3),
    (ConfinementConcept.DENSE_PLASMA_FOCUS, Fuel.DHE3),
    (ConfinementConcept.STAGED_ZPINCH, Fuel.DT),
]


@pytest.mark.parametrize("concept", TARGET_CONCEPTS)
def test_target_concepts_have_factory(concept):
    """Manufactured-target concepts get a nonzero C220108 target factory."""
    r = CostModel(concept=concept, fuel=Fuel.DT).forward(**_FWD)
    assert float(r.cas22_detail["C220108"]) > 0.0


@pytest.mark.parametrize("concept,fuel", IN_SITU_CONCEPTS)
def test_in_situ_concepts_have_no_factory(concept, fuel):
    """In-situ-formation concepts carry no phantom target factory by default.

    This is the fix for the latent defect noted in the closed target-factory
    issues: correctness lives in the model, not in a per-concept C220108=0
    cost override.
    """
    r = CostModel(concept=concept, fuel=fuel).forward(**_FWD)
    assert float(r.cas22_detail["C220108"]) == 0.0


def test_laser_ife_cas80_exceeds_fuel_only():
    """Laser-IFE CAS80 includes the per-shot capsule, so it exceeds the
    isotope-only fuel cost an MFE plant of the same fusion power would see."""
    ife = CostModel(concept=ConfinementConcept.LASER_IFE, fuel=Fuel.DT).forward(**_FWD)
    # Reconstruct the isotope-only CAS80 at the same p_fus / availability.
    fuel_only = cas80_fuel(
        CC,
        p_fus=float(ife.power_table.p_fus),
        n_mod=1,
        availability=0.85,
        inflation_rate=ife.params["inflation_rate"],
        interest_rate=ife.params["interest_rate"],
        lifetime_yr=30,
        construction_time=ife.params["construction_time_yr"],
        fuel=Fuel.DT,
        noak=True,
        burn_fraction=ife.params["burn_fraction"],
        fuel_recovery=ife.params["fuel_recovery"],
        target_unit_cost=0.0,
        n_targets_per_year=0.0,
    )
    assert float(ife.costs.cas80) > float(fuel_only)


def test_cas80_target_term_linear_in_unit_cost():
    """The target consumable contribution is linear in target_unit_cost."""
    base = dict(
        p_fus=2600.0,
        n_mod=1,
        availability=0.85,
        inflation_rate=0.02,
        interest_rate=0.07,
        lifetime_yr=30,
        construction_time=6,
        fuel=Fuel.DT,
        noak=True,
        burn_fraction=0.05,
        fuel_recovery=0.99,
        n_targets_per_year=2.5e8,
    )
    c0 = cas80_fuel(CC, target_unit_cost=0.0, **base)
    c1 = cas80_fuel(CC, target_unit_cost=0.40, **base)
    c2 = cas80_fuel(CC, target_unit_cost=0.80, **base)
    # Equal increments of unit cost give equal increments of CAS80.
    assert float(c2 - c1) == pytest.approx(float(c1 - c0), rel=1e-6)
    assert float(c1) > float(c0)


def test_zero_targets_per_year_no_target_cost():
    """With no shots there is no target consumable, whatever the unit cost."""
    base = dict(
        p_fus=2600.0,
        n_mod=1,
        availability=0.85,
        inflation_rate=0.02,
        interest_rate=0.07,
        lifetime_yr=30,
        construction_time=6,
        fuel=Fuel.DT,
        noak=True,
        burn_fraction=0.05,
        fuel_recovery=0.99,
        n_targets_per_year=0.0,
    )
    assert float(cas80_fuel(CC, target_unit_cost=0.0, **base)) == pytest.approx(
        float(cas80_fuel(CC, target_unit_cost=1000.0, **base))
    )
