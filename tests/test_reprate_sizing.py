"""Tests for the reusable pulsed-forward evaluator (`CostModel._pulsed_forward`).

Task 4 of the rep-rate power sizing feature: extracts the pulsed forward-call
out of `_power_balance`'s PULSED branch so a future rep-rate sizing solver
(Task 5) can evaluate the forward power balance at an explicit (p_fus,
e_driver_mj, f_rep) without duplicating the argument mapping. No solver is
wired up yet; this only exercises the extracted helper directly.
"""

import pytest

from costingfe import CostModel
from costingfe.defaults import load_engineering_defaults
from costingfe.layers.physics import SizingInfeasible
from costingfe.types import ConfinementConcept, Fuel


def _pulsed_plasma_jet_params():
    """Minimal params dict for CostModel(PLASMA_JET, DT)._pulsed_forward.

    Built from the concept's engineering-default YAML plus the one
    cross-cutting key (eta_th) that `forward()` normally injects from the
    power-cycle preset rather than the YAML. `_pulsed_forward` must not be
    reached into via `forward()` internals, so this mirrors only what the
    helper actually reads from `params`.
    """
    params = dict(load_engineering_defaults("pulsed_plasma_jet"))
    params["eta_th"] = 0.40
    return params


def test_pulsed_forward_helper_scales_pfus_with_frep():
    m = CostModel(ConfinementConcept.PLASMA_JET, Fuel.DT)
    params = _pulsed_plasma_jet_params()
    y = params["yield_per_shot_mj"]
    e = params["e_driver_mj"]
    # Fixed shot at two rates: p_fus and driver power both scale with f_rep.
    pt1 = m._pulsed_forward(params, y * 1.0, e, 1.0)
    pt2 = m._pulsed_forward(params, y * 2.0, e, 2.0)
    assert pt2.p_net > pt1.p_net


# --- Task 5: _size_reprate f_rep-from-power solver + dispatch wiring ---
#
# Unit ceilings (net @ max_f_rep, held e_driver_mj fixed, default RANKINE
# eta_th) measured directly against _pulsed_forward before this solver
# existed. Fuel matches each concept's real developer-fuel pairing (Realta
# PLASMA_JET/MAG_TARGET = D-T; Helion PULSED_FRC and THETA_PINCH's D-He3
# baseline = D-He3; LASER_IFE = D-T), per docs/account_justification and
# tests/test_model.py's existing fuel choices for these concepts:
#   PLASMA_JET (DT)     180.7 MW  @ max_f_rep=1.0
#   PULSED_FRC (DHE3)    67.1 MW  @ max_f_rep=1.0
#   THETA_PINCH (DHE3)   19.9 MW  @ max_f_rep=1.0
#   LASER_IFE (DT)      791.2 MW  @ max_f_rep=10.0
#   MAG_TARGET (DT)   -1911.5 MW  @ max_f_rep=1.0 (driver-dominated:
#     e_driver_mj=755 MJ/shot recirculates at eta_pin=0.30 every shot,
#     swamping the 780 MJ/shot yield at any f_rep > 0 -- SizingInfeasible at
#     ANY target).

_CONCEPT_FUEL = {
    ConfinementConcept.PLASMA_JET: Fuel.DT,
    ConfinementConcept.MAG_TARGET: Fuel.DT,
    ConfinementConcept.LASER_IFE: Fuel.DT,
    ConfinementConcept.PULSED_FRC: Fuel.DHE3,
    ConfinementConcept.THETA_PINCH: Fuel.DHE3,
}


def _size(concept, target):
    m = CostModel(concept, _CONCEPT_FUEL[concept])
    return m.forward(
        net_electric_mw=target,
        availability=0.87,
        lifetime_yr=40,
        size_from_power=True,
    )


def test_reprate_single_chamber_hits_target_below_ceiling():
    # 20 MW is well below PLASMA_JET's 180.7 MW single-chamber ceiling:
    # n_mod=1, f_rep <= max_f_rep, and the solved point round-trips to target.
    r = _size(ConfinementConcept.PLASMA_JET, 20.0)
    assert r.solved_n_mod == 1
    assert r.power_table.p_net == pytest.approx(20.0, rel=0.02)
    assert r.power_table.f_rep <= 1.0 + 1e-9  # PLASMA_JET max_f_rep = 1.0


def test_reprate_large_target_bumps_chambers():
    # 1000 MW exceeds one PLASMA_JET chamber's 180.7 MW ceiling:
    # ceil(1000 / 180.7) = 6 chambers, each solved to <= its ceiling.
    r = _size(ConfinementConcept.PLASMA_JET, 1000.0)
    assert r.solved_n_mod > 1
    assert r.power_table.f_rep <= 1.0 + 1e-9  # each chamber at/below its ceiling
    assert r.power_table.p_net * r.solved_n_mod == pytest.approx(1000.0, rel=0.02)


def test_reprate_pinned_n_mod_rejected():
    m = CostModel(ConfinementConcept.PLASMA_JET, Fuel.DT)
    with pytest.raises(ValueError, match="n_mod"):
        m.forward(
            net_electric_mw=20.0,
            availability=0.87,
            lifetime_yr=40,
            n_mod=3,
            size_from_power=True,
        )


def test_reprate_driver_dominated_concept_is_infeasible():
    # MAG_TARGET's default shot design point is driver-dominated (recirc from
    # e_driver_mj swamps the thermal yield at eta_pin=0.30): unreachable at
    # ANY positive target, since bumping n_mod cannot fix a negative unit
    # ceiling. This exercises the SizingInfeasible short-circuit in
    # _n_mod_for_target before bisection is ever attempted.
    with pytest.raises(SizingInfeasible):
        _size(ConfinementConcept.MAG_TARGET, 20.0)


@pytest.mark.parametrize(
    "concept,target",
    [
        (ConfinementConcept.PULSED_FRC, 20.0),
        (ConfinementConcept.THETA_PINCH, 8.0),
        (ConfinementConcept.LASER_IFE, 400.0),
    ],
)
def test_reprate_round_trip_below_ceiling(concept, target):
    # Each target is chosen below that concept's single-chamber ceiling
    # (see the module-level unit-ceiling table above), so n_mod == 1 and the
    # solved f_rep stays within that concept's own max_f_rep.
    m = CostModel(concept, _CONCEPT_FUEL[concept])
    max_f_rep = m._eng_defaults["max_f_rep"]
    r = _size(concept, target)
    assert r.solved_n_mod == 1
    assert r.power_table.f_rep <= max_f_rep + 1e-9
    assert r.power_table.p_net == pytest.approx(target, rel=0.02)


def test_reprate_laser_ife_large_target_bumps_chambers():
    # 2000 MW exceeds LASER_IFE's 791.2 MW single-chamber ceiling:
    # ceil(2000 / 791.2) = 3 chambers.
    r = _size(ConfinementConcept.LASER_IFE, 2000.0)
    assert r.solved_n_mod > 1
    assert r.power_table.f_rep <= 10.0 + 1e-9
    assert r.power_table.p_net * r.solved_n_mod == pytest.approx(2000.0, rel=0.02)
