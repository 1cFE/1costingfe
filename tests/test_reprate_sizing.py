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
# Unit ceilings (net @ max_f_rep, held cited shot fixed) measured against
# _pulsed_forward. Fuel matches each concept's reactor-class source pairing
# (PLASMA_JET / MAG_TARGET / THETA_PINCH = D-T; PULSED_FRC = D-He3; LASER_IFE
# = D-T), per docs/account_justification and concept_power_scaling.md:
#   PLASMA_JET (DT)      180.7 MW  @ max_f_rep=1.0
#   PULSED_FRC (DHE3)     50.5 MW  @ max_f_rep=1.0
#   THETA_PINCH (DT)    1025.6 MW  @ max_f_rep=0.1  (RTPR reactor point)
#   LASER_IFE (DT)       790.5 MW  @ max_f_rep=10.0
#   MAG_TARGET (DT)      161.2 MW  @ max_f_rep=1.0
# MAG_TARGET and THETA_PINCH use the recovered_compression forward: the driver
# energy is largely recovered (mechanical liner rebound / reactive ETS ring-
# down), so the cap-bank store and grid draw are separate, smaller energies
# than the delivered e_driver_mj (see pulsed_*.yaml e_store_mj / e_recirc_mj
# and concept_power_scaling.md). MAG_TARGET nets ~161 MW at the GF E-267
# sourced 0.85 recovery; a single-pass driver at eta_pin=0.30 would instead be
# net-negative, SizingInfeasible at any target.

_CONCEPT_FUEL = {
    ConfinementConcept.PLASMA_JET: Fuel.DT,
    ConfinementConcept.MAG_TARGET: Fuel.DT,
    ConfinementConcept.LASER_IFE: Fuel.DT,
    ConfinementConcept.PULSED_FRC: Fuel.DHE3,
    ConfinementConcept.THETA_PINCH: Fuel.DT,
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


def test_reprate_mag_target_recovers_and_sizes():
    # recovered_compression forward: the 755 MJ/shot liner KE is 85% recovered
    # mechanically (GF E-267 Sankey), and the electrical grid draw is the small
    # explicit e_recirc_mj, not the full driver at eta_pin=0.30. The single-
    # chamber unit ceiling is about +161 MW (a single-pass driver would be net-
    # negative and SizingInfeasible). A target below the ceiling sizes to a
    # single chamber and round-trips, like the other rep-rate concepts.
    r = _size(ConfinementConcept.MAG_TARGET, 100.0)
    assert r.solved_n_mod == 1
    assert r.power_table.p_net == pytest.approx(100.0, rel=0.02)
    assert r.power_table.f_rep <= 1.0 + 1e-9  # MAG_TARGET max_f_rep = 1.0


def test_reprate_mag_target_large_target_bumps_chambers():
    # 500 MW exceeds one MAG_TARGET chamber's ~161 MW ceiling:
    # ceil(500 / 161.2) = 4 chambers, each solved to <= its ceiling.
    r = _size(ConfinementConcept.MAG_TARGET, 500.0)
    assert r.solved_n_mod > 1
    assert r.power_table.f_rep <= 1.0 + 1e-9
    assert r.power_table.p_net * r.solved_n_mod == pytest.approx(500.0, rel=0.02)


@pytest.mark.parametrize(
    "concept,target",
    [
        (ConfinementConcept.PULSED_FRC, 20.0),
        (ConfinementConcept.THETA_PINCH, 400.0),
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


# --- Task 7: cited shot authoritative on the NORMAL (non-sizing) path ---
#
# For a rep-rate concept the normal forward path now HOLDS the cited shot
# (e_driver_mj, yield_per_shot_mj) fixed and SOLVES f_rep to hit the per-module
# net target, instead of growing e_driver at the YAML f_rep ("grow the shot").
# So the same machine has ONE shot definition in both modes, and f_rep becomes
# a solved output (pt.f_rep), not the YAML field.


def test_reprate_normal_path_holds_cited_shot():
    # Below one chamber's ceiling, the non-sizing forward reaches the target on
    # the cited e_driver_mj (not an inverse-grown one) and a solved f_rep.
    m = CostModel(ConfinementConcept.PLASMA_JET, Fuel.DT)
    cited = m._eng_defaults["e_driver_mj"]
    r = m.forward(net_electric_mw=20.0, availability=0.87, lifetime_yr=40)
    assert float(r.power_table.e_driver_mj) == pytest.approx(cited, rel=1e-9)
    assert float(r.power_table.p_net) == pytest.approx(20.0, rel=0.02)
    assert float(r.power_table.f_rep) <= 1.0 + 1e-9  # PLASMA_JET max_f_rep = 1.0


def test_reprate_two_modes_agree_on_shot():
    # Below one chamber's ceiling the normal path (n_mod=1) and the
    # size_from_power path (which also solves n_mod=1) reach the SAME target
    # with the SAME cited shot, the SAME solved f_rep, and the SAME LCOE.
    target = 20.0
    m = CostModel(ConfinementConcept.PLASMA_JET, Fuel.DT)
    normal = m.forward(net_electric_mw=target, availability=0.87, lifetime_yr=40)
    sized = m.forward(
        net_electric_mw=target,
        availability=0.87,
        lifetime_yr=40,
        size_from_power=True,
    )
    assert sized.solved_n_mod == 1
    assert float(normal.power_table.e_driver_mj) == pytest.approx(
        float(sized.power_table.e_driver_mj), rel=1e-9
    )
    assert float(normal.power_table.f_rep) == pytest.approx(
        float(sized.power_table.f_rep), rel=1e-6
    )
    assert float(normal.costs.lcoe) == pytest.approx(float(sized.costs.lcoe), rel=1e-6)


def test_reprate_normal_path_infeasible_above_ceiling():
    # Above one chamber's ceiling with the default n_mod=1, the normal path
    # cannot reach the target and points the user to size_from_power.
    m = CostModel(ConfinementConcept.MAG_TARGET, Fuel.DT)
    with pytest.raises(SizingInfeasible, match="size_from_power"):
        m.forward(net_electric_mw=1000.0, availability=0.87, lifetime_yr=40)
