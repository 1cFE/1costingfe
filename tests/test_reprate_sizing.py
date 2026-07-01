"""Tests for the reusable pulsed-forward evaluator (`CostModel._pulsed_forward`).

Task 4 of the rep-rate power sizing feature: extracts the pulsed forward-call
out of `_power_balance`'s PULSED branch so a future rep-rate sizing solver
(Task 5) can evaluate the forward power balance at an explicit (p_fus,
e_driver_mj, f_rep) without duplicating the argument mapping. No solver is
wired up yet; this only exercises the extracted helper directly.
"""

from costingfe import CostModel
from costingfe.defaults import load_engineering_defaults
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
