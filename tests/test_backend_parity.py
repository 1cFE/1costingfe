"""Forward LCOE must be finite and positive under whichever backend runs.

CI runs the full suite once per backend (default jax, and
COSTINGFE_BACKEND=numpy). This file exercises a representative spread of
concepts so a backend-specific NaN/inf regression is caught in either run.
"""

import math

import pytest

from costingfe import ConfinementConcept, CostModel, Fuel

CASES = [
    (
        ConfinementConcept.TOKAMAK,
        Fuel.DT,
        dict(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30),
    ),
    (
        ConfinementConcept.MIRROR,
        Fuel.DT,
        dict(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30),
    ),
    (
        ConfinementConcept.ORBITRON,
        Fuel.PB11,
        dict(net_electric_mw=0.005, availability=0.85, lifetime_yr=30, n_mod=1),
    ),
]


@pytest.mark.parametrize("concept,fuel,kw", CASES)
def test_forward_lcoe_finite_positive(concept, fuel, kw):
    lcoe = CostModel(concept=concept, fuel=fuel).forward(**kw).costs.lcoe
    assert math.isfinite(lcoe) and lcoe > 0
