"""eta_source_* overrides must reach eta_pin (and thus power/cost tables).

_effective_eta_pin previously read the four eta_source_* constants off the
frozen CostingConstants object (self.cc) instead of the injected params dict,
so a user override (or the sensitivity finite-difference perturbation) was
silently ignored: the resulting power tables and LCOE were bit-identical
regardless of eta_source_nbi, and its elasticity was exactly 0.0.
"""

import warnings

from costingfe import ConfinementConcept as C
from costingfe import CostModel, Fuel

warnings.filterwarnings("ignore")

_BASE = dict(
    net_electric_mw=200.0,
    availability=0.85,
    lifetime_yr=30,
    n_mod=1,
    construction_time_yr=6.0,
    interest_rate=0.07,
    inflation_rate=0.02,
    noak=True,
)


def test_eta_source_nbi_override_changes_power_and_cost():
    """MIRROR is NBI-heated (eta_couple set in its YAML); a higher source
    efficiency must lower recirculating power, raise q_eng, and lower LCOE.
    """
    m = CostModel(C.MIRROR, Fuel.DT)
    lo = m.forward(eta_source_nbi=0.60, **_BASE)  # matches the YAML/constants default
    hi = m.forward(eta_source_nbi=0.90, **_BASE)

    assert float(hi.power_table.rec_frac) < float(lo.power_table.rec_frac)
    assert float(hi.power_table.q_eng) > float(lo.power_table.q_eng)
    assert float(hi.costs.lcoe) < float(lo.costs.lcoe)


def test_eta_source_nbi_sensitivity_elasticity_nonzero():
    """The numpy-backend finite-difference sensitivity must see eta_source_nbi.

    eta_pin is not exposed on result.params (it's derived inside
    _power_balance from a locally-rebuilt params dict), so this checks the
    override actually reaches the physics via the sensitivity elasticity,
    which perturbs params["eta_source_nbi"] and re-runs forward() concretely.
    """
    m = CostModel(C.MIRROR, Fuel.DT)
    r = m.forward(**_BASE)
    sens = m.sensitivity(r.params)
    elasticity = sens["costing"]["eta_source_nbi"]
    assert elasticity != 0.0
    # Higher source efficiency lowers LCOE, so the elasticity is negative.
    assert elasticity < 0.0
