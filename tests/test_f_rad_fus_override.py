"""f_rad_fus_pb11/f_rad_fus_dhe3 overrides must reach p_rad (and thus LCOE).

The steady-state and 0D/sizing power-balance paths previously resolved the
per-fuel steady-state radiation fraction via cc.f_rad_fus(fuel), reading the
frozen CostingConstants object instead of the injected params dict. A user
override (or the sensitivity finite-difference perturbation) on
f_rad_fus_pb11/f_rad_fus_dhe3 was therefore silently ignored: the resulting
power tables and LCOE were bit-identical regardless of the override, and its
elasticity was exactly 0.0.
"""

import warnings

import pytest

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


def test_f_rad_fus_pb11_override_changes_p_rad_and_lcoe():
    """A p-B11 mirror's radiated power must track f_rad_fus_pb11 * p_fus,
    and a changed f_rad_fus_pb11 must move LCOE (it feeds recirculating
    power / thermal-vs-DEC split downstream)."""
    m = CostModel(C.MIRROR, Fuel.PB11)
    base = m.forward(**_BASE)  # default f_rad_fus_pb11 = 0.83
    hi = m.forward(f_rad_fus_pb11=0.90, **_BASE)

    assert float(hi.power_table.p_rad) == pytest.approx(
        0.90 * float(hi.power_table.p_fus), rel=1e-6
    )
    assert float(base.power_table.p_rad) == pytest.approx(
        0.83 * float(base.power_table.p_fus), rel=1e-6
    )
    assert float(hi.costs.lcoe) != pytest.approx(float(base.costs.lcoe))


def test_f_rad_fus_explicit_still_overrides_per_fuel_default():
    """Explicit f_rad_fus kwarg keeps precedence over f_rad_fus_pb11."""
    m = CostModel(C.MIRROR, Fuel.PB11)
    r = m.forward(f_rad_fus=0.5, f_rad_fus_pb11=0.90, **_BASE)

    assert float(r.power_table.p_rad) == pytest.approx(
        0.5 * float(r.power_table.p_fus), rel=1e-6
    )


def test_f_rad_fus_pb11_sensitivity_elasticity_nonzero():
    """The numpy-backend finite-difference sensitivity must see
    f_rad_fus_pb11: it perturbs params["f_rad_fus_pb11"] and re-runs
    forward() concretely, so a live override must show up as a nonzero
    elasticity."""
    m = CostModel(C.MIRROR, Fuel.PB11)
    r = m.forward(**_BASE)
    sens = m.sensitivity(r.params)
    elasticity = sens["costing"]["f_rad_fus_pb11"]
    assert elasticity != 0.0


def test_f_rad_fus_pb11_no_effect_on_dt():
    """DT keeps the full compute_p_rad path; f_rad_fus_pb11 is irrelevant."""
    m = CostModel(C.MIRROR, Fuel.DT)
    lo = m.forward(**_BASE)
    hi = m.forward(f_rad_fus_pb11=0.99, **_BASE)

    assert float(hi.power_table.p_rad) == pytest.approx(float(lo.power_table.p_rad))
    assert float(hi.costs.lcoe) == pytest.approx(float(lo.costs.lcoe))
