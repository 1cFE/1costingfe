"""eta_pin = eta_source(method) x eta_couple(concept): regression + behavior."""

import warnings

import pytest

from costingfe import ConfinementConcept as C
from costingfe import CostModel, Fuel

warnings.filterwarnings("ignore")

_BASE = dict(
    availability=0.85,
    lifetime_yr=30,
    n_mod=1,
    construction_time_yr=6.0,
    interest_rate=0.07,
    inflation_rate=0.02,
    noak=True,
)


@pytest.mark.parametrize(
    "concept,fuel,expected",
    [
        (C.TOKAMAK, Fuel.DT, 233.41),
        (C.MIRROR, Fuel.DT, 186.47),
        (C.STELLARATOR, Fuel.DT, 359.94),
        (C.DIPOLE, Fuel.DHE3, 182.79),
        (C.POLYWELL, Fuel.PB11, 52.98),
    ],
)
def test_benchmark_lcoe_preserved(concept, fuel, expected):
    """Coupling factors are chosen to reproduce each concept's prior eta_pin."""
    r = CostModel(concept=concept, fuel=fuel).forward(net_electric_mw=200.0, **_BASE)
    assert float(r.costs.lcoe) == pytest.approx(expected, abs=0.05)


def test_frc_rf_driver_couples_better_than_nbi():
    """Swapping the FRC driver to RF (with RMF coupling) cuts recirculating power."""
    m = CostModel(concept=C.STEADY_FRC, fuel=Fuel.PB11)
    nbi = m.forward(net_electric_mw=105.0, **_BASE)
    rf = m.forward(
        net_electric_mw=105.0,
        p_nbi=0.0,
        p_icrf=26.0,
        p_ecrh=0.0,
        p_lhcd=0.0,
        eta_couple=0.60,
        **_BASE,
    )
    assert float(rf.power_table.rec_frac) < float(nbi.power_table.rec_frac)


def test_explicit_eta_pin_override_wins():
    """An explicit eta_pin override bypasses the source x coupling derivation."""
    m = CostModel(concept=C.TOKAMAK, fuel=Fuel.DT)
    a = m.forward(net_electric_mw=200.0, **_BASE)
    b = m.forward(net_electric_mw=200.0, eta_pin=0.30, **_BASE)
    assert float(b.power_table.rec_frac) > float(a.power_table.rec_frac)
