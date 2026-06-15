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
        # Tokamak/stellarator re-benchmarked after the bilinear toroidal coil
        # model (total_kAm = G*B*R0*r_coil, replacing the SPARC-calibrated
        # r_bore^2), the recalibrated markups (tokamak 3.09; stellarator 5.87 =
        # 1.9x tokamak, NCSX modular-coil overrun), and the stellarator
        # b_center 18->6 T fix (on-axis field, not peak-on-conductor). Tokamak
        # is ~unchanged (recalibration is cost-neutral at its R0=3.0 reference);
        # stellarator LCOE drops as the field correction outweighs the markup.
        # All re-benchmarked after CAS27 moved to a volume-based blanket-fill
        # build-up (blanket_vol x vol_frac x density x $/kg, keyed on
        # blanket_fill) replacing the special_materials x fill_factor x P_net
        # model. DT/PbLi shifts are <0.5% (a fixed inventory no longer shrinks
        # with net power).
        # Re-benchmarked again after CAS21 buildings/site were fed plant-total
        # power (n_mod x per-module) so they scale with module replication, and
        # the staff-driven administration building moved to a P^0.5 law (was
        # linear in power), matching the staffing accounts CAS40/CAS70. At these
        # 200 MWe / n_mod=1 points only the admin sqrt change bites: sub-1 GWe
        # plants pay slightly more admin building, so LCOE rises <0.1%.
        (C.TOKAMAK, Fuel.DT, 228.07),
        (C.MIRROR, Fuel.DT, 186.95),
        (C.STELLARATOR, Fuel.DT, 343.34),
        # DIPOLE re-benchmarked after the radial-build inversion + spherical
        # geometry dispatch + Li2O blanket fill + C220108 divertor zeroing +
        # external stationary lift coil restructured as
        # stationary_lift_coil_fraction * floating_with_markup (Simpson 2026
        # alignment). +7.4% here is the volumetric CAS27: the dipole's large
        # thin Li2O blanket (C220101 ~$360M) was previously costed at ~$0.6M of
        # fill; it is now ~$235M, consistent with the blanket structure.
        (C.DIPOLE, Fuel.DHE3, 276.24),
        # POLYWELL re-benchmarked after two right-sizings for this electrostatic,
        # copper-magnet concept: (1) C220108 divertor zeroed (charged particles
        # exhaust to the direct converter, no W-monoblock cassette), then (2) the
        # CAS21 cryogenics building zeroed (normal-conducting magnets need no
        # cryoplant). Cumulative -9.6% from the original 53.08.
        (C.POLYWELL, Fuel.PB11, 47.96),
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


def test_eta_pin_rejected_for_heated_concept():
    """eta_pin is derived for NBI/RF concepts; setting it directly is rejected."""
    m = CostModel(concept=C.TOKAMAK, fuel=Fuel.DT)
    with pytest.raises(ValueError, match="eta_pin"):
        m.forward(net_electric_mw=200.0, eta_pin=0.30, **_BASE)


def test_eta_pin_still_accepted_for_electrostatic():
    """Electrostatic concepts keep eta_pin as their direct input (unchanged)."""
    m = CostModel(concept=C.POLYWELL, fuel=Fuel.PB11)
    a = m.forward(net_electric_mw=200.0, **_BASE)
    b = m.forward(net_electric_mw=200.0, eta_pin=0.5, **_BASE)
    assert float(b.power_table.rec_frac) > float(a.power_table.rec_frac)
