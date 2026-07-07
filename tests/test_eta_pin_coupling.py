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
        # re-pinned 2026-06-13: fluence-based CAS72 basis change, see
        # wall_limits_and_fluence.md (TOKAMAK was 228.07, MIRROR 186.95,
        # STELLARATOR 343.34). Steady-state MFE core lifetime now tracks q_n.
        # re-pinned after the tokamak default geometry was re-baselined to the
        # size_from_power 1000 MWe operating point (R0 6.04 m, B 10 T); costing
        # that machine at this 200 MWe point raises LCOE (was 208.41 at R0 3.0).
        # re-pinned: gross-electric reference unified to 1100 MWe (ref_gross_power_mwe;
        # CAS21 was 1150, C220107/C220110 were 1000), small downshift across concepts.
        # re-pinned: D-T blanket unit cost re-anchored to structure-only (0.60 ->
        # 0.35); the breeder/multiplier fill it had baked in is now priced once, in
        # CAS27. Downshift is D-T-only (TOKAMAK 295.32, MIRROR 190.52, STELLARATOR
        # 330.54); DIPOLE (Li2O base) and POLYWELL (aneutronic) pins unchanged.
        (C.TOKAMAK, Fuel.DT, 280.61),
        # re-pinned: mirror central-cell T_i and T_e corrected to the near-Maxwellian
        # Hammir/WHAM value (10 keV). A 2026-06-15 change had raised central T_e to
        # 125 keV (the tandem PLUG hot-electron value that sets the Fowler-Logan
        # potential e*phi = T_e_plug*ln(n_p/n_c)) and read it onto the BULK cell via
        # the non-0D radiation term, inflating the MIRROR-DT benchmark with a spurious
        # high-T_e synchrotron term. Corrected here: 218.36 -> 190.52. Mirror-only;
        # tokamak/stellarator/dipole/polywell pins unchanged. See
        # docs/physics/mirror.md.
        (C.MIRROR, Fuel.DT, 185.26),
        # re-pinned: stellarator coil-center field now derived from the design
        # on-axis field B (like the tokamak) instead of a frozen b_center YAML
        # default. The default B is 5.0 T, below the old frozen 6.0 T, so the
        # coil cost and LCOE drop (323.15 -> 290.70). Stellarator-only.
        (C.STELLARATOR, Fuel.DT, 290.70),
        # DIPOLE re-benchmarked after the radial-build inversion + spherical
        # geometry dispatch + Li2O blanket fill + C220108 divertor zeroing +
        # external stationary lift coil restructured as
        # stationary_lift_coil_fraction * floating_with_markup (Simpson 2026
        # alignment). +7.4% here is the volumetric CAS27: the dipole's large
        # thin Li2O blanket (C220101 ~$360M) was previously costed at ~$0.6M of
        # fill; it is now ~$235M, consistent with the blanket structure.
        # re-fueled to D-T to match the Simpson 2026 config this concept loads
        # (Li2O tritium-breeding blanket, mn=1.053, tritium plant); the prior
        # D-He3 label costed aneutronic fuel on D-T breeding hardware (a $93M/yr
        # He3 fuel bill against a breeding blanket). Combined with the Li2O
        # structure re-anchor to structure-only (0.20 -> 0.35, fill in CAS27,
        # C220101 ~$368M -> ~$644M), the D-T LCOE is 289.25 (was 279.07 D-He3).
        (C.DIPOLE, Fuel.DT, 289.25),
        # POLYWELL re-benchmarked after two right-sizings for this electrostatic,
        # copper-magnet concept: (1) C220108 divertor zeroed (charged particles
        # exhaust to the direct converter, no W-monoblock cassette), then (2) the
        # CAS21 cryogenics building zeroed (normal-conducting magnets need no
        # cryoplant). Cumulative -9.6% from the original 53.08.
        # re-pinned: CAS10 land now sqrt(plant-total power); at this 200 MWe
        # off-reference point the small-plant land rises (was 49.55), then gross
        # reference unified to 1100 (was 49.63).
        (C.POLYWELL, Fuel.PB11, 49.50),
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
