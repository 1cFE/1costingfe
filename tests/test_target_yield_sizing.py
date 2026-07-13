"""Deterministic target-yield sizing (design doc D3) on the first-principles
gain framework: company sets driver energy + ceiling; physics (fuel_coupling_frac
+ universal gain_coeff/gain_exp) sets the yield. Solver scales driver energy
E -> rep rate f -> chamber count n_mod to hit the plant net target.
"""

import pytest

from costingfe import CostModel
from costingfe.defaults import load_engineering_defaults
from costingfe.layers.physics import physics_yield_mj
from costingfe.types import ConfinementConcept, Fuel


def _params(concept_yaml, net_mw, **extra):
    p = dict(load_engineering_defaults(concept_yaml))
    p["eta_th"] = 0.40
    # eta_pin is resolved from laser_driver_type in forward(); these tests call
    # the sizing helpers directly (bypassing forward), so set it explicitly.
    p.setdefault("eta_pin", 0.15)
    p["net_electric_mw"] = net_mw
    p["sizing_axis"] = "target_yield"
    p.update(extra)
    return p


def _laser_params(net_mw, **extra):
    return _params("pulsed_laser_ife", net_mw, **extra)


def _model():
    return CostModel(ConfinementConcept.LASER_IFE, Fuel.DT)


def test_yield_from_e_uses_gain_curve_by_default():
    # Laser defaults enable the size-dependent gain curve (rhoR_ref in the YAML),
    # so _yield_from_e must match physics_yield_mj called WITH the curve args.
    m, p = _model(), _laser_params(1000.0)
    assert p["rhoR_ref_g_cm2"] > 0  # curve on by default for laser
    for e in (2.5, 6.0, 10.0):
        assert m._yield_from_e(p, e) == pytest.approx(
            physics_yield_mj(
                e,
                p["burn_fraction"],
                p.get("fuel_mass_mg_per_mj", 1.1),
                p.get("e_fuel_mj_per_g", 3.4e5),
                m._coupling_frac(p),
                p["rhoR_ref_g_cm2"],
                p["e_rhoR_ref_mj"],
                p.get("gain_hb_g_cm2", 6.0),
            )
        )


def test_higher_rhoR_ref_gives_more_yield():
    # With the curve on, rhoR_ref is the gain lever (higher areal density -> more
    # burn-up -> more yield at the same driver energy).
    m, p = _model(), _laser_params(1000.0)
    p_hi = dict(p, rhoR_ref_g_cm2=3.0)
    assert m._yield_from_e(p_hi, 8.0) > m._yield_from_e(p, 8.0)


def test_flat_burn_fraction_when_curve_off():
    # rhoR_ref=0 reverts to the flat burn_fraction path (MagLIF/Z-pinch default).
    m = _model()
    p = _laser_params(1000.0, rhoR_ref_g_cm2=0.0)
    p_hi = dict(p, burn_fraction=0.30)
    assert m._yield_from_e(p_hi, 8.0) > m._yield_from_e(p, 8.0)


def test_laser_eta_pin_resolves_by_driver_type():
    from costingfe.types import LaserDriverType

    for ldt, expect in [
        (LaserDriverType.DPSSL, 0.15),
        (LaserDriverType.KRF, 0.10),
        (LaserDriverType.FIBER, 0.20),
        (LaserDriverType.NDGLASS, 0.02),
    ]:
        m = CostModel(ConfinementConcept.LASER_IFE, Fuel.DT, laser_driver_type=ldt)
        assert m._laser_eta_pin({}) == pytest.approx(expect)
    # eta_couple (default 1.0) still multiplies, and explicit eta_source wins.
    m = CostModel(ConfinementConcept.LASER_IFE, Fuel.DT)
    assert m._laser_eta_pin({"eta_couple": 0.5}) == pytest.approx(0.15 * 0.5)


def test_lower_wallplug_efficiency_lowers_net():
    # KrF (0.10) is less efficient than DPSSL (0.15): at the same shot it burns
    # more recirculating power, so one chamber's net electric is lower.
    m = _model()
    hi = m._pulsed_net_at_ef(_laser_params(1000.0, eta_pin=0.15), 8.0, 5.0)
    lo = m._pulsed_net_at_ef(_laser_params(1000.0, eta_pin=0.10), 8.0, 5.0)
    assert lo < hi


def test_drive_mode_orders_yield_direct_hybrid_indirect():
    m = _model()
    direct = m._yield_from_e(_laser_params(1000.0, drive_mode="direct"), 8.0)
    hybrid = m._yield_from_e(_laser_params(1000.0, drive_mode="hybrid"), 8.0)
    indirect = m._yield_from_e(_laser_params(1000.0, drive_mode="indirect"), 8.0)
    assert direct > hybrid > indirect
    # Unset drive_mode defaults to direct (unchanged behaviour).
    assert m._yield_from_e(_laser_params(1000.0), 8.0) == pytest.approx(direct)


def test_coupling_frac_override_wins_over_drive_mode():
    m = _model()
    p = _laser_params(1000.0, drive_mode="indirect", coupling_frac=0.9)
    # Explicit coupling_frac beats the indirect (0.5) lookup.
    assert m._coupling_frac(p) == pytest.approx(0.9)


def test_indirect_drive_raises_lcoe_vs_direct():
    # End-to-end: worse coupling -> lower gain -> bigger driver/chamber -> higher
    # LCOE at the same 1 GWe target and rep rate.
    m = _model()
    base = dict(
        sizing_axis="target_yield",
        sizing_mode="single_chamber",
        target_cost_mode="capsule_fab",
        wall_improvement_factor=50.0,
        neutron_wall_load_max_mw_m2=20.0,
    )
    r_direct = m.forward(
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30.0,
        size_from_power=True,
        drive_mode="direct",
        **base,
    )
    r_indirect = m.forward(
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30.0,
        size_from_power=True,
        drive_mode="indirect",
        **base,
    )
    assert r_indirect.costs.lcoe > r_direct.costs.lcoe


def test_net_monotone_in_e_and_f():
    m, p = _model(), _laser_params(1000.0)
    assert m._pulsed_net_at_ef(p, 6.0, 5.0) > m._pulsed_net_at_ef(p, 3.0, 5.0)
    assert m._pulsed_net_at_ef(p, 5.0, 8.0) > m._pulsed_net_at_ef(p, 5.0, 4.0)


def test_solver_hits_target_within_band():
    m, p = _model(), _laser_params(1000.0)
    pt = m._size_target_yield(p, 1)
    assert m._last_n_mod >= 1
    assert p["e_driver_min_mj"] <= pt.e_driver_mj <= p["e_driver_max_mj"]
    assert pt.p_net * m._last_n_mod == pytest.approx(1000.0, rel=1e-2)


def test_small_target_trims_rep_rate():
    m = _model()
    # Below one cited-shot chamber -> hold E at cited, trim f, one chamber.
    p = _laser_params(50.0)
    pt = m._size_target_yield(p, 1)
    assert m._last_n_mod == 1
    assert pt.f_rep < p["f_rep"]
    assert pt.p_net == pytest.approx(50.0, rel=1e-2)


def test_maglif_on_axis_hits_target():
    m = CostModel(ConfinementConcept.MAGLIF, Fuel.DT)
    p = _params("pulsed_maglif", 1000.0)
    pt = m._size_target_yield(p, 1)
    assert pt.p_net * m._last_n_mod == pytest.approx(1000.0, rel=1e-2)
    assert pt.p_fus / pt.f_rep == pytest.approx(m._yield_from_e(p, pt.e_driver_mj))


def test_maglif_defaults_to_target_yield_axis():
    # MagLIF now ships with sizing_axis: target_yield (the calibrated default);
    # clearing the flag reverts it to the generic q_eng path.
    m = CostModel(ConfinementConcept.MAGLIF, Fuel.DT)
    p = dict(load_engineering_defaults("pulsed_maglif"))
    assert m._on_target_yield_axis(p) is True
    p.pop("sizing_axis")
    assert m._on_target_yield_axis(p) is False


def test_pinned_n_mod_rejected():
    m, p = _model(), _laser_params(1000.0)
    with pytest.raises(ValueError, match="n_mod cannot be pinned"):
        m._size_target_yield(p, 2)


# -- single_chamber sizing mode (grow one chamber, no driver ceiling) ---------


def test_single_chamber_hits_target_one_chamber():
    m, p = _model(), _laser_params(1000.0, sizing_mode="single_chamber")
    pt = m._size_target_yield(p, 1)
    assert m._last_n_mod == 1  # always one chamber
    assert pt.p_net == pytest.approx(1000.0, rel=1e-2)
    assert pt.f_rep == pytest.approx(p["f_rep"])  # rep held at cited value


def test_single_chamber_ignores_driver_ceiling():
    # At a LOW rep rate, reaching 1 GWe with one chamber needs far more energy
    # than the nominal engineering ceiling; single_chamber grows E straight past
    # it (driver is linear $/J, so the count/ceiling is invisible to cost).
    m = _model()
    p = _laser_params(1000.0, sizing_mode="single_chamber", f_rep=0.5)
    pt = m._size_target_yield(p, 1)
    assert pt.e_driver_mj > p["e_driver_max_mj"]
    assert pt.f_rep == pytest.approx(0.5)  # rep held; energy did the scaling


def test_single_chamber_scales_e_with_target():
    m = _model()
    pt_lo = m._size_target_yield(_laser_params(500.0, sizing_mode="single_chamber"), 1)
    pt_hi = m._size_target_yield(_laser_params(1500.0, sizing_mode="single_chamber"), 1)
    # More net power at the same rep rate -> proportionally larger driver/yield.
    assert pt_hi.e_driver_mj > pt_lo.e_driver_mj
    assert pt_hi.p_fus > pt_lo.p_fus


def test_single_chamber_small_target_trims_rep():
    # Below the cited shot at cited rep -> hold E at cited, trim f (one chamber).
    m, p = _model(), _laser_params(50.0, sizing_mode="single_chamber")
    pt = m._size_target_yield(p, 1)
    assert m._last_n_mod == 1
    assert pt.e_driver_mj == pytest.approx(p["e_driver_mj"])
    assert pt.f_rep < p["f_rep"]
    assert pt.p_net == pytest.approx(50.0, rel=1e-2)
