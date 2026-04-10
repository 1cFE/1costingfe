"""Tests for pulsed thermal and DEC power balance (forward + inverse)."""

from costingfe.layers.physics import (
    pulsed_dec_forward,
    pulsed_dec_inverse,
    pulsed_thermal_forward,
    pulsed_thermal_inverse,
)
from costingfe.types import Fuel

THERMAL_PARAMS = dict(
    p_fus=2500.0,
    fuel=Fuel.DT,
    e_driver_mj=100.0,
    f_rep=1.0,
    mn=1.1,
    eta_th=0.40,
    eta_pin=0.15,
    f_rad=0.10,
    f_sub=0.03,
    p_pump=1.0,
    p_trit=10.0,
    p_house=4.0,
    p_cryo=0.5,
    p_target=1.0,
    p_coils=0.0,
)


def test_thermal_forward_positive_net():
    pt = pulsed_thermal_forward(**THERMAL_PARAMS)
    assert pt.p_net > 0


def test_thermal_forward_no_dec():
    pt = pulsed_thermal_forward(**THERMAL_PARAMS)
    assert pt.p_dee == 0.0


def test_thermal_forward_pulsed_fields():
    pt = pulsed_thermal_forward(**THERMAL_PARAMS)
    assert pt.e_driver_mj == 100.0
    assert pt.f_rep == 1.0
    assert pt.e_stored_mj > pt.e_driver_mj
    assert pt.f_ch > 0


def test_thermal_forward_driver_thermalizes():
    pt = pulsed_thermal_forward(**THERMAL_PARAMS)
    assert pt.p_th > pt.p_fus * 1.1  # p_th includes driver power


def test_thermal_forward_no_pump_without_thermal():
    params = dict(THERMAL_PARAMS, eta_th=0.0)
    pt = pulsed_thermal_forward(**params)
    assert pt.p_et == 0.0


def test_thermal_inverse_roundtrip():
    pt = pulsed_thermal_forward(**THERMAL_PARAMS)
    inv_params = {
        k: v
        for k, v in THERMAL_PARAMS.items()
        if k not in ("p_fus", "fuel", "e_driver_mj")
    }
    p_fus_recovered, e_driver_recovered = pulsed_thermal_inverse(
        p_net_target=pt.p_net, fuel=Fuel.DT, q_eng=pt.q_eng, **inv_params
    )
    assert abs(p_fus_recovered - 2500.0) < 0.5
    assert abs(e_driver_recovered - 100.0) < 0.5


# ---------------------------------------------------------------------------
# Pulsed DEC (inductive) power balance tests
# ---------------------------------------------------------------------------

# Reference: Helion-like FRC, D-He3, 1 Hz, 12 MJ/pulse
DEC_PARAMS = dict(
    p_fus=500.0,
    fuel=Fuel.DHE3,
    e_driver_mj=12.0,
    f_rep=1.0,
    mn=1.0,
    eta_th=0.0,  # no thermal BOP
    eta_pin=0.95,
    eta_dec=0.85,
    f_pdv=0.80,
    f_rad=0.05,
    f_sub=0.03,
    p_pump=0.0,
    p_trit=0.5,
    p_house=2.0,
    p_cryo=0.0,
    p_target=0.0,
    p_coils=0.5,
)


def test_dec_forward_positive_net():
    pt = pulsed_dec_forward(**DEC_PARAMS)
    assert pt.p_net > 0


def test_dec_forward_has_dec_output():
    pt = pulsed_dec_forward(**DEC_PARAMS)
    assert pt.p_dee > 0


def test_dec_forward_no_thermal_when_eta_th_zero():
    pt = pulsed_dec_forward(**DEC_PARAMS)
    assert pt.p_the == 0.0


def test_dec_forward_pulsed_fields():
    pt = pulsed_dec_forward(**DEC_PARAMS)
    assert pt.e_driver_mj == 12.0
    assert pt.f_rep == 1.0
    assert abs(pt.e_stored_mj - 12.0 / 0.95) < 0.1
    assert pt.f_ch > 0.9  # D-He3 ~0.94


def test_dec_forward_driver_recirc_is_losses_only():
    pt = pulsed_dec_forward(**DEC_PARAMS)
    assert pt.rec_frac < 0.5


def test_dec_forward_with_thermal_bop():
    """D-T with DEC + thermal BOP for neutrons."""
    params_dt = dict(
        p_fus=2000.0,
        fuel=Fuel.DT,
        e_driver_mj=50.0,
        f_rep=1.0,
        mn=1.1,
        eta_th=0.40,
        eta_pin=0.90,
        eta_dec=0.85,
        f_pdv=0.75,
        f_rad=0.10,
        f_sub=0.03,
        p_pump=1.0,
        p_trit=10.0,
        p_house=4.0,
        p_cryo=0.0,
        p_target=0.0,
        p_coils=0.5,
    )
    pt = pulsed_dec_forward(**params_dt)
    assert pt.p_dee > 0
    assert pt.p_the > 0
    assert pt.p_et > pt.p_dee  # gross = DEC + thermal
    assert pt.p_net > 0


def test_dec_inverse_roundtrip():
    pt = pulsed_dec_forward(**DEC_PARAMS)
    # Inverse takes q_sci instead of e_driver_mj, returns (p_fus, e_driver_mj)
    inv_params = {
        k: v for k, v in DEC_PARAMS.items() if k not in ("p_fus", "fuel", "e_driver_mj")
    }
    p_fus_recovered, e_driver_recovered = pulsed_dec_inverse(
        p_net_target=pt.p_net, fuel=Fuel.DHE3, q_sci=pt.q_sci, **inv_params
    )
    assert abs(p_fus_recovered - 500.0) < 0.5, f"Expected ~500, got {p_fus_recovered}"
    assert abs(e_driver_recovered - 12.0) < 0.5, (
        f"Expected ~12, got {e_driver_recovered}"
    )
