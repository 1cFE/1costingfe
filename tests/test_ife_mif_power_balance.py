from costingfe.layers.physics import (
    ife_forward_power_balance,
    ife_inverse_power_balance,
    mif_forward_power_balance,
    mif_inverse_power_balance,
)
from costingfe.types import Fuel

# IFE reference parameters (from pyFECONs CATF IFE config)
IFE_PARAMS = dict(
    p_fus=2500.0,
    fuel=Fuel.DT,
    p_implosion=10.0,
    p_ignition=0.1,
    mn=1.1,
    eta_th=0.46,
    eta_p=0.5,
    eta_pin1=0.10,
    eta_pin2=0.10,
    f_sub=0.03,
    p_pump=1.0,
    p_trit=10.0,
    p_house=4.0,
    p_cryo=0.5,
    p_target=1.0,
)

# MIF reference parameters
MIF_PARAMS = dict(
    p_fus=2500.0,
    fuel=Fuel.DT,
    p_driver=30.0,
    mn=1.1,
    eta_th=0.40,
    eta_p=0.5,
    eta_pin=0.30,
    f_sub=0.03,
    p_pump=1.0,
    p_trit=10.0,
    p_house=4.0,
    p_cryo=0.2,
    p_target=2.0,
    p_coils=0.5,
)


# ---------------------------------------------------------------------------
# IFE tests
# ---------------------------------------------------------------------------


def test_ife_forward_net_power_positive():
    """IFE reference design should produce positive net electric."""
    pt = ife_forward_power_balance(**IFE_PARAMS)
    assert pt.p_net > 0


def test_ife_no_dec():
    """IFE should have zero DEC (no direct energy conversion)."""
    pt = ife_forward_power_balance(**IFE_PARAMS)
    assert pt.p_dee == 0.0
    assert pt.p_dec_waste == 0.0


def test_ife_no_coils():
    """IFE should have zero coil power (no magnets)."""
    pt = ife_forward_power_balance(**IFE_PARAMS)
    assert pt.p_coils == 0.0
    assert pt.p_cool == 0.0


def test_ife_has_target_power():
    """IFE should record target factory power."""
    pt = ife_forward_power_balance(**IFE_PARAMS)
    assert pt.p_target == 1.0


def test_ife_high_recirculating_fraction():
    """IFE with low driver efficiency should have high recirculating fraction."""
    pt = ife_forward_power_balance(**IFE_PARAMS)
    # eta_pin1=0.10 means 10x power draw for implosion laser
    assert pt.rec_frac > 0.05  # significant recirculating fraction


def test_ife_inverse_roundtrip():
    """Forward then inverse should recover original p_fus."""
    pt = ife_forward_power_balance(**IFE_PARAMS)
    eng = {k: v for k, v in IFE_PARAMS.items() if k not in ("p_fus", "fuel")}
    p_fus_recovered = ife_inverse_power_balance(
        p_net_target=pt.p_net, fuel=Fuel.DT, **eng,
    )
    assert abs(p_fus_recovered - 2500.0) < 0.1, f"Expected ~2500, got {p_fus_recovered}"


def test_ife_inverse_1gw_target():
    """1 GW net electric target should give reasonable fusion power."""
    eng = {k: v for k, v in IFE_PARAMS.items() if k not in ("p_fus", "fuel")}
    p_fus = ife_inverse_power_balance(
        p_net_target=1000.0, fuel=Fuel.DT, **eng,
    )
    assert p_fus > 1000
    assert p_fus < 10000


# ---------------------------------------------------------------------------
# MIF tests
# ---------------------------------------------------------------------------


def test_mif_forward_net_power_positive():
    """MIF reference design should produce positive net electric."""
    pt = mif_forward_power_balance(**MIF_PARAMS)
    assert pt.p_net > 0


def test_mif_no_dec():
    """MIF should have zero DEC."""
    pt = mif_forward_power_balance(**MIF_PARAMS)
    assert pt.p_dee == 0.0
    assert pt.p_dec_waste == 0.0


def test_mif_has_target_power():
    """MIF should record liner/target factory power."""
    pt = mif_forward_power_balance(**MIF_PARAMS)
    assert pt.p_target == 2.0


def test_mif_lower_efficiency_than_mfe():
    """MIF with eta_th=0.40 should have lower thermal efficiency."""
    pt = mif_forward_power_balance(**MIF_PARAMS)
    assert pt.p_the < pt.p_th  # thermal conversion < total thermal


def test_mif_inverse_roundtrip():
    """Forward then inverse should recover original p_fus."""
    pt = mif_forward_power_balance(**MIF_PARAMS)
    eng = {k: v for k, v in MIF_PARAMS.items() if k not in ("p_fus", "fuel")}
    p_fus_recovered = mif_inverse_power_balance(
        p_net_target=pt.p_net, fuel=Fuel.DT, **eng,
    )
    assert abs(p_fus_recovered - 2500.0) < 0.1, f"Expected ~2500, got {p_fus_recovered}"


def test_mif_inverse_1gw_target():
    """1 GW net electric target should give reasonable fusion power."""
    eng = {k: v for k, v in MIF_PARAMS.items() if k not in ("p_fus", "fuel")}
    p_fus = mif_inverse_power_balance(
        p_net_target=1000.0, fuel=Fuel.DT, **eng,
    )
    assert p_fus > 1000
    assert p_fus < 10000
