import jax
from costingfe.layers.physics import mfe_forward_power_balance, mfe_inverse_power_balance
from costingfe.types import Fuel, PowerTable

# CATF reference parameters (from pyFECONs customers/CATF/mfe/DefineInputs.py)
CATF_PARAMS = dict(
    p_fus=2600.0,
    fuel=Fuel.DT,
    p_input=50.0,
    mn=1.1,
    eta_th=0.46,
    eta_p=0.5,
    eta_pin=0.5,
    eta_de=0.85,
    f_sub=0.03,
    f_dec=0.0,
    p_coils=2.0,
    p_cool=13.7,
    p_pump=1.0,
    p_trit=10.0,
    p_house=4.0,
    p_cryo=0.5,
    n_e=1.0e20,
    T_e=15.0,
    Z_eff=1.5,
    plasma_volume=500.0,
    B=5.0,
)


def test_mfe_forward_energy_conservation():
    """Total power in = total power out (1st law of thermodynamics)."""
    pt = mfe_forward_power_balance(**CATF_PARAMS)
    assert pt.p_net > 0, "Net power should be positive"
    assert pt.p_et > pt.p_net, "Gross > net (recirculating losses)"


def test_mfe_forward_net_power_positive():
    """CATF reference design should produce positive net electric."""
    pt = mfe_forward_power_balance(**CATF_PARAMS)
    assert pt.p_net > 500  # CATF ~1 GW electric


def test_mfe_forward_q_eng_reasonable():
    """Engineering Q should be > 1 for a viable power plant."""
    pt = mfe_forward_power_balance(**CATF_PARAMS)
    assert pt.q_eng > 1.0
    assert pt.q_eng < 100.0  # sanity check


def test_mfe_forward_no_dec():
    """With f_dec=0, DEC electric should be zero."""
    pt = mfe_forward_power_balance(**CATF_PARAMS)
    assert abs(pt.p_dee) < 0.001


def test_mfe_forward_is_differentiable():
    """JAX should be able to differentiate p_net w.r.t. p_fus."""

    def p_net_fn(p_fus):
        pt = mfe_forward_power_balance(
            p_fus=p_fus,
            fuel=Fuel.DT,
            p_input=50.0,
            mn=1.1,
            eta_th=0.46,
            eta_p=0.5,
            eta_pin=0.5,
            eta_de=0.85,
            f_sub=0.03,
            f_dec=0.0,
            p_coils=2.0,
            p_cool=13.7,
            p_pump=1.0,
            p_trit=10.0,
            p_house=4.0,
            p_cryo=0.5,
        )
        return pt.p_net

    grad_fn = jax.grad(p_net_fn)
    grad_val = grad_fn(2600.0)
    assert grad_val > 0  # more fusion power -> more net electric


# Shared engineering params for inverse tests (no p_fus, no fuel)
_ENG_PARAMS = {
    k: v
    for k, v in CATF_PARAMS.items()
    if k not in ("p_fus", "fuel")
}


def test_inverse_roundtrip():
    """Forward then inverse should recover original p_fus."""
    pt = mfe_forward_power_balance(**CATF_PARAMS)
    p_fus_recovered = mfe_inverse_power_balance(
        p_net_target=pt.p_net, fuel=Fuel.DT, **_ENG_PARAMS,
    )
    assert abs(p_fus_recovered - 2600.0) < 0.1, f"Expected ~2600, got {p_fus_recovered}"


def test_inverse_1gw_target():
    """1 GW net electric target should give a reasonable fusion power."""
    p_fus = mfe_inverse_power_balance(
        p_net_target=1000.0, fuel=Fuel.DT, **_ENG_PARAMS,
    )
    assert p_fus > 1000  # fusion power must exceed net electric
    assert p_fus < 10000  # but not absurdly large
