import jax
from costingfe.layers.physics import ash_neutron_split
from costingfe.types import Fuel


def test_dt_ash_fraction():
    """DT: alpha carries 3.52 MeV of 17.58 MeV total -> ~20.02% charged."""
    p_fus = 1000.0
    p_ash, p_neutron = ash_neutron_split(p_fus, Fuel.DT)
    assert abs(p_ash / p_fus - 0.2002) < 0.001
    assert abs((p_ash + p_neutron) - p_fus) < 0.001  # energy conservation


def test_pb11_fully_aneutronic():
    """pB11: 100% charged particles (3 alphas)."""
    p_fus = 500.0
    p_ash, p_neutron = ash_neutron_split(p_fus, Fuel.PB11)
    assert abs(p_ash - p_fus) < 0.001
    assert abs(p_neutron) < 0.001


def test_dd_semi_catalyzed():
    """DD: semi-catalyzed burn with defaults should give ~56% charged."""
    p_fus = 1000.0
    p_ash, p_neutron = ash_neutron_split(p_fus, Fuel.DD)
    ash_frac = p_ash / p_fus
    assert 0.50 < ash_frac < 0.65
    assert abs((p_ash + p_neutron) - p_fus) < 0.001


def test_dhe3_mostly_aneutronic():
    """DHe3: primary aneutronic with ~7% DD side reactions -> ~95% charged."""
    p_fus = 1000.0
    p_ash, p_neutron = ash_neutron_split(p_fus, Fuel.DHE3)
    ash_frac = p_ash / p_fus
    assert 0.93 < ash_frac < 0.97
    assert abs((p_ash + p_neutron) - p_fus) < 0.001


def test_ash_neutron_split_is_jax_differentiable():
    """Verify JAX can differentiate through the ash/neutron split."""

    def lcoe_proxy(p_fus):
        p_ash, p_neutron = ash_neutron_split(p_fus, Fuel.DT)
        return p_ash

    grad_fn = jax.grad(lcoe_proxy)
    grad_val = grad_fn(1000.0)
    assert abs(grad_val - 0.2002) < 0.001
