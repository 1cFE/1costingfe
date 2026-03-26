import jax

from costingfe.layers.physics import ash_neutron_split
from costingfe.layers.radiation import (
    compute_impurity_fraction,
    compute_p_line,
    compute_p_rad,
    compute_sputtering_yield,
    cooling_rate,
)
from costingfe.types import Fuel, ImpurityMix, WallMaterial

# Default fuel fractions (match YAML defaults)
_FUEL_FRACS = dict(
    dd_f_T=0.969,
    dd_f_He3=0.689,
    dhe3_dd_frac=0.07,
    dhe3_f_T=0.97,
    pb11_f_alpha_n=0.0,
    pb11_f_p_n=0.0,
)


def test_dt_ash_fraction():
    """DT: alpha carries 3.52 MeV of 17.58 MeV total -> ~20.02% charged."""
    p_fus = 1000.0
    p_ash, p_neutron = ash_neutron_split(p_fus, Fuel.DT, **_FUEL_FRACS)
    assert abs(p_ash / p_fus - 0.2002) < 0.001
    assert abs((p_ash + p_neutron) - p_fus) < 0.001  # energy conservation


def test_pb11_fully_aneutronic():
    """pB11: 100% charged particles (3 alphas)."""
    p_fus = 500.0
    p_ash, p_neutron = ash_neutron_split(p_fus, Fuel.PB11, **_FUEL_FRACS)
    assert abs(p_ash - p_fus) < 0.001
    assert abs(p_neutron) < 0.001


def test_dd_semi_catalyzed():
    """DD: semi-catalyzed burn with defaults should give ~56% charged."""
    p_fus = 1000.0
    p_ash, p_neutron = ash_neutron_split(p_fus, Fuel.DD, **_FUEL_FRACS)
    ash_frac = p_ash / p_fus
    assert 0.50 < ash_frac < 0.65
    assert abs((p_ash + p_neutron) - p_fus) < 0.001


def test_dhe3_mostly_aneutronic():
    """DHe3: primary aneutronic with ~7% DD side reactions -> ~95% charged."""
    p_fus = 1000.0
    p_ash, p_neutron = ash_neutron_split(p_fus, Fuel.DHE3, **_FUEL_FRACS)
    ash_frac = p_ash / p_fus
    assert 0.93 < ash_frac < 0.97
    assert abs((p_ash + p_neutron) - p_fus) < 0.001


def test_ash_neutron_split_is_jax_differentiable():
    """Verify JAX can differentiate through the ash/neutron split."""

    def lcoe_proxy(p_fus):
        p_ash, p_neutron = ash_neutron_split(p_fus, Fuel.DT, **_FUEL_FRACS)
        return p_ash

    grad_fn = jax.grad(lcoe_proxy)
    grad_val = grad_fn(1000.0)
    assert abs(grad_val - 0.2002) < 0.001


# ---------------------------------------------------------------------------
# Impurity line radiation model tests
# ---------------------------------------------------------------------------


def test_sputtering_yield_tungsten():
    """W yield at ~100 eV edge (300 eV ion energy) should be very small."""
    Y = compute_sputtering_yield(0.1, WallMaterial.TUNGSTEN)
    assert 0 < float(Y) < 0.01  # Very low yield for W


def test_sputtering_yield_carbon():
    """C yield should be much higher than W at the same edge temperature."""
    Y_C = compute_sputtering_yield(0.1, WallMaterial.CARBON)
    Y_W = compute_sputtering_yield(0.1, WallMaterial.TUNGSTEN)
    assert float(Y_C) > float(Y_W)


def test_cooling_rate_sanity():
    """L_W(10 keV) >> L_C(10 keV) per atom — W has many more electrons."""
    L_W = cooling_rate("W", 10.0)
    L_C = cooling_rate("C", 10.0)
    assert float(L_W) > float(L_C) * 10  # W radiates far more per atom at 10 keV


def test_p_line_tungsten_wall():
    """Full chain: W wall -> f_W -> P_line, check order of magnitude."""
    symbol, f_z = compute_impurity_fraction(
        WallMaterial.TUNGSTEN,
        T_edge_keV=0.1,
        fw_area=600.0,
        plasma_volume=500.0,
        tau_ratio=3.0,
    )
    assert symbol == "W"
    assert float(f_z) > 0

    impurities = ImpurityMix(wall_derived={symbol: f_z}, seeded={})
    n_e = 1e20
    T_e = 15.0
    V = 500.0
    p_line = compute_p_line(n_e, T_e, impurities, V)
    assert float(p_line) > 0  # Should produce some radiation
    assert float(p_line) < 100  # But not absurdly large


def test_p_line_seeded_argon():
    """Seeded Ar at 0.2% -> P_line should be significant."""
    impurities = ImpurityMix(wall_derived={}, seeded={"Ar": 0.002})
    n_e = 1e20
    T_e = 15.0
    V = 500.0
    p_line = compute_p_line(n_e, T_e, impurities, V)
    # At 0.2% Ar, n_e=1e20, T_e=15 keV, V=500 m³ → expect tens of MW
    assert float(p_line) > 1.0


def test_compute_p_rad_with_impurities():
    """P_rad increases when impurities are added."""
    n_e, T_e, Z_eff, V, B = 1e20, 15.0, 1.5, 500.0, 5.0
    p_rad_no_imp = compute_p_rad(n_e, T_e, Z_eff, V, B, impurities=None)
    impurities = ImpurityMix(wall_derived={}, seeded={"Ar": 0.002})
    p_rad_with_imp = compute_p_rad(n_e, T_e, Z_eff, V, B, impurities=impurities)
    assert float(p_rad_with_imp) > float(p_rad_no_imp)


def test_no_impurities_unchanged():
    """Backwards compatibility: wall_material=None -> same result as before."""
    n_e, T_e, Z_eff, V, B = 1e20, 15.0, 1.5, 500.0, 5.0
    p_rad_old = compute_p_rad(n_e, T_e, Z_eff, V, B)
    p_rad_new = compute_p_rad(n_e, T_e, Z_eff, V, B, impurities=None)
    assert float(p_rad_old) == float(p_rad_new)


def test_lithium_wall_low_radiation():
    """Li wall should produce less line radiation than W wall."""
    n_e, T_e, V = 1e20, 15.0, 500.0
    fw_area = 600.0

    _, f_W = compute_impurity_fraction(WallMaterial.TUNGSTEN, 0.1, fw_area, V, 3.0)
    _, f_Li = compute_impurity_fraction(WallMaterial.LITHIUM, 0.1, fw_area, V, 3.0)

    imp_W = ImpurityMix(wall_derived={"W": f_W}, seeded={})
    imp_Li = ImpurityMix(wall_derived={"Li": f_Li}, seeded={})

    p_line_W = compute_p_line(n_e, T_e, imp_W, V)
    p_line_Li = compute_p_line(n_e, T_e, imp_Li, V)
    assert float(p_line_Li) < float(p_line_W)
