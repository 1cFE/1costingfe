from costingfe.defaults import load_costing_constants
from costingfe.layers.cas22 import cas22_reactor_plant_equipment
from costingfe.layers.geometry import RadialBuild, compute_geometry
from costingfe.types import ConfinementConcept, Fuel

CC = load_costing_constants()

# Reference tokamak geometry for tests
RB = RadialBuild(axis_t=6.2, plasma_t=2.0, elon=1.7, blanket_t=0.70)
GEO = compute_geometry(RB, ConfinementConcept.TOKAMAK)
BLANKET_VOL = GEO.firstwall_vol + GEO.blanket_vol + GEO.reflector_vol
SHIELD_VOL = GEO.ht_shield_vol + GEO.lt_shield_vol
STRUCTURE_VOL = GEO.structure_vol
VESSEL_VOL = GEO.vessel_vol


def _make_cas22(fuel=Fuel.DT, n_mod=1, blanket_t=0.70):
    """Helper to compute CAS22 with geometry."""
    rb = RadialBuild(axis_t=6.2, plasma_t=2.0, elon=1.7, blanket_t=blanket_t)
    geo = compute_geometry(rb, ConfinementConcept.TOKAMAK)
    return cas22_reactor_plant_equipment(
        CC, p_net=1000.0, p_th=2500.0, p_et=1100.0, p_fus=2300.0,
        p_cryo=0.5, n_mod=n_mod, fuel=fuel, noak=True,
        blanket_vol=geo.firstwall_vol + geo.blanket_vol + geo.reflector_vol,
        shield_vol=geo.ht_shield_vol + geo.lt_shield_vol,
        structure_vol=geo.structure_vol,
        vessel_vol=geo.vessel_vol,
    )


def test_cas22_dt_has_breeding_blanket():
    """DT should include tritium breeding blanket cost."""
    result = _make_cas22(fuel=Fuel.DT)
    assert result["C220101"] > 0  # first wall + blanket
    assert result["C220000"] > 0  # total


def test_cas22_pb11_no_breeding():
    """pB11 should have cheaper blanket (no breeding)."""
    dt = _make_cas22(fuel=Fuel.DT)
    pb11 = _make_cas22(fuel=Fuel.PB11)
    assert pb11["C220101"] < dt["C220101"]  # no breeding blanket


def test_cas22_isotope_separation_fuel_dependent():
    """Isotope separation should vary by fuel type."""
    dt = _make_cas22(fuel=Fuel.DT)
    pb11 = _make_cas22(fuel=Fuel.PB11)
    assert dt["C220112"] > 0
    assert pb11["C220112"] > 0
    assert dt["C220112"] != pb11["C220112"]


def test_cas22_fuel_handling_tritium_containment():
    """DT should have much higher fuel handling cost (tritium containment)."""
    dt = _make_cas22(fuel=Fuel.DT)
    pb11 = _make_cas22(fuel=Fuel.PB11)
    assert dt["C220500"] > pb11["C220500"]


def test_cas22_scales_with_n_mod():
    """Total CAS22 should scale with number of modules."""
    single = _make_cas22(n_mod=1)
    double = _make_cas22(n_mod=2)
    assert double["C220000"] > single["C220000"]


def test_cas22_all_subaccounts_present():
    """Result should contain all expected sub-account keys."""
    result = _make_cas22()
    expected_keys = [
        "C220101", "C220102", "C220103", "C220104", "C220105",
        "C220106", "C220107", "C220108", "C220109", "C220111",
        "C220112", "C220119", "C220200", "C220300", "C220400",
        "C220500", "C220600", "C220700", "C220000",
    ]
    for key in expected_keys:
        assert key in result, f"Missing sub-account {key}"
        assert result[key] >= 0, f"Sub-account {key} is negative"


def test_cas22_blanket_scales_with_thickness():
    """Thicker blanket should cost more (volume-based costing)."""
    thin = _make_cas22(blanket_t=0.50)
    thick = _make_cas22(blanket_t=0.90)
    assert thick["C220101"] > thin["C220101"]
    assert thick["C220000"] > thin["C220000"]


def test_cas22_shield_volume_based():
    """Shield cost should be proportional to volume."""
    result = _make_cas22()
    expected_shield = CC.shield_unit_cost * SHIELD_VOL * 1.0  # DT scale=1.0
    assert abs(result["C220102"] - expected_shield) < 0.1


def test_cas22_structure_volume_based():
    """Structure cost should use volume-based costing."""
    result = _make_cas22()
    expected = CC.structure_unit_cost * STRUCTURE_VOL
    assert abs(result["C220105"] - expected) < 0.1
