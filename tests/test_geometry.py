import math

from costingfe.layers.geometry import (
    RadialBuild,
    compute_geometry,
)
from costingfe.types import ConfinementConcept

# ARC-like tokamak radial build (from pyFECONs customers/ARC)
ARC_BUILD = RadialBuild(
    R0=3.3,
    plasma_t=1.13,
    elon=1.84,
    vacuum_t=0.10,
    firstwall_t=0.10,
    blanket_t=0.70,
    reflector_t=0.20,
    ht_shield_t=0.20,
    structure_t=0.15,
    gap1_t=0.10,
    vessel_t=0.10,
    coil_t=0.30,
    gap2_t=0.10,
    lt_shield_t=0.15,
    bioshield_t=1.00,
)


def test_tokamak_volumes_positive():
    """All computed volumes should be positive."""
    geo = compute_geometry(ARC_BUILD, ConfinementConcept.TOKAMAK)
    assert geo.plasma_vol > 0
    assert geo.firstwall_vol > 0
    assert geo.blanket_vol > 0
    assert geo.ht_shield_vol > 0
    assert geo.vessel_vol > 0
    assert geo.bioshield_vol > 0


def test_tokamak_blanket_larger_than_firstwall():
    """Blanket (0.70m thick) should have more volume than first wall (0.10m)."""
    geo = compute_geometry(ARC_BUILD, ConfinementConcept.TOKAMAK)
    assert geo.blanket_vol > geo.firstwall_vol


def test_tokamak_firstwall_area_reasonable():
    """First wall area for ARC-like machine should be ~200-400 m^2."""
    geo = compute_geometry(ARC_BUILD, ConfinementConcept.TOKAMAK)
    assert 100 < geo.firstwall_area < 1000


def test_tokamak_plasma_volume_reasonable():
    """Plasma volume for ARC-like machine should be ~100-300 m^3."""
    geo = compute_geometry(ARC_BUILD, ConfinementConcept.TOKAMAK)
    # ARC: V_p ≈ kappa * 2*pi*R * pi*a^2 ≈ 1.84 * 2*pi*3.3 * pi*1.13^2 ≈ 153 m^3
    assert 100 < geo.plasma_vol < 300


def test_mirror_uses_cylinder():
    """Mirror geometry should use cylindrical volumes."""
    rb = RadialBuild(
        R0=2.0,
        plasma_t=1.0,
        chamber_length=10.0,
        coil_t=0.0,  # Mirror has no TF coils in radial build
    )
    geo = compute_geometry(rb, ConfinementConcept.MIRROR)
    # Plasma volume = height * pi * r^2 = 10 * pi * 1^2 ≈ 31.4
    assert abs(geo.plasma_vol - 10 * math.pi) < 0.1
    assert geo.blanket_vol > 0


def test_ife_uses_sphere():
    """IFE geometry should use spherical volumes."""
    rb = RadialBuild(
        R0=5.0,  # Chamber radius
        plasma_t=1.0,
        coil_t=0.0,
    )
    geo = compute_geometry(rb, ConfinementConcept.LASER_IFE)
    # Plasma vol = 4/3 * pi * r^3 = 4/3 * pi * 1^3 ≈ 4.19
    assert abs(geo.plasma_vol - (4.0 / 3.0) * math.pi) < 0.1
    assert geo.blanket_vol > 0


def test_mif_uses_sphere():
    """MIF geometry should also use spherical volumes."""
    rb = RadialBuild(
        R0=3.0,
        plasma_t=0.5,
        coil_t=0.0,
    )
    geo = compute_geometry(rb, ConfinementConcept.MAG_TARGET)
    assert geo.plasma_vol > 0
    assert geo.blanket_vol > 0


def test_elongation_scales_volume():
    """Higher elongation should give larger volumes."""
    rb_circ = RadialBuild(R0=3.3, plasma_t=1.13, elon=1.0)
    rb_elong = RadialBuild(R0=3.3, plasma_t=1.13, elon=2.0)
    geo_circ = compute_geometry(rb_circ, ConfinementConcept.TOKAMAK)
    geo_elong = compute_geometry(rb_elong, ConfinementConcept.TOKAMAK)
    assert geo_elong.plasma_vol > geo_circ.plasma_vol
    ratio = geo_elong.plasma_vol / geo_circ.plasma_vol
    assert abs(ratio - 2.0) < 0.01  # Should scale linearly with kappa


def test_outer_radii_increase_outward():
    """Outer radii should increase: blanket < vessel < bioshield."""
    geo = compute_geometry(ARC_BUILD, ConfinementConcept.TOKAMAK)
    assert geo.blanket_or < geo.vessel_or < geo.bioshield_or
