"""Layer 3: Engineering — radial build geometry for reactor components.

Computes component radii, volumes, and surface areas from radial build
thicknesses. Different volume formulas per concept:
  - Tokamak: hollow torus (2*pi*R * pi*a^2)
  - Mirror: cylindrical ring (height * pi * (r_out^2 - r_in^2))
  - IFE/MIF: spherical shell (4/3 * pi * (r_out^3 - r_in^3))

Source: pyFECONs costing/calculations/volume.py, cas220101_reactor_equipment.py
"""

import math
from dataclasses import dataclass

from costingfe.types import ConfinementConcept, CONCEPT_TO_FAMILY, ConfinementFamily


@dataclass(frozen=True)
class RadialBuild:
    """Input radial build thicknesses (meters), from center outward."""

    # Core geometry
    axis_t: float = 6.2   # Major radius R0 (tokamak) or chamber radius (mirror/IFE)
    plasma_t: float = 2.0  # Minor radius a (tokamak) or plasma thickness
    elon: float = 1.0      # Elongation kappa (tokamak only, 1.0 = circular)
    chamber_length: float = 0.0  # Chamber length (mirror only)

    # Radial build layers (center → outboard)
    vacuum_t: float = 0.10
    firstwall_t: float = 0.05
    blanket_t: float = 0.70
    reflector_t: float = 0.20
    ht_shield_t: float = 0.20
    structure_t: float = 0.15
    gap1_t: float = 0.10
    vessel_t: float = 0.10
    coil_t: float = 0.30    # TF coil (MFE only, 0 for IFE)
    gap2_t: float = 0.10
    lt_shield_t: float = 0.15
    bioshield_t: float = 1.00


@dataclass(frozen=True)
class Geometry:
    """Computed geometry: radii, volumes (m^3), and surface areas (m^2)."""

    # Component volumes [m^3]
    plasma_vol: float
    firstwall_vol: float
    blanket_vol: float
    reflector_vol: float
    ht_shield_vol: float
    structure_vol: float
    vessel_vol: float
    lt_shield_vol: float
    bioshield_vol: float

    # Surface areas [m^2]
    firstwall_area: float  # Inner surface of first wall (plasma-facing)

    # Key outer radii [m]
    blanket_or: float   # Outer radius of blanket
    vessel_or: float    # Outer radius of vessel
    bioshield_or: float  # Outer radius of bioshield


def _torus_shell_volume(R: float, r_in: float, r_out: float, kappa: float) -> float:
    """Volume of a toroidal shell with elongation.

    V = kappa * 2*pi*R * pi*(r_out^2 - r_in^2)
    where R = major radius, r = minor radii (measured from magnetic axis).
    """
    return kappa * 2 * math.pi * R * math.pi * (r_out**2 - r_in**2)


def _torus_surface_area(R: float, a: float, kappa: float) -> float:
    """Surface area of a torus (approximate with elongation).

    SA ≈ kappa * 4*pi^2*R*a
    """
    return kappa * 4 * math.pi**2 * R * a


def _cylinder_shell_volume(height: float, r_in: float, r_out: float) -> float:
    """Volume of a cylindrical ring."""
    return height * math.pi * (r_out**2 - r_in**2)


def _sphere_shell_volume(r_in: float, r_out: float) -> float:
    """Volume of a spherical shell."""
    return (4.0 / 3.0) * math.pi * (r_out**3 - r_in**3)


def compute_geometry(rb: RadialBuild, concept: ConfinementConcept) -> Geometry:
    """Compute component volumes and surface areas from radial build.

    Dispatches volume formula by concept family:
    - MFE tokamak/stellarator: torus
    - MFE mirror: cylinder
    - IFE/MIF: sphere
    """
    family = CONCEPT_TO_FAMILY[concept]

    # Cumulative radii from magnetic axis outward
    # (for tokamak: measured from axis, so plasma outer = a)
    plasma_or = rb.plasma_t
    vacuum_or = plasma_or + rb.vacuum_t
    firstwall_or = vacuum_or + rb.firstwall_t
    blanket_or = firstwall_or + rb.blanket_t
    reflector_or = blanket_or + rb.reflector_t
    ht_shield_or = reflector_or + rb.ht_shield_t
    structure_or = ht_shield_or + rb.structure_t
    gap1_or = structure_or + rb.gap1_t
    vessel_or = gap1_or + rb.vessel_t
    coil_or = vessel_or + rb.coil_t
    gap2_or = coil_or + rb.gap2_t
    lt_shield_or = gap2_or + rb.lt_shield_t
    bioshield_or = lt_shield_or + rb.bioshield_t

    # Select volume function
    if family == ConfinementFamily.MFE and concept == ConfinementConcept.MIRROR:
        h = rb.chamber_length

        def vol(r_in, r_out):
            return _cylinder_shell_volume(h, r_in, r_out)

        firstwall_area = 2 * math.pi * vacuum_or * h
    elif family == ConfinementFamily.MFE:
        # Tokamak / stellarator: torus
        R = rb.axis_t
        k = rb.elon

        def vol(r_in, r_out):
            return _torus_shell_volume(R, r_in, r_out, k)

        firstwall_area = _torus_surface_area(R, vacuum_or, k)
    else:
        # IFE / MIF: spherical
        def vol(r_in, r_out):
            return _sphere_shell_volume(r_in, r_out)

        firstwall_area = 4 * math.pi * vacuum_or**2

    return Geometry(
        plasma_vol=vol(0, plasma_or),
        firstwall_vol=vol(vacuum_or, firstwall_or),
        blanket_vol=vol(firstwall_or, blanket_or),
        reflector_vol=vol(blanket_or, reflector_or),
        ht_shield_vol=vol(reflector_or, ht_shield_or),
        structure_vol=vol(ht_shield_or, structure_or),
        vessel_vol=vol(gap1_or, vessel_or),
        lt_shield_vol=vol(gap2_or, lt_shield_or),
        bioshield_vol=vol(lt_shield_or, bioshield_or),
        firstwall_area=firstwall_area,
        blanket_or=blanket_or,
        vessel_or=vessel_or,
        bioshield_or=bioshield_or,
    )
