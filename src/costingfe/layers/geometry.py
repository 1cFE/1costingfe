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

from costingfe.types import CONCEPT_TO_FAMILY, ConfinementConcept, ConfinementFamily


@dataclass(frozen=True)
class RadialBuild:
    """Input radial build thicknesses (meters), from center outward."""

    # Core geometry
    R0: float = 6.2  # Major radius R0 (tokamak) or chamber radius (mirror/IFE)
    plasma_t: float = 2.0  # Minor radius a (tokamak) or plasma thickness
    elon: float = 1.0  # Elongation kappa (tokamak only, 1.0 = circular)
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
    coil_t: float = 0.30  # TF coil (MFE only, 0 for IFE)
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
    blanket_or: float  # Outer radius of blanket
    vessel_or: float  # Outer radius of vessel
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


def chamber_radius_m(
    yield_per_shot_mj: float,
    r_ref_m: float,
    yield_ref_mj: float,
    wall_improvement_factor: float,
    p_neutron_mw: float = 0.0,
    neutron_wall_load_max_mw_m2: float = 0.0,
) -> float:
    """First-wall radius [m] for an IFE/MIF chamber = the larger of TWO vessel
    constraints, ``R_fw = max(R_fluence, R_power)``:

    1. Per-shot survivability (fluence), which PENALIZES high single-shot yield:

           R_fluence = r_ref_m * sqrt(yield_per_shot_mj / (yield_ref_mj * f_wall))

       Carried from GEM's HAPL dry-wall chamber (Sviatoslavsky et al., FST 47,
       535 (2005)): r_ref_m = 6.5 m at yield_ref_mj = 150 MJ. The fluence limit
       is a materials-physics property (wall neutron-dpa / thermal-shock
       tolerance), not a learning-curve quantity, so the FOAK constant is a valid
       NOAK dry-wall base. ``wall_improvement_factor`` (f_wall >= 1) raises the
       tolerable areal fluence and shrinks the chamber:
         - 1.0    : GEM dry wall (default)
         - ~1.5-3 : advanced dry-wall materials (NOAK material advancement)
         - ~40-60 : thick-liquid wall (HYLIFE-II / Xcimer) — the flowing liquid
           absorbs the pulse and moves the structural wall inward, pulling the
           naive ~22 m dry-wall radius at 1.8 GJ down to a HYLIFE-class ~3 m.
       The liquid inventory itself is costed separately in CAS27, so f_wall
       shrinks only the structural chamber (no double benefit).

    2. Time-averaged neutron wall loading (power density), which PENALIZES high
       total power. A spherical first wall of area 4*pi*R^2 carrying neutron
       power P_n must keep Gamma_n = P_n / (4*pi*R^2) below a wall-type limit:

           R_power = sqrt(P_n / (4*pi * neutron_wall_load_max_mw_m2))

       Without this floor a LOW-yield/HIGH-rep concept gets a tiny fluence-sized
       chamber carrying tens of MW/m^2 — a first-wall power density no real wall
       survives. That is a free pass for exactly the high-rep design corner the
       fluence term (constraint 1) is supposed to reward, so the two constraints
       must both bind. Pass p_neutron_mw <= 0 (or limit <= 0) to disable the
       floor (pure-fluence behaviour, e.g. unit tests / legacy callers).

    See docs/plans/2026-07-07-target-yield-sizing-design.md (D2).
    """
    r_fluence = r_ref_m * math.sqrt(
        yield_per_shot_mj / (yield_ref_mj * wall_improvement_factor)
    )
    if p_neutron_mw > 0.0 and neutron_wall_load_max_mw_m2 > 0.0:
        r_power = math.sqrt(
            p_neutron_mw / (4.0 * math.pi * neutron_wall_load_max_mw_m2)
        )
        return max(r_fluence, r_power)
    return r_fluence


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
    if family == ConfinementFamily.STEADY_STATE and concept in (
        ConfinementConcept.MIRROR,
        ConfinementConcept.STEADY_FRC,
    ):
        h = rb.chamber_length

        def vol(r_in, r_out):
            return _cylinder_shell_volume(h, r_in, r_out)

        firstwall_area = 2 * math.pi * vacuum_or * h
    elif concept == ConfinementConcept.DIPOLE:
        # Levitated dipole: roughly spherical vessel surrounding a floating
        # ring coil. The torus branch is unsafe here because the minor radius
        # exceeds the major radius for dipole-class machines (Simpson 2026
        # Reactor A: inner VV ~20.6 m), causing the torus to self-intersect.
        def vol(r_in, r_out):
            return _sphere_shell_volume(r_in, r_out)

        firstwall_area = 4 * math.pi * vacuum_or**2
    elif family == ConfinementFamily.STEADY_STATE:
        # Tokamak / stellarator: torus
        R = rb.R0
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
