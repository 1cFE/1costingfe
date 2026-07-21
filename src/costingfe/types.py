from dataclasses import dataclass, field
from enum import Enum


class ConfinementFamily(Enum):
    STEADY_STATE = "steady_state"
    PULSED = "pulsed"


class ConfinementConcept(Enum):
    TOKAMAK = "tokamak"
    STELLARATOR = "stellarator"
    MIRROR = "mirror"
    STEADY_FRC = "steady_frc"
    DIPOLE = "dipole"
    LASER_IFE = "laser_ife"
    ZPINCH = "zpinch"
    HEAVY_ION = "heavy_ion"
    MAG_TARGET = "mag_target"
    PLASMA_JET = "plasma_jet"
    PULSED_FRC = "pulsed_frc"
    MAGLIF = "maglif"
    THETA_PINCH = "theta_pinch"
    DENSE_PLASMA_FOCUS = "dense_plasma_focus"
    STAGED_ZPINCH = "staged_zpinch"
    ORBITRON = "orbitron"
    POLYWELL = "polywell"


class PulsedConversion(Enum):
    THERMAL = "thermal"
    INDUCTIVE_DEC = "inductive_dec"
    # Reactive/mechanically recovered drive + thermal (neutron) output. The
    # driver's per-pulse energy is largely recovered each cycle, so the stored,
    # consumed, and delivered energies decouple (e_store_mj / e_recirc_mj /
    # e_driver_mj). Used by THETA_PINCH (reactive ETS) and MAG_TARGET (mechanical
    # liner rebound). See pulsed_recovered_compression_forward.
    RECOVERED_COMPRESSION = "recovered_compression"


class LaserDriverType(Enum):
    """Laser-IFE driver architecture.

    Selects, for LASER_IFE, the C220104 capital coefficient ($/MJ), the CAS72
    scheduled-replacement subsystem set, and the wall-plug efficiency
    (eta_source_* by type). Chosen via the laser_driver_type parameter
    (concept-YAML default + per-run override), not a separate concept.

    DPSSL is the commercial baseline (LIFE / HiPER / Focused Energy / Marvel);
    KRF carries the NRL Electra / Xcimer heritage; FIBER is the coherent-combined
    fiber / blue-laser architecture (Blue Laser Fusion / XCAN); NDGLASS
    (NIF-class) is flagged commercially marginal — flashlamp shot life is
    Xe-arc-limited and its wall-plug efficiency is fundamentally low.
    """

    DPSSL = "dpssl"  # diode-pumped solid-state
    KRF = "krf"  # KrF excimer
    FIBER = "fiber"  # coherent-combined fiber / blue laser (BLF / XCAN)
    NDGLASS = "nd_glass"  # flashlamp-pumped Nd:Glass (NIF-class)


class WallType(Enum):
    """First-wall protection scheme for IFE/MIF chamber sizing.

    Selects the wall_improvement_factor default (tolerable areal fluence
    relative to the GEM/HAPL dry-wall base) used by
    ``layers/geometry.chamber_radius_m``. See
    docs/plans/2026-07-07-target-yield-sizing-design.md (D2).
    """

    DRY = "dry"  # GEM/HAPL dry solid wall (f_wall ~ 1)
    ADVANCED_DRY = "advanced_dry"  # advanced-material dry wall (f_wall ~ 1.5-3)
    THICK_LIQUID = "thick_liquid"  # HYLIFE/Xcimer liquid wall (f_wall ~ 40-60)


class DriveMode(Enum):
    """Laser-IFE drive configuration for the target-yield gain (design doc D-
    unified). Sets the coupling fraction (share of delivered driver energy that
    actually assembles/compresses the fuel) used by physics_yield_mj: the fuel
    mass loaded per driver joule scales with coupling, so gain is
    burn_fraction * loading * coupling * e_DT. The compressed fuel burns the same
    (burn_fraction is unchanged) — drive mode changes only how efficiently the
    driver assembles it.

    DIRECT: laser ablates the capsule directly — highest coupling (reference).
    HYBRID: combined / novel drive (e.g. Xcimer-class) — intermediate coupling.
    INDIRECT: laser -> hohlraum -> X-rays -> capsule; most driver energy is lost
      to X-ray conversion + hohlraum re-absorption, so the least fuel is
      assembled per joule -> lowest gain (and the hohlraum raises target cost).
    """

    DIRECT = "direct"
    HYBRID = "hybrid"
    INDIRECT = "indirect"


class TargetCostMode(Enum):
    """Per-shot target/consumable cost model (design doc D4). Size-scaled with
    driver energy so the per-shot cost differs between a small DPSSL capsule and
    a large Xcimer/MagLIF shot. Calibrated to reproduce the concept's existing
    NOAK target_unit_cost at its reference driver energy.

    METAL_LINER (MagLIF, Z-pinch): material-dominated — volumetric liner metal
      (scales ~E) * machining markup + recyclable-transmission-line remanufacture
      (~flat). Markup ~1.3x; the material leg carries the cost.
    CAPSULE_FAB (laser, heavy-ion): fabrication-dominated — cryo layering
      (~E, fuel volume), coating/metrology (~E^2/3, shell area), and per-target
      assembly (~flat), plus a negligible material floor.
    """

    METAL_LINER = "metal_liner"
    CAPSULE_FAB = "capsule_fab"


CONCEPT_TO_FAMILY = {
    ConfinementConcept.TOKAMAK: ConfinementFamily.STEADY_STATE,
    ConfinementConcept.STELLARATOR: ConfinementFamily.STEADY_STATE,
    ConfinementConcept.MIRROR: ConfinementFamily.STEADY_STATE,
    ConfinementConcept.STEADY_FRC: ConfinementFamily.STEADY_STATE,
    ConfinementConcept.DIPOLE: ConfinementFamily.STEADY_STATE,
    ConfinementConcept.LASER_IFE: ConfinementFamily.PULSED,
    ConfinementConcept.ZPINCH: ConfinementFamily.PULSED,
    ConfinementConcept.HEAVY_ION: ConfinementFamily.PULSED,
    ConfinementConcept.MAG_TARGET: ConfinementFamily.PULSED,
    ConfinementConcept.PLASMA_JET: ConfinementFamily.PULSED,
    ConfinementConcept.PULSED_FRC: ConfinementFamily.PULSED,
    ConfinementConcept.MAGLIF: ConfinementFamily.PULSED,
    ConfinementConcept.THETA_PINCH: ConfinementFamily.PULSED,
    ConfinementConcept.DENSE_PLASMA_FOCUS: ConfinementFamily.PULSED,
    ConfinementConcept.STAGED_ZPINCH: ConfinementFamily.PULSED,
    ConfinementConcept.ORBITRON: ConfinementFamily.STEADY_STATE,
    ConfinementConcept.POLYWELL: ConfinementFamily.STEADY_STATE,
}

# Concepts whose plant power scales by REPLICATING a fixed module (n_mod), not by
# growing a single device (volume, like tokamak/mirror) or by rep-rate/yield.
# These support size_from_power via integer module-count solve (model._size_modular).
# Membership is published-developer-grounded; see docs/account_justification/
# concept_power_scaling.md. STAGED_ZPINCH is Zap's sheared-flow pinch (see its YAML).
N_MOD_SIZED_CONCEPTS = frozenset(
    {
        ConfinementConcept.ORBITRON,
        ConfinementConcept.DENSE_PLASMA_FOCUS,
        ConfinementConcept.STAGED_ZPINCH,
        ConfinementConcept.STEADY_FRC,
    }
)

# Concepts whose plant power scales by rep-rate x per-shot fusion yield (single
# device, repeated pulses), not by growing a device (tokamak/mirror) or by
# module replication (N_MOD_SIZED_CONCEPTS). Each concept YAML carries a
# sourced or explicitly-flagged-illustrative shot design point
# (e_driver_mj, yield_per_shot_mj, max_f_rep); see the "Rep-rate shot design
# points" table and citations in docs/physics/concept_power_scaling.md.
REP_RATE_SIZED_CONCEPTS = frozenset(
    {
        ConfinementConcept.PULSED_FRC,
        ConfinementConcept.MAG_TARGET,
        ConfinementConcept.PLASMA_JET,
        ConfinementConcept.THETA_PINCH,
        ConfinementConcept.LASER_IFE,
    }
)

# Concepts that support the opt-in target-yield sizing axis (design doc D3):
# scale the shot (driver energy -> yield via the gain curve -> chamber via
# R~sqrt(yield)) to hit the power target with one chamber. LASER_IFE is already
# rep-rate-sized; MAGLIF and ZPINCH move OFF their q_eng inverse path onto the
# cited-shot / target-yield path only when a concept sets
# `sizing_axis: target_yield` (otherwise their behavior is unchanged).
TARGET_YIELD_CONCEPTS = frozenset(
    {
        ConfinementConcept.LASER_IFE,
        ConfinementConcept.MAGLIF,
        ConfinementConcept.ZPINCH,
    }
)

CONCEPT_DEFAULT_CONVERSION = {
    ConfinementConcept.LASER_IFE: PulsedConversion.THERMAL,
    ConfinementConcept.ZPINCH: PulsedConversion.THERMAL,
    ConfinementConcept.HEAVY_ION: PulsedConversion.THERMAL,
    ConfinementConcept.MAG_TARGET: PulsedConversion.RECOVERED_COMPRESSION,
    ConfinementConcept.PLASMA_JET: PulsedConversion.THERMAL,
    ConfinementConcept.PULSED_FRC: PulsedConversion.INDUCTIVE_DEC,
    ConfinementConcept.MAGLIF: PulsedConversion.THERMAL,
    ConfinementConcept.THETA_PINCH: PulsedConversion.RECOVERED_COMPRESSION,
    ConfinementConcept.DENSE_PLASMA_FOCUS: PulsedConversion.THERMAL,
    ConfinementConcept.STAGED_ZPINCH: PulsedConversion.THERMAL,
}


class WallMaterial(Enum):
    TUNGSTEN = "W"
    CARBON = "C"
    BERYLLIUM = "Be"
    MOLYBDENUM = "Mo"
    SIC = "SiC"
    LITHIUM = "Li"


@dataclass
class ImpurityMix:
    """Impurity species and concentrations (f_z = n_z/n_e)."""

    wall_derived: dict[str, float]
    seeded: dict[str, float]


class FirstWallClass(Enum):
    """Hardware class of the plasma-facing first wall, for aneutronic machines
    (blanket_form NONE) whose wall is priced as surface hardware rather than
    inside a blanket structure account. Discrete classes mirror the hardware
    reality: qualified wall products exist as actively-cooled panels and as
    divertor-grade high-heat-flux components, with nothing in between."""

    PANEL = "panel"  # actively-cooled W/steel panel wall (ITER FW panel class)
    HHF = "hhf"  # divertor-grade monoblock/hypervapotron wall (ITER divertor class)
    BEAM_DUMP = "beam_dump"  # bare-CuCrZr swirl-tube/hypervapotron panel wall
    # (ITER NBI calorimeter/RID class; mass-build-up priced, no public
    # procurement value)


class CoilMaterial(Enum):
    REBCO_HTS = "rebco_hts"
    NB3SN = "nb3sn"
    NBTI = "nbti"
    COPPER = "copper"

    @property
    def is_superconducting(self) -> bool:
        """True for superconductors, which require a cryogenic plant; COPPER
        (normal-conducting) does not."""
        return self is not CoilMaterial.COPPER


class BlanketForm(Enum):
    LIQUID_METAL = "liquid_metal"
    MOLTEN_SALT = "molten_salt"
    SOLID_BREEDER = "solid_breeder"
    # Water-cooled steel energy-capture blanket: no breeder, no multiplier. For
    # D-D, which breeds its own tritium but is neutronic and so needs a low-Z
    # moderator to capture the neutron energy as heat.
    WATER_COOLED = "water_cooled"
    NONE = "none"

    @property
    def structure_factor(self) -> float:
        """Multiplier on the per-fuel blanket_unit_cost_<fuel> in CAS22.01."""
        return _BLANKET_STRUCTURE_FACTOR[self]

    @property
    def valid_fills(self) -> set["BlanketFill"]:
        """Set of BlanketFill values physically compatible with this form."""
        return _BLANKET_FORM_VALID_FILLS[self]

    @property
    def default_fill(self) -> "BlanketFill":
        """The default BlanketFill to use when only the form is specified."""
        return _BLANKET_FORM_DEFAULT_FILL[self]


_BLANKET_STRUCTURE_FACTOR = {
    BlanketForm.LIQUID_METAL: 1.0,
    BlanketForm.MOLTEN_SALT: 1.3,
    BlanketForm.SOLID_BREEDER: 1.2,
    # Steel flow-channel structure comparable to the liquid-metal baseline, but
    # water-cooled (no MHD inserts, no online tritium extraction).
    BlanketForm.WATER_COOLED: 1.0,
    BlanketForm.NONE: 0.0,
}


class BlanketFill(Enum):
    PBLI = "pbli"
    LI = "li"
    FLIBE = "flibe"
    BE_CERAMIC = "be_ceramic"
    CERAMIC_ONLY = "ceramic_only"
    LI2O = "li2o"
    WATER = "water"  # Light-water moderator/coolant, no breeder (D-D)
    NONE = "none"

    # CAS27 cost is a volume-based mass build-up keyed on this fill via
    # cc.cas27_fill_materials[fill.value]; see costs.cas27_special_materials.


_BLANKET_FORM_VALID_FILLS = {
    BlanketForm.LIQUID_METAL: {BlanketFill.PBLI, BlanketFill.LI},
    BlanketForm.MOLTEN_SALT: {BlanketFill.FLIBE},
    # Li2O is a solid ceramic breeder (no Be multiplier needed when a separate
    # W neutron multiplier handles TBR, as in OpenStar / Simpson 2026).
    BlanketForm.SOLID_BREEDER: {
        BlanketFill.BE_CERAMIC,
        BlanketFill.CERAMIC_ONLY,
        BlanketFill.LI2O,
    },
    BlanketForm.WATER_COOLED: {BlanketFill.WATER},
    BlanketForm.NONE: {BlanketFill.NONE},
}

_BLANKET_FORM_DEFAULT_FILL = {
    BlanketForm.LIQUID_METAL: BlanketFill.PBLI,
    BlanketForm.MOLTEN_SALT: BlanketFill.FLIBE,
    BlanketForm.SOLID_BREEDER: BlanketFill.BE_CERAMIC,
    BlanketForm.WATER_COOLED: BlanketFill.WATER,
    BlanketForm.NONE: BlanketFill.NONE,
}


class PowerCycle(Enum):
    RANKINE = "rankine"
    BRAYTON_SCO2 = "brayton_sco2"
    COMBINED = "combined"


class Fuel(Enum):
    DT = "dt"
    DD = "dd"
    DHE3 = "dhe3"
    PB11 = "pb11"


@dataclass
class PowerTable:
    """All power flow values computed by Layer 2 (physics)."""

    p_fus: float  # Fusion power [MW]
    p_ash: float  # Charged fusion product power [MW]
    p_neutron: float  # Neutron power [MW]
    p_rad: float  # Plasma radiation power [MW] (bremsstrahlung + synchrotron + line)
    p_wall: float  # Ash thermal on walls [MW]
    p_dee: float  # Direct energy extracted electric [MW]
    p_dec_waste: float  # DEC waste heat [MW]
    p_th: float  # Total thermal power [MW]
    p_the: float  # Thermal electric power [MW]
    p_et: float  # Gross electric power [MW]
    p_loss: float  # Lost power [MW]
    p_net: float  # Net electric power [MW]
    p_pump: float  # Pumping power [MW]
    p_sub: float  # Subsystem power [MW]
    p_aux: float  # Auxiliary power [MW]
    p_input: (
        float  # Effective heating power [MW] (may exceed user value if P_rad > P_ash)
    )
    p_coils: float  # Coil power [MW] (MFE)
    p_cool: float  # Cooling power [MW] (MFE)
    p_cryo: float  # Cryogenic system power [MW]
    p_target: float  # Target factory power [MW] (IFE/MIF)
    q_sci: float  # Scientific Q
    q_eng: float  # Engineering Q
    rec_frac: float  # Recirculating power fraction
    e_driver_mj: float = 0.0  # Per-pulse driver energy delivered [MJ]
    e_stored_mj: float = 0.0  # Per-pulse cap bank / store energy [MJ] (-> C220107)
    e_recirc_mj: float = 0.0  # Per-pulse net electrical grid draw for the driver [MJ]
    f_rep: float = 0.0  # Repetition rate [Hz]
    f_ch: float = 0.0  # Charged-particle fraction


@dataclass
class CostResult:
    """Per-CAS cost breakdown in millions USD."""

    cas10: float = 0.0  # Pre-construction
    cas21: float = 0.0  # Buildings
    cas22: float = 0.0  # Reactor plant equipment
    cas23: float = 0.0  # Turbine plant equipment
    cas24: float = 0.0  # Electric plant equipment
    cas25: float = 0.0  # Misc plant equipment
    cas26: float = 0.0  # Heat rejection
    cas27: float = 0.0  # Special materials
    cas28: float = 0.0  # Digital twin
    cas29: float = 0.0  # Contingency
    cas20: float = 0.0  # Total direct costs (sum CAS21-29)
    cas30: float = 0.0  # Indirect service costs
    cas40: float = 0.0  # Owner's costs
    cas50: float = 0.0  # Supplementary costs
    cas60: float = 0.0  # Capitalized financial costs
    cas70: float = 0.0  # Annualized O&M + replacement (CAS71 + CAS72)
    cas71: float = 0.0  # Annualized O&M
    cas72: float = 0.0  # Annualized scheduled replacement
    cas80: float = 0.0  # Annualized fuel
    cas90: float = 0.0  # Annualized financial (capital)
    total_capital: float = 0.0  # CAS10-60 sum
    lcoe: float = 0.0  # $/MWh
    overnight_cost: float = 0.0  # CAS10-50 sum (M$, excludes IDC)
    capital_per_kw: float = 0.0  # $/kW (total_capital per net electric kW)

    _LABELS = {
        "cas10": ("CAS10", "Pre-construction"),
        "cas21": ("CAS21", "Buildings"),
        "cas22": ("CAS22", "Reactor plant equipment"),
        "cas23": ("CAS23", "Turbine plant equipment"),
        "cas24": ("CAS24", "Electric plant equipment"),
        "cas25": ("CAS25", "Misc plant equipment"),
        "cas26": ("CAS26", "Heat rejection"),
        "cas27": ("CAS27", "Special materials"),
        "cas28": ("CAS28", "Digital twin"),
        "cas29": ("CAS29", "Contingency"),
        "cas20": ("CAS20", "Total direct costs"),
        "cas30": ("CAS30", "Indirect service costs"),
        "cas40": ("CAS40", "Owner's costs"),
        "cas50": ("CAS50", "Supplementary costs"),
        "cas60": ("CAS60", "Interest during construction"),
        "cas70": ("CAS70", "O&M + replacement (ann.)"),
        "cas71": ("  71", "  O&M (ann.)"),
        "cas72": ("  72", "  Scheduled replacement (ann.)"),
        "cas80": ("CAS80", "Fuel (ann.)"),
        "cas90": ("CAS90", "Financial (ann.)"),
    }

    def __str__(self) -> str:
        lines = []
        lines.append(f"{'Code':<8} {'Account':<30} {'M$':>10}")
        lines.append("-" * 50)
        for attr, (code, label) in self._LABELS.items():
            val = float(getattr(self, attr))
            lines.append(f"{code:<8} {label:<30} {val:>10.1f}")
        lines.append("-" * 50)
        lines.append(
            f"{'':8} {'Overnight (M$)':<30} {float(self.overnight_cost):>10.1f}"
        )
        lines.append(f"{'':8} {'Total capital':<30} {float(self.total_capital):>10.1f}")
        lines.append(
            f"{'':8} {'Capital ($/kW)':<30} {float(self.capital_per_kw):>10.0f}"
        )
        lines.append(f"{'':8} {'LCOE ($/MWh)':<30} {float(self.lcoe):>10.1f}")
        return "\n".join(lines)


@dataclass
class ForwardResult:
    """Complete result from a forward costing run."""

    power_table: PowerTable
    costs: CostResult
    params: dict  # All input params (for sensitivity analysis)
    overridden: list[str] = field(default_factory=list)  # Keys that were overridden
    cas22_detail: dict[str, float] = field(default_factory=dict)  # CAS22 sub-accounts
    plasma_state: object = None  # PlasmaState when 0D model is active
    solved_n_mod: int | None = None  # module count solved by n_mod size_from_power
