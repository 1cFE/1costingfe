from dataclasses import dataclass, field
from enum import Enum


class ConfinementFamily(Enum):
    STEADY_STATE = "steady_state"
    PULSED = "pulsed"


class ConfinementConcept(Enum):
    TOKAMAK = "tokamak"
    STELLARATOR = "stellarator"
    MIRROR = "mirror"
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


CONCEPT_TO_FAMILY = {
    ConfinementConcept.TOKAMAK: ConfinementFamily.STEADY_STATE,
    ConfinementConcept.STELLARATOR: ConfinementFamily.STEADY_STATE,
    ConfinementConcept.MIRROR: ConfinementFamily.STEADY_STATE,
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

CONCEPT_DEFAULT_CONVERSION = {
    ConfinementConcept.LASER_IFE: PulsedConversion.THERMAL,
    ConfinementConcept.ZPINCH: PulsedConversion.THERMAL,
    ConfinementConcept.HEAVY_ION: PulsedConversion.THERMAL,
    ConfinementConcept.MAG_TARGET: PulsedConversion.THERMAL,
    ConfinementConcept.PLASMA_JET: PulsedConversion.THERMAL,
    ConfinementConcept.PULSED_FRC: PulsedConversion.INDUCTIVE_DEC,
    ConfinementConcept.MAGLIF: PulsedConversion.THERMAL,
    ConfinementConcept.THETA_PINCH: PulsedConversion.INDUCTIVE_DEC,
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


class CoilMaterial(Enum):
    REBCO_HTS = "rebco_hts"
    NB3SN = "nb3sn"
    NBTI = "nbti"
    COPPER = "copper"

    @property
    def default_cost_per_kAm(self) -> float:
        """Default conductor cost in $/kAm."""
        return _COIL_MATERIAL_COST[self]


_COIL_MATERIAL_COST = {
    CoilMaterial.REBCO_HTS: 50.0,
    CoilMaterial.NB3SN: 7.0,
    CoilMaterial.NBTI: 7.0,
    CoilMaterial.COPPER: 1.0,
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
    e_driver_mj: float = 0.0  # Per-pulse driver energy [MJ]
    e_stored_mj: float = 0.0  # Per-pulse cap bank energy [MJ]
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
    overnight_cost: float = 0.0  # $/kW


@dataclass
class ForwardResult:
    """Complete result from a forward costing run."""

    power_table: PowerTable
    costs: CostResult
    params: dict  # All input params (for sensitivity analysis)
    overridden: list[str] = field(default_factory=list)  # Keys that were overridden
    cas22_detail: dict[str, float] = field(default_factory=dict)  # CAS22 sub-accounts
    plasma_state: object = None  # PlasmaState when 0D model is active
