from enum import Enum
from dataclasses import dataclass, field


class ConfinementFamily(Enum):
    MFE = "mfe"
    IFE = "ife"
    MIF = "mif"


class ConfinementConcept(Enum):
    TOKAMAK = "tokamak"
    STELLARATOR = "stellarator"
    MIRROR = "mirror"
    LASER_IFE = "laser_ife"
    ZPINCH = "zpinch"
    HEAVY_ION = "heavy_ion"
    MAG_TARGET = "mag_target"
    PLASMA_JET = "plasma_jet"


CONCEPT_TO_FAMILY = {
    ConfinementConcept.TOKAMAK: ConfinementFamily.MFE,
    ConfinementConcept.STELLARATOR: ConfinementFamily.MFE,
    ConfinementConcept.MIRROR: ConfinementFamily.MFE,
    ConfinementConcept.LASER_IFE: ConfinementFamily.IFE,
    ConfinementConcept.ZPINCH: ConfinementFamily.IFE,
    ConfinementConcept.HEAVY_ION: ConfinementFamily.IFE,
    ConfinementConcept.MAG_TARGET: ConfinementFamily.MIF,
    ConfinementConcept.PLASMA_JET: ConfinementFamily.MIF,
}


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
    p_coils: float  # Coil power [MW] (MFE)
    p_cool: float  # Cooling power [MW] (MFE)
    p_cryo: float  # Cryogenic system power [MW]
    p_target: float  # Target factory power [MW] (IFE/MIF)
    q_sci: float  # Scientific Q
    q_eng: float  # Engineering Q
    rec_frac: float  # Recirculating power fraction


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
    cas70: float = 0.0  # Annualized O&M
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
