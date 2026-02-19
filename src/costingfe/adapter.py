"""Fusion-tea adapter: typed interface between SysML pipeline and CostModel.

Provides:
- FusionTeaInput: flat dict from SysML-extracted parameters
- FusionTeaOutput: CAS-code-keyed cost breakdown + LCOE
- run_costing(): input -> output (the single function fusion-tea calls)
"""

from dataclasses import dataclass, field

from costingfe.defaults import load_costing_constants
from costingfe.model import CostModel
from costingfe.types import ConfinementConcept, Fuel


@dataclass
class FusionTeaInput:
    """Parameters extracted from SysML model by fusion-tea."""

    concept: str         # e.g. "tokamak", "laser_ife", "mag_target"
    fuel: str            # e.g. "dt", "dd", "dhe3", "pb11"
    net_electric_mw: float
    availability: float
    lifetime_yr: float
    n_mod: int = 1
    construction_time_yr: float = 6.0
    interest_rate: float = 0.07
    inflation_rate: float = 0.0245
    noak: bool = True
    overrides: dict = field(default_factory=dict)
    cost_overrides: dict[str, float] = field(default_factory=dict)      # CAS account → M$
    costing_overrides: dict[str, float] = field(default_factory=dict)   # CostingConstants field → value


@dataclass
class FusionTeaOutput:
    """Cost breakdown in CAS-code-keyed format for fusion-tea."""

    lcoe: float                  # $/MWh
    overnight_cost: float        # $/kW
    total_capital: float         # M$
    costs: dict[str, float]      # CAS code -> M$ (e.g. "CAS10" -> 16.0)
    power_table: dict[str, float]  # Power flow values (e.g. "p_fus" -> 2300.0)
    sensitivity: dict[str, dict[str, float]]  # {"engineering": {...}, "financial": {...}}
    overridden: list[str] = field(default_factory=list)  # Keys that were overridden


def run_costing(inp: FusionTeaInput) -> FusionTeaOutput:
    """Single entry point for fusion-tea pipeline.

    Maps SysML-extracted parameters to CostModel, runs forward costing,
    and returns CAS-code-keyed results.
    """
    concept = ConfinementConcept(inp.concept)
    fuel = Fuel(inp.fuel)

    cc = load_costing_constants()
    if inp.costing_overrides:
        cc = cc.replace(**inp.costing_overrides)

    model = CostModel(concept=concept, fuel=fuel, costing_constants=cc)
    result = model.forward(
        net_electric_mw=inp.net_electric_mw,
        availability=inp.availability,
        lifetime_yr=inp.lifetime_yr,
        n_mod=inp.n_mod,
        construction_time_yr=inp.construction_time_yr,
        interest_rate=inp.interest_rate,
        inflation_rate=inp.inflation_rate,
        noak=inp.noak,
        cost_overrides=inp.cost_overrides or None,
        **inp.overrides,
    )

    c = result.costs
    costs = {
        "CAS10": float(c.cas10),
        "CAS21": float(c.cas21),
        "CAS22": float(c.cas22),
        "CAS23": float(c.cas23),
        "CAS24": float(c.cas24),
        "CAS25": float(c.cas25),
        "CAS26": float(c.cas26),
        "CAS27": float(c.cas27),
        "CAS28": float(c.cas28),
        "CAS29": float(c.cas29),
        "CAS20": float(c.cas20),
        "CAS30": float(c.cas30),
        "CAS40": float(c.cas40),
        "CAS50": float(c.cas50),
        "CAS60": float(c.cas60),
        "CAS70": float(c.cas70),
        "CAS80": float(c.cas80),
        "CAS90": float(c.cas90),
    }
    # Include CAS22 sub-account detail
    for key, val in result.cas22_detail.items():
        costs[key] = float(val)

    pt = result.power_table
    power_table = {
        "p_fus": float(pt.p_fus),
        "p_th": float(pt.p_th),
        "p_et": float(pt.p_et),
        "p_net": float(pt.p_net),
        "q_sci": float(pt.q_sci),
        "q_eng": float(pt.q_eng),
        "rec_frac": float(pt.rec_frac),
    }

    sens = model.sensitivity(result.params, cost_overrides=inp.cost_overrides or None)

    return FusionTeaOutput(
        lcoe=float(c.lcoe),
        overnight_cost=float(c.overnight_cost),
        total_capital=float(c.total_capital),
        costs=costs,
        power_table=power_table,
        sensitivity=sens,
        overridden=result.overridden,
    )
