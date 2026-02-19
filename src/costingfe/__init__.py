from dataclasses import dataclass

from costingfe.model import CostModel
from costingfe.types import (
    CONCEPT_TO_FAMILY,
    ConfinementConcept,
    ConfinementFamily,
    CostResult,
    ForwardResult,
    Fuel,
    PowerTable,
)
from costingfe.validation import CostingInput as CostingInput


@dataclass
class ComparisonResult:
    concept: ConfinementConcept
    fuel: Fuel
    lcoe: float
    result: ForwardResult


def compare_all(
    net_electric_mw: float,
    availability: float,
    lifetime_yr: float,
    concepts: list[ConfinementConcept] | None = None,
    fuels: list[Fuel] | None = None,
    **kwargs,
) -> list[ComparisonResult]:
    """Run all concept x fuel combinations, rank by LCOE."""
    if concepts is None:
        concepts = [
            ConfinementConcept.TOKAMAK,
            ConfinementConcept.STELLARATOR,
            ConfinementConcept.MIRROR,
            ConfinementConcept.LASER_IFE,
            ConfinementConcept.ZPINCH,
            ConfinementConcept.HEAVY_ION,
            ConfinementConcept.MAG_TARGET,
            ConfinementConcept.PLASMA_JET,
        ]
    if fuels is None:
        fuels = list(Fuel)

    results = []
    for concept in concepts:
        for fuel in fuels:
            try:
                model = CostModel(concept=concept, fuel=fuel)
                result = model.forward(
                    net_electric_mw=net_electric_mw,
                    availability=availability,
                    lifetime_yr=lifetime_yr,
                    **kwargs,
                )
                results.append(
                    ComparisonResult(
                        concept=concept,
                        fuel=fuel,
                        lcoe=result.costs.lcoe,
                        result=result,
                    )
                )
            except Exception:
                continue  # skip non-viable combinations

    return sorted(results, key=lambda r: r.lcoe)
