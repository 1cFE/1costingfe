from costingfe.types import (
    ConfinementFamily,
    ConfinementConcept,
    Fuel,
    CONCEPT_TO_FAMILY,
    PowerTable,
    CostResult,
    ForwardResult,
)
from costingfe.model import CostModel

from dataclasses import dataclass


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
            ConfinementConcept.LASER_IFE,
            ConfinementConcept.MAG_TARGET,
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
