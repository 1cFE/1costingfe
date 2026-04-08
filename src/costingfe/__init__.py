import os as _os

# Default to CPU — suppresses "NVIDIA GPU may be present" warning.
# Users with CUDA-enabled jaxlib can set JAX_PLATFORMS=cuda to override.
_os.environ.setdefault("JAX_PLATFORMS", "cpu")

from dataclasses import dataclass

from costingfe.layers.tokamak import PlasmaState as PlasmaState
from costingfe.model import CostModel
from costingfe.types import (
    CONCEPT_DEFAULT_CONVERSION as CONCEPT_DEFAULT_CONVERSION,
)
from costingfe.types import (
    CONCEPT_TO_FAMILY as CONCEPT_TO_FAMILY,
)
from costingfe.types import (
    ConfinementConcept,
    ForwardResult,
    Fuel,
)
from costingfe.types import (
    ConfinementFamily as ConfinementFamily,
)
from costingfe.types import (
    CostResult as CostResult,
)
from costingfe.types import (
    PowerCycle as PowerCycle,
)
from costingfe.types import (
    PowerTable as PowerTable,
)
from costingfe.types import (
    PulsedConversion as PulsedConversion,
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
            ConfinementConcept.PULSED_FRC,
            ConfinementConcept.MAGLIF,
            ConfinementConcept.THETA_PINCH,
            ConfinementConcept.DENSE_PLASMA_FOCUS,
            ConfinementConcept.STAGED_ZPINCH,
            ConfinementConcept.ORBITRON,
            ConfinementConcept.POLYWELL,
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
