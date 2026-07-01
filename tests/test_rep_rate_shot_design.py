"""Per-shot design-point fields for REP_RATE_SIZED_CONCEPTS.

Forward-path YAML data only (Task 3 of rep-rate power sizing): each of the
five rep-rate-sized concepts carries e_driver_mj, yield_per_shot_mj, and
max_f_rep in its concept-default YAML. Three are sourced from published
design points; two (PULSED_FRC yield, THETA_PINCH yield + max_f_rep) are
explicitly-flagged illustrative placeholders. See docs/physics/
concept_power_scaling.md ("Rep-rate shot design points") for citations and
derivations. No solver consumes these fields yet (Tasks 4-5).
"""

import pytest

from costingfe.defaults import load_engineering_defaults
from costingfe.types import (
    CONCEPT_TO_FAMILY,
    REP_RATE_SIZED_CONCEPTS,
    ConfinementConcept,
)

# (concept, e_driver_mj, yield_per_shot_mj, max_f_rep): exact pinned values,
# see CONTROLLER DECISIONS in the Task 3 sourcing record.
_EXPECTED = [
    (ConfinementConcept.MAG_TARGET, 755.0, 780.0, 1.0),
    (ConfinementConcept.PLASMA_JET, 31.3, 736.0, 1.0),
    (ConfinementConcept.LASER_IFE, 2.5, 250.0, 10.0),
    (ConfinementConcept.PULSED_FRC, 50.0, 101.4, 1.0),
    (ConfinementConcept.THETA_PINCH, 3.5, 35.0, 1.0),
]


@pytest.mark.parametrize("concept,e_driver_mj,yield_per_shot_mj,max_f_rep", _EXPECTED)
def test_shot_design_point_fields(concept, e_driver_mj, yield_per_shot_mj, max_f_rep):
    family = CONCEPT_TO_FAMILY[concept]
    defaults = load_engineering_defaults(f"{family.value}_{concept.value}")
    assert defaults["e_driver_mj"] == pytest.approx(e_driver_mj)
    assert defaults["yield_per_shot_mj"] == pytest.approx(yield_per_shot_mj)
    assert defaults["max_f_rep"] == pytest.approx(max_f_rep)


def test_all_rep_rate_sized_concepts_have_shot_design_fields():
    # Every REP_RATE_SIZED_CONCEPTS member (not just the ones in _EXPECTED)
    # must carry the three fields; guards against a concept being added to
    # the set without its YAML being updated.
    for concept in REP_RATE_SIZED_CONCEPTS:
        family = CONCEPT_TO_FAMILY[concept]
        defaults = load_engineering_defaults(f"{family.value}_{concept.value}")
        for key in ("e_driver_mj", "yield_per_shot_mj", "max_f_rep"):
            assert key in defaults, f"{concept.value} missing {key}"
            assert defaults[key] > 0.0
