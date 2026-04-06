from costingfe.types import (
    CONCEPT_DEFAULT_CONVERSION,
    CONCEPT_TO_FAMILY,
    ConfinementConcept,
    ConfinementFamily,
    Fuel,
    PowerTable,
    PulsedConversion,
)


def test_confinement_family_values():
    assert ConfinementFamily.STEADY_STATE.value == "steady_state"
    assert ConfinementFamily.PULSED.value == "pulsed"
    assert not hasattr(ConfinementFamily, "MFE")
    assert not hasattr(ConfinementFamily, "IFE")
    assert not hasattr(ConfinementFamily, "MIF")


def test_pulsed_conversion_enum():
    assert PulsedConversion.THERMAL.value == "thermal"
    assert PulsedConversion.INDUCTIVE_DEC.value == "inductive_dec"


def test_concept_to_family_mapping():
    assert (
        CONCEPT_TO_FAMILY[ConfinementConcept.TOKAMAK] == ConfinementFamily.STEADY_STATE
    )
    assert (
        CONCEPT_TO_FAMILY[ConfinementConcept.MIRROR] == ConfinementFamily.STEADY_STATE
    )
    assert CONCEPT_TO_FAMILY[ConfinementConcept.LASER_IFE] == ConfinementFamily.PULSED
    assert CONCEPT_TO_FAMILY[ConfinementConcept.MAG_TARGET] == ConfinementFamily.PULSED


def test_all_concepts_have_family():
    for concept in ConfinementConcept:
        assert concept in CONCEPT_TO_FAMILY, f"{concept} missing from CONCEPT_TO_FAMILY"


def test_fuel_enum():
    assert len(Fuel) == 4
    assert Fuel.DT.value == "dt"


def test_power_table_has_pulsed_fields():
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(PowerTable)}
    assert "e_driver_mj" in field_names
    assert "e_stored_mj" in field_names
    assert "f_rep" in field_names
    assert "f_ch" in field_names


def test_concept_default_conversion():
    assert (
        CONCEPT_DEFAULT_CONVERSION[ConfinementConcept.LASER_IFE]
        == PulsedConversion.THERMAL
    )
    assert (
        CONCEPT_DEFAULT_CONVERSION[ConfinementConcept.MAG_TARGET]
        == PulsedConversion.THERMAL
    )
