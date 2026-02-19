"""Tests for CostingInput validation."""

import pytest
import warnings
from pydantic import ValidationError

from costingfe.validation import CostingInput
from costingfe.types import ConfinementConcept, Fuel


class TestTier1FieldConstraints:
    """Tier 1: pydantic Field() constraints."""

    def test_valid_minimal_input(self):
        """Required fields only — should succeed."""
        inp = CostingInput(
            concept=ConfinementConcept.TOKAMAK,
            fuel=Fuel.DT,
            net_electric_mw=1000.0,
        )
        assert inp.net_electric_mw == 1000.0
        assert inp.availability == 0.85  # default
        assert inp.lifetime_yr == 40.0  # default

    def test_net_electric_mw_must_be_positive(self):
        with pytest.raises(ValidationError, match="net_electric_mw"):
            CostingInput(
                concept=ConfinementConcept.TOKAMAK,
                fuel=Fuel.DT,
                net_electric_mw=-100.0,
            )

    def test_net_electric_mw_zero_rejected(self):
        with pytest.raises(ValidationError, match="net_electric_mw"):
            CostingInput(
                concept=ConfinementConcept.TOKAMAK,
                fuel=Fuel.DT,
                net_electric_mw=0.0,
            )

    def test_availability_must_be_in_range(self):
        with pytest.raises(ValidationError, match="availability"):
            CostingInput(
                concept=ConfinementConcept.TOKAMAK,
                fuel=Fuel.DT,
                net_electric_mw=1000.0,
                availability=1.5,
            )

    def test_availability_zero_rejected(self):
        with pytest.raises(ValidationError, match="availability"):
            CostingInput(
                concept=ConfinementConcept.TOKAMAK,
                fuel=Fuel.DT,
                net_electric_mw=1000.0,
                availability=0.0,
            )

    def test_lifetime_must_be_positive(self):
        with pytest.raises(ValidationError, match="lifetime_yr"):
            CostingInput(
                concept=ConfinementConcept.TOKAMAK,
                fuel=Fuel.DT,
                net_electric_mw=1000.0,
                lifetime_yr=-5.0,
            )

    def test_n_mod_must_be_integer(self):
        with pytest.raises(ValidationError, match="n_mod"):
            CostingInput(
                concept=ConfinementConcept.TOKAMAK,
                fuel=Fuel.DT,
                net_electric_mw=1000.0,
                n_mod=1.5,
            )

    def test_n_mod_must_be_at_least_one(self):
        with pytest.raises(ValidationError, match="n_mod"):
            CostingInput(
                concept=ConfinementConcept.TOKAMAK,
                fuel=Fuel.DT,
                net_electric_mw=1000.0,
                n_mod=0,
            )

    def test_interest_rate_must_be_positive(self):
        with pytest.raises(ValidationError, match="interest_rate"):
            CostingInput(
                concept=ConfinementConcept.TOKAMAK,
                fuel=Fuel.DT,
                net_electric_mw=1000.0,
                interest_rate=-0.01,
            )

    def test_inflation_rate_can_be_negative(self):
        """Deflation is valid."""
        inp = CostingInput(
            concept=ConfinementConcept.TOKAMAK,
            fuel=Fuel.DT,
            net_electric_mw=1000.0,
            inflation_rate=-0.01,
        )
        assert inp.inflation_rate == -0.01

    def test_construction_time_must_be_positive(self):
        with pytest.raises(ValidationError, match="construction_time_yr"):
            CostingInput(
                concept=ConfinementConcept.TOKAMAK,
                fuel=Fuel.DT,
                net_electric_mw=1000.0,
                construction_time_yr=0.0,
            )

    def test_concept_string_accepted(self):
        """Concept can be passed as string (adapter path)."""
        inp = CostingInput(
            concept="tokamak",
            fuel="dt",
            net_electric_mw=1000.0,
        )
        assert inp.concept == ConfinementConcept.TOKAMAK

    def test_invalid_concept_rejected(self):
        with pytest.raises(ValidationError, match="concept"):
            CostingInput(
                concept="not_a_concept",
                fuel=Fuel.DT,
                net_electric_mw=1000.0,
            )

    def test_all_customer_defaults(self):
        """All customer params have sensible defaults."""
        inp = CostingInput(
            concept=ConfinementConcept.TOKAMAK,
            fuel=Fuel.DT,
            net_electric_mw=1000.0,
        )
        assert inp.availability == 0.85
        assert inp.lifetime_yr == 40.0
        assert inp.n_mod == 1
        assert inp.construction_time_yr == 6.0
        assert inp.interest_rate == 0.07
        assert inp.inflation_rate == 0.02
        assert inp.noak is True
        assert inp.cost_overrides == {}
        assert inp.costing_overrides == {}
