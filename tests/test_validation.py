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


class TestTier2FamilyRequiredParams:
    """Tier 2: After template merge, all family-required params must be present."""

    def test_mfe_missing_p_input_rejected(self):
        """MFE requires p_input — should fail if None after merge."""
        with pytest.raises(ValidationError, match="p_input"):
            CostingInput(
                concept=ConfinementConcept.TOKAMAK,
                fuel=Fuel.DT,
                net_electric_mw=1000.0,
                mn=1.1, eta_th=0.46, eta_p=0.5, f_sub=0.03,
                p_pump=1.0, p_trit=10.0, p_house=4.0, p_cryo=0.5,
                blanket_t=0.7, ht_shield_t=0.2, structure_t=0.15,
                vessel_t=0.1, plasma_t=2.0,
                eta_pin=0.5, eta_de=0.85, f_dec=0.0,
                p_coils=2.0, p_cool=13.7, axis_t=6.2, elon=1.7,
            )

    def test_ife_missing_p_implosion_rejected(self):
        with pytest.raises(ValidationError, match="p_implosion"):
            CostingInput(
                concept=ConfinementConcept.LASER_IFE,
                fuel=Fuel.DT,
                net_electric_mw=1000.0,
                mn=1.1, eta_th=0.46, eta_p=0.5, f_sub=0.03,
                p_pump=1.0, p_trit=10.0, p_house=4.0, p_cryo=0.5,
                blanket_t=0.8, ht_shield_t=0.25, structure_t=0.15,
                vessel_t=0.1, plasma_t=4.0,
                p_ignition=0.1, eta_pin1=0.1, eta_pin2=0.1, p_target=1.0,
            )

    def test_mif_missing_p_driver_rejected(self):
        with pytest.raises(ValidationError, match="p_driver"):
            CostingInput(
                concept=ConfinementConcept.MAG_TARGET,
                fuel=Fuel.DT,
                net_electric_mw=1000.0,
                mn=1.1, eta_th=0.4, eta_p=0.5, f_sub=0.03,
                p_pump=1.0, p_trit=10.0, p_house=4.0, p_cryo=0.2,
                blanket_t=0.7, ht_shield_t=0.2, structure_t=0.15,
                vessel_t=0.1, plasma_t=3.0,
                eta_pin=0.3, p_target=2.0, p_coils=0.5,
            )

    def test_none_engineering_params_ok_when_template_will_fill(self):
        """When no engineering params given (all None), Tier 2 is skipped."""
        inp = CostingInput(
            concept=ConfinementConcept.TOKAMAK,
            fuel=Fuel.DT,
            net_electric_mw=1000.0,
        )
        assert inp.mn is None

    def test_mfe_complete_params_accepted(self):
        """All MFE params provided — should pass."""
        inp = CostingInput(
            concept=ConfinementConcept.TOKAMAK,
            fuel=Fuel.DT,
            net_electric_mw=1000.0,
            mn=1.1, eta_th=0.46, eta_p=0.5, f_sub=0.03,
            p_pump=1.0, p_trit=10.0, p_house=4.0, p_cryo=0.5,
            blanket_t=0.7, ht_shield_t=0.2, structure_t=0.15,
            vessel_t=0.1, plasma_t=2.0,
            p_input=50.0, eta_pin=0.5, eta_de=0.85, f_dec=0.0,
            p_coils=2.0, p_cool=13.7, axis_t=6.2, elon=1.7,
        )
        assert inp.p_input == 50.0


class TestTier3PhysicsChecks:
    """Tier 3: Cross-field and physics validation."""

    def _make_mfe_input(self, **overrides):
        """Helper: complete MFE tokamak input with all params."""
        defaults = dict(
            concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT,
            net_electric_mw=1000.0,
            mn=1.1, eta_th=0.46, eta_p=0.5, f_sub=0.03,
            p_pump=1.0, p_trit=10.0, p_house=4.0, p_cryo=0.5,
            blanket_t=0.7, ht_shield_t=0.2, structure_t=0.15,
            vessel_t=0.1, plasma_t=2.0,
            p_input=50.0, eta_pin=0.5, eta_de=0.85, f_dec=0.0,
            p_coils=2.0, p_cool=13.7, axis_t=6.2, elon=1.7,
        )
        defaults.update(overrides)
        return CostingInput(**defaults)

    def test_eta_th_warning_when_high(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._make_mfe_input(eta_th=0.70)
            assert any("eta_th" in str(warning.message) for warning in w)

    def test_eta_th_no_warning_when_normal(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._make_mfe_input(eta_th=0.46)
            assert not any("eta_th" in str(warning.message) for warning in w)

    def test_eta_p_warning_when_high(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._make_mfe_input(eta_p=0.98)
            assert any("eta_p" in str(warning.message) for warning in w)

    def test_mn_warning_when_outside_range(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._make_mfe_input(mn=2.0)
            assert any("mn" in str(warning.message) for warning in w)

    def test_f_sub_warning_when_high(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._make_mfe_input(f_sub=0.35)
            assert any("f_sub" in str(warning.message) for warning in w)

    def test_p_net_negative_raises_error(self):
        """p_net < 0 is a hard error — plant consumes more than it produces."""
        with pytest.raises(ValidationError, match="p_net"):
            self._make_mfe_input(
                net_electric_mw=1.0,
                p_input=500.0,
                eta_pin=0.1,
            )

    def test_q_sci_warning_when_low(self):
        """Q_sci < 2 means fusion power is low relative to injected heating."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._make_mfe_input(p_input=5000.0, eta_pin=0.9)
            assert any("Q_sci" in str(warning.message) for warning in w)

    def test_rec_frac_warning_when_high(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._make_mfe_input(eta_pin=0.05)
            assert any("rec" in str(warning.message).lower() for warning in w)


from costingfe.model import CostModel


class TestForwardIntegration:
    """Validation fires when calling CostModel.forward()."""

    def test_forward_rejects_negative_net_electric(self):
        model = CostModel(
            concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT,
        )
        with pytest.raises(ValidationError, match="net_electric_mw"):
            model.forward(net_electric_mw=-100, availability=0.85, lifetime_yr=40)

    def test_forward_rejects_invalid_availability(self):
        model = CostModel(
            concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT,
        )
        with pytest.raises(ValidationError, match="availability"):
            model.forward(net_electric_mw=1000, availability=2.0, lifetime_yr=40)

    def test_forward_still_works_with_valid_input(self):
        model = CostModel(
            concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT,
        )
        result = model.forward(
            net_electric_mw=1000, availability=0.85, lifetime_yr=40,
        )
        assert result.costs.lcoe > 0


from costingfe.adapter import FusionTeaInput, run_costing


class TestAdapterIntegration:
    """Validation fires when calling run_costing()."""

    def test_adapter_rejects_negative_net_electric(self):
        inp = FusionTeaInput(
            concept="tokamak", fuel="dt",
            net_electric_mw=-100, availability=0.85, lifetime_yr=40,
        )
        with pytest.raises(ValidationError, match="net_electric_mw"):
            run_costing(inp)

    def test_adapter_rejects_invalid_availability(self):
        inp = FusionTeaInput(
            concept="tokamak", fuel="dt",
            net_electric_mw=1000, availability=2.0, lifetime_yr=40,
        )
        with pytest.raises(ValidationError, match="availability"):
            run_costing(inp)

    def test_adapter_still_works_with_valid_input(self):
        inp = FusionTeaInput(
            concept="tokamak", fuel="dt",
            net_electric_mw=1000, availability=0.85, lifetime_yr=40,
        )
        output = run_costing(inp)
        assert output.lcoe > 0
