"""Tests for CostingInput validation."""

import warnings

import pytest
from pydantic import ValidationError

from costingfe.adapter import FusionTeaInput, run_costing
from costingfe.model import CostModel
from costingfe.types import BlanketFill, BlanketForm, ConfinementConcept, Fuel
from costingfe.validation import CostingInput


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

    def test_n_mod_accepts_positive_float(self):
        """FR-1 of costingfe-library-preconditions: n_mod must accept positive
        real values for the two-knob projection (n_mod = 1000 / P_native)."""
        valid = CostingInput(
            concept=ConfinementConcept.TOKAMAK,
            fuel=Fuel.DT,
            net_electric_mw=1000.0,
            n_mod=1.5,
        )
        assert valid.n_mod == 1.5

    def test_n_mod_must_be_positive(self):
        """FR-1: n_mod accepts any positive real value (gt=0). Zero and
        negative values are rejected."""
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

    def test_override_reference_mw_must_be_positive(self):
        with pytest.raises(ValidationError, match="override_reference_mw"):
            CostingInput(
                concept=ConfinementConcept.TOKAMAK,
                fuel=Fuel.DT,
                net_electric_mw=1000.0,
                override_reference_mw=0.0,
            )

    def test_override_reference_mw_defaults_none(self):
        inp = CostingInput(
            concept=ConfinementConcept.TOKAMAK,
            fuel=Fuel.DT,
            net_electric_mw=1000.0,
        )
        assert inp.override_reference_mw is None

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
        assert inp.construction_time_yr is None  # deferred to the concept YAML
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
                mn=1.1,
                eta_th=0.46,
                eta_p=0.5,
                f_sub=0.03,
                p_pump=1.0,
                p_trit=10.0,
                p_house=4.0,
                p_cryo=0.5,
                blanket_t=0.7,
                ht_shield_t=0.2,
                structure_t=0.15,
                vessel_t=0.1,
                plasma_t=2.0,
                eta_pin=0.5,
                eta_de=0.85,
                f_dec=0.0,
                p_coils=2.0,
                p_cool=13.7,
                R0=6.2,
                elon=1.7,
            )

    def test_pulsed_missing_q_eng_rejected(self):
        """Pulsed concept missing q_eng should be rejected."""
        with pytest.raises(ValidationError, match="q_eng"):
            CostingInput(
                concept=ConfinementConcept.LASER_IFE,
                fuel=Fuel.DT,
                net_electric_mw=1000.0,
                mn=1.1,
                eta_th=0.46,
                f_sub=0.03,
                p_pump=1.0,
                p_trit=10.0,
                p_house=4.0,
                p_cryo=0.5,
                blanket_t=0.8,
                ht_shield_t=0.25,
                structure_t=0.15,
                vessel_t=0.1,
                plasma_t=4.0,
                f_rep=10.0,
                eta_pin=0.1,
                p_target=1.0,
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
            mn=1.1,
            eta_th=0.46,
            eta_p=0.5,
            f_sub=0.03,
            p_pump=1.0,
            p_trit=10.0,
            p_house=4.0,
            p_cryo=0.5,
            blanket_t=0.7,
            ht_shield_t=0.2,
            structure_t=0.15,
            vessel_t=0.1,
            plasma_t=2.0,
            burn_fraction=0.05,
            fuel_recovery=0.99,
            p_input=50.0,
            eta_pin=0.5,
            eta_de=0.85,
            f_dec=0.0,
            p_coils=2.0,
            p_cool=13.7,
            R0=6.2,
            elon=1.7,
        )
        assert inp.p_input == 50.0


class TestTier3PhysicsChecks:
    """Tier 3: Cross-field and physics validation."""

    def _make_mfe_input(self, **overrides):
        """Helper: complete MFE tokamak input with all params."""
        defaults = dict(
            concept=ConfinementConcept.TOKAMAK,
            fuel=Fuel.DT,
            net_electric_mw=1000.0,
            mn=1.1,
            eta_th=0.46,
            eta_p=0.5,
            f_sub=0.03,
            p_pump=1.0,
            p_trit=10.0,
            p_house=4.0,
            p_cryo=0.5,
            blanket_t=0.7,
            ht_shield_t=0.2,
            structure_t=0.15,
            vessel_t=0.1,
            plasma_t=2.0,
            burn_fraction=0.05,
            fuel_recovery=0.99,
            p_input=50.0,
            eta_pin=0.5,
            eta_de=0.85,
            f_dec=0.0,
            p_coils=2.0,
            p_cool=13.7,
            R0=6.2,
            elon=1.7,
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

    def test_mfe_physics_uses_instance_plasma_not_representative(self):
        """The feasibility check must evaluate the concept's own plasma, not
        hardcoded representative-thermal values. A low-density non-thermal
        plasma that is genuinely feasible must not be rejected as if it were a
        dense thermal plasma."""
        common = dict(
            concept=ConfinementConcept.ORBITRON,
            fuel=Fuel.PB11,
            net_electric_mw=0.005,
            mn=1.0,
            eta_th=0.4,
            eta_p=0.5,
            f_sub=0.03,
            p_pump=0.00005,
            p_trit=0.0,
            p_house=0.0001,
            p_cryo=0.0,
            blanket_t=0.02,
            ht_shield_t=0.02,
            structure_t=0.10,
            vessel_t=0.05,
            plasma_t=0.3,
            burn_fraction=0.05,
            fuel_recovery=0.99,
            p_input=0.0025,
            eta_pin=0.80,
            eta_de=0.70,
            f_dec=0.90,
            p_coils=0.0001,
            p_cool=0.0002,
            R0=0.0,
            elon=1.0,
        )
        # Dense thermal plasma -> enormous bremsstrahlung -> infeasible.
        with pytest.raises(ValidationError, match="p_net"):
            CostingInput(
                **common, n_e=1.0e20, T_e=15.0, Z_eff=1.5, plasma_volume=500.0, B=5.0
            )
        # The actual low-density non-thermal plasma -> feasible (no raise).
        CostingInput(**common, n_e=1.0e18, T_e=5.0, Z_eff=1.5, plasma_volume=0.5, B=1.0)

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


class TestForwardIntegration:
    """Validation fires when calling CostModel.forward()."""

    def test_forward_rejects_negative_net_electric(self):
        model = CostModel(
            concept=ConfinementConcept.TOKAMAK,
            fuel=Fuel.DT,
        )
        with pytest.raises(ValidationError, match="net_electric_mw"):
            model.forward(net_electric_mw=-100, availability=0.85, lifetime_yr=40)

    def test_forward_rejects_invalid_availability(self):
        model = CostModel(
            concept=ConfinementConcept.TOKAMAK,
            fuel=Fuel.DT,
        )
        with pytest.raises(ValidationError, match="availability"):
            model.forward(net_electric_mw=1000, availability=2.0, lifetime_yr=40)

    def test_forward_still_works_with_valid_input(self):
        model = CostModel(
            concept=ConfinementConcept.TOKAMAK,
            fuel=Fuel.DT,
        )
        result = model.forward(
            net_electric_mw=1000,
            availability=0.85,
            lifetime_yr=40,
        )
        assert result.costs.lcoe > 0


class TestAdapterIntegration:
    """Validation fires when calling run_costing()."""

    def test_adapter_rejects_negative_net_electric(self):
        inp = FusionTeaInput(
            concept="tokamak",
            fuel="dt",
            net_electric_mw=-100,
            availability=0.85,
            lifetime_yr=40,
        )
        with pytest.raises(ValidationError, match="net_electric_mw"):
            run_costing(inp)

    def test_adapter_rejects_invalid_availability(self):
        inp = FusionTeaInput(
            concept="tokamak",
            fuel="dt",
            net_electric_mw=1000,
            availability=2.0,
            lifetime_yr=40,
        )
        with pytest.raises(ValidationError, match="availability"):
            run_costing(inp)

    def test_adapter_still_works_with_valid_input(self):
        inp = FusionTeaInput(
            concept="tokamak",
            fuel="dt",
            net_electric_mw=1000,
            availability=0.85,
            lifetime_yr=40,
        )
        output = run_costing(inp)
        assert output.lcoe > 0


def test_blanket_fill_must_match_form():
    """Schema check: solid_breeder cannot use pbli fill."""
    with pytest.raises(ValidationError, match="not valid for blanket_form"):
        CostingInput(
            concept=ConfinementConcept.TOKAMAK,
            fuel=Fuel.DT,
            net_electric_mw=1000.0,
            blanket_form=BlanketForm.SOLID_BREEDER,
            blanket_fill=BlanketFill.PBLI,
        )


def test_dt_requires_breeding_blanket():
    """Physics check: DT without breeding blanket raises."""
    with pytest.raises(ValidationError, match="DT fuel requires a breeding blanket"):
        CostingInput(
            concept=ConfinementConcept.TOKAMAK,
            fuel=Fuel.DT,
            net_electric_mw=1000.0,
            blanket_form=BlanketForm.NONE,
            blanket_fill=BlanketFill.NONE,
        )


def test_liquid_metal_with_none_fill_rejected():
    """Schema check: NONE fill is not in LIQUID_METAL.valid_fills (form-fill rule)."""
    with pytest.raises(ValidationError, match="not valid for blanket_form"):
        CostingInput(
            concept=ConfinementConcept.TOKAMAK,
            fuel=Fuel.DT,
            net_electric_mw=1000.0,
            blanket_form=BlanketForm.LIQUID_METAL,
            blanket_fill=BlanketFill.NONE,
        )


def test_aneutronic_with_blanket_warns():
    """Economics check: p-B11 with non-none blanket emits UserWarning."""
    with pytest.warns(
        UserWarning, match="aneutronic fuels do not need a breeding blanket"
    ):
        CostingInput(
            concept=ConfinementConcept.TOKAMAK,
            fuel=Fuel.PB11,
            net_electric_mw=1000.0,
            blanket_form=BlanketForm.LIQUID_METAL,
            blanket_fill=BlanketFill.PBLI,
        )


def test_all_valid_form_fill_pairs_accepted_for_dt():
    """Every valid pair (except NONE/NONE which DT rejects) is accepted."""
    for form in BlanketForm:
        if form == BlanketForm.NONE:
            continue
        for fill in form.valid_fills:
            CostingInput(
                concept=ConfinementConcept.TOKAMAK,
                fuel=Fuel.DT,
                net_electric_mw=1000.0,
                blanket_form=form,
                blanket_fill=fill,
            )
