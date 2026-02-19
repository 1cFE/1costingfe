"""Input validation for the costing model.

Pydantic-based CostingInput with three validation tiers:
- Tier 1: Field-level constraints (pydantic Field)
- Tier 2: Family-aware required engineering parameters
- Tier 3: Cross-field physics checks
"""

import warnings

from pydantic import BaseModel, Field, model_validator

from costingfe.types import (
    CONCEPT_TO_FAMILY,
    ConfinementConcept,
    ConfinementFamily,
    Fuel,
)


class CostingInput(BaseModel):
    """Validated input for the costing model.

    Required fields: concept, fuel, net_electric_mw.
    Customer parameters have defaults.
    Engineering parameters default to None (filled from YAML templates).
    """

    # --- Required (no defaults) ---
    concept: ConfinementConcept
    fuel: Fuel
    net_electric_mw: float = Field(gt=0)

    # --- Customer parameters (with defaults) ---
    availability: float = Field(default=0.85, gt=0, le=1)
    lifetime_yr: float = Field(default=40.0, gt=0)
    n_mod: int = Field(default=1, ge=1, strict=True)
    construction_time_yr: float = Field(default=6.0, gt=0)
    interest_rate: float = Field(default=0.07, gt=0)
    inflation_rate: float = 0.02
    noak: bool = True
    cost_overrides: dict[str, float] = Field(default_factory=dict)
    costing_overrides: dict[str, float] = Field(default_factory=dict)

    # --- Engineering parameters (None = use YAML template) ---
    # Common (all families)
    mn: float | None = None
    eta_th: float | None = None
    eta_p: float | None = None
    f_sub: float | None = None
    p_pump: float | None = None
    p_trit: float | None = None
    p_house: float | None = None
    p_cryo: float | None = None
    blanket_t: float | None = None
    ht_shield_t: float | None = None
    structure_t: float | None = None
    vessel_t: float | None = None
    plasma_t: float | None = None

    # MFE only
    p_input: float | None = None
    eta_pin: float | None = None
    eta_de: float | None = None
    f_dec: float | None = None
    p_coils: float | None = None
    p_cool: float | None = None
    axis_t: float | None = None
    elon: float | None = None

    # IFE only
    p_implosion: float | None = None
    p_ignition: float | None = None
    eta_pin1: float | None = None
    eta_pin2: float | None = None
    p_target: float | None = None  # shared with MIF

    # MIF only
    p_driver: float | None = None
    # eta_pin: already declared above (shared MFE/MIF)
    # p_target: already declared above (shared IFE/MIF)
    # p_coils: already declared above (shared MFE/MIF)

    # Plasma parameters (MFE radiation calculation)
    n_e: float | None = None
    T_e: float | None = None
    Z_eff: float | None = None
    plasma_volume: float | None = None
    B: float | None = None

    # --- Tier 2: family-required parameter lists ---
    _COMMON_REQUIRED = [
        "mn", "eta_th", "eta_p", "f_sub",
        "p_pump", "p_trit", "p_house", "p_cryo",
        "blanket_t", "ht_shield_t", "structure_t", "vessel_t", "plasma_t",
    ]
    _MFE_REQUIRED = [
        "p_input", "eta_pin", "eta_de", "f_dec",
        "p_coils", "p_cool", "axis_t", "elon",
    ]
    _IFE_REQUIRED = [
        "p_implosion", "p_ignition", "eta_pin1", "eta_pin2", "p_target",
    ]
    _MIF_REQUIRED = [
        "p_driver", "eta_pin", "p_target", "p_coils",
    ]

    @model_validator(mode="after")
    def check_family_required_params(self):
        """Tier 2: If any eng param is set, all family-required must be present."""
        family = CONCEPT_TO_FAMILY[self.concept]

        all_eng = (
            self._COMMON_REQUIRED
            + self._MFE_REQUIRED + self._IFE_REQUIRED + self._MIF_REQUIRED
        )
        any_set = any(getattr(self, k) is not None for k in all_eng)
        if not any_set:
            return self

        family_required = {
            ConfinementFamily.MFE: self._MFE_REQUIRED,
            ConfinementFamily.IFE: self._IFE_REQUIRED,
            ConfinementFamily.MIF: self._MIF_REQUIRED,
        }
        required = self._COMMON_REQUIRED + family_required.get(family, [])

        missing = [k for k in required if getattr(self, k) is None]
        if missing:
            raise ValueError(
                f"Missing required engineering parameters for "
                f"{family.value}: {', '.join(missing)}"
            )
        return self

    @model_validator(mode="after")
    def check_physics(self):
        """Tier 3: Cross-field physics checks (warnings + errors).

        Only runs when all engineering params are present (not None).
        """
        family = CONCEPT_TO_FAMILY[self.concept]

        # --- Simple field warnings (no computation needed) ---
        if self.eta_th is not None and self.eta_th > 0.65:
            warnings.warn(
                f"eta_th = {self.eta_th} is unusually high (> 0.65)",
                stacklevel=2,
            )
        if self.eta_p is not None and self.eta_p > 0.95:
            warnings.warn(
                f"eta_p = {self.eta_p} is unusually high (> 0.95)",
                stacklevel=2,
            )
        if self.mn is not None and not (1.0 <= self.mn <= 1.5):
            warnings.warn(
                f"mn = {self.mn} is outside typical range [1.0, 1.5]",
                stacklevel=2,
            )
        if self.f_sub is not None and self.f_sub > 0.3:
            warnings.warn(
                f"f_sub = {self.f_sub} is unusually high (> 0.3)",
                stacklevel=2,
            )

        # --- Physics checks requiring power balance computation ---
        if any(getattr(self, k) is None for k in self._COMMON_REQUIRED):
            return self

        if family == ConfinementFamily.MFE:
            self._check_mfe_physics()
        elif family == ConfinementFamily.IFE:
            self._check_ife_physics()
        elif family == ConfinementFamily.MIF:
            self._check_mif_physics()

        return self

    def _check_mfe_physics(self):
        from costingfe.layers.physics import (
            mfe_forward_power_balance,
            mfe_inverse_power_balance,
        )

        mfe_params = [self.p_input, self.eta_pin, self.eta_de, self.f_dec,
                      self.p_coils, self.p_cool]
        if any(v is None for v in mfe_params):
            return

        p_net_per_mod = self.net_electric_mw / self.n_mod
        p_fus = mfe_inverse_power_balance(
            p_net_target=p_net_per_mod, fuel=self.fuel,
            p_input=self.p_input, mn=self.mn, eta_th=self.eta_th,
            eta_p=self.eta_p, eta_pin=self.eta_pin, eta_de=self.eta_de,
            f_sub=self.f_sub, f_dec=self.f_dec, p_coils=self.p_coils,
            p_cool=self.p_cool, p_pump=self.p_pump, p_trit=self.p_trit,
            p_house=self.p_house, p_cryo=self.p_cryo,
        )
        pt = mfe_forward_power_balance(
            p_fus=p_fus, fuel=self.fuel,
            p_input=self.p_input, mn=self.mn, eta_th=self.eta_th,
            eta_p=self.eta_p, eta_pin=self.eta_pin, eta_de=self.eta_de,
            f_sub=self.f_sub, f_dec=self.f_dec, p_coils=self.p_coils,
            p_cool=self.p_cool, p_pump=self.p_pump, p_trit=self.p_trit,
            p_house=self.p_house, p_cryo=self.p_cryo,
        )
        self._check_power_table(pt, p_fus)

    def _check_ife_physics(self):
        from costingfe.layers.physics import (
            ife_forward_power_balance,
            ife_inverse_power_balance,
        )

        ife_params = [self.p_implosion, self.p_ignition,
                      self.eta_pin1, self.eta_pin2, self.p_target]
        if any(v is None for v in ife_params):
            return

        p_net_per_mod = self.net_electric_mw / self.n_mod
        p_fus = ife_inverse_power_balance(
            p_net_target=p_net_per_mod, fuel=self.fuel,
            p_implosion=self.p_implosion, p_ignition=self.p_ignition,
            mn=self.mn, eta_th=self.eta_th, eta_p=self.eta_p,
            eta_pin1=self.eta_pin1, eta_pin2=self.eta_pin2,
            f_sub=self.f_sub, p_pump=self.p_pump, p_trit=self.p_trit,
            p_house=self.p_house, p_cryo=self.p_cryo, p_target=self.p_target,
        )
        pt = ife_forward_power_balance(
            p_fus=p_fus, fuel=self.fuel,
            p_implosion=self.p_implosion, p_ignition=self.p_ignition,
            mn=self.mn, eta_th=self.eta_th, eta_p=self.eta_p,
            eta_pin1=self.eta_pin1, eta_pin2=self.eta_pin2,
            f_sub=self.f_sub, p_pump=self.p_pump, p_trit=self.p_trit,
            p_house=self.p_house, p_cryo=self.p_cryo, p_target=self.p_target,
        )
        self._check_power_table(pt, p_fus)

    def _check_mif_physics(self):
        from costingfe.layers.physics import (
            mif_forward_power_balance,
            mif_inverse_power_balance,
        )

        mif_params = [self.p_driver, self.eta_pin, self.p_target]
        if any(v is None for v in mif_params):
            return

        p_net_per_mod = self.net_electric_mw / self.n_mod
        p_fus = mif_inverse_power_balance(
            p_net_target=p_net_per_mod, fuel=self.fuel,
            p_driver=self.p_driver, mn=self.mn, eta_th=self.eta_th,
            eta_p=self.eta_p, eta_pin=self.eta_pin, f_sub=self.f_sub,
            p_pump=self.p_pump, p_trit=self.p_trit, p_house=self.p_house,
            p_cryo=self.p_cryo, p_target=self.p_target,
            p_coils=self.p_coils or 0.0,
        )
        pt = mif_forward_power_balance(
            p_fus=p_fus, fuel=self.fuel,
            p_driver=self.p_driver, mn=self.mn, eta_th=self.eta_th,
            eta_p=self.eta_p, eta_pin=self.eta_pin, f_sub=self.f_sub,
            p_pump=self.p_pump, p_trit=self.p_trit, p_house=self.p_house,
            p_cryo=self.p_cryo, p_target=self.p_target,
            p_coils=self.p_coils or 0.0,
        )
        self._check_power_table(pt, p_fus)

    def _check_power_table(self, pt, p_fus):
        """Check derived physics values from power balance."""
        rec_frac = float(pt.rec_frac)
        q_sci = float(pt.q_sci)

        if float(p_fus) <= 0 or rec_frac > 0.95:
            raise ValueError(
                f"p_net is effectively non-positive (rec_frac = "
                f"{rec_frac:.3f}, p_fus = {float(p_fus):.1f} MW) — "
                f"plant consumes more power than it produces"
            )
        if q_sci < 2:
            warnings.warn(
                f"Q_sci = {q_sci:.3f} < 2 — "
                f"fusion power is low relative to injected heating",
                stacklevel=4,
            )
        if rec_frac > 0.5:
            warnings.warn(
                f"Recirculating fraction = {rec_frac:.3f} > 0.5 — "
                f"excessive parasitic power load",
                stacklevel=4,
            )
