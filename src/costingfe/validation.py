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
    BlanketFill,
    BlanketForm,
    ConfinementConcept,
    ConfinementFamily,
    Fuel,
)

_PLASMA_0D_FIELDS = ["q95", "f_GW", "M_ion", "lambda_q", "use_0d_model"]


def default_availability(concept: ConfinementConcept) -> float:
    """Concept-aware availability default.

    Linear/open-end geometries (mirror) achieve higher steady-state availability
    than port-limited toroidal geometries because blanket and first-wall
    components can be exchanged axially without re-establishing toroidal
    vacuum/structural continuity. Same physical basis as the C220110 0.55x
    remote-handling capex scaling and the CAS70 0.85x O&M scaling.

    Tokamak/stellarator: 0.85 (ARIES heritage).
    Mirror: 0.87 (shorter scheduled outages).
    Other concepts: 0.85 (no concept-specific basis to claim better).
    """
    if concept == ConfinementConcept.MIRROR:
        return 0.87
    return 0.85


_VALIDATION_PHYSICS = dict(
    n_e=1.0e20,
    T_e=15.0,
    Z_eff=1.5,
    plasma_volume=500.0,
    B=5.0,
    dd_f_T=0.969,
    dd_f_He3=0.689,
    dhe3_dd_frac=0.131,
    dhe3_f_T=0.5,
    # representative MFE-class value (bf=0.05, fr=0.99); concept-specific
    # values flow through compute()
    dhe3_f_He3=0.84,
    pb11_f_alpha_n=0.0,
    pb11_f_p_n=0.0,
    wall_material=None,
    T_edge=0.05,
    tau_ratio=3.0,
    fw_area=0.0,
    R_major=0.0,
    a_minor=0.0,
    kappa=1.7,
    R_w=0.6,
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
    # availability defaults to None → resolved per-concept in default_availability().
    # Concept-aware default reflects the planned-outage advantage of linear
    # geometries: a mirror's blanket rings can be lifted axially without
    # re-establishing toroidal continuity, shortening scheduled-outage duration
    # versus port-limited toroidal access. Override by passing a value.
    availability: float | None = Field(default=None, gt=0, le=1)
    lifetime_yr: float = Field(default=40.0, gt=0)
    n_mod: float = Field(default=1.0, gt=0)  # FR-1 of costingfe-library-preconditions spec: non-integer n_mod required for two-knob projection (n_mod = 1000 / P_native)
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
    burn_fraction: float | None = Field(default=None, gt=0, le=1)
    fuel_recovery: float | None = Field(default=None, gt=0, le=1)

    # MFE only
    p_input: float | None = None
    eta_pin: float | None = None
    eta_couple: float | None = None  # heating delivered->plasma coupling (concept)
    eta_de: float | None = None
    f_dec: float | None = None
    p_coils: float | None = None
    p_cool: float | None = None
    R0: float | None = None
    elon: float | None = None

    # Pulsed shared
    p_target: float | None = None
    target_unit_cost: float | None = Field(
        default=None, ge=0
    )  # $/shot consumed target hardware (CAS80)
    # eta_pin: already declared above (shared MFE/PULSED)
    # p_coils: already declared above (shared MFE/PULSED)

    # Pulsed (new unified parameter set)
    q_eng: float | None = None
    f_rep: float | None = None
    f_rad: float | None = None
    eta_dec: float | None = None
    f_pdv: float | None = None

    # Plasma parameters (MFE radiation calculation)
    n_e: float | None = None
    T_e: float | None = None
    Z_eff: float | None = None
    plasma_volume: float | None = None
    B: float | None = None

    # 0D plasma model (tokamak only)
    use_0d_model: bool = False
    q95: float | None = None
    f_GW: float | None = None
    M_ion: float | None = None
    lambda_q: float | None = None

    # Blanket configuration (YAML-driven, validated below)
    blanket_form: BlanketForm | None = None
    blanket_fill: BlanketFill | None = None

    # --- Tier 2: family-required parameter lists ---
    _COMMON_REQUIRED = [
        "mn",
        "f_sub",
        "p_pump",
        "p_trit",
        "p_house",
        "p_cryo",
        "blanket_t",
        "ht_shield_t",
        "structure_t",
        "vessel_t",
        "plasma_t",
        "burn_fraction",
        "fuel_recovery",
    ]
    _MFE_REQUIRED = [
        "p_input",
        "eta_p",
        "eta_de",
        "f_dec",
        "p_coils",
        "p_cool",
        "R0",
        "elon",
    ]
    _PULSED_REQUIRED = [
        "q_eng",
        "f_rep",
        "eta_pin",
        "p_target",
        "target_unit_cost",
    ]

    @model_validator(mode="after")
    def fill_concept_defaults(self):
        """Fill customer parameters that depend on the concept when not set."""
        if self.availability is None:
            self.availability = default_availability(self.concept)
        return self

    @model_validator(mode="after")
    def check_family_required_params(self):
        """Tier 2: If any eng param is set, all family-required must be present."""
        family = CONCEPT_TO_FAMILY[self.concept]

        all_eng = self._COMMON_REQUIRED + self._MFE_REQUIRED + self._PULSED_REQUIRED
        any_set = any(getattr(self, k) is not None for k in all_eng)
        if not any_set:
            return self

        if family == ConfinementFamily.PULSED:
            family_keys = self._PULSED_REQUIRED
        else:
            family_keys = (
                self._MFE_REQUIRED if family == ConfinementFamily.STEADY_STATE else []
            )
        required = self._COMMON_REQUIRED + family_keys

        missing = [k for k in required if getattr(self, k) is None]
        if missing:
            raise ValueError(
                f"Missing required engineering parameters for "
                f"{family.value}: {', '.join(missing)}"
            )

        # Steady-state heating efficiency: either an explicit eta_pin
        # (electrostatic / direct-injection concepts) or an eta_couple that
        # combines with the per-method eta_source to form eta_pin.
        if (
            family == ConfinementFamily.STEADY_STATE
            and self.eta_pin is None
            and self.eta_couple is None
        ):
            raise ValueError(
                "steady-state concept requires either eta_pin or eta_couple"
            )

        # 0D model requires q95 and f_GW
        if self.use_0d_model:
            if self.concept != ConfinementConcept.TOKAMAK:
                raise ValueError("use_0d_model is only supported for TOKAMAK concept")
            od_missing = []
            if self.q95 is None:
                od_missing.append("q95")
            if self.f_GW is None:
                od_missing.append("f_GW")
            if od_missing:
                raise ValueError(f"0D model requires: {', '.join(od_missing)}")

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

        if family == ConfinementFamily.STEADY_STATE:
            self._check_mfe_physics()
        elif family == ConfinementFamily.PULSED:
            self._check_pulsed_physics()

        return self

    @model_validator(mode="after")
    def check_blanket_compatibility(self):
        """Tier 3: blanket form/fill compatibility and fuel-physics coupling.

        Errors:
          - blanket_fill must be in blanket_form.valid_fills
          - DT fuel with blanket_form=NONE or blanket_fill=NONE is unphysical
        Warnings:
          - Aneutronic fuels (DHe3, pB11) with non-none blanket are wasteful
        """
        if self.blanket_form is None or self.blanket_fill is None:
            return self  # presence is enforced at the materialization point

        if self.blanket_fill not in self.blanket_form.valid_fills:
            raise ValueError(
                f"blanket_fill={self.blanket_fill.value!r} not valid for "
                f"blanket_form={self.blanket_form.value!r}. "
                f"Valid: {sorted(f.value for f in self.blanket_form.valid_fills)}"
            )

        if self.fuel == Fuel.DT and (
            self.blanket_form == BlanketForm.NONE
            or self.blanket_fill == BlanketFill.NONE
        ):
            raise ValueError(
                "DT fuel requires a breeding blanket "
                f"(got blanket_form={self.blanket_form.value!r}, "
                f"blanket_fill={self.blanket_fill.value!r})."
            )

        if (
            self.fuel in (Fuel.DHE3, Fuel.PB11)
            and self.blanket_form != BlanketForm.NONE
        ):
            warnings.warn(
                f"{self.fuel.value} with blanket_form={self.blanket_form.value!r}: "
                "aneutronic fuels do not need a breeding blanket.",
                stacklevel=2,
            )

        return self

    def _check_mfe_physics(self):
        from costingfe.layers.physics import (
            mfe_forward_power_balance,
            mfe_inverse_power_balance,
        )

        mfe_params = [
            self.p_input,
            self.eta_pin,
            self.eta_de,
            self.f_dec,
            self.p_coils,
            self.p_cool,
        ]
        if any(v is None for v in mfe_params):
            return

        p_net_per_mod = self.net_electric_mw / self.n_mod
        p_fus = mfe_inverse_power_balance(
            p_net_target=p_net_per_mod,
            fuel=self.fuel,
            p_input=self.p_input,
            mn=self.mn,
            eta_th=self.eta_th,
            eta_p=self.eta_p,
            eta_pin=self.eta_pin,
            eta_de=self.eta_de,
            f_sub=self.f_sub,
            f_dec=self.f_dec,
            p_coils=self.p_coils,
            p_cool=self.p_cool,
            p_pump=self.p_pump,
            p_trit=self.p_trit,
            p_house=self.p_house,
            p_cryo=self.p_cryo,
            **_VALIDATION_PHYSICS,
        )
        pt = mfe_forward_power_balance(
            p_fus=p_fus,
            fuel=self.fuel,
            p_input=self.p_input,
            mn=self.mn,
            eta_th=self.eta_th,
            eta_p=self.eta_p,
            eta_pin=self.eta_pin,
            eta_de=self.eta_de,
            f_sub=self.f_sub,
            f_dec=self.f_dec,
            p_coils=self.p_coils,
            p_cool=self.p_cool,
            p_pump=self.p_pump,
            p_trit=self.p_trit,
            p_house=self.p_house,
            p_cryo=self.p_cryo,
            **_VALIDATION_PHYSICS,
        )
        self._check_power_table(pt, p_fus)

    def _check_pulsed_physics(self):
        from costingfe.layers.physics import (
            pulsed_thermal_forward,
            pulsed_thermal_inverse,
        )

        pulsed_params = [
            self.q_eng,
            self.f_rep,
            self.eta_pin,
            self.p_target,
        ]
        if any(v is None for v in pulsed_params):
            return

        # Thermal inverse requires eta_th > 0; skip for DEC (eta_th=0)
        if self.eta_th is not None and self.eta_th == 0.0:
            return

        p_net_per_mod = self.net_electric_mw / self.n_mod
        common_kw = dict(
            fuel=self.fuel,
            f_rep=self.f_rep,
            mn=self.mn,
            eta_th=self.eta_th,
            eta_pin=self.eta_pin,
            f_rad=0.1,
            f_sub=self.f_sub,
            p_pump=self.p_pump,
            p_trit=self.p_trit,
            p_house=self.p_house,
            p_cryo=self.p_cryo,
            p_target=self.p_target,
            p_coils=self.p_coils or 0.0,
        )
        p_fus, e_driver_solved = pulsed_thermal_inverse(
            p_net_target=p_net_per_mod,
            q_eng=self.q_eng,
            **common_kw,
        )
        common_kw["e_driver_mj"] = e_driver_solved
        pt = pulsed_thermal_forward(
            p_fus=p_fus,
            **common_kw,
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
