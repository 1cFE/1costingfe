"""Input validation for the costing model.

Pydantic-based CostingInput with three validation tiers:
- Tier 1: Field-level constraints (pydantic Field)
- Tier 2: Family-aware required engineering parameters
- Tier 3: Cross-field physics checks
"""

import warnings
from typing import Optional

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
    mn: Optional[float] = None
    eta_th: Optional[float] = None
    eta_p: Optional[float] = None
    f_sub: Optional[float] = None
    p_pump: Optional[float] = None
    p_trit: Optional[float] = None
    p_house: Optional[float] = None
    p_cryo: Optional[float] = None
    blanket_t: Optional[float] = None
    ht_shield_t: Optional[float] = None
    structure_t: Optional[float] = None
    vessel_t: Optional[float] = None
    plasma_t: Optional[float] = None

    # MFE only
    p_input: Optional[float] = None
    eta_pin: Optional[float] = None
    eta_de: Optional[float] = None
    f_dec: Optional[float] = None
    p_coils: Optional[float] = None
    p_cool: Optional[float] = None
    axis_t: Optional[float] = None
    elon: Optional[float] = None

    # IFE only
    p_implosion: Optional[float] = None
    p_ignition: Optional[float] = None
    eta_pin1: Optional[float] = None
    eta_pin2: Optional[float] = None
    p_target: Optional[float] = None  # shared with MIF

    # MIF only
    p_driver: Optional[float] = None
    # eta_pin: already declared above (shared MFE/MIF)
    # p_target: already declared above (shared IFE/MIF)
    # p_coils: already declared above (shared MFE/MIF)

    # Plasma parameters (MFE radiation calculation)
    n_e: Optional[float] = None
    T_e: Optional[float] = None
    Z_eff: Optional[float] = None
    plasma_volume: Optional[float] = None
    B: Optional[float] = None
