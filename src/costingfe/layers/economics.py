"""Layer 5: Economics — CRF, levelized costs, LCOE."""

import numpy as jnp


def compute_crf(interest_rate: float, plant_lifetime: float) -> float:
    """Capital Recovery Factor: CRF = i*(1+i)^n / ((1+i)^n - 1)."""
    i = interest_rate
    n = plant_lifetime
    return (i * (1 + i) ** n) / (((1 + i) ** n) - 1)


def levelized_annual_cost(
    annual_cost: float,
    interest_rate: float,
    inflation_rate: float,
    plant_lifetime: float,
    construction_time: float,
) -> float:
    """Levelized annual cost of a nominally-growing cost stream.

    Converts an annual cost (in today's dollars) into a level annual
    payment over the plant lifetime, accounting for:
    1. Inflation during construction (shifts to first-year-of-operation $)
    2. Continued inflation over the operating lifetime (growing annuity)
    3. Discounting at the nominal interest rate (time value of money)
    4. Annualization via CRF

    Formula:
      A_1 = annual_cost * (1 + g)^Tc          (first-year cost)
      PV  = A_1 * (1 - ((1+g)/(1+i))^n) / (i - g)  (growing annuity PV)
      levelized = CRF(i, n) * PV

    When i == g (L'Hopital limit): PV = A_1 * n / (1 + i)

    See docs/account_justification/CAS70_levelized_annual_cost.md
    """
    i = interest_rate
    g = inflation_rate
    n = plant_lifetime
    # Inflate to first-year-of-operation dollars
    a1 = annual_cost * (1 + g) ** construction_time
    # PV of growing annuity discounted at nominal rate
    # Use jnp.where for JAX traceability (both branches always evaluated)
    pv_normal = a1 * (1 - ((1 + g) / (1 + i)) ** n) / (i - g + 1e-30)
    pv_equal = a1 * n / (1 + i)
    pv = jnp.where(jnp.abs(i - g) < 1e-9, pv_equal, pv_normal)
    # Annualize with plain CRF
    crf = compute_crf(i, n)
    return crf * pv


def compute_lcoe(
    cas90: float,
    cas70: float,
    cas80: float,
    p_net: float,
    n_mod: int,
    availability: float,
) -> float:
    """LCOE in $/MWh. CAS values in M$, p_net in MW.

    LCOE = (CAS90 + CAS70 + CAS80) * 1e6 / (8760 * p_net * n_mod * availability)
    """
    annual_energy_mwh = 8760 * p_net * n_mod * availability
    total_annual_cost_usd = (cas90 + cas70 + cas80) * 1e6
    return total_annual_cost_usd / annual_energy_mwh
