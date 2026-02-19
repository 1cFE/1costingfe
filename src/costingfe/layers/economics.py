"""Layer 5: Economics — CRF, levelized costs, LCOE."""


def compute_crf(interest_rate: float, plant_lifetime: float) -> float:
    """Capital Recovery Factor: CRF = i*(1+i)^n / ((1+i)^n - 1)."""
    i = interest_rate
    n = plant_lifetime
    return (i * (1 + i) ** n) / (((1 + i) ** n) - 1)


def compute_effective_crf(
    interest_rate: float,
    plant_lifetime: float,
    construction_time: float,
) -> float:
    """CRF adjusted for construction time: CRF * (1+i)^Tc."""
    crf = compute_crf(interest_rate, plant_lifetime)
    return crf * (1 + interest_rate) ** construction_time


def levelized_annual_cost(
    annual_cost: float,
    inflation_rate: float,
    construction_time: float,
) -> float:
    """Adjust an annual cost from today's dollars to operation-start dollars.

    annual_cost is in today's dollars. Inflation shifts it forward to the
    dollar-year when operation begins (after construction):
      annual_cost_at_operation = annual_cost * (1 + inflation)^construction_time
    """
    return annual_cost * (1 + inflation_rate) ** construction_time


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
