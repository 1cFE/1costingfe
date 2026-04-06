"""Backcasting: given a target LCOE, find required parameter values.

Uses bisection for single-parameter backcasting (robust, bracketed
root-finding). The forward model is the oracle — no analytic inversion
needed.
"""

from costingfe.model import CostModel


def backcast_single(
    model: CostModel,
    target_lcoe: float,
    param_name: str,
    param_range: tuple[float, float],
    base_params: dict | None = None,
    tol: float = 0.01,
) -> float:
    """Find the value of `param_name` that achieves `target_lcoe`.

    Uses Brent's method (bracketed root-finding) on:
        f(x) = forward(params with param_name=x).lcoe - target_lcoe

    Args:
        model: CostModel instance (concept + fuel already set).
        target_lcoe: Target LCOE in $/MWh.
        param_name: Engineering parameter to solve for (e.g. "eta_th").
        param_range: (low, high) bracket — target must be achievable
            within this range.
        base_params: Base forward() params. If None, uses model defaults.
        tol: Absolute tolerance on LCOE ($/MWh).

    Returns:
        Parameter value that achieves the target LCOE.

    Raises:
        ValueError: If target LCOE is not achievable within the bracket.
    """
    # Params that are named forward() args, not **overrides
    named_args = {
        "net_electric_mw",
        "availability",
        "lifetime_yr",
        "n_mod",
        "construction_time_yr",
        "interest_rate",
        "inflation_rate",
        "noak",
        "fuel",
        "concept",
    }

    if base_params is None:
        # Run once with defaults to get the full param set
        base_result = model.forward(
            net_electric_mw=1000.0,
            availability=0.85,
            lifetime_yr=30,
        )
        base_params = base_result.params

    def _lcoe_at(val):
        eng = {k: v for k, v in base_params.items() if k not in named_args}
        if param_name in eng:
            eng[param_name] = val
        result = model.forward(
            net_electric_mw=base_params["net_electric_mw"],
            availability=base_params["availability"],
            lifetime_yr=base_params["lifetime_yr"],
            n_mod=base_params.get("n_mod", 1),
            construction_time_yr=base_params.get("construction_time_yr", 6.0),
            interest_rate=(
                val
                if param_name == "interest_rate"
                else base_params.get("interest_rate", 0.07)
            ),
            inflation_rate=(
                val
                if param_name == "inflation_rate"
                else base_params.get("inflation_rate", 0.02)
            ),
            noak=base_params.get("noak", True),
            **eng,
        )
        return float(result.costs.lcoe)

    lo, hi = param_range
    f_lo = _lcoe_at(lo) - target_lcoe
    f_hi = _lcoe_at(hi) - target_lcoe

    if f_lo * f_hi > 0:
        lcoe_lo = _lcoe_at(lo)
        lcoe_hi = _lcoe_at(hi)
        raise ValueError(
            f"Target LCOE {target_lcoe:.1f} $/MWh not achievable within "
            f"{param_name}=[{lo}, {hi}]. "
            f"LCOE range: [{min(lcoe_lo, lcoe_hi):.1f}, {max(lcoe_lo, lcoe_hi):.1f}]"
        )

    # Bisection root-finding (replaces scipy.optimize.brentq)
    xtol = tol * 0.01
    a, b = lo, hi
    fa = f_lo
    for _ in range(100):  # 100 iterations gives ~1e-30 relative precision
        mid = (a + b) / 2.0
        if (b - a) < xtol:
            break
        fm = _lcoe_at(mid) - target_lcoe
        if fa * fm <= 0:
            b = mid
        else:
            a, fa = mid, fm
    return (a + b) / 2.0


def backcast_multi(
    model: CostModel,
    target_lcoe: float,
    param_ranges: dict[str, tuple[float, float]],
    base_params: dict | None = None,
    tol: float = 0.01,
) -> dict[str, float]:
    """Find each parameter's required value independently.

    For each parameter in param_ranges, holds all others at base values
    and solves for the value that achieves target_lcoe. This gives the
    "what if we only improved X?" answer for each parameter.

    Returns dict of param_name -> required_value (or None if not achievable).
    """
    results = {}
    for param_name, param_range in param_ranges.items():
        try:
            val = backcast_single(
                model,
                target_lcoe,
                param_name,
                param_range,
                base_params=base_params,
                tol=tol,
            )
            results[param_name] = val
        except ValueError:
            results[param_name] = None
    return results
