"""Probabilistic uncertainty analysis for fusion power plant LCOE.

Demonstrates three scenarios:

1. Basic Monte Carlo -- independent distributions on 5 key parameters
2. Correlated parameters -- eta_th and availability negatively correlated
3. Full uncertainty -- also varies CostingConstants (blanket + shield cost)
"""

import numpy as np

from costingfe import ConfinementConcept, CostModel, Fuel
from costingfe.analysis.uncertainty import (
    Normal,
    Triangular,
    Uniform,
    run_uncertainty,
    run_uncertainty_full,
)


def main():
    # ── Setup ──────────────────────────────────────────────────────────
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    base = model.forward(
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
    )
    print(f"Deterministic LCOE: {base.costs.lcoe:.2f} $/MWh\n")

    # ── 1. Basic uncertainty (independent) ─────────────────────────────
    dists = {
        "eta_th": Normal(mean=0.40, sigma=0.03),
        "availability": Triangular(low=0.75, mode=0.85, high=0.92),
        "interest_rate": Uniform(low=0.05, high=0.10),
        "inflation_rate": Normal(mean=0.025, sigma=0.005),
        "construction_time_yr": Triangular(low=4.0, mode=6.0, high=10.0),
    }

    res = run_uncertainty(
        model,
        base.params,
        param_distributions=dists,
        n_samples=2000,
        seed=42,
    )

    print("─── 1. Basic uncertainty (5 params, independent) ───")
    print(f"  Samples : {res.n_samples}")
    print(f"  Mean    : {res.mean:.2f} $/MWh")
    print(f"  Std     : {res.std:.2f} $/MWh")
    print(f"  P10     : {res.p10:.2f} $/MWh")
    print(f"  P50     : {res.p50:.2f} $/MWh")
    print(f"  P90     : {res.p90:.2f} $/MWh")
    print(f"  80% CI  : [{res.ci_80[0]:.2f}, {res.ci_80[1]:.2f}]")
    print(f"  90% CI  : [{res.ci_90[0]:.2f}, {res.ci_90[1]:.2f}]")
    print()

    # ── 2. Correlated parameters ───────────────────────────────────────
    corr_dists = {
        "eta_th": Normal(mean=0.40, sigma=0.03),
        "availability": Normal(mean=0.85, sigma=0.04),
    }
    # Negative correlation: higher efficiency plants tend to have
    # lower availability (more complex systems)
    corr_matrix = np.array(
        [
            [1.0, -0.6],
            [-0.6, 1.0],
        ]
    )

    res_corr = run_uncertainty(
        model,
        base.params,
        param_distributions=corr_dists,
        correlation_matrix=corr_matrix,
        n_samples=2000,
        seed=42,
    )

    # Compare with independent
    res_ind = run_uncertainty(
        model,
        base.params,
        param_distributions=corr_dists,
        n_samples=2000,
        seed=42,
    )

    print("─── 2. Correlated parameters (eta_th ~ availability) ───")
    r = res_ind
    print(
        f"  Independent:  P10={r.p10:.2f}  P50={r.p50:.2f}"
        f"  P90={r.p90:.2f}  std={r.std:.2f}"
    )
    r = res_corr
    print(
        f"  Correlated:   P10={r.p10:.2f}  P50={r.p50:.2f}"
        f"  P90={r.p90:.2f}  std={r.std:.2f}"
    )
    print()

    # ── 3. Full uncertainty (with CostingConstants) ────────────────────
    res_full = run_uncertainty_full(
        concept=ConfinementConcept.TOKAMAK,
        fuel=Fuel.DT,
        base_params=base.params,
        param_distributions={
            "eta_th": Normal(mean=0.40, sigma=0.03),
            "availability": Triangular(low=0.75, mode=0.85, high=0.92),
        },
        cc_distributions={
            "blanket_unit_cost_dt": Uniform(0.5, 2.0),
            "shield_unit_cost": Triangular(0.5, 0.74, 1.2),
        },
        n_samples=1000,
        n_cc_bins=5,
        seed=42,
    )

    print("─── 3. Full uncertainty (with CostingConstants) ───")
    print(f"  Samples : {res_full.n_samples}")
    print(f"  Mean    : {res_full.mean:.2f} $/MWh")
    print(f"  P10     : {res_full.p10:.2f} $/MWh")
    print(f"  P50     : {res_full.p50:.2f} $/MWh")
    print(f"  P90     : {res_full.p90:.2f} $/MWh")
    print(f"  90% CI  : [{res_full.ci_90[0]:.2f}, {res_full.ci_90[1]:.2f}]")
    print()


if __name__ == "__main__":
    main()
