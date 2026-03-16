"""Back-solve: what does a 1 ¢/kWh DT mirror look like?

Starting from the 500 MWe reference case (8.93 ¢/kWh), sweep key
parameters to find combinations that reach ~1 ¢/kWh (10 $/MWh).
Uses brute-force grid search over the highest-elasticity levers.
"""

from costingfe import ConfinementConcept, CostModel, Fuel

model = CostModel(concept=ConfinementConcept.MIRROR, fuel=Fuel.DT)

# ── Baseline ──────────────────────────────────────────────────────────
base_kwargs = dict(
    net_electric_mw=500.0,
    availability=0.85,
    lifetime_yr=30,
    n_mod=1,
    construction_time_yr=5.0,
    interest_rate=0.07,
    inflation_rate=0.02,
    noak=True,
    R0=0.0,
    plasma_t=1.5,
    chamber_length=20.0,
    blanket_t=0.60,
    ht_shield_t=0.20,
    structure_t=0.15,
    vessel_t=0.10,
    p_input=40.0,
    mn=1.1,
    eta_th=0.40,
    eta_p=0.5,
    eta_pin=0.5,
    eta_de=0.60,
    f_sub=0.03,
    f_dec=0.30,
    p_coils=5.0,
    p_cool=20.0,
    p_pump=1.5,
    p_trit=10.0,
    p_house=4.0,
    p_cryo=1.0,
)

base = model.forward(**base_kwargs)
bl = float(base.costs.lcoe)
print(f"Baseline: {bl:.1f} $/MWh ({bl / 10:.2f} ¢/kWh)")
print()

# ── Parameter grid (optimistic but physically plausible ranges) ──────
sweeps = {
    # Scale: bigger plant dilutes fixed costs
    "net_electric_mw": [500, 1000, 2000],
    # Availability: biggest single lever
    "availability": [0.85, 0.90, 0.95],
    # Construction: faster = less IDC
    "construction_time_yr": [5.0, 3.0, 2.0],
    # Finance: lower cost of capital
    "interest_rate": [0.07, 0.05, 0.03],
    # Thermal efficiency: advanced cycles
    "eta_th": [0.40, 0.50, 0.60],
    # DEC: aggressive end-loss recovery
    "f_dec": [0.30, 0.50, 0.70],
    "eta_de": [0.60, 0.70, 0.80],
    # Heating wall-plug efficiency
    "eta_pin": [0.50, 0.65, 0.80],
    # Compact geometry (lower material costs)
    "plasma_t": [1.5, 1.0, 0.7],
    "blanket_t": [0.60, 0.40, 0.25],
}

TARGET = 10.0  # $/MWh = 1 ¢/kWh

# ── Sweep: vary one parameter at a time ──────────────────────────────
print("Single-parameter sweeps:")
print(f"{'Parameter':<24} {'Value':>8} {'LCOE $/MWh':>12} {'¢/kWh':>8} {'vs base':>8}")
print("-" * 64)

for param, values in sweeps.items():
    for val in values:
        kwargs = {**base_kwargs, param: val}
        r = model.forward(**kwargs)
        lcoe = float(r.costs.lcoe)
        marker = " <--" if lcoe <= TARGET else ""
        delta = lcoe / float(base.costs.lcoe) * 100 - 100
        print(
            f"  {param:<22} {val:>8.2f} {lcoe:>12.1f}"
            f" {lcoe / 10:>8.2f} {delta:>+7.1f}%{marker}"
        )
    print()

# ── Combined optimistic case ─────────────────────────────────────────
print("=" * 64)
print("Combined scenarios:")
print("=" * 64)

scenarios = {
    "Aggressive engineering": dict(
        net_electric_mw=1000,
        eta_th=0.55,
        eta_pin=0.70,
        f_dec=0.50,
        eta_de=0.75,
        plasma_t=1.0,
        blanket_t=0.40,
    ),
    "Aggressive finance": dict(
        construction_time_yr=3.0,
        interest_rate=0.04,
        availability=0.92,
    ),
    "Aggressive all": dict(
        net_electric_mw=1000,
        availability=0.92,
        construction_time_yr=3.0,
        interest_rate=0.04,
        eta_th=0.55,
        eta_pin=0.70,
        f_dec=0.50,
        eta_de=0.75,
        plasma_t=1.0,
        blanket_t=0.40,
    ),
    "Extreme (stretch targets)": dict(
        net_electric_mw=2000,
        availability=0.95,
        construction_time_yr=2.0,
        interest_rate=0.03,
        eta_th=0.60,
        eta_pin=0.80,
        f_dec=0.70,
        eta_de=0.80,
        plasma_t=0.7,
        blanket_t=0.25,
        p_input=20.0,
        p_cool=10.0,
        p_coils=2.0,
        p_cryo=0.5,
    ),
}

# ── Scenarios with cost overrides ─────────────────────────────────────
print("\n" + "=" * 64)
print("Combined scenarios with cost overrides:")
print("=" * 64)

override_scenarios = {
    "Aggressive all + $250M buildings": (
        dict(
            net_electric_mw=1000,
            availability=0.92,
            construction_time_yr=3.0,
            interest_rate=0.04,
            eta_th=0.55,
            eta_pin=0.70,
            f_dec=0.50,
            eta_de=0.75,
            plasma_t=1.0,
            blanket_t=0.40,
        ),
        {"CAS21": 250.0},
    ),
    "Extreme + $250M buildings": (
        dict(
            net_electric_mw=2000,
            availability=0.95,
            construction_time_yr=2.0,
            interest_rate=0.03,
            eta_th=0.60,
            eta_pin=0.80,
            f_dec=0.70,
            eta_de=0.80,
            plasma_t=0.7,
            blanket_t=0.25,
            p_input=20.0,
            p_cool=10.0,
            p_coils=2.0,
            p_cryo=0.5,
        ),
        {"CAS21": 250.0},
    ),
}

for name, (overrides, cost_ovr) in override_scenarios.items():
    kwargs = {**base_kwargs, **overrides, "cost_overrides": cost_ovr}
    r = model.forward(**kwargs)
    lcoe = float(r.costs.lcoe)
    pt = r.power_table
    c = r.costs
    marker = " *** TARGET ***" if lcoe <= TARGET else ""
    print(f"\n{name}:{marker}")
    print(f"  LCOE: {lcoe:.1f} $/MWh ({lcoe / 10:.2f} ¢/kWh)")
    print(
        f"  P_fus: {pt.p_fus:.0f} MW | P_net: {pt.p_net:.0f} MW | Q_eng: {pt.q_eng:.1f}"
    )
    print(
        f"  Overnight: {c.overnight_cost:.0f} $/kW"
        f" | Capital: {float(c.total_capital):.0f} M$"
    )
    print(f"  CAS21 (buildings): {float(c.cas21):.0f} M$")
    print(f"  Cost overrides: {cost_ovr}")
    print(f"  Param overrides: {overrides}")

for name, overrides in scenarios.items():
    kwargs = {**base_kwargs, **overrides}
    r = model.forward(**kwargs)
    lcoe = float(r.costs.lcoe)
    pt = r.power_table
    c = r.costs
    marker = " *** TARGET ***" if lcoe <= TARGET else ""
    print(f"\n{name}:{marker}")
    print(f"  LCOE: {lcoe:.1f} $/MWh ({lcoe / 10:.2f} ¢/kWh)")
    print(
        f"  P_fus: {pt.p_fus:.0f} MW | P_net: {pt.p_net:.0f} MW | Q_eng: {pt.q_eng:.1f}"
    )
    print(
        f"  Overnight: {c.overnight_cost:.0f} $/kW"
        f" | Capital: {float(c.total_capital):.0f} M$"
    )
    print(f"  Overrides: {overrides}")
