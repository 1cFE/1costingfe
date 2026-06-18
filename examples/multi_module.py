"""Example: Multi-module plants — economy of scale via n_mod."""

from costingfe import ConfinementConcept, CostModel, Fuel

model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)

print("Multi-Module Economy of Scale — DT Tokamak, 1 GWe total")
print(
    f"{'n_mod':>6} {'CAS22/mod':>12} {'CAS22 total':>12} {'Overnight':>12} {'LCOE':>10}"
)
print(f"{'':>6} {'M$':>12} {'M$':>12} {'$/kW':>12} {'$/MWh':>10}")
print("-" * 54)

for n in [1, 2, 4]:
    r = model.forward(
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
        inflation_rate=0.0245,
        n_mod=n,
    )
    # Sum per-module sub-accounts (C2201xx) to get per-module CAS22 contribution
    per_mod_keys = [k for k in r.cas22_detail if k.startswith("C2201")]
    per_mod_cas22 = sum(r.cas22_detail[k] for k in per_mod_keys)
    total_cas22 = r.costs.cas22

    print(
        f"{n:>6} {per_mod_cas22:>12.0f} {total_cas22:>12.0f} "
        f"{r.costs.capital_per_kw:>12.0f} {r.costs.lcoe:>10.1f}"
    )
