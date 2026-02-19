"""Example: Generate fusion-backcasting subsystems from 1costingfe physics model.

Shows how 1costingfe populates fusion-backcasting's subsystem costs
with physics-based values instead of static defaults.
"""

import json
from costingfe.backcasting_bridge import generate_subsystems, generate_subsystems_json

# ── Generate subsystems for a DT tokamak ────────────────────────────
subsystems, financial = generate_subsystems(
    concept="tokamak", fuel="dt",
    net_electric_mw=1000.0, availability=0.85, lifetime_yr=30,
)

print("fusion-backcasting Subsystems (from 1costingfe physics model)")
print(f"{'Account':<10} {'Name':<28} {'CapEx M$':>10} {'O&M M$/yr':>10}")
print("-" * 62)

total_capex = 0
for s in subsystems:
    print(f"{s['account']:<10} {s['name']:<28} {s['absolute_capital_cost']:>10.1f} "
          f"{s['absolute_fixed_om']:>10.1f}")
    total_capex += s["absolute_capital_cost"]
print("-" * 62)
print(f"{'':10} {'Total':28} {total_capex:>10.1f}")

print(f"\nFinancial: WACC={financial['wacc']:.0%}, "
      f"CF={financial['capacity_factor']:.0%}, "
      f"Life={financial['lifetime']}yr, "
      f"Capacity={financial['capacity_mw']:.0f}MW")

# ── Compare across fuels ────────────────────────────────────────────
print("\n\nSubsystem costs by fuel (tokamak, M$):")
print(f"{'Account':<10} {'Name':<22} {'DT':>8} {'DD':>8} {'DHe3':>8} {'pB11':>8}")
print("-" * 60)

fuel_data = {}
for fuel in ["dt", "dd", "dhe3", "pb11"]:
    subs, _ = generate_subsystems(concept="tokamak", fuel=fuel)
    fuel_data[fuel] = {s["account"]: s["absolute_capital_cost"] for s in subs}

accounts = [(s["account"], s["name"]) for s in subsystems]
for acc, name in accounts:
    vals = [fuel_data[f].get(acc, 0) for f in ["dt", "dd", "dhe3", "pb11"]]
    print(f"{acc:<10} {name:<22} {vals[0]:>8.1f} {vals[1]:>8.1f} {vals[2]:>8.1f} {vals[3]:>8.1f}")

# ── Export as JSON (drop-in for fusion-backcasting) ─────────────────
print("\n\nJSON output (for fusion-backcasting API):")
data = generate_subsystems_json(concept="tokamak", fuel="dt")
print(json.dumps(data, indent=2)[:500] + "\n  ...")
