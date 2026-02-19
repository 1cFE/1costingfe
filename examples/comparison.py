"""Example: Compare all reactor concepts and fuel types."""

from costingfe import CostModel, ConfinementConcept, Fuel, compare_all

# ── Fuel Comparison (DT Tokamak) ────────────────────────────────────
print("Fuel Comparison — 1 GWe Tokamak")
print(f"{'Fuel':<8} {'LCOE':>8} {'Capital':>10} {'CAS22':>10} {'Overnight':>10} {'P_fus':>8}")
print(f"{'':8} {'$/MWh':>8} {'M$':>10} {'M$':>10} {'$/kW':>10} {'MW':>8}")
print("-" * 58)

for fuel in Fuel:
    m = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=fuel)
    r = m.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    c = r.costs
    print(f"{fuel.value:<8} {c.lcoe:>8.1f} {c.total_capital:>10.0f} "
          f"{c.cas22:>10.0f} {c.overnight_cost:>10.0f} {r.power_table.p_fus:>8.0f}")

# ── Concept Comparison (DT fuel) ───────────────────────────────────
print("\nConcept Comparison — 1 GWe DT")
print(f"{'Concept':<16} {'LCOE':>8} {'Capital':>10} {'Overnight':>10} "
      f"{'Q_eng':>8} {'Recirc':>8}")
print("-" * 64)

for concept in ConfinementConcept:
    m = CostModel(concept=concept, fuel=Fuel.DT)
    r = m.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    c = r.costs
    pt = r.power_table
    print(f"{concept.value:<16} {c.lcoe:>8.1f} {c.total_capital:>10.0f} "
          f"{c.overnight_cost:>10.0f} {pt.q_eng:>8.1f} {pt.rec_frac*100:>7.1f}%")

# ── Full Ranking (all combinations) ────────────────────────────────
print("\nFull Ranking — All Concept x Fuel (top 15)")
print(f"{'#':>3} {'Concept':<16} {'Fuel':<6} {'LCOE':>8} {'Capital':>10} {'Overnight':>10}")
print("-" * 56)

results = compare_all(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
for i, r in enumerate(results[:15]):
    c = r.result.costs
    print(f"{i+1:>3} {r.concept.value:<16} {r.fuel.value:<6} "
          f"{r.lcoe:>8.1f} {c.total_capital:>10.0f} {c.overnight_cost:>10.0f}")

print(f"\n{len(results)} viable combinations total.")
