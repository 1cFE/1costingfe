"""Example: Scaling cost overrides from a reference plant to a target plant.

Use case: you have cost data for a 400 MWe plant design (e.g. vendor quotes,
engineering estimates, or published cost breakdowns) and want to estimate the
LCOE of a 1000 MWe plant using 1costingfe's built-in scaling laws.

The override_reference_mw parameter tells the framework that your cost
overrides are valid at 400 MWe.  It runs the model at both 400 and 1000 MWe,
computes the ratio for each overridden account, and applies that ratio to
your data.  This preserves the model's per-account scaling (which depends on
different power quantities: p_net, p_th, p_fus, p_et) while anchoring to
your empirical numbers.
"""

from costingfe import ConfinementConcept, CostModel, Fuel

model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)

REFERENCE_MW = 400.0
TARGET_MW = 1000.0

# Cost data at 400 MWe (M$).
# These could come from a published design study, vendor estimates,
# or a detailed bottom-up analysis at the reference scale.
overrides_at_ref = {
    "CAS21": 450.0,  # Buildings: site-specific estimate
    "CAS22": 2500.0,  # Reactor plant equipment: engineering estimate
    "CAS23": 120.0,  # Turbine plant: vendor quote
}

# ── Baseline at target power (no overrides) ──────────────────────
baseline = model.forward(
    net_electric_mw=TARGET_MW,
    availability=0.85,
    lifetime_yr=30,
)

# ── Scaled overrides ─────────────────────────────────────────────
# The framework scales each override using the model's own account
# ratios between 400 and 1000 MWe.
scaled = model.forward(
    net_electric_mw=TARGET_MW,
    availability=0.85,
    lifetime_yr=30,
    cost_overrides=overrides_at_ref,
    override_reference_mw=REFERENCE_MW,
)

# ── For comparison: what the model computes at the reference power ─
ref_baseline = model.forward(
    net_electric_mw=REFERENCE_MW,
    availability=0.85,
    lifetime_yr=30,
)

# ── Results ──────────────────────────────────────────────────────
print(f"Scaled Cost Overrides: {REFERENCE_MW:.0f} MWe -> {TARGET_MW:.0f} MWe")
print("DT Tokamak, NOAK\n")

header = (
    f"{'Account':<16}"
    f"{'Model @':>10} {'Your @':>10} {'Model @':>10} {'Scaled @':>10}"
    f"{'Scale':>8}"
)
units = (
    f"{'':16}"
    f"{f'{REFERENCE_MW:.0f} MWe':>10} {f'{REFERENCE_MW:.0f} MWe':>10}"
    f"{f'{TARGET_MW:.0f} MWe':>10} {f'{TARGET_MW:.0f} MWe':>10}"
    f"{'factor':>8}"
)
print(header)
print(units)
print("-" * 74)

account_map = {
    "CAS21": ("cas21", "Buildings"),
    "CAS22": ("cas22", "Reactor equip"),
    "CAS23": ("cas23", "Turbine plant"),
}

for key, (attr, label) in account_map.items():
    ref_model = getattr(ref_baseline.costs, attr)
    ref_yours = overrides_at_ref[key]
    tgt_model = getattr(baseline.costs, attr)
    tgt_scaled = getattr(scaled.costs, attr)
    factor = tgt_scaled / ref_yours if ref_yours > 0 else 0

    print(
        f"{label:<16}"
        f"{ref_model:>10.1f} {ref_yours:>10.1f}"
        f"{tgt_model:>10.1f} {tgt_scaled:>10.1f}"
        f"{factor:>8.3f}"
    )

print("-" * 74)
print(
    f"{'Total capital':<16}"
    f"{'':>10} {'':>10}"
    f"{baseline.costs.total_capital:>10.0f} {scaled.costs.total_capital:>10.0f}"
)
print(
    f"{'LCOE ($/MWh)':<16}"
    f"{'':>10} {'':>10}"
    f"{baseline.costs.lcoe:>10.1f} {scaled.costs.lcoe:>10.1f}"
)
print(f"\nOverridden accounts: {', '.join(scaled.overridden)}")
