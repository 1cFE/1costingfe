"""Tornado figure for blog post #3. Run from the freeze worktree:

cd /mnt/c/Users/talru/1cfe/1costingfe-freeze
uv run python \
    "/mnt/c/Users/talru/1cfe/1costingfe/docs/blog/3 Intro/blog3_tornado_figure.py"
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from costingfe import ConfinementConcept, CostModel, Fuel

LABELS = {
    "availability": "Availability",
    "interest_rate": "Cost of capital (WACC)",
    "construction_time_yr": "Construction time",
    "lifetime_yr": "Plant lifetime",
    "eta_th": "Thermal cycle efficiency",
    "eta_pin": "Heating wall-plug efficiency",
    "eta_couple": "Heating coupling efficiency",
    "B": "On-axis field",
    "b_center": "On-axis field",
    "r_bore": "Coil winding radius",
    "net_electric_mw": "Net electric power",
    "inflation_rate": "Inflation rate",
    "R0": "Major radius",
    "elon": "Plasma elongation",
    "plasma_t": "Plasma minor radius",
    "blanket_t": "Blanket thickness",
    "mn": "Blanket energy multiplication",
}

model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
result = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30.0)
s = model.sensitivity(result.params)
merged = {**s["engineering"], **s["financial"]}
top = sorted(merged.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
top.reverse()  # largest bar on top after barh

names = [LABELS.get(k, k) for k, _ in top]
vals = [v for _, v in top]
colors = ["#dc2626" if v > 0 else "#16a34a" for v in vals]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(names, vals, color=colors)
ax.axvline(0, color="#9ca3af", lw=0.8)
ax.set_xlabel("LCOE elasticity (% change in LCOE per % change in parameter)")
ax.set_title("What moves the cost of a 1 GWe D-T tokamak")
fig.tight_layout()
out = Path(__file__).parent / "tornado_dt_tokamak.png"
fig.savefig(out, dpi=200)
print(f"wrote {out}")
print({k: round(v, 3) for k, v in top})
