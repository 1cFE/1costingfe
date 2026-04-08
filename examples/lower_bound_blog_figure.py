"""Generate the bar chart for the blog post:
'The Lower Bound for Fusion Energy Cost'

Shows LCOE floor by scenario for DT (full/half/zero staff), D-He3
(excl. fuel), and pB11, with the 1-cent target line and budget
annotations on bars near or below the target.
"""

import matplotlib.pyplot as plt
import numpy as np

from costingfe import ConfinementConcept, CostModel, Fuel
from costingfe.types import PowerCycle

FREE_CORE = {"CAS22": 0.0, "CAS27": 0.0}
INFLATION = 0.0245
TARGET = 10.0

m_dt = CostModel(
    concept=ConfinementConcept.TOKAMAK,
    fuel=Fuel.DT,
    power_cycle=PowerCycle.BRAYTON_SCO2,
)
m_dhe3 = CostModel(
    concept=ConfinementConcept.MIRROR,
    fuel=Fuel.DHE3,
    power_cycle=PowerCycle.BRAYTON_SCO2,
)
m_pb11 = CostModel(
    concept=ConfinementConcept.MIRROR,
    fuel=Fuel.PB11,
    power_cycle=PowerCycle.BRAYTON_SCO2,
)

scenarios = [
    (
        "1 GWe\nbaseline",
        dict(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30),
    ),
    (
        "2 GWe\nbaseline",
        dict(net_electric_mw=2000.0, availability=0.85, lifetime_yr=30),
    ),
    (
        "2 GWe\naggressive",
        dict(
            net_electric_mw=2000.0,
            availability=0.95,
            lifetime_yr=50,
            interest_rate=0.03,
            construction_time_yr=3.0,
        ),
    ),
    (
        "3 GWe\naggressive",
        dict(
            net_electric_mw=3000.0,
            availability=0.95,
            lifetime_yr=50,
            interest_rate=0.03,
            construction_time_yr=3.0,
        ),
    ),
    (
        "5 GWe\nmega",
        dict(
            net_electric_mw=5000.0,
            availability=0.95,
            lifetime_yr=50,
            interest_rate=0.02,
            construction_time_yr=3.0,
        ),
    ),
]

# Collect data: 5 series
dt_full = []
dt_half = []
dt_zero = []
dhe3_bop = []  # excluding fuel cost
pb_full = []
labels = []

for label, kw in scenarios:
    labels.append(label)
    energy = 8760 * kw["net_electric_mw"] * kw["availability"]

    # D-T full staffing
    r_dt = m_dt.forward(**kw, inflation_rate=INFLATION, cost_overrides=FREE_CORE)
    om = r_dt.costs.cas70 * 1e6 / energy
    staff = r_dt.costs.cas71 * 1e6 / energy
    dt_full.append(float(r_dt.costs.lcoe))
    dt_half.append(float(r_dt.costs.lcoe - staff * 0.5))
    dt_zero.append(float(r_dt.costs.lcoe - staff))

    # D-He3 BOP only (subtract fuel)
    r_dhe3 = m_dhe3.forward(**kw, inflation_rate=INFLATION, cost_overrides=FREE_CORE)
    fuel_mwh = r_dhe3.costs.cas80 * 1e6 / energy
    dhe3_bop.append(float(r_dhe3.costs.lcoe - fuel_mwh))

    # p-B11
    r_pb = m_pb11.forward(**kw, inflation_rate=INFLATION, cost_overrides=FREE_CORE)
    pb_full.append(float(r_pb.costs.lcoe))

x = np.arange(len(labels))
n_series = 5
total_width = 0.8
w = total_width / n_series

series = [
    (x - 2 * w, dt_full, "D-T", "#c44e52"),
    (x - 1 * w, dt_half, "D-T (half staff)", "#e89c9e"),
    (x + 0 * w, dt_zero, "D-T (zero staff)", "#f2cdce"),
    (x + 1 * w, dhe3_bop, "D-He3 (excl. fuel)", "#8dbb72"),
    (x + 2 * w, pb_full, "p-B11", "#4c72b0"),
]

fig, ax = plt.subplots(figsize=(12, 6.5))

TARGET_CKW = TARGET / 10  # 1 cent/kWh

for pos, vals, lbl, color in series:
    vals_ckw = [v / 10 for v in vals]
    ax.bar(pos, vals_ckw, w, label=lbl, color=color, edgecolor="white", linewidth=0.5)

# 1-cent target line
ax.axhline(y=TARGET_CKW, color="black", linestyle="--", linewidth=1.5, zorder=5)
ax.text(
    len(labels) - 0.5,
    TARGET_CKW + 0.03,
    "1 cent target",
    ha="right",
    fontsize=10,
    fontstyle="italic",
)

# Annotate budget only on bars near or below the target
for pos, vals, lbl, color in series:
    for i, v in enumerate(vals):
        budget = TARGET - v
        if True:  # annotate all bars
            budget_ckw = budget / 10
            sign = "+" if budget_ckw >= 0 else ""
            ax.annotate(
                f"{sign}{budget_ckw:.2f}",
                xy=(pos[i], v / 10),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=color,
                fontweight="bold",
            )

ax.set_ylabel("LCOE floor (\u00a2/kWh)", fontsize=12)
ax.set_xlabel("Scenario", fontsize=12)
ax.set_title("Free-core LCOE floor: fuel choice, scale, and staffing", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.legend(fontsize=9, loc="upper right", ncol=2)
ax.set_ylim(0, max(dt_full) / 10 * 1.12)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig("docs/blog/lower_bound_floor_chart.png", dpi=150)
print("Saved to docs/blog/lower_bound_floor_chart.png")
plt.show()
