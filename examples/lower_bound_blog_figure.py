"""Generate the bar chart for the blog post:
'The Lower Bound for Fusion Energy Cost'

Shows LCOE floor by scenario for DT (full/half/zero staff), D-He3
(full/zero staff, excl. fuel), and pB11 (full/zero staff), with the
1-cent target line and budget annotations.
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
]

# Collect data: 7 series
dt_full = []
dt_half = []
dt_zero = []
dhe3_full = []  # excluding fuel cost
dhe3_zero = []
pb_full = []
pb_zero = []
labels = []

for label, kw in scenarios:
    labels.append(label)
    energy = 8760 * kw["net_electric_mw"] * kw["availability"]

    # D-T
    r_dt = m_dt.forward(
        **kw, inflation_rate=INFLATION, f_dec=0.0, cost_overrides=FREE_CORE
    )
    staff_dt = r_dt.costs.cas71 * 1e6 / energy
    dt_full.append(float(r_dt.costs.lcoe))
    dt_half.append(float(r_dt.costs.lcoe - staff_dt * 0.5))
    dt_zero.append(float(r_dt.costs.lcoe - staff_dt))

    # D-He3 BOP only (subtract fuel)
    r_dhe3 = m_dhe3.forward(
        **kw, inflation_rate=INFLATION, f_dec=0.0, cost_overrides=FREE_CORE
    )
    fuel_mwh = r_dhe3.costs.cas80 * 1e6 / energy
    staff_dhe3 = r_dhe3.costs.cas71 * 1e6 / energy
    bop = float(r_dhe3.costs.lcoe - fuel_mwh)
    dhe3_full.append(bop)
    dhe3_zero.append(bop - staff_dhe3)

    # p-B11
    r_pb = m_pb11.forward(
        **kw, inflation_rate=INFLATION, f_dec=0.0, cost_overrides=FREE_CORE
    )
    staff_pb = r_pb.costs.cas71 * 1e6 / energy
    pb_full.append(float(r_pb.costs.lcoe))
    pb_zero.append(float(r_pb.costs.lcoe - staff_pb))

x = np.arange(len(labels))
n_series = 7
total_width = 0.85
w = total_width / n_series

series = [
    (x - 3 * w, dt_full, "D-T", "#c44e52"),
    (x - 2 * w, dt_half, "D-T (half staff)", "#d4787b"),
    (x - 1 * w, dt_zero, "D-T (zero staff)", "#e89c9e"),
    (x + 0 * w, dhe3_full, "D-He3 (excl. fuel)", "#5a9a3c"),
    (x + 1 * w, dhe3_zero, "D-He3 (zero staff)", "#8dbb72"),
    (x + 2 * w, pb_full, "p-B11", "#4c72b0"),
    (x + 3 * w, pb_zero, "p-B11 (zero staff)", "#7295c4"),
]

fig, ax = plt.subplots(figsize=(12, 7))

for pos, vals, lbl, color in series:
    ax.bar(
        pos,
        vals,
        w,
        label=lbl,
        color=color,
        edgecolor="white",
        linewidth=0.5,
    )

# 1-cent target line
ax.axhline(y=TARGET, color="black", linestyle="--", linewidth=1.5, zorder=5)
ax.text(
    len(labels) - 0.5,
    TARGET + 0.3,
    "$10/MWh target",
    ha="right",
    fontsize=13,
    fontstyle="italic",
)

# Annotate all bars with budget
for pos, vals, lbl, color in series:
    for i, v in enumerate(vals):
        budget = TARGET - v
        if budget >= 0:
            text = f"+${budget:.1f}"
        else:
            text = f"-${-budget:.1f}"
        ax.annotate(
            text,
            xy=(pos[i], v),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=color,
            fontweight="bold",
        )

ax.set_ylabel("LCOE floor ($/MWh)", fontsize=14)
ax.set_xlabel("Scenario", fontsize=14)
ax.set_title(
    "Free-core LCOE floor: fuel choice, scale, and staffing",
    fontsize=16,
)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.legend(fontsize=10, loc="upper right", ncol=3)
ax.set_ylim(0, max(dt_full) * 1.12)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig("docs/blog/1 Floor/lower_bound_floor_chart.png", dpi=150)
print("Saved to docs/blog/1 Floor/lower_bound_floor_chart.png")
plt.show()
