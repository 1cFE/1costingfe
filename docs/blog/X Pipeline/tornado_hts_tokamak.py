"""Generate sensitivity tornado chart for the HTS compact tokamak (Concept A).

Reproduces the sensitivity analysis from the blog post:
'From papers to plant economics'
"""

from costingfe import ConfinementConcept, CostModel, Fuel

# CPI inflation 2014 -> 2024
_CPI = 1.34

# ARC-specific cost overrides (Sorbom et al. 2015, CPI-adjusted)
OVERRIDES = {
    "C220103": round(5150.0 * _CPI, 1),  # Magnets + structure
    "C220101": round(260.0 * _CPI, 1),  # FLiBe blanket
    "C220106": round(92.0 * _CPI, 1),  # Vacuum vessel
    "CAS27": 146.0,  # FLiBe inventory
}

model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)

result = model.forward(
    net_electric_mw=261.0,
    availability=0.75,
    lifetime_yr=30,
    n_mod=1,
    construction_time_yr=5.0,
    interest_rate=0.07,
    inflation_rate=0.0245,
    noak=True,
    R0=3.3,
    plasma_t=1.13,
    elon=1.84,
    blanket_t=0.80,
    ht_shield_t=0.20,
    structure_t=0.20,
    vessel_t=0.20,
    p_input=38.6,
    mn=1.1,
    eta_th=0.46,
    eta_p=0.5,
    eta_pin=0.5,
    eta_de=0.85,
    f_sub=0.03,
    f_dec=0.0,
    p_coils=2.0,
    p_cool=13.7,
    p_pump=1.0,
    p_trit=10.0,
    p_house=4.0,
    p_cryo=0.5,
    cost_overrides=OVERRIDES,
)

sens = model.sensitivity(result.params, cost_overrides=OVERRIDES)

# Merge all categories, skip near-zero
all_params = {}
for category in ("engineering", "financial"):
    for param, elasticity in sens[category].items():
        if abs(elasticity) > 0.005:
            all_params[param] = elasticity

# Sort by absolute elasticity
ranked = sorted(all_params.items(), key=lambda x: abs(x[1]), reverse=True)

# Friendly names
LABELS = {
    "availability": "Availability",
    "construction_time_yr": "Construction time",
    "interest_rate": "Cost of capital (WACC)",
    "inflation_rate": "Inflation rate",
    "eta_th": "Thermal efficiency",
    "eta_pin": "Heating wall-plug eff.",
    "p_input": "Auxiliary heating power",
    "p_nbi": "NBI heating cost",
    "r_coil": "Coil winding radius",
    "R0": "Major radius",
    "elon": "Elongation",
    "b_max": "Peak field on conductor",
    "mn": "Neutron multiplier",
    "blanket_t": "Blanket thickness",
    "ht_shield_t": "Shield thickness",
    "p_cool": "Cooling power",
    "f_sub": "Subsystem fraction",
    "p_trit": "Tritium processing power",
}

# ASCII tornado
BAR_WIDTH = 30
max_abs = max(abs(e) for _, e in ranked) if ranked else 1.0

print("Sensitivity Tornado: HTS Compact Tokamak (Concept A)")
print(f"Baseline LCOE: {result.costs.lcoe:.1f} $/MWh")
print("261 MWe, 75% CF, 30 yr, 7% WACC, NOAK")
print("Published magnet costs (Sorbom et al. 2015, CPI-adjusted)\n")

print(
    f"{'Parameter':<28} {'Elasticity':>10}  {'cost decrease':>{BAR_WIDTH}} "
    f"| {'cost increase'}"
)
print("-" * (28 + 10 + 2 + BAR_WIDTH + 3 + BAR_WIDTH))

for param, elast in ranked[:15]:
    label = LABELS.get(param, param)
    bar_len = int(abs(elast) / max_abs * BAR_WIDTH)
    if elast < 0:
        left_bar = " " * (BAR_WIDTH - bar_len) + "\u2588" * bar_len
        right_bar = ""
    else:
        left_bar = " " * BAR_WIDTH
        right_bar = "\u2588" * bar_len
    print(f"{label:<28} {elast:>+10.4f}  {left_bar} | {right_bar}")

print("\nElasticity = % change in LCOE per 1% change in parameter.")
print("A 1% increase in availability reduces LCOE by 0.96%.")
print("A 1% increase in construction time raises LCOE by 0.30%.")
