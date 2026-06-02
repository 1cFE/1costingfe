"""Generate the Section 2.3 sensitivity tornado for the paper.

Reads the canonical mirror+D-3He+DEC scenario, computes elasticities via
JAX autodiff (model.sensitivity), and renders a horizontal bar chart with
three colored bands: physics outputs, cost unit prices, financial /
methodology.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from costingfe import ConfinementConcept, CostModel, ForwardResult, Fuel

# Three-bucket assignment.
# model.sensitivity() returns keys: "engineering", "financial", "costing".
# "engineering" -> physics outputs / conversion physics -> bucket "physics"
# "costing"     -> cost unit prices                    -> bucket "costing"
# "financial"   -> financial / methodology             -> bucket "financial"
_SENS_TO_BUCKET: dict[str, str] = {
    "engineering": "physics",
    "costing": "costing",
    "financial": "financial",
}
_COLORS = {
    "physics": "#1f77b4",
    "costing": "#d62728",
    "financial": "#2ca02c",
}
_LABELS = {
    "physics": "Physics outputs",
    "costing": "Cost unit prices",
    "financial": "Financial / methodology",
}

assert set(_LABELS) == set(_COLORS) == set(_SENS_TO_BUCKET.values()), (
    "_LABELS, _COLORS, and _SENS_TO_BUCKET must share the same bucket keys"
)

_SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = _SCRIPT_DIR.parent / "figures"

# Canonical kwargs matching examples/external_physics_handoff.py exactly.
_BASE_KWARGS: dict = dict(
    net_electric_mw=1000.0,
    availability=0.87,
    lifetime_yr=30,
    construction_time_yr=6.0,
    interest_rate=0.07,
    inflation_rate=0.02,
    noak=True,
    R0=0.0,
    chamber_length=80.0,
    plasma_t=0.4,
    elon=1.0,
    blanket_t=0.30,
    ht_shield_t=0.20,
    structure_t=0.15,
    vessel_t=0.10,
    b_center=12.0,
    r_bore=1.85,
    p_input=50.0,
    eta_pin=0.60,
    eta_p=0.50,
    p_coils=5.0,
    p_cool=25.0,
    p_pump=1.5,
    p_house=4.0,
    p_cryo=1.0,
    f_sub=0.03,
    mn=1.05,
    eta_th=0.40,
    eta_de=0.70,
    f_dec=0.90,
    n_e=3.3e19,
    T_e=70.0,
    Z_eff=1.3,
    B=3.0,
    plasma_volume=400.0,
    T_edge=0.20,
    tau_ratio=3.0,
    R_w=0.4,
    dhe3_f_T=0.5,
    dhe3_f_He3=0.1,
    dhe3_dd_frac=0.131,
    dd_f_T=0.969,
    dd_f_He3=0.689,
)


def base_model_and_result() -> tuple[CostModel, ForwardResult]:
    model = CostModel(concept=ConfinementConcept.MIRROR, fuel=Fuel.DHE3)
    result = model.forward(**_BASE_KWARGS)
    return model, result


def collect_elasticities(model: CostModel, base) -> list[tuple[str, float, str]]:
    sens = model.sensitivity(base.params)
    rows: list[tuple[str, float, str]] = []
    for sens_key, bucket in _SENS_TO_BUCKET.items():
        for param, e in sens[sens_key].items():
            if abs(e) > 1e-4:
                rows.append((param, float(e), bucket))
    rows.sort(key=lambda x: abs(x[1]), reverse=True)
    return rows


def render_tornado(rows: list[tuple[str, float, str]], out_path: Path) -> None:
    rows = rows[:18]  # top 18 rows fit on one PDF page at this figsize
    rows_plot = list(reversed(rows))  # largest at top in matplotlib horizontal bar
    fig, ax = plt.subplots(figsize=(7.0, 0.32 * len(rows_plot) + 1.2))
    for i, (param, e, bucket) in enumerate(rows_plot):
        ax.barh(i, e, color=_COLORS[bucket])
    ax.set_yticks(range(len(rows_plot)))
    ax.set_yticklabels([param for param, _, _ in rows_plot])
    ax.set_xlabel("LCOE elasticity (% change in LCOE per 1% change in parameter)")
    ax.axvline(0, color="black", lw=0.5)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_COLORS[k]) for k in _SENS_TO_BUCKET.values()
    ]
    ax.legend(
        handles,
        [_LABELS[k] for k in _SENS_TO_BUCKET.values()],
        loc="lower left",
        frameon=False,
        fontsize="small",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    model, base = base_model_and_result()
    rows = collect_elasticities(model, base)
    render_tornado(rows, FIG_DIR / "tornado.pdf")
    print(f"Wrote {FIG_DIR / 'tornado.pdf'}")


if __name__ == "__main__":
    main()
