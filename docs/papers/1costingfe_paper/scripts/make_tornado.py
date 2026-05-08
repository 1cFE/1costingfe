"""Generate the Section 2.3 sensitivity tornado for the paper.

Reads the canonical mirror+D-3He+DEC scenario, computes elasticities via
JAX autodiff (model.sensitivity), and renders a horizontal bar chart with
three colored bands: physics outputs, cost unit prices, financial /
methodology. Also writes a small text table comparing the top-5 autodiff
elasticities against centered finite differences as an exactness check.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from costingfe import ConfinementConcept, CostModel, Fuel

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

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR.parent / "figures"
TABLE_OUT = OUT_DIR / "_outputs" / "tornado_table.txt"

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
    b_max=12.0,
    r_coil=1.85,
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


def base_model_and_result():
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
    rows = rows[:18]  # keep top 18 to fit on one page
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
        loc="lower right",
        frameon=False,
        fontsize="small",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _centered_fd_elasticity(
    model: CostModel, base, param: str, h: float = 1e-3
) -> float:
    """Centered finite-difference elasticity at the base point.

    Uses a 0.1% perturbation of the named parameter.  Reconstructs the
    full kwargs dict from base.params, overriding only the one parameter.
    """
    p0 = float(base.params[param])
    p_lo = p0 * (1.0 - h)
    p_hi = p0 * (1.0 + h)

    # Use batch_lcoe: two-point evaluation via a single vmap call.
    lcoe_lo, lcoe_hi = model.batch_lcoe(
        param_sets={param: [p_lo, p_hi]},
        params=base.params,
    )
    base_lcoe = float(base.costs.lcoe)
    d_lcoe = lcoe_hi - lcoe_lo
    d_param = p_hi - p_lo
    return (d_lcoe / d_param) * (p0 / base_lcoe)


def write_fd_table(
    model: CostModel,
    base,
    top_rows: list[tuple[str, float, str]],
    out_path: Path,
    k: int = 5,
) -> None:
    """Write centered finite differences vs autodiff for the top-k elasticities."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "param            autodiff       FD             rel_err",
        "----------------------------------------------------------",
    ]
    for param, e_auto, _ in top_rows[:k]:
        e_fd = _centered_fd_elasticity(model, base, param)
        rel = abs(e_auto - e_fd) / max(abs(e_auto), 1e-12)
        lines.append(f"{param:<16s} {e_auto:+.6f}  {e_fd:+.6f}  {rel:.2e}")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    model, base = base_model_and_result()
    rows = collect_elasticities(model, base)
    render_tornado(rows, FIG_DIR / "tornado.pdf")
    write_fd_table(model, base, rows, TABLE_OUT)
    print(f"Wrote {FIG_DIR / 'tornado.pdf'}")
    print(f"Wrote {TABLE_OUT}")


if __name__ == "__main__":
    main()
