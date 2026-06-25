"""Build the side-by-side stacked-bar figure of capital cost composition
for ARC and ARIES-AT, reading the JSON outputs of benchmark_arc.py and
benchmark_aries_at.py.

Run as a script (after the two benchmark scripts have been run):
    python docs/papers/1costingfe_paper/scripts/make_benchmark_bars.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

GROUPS: dict[str, tuple[str, ...]] = {
    "Buildings (CAS21)": ("cas21",),
    "Reactor (CAS22)": ("cas22",),
    "BoP (CAS23-26)": ("cas23", "cas24", "cas25", "cas26"),
    "Other capital": (
        "cas10",
        "cas27",
        "cas28",
        "cas29",
        "cas30",
        "cas40",
        "cas50",
    ),
    "Financial (CAS60)": ("cas60",),
}


def _aggregate(payload: dict) -> dict[str, float]:
    cas = payload["cas"]
    return {label: sum(cas[k] for k in keys) for label, keys in GROUPS.items()}


def run(input_dir: Path, figure_dir: Path) -> Path:
    """Read the two benchmark JSONs from input_dir; write the PDF into figure_dir."""
    input_dir = Path(input_dir)
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    arc = json.loads((input_dir / "arc.json").read_text())
    aries = json.loads((input_dir / "aries_at.json").read_text())

    arc_groups = _aggregate(arc)
    aries_groups = _aggregate(aries)
    labels = list(GROUPS.keys())

    fig, ax = plt.subplots(figsize=(7, 5))

    # One fixed color per CAS group, shared by both bars, so the single legend
    # entry per group matches the ARC and ARIES-AT stacks alike.
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    bottoms_arc = 0.0
    bottoms_aries = 0.0
    for i, label in enumerate(labels):
        color = colors[i % len(colors)]
        ax.bar(
            [0],
            [arc_groups[label]],
            bottom=bottoms_arc,
            label=label,
            width=0.6,
            color=color,
        )
        ax.bar(
            [1],
            [aries_groups[label]],
            bottom=bottoms_aries,
            width=0.6,
            color=color,
        )
        bottoms_arc += arc_groups[label]
        bottoms_aries += aries_groups[label]

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["ARC (270 MWe)", "ARIES-AT (1 GWe)"])
    ax.set_ylabel("Capital cost (M\\$, 2025)")
    ax.set_title("Capital cost composition by major CAS account group")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.tight_layout()

    out = figure_dir / "benchmark_lcoe_stacks.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    here = Path(__file__).parent
    out_path = run(input_dir=here / "_outputs", figure_dir=here.parent / "figures")
    print(f"Wrote figure to {out_path}")
