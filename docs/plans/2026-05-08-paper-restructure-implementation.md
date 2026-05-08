# Paper Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure `1costingfe_paper.tex` into an operational reference: add a worked tour Section 2 (mirror + D-${}^3$He + DEC, with JAX-derived sensitivity tornado), tighten the physics chapter by moving pulsed + inverse balance to a new Appendix C, compress the CAS chapter via a Disposition column, finish three deferred benchmarking discussions, and apply small edits to abstract / conclusion / intro / code availability.

**Architecture:** Two new Python artifacts produce all numerical content for Section 2: `examples/external_physics_handoff.py` (a runnable example) emits the §2.2 reference rollup, and `docs/papers/1costingfe_paper/scripts/make_tornado.py` emits both `figures/tornado.pdf` and an autodiff-vs-finite-difference comparison table. Reuse the sensitivity API exercised in the existing `examples/tornado_plot.py` rather than rolling the gradient logic from scratch. The other tasks are LaTeX edits with build verification.

**Tech Stack:** LaTeX (pdflatex, TikZ, listings, booktabs, cleveref); Python 3.11+; JAX (already a project dependency); matplotlib for the tornado PDF; pytest for smoke tests.

**Spec:** `docs/plans/2026-05-07-paper-restructure-design.md`. The reference scenario (mirror + D-${}^3$He + venetian-blind DEC) is locked there.

**Memory rules to honor throughout:**
- No em dashes (Unicode `—`) in `paper.tex` prose; replace with commas, parentheses, or LaTeX `--`.
- No tildes for approximate values; write "approximately" or use ranges.
- No "earlier draft said X" / "previously this section did Y" framing in `paper.tex`.
- One-liner commit messages; no `Co-Authored-By` line.

---

## File Structure

**Files to create:**
- `examples/external_physics_handoff.py` -- runnable mirror + D-${}^3$He + DEC reference; emits the rollup numbers cited in §2.2.
- `docs/papers/1costingfe_paper/scripts/make_tornado.py` -- generates `figures/tornado.pdf` and the autodiff-vs-FD table for §2.3.
- `docs/papers/1costingfe_paper/figures/tornado.pdf` -- output of the above.
- `tests/test_external_physics_handoff.py` -- smoke test that the example runs without error.

**Files to modify:**
- `docs/papers/1costingfe_paper/1costingfe_paper.tex` -- the bulk of the restructure.
- `docs/papers/1costingfe_paper/todo.md` -- mark items resolved.

**Files left untouched:** package source (`src/costingfe/`), other examples, sister paper, account-justification docs.

---

## Task 1: Add Section 2 skeleton + `listings` package

**Files:**
- Modify: `docs/papers/1costingfe_paper/1costingfe_paper.tex` (preamble lines 1-21, plus a new section between `\section{Introduction}` and `\section{Economics Module}`)

- [ ] **Step 1: Add the `listings` package and configure it for short Python listings.**

In the preamble (after `\usepackage{xcolor}` near line 20), insert:

```latex
\usepackage{listings}
\lstset{
  basicstyle=\ttfamily\small,
  language=Python,
  showstringspaces=false,
  columns=fullflexible,
  keepspaces=true,
  breaklines=true,
  frame=single,
  framesep=4pt,
  xleftmargin=0pt,
}
```

- [ ] **Step 2: Insert the empty Section 2 skeleton between Sections 1 and 2 (currently 2 = Economics).**

Locate `\section{Economics Module}` (currently line 259) and insert immediately above it:

```latex
%% ============================================================
\section{From Customer Requirements to LCOE: A Worked Tour}
\label{sec:tour}

\subsection{Pipeline overview}
\label{sec:tour-pipeline}

% TASK 4 fills this section.

\subsection{Forward call from physics outputs}
\label{sec:tour-forward}

% TASK 3 fills this section.

\subsection{Sensitivity from automatic differentiation}
\label{sec:tour-sensitivity}

% TASK 6 fills this section.

\subsection{Other API surfaces}
\label{sec:tour-api}

% TASK 7 fills this section.
```

The four `% TASK N fills` markers are placeholders that later tasks remove when they populate each subsection.

- [ ] **Step 3: Build to verify nothing is broken.**

Run from `docs/papers/1costingfe_paper/`:
```
pdflatex -interaction=nonstopmode 1costingfe_paper.tex
```
Expected: PDF builds; new "From Customer Requirements to LCOE: A Worked Tour" appears as Section 2 (Economics shifts to Section 3, etc.). Cross-references to other sections resolve as before.

- [ ] **Step 4: Commit.**

```
git add docs/papers/1costingfe_paper/1costingfe_paper.tex
git commit -m "Add Section 2 skeleton (worked tour) and listings package"
```

---

## Task 2: Write `examples/external_physics_handoff.py`

**Files:**
- Create: `examples/external_physics_handoff.py`
- Create: `tests/test_external_physics_handoff.py`

The reference scenario from the spec: 1 GWe D-${}^3$He steady-state mirror with venetian-blind DEC. Physics outputs to feed in are listed in the spec §2.2 table.

- [ ] **Step 1: Read the existing mirror + D-${}^3$He examples to confirm the API surface for this concept + fuel + DEC combination.**

```bash
sed -n '1,80p' /mnt/c/Users/talru/1cfe/1costingfe/examples/dhe3_mix_optimization.py
sed -n '1,80p' /mnt/c/Users/talru/1cfe/1costingfe/examples/dt_mirror.py
sed -n '1,40p' /mnt/c/Users/talru/1cfe/1costingfe/examples/tornado_plot.py
```

These three files together show: how to instantiate a mirror `CostModel`, how to feed physics outputs as forward kwargs, and how to call `sensitivity()`. Reuse their patterns; do not invent a new API surface.

If `model.forward()` for `MIRROR` requires the in-flight 0D mirror physics layer that has not yet merged, fall back to passing `use_0d_model=False` (see `src/costingfe/validation.py:_PLASMA_0D_FIELDS`) and supplying `p_fus` and machine geometry directly. Confirm this path runs before continuing.

- [ ] **Step 2: Write the failing smoke test.**

Create `tests/test_external_physics_handoff.py`:

```python
"""Smoke test: the canonical Section 2.2 reference scenario runs and produces
group-level LCOE rollups."""

import subprocess
import sys
from pathlib import Path


def test_external_physics_handoff_runs():
    """Running the example produces a non-zero LCOE and the rollup keys the
    paper's Table relies on."""
    script = Path(__file__).resolve().parents[1] / "examples" / "external_physics_handoff.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    # Required group-level rollup keys appear, in M$ rows.
    for key in ("CAS10", "CAS21", "CAS22", "CAS40", "CAS50", "CAS60",
                "CAS70", "CAS80", "CAS90", "Total overnight", "LCOE"):
        assert key in out, f"missing rollup row: {key}"
```

- [ ] **Step 3: Run the test to verify it fails (script does not yet exist).**

```bash
cd /mnt/c/Users/talru/1cfe/1costingfe && pytest tests/test_external_physics_handoff.py -v
```

Expected: FAIL with the example file not existing (returncode != 0).

- [ ] **Step 4: Write the example.**

Create `examples/external_physics_handoff.py`:

```python
"""Canonical 1 GWe D-3He steady-state mirror reference for the 1costingfe paper.

This is the example cited in Section 2.2: physics outputs from an external
model (P_fus, central-cell length, plasma radius, fields, temperatures,
densities, secondary burn fractions, DEC efficiencies) hand off to the
1costingfe forward call, which produces a complete CAS-account rollup and an
LCOE figure.

Numbers are illustrative; the point is the handoff pattern, not the design.
"""

from costingfe import ConfinementConcept, CostModel, Fuel


def main() -> None:
    model = CostModel(concept=ConfinementConcept.MIRROR, fuel=Fuel.DHE3)

    result = model.forward(
        # Customer parameters (availability defaults to 0.87 for mirror)
        net_electric_mw=1000.0,
        lifetime_yr=30,
        construction_time_yr=6.0,

        # Physics outputs (see paper Section 2.2 Table)
        p_fus=2400.0,             # MW
        chamber_length=80.0,      # m, central cell length
        plasma_t=0.4,             # m, plasma radius at midplane
        b_max=12.0,               # T, peak field on conductor
        # Note: midplane field B_cc and mirror ratio R_m are derived in the
        # 0D mirror model; here we feed b_max directly for coil costing.
        T_e=70.0,
        n_e=3.3e19,

        # Conversion (DEC + thermal)
        eta_th=0.40,
        eta_dec=0.70,
        f_dec=0.90,

        # D-3He secondary-burn fractions (steady-state mirror defaults)
        f_T_secondary=0.5,
        f_He3_secondary=0.1,

        # If the 0D mirror physics layer has not yet merged at execution time,
        # set use_0d_model=False and supply p_fus + b_max as above.
        use_0d_model=False,
    )

    costs = result.costs

    # Group-level rollup (matches the paper Section 2.2 table)
    rows = [
        ("CAS10",                costs.cas10),
        ("CAS21",                costs.cas21),
        ("CAS22",                costs.cas22),
        ("CAS23-26",             costs.cas23 + costs.cas24 + costs.cas25 + costs.cas26),
        ("CAS27-30",             costs.cas27 + costs.cas28 + costs.cas29 + costs.cas30),
        ("CAS40",                costs.cas40),
        ("CAS50",                costs.cas50),
        ("CAS60",                costs.cas60),
        ("Total overnight",      costs.overnight_cost_musd),
        ("CAS70 (M$/yr)",        costs.cas70),
        ("CAS80 (M$/yr)",        costs.cas80),
        ("CAS90 (M$/yr)",        costs.cas90),
        ("LCOE",                 costs.lcoe),
    ]
    print("1 GWe D-3He mirror with venetian-blind DEC, NOAK reference\n")
    for label, value in rows:
        print(f"  {label:<25s} {value:>10.1f}")
    print(f"\nOvernight specific cost: {costs.overnight_cost / 1000:.0f} $/kW")
    print(f"LCOE: {costs.lcoe:.1f} $/MWh")


if __name__ == "__main__":
    main()
```

Inspect the actual `Costs` dataclass fields in `src/costingfe/types.py` (or wherever `result.costs` is defined) and adjust attribute names if they differ from the placeholders above. The required output rows are fixed by the test in Step 2.

- [ ] **Step 5: Run the smoke test to verify it passes.**

```bash
cd /mnt/c/Users/talru/1cfe/1costingfe && pytest tests/test_external_physics_handoff.py -v
```

Expected: PASS. If it fails, the failure is in the example script, not in the test; fix the script.

- [ ] **Step 6: Capture the script's output for use in Task 3.**

```bash
cd /mnt/c/Users/talru/1cfe/1costingfe && python examples/external_physics_handoff.py | tee docs/papers/1costingfe_paper/scripts/_outputs/section2_rollup.txt
```

This file is consumed by Task 3 to populate the §2.2 table.

- [ ] **Step 7: Commit.**

```
git add examples/external_physics_handoff.py tests/test_external_physics_handoff.py docs/papers/1costingfe_paper/scripts/_outputs/section2_rollup.txt
git commit -m "Add canonical mirror+D-3He+DEC example for paper Section 2.2"
```

---

## Task 3: Populate §2.2 (Forward call from physics outputs)

**Files:**
- Modify: `docs/papers/1costingfe_paper/1costingfe_paper.tex` (the §2.2 placeholder area)

- [ ] **Step 1: Open `docs/papers/1costingfe_paper/scripts/_outputs/section2_rollup.txt` to read the M$ values.**

Use the values verbatim in the table below; round to one decimal place where the script does so.

- [ ] **Step 2: Replace the `% TASK 3 fills this section.` line with the §2.2 body.**

Template:

```latex
The framework's headline use case is consuming physics outputs from an
external model and producing a complete CAS-account rollup and LCOE.
This subsection walks one such handoff for a 1\,GWe D-${}^3$He steady-state
mirror with venetian-blind direct energy conversion. Numbers are
illustrative; the canonical script is
\texttt{examples/external\_physics\_handoff.py}.

The physics outputs supplied as forward arguments are summarised in
\cref{tab:tour-inputs}.

\begin{table}[htbp]
\caption{Physics outputs handed to \textsc{1costingfe} for the
Section~\ref{sec:tour} reference scenario.}
\label{tab:tour-inputs}
\centering
\begin{tabular}{lll}
\toprule
Quantity & Value & Notes \\
\midrule
Concept       & Mirror              & linear, tandem-style \\
Fuel          & D-${}^3$He          & aneutronic primary; D-D side reactions \\
$P_{\text{fus}}$ & 2400 MW          & from external physics \\
$L$           & 80 m                & central cell length \\
$r_p$         & 0.4 m               & plasma radius at midplane \\
$B_{\max}$    & 12 T                & peak field on conductor \\
$T_i$         & 70 keV              & per Section~3.1.3 example \\
$n_e$         & $3.3 \times 10^{19}$ m${}^{-3}$ & per Section~3.1.3 example \\
$\eta_{\text{th}}$ & 0.40           & thermal cycle for residual neutrons + bremsstrahlung \\
$\eta_{\text{dec}}$ & 0.70          & venetian-blind direct conversion \\
$f_{\text{dec}}$ & 0.90             & charged-particle collection fraction \\
$f_T^*$       & 0.5                 & secondary D-T burn (steady-state mirror default) \\
$f_{{}^3\text{He}}^*$ & 0.1         & secondary D-${}^3$He burn (steady-state default) \\
\bottomrule
\end{tabular}
\end{table}

The forward call is:

\begin{lstlisting}
from costingfe import ConfinementConcept, CostModel, Fuel
model  = CostModel(concept=ConfinementConcept.MIRROR, fuel=Fuel.DHE3)
result = model.forward(
    net_electric_mw=1000.0, lifetime_yr=30,
    p_fus=2400.0, chamber_length=80.0, plasma_t=0.4, b_max=12.0,
    T_e=70.0, n_e=3.3e19,
    eta_th=0.40, eta_dec=0.70, f_dec=0.90,
    f_T_secondary=0.5, f_He3_secondary=0.1,
)
\end{lstlisting}

The resulting group-level rollup is shown in \cref{tab:tour-rollup}.

\begin{table}[htbp]
\caption{CAS-account rollup for the Section~\ref{sec:tour} reference
scenario, in 2025\,USD. Reproduced by
\texttt{examples/external\_physics\_handoff.py}.}
\label{tab:tour-rollup}
\centering
\begin{tabular}{lr}
\toprule
Group & M\$ \\
\midrule
CAS10                  & VALUE \\
CAS21                  & VALUE \\
CAS22                  & VALUE \\
CAS23--26              & VALUE \\
CAS27--30              & VALUE \\
CAS40                  & VALUE \\
CAS50                  & VALUE \\
CAS60                  & VALUE \\
\midrule
Total overnight        & VALUE \\
Overnight specific cost (\$/kW) & VALUE \\
\midrule
CAS70 (M\$/yr)         & VALUE \\
CAS80 (M\$/yr)         & VALUE \\
CAS90 (M\$/yr)         & VALUE \\
\midrule
\textbf{LCOE (\$/MWh)} & \textbf{VALUE} \\
\bottomrule
\end{tabular}
\end{table}

This rollup is the canonical citable reference: other papers and
analyses can point to \cref{tab:tour-rollup} for the 1\,GWe D-${}^3$He
mirror NOAK breakdown.
```

Replace each `VALUE` with the corresponding number from `_outputs/section2_rollup.txt`.

- [ ] **Step 3: Build to verify the section renders.**

```bash
cd docs/papers/1costingfe_paper && pdflatex -interaction=nonstopmode 1costingfe_paper.tex
```

Expected: PDF builds; Section 2.2 shows the inputs table, the listing, and the rollup table with no `??` cross-reference warnings.

- [ ] **Step 4: Commit.**

```
git add docs/papers/1costingfe_paper/1costingfe_paper.tex
git commit -m "Populate paper Section 2.2 with mirror+D-3He+DEC forward-call rollup"
```

---

## Task 4: TikZ pipeline figure for §2.1

**Files:**
- Modify: `docs/papers/1costingfe_paper/1costingfe_paper.tex` (the §2.1 placeholder area)

- [ ] **Step 1: Replace the `% TASK 4 fills this section.` line with the §2.1 body.**

Template:

```latex
\textsc{1costingfe} is a thin physics layer over a thick costing
layer. Users supply whichever physics outputs they have for whichever
confinement concept; the framework backs out the rest from defaults
and sizes the cost stack accordingly. Vendor quotes or known costs
enter through CAS-level overrides without re-running the physics.

\Cref{fig:pipeline} sketches the dataflow. Inputs on the left
column are physics-output engineering parameters: geometry,
$n_e$, $T_e$ (or $T_i$), magnetic field configuration,
$\eta_{\text{th}}$, and (where applicable) $\eta_{\text{dec}}$,
$f_{\text{dec}}$, secondary burn fractions for catalyzed cycles.
Costing parameters (WACC, $T_c$, NOAK switch, conductor cost per
kA-m, ${}^3$He market price, etc.) feed the right-hand columns.
The included 0D models (\cref{sec:tokamak-model} for tokamaks) are
convenience layers for users without an external physics model in
hand and are not required.

\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
  x=1.0cm, y=1.0cm,
  box/.style={draw=lightfg, thick, rounded corners=2pt,
              minimum width=2.7cm, minimum height=1.4cm,
              align=center, font=\small},
  arrow/.style={-{Stealth}, thick, color=lightfg},
  side/.style={-{Stealth}, thick, color=blue!70!black, dashed},
]
  \node[box] (phys)   at (0,0)  {External\\physics model};
  \node[box] (mod)    at (3.6,0) {1costingfe\\physics module};
  \node[box] (eng)    at (7.2,0) {Engineering\\sizing};
  \node[box] (cas)    at (10.8,0) {CAS\\accounts};
  \node[box] (lcoe)   at (14.4,0) {LCOE\\(\$/MWh)};

  \draw[arrow] (phys) -- (mod);
  \draw[arrow] (mod)  -- (eng);
  \draw[arrow] (eng)  -- (cas);
  \draw[arrow] (cas)  -- (lcoe);

  \node[align=center, font=\scriptsize, color=lightfg]
    at (1.8,-1.6) {$P_{\text{fus}}$, geometry,\\$n_e, T_e, B,
                   \eta_{\text{th}}, \eta_{\text{dec}}$};

  \node[align=center, font=\scriptsize, color=blue!70!black]
    at (10.8,-2.4) {Vendor quotes\\(\texttt{cost\_overrides})};
  \draw[side] (10.8,-1.8) -- (cas);
\end{tikzpicture}
\caption{Pipeline from external physics outputs to LCOE. Solid path:
forward computation. Dashed path: known-cost overrides bypass the
parametric estimate at any CAS account. The included 0D models
(Appendix~A) substitute for the leftmost box when the user has no
external physics in hand.}
\label{fig:pipeline}
\end{figure}
```

The figure uses the same `lightfg` color macro defined at the top of the file (line ~31), so it inherits dark/light mode handling.

- [ ] **Step 2: Build to verify TikZ renders.**

```bash
cd docs/papers/1costingfe_paper && pdflatex -interaction=nonstopmode 1costingfe_paper.tex
```

Expected: PDF builds; Section 2.1 shows the five-box pipeline figure with the dashed override arrow.

- [ ] **Step 3: Commit.**

```
git add docs/papers/1costingfe_paper/1costingfe_paper.tex
git commit -m "Add Section 2.1 pipeline figure (TikZ)"
```

---

## Task 5: Write `scripts/make_tornado.py`

**Files:**
- Create: `docs/papers/1costingfe_paper/scripts/make_tornado.py`
- Create: `docs/papers/1costingfe_paper/figures/tornado.pdf` (output of the script)
- Create: `docs/papers/1costingfe_paper/scripts/_outputs/tornado_table.txt` (autodiff vs FD sidebar)

The existing `examples/tornado_plot.py` shows the API surface for `model.sensitivity()`; reuse the category split (engineering / financial / costing) which maps cleanly to the spec's three buckets (physics outputs / cost unit prices / financial-methodology).

- [ ] **Step 1: Write the script.**

Create `docs/papers/1costingfe_paper/scripts/make_tornado.py`:

```python
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

# Three-bucket assignment. Members of "engineering" returned by
# model.sensitivity() that originate in physics outputs go to bucket 1; the
# remaining engineering parameters (heating efficiencies, etc.) also go to
# bucket 1 since they are conversion-physics, not procurement.
_BUCKET = {
    "physics": "engineering",
    "costing": "costing",
    "financial": "financial",
}
_COLORS = {
    "physics":   "#1f77b4",
    "costing":   "#d62728",
    "financial": "#2ca02c",
}
_LABELS = {
    "physics":   "Physics outputs",
    "costing":   "Cost unit prices",
    "financial": "Financial / methodology",
}

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR.parent / "figures"
TABLE_OUT = OUT_DIR / "_outputs" / "tornado_table.txt"


def base_model_and_result():
    model = CostModel(concept=ConfinementConcept.MIRROR, fuel=Fuel.DHE3)
    result = model.forward(
        net_electric_mw=1000.0, lifetime_yr=30,
        p_fus=2400.0, chamber_length=80.0, plasma_t=0.4, b_max=12.0,
        T_e=70.0, n_e=3.3e19,
        eta_th=0.40, eta_dec=0.70, f_dec=0.90,
        f_T_secondary=0.5, f_He3_secondary=0.1,
        use_0d_model=False,
    )
    return model, result


def collect_elasticities(model, base):
    sens = model.sensitivity(base.params)
    rows = []  # (param, elasticity, bucket)
    for bucket, category in _BUCKET.items():
        for param, e in sens[category].items():
            if abs(e) > 1e-4:
                rows.append((param, float(e), bucket))
    rows.sort(key=lambda x: abs(x[1]), reverse=True)
    return rows


def render_tornado(rows, out_path: Path):
    rows = rows[:18]  # keep the top 18 entries to fit on one page
    rows = list(reversed(rows))  # largest at top in matplotlib horizontal bar
    fig, ax = plt.subplots(figsize=(7.0, 0.32 * len(rows) + 1.2))
    for i, (param, e, bucket) in enumerate(rows):
        ax.barh(i, e, color=_COLORS[bucket])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([param for param, _, _ in rows])
    ax.set_xlabel("LCOE elasticity (% change in LCOE per 1% change in parameter)")
    ax.axvline(0, color="black", lw=0.5)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_COLORS[k]) for k in _BUCKET
    ]
    ax.legend(handles, [_LABELS[k] for k in _BUCKET], loc="lower right",
              frameon=False, fontsize="small")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)


def write_fd_table(model, base, top_rows, out_path: Path, k: int = 5):
    """Centered finite differences vs autodiff for the top-k elasticities."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["param            autodiff       FD             rel_err",
             "----------------------------------------------------------"]
    for param, e_auto, _ in top_rows[:k]:
        e_fd = _centered_fd_elasticity(model, base, param)
        rel = abs(e_auto - e_fd) / max(abs(e_auto), 1e-12)
        lines.append(f"{param:<16s} {e_auto:+.6f}  {e_fd:+.6f}  {rel:.2e}")
    out_path.write_text("\n".join(lines) + "\n")


def _centered_fd_elasticity(model, base, param: str, h: float = 1e-3):
    """Centered finite-difference elasticity at the base point, using a 0.1%
    perturbation of the parameter. Reads/writes through model.params via
    keyword override (use whatever the model's actual override surface is)."""
    p0 = base.params[param]
    p_lo = p0 * (1 - h)
    p_hi = p0 * (1 + h)
    lcoe_lo = model.forward(**{**base.kwargs, param: p_lo}).costs.lcoe
    lcoe_hi = model.forward(**{**base.kwargs, param: p_hi}).costs.lcoe
    return (lcoe_hi - lcoe_lo) / (p_hi - p_lo) * (p0 / base.costs.lcoe)


def main() -> None:
    model, base = base_model_and_result()
    rows = collect_elasticities(model, base)
    render_tornado(rows, FIG_DIR / "tornado.pdf")
    write_fd_table(model, base, rows, TABLE_OUT)
    print(f"Wrote {FIG_DIR / 'tornado.pdf'}")
    print(f"Wrote {TABLE_OUT}")


if __name__ == "__main__":
    main()
```

The `_centered_fd_elasticity` helper assumes `base.params` is a dict and that `base.kwargs` exists; if the actual API differs, adapt to whatever lets you forward-call with one parameter overridden. Look at `examples/parameter_sweeps.py` for the established pattern.

- [ ] **Step 2: Run the script.**

```bash
cd /mnt/c/Users/talru/1cfe/1costingfe && python docs/papers/1costingfe_paper/scripts/make_tornado.py
```

Expected: prints two `Wrote ...` lines. `figures/tornado.pdf` exists (non-empty); `scripts/_outputs/tornado_table.txt` exists.

- [ ] **Step 3: Eyeball the FD table.**

```bash
cat /mnt/c/Users/talru/1cfe/1costingfe/docs/papers/1costingfe_paper/scripts/_outputs/tornado_table.txt
```

Expected: `rel_err` column is small (< 1e-3) for all 5 rows. If any row diverges, the parameter has a discontinuous response (e.g. a `where` clamp); investigate and either pick a base point that avoids the clamp or annotate the divergence in §2.3.

- [ ] **Step 4: Commit.**

```
git add docs/papers/1costingfe_paper/scripts/make_tornado.py docs/papers/1costingfe_paper/figures/tornado.pdf docs/papers/1costingfe_paper/scripts/_outputs/tornado_table.txt
git commit -m "Add tornado script and figure for paper Section 2.3"
```

---

## Task 6: Populate §2.3 (Sensitivity tornado)

**Files:**
- Modify: `docs/papers/1costingfe_paper/1costingfe_paper.tex` (the §2.3 placeholder area)

- [ ] **Step 1: Read the autodiff-vs-FD table.**

```bash
cat /mnt/c/Users/talru/1cfe/1costingfe/docs/papers/1costingfe_paper/scripts/_outputs/tornado_table.txt
```

Use the values verbatim in the §2.3 sidebar table.

- [ ] **Step 2: Replace the `% TASK 6 fills this section.` line with the §2.3 body.**

Template:

```latex
The forward pipeline of \cref{fig:pipeline} is differentiable end-to-end.
JAX reverse-mode automatic differentiation produces exact partial
derivatives of LCOE with respect to every input parameter in a single
backward pass. \Cref{fig:tornado} shows the resulting tornado for the
Section~\ref{sec:tour} reference scenario, organised into three bands:
physics outputs (the parameters supplied by the external physics
model), cost unit prices (procurement-grounded \$/unit values that
multiply physical quantities), and financial / methodology (cost-of-money,
construction time, and contingency assumptions).

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\linewidth]{figures/tornado.pdf}
\caption{Sensitivity tornado for the canonical 1\,GWe D-${}^3$He
mirror reference. Bars give LCOE elasticity
$\epsilon_p = (\partial \text{LCOE}/\partial p)(p/\text{LCOE})$
computed by JAX reverse-mode autodiff on the full forward pipeline.
A bar of length $0.10$ means a $+1\%$ perturbation of the parameter
moves LCOE by $+0.10\%$. Reproduced by
\texttt{scripts/make\_tornado.py}.}
\label{fig:tornado}
\end{figure}

The tornado is the headline operational output of the framework: a
single figure tells a physics team where simulation effort buys the
most LCOE certainty, and a procurement team where vendor data does
the same. For the reference scenario, the most consequential
physics-output parameter is \textbf{TOP\_PHYSICS}, the most
consequential cost unit price is \textbf{TOP\_COSTING}, and the most
consequential financial parameter is \textbf{TOP\_FINANCIAL}.
[Replace the three placeholders with the actual top entries from the
generated figure.]

Because the gradients are exact rather than approximate, autodiff and
finite differences agree to machine precision at any non-singular
point. \Cref{tab:tornado-fd} reports a centered finite-difference
spot-check at the base point for the top five entries.

\begin{table}[htbp]
\caption{Autodiff elasticities versus centered finite differences for
the top five entries of \cref{fig:tornado}. Reproduced by
\texttt{scripts/make\_tornado.py}.}
\label{tab:tornado-fd}
\centering
\begin{tabular}{lrrr}
\toprule
Parameter & Autodiff & FD & Relative error \\
\midrule
PARAM1 & VALUE & VALUE & VALUE \\
PARAM2 & VALUE & VALUE & VALUE \\
PARAM3 & VALUE & VALUE & VALUE \\
PARAM4 & VALUE & VALUE & VALUE \\
PARAM5 & VALUE & VALUE & VALUE \\
\bottomrule
\end{tabular}
\end{table}
```

Fill in the `TOP_PHYSICS / TOP_COSTING / TOP_FINANCIAL` placeholders by reading the figure (or by adding a small print to `make_tornado.py` that emits the top per bucket). Fill in the `PARAMn / VALUE` entries from `_outputs/tornado_table.txt`.

- [ ] **Step 3: Build to verify figure include and cross-references.**

```bash
cd docs/papers/1costingfe_paper && pdflatex -interaction=nonstopmode 1costingfe_paper.tex
```

Expected: PDF builds; tornado figure renders; FD table renders; `\cref{fig:tornado}` and `\cref{tab:tornado-fd}` resolve.

- [ ] **Step 4: Commit.**

```
git add docs/papers/1costingfe_paper/1costingfe_paper.tex
git commit -m "Populate paper Section 2.3 with sensitivity tornado and FD spot-check"
```

---

## Task 7: §2.4 (Other API surfaces)

**Files:**
- Modify: `docs/papers/1costingfe_paper/1costingfe_paper.tex` (the §2.4 placeholder area)

- [ ] **Step 1: Replace the `% TASK 7 fills this section.` line with the §2.4 body.**

Template:

```latex
The forward call of \cref{sec:tour-forward} is the headline entry
point. Three other API surfaces address common variants on the
underlying question.

\paragraph{Cost overrides.} A vendor quote may replace a parametric
estimate without re-running the physics:

\begin{lstlisting}
result = model.forward(
    net_electric_mw=1000.0, lifetime_yr=30,
    cost_overrides={"C220103": 80.0},  # M$, mirror coil quote
)
\end{lstlisting}

Downstream rollups (CAS22, total capital, LCOE) recompute
automatically. Mirror coils are typically smaller absolute cost than
tokamak TF/CS/PF systems, reflecting the simpler solenoid topology.

\paragraph{Batch sweeps.} Vectorised over JAX \texttt{vmap},
suitable for uncertainty-band propagation and Monte Carlo:

\begin{lstlisting}
lcoes = model.batch_lcoe(
    {"f_T_secondary": [0.3, 0.4, 0.5, 0.6, 0.7]},
    base_params=result.params,
)
\end{lstlisting}

The swept parameter here is the secondary D-T burn fraction, which
controls how much of the D-${}^3$He plant's neutron load comes from
D-D-bred tritium.

\paragraph{Backcasting.} Solves the inverse problem: which value of a
single parameter hits a target LCOE, given everything else fixed.

\begin{lstlisting}
from costingfe.analysis.backcast import backcast_single
eta_dec_target = backcast_single(
    model, target_lcoe=60.0, param_name="eta_dec",
    param_range=(0.50, 0.85), base_params=result.params,
)
\end{lstlisting}

The question above is what venetian-blind efficiency would bring the
mirror reference plant to a 60\,\$/MWh LCOE.
```

Adjust `{"C220103": 80.0}` to a value that's plausible relative to the §2.2 rollup's CAS22 line (mirror coils are part of CAS22.01.03; the parametric estimate from the rollup is the right ballpark, set the override 10-20% lower than that to motivate "vendor came in below").

- [ ] **Step 2: Build.**

```bash
cd docs/papers/1costingfe_paper && pdflatex -interaction=nonstopmode 1costingfe_paper.tex
```

Expected: PDF builds; §2.4 has three labeled paragraphs and three short listings; cross-reference `\cref{sec:tour-forward}` resolves.

- [ ] **Step 3: Commit.**

```
git add docs/papers/1costingfe_paper/1costingfe_paper.tex
git commit -m "Populate paper Section 2.4 with override, sweep, and backcast listings"
```

---

## Task 8: Move pulsed and inverse balances to Appendix C

**Files:**
- Modify: `docs/papers/1costingfe_paper/1costingfe_paper.tex` (sections 3.4, 3.5, and the appendix area)

- [ ] **Step 1: Find the boundaries of the content to move.**

```bash
grep -n "^\\\\subsection\|^\\\\appendix\|^\\\\section" /mnt/c/Users/talru/1cfe/1costingfe/docs/papers/1costingfe_paper/1costingfe_paper.tex | head -30
```

The relevant regions (line numbers will shift as Section 2 grows; locate by content, not number):
- `\subsection{Pulsed Power Balance}` (`\label{sec:pulsed-balance}`) and its three `\subsubsection` children.
- `\subsection{Inverse Power Balance}` (`\label{sec:inverse-balance}`) and its two `\subsubsection` children.

Both are inside `\section{Physics Module}`.

- [ ] **Step 2: Cut the two subsections from the Physics Module section.**

Remove lines beginning at `\subsection{Pulsed Power Balance}` through the end of `\subsubsection{Steady-state inverse.}` (the entire block).

- [ ] **Step 3: Cut the forward-looking thermodynamics paragraph.**

In the Physics Module, locate and delete the paragraph beginning "Lumped efficiency parameters are used for characterizing the power flow and thermal cycle. A more involved power plant thermodynamics model could be implemented in the future". Per the spec, this ambient text is removed for parsimony.

- [ ] **Step 4: Insert the new Appendix C after Appendix B (Synchrotron model).**

Locate `\section{Synchrotron radiation model details}` and its trailing subsections; insert immediately after the last line of that appendix:

```latex
%% ============================================================
\section{Pulsed and inverse power balances}
\label{sec:pulsed-and-inverse-balance}

\subsection{Pulsed power balance}
\label{sec:pulsed-balance}

% paste the three subsubsections originally under "Pulsed Power Balance"
% (Per-pulse energy framework, Pulsed thermal conversion, Pulsed
% inductive DEC conversion) verbatim.

\subsection{Inverse power balance}
\label{sec:inverse-balance}

% paste the two subsubsections originally under "Inverse Power Balance"
% (Pulsed inverse, Steady-state inverse) verbatim.
```

The labels `sec:pulsed-balance` and `sec:inverse-balance` are preserved so existing `\cref{}` calls continue to resolve.

- [ ] **Step 5: Demote the headings.**

In the moved content, the previous `\subsection{...}` becomes the appendix's subsection (still `\subsection`), and previous `\subsubsection{...}` stays `\subsubsection`. No demotion needed -- they were already at `\subsection` / `\subsubsection` inside `\section{Physics Module}`.

- [ ] **Step 6: Add a forward-pointer in the Physics Module narrative.**

At the end of the steady-state physics narrative (just before the start of Cost Account Structure), insert a one-sentence pointer:

```latex
Pulsed concepts (laser IFE, Z-pinch, inductive recovery) and the
inverse-balance solver used by the costing model are presented in
\cref{sec:pulsed-and-inverse-balance}.
```

- [ ] **Step 7: Build and verify cross-refs.**

```bash
cd docs/papers/1costingfe_paper && pdflatex -interaction=nonstopmode 1costingfe_paper.tex && pdflatex -interaction=nonstopmode 1costingfe_paper.tex
```

Two passes are needed for `\cref{}` to settle. Expected: no `??` warnings; Appendix C appears after Appendix B; Physics Module is shorter and ends with the forward-pointer; CAS22.01.07 capacitor-bank reference (which currently points to the old `sec:cas2207` subsection) and any other in-body `\cref{sec:pulsed-balance}` / `\cref{sec:inverse-balance}` calls still resolve.

- [ ] **Step 8: Commit.**

```
git add docs/papers/1costingfe_paper/1costingfe_paper.tex
git commit -m "Move pulsed and inverse balance to new Appendix C"
```

---

## Task 9: CAS overview table -- add Disposition column

**Files:**
- Modify: `docs/papers/1costingfe_paper/1costingfe_paper.tex` (the `tab:cas-overview` table)

- [ ] **Step 1: Locate the current table.**

Search for `\label{tab:cas-overview}` (currently around line 1130).

- [ ] **Step 2: Replace the column spec and rebuild every row.**

Change `\begin{tabular}{llp{7.0cm}}` to `\begin{tabular}{llcp{6.0cm}}` (adds a centered Disposition column).

Update the header row from
```
CAS & Description & Method \\
```
to
```
CAS & Description & Disposition & Method \\
```

For every body row, insert the Disposition value as the third column. Use the disposition-table mapping in the spec (`docs/plans/2026-05-07-paper-restructure-design.md`, Section 5). Disposition values are exactly one of: `Inherited`, `Extended`, `Replaced`. Examples:

```
10$^\dagger$ & Pre-construction & Replaced & Land intensity scaling, fuel-dependent licensing \\
21$^\dagger$ & Buildings \& structures & Replaced & Per-building, per-fuel, industrial-grade (18 buildings) \\
\hspace{1em}22.01.01 & First wall + blanket & Inherited & Volume $\times$ thermal intensity, fuel-dependent \\
\hspace{1em}22.01.03$^\dagger$ & Coils (magnets) & Replaced & Conductor kAm $\times$ \$/kAm $\times$ markup; \$0 for IFE \\
\hspace{1em}22.01.04$^\dagger$ & Heating (MFE) / driver (pulsed) & Extended & Per-MW linear; concept-specific driver capital for pulsed \\
\hspace{1em}22.01.07$^\dagger$ & Power supplies & Replaced & Power-scaled (MFE); \$/J stored basis for all pulsed \\
70$^\dagger$ & Annualized O\&M + replacement & Replaced & Growing-annuity levelization \\
```

Walk every existing row; add Disposition exactly once. Do not change the rows themselves, only insert one cell.

- [ ] **Step 3: Add a one-sentence legend just above the table.**

```latex
The \emph{Disposition} column flags whether the account uses prior
conventions unchanged (\emph{Inherited}), keeps the prior backbone
with additional sub-cases (\emph{Extended}), or has been re-derived
from procurement and first principles (\emph{Replaced}). Subsections
in this chapter present methodology only for accounts in the latter
two categories.
```

- [ ] **Step 4: Build.**

```bash
cd docs/papers/1costingfe_paper && pdflatex -interaction=nonstopmode 1costingfe_paper.tex
```

Expected: table fits on the page; Disposition column is populated for every row; legend appears immediately above.

- [ ] **Step 5: Commit.**

```
git add docs/papers/1costingfe_paper/1costingfe_paper.tex
git commit -m "Add Disposition column to CAS overview table"
```

---

## Task 10: Compress inherited CAS22 subsections

**Files:**
- Modify: `docs/papers/1costingfe_paper/1costingfe_paper.tex` (the CAS22 subsection area)

CAS22.01.01, .02, .05, .06, .08, .11 are all `Inherited` per the spec. The current paper does not have prose subsections for these (it skips from .03 to .04 to .07 etc.), so this task is mostly about the cluster paragraph and a one-sentence stub explaining why the inherited accounts are absent.

- [ ] **Step 1: After the CAS22 overview text and before the first `Replaced` subsection (CAS22.01.03), add a stub paragraph.**

Locate the existing `\subsection{CAS22.01.03: Superconducting Coils}` and insert immediately above it:

```latex
\subsection{Inherited sub-accounts}
\label{sec:cas22-inherited}

Sub-accounts CAS22.01.01 (first wall + blanket), 22.01.02 (shield),
22.01.05 (primary structure), 22.01.06 (vacuum system), 22.01.08
(divertor), and 22.01.11 (installation labor) inherit the methodology
of the pyFECONS implementation \citep{woodruff2026} unchanged: hybrid
volume + thermal-intensity scaling for component accounts and
percentage-of-subtotal for installation labor. They are not
re-derived in this paper.
```

- [ ] **Step 2: Replace the existing `CAS22.02--.07` subsection (or insert if absent) with a single short cluster paragraph.**

Locate `\subsection{CAS22.02--.07}` if present; otherwise insert before the start of CAS23 discussion. The subsection should be:

```latex
\subsection{CAS22.02--.07: Plant-wide reactor systems}
\label{sec:cas2207-cluster}

The plant-wide reactor systems CAS22.02 (main and secondary coolant),
CAS22.03 (auxiliary cooling and cryoplant), CAS22.04 (radioactive
waste management), CAS22.05 (fuel handling and storage), CAS22.06
(other reactor plant equipment), and CAS22.07 (instrumentation and
control) inherit pyFECONS power-scaling laws with exponents in the
range 0.65--0.85 and pyFECONS reference-plant unit costs. CAS22.05
is fuel-dependent (D-T tritium accounting versus aneutronic fuel
handling); the others are concept- and fuel-agnostic.
```

- [ ] **Step 3: Build.**

```bash
cd docs/papers/1costingfe_paper && pdflatex -interaction=nonstopmode 1costingfe_paper.tex
```

Expected: PDF builds; new short subsections appear; CAS chapter is shorter; the Replaced subsections that follow (CAS22.01.03, .04, etc.) are unchanged.

- [ ] **Step 4: Commit.**

```
git add docs/papers/1costingfe_paper/1costingfe_paper.tex
git commit -m "Add inherited and plant-wide-cluster CAS22 stubs"
```

---

## Task 11: CAS28 + CAS22.01.07 sourcing annotations

**Files:**
- Modify: `docs/papers/1costingfe_paper/1costingfe_paper.tex` (the CAS28 subsection and the CAS22.01.07 subsection)

Project rule: `pyFECONS` is the least-trusted source; vendor systems should use procurement data, not material build-ups or pyFECONS internals.

- [ ] **Step 1: Append the CAS28 sourcing flag.**

Locate `\subsection{CAS28: Digital Twin}` (around line 1779). At the end of the subsection, before the next `\subsection`, append:

```latex
The \$5\,M figure is sourced from a single pyFECONS internal note;
no independent benchmark has yet been adopted. CAS28 is therefore
the lowest-information account in the framework and the figure is
treated as a placeholder with wide uncertainty. Closing this gap
requires sourcing one or more independent quotes for plant digital
twin integration; users wishing to vary the figure should override
\texttt{cost\_overrides=\{"CAS28": <value>\}} until that work
completes.
```

- [ ] **Step 2: Append the CAS22.01.07 vendor-source flag.**

Locate `\subsection{CAS22.01.07: Power Supplies}` (around line 1448). At the end of the subsection (after the `Pulsed concepts.` subsubsection), append:

```latex
\subsubsection{Source provenance.}
The steady-state MFE figure draws partly on the ARIES-CS
\citep{najmabadi2008stellarator} power-supply scope, which is a
systems study rather than a procurement reference. The capacitor-bank
\$/J basis used for pulsed concepts is procurement-grounded
(commercial high-voltage capacitor pricing). Substituting a
procurement reference for the steady-state MFE figure is left to
follow-on work; users with a vendor quote should override
\texttt{cost\_overrides=\{"C220107": <value>\}} until that
substitution lands.
```

- [ ] **Step 3: Build.**

```bash
cd docs/papers/1costingfe_paper && pdflatex -interaction=nonstopmode 1costingfe_paper.tex
```

Expected: PDF builds; both subsections gain the new annotations; no `??` cross-reference warnings.

- [ ] **Step 4: Commit.**

```
git add docs/papers/1costingfe_paper/1costingfe_paper.tex
git commit -m "Annotate CAS28 and CAS22.01.07 sourcing per project rules"
```

---

## Task 12: Finish three benchmarking TODO discussion paragraphs

**Files:**
- Modify: `docs/papers/1costingfe_paper/1costingfe_paper.tex` (the three `\textbf{TODO: ...}` blocks in §6)

The drafts already in the body lay out the substance. The editing task is to lift the markers, tighten the prose, and make sure no "earlier draft said X" framing appears.

- [ ] **Step 1: Locate the three blocks.**

```bash
grep -n "TODO" /mnt/c/Users/talru/1cfe/1costingfe/docs/papers/1costingfe_paper/1costingfe_paper.tex
```

Expected: three `% TODO` or `\textbf{TODO:` entries in the Benchmarking section (currently around lines 2320, 2394, 2421).

- [ ] **Step 2: For each block, remove the `\textbf{TODO: ...}` wrapper and replace with the cleaned prose.**

For block 1 (ARC discussion, currently around line 2320), the existing draft is reasonable; just remove the `\textbf{TODO: discussion paragraph -- ` opener and the closing `}`, and tighten any awkward phrasing.

For block 2 (ARIES-AT discussion, currently around line 2394), same treatment: remove `\textbf{TODO: ARIES-AT discussion paragraph -- ` and the closing `}`.

For block 3 (LCOE composition discussion, currently around line 2421), same treatment.

The result should read as finished prose. No `TODO` strings remain in the body.

- [ ] **Step 3: Build and read the rendered Benchmarking section in the PDF.**

```bash
cd docs/papers/1costingfe_paper && pdflatex -interaction=nonstopmode 1costingfe_paper.tex
```

Expected: PDF builds; Section 6 reads as finished prose; no TODO markers remain.

```bash
grep -n "TODO" /mnt/c/Users/talru/1cfe/1costingfe/docs/papers/1costingfe_paper/1costingfe_paper.tex
```

Expected output: empty (no matches).

- [ ] **Step 4: Update `todo.md`.**

Open `docs/papers/1costingfe_paper/todo.md` and remove the "Three `% TODO` discussion paragraphs..." note from the Validation-against-prior-tools section, since those paragraphs are now finished. Mark the gradient-demo and pipeline-figure items as resolved.

- [ ] **Step 5: Commit.**

```
git add docs/papers/1costingfe_paper/1costingfe_paper.tex docs/papers/1costingfe_paper/todo.md
git commit -m "Finalise three benchmarking discussion paragraphs"
```

---

## Task 13: Abstract, Section 3 (Economics), conclusion, introduction edits

**Files:**
- Modify: `docs/papers/1costingfe_paper/1costingfe_paper.tex` (abstract, economics intro, conclusion, introduction)

- [ ] **Step 1: Abstract -- add the worked-tour sentence.**

Locate the abstract (between `\begin{abstract}` and `\end{abstract}`, currently lines 100-133). Find the sentence ending "exposing exact LCOE gradients to support sensitivity analysis, and Monte Carlo uncertainty propagation." Immediately after that sentence, before "The economics, physics, and cost-account methodology are described...", insert:

```latex
A worked tour from physics outputs to LCOE, including JAX-derived
sensitivities across physics, cost, and financial parameters, is
presented as the canonical use case (\cref{sec:tour}).
```

Per the spec, the existing coverage claim ("It spans the several confinement families...") is left in place. No other abstract edits.

- [ ] **Step 2: Section 3 (Economics) -- add the NOAK / financial-knob anchor sentence.**

Locate `\section{Economics Module}` (currently around line 259). Immediately after the opening paragraph that introduces LCOE, before `\subsection{Capital Recovery Factor}`, insert:

```latex
Defaults throughout this section are NOAK; the financial knobs
$(i, T_c, n)$ that appear among the dominant entries of the
Section~\ref{sec:tour-sensitivity} sensitivity tornado are defined
in the subsections that follow, with reference values stated
inline.
```

This is the spec's "one-sentence anchor"; do not expand further. No structural changes to the Economics section.

- [ ] **Step 3: Conclusion -- remove the "gradients not demonstrated" sentence.**

Locate the conclusion (currently around line 2437). Find and delete the sentence:

> Gradient-enabled use cases (sensitivity tornados, target-driven inverse design, autodiff-versus-finite-difference comparisons) are supported by the framework but are not demonstrated in the present text.

Replace with nothing; the surrounding text closes cleanly.

- [ ] **Step 4: Introduction -- add the Section 2 anchor paragraph.**

After the existing roadmap paragraph (`The remainder of this paper is organized as follows...`, currently around lines 250-256), insert a new short paragraph:

```latex
The worked tour in \cref{sec:tour} is the headline operational
example: a single concept handed off from an external physics model
into the costing pipeline, with a sensitivity tornado over the
combined parameter set. Concept-agnosticism is exercised by running
the tour on a non-tokamak case (a D-${}^3$He mirror with
venetian-blind direct energy conversion); the included 0D models in
the appendix are convenience layers and are not required.
```

- [ ] **Step 5: Build (two passes for `\cref{}` to settle).**

```bash
cd docs/papers/1costingfe_paper && pdflatex -interaction=nonstopmode 1costingfe_paper.tex && pdflatex -interaction=nonstopmode 1costingfe_paper.tex
```

Expected: PDF builds; abstract has the new tour sentence; Economics has the NOAK anchor sentence; conclusion does not mention "not demonstrated"; introduction has the new anchor paragraph; cross-refs to `sec:tour` and `sec:tour-sensitivity` resolve.

- [ ] **Step 6: Commit.**

```
git add docs/papers/1costingfe_paper/1costingfe_paper.tex
git commit -m "Abstract, Economics, conclusion, intro edits for worked-tour Section 2"
```

---

## Task 14: Code Availability extensions

**Files:**
- Modify: `docs/papers/1costingfe_paper/1costingfe_paper.tex` (Code Availability section)

- [ ] **Step 1: Locate the Code Availability section.**

Around line 2472. The current text already mentions the three benchmarking scripts. Append two pointers.

- [ ] **Step 2: Append after the existing pointers.**

```latex
The Section~\ref{sec:tour} worked tour is reproduced by
\texttt{examples/external\_physics\_handoff.py} (the rollup table)
and \texttt{docs/papers/1costingfe\_paper/scripts/make\_tornado.py}
(the sensitivity tornado figure and the autodiff-vs-finite-difference
comparison table).
```

- [ ] **Step 3: Build.**

```bash
cd docs/papers/1costingfe_paper && pdflatex -interaction=nonstopmode 1costingfe_paper.tex
```

Expected: PDF builds; Code Availability lists both new artifacts.

- [ ] **Step 4: Commit.**

```
git add docs/papers/1costingfe_paper/1costingfe_paper.tex
git commit -m "Code Availability: add Section 2 worked-tour artifacts"
```

---

## Task 15: Final build + diff walk

**Files:**
- All previously modified files; sanity check only.

- [ ] **Step 1: Clean build.**

```bash
cd docs/papers/1costingfe_paper && \
  rm -f *.aux *.bbl *.blg *.log *.out *.toc && \
  pdflatex -interaction=nonstopmode 1costingfe_paper.tex && \
  bibtex 1costingfe_paper && \
  pdflatex -interaction=nonstopmode 1costingfe_paper.tex && \
  pdflatex -interaction=nonstopmode 1costingfe_paper.tex
```

Expected: PDF builds; bibliography resolves; no `??` cross-reference warnings; no overfull-hbox errors of consequence.

- [ ] **Step 2: Page-by-page sanity scan.**

Open the PDF and verify, in order:
- Abstract has the worked-tour sentence; coverage claim still present.
- Section 2 has four subsections (pipeline figure, forward call + rollup, tornado + FD table, three API listings) and renders without overflow.
- Section 3 (Economics) is unchanged.
- Section 4 (Physics) ends with the forward-pointer to Appendix C and is shorter than before.
- Section 5 (CAS) overview table has a Disposition column; Inherited stub and Cluster paragraph appear; CAS28 and CAS22.01.07 have sourcing notes.
- Section 6 (Benchmarking) has no remaining `TODO` markers; three discussion paragraphs are clean prose.
- Section 7 (Conclusion) does not contain "not demonstrated".
- Code Availability lists the new artifacts.
- Appendix A (0D Tokamak) and B (Synchrotron) unchanged.
- Appendix C (Pulsed and inverse balances) appears after B with the moved content.

- [ ] **Step 3: Pdf-level grep sanity.**

```bash
grep -ic "TODO\|FIXME\|TBD\|XXX" docs/papers/1costingfe_paper/1costingfe_paper.tex
```

Expected: 0.

- [ ] **Step 4: Run the smoke test one more time.**

```bash
cd /mnt/c/Users/talru/1cfe/1costingfe && pytest tests/test_external_physics_handoff.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit if anything changed.**

```
git status -s
```

If any files are modified beyond build artifacts: `git add` and commit with a one-line message describing what was caught in the diff walk. Build artifacts (`.aux`, `.bbl`, etc.) should not be committed.

---

## Out of scope (per spec)

- Stub 0D physics models for stellarator, FRC, IFE.
- Per-CAS-account ARIES-AT cross-walk against pyFECONS.
- Procurement substitute for CAS22.01.07 (annotated, not closed).
- Independent benchmark for CAS28 (annotated, not closed).
- Splitting into a methods + companion paper.
- 0D mirror physics layer (in flight in `docs/plans/mirror_physics_model.md`); this restructure does not block on it. If the layer merges before this restructure does, Task 2's `use_0d_model=False` branch can be dropped.
