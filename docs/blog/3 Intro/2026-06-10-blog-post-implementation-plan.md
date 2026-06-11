# Blog Post #3 (1costingFE Intro) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A publishable ~1,500-word blog post introducing 1costingFE: what it does, how to try it live, how to clone and run it, with every number and snippet verified at tag `v0.1.0-alpha.1`.

**Architecture:** Rewrite from scratch (the May 27 draft is superseded; mine it for facts/links only). All numbers regenerate from the freeze worktree at `/mnt/c/Users/talru/1cfe/1costingfe-freeze` (created by the explorer plan, Task 1). Output lives in the untracked `docs/blog/3 Intro/` folder.

**Tech Stack:** Markdown, matplotlib (tornado figure), fresh-clone snippet verification.

**Hard constraint:** Never switch branches, modify tracked files, or commit in `/mnt/c/Users/talru/1cfe/1costingfe`. All writes go to `docs/blog/3 Intro/` (untracked) or `/tmp`.

**Dependencies on the explorer plan:** Task 1 (freeze tag + worktree) must be done first. Task 6 here (explorer link + screenshot) needs the Vercel deployment; everything else can proceed in parallel.

**Style rules (user preferences, mandatory):** bare `$` in markdown (no escaping); no em dashes (use commas, parentheses, or dashes); no tildes for approximate values (write "about 23%"); never source a cost VALUE from pyFECONS (account-structure and landscape mentions are fine); don't call pulsed concepts "FRC-style".

---

### Task 1: Regenerate canonical numbers at the freeze

**Files:**
- Create: `docs/blog/3 Intro/freeze_outputs/dt_tokamak_output.txt`
- Create: `docs/blog/3 Intro/freeze_outputs/tornado_output.txt`

- [ ] **Step 1: Run the canonical examples in the freeze worktree**

```bash
cd /mnt/c/Users/talru/1cfe/1costingfe-freeze
mkdir -p "/mnt/c/Users/talru/1cfe/1costingfe/docs/blog/3 Intro/freeze_outputs"
uv run python examples/dt_tokamak.py > "/mnt/c/Users/talru/1cfe/1costingfe/docs/blog/3 Intro/freeze_outputs/dt_tokamak_output.txt" 2>&1
uv run python examples/tornado_plot.py > "/mnt/c/Users/talru/1cfe/1costingfe/docs/blog/3 Intro/freeze_outputs/tornado_output.txt" 2>&1
```

Expected: both scripts run clean. These two files are the ONLY source for any LCOE value or elasticity quoted in the post (per the no-inline-python rule, companion scripts are the source of truth).

- [ ] **Step 2: Record headline numbers**

Read the two output files and note: baseline D-T tokamak LCOE, overnight capex, and the top 8 elasticities (engineering + financial). These replace the old draft's "illustrative" table, which was flagged as unverified.

### Task 2: Tornado figure

**Files:**
- Create: `docs/blog/3 Intro/blog3_tornado_figure.py`
- Create: `docs/blog/3 Intro/tornado_dt_tokamak.png` (generated)

- [ ] **Step 1: Write the figure script**

`docs/blog/3 Intro/blog3_tornado_figure.py`:

```python
"""Tornado figure for blog post #3. Run from the freeze worktree:

    cd /mnt/c/Users/talru/1cfe/1costingfe-freeze
    uv run python "/mnt/c/Users/talru/1cfe/1costingfe/docs/blog/3 Intro/blog3_tornado_figure.py"
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
    "b_center": "Peak field on coil",
    "r_bore": "Coil winding radius",
    "net_electric_mw": "Net electric power",
    "inflation_rate": "Inflation rate",
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
```

- [ ] **Step 2: Generate and inspect**

```bash
cd /mnt/c/Users/talru/1cfe/1costingfe-freeze
uv run python "/mnt/c/Users/talru/1cfe/1costingfe/docs/blog/3 Intro/blog3_tornado_figure.py"
```

Expected: PNG written; printed elasticities consistent with `freeze_outputs/tornado_output.txt`. View the PNG (Read tool) — labels legible, availability and WACC should dominate.

### Task 3: Verify the post's code snippets from a fresh clone

**Files:**
- Create: `/tmp/blog3_verify/snippet_forward.py`, `/tmp/blog3_verify/snippet_grad.py`

- [ ] **Step 1: Fresh clone at the tag, fresh venv**

```bash
rm -rf /tmp/blog3_verify && mkdir -p /tmp/blog3_verify && cd /tmp/blog3_verify
git clone --depth 1 --branch v0.1.0-alpha.1 /mnt/c/Users/talru/1cfe/1costingfe repo
cd repo && python3 -m venv .venv && .venv/bin/pip install -e . -q
```

(Local clone stands in for the GitHub clone; same tag, same install path a reader follows.)

- [ ] **Step 2: Write the exact snippets the post will print**

`/tmp/blog3_verify/snippet_forward.py` (this IS the post's main snippet — if it needs changes to run, change the post, not just the file):

```python
from costingfe import CostModel, ConfinementConcept, Fuel

model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
result = model.forward(
    net_electric_mw=1000.0,
    availability=0.85,
    lifetime_yr=30,
)

print(f"LCOE: ${float(result.costs.lcoe):.1f}/MWh")
print(f"Overnight capital: ${float(result.costs.overnight_cost) / 1000:.2f}B")
for code, value in sorted(result.cas22_detail.items(), key=lambda kv: -kv[1])[:5]:
    print(f"  {code}: ${float(value):.0f}M")
```

`/tmp/blog3_verify/snippet_grad.py`:

```python
from costingfe import CostModel, ConfinementConcept, Fuel

model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
result = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)

s = model.sensitivity(result.params)
for key, e in sorted(
    {**s["engineering"], **s["financial"]}.items(), key=lambda kv: -abs(kv[1])
)[:8]:
    print(f"{key:24s} {e:+.2f}")
```

- [ ] **Step 3: Run both, capture output**

```bash
cd /tmp/blog3_verify/repo
.venv/bin/python ../snippet_forward.py | tee "/mnt/c/Users/talru/1cfe/1costingfe/docs/blog/3 Intro/freeze_outputs/snippet_forward_output.txt"
.venv/bin/python ../snippet_grad.py | tee "/mnt/c/Users/talru/1cfe/1costingfe/docs/blog/3 Intro/freeze_outputs/snippet_grad_output.txt"
```

Expected: both run clean on a bare `pip install -e .` (jax, pyyaml, pydantic pulled in automatically). The captured outputs are pasted verbatim into the post as the "what you'll see" blocks. If either snippet errors, fix the snippet (not the library) and re-run.

### Task 4: Resolve the carried-over fact-check items

The old draft's header flagged these. Resolve each; record answers in a comment block at the top of the new post until final review removes it.

- [ ] **Step 1: Verify the Schulte 1978 PNL-2648 OSTI link** (search OSTI for "PNL-2648 Fusion Reactor Design Studies Standard Accounts"; the old draft's guess was https://www.osti.gov/biblio/6395456 — confirm or replace)
- [ ] **Step 2: Confirm published URLs of the three prior posts** on 1cf.energy: cost-floor post, DEC post, and the fusion-tea pipeline post ("From Papers to Plant Economics"); the copies in `docs/blog/` carry working titles, the live slugs may differ
- [ ] **Step 3: Confirm the GitHub repo URL and visibility** (`gh repo view 1cfe/1costingfe --json visibility,url`); if private, raise with the user — the post invites cloning, so it must be public (or have a public mirror) at publication
- [ ] **Step 4: Confirm install instructions**: package is NOT on PyPI; the post must say clone + `pip install -e .` (exactly what Task 3 verified) and must NOT promise `pip install 1costingfe`

### Task 5: Write the post

**Files:**
- Create: `docs/blog/3 Intro/1costingfe-intro-post.md`

- [ ] **Step 1: Write the full draft to the outline below**

Section budgets (about 1,500 words total):

1. **Hook** (150 w). Three prior posts ran on one engine: the cost-floor post and the DEC post made concrete numerical claims with it, and the fusion-tea pipeline post used it as the deterministic costing layer under all 38 concept analyses. This post hands the reader that engine. Link all three.
2. **What it is** (250 w). 1costingFE: open-source, JAX-native costing framework. Three modules in one `forward()` call: economics (LCOE with capital recovery, interest during construction, growing-annuity O&M), physics (first-principles power partitioning for D-T, D-D, D-He3, p-B11; bremsstrahlung, synchrotron, and impurity radiation; steady-state and pulsed power balances), and the full Schulte/ARIES Code of Accounts with 19 reactor-plant sub-accounts, every account overridable and tracked. 17 confinement concepts, 4 fuels. Cite tag `v0.1.0-alpha.1` as the version described.
3. **Try it live** (100 w). The explorer at the Vercel URL: pick a concept and fuel, drag sliders, watch LCOE, the account breakdown, and the elasticity tornado update live. One screenshot (`explorer_screenshot.png`). State plainly: it is the same model, numpy-ported, computing on every slider move, not a lookup table.
4. **Clone and run** (350 w). Install block (clone at tag, `pip install -e .`), then `snippet_forward.py` verbatim, then its captured output verbatim. One sentence per output artifact: LCOE, overnight capital, the named CAS22 sub-accounts. Mention `examples/` (27 entries) including the scripts that reproduce the prior posts' numbers, and the per-account justification docs in `docs/account_justification/`.
5. **The differentiable part** (250 w). `snippet_grad.py` verbatim plus captured output. The tornado figure. Two points, briefly: one backward pass gives every elasticity at once (and on the dashboard the same quantities come from finite differences on the numpy port); and the elasticity vector is the discipline against over-tuned models — parameters with near-zero elasticity do not deserve calibration effort.
6. **Landscape** (100 w). One paragraph: PROCESS, bluemira, FUSE.jl, and FAROES are deeper on tokamak engineering and D-T; pyFECONS and the CATF fork span concepts and fuels as parameterized cost walkers. No existing open tool combines concept-and-fuel breadth with concept-aware physics and autodiff sensitivity; that intersection is the slot 1costingFE fills. Link the paper for the full comparison.
7. **What's shaky** (150 w). Alpha, and specifically: stellarator and mirror geometry use framework-default sizing (tokamak has the only concept-specific 0D equilibrium); aneutronic building and tritium-plant scope is calibrated against thin reference data; pulsed inductive DEC has no reactor-scale validation anywhere, treat its defaults skeptically. Invite correction over deference.
8. **Invitation** (150 w). In order of value to us: reference cost data for any account (sourced beats anonymized); account-level review of a single `docs/account_justification/` file; validation cases (published design with full cost breakdown we fail to reproduce); bug reports on GitHub issues. Close with the three links: explorer, repo at tag, paper.

Constraints while drafting:
- Every number in the text comes from a `freeze_outputs/` file. No inline-computed numbers.
- No mention of: power-to-geometry sizing, H_factor wiring, magnet selection tables, target-factory three-term capex internals, or anything from June 9-10.
- pyFECONS/NtTau appear only in the landscape paragraph, never as a value source.
- Style rules from the header of this plan.

- [ ] **Step 2: Self-check against the old draft** — skim `1costingFE intro.md` once for facts worth keeping that the new post lost (links, references); do NOT import its structure, conclusions section, roadmap, or comparison table.

### Task 6: Insert explorer link and screenshot (blocked on explorer Task 11)

- [ ] **Step 1: Replace the explorer URL placeholder** with the production Vercel URL from the explorer plan Task 11; verify it loads logged-out (private window).
- [ ] **Step 2: Confirm `explorer_screenshot.png`** (taken in explorer plan Task 10) shows a non-default, interesting state — e.g., pulsed FRC D-He3 with the tornado visible. Reference it in section 3.

### Task 7: Final verification pass

- [ ] **Step 1: Link check** — extract every URL from the post and curl each for HTTP 200 (or confirm by eye for sites that block bots):

```bash
grep -oE 'https?://[^) ]+' "/mnt/c/Users/talru/1cfe/1costingfe/docs/blog/3 Intro/1costingfe-intro-post.md" | sort -u | while read u; do code=$(curl -s -o /dev/null -w "%{http_code}" -L --max-time 15 "$u"); echo "$code $u"; done
```

- [ ] **Step 2: Style sweep** — search the post for: `—` (em dash), `~` before digits, escaped `\$`, "FRC-style", praise filler. All must be absent. Word count: `wc -w` between 1,200 and 1,800.
- [ ] **Step 3: Number cross-check** — every LCOE/elasticity/count in the post matches a `freeze_outputs/` file or a verified repo fact (17 concepts, 4 fuels, 27 example entries, test count as reported by `uv run pytest --collect-only -q | tail -1` in the freeze worktree).
- [ ] **Step 4: Remove the fact-check comment block** from the top of the post once all items are resolved.
- [ ] **Step 5: Hand to user for editorial review.** Publication itself (posting to 1cf.energy) is the user's step.
