# Laser-IFE driver: per-architecture capital + scheduled-replacement (CAS72)

**Date:** 2026-06-04
**Status:** Design approved, pre-implementation
**Supersedes:** PR #28 (`feat/laser-ife-driver-replacement`) — this design reworks it
**Affects:** CAS22.01.04 (driver capital), CAS72 (scheduled replacement), `LASER_IFE`

## Problem

`LASER_IFE` budgets no scheduled driver replacement, understating O&M for the
LCOE-dominant cost of a rep-rated laser plant. PR #28 added a replacement term
but has three defects this design fixes:

1. **One capital, three replacement profiles.** `C220104` for `LASER_IFE` is
   always the DPSSL figure (205 M$/MJ). The PR charged KrF "amplifier-tube" and
   Nd:Glass "flashlamp" replacement against DPSSL hardware that has neither.
2. **Wrong replacement math.** The PR used a continuous straight-line
   (`frac·capex·shots/yr / life`). Driver subsystems span replacement intervals
   from sub-annual (flashlamps) to multi-decade (pump diodes); the straight-line
   over-charges long-lived items (it bills ~0.8 of a diode set the plant never
   replaces, on top of the diode set already in capex) and the capped discrete
   loop used elsewhere (`MAX_REP=20`) under-charges near-annual optics and cannot
   represent sub-annual flashlamps. No existing pattern is correct across
   1e4–1e10 shot lifetimes.
3. **Defaults in code, not YAML.** New constants landed only in `defaults.py`,
   and the driver-type default was hardcoded in `model.py`
   (`if concept==LASER_IFE: DPSSL`), violating the defaults-in-YAML convention.

## Decisions (locked with user)

- **Scope:** all three driver types are first-class — branch *both* capital and
  replacement by type. One `LASER_IFE` concept; the architecture is a parameter,
  not a separate concept (parallel to `pulsed_conversion`).
- **KrF capital = 40 M$/MJ** (leans to the Xcimer/ASPEN architectural claim of
  cheaper large-aperture optics; range 20–200 documented).
- **Replacement model = geometric closed-form** PV (exact across all regimes).

## Architecture

### 1. Capital — branch C220104 by driver type

`C220104 = c_driver[type] × E_drv` for `LASER_IFE`. Coefficients (M$/MJ of
delivered pulse energy, driver hardware, NOAK):

| Type | M$/MJ | Range | Basis / source |
|---|---|---|---|
| DPSSL | **205** | 200–700 | Existing default; diode-cost-gated. Zuegel, ARPA-E IFE workshop 2023 |
| KrF | **40** | 20–200 | Xcimer/ASPEN $10–20/J optical-on-target (30× vs NIF claim, unproven long-pulse optics) vs NRL/Sethian engineering baseline $200/J (Sethian, *Fusion Sci. Tech.* 64, 2013). 40 leans architectural-claim-but-not-promotional |
| Nd:Glass | **1000** | 500–2000 | NIF $3.5–4.2B / 1.1–1.9 MJ UV ≈ $2000/J facility, driver-only ~half |

The MagLIF preheat line (`laser_preheat_per_mj`) is unchanged — it is a separate
concept and stays DPSSL-class.

`cas22.py` selects the coefficient for `LASER_IFE` from `laser_driver_type`; all
other concepts unchanged. This requires threading `laser_driver_type` into the
`cas22` forward call.

### 2. Replacement — one shared geometric closed-form helper

Replacement costing already exists inline in `cas70_om`, in two forms: a
discrete bounded PV loop (`MAX_REP=20`) used by **core**, **DEC grid**, and
**cap bank**, and a continuous straight-line (electrode). The geometric
closed-form below *is the discrete loop's exact closed form*
(`Σ_{k=1}^{n_rep} E·s^k = E·s(1−s^n_rep)/(1−s)`) with no iteration cap — so it is
not a new method. Extract it once and route the three discrete blocks **and**
the laser block through it:

```python
def levelized_replacement_cost(event_cost, t_replace, interest_rate, lifetime_yr):
    """Level annual cost of replacing an item every t_replace years over the
    plant life. Exact for any interval (sub-annual to multi-decade). Nominal
    discount only, PV at operation start, annualized by CRF — the convention the
    core/DEC/cap-bank blocks already use. The first set is capital, so only
    replacements *beyond* it are charged (n_rep = ceil(n/t)-1)."""
    s = (1.0 + interest_rate) ** (-t_replace)            # per-interval discount, <1 for i>0
    n_rep = jnp.maximum(0.0, jnp.ceil(lifetime_yr / t_replace) - 1.0)
    pv = event_cost * s * (1.0 - s ** n_rep) / (1.0 - s) # geometric series; n_rep=0 -> 0
    return pv * compute_crf(interest_rate, lifetime_yr)
```

Callers pass `t_replace` in years: core → `core_lifetime_cal`; DEC →
`dec_grid_life_cal`; cap bank → `cap_shot_lifetime / n_shots_per_year`; laser
subsystem → `shot_lifetime / n_shots_per_year`, with
`event_cost = replace_frac × C220104 × n_mod`.

**Effect of the refactor:**
- **No shipped numbers change.** At every shipped default `n_rep ≤ 20`
  (core ≤6, cap bank = 8 at 1 Hz), where the closed form equals the loop exactly.
- **Fixes a latent cap-bank undercount.** The `MAX_REP=20` loop silently
  truncates when inductive-DEC is run at high `f_rep` (e.g. selected for
  laser-IFE/heavy-ion at 5–10 Hz needs ~40–80 events). The closed form is exact.
- **Electrode is left as-is** (continuous + inflation-escalation convention via
  `levelized_annual_cost`; migrating it would recalibrate its ~$25M/yr).

Correct in every laser regime: diodes (`t≈37 yr` > plant → `n_rep=0` → $0, no
double-count), final optics (`t≈1.1 yr` → ~26 events, exact), flashlamps
(`t≈4e-5 yr` → perpetuity limit, prohibitive but finite — surfaces Nd:Glass
non-viability). JAX-safe: `i>0 ⇒ s<1 ⇒ 1−s>0`; `s**n_rep` via `exp(n_rep·log s)`
stays finite; `ceil` has zero gradient.

### 3. Subsystems and per-subsystem defaults

`replace_frac` = fraction of `C220104`; `shot_lifetime` = NOAK projection.
Lifetimes are mature-line targets, not demonstrated (demonstrated noted in the
account justification). KrF/Nd:Glass cost shares are engineering estimates
(no verified component breakdown) and are flagged as such in the doc.

| Type | Subsystem | replace_frac | shot_lifetime | Note |
|---|---|---|---|---|
| DPSSL | Pump diodes | 0.50 | 1.0e10 | ≈ plant life ⇒ ~capital; demonstrated ~1e8 (Mercury). The make-or-break sensitivity lever |
| DPSSL | Conversion crystals (KDP/DKDP) | 0.03 | 3.0e9 | Small, long-lived |
| DPSSL | Final optics (GIMM/transport/debris shields) | 0.05 | 3.0e8 | Dominant O&M; near-annual. GIMM NOAK target >3e8, demonstrated ~1e5 |
| KrF | Hibachi foil + windows | 0.04 | 3.0e8 | Electra durability target >3e8; demonstrated ~1e4–1e5 (engineering estimate) |
| KrF | E-beam diode + gas system | 0.06 | 3.0e8 | Engineering estimate |
| Nd:Glass | Flashlamps | 0.10 | 1.0e4 | Xe-arc-limited; demonstrated O(1e3–1e4). Glass slabs = capital |

Dispatch table in `costs.py` maps `LaserDriverType → [(frac_attr, life_attr), ...]`;
`cas72 += Σ levelized_replacement_cost(...)`. Guards unchanged from PR:
`concept==LASER_IFE and f_rep>0 and laser_driver_type is not None`.

### 4. Selection mechanism (how a tech is chosen)

Parallel to `pulsed_conversion`:

- **Concept default:** `laser_driver_type: dpssl` in
  `data/defaults/pulsed_laser_ife.yaml` (where `pulsed_conversion: thermal` lives).
- **Adapter input:** add `laser_driver_type: str = ""` to `FusionTeaInput`;
  `""` → concept default, else `LaserDriverType(inp.laser_driver_type)`.
- **Per-run override:** `CostModel(..., laser_driver_type=LaserDriverType.KRF)`.
- Remove the hardcoded `if concept==LASER_IFE: DPSSL` from `model.py`; resolve
  the default from the concept YAML (`self._eng_defaults`).

### 5. Defaults live in YAML

Add to `data/defaults/costing_constants.yaml` (alongside `cap_shot_lifetime`):
the three capital coefficients (`driver_laser_per_mj: 205`, `driver_krf_per_mj: 40`,
`driver_ndglass_per_mj: 1000`) and all six `(replace_frac, shot_lifetime)` pairs.
Dataclass fields in `defaults.py` carry the same values as fallback.

## Documentation (no new files)

1. **`docs/account_justification/CAS22_reactor_components.md`**
   - Extend `#### Driver costs` table: branch the Laser IFE row by driver type;
     add KrF (40 M$/MJ) and Nd:Glass (1000 M$/MJ) with bases/sources above.
   - Add `#### Laser-driver scheduled replacement (CAS72 O&M)` after the
     electrode section, mirroring its style: the geometric formula, the
     per-subsystem table (demonstrated vs NOAK lifetime, replace_frac, source),
     calibration (DPSSL ≈ $48M/yr at $1B/10 Hz/0.85, optics-dominated; diodes
     ~0 as capital; Nd:Glass prohibitive), and the KrF/Nd:Glass
     estimate caveat.
2. **`docs/papers/1costingfe_paper/1costingfe_paper.tex`**, §CAS22.01.04
   (`sec:cas2204`):
   - Extend the laser-coefficient paragraph and `tab:cas2204-driver` to state the
     three driver-type capital figures (DPSSL 205, KrF 40, Nd:Glass 1000 M$/MJ).
   - Add a short scheduled-replacement paragraph (geometric model; diodes are
     capital at NOAK life, optics dominate, flashlamps non-viable). No history.

## Code touch-list (added by the refactor decision)

`layers/economics.py` (new `levelized_replacement_cost` helper);
`layers/costs.py` (route core/DEC/cap-bank through the helper, removing the 3
`MAX_REP` loops; add laser dispatch); plus the items in §1–5
(`defaults.py`, `types.py`, `cas22.py`, `model.py`, `adapter.py`, two YAMLs).

## Test plan (`tests/test_costs.py`, rework PR's 10)

- **Capital branch:** `C220104` = 205/40/1000 × `E_drv` for DPSSL/KrF/Nd:Glass.
- **Replacement magnitude:** exact match to `levelized_replacement_cost` summed
  over each type's subsystems (DPSSL, KrF, Nd:Glass).
- **Refactor regression:** core/DEC/cap-bank CAS72 unchanged vs current code at
  shipped defaults (n_rep ≤ 20).
- **Cap-bank truncation fix:** at inductive-DEC + high `f_rep` (n_rep > 20) the
  helper exceeds the old truncated loop (documents the bug fix).
- **Diode no-double-count:** at `dpssl_diode_shot_lifetime=1e10` (>plant) the
  diode term ≈ 0; dropping it to 1e8 makes it dominate (sensitivity lever works).
- **Optics near-annual:** DPSSL CAS72 dominated by the optics subsystem.
- **Nd:Glass prohibitive:** Nd:Glass CAS72 ≫ DPSSL ≫ (or vs) KrF.
- **Selection:** YAML default resolves to DPSSL; overriding to KRF changes both
  C220104 and CAS72.
- **No-op guards:** non-`LASER_IFE`, `f_rep=0`, `laser_driver_type=None`.
- **Backward-compat:** default `LASER_IFE` (DPSSL) keeps `C220104` at 205×E_drv.

## Backward compatibility

Default driver type is DPSSL with unchanged 205 M$/MJ capital, so existing
`LASER_IFE` runs change only by the new CAS72 replacement term (which is the
intended addition). KrF/Nd:Glass are opt-in.

## Open uncertainties (documented, not blocking)

- KrF capital (40 M$/MJ) rests on Xcimer's unproven long-pulse-optics scaling;
  conservative bound 200 M$/MJ is recorded.
- KrF/Nd:Glass subsystem cost shares are engineering estimates — no verified
  component cost breakdown exists in the literature.
- DPSSL NOAK diode life (1e10) is a defensible derivation from LIFE 30-yr/16-Hz
  arithmetic, not a single cited line item.

## Sources

LIFE/DPSSL: Dunne et al. (LIFE laser systems); Orth/Bibeau DPSSL system studies
(OSTI 15013230, 102290); Mercury (OSTI 1019071); Zuegel ARPA-E IFE 2023; Häfner
diode pumps (LLNL IFE 2022). Final optics: Latkowski et al. fused-silica final
optics (OSTI 20845924); GIMM (UCSD-CER-05-08; *Fusion Sci. Tech.* 56(1)). KrF:
Sethian et al. Electra (DTIC ADA480681); Sethian *Fusion Sci. Tech.* 64 (2013);
Xcimer/ASPEN (Galloway, LLNL IFE 2022; Xcimer FAQ). Nd:Glass/NIF: lasers.llnl.gov;
NIF cost/flashlamp specs.
