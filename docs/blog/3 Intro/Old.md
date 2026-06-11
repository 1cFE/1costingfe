# Goals of this post

1. Demonstrate current capabilities and features of 1costingFE and give the reader a sense of what the framework makes possible: concept and fuel coverage across all major confinement families, JAX-native autodiff for elasticities and Monte Carlo, and cost overrides at every account level.
2. Recruit sophisticated users to pull the code and engage with us on bugs, missing fuel and concept cases, account-level review, reference cost data, and feature requests. Think: Simon Woodruff, Layla Araiinejad, Jacob Schwartz.
    1. Think distribution to **Open Source Software for Fusion Energy** https://ossfe.org/’s mailing list

## Draft

*Working public title: A Differentiable, Open-Source Framework for Fusion Plant Economics*

*Follows the free core and  DEC posts which both reference and use 1costingfe. Paired with the worked-example post at slot 4.1.*

---

# 1costingFE: A Differentiable, Open-Source Framework for Fusion Plant Economics

## Opening

Every fusion design paper we read has the same shape. The physics is detailed. The geometry is specified. The economics is a footnote, or missing, or asserted. A 2015 compact tokamak paper [1] gives costs for three components and leaves the balance of plant out of scope. A recent stellarator preconceptual design publishes a target LCOE with no breakdown. The few public fusion costing tools are either calibrated to a single confinement family (usually D-T tokamaks) or are parameterized cost walkers rather than concept-aware models.

Every fusion design paper has the same shape. The physics is detailed. The geometry is specified. The economics is a footnote, asserted, or absent. A 2015 compact tokamak paper gives costs for three components and leaves the balance of plant out of scope (Sorbom et al., 2015). A recent stellarator preconceptual design publishes a target LCOE with no breakdown. The few public fusion costing tools are either calibrated to a single confinement family (usually D-T tokamaks) or are parameterized cost walkers without concept-aware physics.

1cFE's question is whether any fusion concept can reach $0.01/kWh. Answering it requires a costing framework that spans confinement families and fuel cycles, computes sensitivities at scale, and carries its assumptions on the surface where they can be challenged. We wrote one: 1costingFE, a JAX-native, open-source costing framework that takes a concept, a fuel cycle, and a target power, and returns a full Code of Accounts Structure breakdown, a closed power balance, and a complete set of sensitivity elasticities from one backward pass.

The two prior dispatches on this site asked a sharp version of the corridor question: can fusion electricity reach 1 cent per kWh? The first post showed that even with a free fusion core, the D-T balance-of-plant floor is roughly $29/MWh, and only aneutronic fuels at large scale push the floor below $10/MWh. The second post showed that direct energy conversion does not move that floor much; its leverage shows up on core size and fuel burn. Both posts make concrete numerical claims. Both posts use the same engine.

The code is public. The methodology paper is public. We want users.

## What it is

1costingFE is three modules wired together. The economics module converts a capital cost and a set of financial parameters into a levelized cost per MWh. The physics module converts a fusion power and a set of engineering coefficients into a net electric output, across four fuels (D-T, D-D, D-³He, p-¹¹B) and both steady-state and pulsed concepts. The CAS module carries the cost account structure developed by Schulte and colleagues at Pacific Northwest Laboratory in 1978 [2], adopted by the ARIES program and the Generation IV Economic Modeling Working Group, and implemented in pyFECONS by Woodruff Scientific [3]. We use pyFECONS as a reference implementation and a validation target; our framework differs in three ways that matter for the corridor question:

1. **Concept and fuel coverage.** One model handles tokamaks, stellarators, mirrors, IFE, Z-pinch, and inductive-recovery concepts, across all four fusion fuels, from one call site.
2. **Automatic differentiation.** The JAX backbone gives us exact elasticities of LCOE with respect to every input parameter in a single backward pass, vectorised Monte Carlo across assumption distributions, and JIT-compiled sweeps.
3. **Parsimony by design.** Each account is independently justified from literature or first principles, parameters are added only where data or physics supports them, and every default value is tagged so users can see what came from the framework and what came from the source paper.

The user surface is a single call:

```python
from costingfe import ConfinementConcept, CostModel, Fuel

model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
result = model.forward(
    net_electric_mw=1000.0,
    availability=0.85,
    lifetime_yr=30,
)

print(result.lcoe_usd_per_mwh)        # 87
print(result.overnight_capex_per_kw)  # 6166
print(result.cas22_detail)            # dict of reactor plant equipment accounts
```

That is the baseline: a generic D-T tokamak at 1 GWe, roughly $87/MWh NOAK. The returned object exposes every CAS account from 10 through 90, every power balance quantity, and every intermediate, all addressable by name.

## Why differentiable

The JAX backbone is the design decision that most shapes what you can do with the framework.

**Exact elasticities in a single backward pass.** The sensitivity of LCOE to every input parameter comes out of `jax.grad` without a parameter sweep. For a model with fifty inputs, you get fifty elasticities in one backward pass, not fifty forward runs. This moves sensitivity analysis from a batch process to an interactive one.

**Vectorised Monte Carlo over assumption distributions.** `jax.vmap` samples assumption distributions and propagates them through the full cost computation in parallel. A thousand-sample uncertainty analysis fits in one call.

**JIT-compiled sweeps.** Once compiled, a parametric sweep across concepts or sizes runs at native speed. The compile step is paid once.

The flip side of this power is a discipline constraint. A model with too many tunable parameters, in a regime where calibration data is sparse, overfits. If you can back-solve any LCOE you want by wiggling enough knobs, the model tells you nothing. 1costingFE is built bottom-up with parameters introduced only where literature or physics supports them. The autodiff machinery then tells you which of those few parameters actually matter, so attention can go to calibrating the dominant ones rather than tuning the noise floor.

## The three modules

### Economics

The economics module computes:

$\text{LCOE} = \frac{C_\text{annual} + M_\text{annual}}{8760 \cdot P_\text{net} \cdot A}$

where $C_\text{annual}$ is the annualised capital cost, $M_\text{annual}$ the annualised operating cost, $P_\text{net}$ the per-module net electric power, and $A$ the availability. Three details merit attention because they tend to go wrong in simpler treatments:

- **Capital Recovery Factor** converts overnight cost plus interest during construction into equal annual payments. Standard formula; easy to implement wrong if construction-period compounding is neglected.
- **Interest During Construction** accrues as capital is disbursed over the build window before any revenue arrives. At 7% WACC and a 6-year build, IDC adds about 23% to the overnight cost.
- **Growing-annuity correction for operating costs.** Annual O&M is stated in base-year dollars but escalates at the inflation rate in nominal terms. A naive flat-inflation treatment understates present-value operating costs by tens of percent over plant life. We implement the closed-form growing annuity.

Full derivations are in the paper, §2.

### Physics

The physics module has two jobs: compute the fusion power split between neutrons, charged particles, and photons from first-principles nuclear data; and close the power balance from fusion power through blanket multiplication, thermal cycle, and recirculating loads to net electric output.

Fuel-specific power splits are where most TEA tools stop being concept-agnostic. 1costingFE treats all four major fuels as first class. D-T is the simple case: 17.6 MeV per reaction, 80% in neutrons, alpha ash inert. D-D is the hard case: two primary branches with secondary D-T and D-³He burns whose burn fractions are parameterised and affect both the effective energy per event and the neutron fraction. D-³He carries a small D-D side reaction that produces tritium and thus a nonzero, economically relevant neutron fraction. p-¹¹B is the most complex: the ¹¹B(α,n) and ¹¹B(p,n) side reactions are parameterised, and the default treatment follows Ochs et al. [4] on alpha ash management, which gives a clean aneutronic default unless the user chooses otherwise.

Radiation losses are modelled with Wesson's bremsstrahlung, Albajar's global synchrotron model [5] with Fidone's wall-reflectivity correction, and a first-wall-sputtering-driven impurity line radiation model with coronal-equilibrium $L_z(T_e)$ for eight wall materials. Power balance covers both steady-state (tokamak, stellarator, mirror) and pulsed (IFE, Z-pinch, FRC-style inductive DEC) concepts. The pulsed inductive DEC path distinguishes capacitor-bank round-trip losses from full grid draw, which matters for concepts where the capacitor bank recirculates most of its energy each cycle.

Concept-specific physics (for example, the 0D tokamak equilibrium that sizes the coils from field and bore) lives in concept-specific modules. The tokamak model is functional today; stellarator and mirror layers are on the roadmap.

### Cost Accounts

The Code of Accounts Structure is a hierarchical decomposition of every plant cost. Capital accounts CAS10 through CAS60 cover land through financing. Annual accounts CAS70 through CAS90 cover operations and maintenance, fuel, and the annualised capital charge. Each account has its own internal justification documenting how the cost is computed, what it is calibrated against, and where the uncertainty lives.

Every account accepts an override. If the published design paper gives a magnet cost, you override CAS220103. If a vendor quotes a blanket cost, you override CAS220101. Overrides are tracked in the output so framework defaults are always distinguishable from user-supplied numbers.

## A working demo

The baseline call returns an `LCOEResult` with the full plant breakdown. What gets interesting is the differentiability:

```python
from jax import grad
import jax.numpy as jnp

def lcoe_from_params(net_mw, avail, wacc, construction_yr):
    return model.forward(
        net_electric_mw=net_mw,
        availability=avail,
        interest_rate=wacc,
        construction_time_yr=construction_yr,
    ).lcoe_usd_per_mwh

partials = grad(lcoe_from_params, argnums=(0, 1, 2, 3))(1000.0, 0.85, 0.07, 6.0)
```

The call returns the partial derivatives of LCOE with respect to each argument, simultaneously. Multiplying each partial by the reference value of its argument and dividing by LCOE gives the elasticity vector. For the baseline D-T tokamak at 1 GWe:

```
Parameter                       Elasticity
─────────────────────────────────────────────────────
Availability                       -0.96
Cost of capital (WACC)             +0.77
Construction time                  +0.30
Thermal efficiency                 -0.13
Coil winding radius                +0.12
Peak toroidal field                +0.08
Plasma major radius                +0.02
Heating wall-plug efficiency       -0.01
```

*Numbers illustrative; to be regenerated from a fresh `quickstart.ipynb` run before publication.*

In a conventional framework, producing this table requires a parameter sweep: forty-plus runs, each with its own finite-difference step. Here it costs one additional backward pass on top of the forward run.

Monte Carlo over assumption distributions is similarly cheap. The call below draws a thousand samples from independent uniform priors on availability, WACC, and thermal efficiency, and runs the full model on each in parallel:

```python
keys = jax.random.split(jax.random.PRNGKey(0), 1000)
samples = jax.vmap(draw_assumption_sample)(keys)
lcoes = jax.vmap(lcoe_from_params)(*samples.T)
```

Elapsed time on a laptop is measured in seconds, not minutes.

## Try it

```bash
pip install 1costingfe
```

The repo contains:

- `examples/` runnable Jupyter notebooks reproducing every number in this post and the paper
- `costingfe/` the framework
- `tex/` the methodology paper and build
- `tests/` account-level unit tests and end-to-end integration checks

A ten-minute walkthrough sits at `examples/quickstart.ipynb`. Working from the baseline D-T tokamak to a concept-overridden model with sensitivity analysis takes about that long.

## What's supported today

| Confinement | Concept-specific physics | CAS coverage | Status |
| --- | --- | --- | --- |
| Tokamak | 0D equilibrium, coil sizing | Full | Working |
| Stellarator | Framework defaults only | Full | Working without concept-specific sizing |
| Mirror | Framework defaults only | Full | Experimental |
| IFE / Z-pinch | Pulsed power balance | Full | Working |
| Inductive DEC (FRC-style) | Pulsed inductive balance | Full | Working, undervalidated |
| Fuel | Primary reaction | Secondary chains | Default treatment |
| --- | --- | --- | --- |
| D-T | Complete | None | Neutronic |
| D-D | Complete | D-T and D-³He burns parameterised | Neutronic |
| D-³He | Complete | D-D side reactions | Low-neutron |
| p-¹¹B | Complete | ¹¹B(α,n), ¹¹B(p,n) parameterised | Aneutronic |

Direct energy conversion is implemented for pulsed inductive concepts and as an optional override for steady-state ones. The default is thermal conversion only.

## What's on the roadmap

Priorities for the next three months:

1. **Stellarator and mirror physics modules.** Concept-specific sizing that pulls magnet geometry from a concept-appropriate equilibrium model rather than using framework defaults.
2. **Calibrated aneutronic balance-of-plant.** The CAS26 tritium plant account is independently justified for D-T; the reduced obligation for aneutronic fuels is currently approximated. Buildings (CAS21) is in the same state.
3. **DEC for steady-state concepts.** Today DEC is off by default for MFE. As our lower-bound dispatch on fusion's cost floor and DEC deep dive argue, p-¹¹B with direct energy conversion is where the sub-cent corridor sits, and we want a defensible cost model for it.
4. **Integration with Fusion-TEA.** The SysML v2 pipeline described in our methodology post ingests papers, builds system models, and currently emits hand-written 1costingFE scripts. Code generation from SysML to 1costingFE is the next integration target.
5. **Stable release cadence on PyPI.**

## What we'd love feedback on

This is early code. We expect to be wrong in places. In rough order of what helps us most:

**Reference data.** If you have NOAK or FOAK cost numbers for any CAS account, tell us. Magnet costs at contemporary REBCO prices, blanket costs at modern FLiBe or lithium prices, balance of plant from recent fission or combined-cycle projects. Anonymised is fine.

**Missing fuel or concept cases.** Hybrid fuel cycles, muon-catalyzed concepts, mass-accelerated fuel, inertial electrostatic variants. If your concept doesn't fit the framework's abstractions cleanly, we want to hear about it.

**Account-level review.** Pick any single CAS account, read the internal justification, and tell us where we're wrong. Specific is better than general.

**Bugs.** File issues on GitHub. We respond.

The code is at github.com/1cfe/1costingfe. The paper is at github.com/1cfe/1costingfe/blob/master/tex/paper.tex. 

## References

1. Sorbom, B. N. et al. "ARC: A compact, high-field, fusion nuclear science facility and demonstration power plant with demountable magnets." *Fusion Engineering and Design* 100, 378 (2015). DOI
2. Schulte, S. C. et al. "Fusion Reactor Design Studies: Standard Accounts for Cost Estimates." PNL-2648, Pacific Northwest Laboratory (1978).
3. Woodruff, S. "A Costing Framework for Fusion Power Plants." arXiv:2601.21724 (2026). Link
4. Ochs, I. E. et al. "Steady-state relaxation of p-¹¹B plasmas." (2025). *Final citation to be supplied by Tal.*
5. Albajar, F. et al. "Synchrotron radiation loss in tokamak plasmas." *Nuclear Fusion* 41, 665 (2001). DOI
6. CATF Investors Working Group. "Assessing the cost of fusion energy." arXiv:2602.19389 (2025). Link

---

<!--
TO VERIFY BEFORE PUBLISHING:

- PNL-2648 OSTI link (https://www.osti.gov/biblio/6395456) is a plausible guess, not verified. Confirm the correct OSTI ID for Schulte et al. 1978 and update reference 2.
- Cross-link slug for the DEC post (https://1cf.energy/direct-energy-conversion-and-the-cost-floor/) was extrapolated from the post title; confirm against the actual published URL.
- Illustrative elasticity table (Availability, WACC, Construction time, etc.) is carried over from the early draft; regenerate against the tagged release before publication.
- Albajar DOI corrected to 10.1088/0029-5515/41/6/301; double-check.
- Confirm `pip install 1costingfe` is the intended PyPI name (vs. `costingfe`), since the import name is `costingfe`.
- One failing test (279/280) is disclosed in the limitations section; fix or remove the disclosure before tagging.
-->

This post introduces that engine. 1costingFE is a JAX-native, open-source costing framework that takes a confinement concept, a fuel cycle, and a target net electric power, and returns a full Code of Accounts breakdown, a closed power balance, and a complete set of sensitivity elasticities from one backward pass. It covers 15 confinement concepts and 4 fusion fuels, all callable from one API.

We wrote 1costingFE because no existing open tool natively handles aneutronic fuels (p-B11, D-He3) with direct energy conversion across multiple confinement families with autodiff sensitivity in one call. The code is public. The methodology paper is public. The repo has 22 example scripts, 280 unit tests, and the scripts that reproduce every number in the prior two posts. We want users.

## What 1costingFE is

1costingFE is three modules wired together.

1. **Economics** converts a capital cost and a set of financial parameters into a levelized cost per MWh, with explicit treatment of construction-period interest and growing-annuity O&M.
2. **Physics** converts a fusion power and a set of engineering coefficients into a net electric output, across four fuels (D-T, D-D, D-He3, p-B11) and both steady-state and pulsed concepts, with first-principles fuel partitioning, Wesson bremsstrahlung, Albajar synchrotron with Fidone wall reflectivity, and an 8-material first-wall sputtering and impurity-radiation model.
3. **Cost accounts** carry the full Code of Accounts (CAS10 through CAS90) developed by Schulte et al. at Pacific Northwest Laboratory in 1978, adopted by the ARIES program and the Generation IV Economic Modeling Working Group, with 19 CAS22 reactor-plant-equipment sub-accounts addressable by name and overridable from user input.

The user surface is one call:

```python
from costingfe import CostModel, ConfinementConcept, Fuel

model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
result = model.forward(
    net_electric_mw=1000.0,
    availability=0.85,
    lifetime_yr=30,
)

print(result.costs.lcoe_usd_per_mwh)        # baseline DT tokamak
print(result.costs.overnight_capex_per_kw)  # overnight capital
print(result.cas22_detail)                  # 19 reactor-plant sub-accounts
print(result.power_table)                   # closed power balance
print(result.overridden)                    # which accounts came from user input
```

The returned `ForwardResult` exposes `power_table` (the full power balance from fusion power through to grid), `costs` (the LCOE, CRF, IDC, and CAS rollup), `cas22_detail` (the 19 reactor-plant-equipment sub-accounts including blanket C220101, coils C220103, DEC C220109), `params` (every numeric input and intermediate, addressable by name), `overridden` (the set of accounts where user input replaced framework defaults), and `plasma_state` (for concepts with concept-specific physics, the equilibrium quantities used downstream).

That single call also drives the pulsed-FRC D-He3 case from the DEC post and the free-core D-T case from the cost-floor post: same API, different concept and fuel.

## Why it is differentiable

The JAX backbone is the design decision that most shapes what you can do with the framework.

**Exact elasticities in one backward pass.** `jax.grad` returns the partial derivatives of LCOE with respect to every model input simultaneously. For a model with 50 inputs you get 50 elasticities for the cost of one backward pass on top of the forward run, not 50 finite-difference forward sweeps. This moves sensitivity analysis from a batch process to an interactive one.

```python
from jax import grad

def lcoe_from_params(net_mw, avail, wacc, construction_yr):
    return model.forward(
        net_electric_mw=net_mw,
        availability=avail,
        interest_rate=wacc,
        construction_time_yr=construction_yr,
    ).costs.lcoe_usd_per_mwh

partials = grad(lcoe_from_params, argnums=(0, 1, 2, 3))(1000.0, 0.85, 0.07, 6.0)
```

For convenience, `model.sensitivity()` returns elasticities for engineering, financial, and costing parameters as a single labelled dict. Illustrative values for a baseline D-T tokamak at 1 GWe (numbers regenerate from the released `examples/dt_tokamak.py`):

| Parameter | Elasticity |
| --- | --- |
| Availability | -0.96 |
| Cost of capital (WACC) | +0.77 |
| Construction time | +0.30 |
| Thermal cycle efficiency | -0.13 |
| Coil winding radius | +0.12 |
| Peak toroidal field | +0.08 |
| Plasma major radius | +0.02 |
| Heating wall-plug efficiency | -0.01 |

*Numbers above are illustrative for the present alpha and will be regenerated against the tagged release.*

**Vectorised Monte Carlo.** `jax.vmap` propagates assumption distributions through the full cost computation in parallel. A thousand-sample uncertainty analysis fits in one call. The included `model.batch_lcoe()` wraps this idiom for parameter sweeps:

```python
import jax, jax.numpy as jnp

keys = jax.random.split(jax.random.PRNGKey(0), 1000)
samples = jax.vmap(draw_assumption_sample)(keys)
lcoes = jax.vmap(lcoe_from_params)(*samples.T)
```

Elapsed time on a laptop is measured in seconds, not minutes. The `examples/uncertainty_analysis.py` script demonstrates the full Monte Carlo workflow; `examples/tornado_plot.py` demonstrates the elasticity-driven tornado plot.

**JIT-compiled sweeps.** Once compiled, a parametric sweep across concepts or sizes runs at native speed. The compile step is paid once.

The flip side of this power is a discipline constraint. A model with too many tunable parameters in a regime where calibration data is sparse overfits. If you can back-solve any LCOE you want by wiggling enough knobs, the model tells you nothing. 1costingFE is built bottom-up with parameters introduced only where literature or physics supports them. The autodiff machinery then tells you which of those few parameters actually matter, so calibration attention can go to the dominant ones rather than the noise floor. To make this explicit: every internal value is traced, all conditionals route through `jnp.where` rather than Python control flow, and the framework exposes the elasticity vector directly.

## The economics module in detail

The LCOE formula is standard:

$\text{LCOE} = \frac{C_\text{annual} + M_\text{annual}}{8760 \cdot P_\text{net} \cdot A}$

where $C_\text{annual}$ is the annualised capital cost (overnight capex plus IDC, multiplied by the capital recovery factor), $M_\text{annual}$ is the present-value-equivalent annual operating cost, $P_\text{net}$ is the net electric power per module, and $A$ is the availability. Three details merit attention:

- **Capital Recovery Factor (CRF)** converts overnight cost plus interest during construction into equal annual payments over the plant lifetime at the WACC. Standard formula, easy to implement wrong if construction-period compounding is neglected.
- **Interest During Construction (IDC)** accrues as capital is disbursed over the build window before any revenue arrives. At 7% WACC and a 6-year build, IDC adds about 23% to the overnight cost. Both build window and disbursement schedule are exposed as parameters.
- **Growing-annuity correction for operating costs.** Annual O&M is stated in base-year dollars but escalates at the inflation rate in nominal terms. A naive flat-inflation treatment understates present-value O&M by tens of percent over plant life. We implement the closed-form growing annuity so the LCOE reflects the actual nominal stream over the plant lifetime.

CAS72 component replacements (blanket modules, divertor cassettes, magnets) are scheduled against fuel-dependent component lifetimes and present-value-discounted into the annual O&M stream rather than smeared as a flat fraction. Multi-module plants share staffing, indirects, and site costs across modules; `multi_module.py` in `examples/` walks through the convention.

## The physics module in detail

The physics module has two jobs: compute the fusion power split between neutrons, charged particles, and photons from first-principles nuclear data, and close the power balance from fusion power through blanket multiplication, thermal cycle, and recirculating loads to net electric output.

Fuel-specific power splits are where most TEA tools stop being concept-agnostic. 1costingFE treats all four major fuels as first-class:

| Fuel | Primary Q-value | Secondary chains | Default partition |
| --- | --- | --- | --- |
| D-T | 17.6 MeV | None | 80% neutrons, 20% charged |
| D-D | 3.65 / 4.03 MeV branches | D-T and D-He3 burns parameterised (`dd_f_T`, `dd_f_He3`) | Neutron fraction depends on burn |
| D-He3 | 18.3 MeV | D-D side reactions parameterised (`dhe3_dd_frac`, `dhe3_f_T`) | About 5% neutrons from D-D |
| p-B11 | 8.7 MeV | 11B(α,n)14N (Q = +0.158 MeV) and 11B(p,n)11C (Q = -2.765 MeV) parameterised (`pb11_f_alpha_n`, `pb11_f_p_n`) | About 0.2% neutrons |

For p-B11, the default treatment follows Ochs et al. (2022) on alpha-channeling and non-equilibrium operation, which sets the cleanest aneutronic baseline available unless the user overrides.

Radiation losses are modelled with Wesson's bremsstrahlung formula, the Albajar et al. (2001) global synchrotron model with Fidone's wall-reflectivity correction, and a first-wall-sputtering-driven impurity line-radiation model with coronal-equilibrium $L_z(T_e)$ for eight wall materials (W, C, Be, Mo, Si, Li, Ne, Ar) using Post-Jensen and Mavrin curves, plus Bohdansky and Eckstein sputtering yields and SOL screening. This is more impurity physics than any other costing framework we know of, because for aneutronic fuels the radiated fraction sets whether DEC can pay off at all.

Power balance covers steady-state (tokamak, stellarator, mirror, orbitron, polywell) and pulsed (laser IFE, Z-pinch, heavy-ion, magnetised target, plasma jet, pulsed FRC, MagLIF, theta pinch, dense plasma focus, staged Z-pinch) concepts. The pulsed inductive DEC path distinguishes capacitor-bank round-trip losses from the full grid draw, which matters for concepts where the bank recirculates most of its energy each cycle.

Three power cycles are selectable: Rankine (40%), supercritical CO2 Brayton (47%), and combined cycle (53%). DEC is implemented as pulsed inductive (the default for PULSED_FRC and THETA_PINCH) and as steady-state Venetian-blind (an opt-in overlay parameterised by `f_dec` and `eta_de`).

The tokamak has a 0D plasma equilibrium model that sizes coils from peak toroidal field and bore radius, with a Greenwald-density and q95-dependent disruption-rate estimator (`use_0d_model=True`). Other concepts use framework-default geometry sizing today; concept-specific equilibria for stellarators and mirrors are on the roadmap.

## The cost-account module in detail

The Code of Accounts is a hierarchical decomposition of every plant cost. CAS10 through CAS60 cover land, structures, reactor-plant equipment, turbine-plant equipment, electrical, and indirects. CAS70 through CAS90 cover annual operations, fuel, and the annualised capital charge. Each account has its own internal justification documenting how the cost is computed, what it is calibrated against, and where the uncertainty lives.

The 19 CAS22 reactor-plant sub-accounts cover blanket and first wall (C220101), shield (C220102), magnets (C220103), supplemental heating (C220104), primary structure (C220105), reactor vacuum systems (C220106), power supplies (C220107), impurity control (C220108), DEC (C220109), and ten more. Every account is overridable:

```python
result = model.forward(
    net_electric_mw=1000.0,
    cost_overrides={
        "C220103": 480e6,         # vendor magnet quote
        "CAS21":   570e6,         # bottom-up building estimate
    },
)
print(result.overridden)          # {'C220103', 'CAS21'}
```

Overridden accounts are tracked in the output, so framework defaults are always distinguishable from user-supplied numbers. Power-scaled overrides (`override_reference_mw=...`) let you supply a cost at one plant scale and have it scaled correctly to the target. This is the mechanism we used for the building-cost gap analysis between D-T and p-B11 in the first post.

The framework also supports back-casting (given a target LCOE, solve for one parameter), Monte Carlo uncertainty quantification, multi-module plant costing, FOAK-vs-NOAK learning-curve treatments, and an adapter to the fusion-tea SysML v2 MBSE pipeline. The `examples/` folder includes `path_to_1cent.py`, `foak_vs_noak.py`, and `backcasting_bridge.py` for these workflows.

## What is supported today

| Confinement family | Concept | Concept-specific physics | CAS coverage |
| --- | --- | --- | --- |
| Steady-state MFE | Tokamak | 0D equilibrium, coil sizing, disruption rate | Full |
| Steady-state MFE | Stellarator | Framework-default sizing | Full |
| Steady-state MFE | Mirror | Framework-default sizing | Full |
| Steady-state MFE | Orbitron, Polywell | Framework-default sizing | Full |
| Pulsed IFE | Laser IFE, heavy-ion | Pulsed power balance | Full |
| Pulsed MIF | Z-pinch (incl. staged), MagLIF, theta pinch, dense plasma focus, plasma jet, magnetised target | Pulsed power balance | Full |
| Pulsed FRC | Pulsed FRC | Pulsed inductive DEC default | Full |

| Fuel | Primary partition | Secondary chains parameterised | Default |
| --- | --- | --- | --- |
| D-T | 17.6 MeV; 80/20 split | None | Neutronic |
| D-D | dual-branch | D-T and D-He3 burn fractions | Neutronic |
| D-He3 | 18.3 MeV; charged-dominated | D-D side reactions | Low-neutron |
| p-B11 | 8.7 MeV; charged-dominated | 11B(α,n) and 11B(p,n) | Aneutronic |

Power cycles: Rankine, sCO2 Brayton, combined cycle. DEC: pulsed inductive (default for PULSED_FRC and THETA_PINCH) and steady-state Venetian-blind (opt-in overlay).

Tests: 280 unit tests across all concepts, fuels, and CAS accounts, plus JAX-path tests for `grad`, `vmap`, and `jit`. Examples: 22 scripts in `examples/`, including the `dec_blog_numbers.py` and `lower_bound_blog_numbers.py` that reproduce every number in the prior two posts.

## How 1costingFE compares to the landscape

The fusion TEA tool space is small but real. We owe several of these projects clear acknowledgement, and we use them as validation targets where the scope overlaps. The honest summary is that no existing open tool handles aneutronic fuels with DEC across multiple confinement families with autodiff sensitivity in one call. The gap is what 1costingFE fills.

| Tool | Concept breadth | Fuel breadth | Sensitivity | Cost-account fidelity | License |
| --- | --- | --- | --- | --- | --- |
| PROCESS (UKAEA) | Tokamak, stellarator, ST | D-T anchored | Optimisation, finite difference | ARIES-derived | MIT |
| bluemira (UKAEA et al.) | Tokamak | D-T anchored | Limited; least-mature module | Coupled to engineering, evolving | LGPL 2.1 |
| FUSE.jl (General Atomics) | Tokamak (advanced) | D-T anchored | UQ + ML surrogates | ARIES + Sheffield | Apache 2.0 |
| FAROES (Princeton) | Tokamak (steady-state) | D-T | OpenMDAO derivatives | Sheffield-derived | MIT |
| pyFECONS (Woodruff Scientific) | MFE, IFE, MIF (parameterised) | All fuels (parameterised) | None native | Schulte / ARIES | BSD |
| CATF/IWG fork of pyFECONS | MFE, IFE, MIF | All fuels | None native | Schulte + financial layer | BSD |
| 1costingFE | 15 concepts (MFE/IFE/MIF) | 4 fuels with secondary chains | JAX autodiff, vmap, JIT | Full CAS10-90 + CAS22 detail | MIT (planned) |

A few notes on positioning:

- PROCESS (Kovari et al., 2014; arXiv:1601.06147) is the de-facto reference systems code for EU-DEMO and UK STEP. It has the deepest tokamak engineering coupling of any open tool. For a detailed tokamak design study, PROCESS or FUSE is the right tool today; 1costingFE is the right tool when you need to compare a tokamak to a stellarator to a pulsed FRC at the same confidence level.
- bluemira is the BLUEPRINT and MIRA merger; the geometry, neutronics, and engineering coupling is excellent and the economics module is, by the project's own documentation, the least-mature piece. The two tools have complementary strengths; bluemira's geometry could feed 1costingFE's accounts.
- FUSE.jl (Meneghini et al., 2024; arXiv:2409.05894) carries two cost models (ARIES and Sheffield), modern UQ, and ML surrogates. Tokamak-centric and D-T-centric. The ML-surrogate strategy is one we may borrow for high-throughput corridor sweeps.
- FAROES is OpenMDAO-based with gradient-based optimisation via OpenMDAO derivatives. Steady-state tokamak only. The differentiability story is the closest precedent in the space, and we share the conviction that gradient information is essential for fusion TEA at scale.
- pyFECONS (Woodruff, 2025) has the broadest concept coverage of any prior open framework: MFE, IFE, and MIF in one walker. Where it differs from 1costingFE is that it is a parameterised cost walker rather than a concept-aware physics-plus-cost model: cost as a function of inputs, with the physics and concept assumptions encoded in those inputs. We treat it as a reference implementation and validation target.
- The CATF Investors Working Group extension reorganises pyFECONS around three architecture-defining cost-driver tracks (coils, lasers, pulsed power) and adds a thorough financial-analysis layer (NPV, IRR, learning curves, FOAK and NOAK). The pulsed-power capacitor-cost basis we use in our DEC post comes from this fork.
- Closed or paper-only tools include ARIES (the cost-account documentation is public; the code is not), GASC, Generomak, Sheffield-Milora (spreadsheet lineage), SYCOMORE (CEA), FRESCO (ENEA), and MIRA (now folded into bluemira).

The bottom line: PROCESS, bluemira, FUSE, and FAROES are tokamak and D-T anchored. pyFECONS and CATF span concepts and fuels but encode physics in the inputs rather than in the model. 1costingFE is built for the corridor question, where the answer requires comparing aneutronic and neutronic, steady-state and pulsed, charged-particle DEC and thermal, with sensitivity machinery that does not depend on a parameter sweep.

## Honest limitations

We are at version 0.1.0 alpha. Calling out what does not work yet:

- **Stellarator and mirror layers** use framework-default geometry sizing today. Concept-specific 0D equilibrium sizing (the way the tokamak does it) is on the roadmap. Today's stellarator coil cost uses a `path_factor` of 2.0 for 3D winding routing, conservative compared to HELIAS-anchored estimates.
- **Aneutronic balance-of-plant.** The framework knows D-T plants need shielding, hot cells, tritium plant, and nuclear-rated HVAC; it knows aneutronic plants do not. The CAS21 building scope and CAS26 tritium plant scope for aneutronic fuels are calibrated against limited reference data. We expect the calibration to tighten as more reference designs land.
- **Pulsed inductive DEC** is physically modelled, but no system at reactor scale exists for validation. The inductive-recovery efficiency is parameterised; the reader should treat its default with the same skepticism we did in the DEC post.
- **Synchrotron model** is tuned for toroidal geometry; mirror loss-cone treatment is simplified.
- **0D plasma model** assumes T_i = T_e. For p-B11 at low collisionality, T_i can differ substantially from T_e; that is on the TODO list.
- **PyPI release**: not yet. The package name on import is `costingfe`; the project is `1costingfe`. To install today, clone the repo and run `pip install -e .`, or `pip install git+https://github.com/1cfe/1costingfe`. PyPI release is imminent.
- **Quickstart notebook**: not yet. The closest entry point is `examples/dt_tokamak.py`, which runs end-to-end against a fresh install. A `quickstart.ipynb` is on the roadmap.
- **Test suite**: 279 of 280 tests pass today; one test is failing on a numerical default and will be fixed before the tagged release.

## Roadmap

Priorities for the next three months:

1. **Stellarator and mirror physics modules.** Concept-specific sizing pulled from a concept-appropriate equilibrium model rather than framework defaults.
2. **Calibrated aneutronic balance-of-plant.** Tighten the CAS21 building and CAS26 tritium-plant scope for D-He3 and p-B11 against more reference designs.
3. **DEC for steady-state concepts.** Pulsed inductive is implemented; Venetian-blind is implemented as an overlay; the calibration of both against future experimental data (Realta TMX-style end-loss demonstrations, Helion's Polaris) is open.
4. **Integration with fusion-tea.** The SysML v2 MBSE pipeline ingests papers, builds system models, and currently emits hand-written 1costingFE scripts. Code generation from SysML to 1costingFE is the next integration target.
5. **PyPI alpha release** (`pip install 1costingfe`).
6. **Quickstart notebook** at `examples/quickstart.ipynb` reproducing the baselines from the prior posts in a 10-minute walkthrough.

## What we would value feedback on

This is early code. We expect to be wrong in places. In rough order of what helps us most:

**Reference cost data.** If you have NOAK or FOAK cost numbers for any CAS account, tell us. Magnet costs at contemporary REBCO prices, blanket costs at modern FLiBe or lithium prices, balance of plant from recent fission or combined-cycle projects, pulsed-power capacitor pricing at relevant duty cycles. Anonymised is fine; sourced is better.

**Missing fuel or concept cases.** Hybrid fuel cycles, muon-catalysed concepts, mass-accelerated fuel, inertial-electrostatic variants, Z-pinch architectures we have not enumerated. If your concept does not fit the framework's abstractions cleanly, we want to hear about it before we entrench the abstractions.

**Account-level review.** Pick a single CAS account, read the internal justification (in `docs/account_justification/`), and tell us where we are wrong. Specific is better than general. The accounts we are least confident about are CAS21 (aneutronic buildings), CAS26 (aneutronic tritium plant), and C220103 (stellarator coils with 3D path factor).

**Bug reports.** File issues on GitHub. We respond.

**Validation cases.** If you have a published design with a full cost breakdown and we cannot reproduce it within reasonable tolerances by overriding the relevant accounts, that is a case we want to study.

The code is at github.com/1cfe/1costingfe. The methodology paper is at github.com/1cfe/1costingfe/blob/master/tex/paper.pdf. Examples reproducing the prior two posts are at `examples/lower_bound_blog_numbers.py` and `examples/dec_blog_numbers.py`.

## Conclusions

**1. The corridor question requires breadth and differentiability simultaneously.** Asking whether any fusion concept can reach 1 cent per kWh is not a tokamak question or a D-T question. It is a question about which combinations of fuel, confinement, and conversion sit closest to the floor, and which parameters in those combinations move LCOE the most. Existing open tools span concepts at the cost of physics, or carry physics at the cost of concept breadth. 1costingFE is built for the intersection.

**2. Autodiff is not a luxury for fusion TEA.** A 50-input forward model, computed once, returns a 50-element elasticity vector for the cost of one backward pass. The same framework that produces the LCOE produces the sensitivity to every input. This is the discipline that prevents over-tuned models: if a parameter has near-zero elasticity, calibration effort spent on it is wasted.

**3. Concept and fuel coverage is what 1costingFE buys you over PROCESS, FUSE, bluemira, and FAROES.** For a detailed tokamak engineering study, those tools are more mature today, and we use them as validation targets. For a comparison across confinement families and fuels at a comparable confidence level, no other open tool fills this slot.

**4. pyFECONS and the CATF fork are the closest prior art on concept and fuel breadth.** They are parameterised cost walkers; their physics lives in the inputs. 1costingFE adds first-principles fuel partitioning, an 8-material radiation model, concept-specific power balances, and the autodiff sensitivity layer. We treat pyFECONS as a reference implementation and validation target rather than a competitor.

**5. The framework is alpha and we know it.** Stellarator and mirror sizing, aneutronic BOP calibration, and pulsed-inductive DEC validation are open. PyPI release and the quickstart notebook are on the roadmap. We are publishing now because the framework is already producing numbers that the prior two posts on this site rely on, and because the sooner sophisticated users find the bugs, the sooner the framework can carry the corridor analysis it was built for.

**6. We want users.** The OSSFE community, the systems-code community, and individual researchers with domain expertise on a single account or a single concept are the people who can make this framework better fastest. Pull the code. Run the examples. Tell us where we are wrong.

## References

1. Sorbom, B. N. et al. "ARC: A compact, high-field, fusion nuclear science facility and demonstration power plant with demountable magnets." *Fusion Engineering and Design* 100, 378-405 (2015). DOI
2. Schulte, S. C. et al. "Fusion Reactor Design Studies: Standard Accounts for Cost Estimates." PNL-2648, Pacific Northwest Laboratory (1978). Link
3. Waganer, L. "ARIES Cost Account Documentation." UCSD-CER-13-01 (2013). Link
4. Woodruff, S. "A Costing Framework for Fusion Power Plants." arXiv:2601.21724 (2025). Link
5. CATF Investors Working Group. "Extension of the Fusion Power Plant Costing Standard." arXiv:2602.19389 (2026). Link
6. Kovari, M. et al. "PROCESS': A systems code for fusion power plants - Part 1: Physics." *Fusion Engineering and Design* 89, 3054-3069 (2014). arXiv
7. Meneghini, O. et al. "FUSE: a next-generation framework for integrated design of fusion pilot plants." arXiv:2409.05894 (2024). Link
8. Albajar, F., Johner, J. & Granata, G. "Improved calculation of synchrotron radiation losses in realistic tokamak plasmas." *Nuclear Fusion* 41, 665-678 (2001). DOI
9. Ochs, I. E. et al. "Improving the feasibility of economical proton-boron-11 fusion via alpha channeling with a hybrid fast and thermal proton scheme." *Physical Review E* 106, 055215 (2022). DOI
10. 1cFE. "Fusion's cost floor: what if the core were free?" (2026). Link
11. 1cFE. "Direct energy conversion for fusion: fuel, confinement, and the cost question." (2026). Link
12. 1cFE. "1costingfe: open-source fusion techno-economic model." GitHub