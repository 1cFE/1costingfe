# 1costingFE intro

Post Title: A Differentiable, Open-Source Framework for Fusion Plant Economics
Target Date: April 28, 2026
Status: Drafting
Lead Author: Tal Rubin
Type: Milestone
Workstreams: WS2 TEA Engine
Assets Needed: PyPI alpha release, runnable quickstart.ipynb in examples/, optional CAS22 waterfall figure with elasticities overlaid.
Internal Dependencies: Alpha release on PyPI (pip install 1costingfe resolves); examples/quickstart.ipynb runs end-to-end; paper.tex stable enough to be the canonical methodology reference; generic DT tokamak baseline numbers reproducible from a fresh install.
Risks: Overselling capability. Confusing readers about DEC maturity. Mitigate by the explicit roadmap table in the post and repeated 'what we don't do yet.' False precision on the elasticity numbers: these are illustrative until regenerated from a fresh run.

# Goals of this post

1. Demonstrate current capabilities and features of 1costingFE and give the reader a sense of what the framework makes possible: concept and fuel coverage across all major confinement families, JAX-native autodiff for elasticities and Monte Carlo, and cost overrides at every account level.
2. Recruit sophisticated users to pull the code and engage with us on bugs, missing fuel and concept cases, account-level review, reference cost data, and feature requests. Think: Simon Woodruff, Layla Araiinejad, Jacob Schwartz.
    1. Think distribution to **Open Source Software for Fusion Energy** [https://ossfe.org/](https://ossfe.org/OSSFE_2026/)’s mailing list

## Draft

*Working public title: A Differentiable, Open-Source Framework for Fusion Plant Economics*

*Follows the free core and  DEC posts which both reference and use 1costingfe. Paired with the worked-example post at slot 4.1.*

---

## Opening

Every fusion design paper we read has the same shape. The physics is detailed. The geometry is specified. The economics is a footnote, or missing, or asserted. A 2015 compact tokamak paper [1] gives costs for three components and leaves the balance of plant out of scope. A [recent stellarator preconceptual design](https://arxiv.org/abs/2512.08027) publishes a target LCOE with no breakdown. The few public fusion costing tools are either calibrated to a single confinement family (usually D-T tokamaks) or are parameterized cost walkers rather than concept-aware models.

1cFE's question is whether any fusion concept can reach $0.01/kWh. Answering it requires a costing framework that spans confinement families and fuel cycles, computes sensitivities at scale, and carries its assumptions on the surface where they can be challenged. We wrote one: [1costingFE](https://github.com/1cfe/1costingfe), a JAX-native, open-source costing framework that takes a concept, a fuel cycle, and a target power, and returns a full Code of Accounts Structure breakdown, a closed power balance, and a complete set of sensitivity elasticities from one backward pass.

The code is public. The [methodology paper](https://github.com/1cfe/1costingfe/blob/master/tex/paper.tex) is public. We want users.

## What it is

1costingFE is three modules wired together. The economics module converts a capital cost and a set of financial parameters into a levelized cost per MWh. The physics module converts a fusion power and a set of engineering coefficients into a net electric output, across four fuels (D-T, D-D, D-³He, p-¹¹B) and both steady-state and pulsed concepts. The CAS module carries the cost account structure developed by Schulte and colleagues at Pacific Northwest Laboratory in 1978 [2], adopted by the ARIES program and the Generation IV Economic Modeling Working Group, and implemented in [pyFECONS](https://github.com/woodruff-scientific-ltd/pyfecons) by Woodruff Scientific [3]. We use pyFECONS as a reference implementation and a validation target; our framework differs in three ways that matter for the corridor question:

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

Full derivations are in the [paper](https://github.com/1cfe/1costingfe/blob/master/tex/paper.tex), §2.

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
3. **DEC for steady-state concepts.** Today DEC is off by default for MFE. As our [lower-bound dispatch on fusion's cost floor](https://1cf.energy/blog/fusions-cost-floor/) and [DEC deep dive](#TODO-dec-link) argue, p-¹¹B with direct energy conversion is where the sub-cent corridor sits, and we want a defensible cost model for it.
4. **Integration with [Fusion-TEA](https://github.com/1cfe/fusion-tea).** The SysML v2 pipeline described in our [methodology post](https://1cf.energy/searching-the-fusion-design-space-systematically/) ingests papers, builds system models, and currently emits hand-written 1costingFE scripts. Code generation from SysML to 1costingFE is the next integration target.
5. **Stable release cadence on PyPI.**

## What we'd love feedback on

This is early code. We expect to be wrong in places. In rough order of what helps us most:

**Reference data.** If you have NOAK or FOAK cost numbers for any CAS account, tell us. Magnet costs at contemporary REBCO prices, blanket costs at modern FLiBe or lithium prices, balance of plant from recent fission or combined-cycle projects. Anonymised is fine.

**Missing fuel or concept cases.** Hybrid fuel cycles, muon-catalyzed concepts, mass-accelerated fuel, inertial electrostatic variants. If your concept doesn't fit the framework's abstractions cleanly, we want to hear about it.

**Account-level review.** Pick any single CAS account, read the internal justification, and tell us where we're wrong. Specific is better than general.

**Bugs.** File issues on GitHub. We respond.

The code is at [github.com/1cfe/1costingfe](http://github.com/1cfe/1costingfe). The paper is at [github.com/1cfe/1costingfe/blob/master/tex/paper.tex](http://github.com/1cfe/1costingfe/blob/master/tex/paper.tex). 

## References

1. Sorbom, B. N. et al. "ARC: A compact, high-field, fusion nuclear science facility and demonstration power plant with demountable magnets." *Fusion Engineering and Design* 100, 378 (2015). [DOI](https://doi.org/10.1016/j.fusengdes.2015.07.008)
2. Schulte, S. C. et al. "Fusion Reactor Design Studies: Standard Accounts for Cost Estimates." PNL-2648, Pacific Northwest Laboratory (1978).
3. Woodruff, S. "A Costing Framework for Fusion Power Plants." arXiv:2601.21724 (2026). [Link](https://arxiv.org/abs/2601.21724)
4. Ochs, I. E. et al. "Steady-state relaxation of p-¹¹B plasmas." (2025). *Final citation to be supplied by Tal.*
5. Albajar, F. et al. "Synchrotron radiation loss in tokamak plasmas." *Nuclear Fusion* 41, 665 (2001). [DOI](https://doi.org/10.1088/0029-5515/41/6/301)
6. CATF Investors Working Group. "Assessing the cost of fusion energy." arXiv:2602.19389 (2025). [Link](https://arxiv.org/abs/2602.19389)

---

## External Review

**Reviewers invited**

| Name | Affiliation | Invited | Responded | Acknowledgement accepted |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

**Substantive feedback received**

- 

**Acknowledgement text to appear in the published piece**

> Thanks to [Names] for review and comments on an earlier draft. Any errors are ours.
> 

**Notes on disagreements or suggestions not adopted**

- 

---

## Notes, References, and Assets

**Pre-publication checklist:**

- [ ]  Tal confirms Ochs citation (ref 4)
- [ ]  DEC deep dive link inserted (replacing #TODO-dec-link)
- [x]  Post B link inserted (replacing #TODO-post-b-link)
- [ ]  Elasticity table regenerated from fresh `quickstart.ipynb` run
- [ ]  PyPI alpha tagged and `pip install 1costingfe` verified on Linux and macOS
- [ ]  `examples/quickstart.ipynb` runs end-to-end on a clean environment
- [x]  Lead Author assigned (Tal)
- [ ]  Target Date set

**Assets to produce:**

- [ ]  PyPI alpha release
- [ ]  `examples/quickstart.ipynb` (ten-minute walkthrough)
- [ ]  Optional: CAS22 waterfall figure with elasticities overlaid