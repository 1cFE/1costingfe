# 1costingfe Design Document

**Date:** 2026-02-18
**Status:** Approved
**Context:** Implements recommendation 8.1 from "Analysis: Woodruff Paper Claims vs Actual Capabilities" — use pyFECONs as intellectual scaffolding, not production code. Build a computationally flexible, JAX-native fusion costing framework.

---

## 1. Purpose

1costingfe is a Python-native fusion power plant costing framework built on JAX. It replaces the PyFECONS adapter in the fusion-tea SysML pipeline and provides capabilities no existing tool offers:

- **Customer-first framing**: Start from net electric power, not fusion power
- **Bidirectional computation**: Forward costing and backcasting (target LCOE to required parameters)
- **Native sensitivity analysis**: Exact gradients via JAX autodiff, not finite differences
- **Cross-concept comparison**: All confinement concepts x all fuel types, ranked by LCOE
- **All architectures and fuels**: MFE (tokamak, stellarator, mirror), IFE (laser, Z-pinch, heavy-ion), MIF (magnetized target, plasma jet) x DT, DD, DHe3, pB11

---

## 2. Architecture: Five-Layer Customer-First Pipeline

```
Layer 1: Customer Requirements (net electric, availability, lifetime, targets)
    |  inverse power balance
Layer 2: Physics (required fusion power, gain, fuel physics)
    |  geometry + component sizing
Layer 3: Engineering (radial build, materials, component specs)
    |  per-account costing
Layer 4: Costs (CAS10-CAS90, COA standard)
    |  financial aggregation
Layer 5: Economics (LCOE, NPV, learning curves)
```

The model's natural forward direction IS the customer's perspective: start from "I want 1 GW net electric" and work downward to determine what that costs. The inverse power balance (net electric to required fusion power) is algebraic, not numerical.

### Why JAX

JAX provides three capabilities that are core to 1costingfe's value:

1. **`jax.grad`**: Exact partial derivatives of LCOE w.r.t. every parameter in one backward pass. Sensitivity analysis and tornado charts are free.
2. **`jax.vmap`**: Write scalar equations, vectorize automatically across Monte Carlo samples or parameter sweeps.
3. **`jax.jit`**: JIT compilation for fast execution of large parameter sweeps.

For backcasting, `jax.jacobian` provides the full Jacobian matrix (how every output responds to every input), enabling constraint propagation without brute-force search. Gradient-based optimization (phase 2) uses these same gradients.

---

## 3. Confinement Taxonomy

Two-level hierarchy: **family** determines power balance, **concept** determines engineering details.

```
ConfinementFamily          ConfinementConcept
MFE (Magnetic)       -->   Tokamak, Stellarator, Mirror/FRC
IFE (Inertial)       -->   Laser-driven, Z-pinch, Heavy-ion
MIF (Magneto-Inertial) --> Magnetized target, Plasma jet
```

### Fuel types

DT, DD, DHe3, pB11 — each with fuel-specific physics (ash/neutron split, secondary burn models).

---

## 4. Layer Details

### Layer 1: Customer Requirements

Inputs:
- `net_electric_mw`: desired net electric output
- `availability`: plant capacity factor
- `lifetime_yr`: plant operational lifetime
- `n_units`: number of reactor units
- `construction_time_yr`: time to build
- `cost_year`: dollar year for outputs

Optional targets (for backcasting):
- `target_lcoe`: desired LCOE in $/MWh
- `target_overnight_cost`: desired overnight capital cost in $/kW

### Layer 2: Physics (Power Balance + Fuel Physics)

**Direction:** Inverse by default (net electric to fusion power), matching the customer-first framing. Forward also available for validation.

**Fuel physics** (from pyFECONs equations):
- DT: 20% charged (alpha), 80% neutron
- DD: semi-catalyzed model with parameterized secondary burn fractions
- DHe3: primary aneutronic + DD side reactions
- pB11: fully aneutronic

**Power balance** per architecture family:
- MFE: recirculating includes coils, cryo, divertor cooling, plasma heating (NBI/ICRF/LHCD)
- IFE: recirculating includes laser drivers, target injection
- MIF: recirculating includes pulsed power charging

Output: `PowerTable` (~25 computed power fields, consumed by all downstream layers).

### Layer 3: Engineering

**Geometry as interface:** Simple geometries (tokamak torus, mirror cylinder, IFE sphere) computed internally. Complex geometries (stellarator 3D, novel ICF) provided externally from fusion-tea SysML or external tools (Paramak, VMEC).

```python
# Geometry dataclass: the interface (volumes, areas, component specs)
# compute_tokamak_geometry()    — built-in
# compute_mirror_geometry()     — built-in
# compute_simple_ife_geometry() — built-in
# Stellarator: always external input
```

**Architecture-specific modules:**
- `mfe/tokamak.py`: identical TF coils, divertor, simple torus
- `mfe/stellarator.py`: unique coils (external geometry), baffles
- `mfe/mirror.py`: solenoid coils, end plugs, cylinder
- `ife/laser.py`: NIF-scaled lasers, spherical chamber
- `ife/zpinch.py`: pulsed power driver, cylindrical chamber
- `ife/heavy_ion.py`: accelerator driver
- `mif/mag_target.py`: liner + plasma gun
- `mif/plasma_jet.py`: converging jets

**Common modules:** blanket, shield, vacuum, primary structure, BOP — shared across all concepts.

**Default engineering parameters:** Each concept x fuel combination has reasonable defaults (YAML files) so cross-concept comparison works out of the box. Fusion-tea SysML bindings override these for specific designs.

### Layer 4: Costs (CAS Accounts)

Each CAS account is a pure JAX function. Follows COA standard (CAS10-CAS90).

**Fuel-dependent configuration matrix:**

| CAS Account | DT | DD | DHe3 | pB11 |
|---|---|---|---|---|
| 220101 Blanket | Breeding (TBR>1.05) | Energy-capture (no breeding) | Minimal (X-ray + ~5% neutron) | Minimal (X-ray only) |
| 220101 Neutron Multiplier | Required (Be/Pb) | Not needed | Not needed | Not needed |
| 220102 Shield | Heavy 1-2m | Mixed 0.5-1m | Light | Minimal |
| 220109 DEC | Optional (unproven) | Optional (unproven) | Optional (unproven) | Optional (unproven) |
| 220112 Isotope Sep | D2O + T purification + Li-6 enrichment | D2O extraction | D2O + He-3 extraction | H-1 purification + B-11 enrichment |
| 220119 Replacements | 5-10 FPY | 10-15 FPY | 30+ FPY | 50+ FPY |
| 2205 Fuel Handling | Full tritium processing (~56 kg/yr) | Small-scale T handling (~5 kg/yr) | He-3 handling | B powder injection |
| CAS23 Steam Turbine | Required (33-40% eta) | Required (33-40% eta) | Required (30-35% eta) | Required (30-35% eta) |
| CAS70 O&M | High (remote handling) | Moderate | Reduced | Low (contact OK) |
| CAS80 Fuel feedstock | D + Li-6: ~$0.2-1.1M/yr | D: ~$0.3-1.5M/yr | D + He-3: infeasible* | H + B-11: ~$3-4M/yr |
| CAS10 Licensing | $5M | $3M | $1M | $0.1M |

*Unless He-3 self-produced from DD.

**CAS accounting note:** DT breeding infrastructure is NOT CAS80. It splits across CAS220101 (blanket capital), CAS220119 (replacement), CAS2205 (tritium processing), CAS70 (O&M). CAS80 is feedstock only.

**Architecture dispatch:** Concept-specific CAS functions (e.g., cas220103_tokamak_coils vs cas220103_lasers) are bound at model construction time, outside the JIT boundary. Inside JIT, the bound function is called directly.

### Layer 5: Economics

Core outputs:
- **LCOE** = (CAS90 + CAS70 + CAS80) / (8760 x p_net x n_units x availability)
- **Overnight capital cost** = (CAS10-CAS60) / (p_net x n_units) in $/kW
- **Capital recovery factor** = f(discount_rate, lifetime)

Future additions:
- NPV / IRR over plant lifetime
- FOAK vs NOAK learning curves
- Cost interpolation with optimism/learning axes (compatible with fusion-tea's bilinear interpolation CalcDef)
- Financing structure (debt/equity split, WACC)

---

## 5. Analysis Layer

### Sensitivity Analysis

Exact gradients via `jax.grad` — partial derivative of LCOE w.r.t. every continuous parameter in one backward pass. Outputs: tornado charts, parameter importance ranking, elasticity.

### Backcasting Phase 1: Constraint Propagation (MVP)

Given a target LCOE and fixed parameters, determine required values for free parameters using the Jacobian. Reports feasibility and binding constraints (e.g., required gain exceeds physical limits).

### Backcasting Phase 2: Gradient-Based Optimization (future)

Full optimization over continuous parameters, respecting physics/engineering constraints. Uses JAX gradients with scipy.optimize or optax.

### Cross-Concept Comparison

Enumerate all concept x fuel combinations with the same Layer 1 inputs, rank by LCOE. Includes per-combination sensitivity. Uses `jax.vmap` to batch runs within each architecture family.

---

## 6. Fusion-Tea Integration

1costingfe replaces the PyFECONS adapter in fusion-tea. The integration point:

1. Fusion-tea extracts parameters from SysML via its AST walker
2. Maps to 1costingfe's typed API (CostModel with concept + fuel)
3. SysML bindings override default engineering parameters at any layer
4. 1costingfe returns per-CAS costs
5. Fusion-tea writes costs back to SysML design

**Parameter override pattern:**
- Layer 1 (customer): always from SysML design
- Layer 2 (physics): override if SysML specifies
- Layer 3 (engineering): override if SysML specifies; defaults otherwise
- Layer 4 (costs): CostingConstants from SysML or built-in
- Layer 5 (economics): financial params from SysML or defaults

A minimal SysML design (concept + fuel + net electric) gets a full cost estimate from defaults. A detailed design overrides specific parameters.

**Cost interpolation compatibility:** 1costingfe exposes optimism and learning as parameters, compatible with fusion-tea's bilinear interpolation CalcDef. `jax.grad` through the learning parameter quantifies the value of R&D investment.

---

## 7. Validation Strategy

**NOT validating against pyFECONs as ground truth.** pyFECONs is a sanity-check reference, not an oracle.

1. **Physics first principles:**
   - Energy conservation: power in = power out + losses (every run)
   - Carnot limit: thermal efficiency < 1 - T_cold/T_hot
   - Lawson criterion: gain is physically achievable for given fuel

2. **Published design studies:**
   - ARIES-ACT, EU-DEMO, ARC: run 1costingfe with same parameters, compare LCOE and per-account costs
   - Expect "same ballpark" (within 2x), not exact match

3. **Internal consistency:**
   - Increasing availability should decrease LCOE proportionally
   - Switching from DT to aneutronic should eliminate breeding blanket cost
   - All fuel types produce steam turbine costs (X-ray/heat on first wall is universal)

4. **Cross-validation with pyFECONs:**
   - Run both with identical inputs as sanity check
   - Document where and why they differ

5. **Gradient correctness:**
   - Finite-difference check: compare `jax.grad` to (f(x+h) - f(x))/h

---

## 8. Package Structure

```
1costingfe/
  pyproject.toml              # uv, JAX dependency
  src/
    costingfe/
      __init__.py
      model.py                # CostModel: top-level API
      types.py                # Enums, dataclasses
      layers/
        customer.py           # Layer 1: requirements
        physics.py            # Layer 2: power balance, fuel physics
        engineering/
          mfe/
            tokamak.py
            stellarator.py
            mirror.py
          ife/
            laser.py
            zpinch.py
            heavy_ion.py
          mif/
            mag_target.py
            plasma_jet.py
          common.py           # blanket, shield, vacuum, BOP
          geometry.py         # Geometry interface + simple solvers
        costs.py              # Layer 4: CAS accounts
        economics.py          # Layer 5: LCOE, NPV
      analysis/
        sensitivity.py        # jax.grad / jax.jacobian wrappers
        backcast.py           # constraint propagation + optimization
        compare.py            # cross-concept comparison
      data/
        defaults/             # default params per concept x fuel (YAML)
      validation.py           # input validation, physics sanity checks
  tests/
    test_power_balance.py
    test_accounts.py
    test_lcoe.py
    test_sensitivity.py
```

---

## 9. Top-Level API

```python
from costingfe import CostModel, ConfinementConcept, Fuel

# Forward: customer requirements to LCOE
model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
result = model.forward(net_electric_mw=1000, availability=0.9, lifetime_yr=40)

# Sensitivity: which parameters matter most?
sens = model.sensitivity(result.params)

# Backcast: what must be true to hit a target?
feasible = model.backcast(target_lcoe=10.0, fixed={"net_electric_mw": 1000})

# Cross-concept comparison
from costingfe import compare_all
ranking = compare_all(net_electric_mw=1000, availability=0.9, lifetime_yr=40)
```

---

## 10. Technology Stack

| Component | Choice | Rationale |
|---|---|---|
| Language | Python 3.10+ | Consistent with 1cFE ecosystem |
| Computational backend | JAX | Autodiff, vmap, JIT |
| Package manager | uv | Consistent with fusion-tea |
| Testing | pytest | Standard |
| Data formats | YAML (defaults), JSON (results) | Human-readable |
| Type checking | Dataclasses + type hints | IDE support, validation |

---

## 11. Key Design Decisions Summary

| Decision | Choice | Rationale |
|---|---|---|
| Starting point | Hybrid: pyFECONs as intellectual scaffolding, not production code | Rec 8.1 from Woodruff analysis |
| Computational backend | JAX | Autodiff for sensitivity/backcasting, vmap for Monte Carlo |
| Architecture | 5-layer customer-first pipeline | Natural for "start from electric power" |
| Confinement taxonomy | Family + Concept (two-level) | Family = power balance, concept = engineering |
| Geometry | Interface + built-in solvers for simple cases | Stellarators and novel ICF get geometry externally |
| Fuel configuration | Matrix of active CAS accounts per fuel | Breeding blanket DT only, DEC always optional, steam turbines universal |
| Backcasting | Phase 1: constraint propagation. Phase 2: optimization | JAX makes both cheap |
| Validation | Physics first principles + published studies | pyFECONs as reference only, not oracle |
| fusion-tea integration | Typed Python API, replaces PyFECONS adapter | SysML overrides defaults at any layer |

---

## References

- "Analysis: Woodruff Paper Claims vs Actual Capabilities" (2026) — motivation and recommendation 8.1
- fusion-tea plan.md — pipeline architecture and PyFECONS adapter design
- fusion-tea/knowledge/research/approved/20260208-fusion-reactor-subsystems-by-fuel-type.md — fuel-dependent configuration matrix
- fusion-tea/knowledge/research/approved/20260211-fusion-fuel-isotope-sourcing.md — fuel cost and supply chain data
- Woodruff (2026), "A Costing Framework for Fusion Power Plants" — COA structure and equations reference
- pyFECONs source code — account structure and power balance equations reference
