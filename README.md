# 1costingfe

JAX-native fusion power plant costing framework. Replaces the PyFECONS adapter in the fusion-tea SysML pipeline with a differentiable, 5-layer customer-first model.

## Install

```bash
pip install -e .
# or with dev dependencies:
pip install -e ".[dev]"
```

Requires Python 3.10+.

## Quick Start

```python
from costingfe import CostModel, ConfinementConcept, Fuel

# Create a model for a DT tokamak
model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)

# Forward costing: customer requirements -> LCOE
result = model.forward(
    net_electric_mw=1000.0,
    availability=0.85,
    lifetime_yr=30,
)

print(f"LCOE: {result.costs.lcoe:.1f} $/MWh")
print(f"Overnight cost: {result.costs.overnight_cost:.0f} $/kW")
print(f"Fusion power: {result.power_table.p_fus:.0f} MW")
```

## Sensitivity Analysis (JAX autodiff)

```python
sens = model.sensitivity(result.params)

# Engineering levers (sorted by |elasticity|)
for k, v in sorted(sens["engineering"].items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {k:25s} {v:+.4f}")

# Financial parameters
for k, v in sens["financial"].items():
    print(f"  {k:25s} {v:+.4f}")
```

Elasticity = (dLCOE/dp) * (p/LCOE) -- dimensionless, comparable across parameters.

## Batch Parameter Sweeps (JAX vmap)

```python
# Sweep blanket thickness from 0.5m to 1.0m
lcoes = model.batch_lcoe(
    {"blanket_t": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]},
    result.params,
)
```

## Cross-Concept Comparison

```python
from costingfe import compare_all

results = compare_all(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
for r in results[:5]:
    print(f"  {r.concept.value:15s} {r.fuel.value:5s} {r.lcoe:6.1f} $/MWh")
```

## Backcasting

```python
from costingfe.analysis.backcast import backcast_single

# What availability achieves 60 $/MWh?
avail = backcast_single(
    model, target_lcoe=60.0, param_name="availability",
    param_range=(0.70, 0.98), base_params=result.params,
)
```

## Fusion-Tea Adapter

```python
from costingfe.adapter import FusionTeaInput, run_costing

inp = FusionTeaInput(
    concept="tokamak",
    fuel="dt",
    net_electric_mw=1000.0,
    availability=0.85,
    lifetime_yr=30,
)
out = run_costing(inp)
# out.lcoe, out.costs (CAS-keyed dict), out.power_table, out.sensitivity
```

## Supported Concepts

| Family | Concept | Key features |
|--------|---------|-------------|
| MFE | `tokamak` | Toroidal confinement, TF/CS/PF coils |
| MFE | `stellarator` | Steady-state, complex 3D coils |
| MFE | `mirror` | Cylindrical, end-loss DEC opportunity |
| IFE | `laser_ife` | Split laser drivers, target factory |
| IFE | `zpinch` | Pulsed power driver |
| IFE | `heavy_ion` | Heavy ion accelerator |
| MIF | `mag_target` | Magnetized target, liner factory |
| MIF | `plasma_jet` | Plasma jet driver |

## Fuels

- `dt` -- Deuterium-Tritium (breeding blanket, heavy shielding)
- `dd` -- Deuterium-Deuterium (no breeding, moderate shielding)
- `dhe3` -- Deuterium-Helium-3 (mostly aneutronic)
- `pb11` -- Proton-Boron-11 (fully aneutronic, minimal shielding)

## Engineering Overrides

Pass any engineering parameter as a keyword argument:

```python
result = model.forward(
    net_electric_mw=1000.0,
    availability=0.85,
    lifetime_yr=30,
    eta_th=0.50,        # Override thermal efficiency
    blanket_t=0.90,     # Thicker blanket
    axis_t=7.0,         # Larger major radius
)
```

See `src/costingfe/data/defaults/` YAML files for all available parameters.

## Tests

```bash
pytest tests/ -v
```

## Architecture

```
Customer Requirements (net_electric_mw, availability, lifetime_yr)
    |
    v
Layer 2: Physics (power balance, inverse for target p_net)
    |
    v
Layer 3: Engineering (radial build -> geometry -> volumes)
    |
    v
Layer 4: Costs (CAS 10-60 accounts, volume-based + power-scaled)
    |
    v
Layer 5: Economics (CAS 70-90, LCOE)
```
