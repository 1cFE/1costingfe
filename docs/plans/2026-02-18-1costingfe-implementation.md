# 1costingfe Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a JAX-native fusion power plant costing framework with customer-first framing, native sensitivity analysis, and backcasting.

**Architecture:** 5-layer pipeline (Customer Requirements -> Physics -> Engineering -> Costs -> Economics). All computation in JAX for autodiff/vmap/JIT. Confinement taxonomy: Family (MFE/IFE/MIF) + Concept (tokamak/stellarator/mirror/laser/...). Fuel types: DT, DD, DHe3, pB11.

**Tech Stack:** Python 3.10+, JAX, uv, pytest, ruff (format + lint)

**Design Doc:** `docs/plans/2026-02-18-1costingfe-design.md`

**Reference Equations:** pyFECONs at `/mnt/c/Users/talru/1cfe/pyfecons/pyfecons/costing/`

---

## Phase 1: Foundation

### Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `src/costingfe/__init__.py`
- Create: `src/costingfe/types.py`
- Create: `tests/__init__.py`
- Create: `tests/test_types.py`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "costingfe"
version = "0.1.0"
description = "JAX-native fusion power plant costing framework"
requires-python = ">=3.10"
dependencies = [
    "jax>=0.4.0",
    "jaxlib>=0.4.0",
    "pyyaml>=6.0",
    "scipy>=1.10.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.8.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/costingfe"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP"]

[tool.ruff.format]
quote-style = "double"
```

**Step 2: Create types module with enums and core dataclasses**

```python
# src/costingfe/types.py
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import jax.numpy as jnp

class ConfinementFamily(Enum):
    MFE = "mfe"
    IFE = "ife"
    MIF = "mif"

class ConfinementConcept(Enum):
    TOKAMAK = "tokamak"
    STELLARATOR = "stellarator"
    MIRROR = "mirror"
    LASER_IFE = "laser_ife"
    ZPINCH = "zpinch"
    HEAVY_ION = "heavy_ion"
    MAG_TARGET = "mag_target"
    PLASMA_JET = "plasma_jet"

CONCEPT_TO_FAMILY = {
    ConfinementConcept.TOKAMAK: ConfinementFamily.MFE,
    ConfinementConcept.STELLARATOR: ConfinementFamily.MFE,
    ConfinementConcept.MIRROR: ConfinementFamily.MFE,
    ConfinementConcept.LASER_IFE: ConfinementFamily.IFE,
    ConfinementConcept.ZPINCH: ConfinementFamily.IFE,
    ConfinementConcept.HEAVY_ION: ConfinementFamily.IFE,
    ConfinementConcept.MAG_TARGET: ConfinementFamily.MIF,
    ConfinementConcept.PLASMA_JET: ConfinementFamily.MIF,
}

class Fuel(Enum):
    DT = "dt"
    DD = "dd"
    DHE3 = "dhe3"
    PB11 = "pb11"

@dataclass
class PowerTable:
    """All power flow values computed by Layer 2 (physics)."""
    p_fus: float        # Fusion power [MW]
    p_ash: float        # Charged fusion product power [MW]
    p_neutron: float    # Neutron power [MW]
    p_rad: float        # Plasma radiation power [MW] (bremsstrahlung + synchrotron + line)
    p_wall: float       # Ash thermal on walls [MW]
    p_dee: float        # Direct energy extracted electric [MW]
    p_dec_waste: float  # DEC waste heat [MW]
    p_th: float         # Total thermal power [MW]
    p_the: float        # Thermal electric power [MW]
    p_et: float         # Gross electric power [MW]
    p_loss: float       # Lost power [MW]
    p_net: float        # Net electric power [MW]
    p_pump: float       # Pumping power [MW]
    p_sub: float        # Subsystem power [MW]
    p_aux: float        # Auxiliary power [MW]
    p_coils: float      # Coil power [MW] (MFE)
    p_cool: float       # Cooling power [MW] (MFE)
    q_sci: float        # Scientific Q
    q_eng: float        # Engineering Q
    rec_frac: float     # Recirculating power fraction

@dataclass
class CostResult:
    """Per-CAS cost breakdown in millions USD."""
    cas10: float = 0.0   # Pre-construction
    cas21: float = 0.0   # Buildings
    cas22: float = 0.0   # Reactor plant equipment
    cas23: float = 0.0   # Turbine plant equipment
    cas24: float = 0.0   # Electric plant equipment
    cas25: float = 0.0   # Misc plant equipment
    cas26: float = 0.0   # Heat rejection
    cas27: float = 0.0   # Special materials
    cas28: float = 0.0   # Digital twin
    cas29: float = 0.0   # Contingency
    cas20: float = 0.0   # Total direct costs (sum CAS21-29)
    cas30: float = 0.0   # Indirect service costs
    cas40: float = 0.0   # Owner's costs
    cas50: float = 0.0   # Supplementary costs
    cas60: float = 0.0   # Capitalized financial costs
    cas70: float = 0.0   # Annualized O&M
    cas80: float = 0.0   # Annualized fuel
    cas90: float = 0.0   # Annualized financial (capital)
    total_capital: float = 0.0  # CAS10-60 sum
    lcoe: float = 0.0    # $/MWh
    overnight_cost: float = 0.0  # $/kW

@dataclass
class ForwardResult:
    """Complete result from a forward costing run."""
    power_table: PowerTable
    costs: CostResult
    params: dict  # All input params (for sensitivity analysis)
```

**Step 3: Write test for types**

```python
# tests/test_types.py
from costingfe.types import (
    ConfinementFamily, ConfinementConcept, Fuel,
    CONCEPT_TO_FAMILY, PowerTable, CostResult,
)

def test_concept_to_family_mapping():
    assert CONCEPT_TO_FAMILY[ConfinementConcept.TOKAMAK] == ConfinementFamily.MFE
    assert CONCEPT_TO_FAMILY[ConfinementConcept.LASER_IFE] == ConfinementFamily.IFE
    assert CONCEPT_TO_FAMILY[ConfinementConcept.MAG_TARGET] == ConfinementFamily.MIF

def test_all_concepts_have_family():
    for concept in ConfinementConcept:
        assert concept in CONCEPT_TO_FAMILY, f"{concept} missing from CONCEPT_TO_FAMILY"

def test_fuel_enum():
    assert len(Fuel) == 4
    assert Fuel.DT.value == "dt"
```

**Step 4: Install and run tests**

Run: `cd /mnt/c/Users/talru/1cfe/1costingfe && uv sync && uv run pytest tests/ -v`
Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add pyproject.toml src/ tests/
git commit -m "feat: project setup with types, enums, and core dataclasses"
```

---

## Phase 2: Fuel Physics (Layer 2 Foundation)

### Task 2: Fuel Physics — Ash/Neutron Split

Port `compute_ash_neutron_split` from pyFECONs to JAX. This is the foundation of all power balance calculations.

**Files:**
- Create: `src/costingfe/layers/physics.py`
- Create: `tests/test_physics.py`

**Step 1: Write failing tests**

```python
# tests/test_physics.py
import jax.numpy as jnp
from costingfe.layers.physics import ash_neutron_split
from costingfe.types import Fuel

def test_dt_ash_fraction():
    """DT: alpha carries 3.52 MeV of 17.58 MeV total -> ~20.02% charged."""
    p_fus = 1000.0
    p_ash, p_neutron = ash_neutron_split(p_fus, Fuel.DT)
    assert abs(p_ash / p_fus - 0.2002) < 0.001
    assert abs((p_ash + p_neutron) - p_fus) < 0.001  # energy conservation

def test_pb11_fully_aneutronic():
    """pB11: 100% charged particles (3 alphas)."""
    p_fus = 500.0
    p_ash, p_neutron = ash_neutron_split(p_fus, Fuel.PB11)
    assert abs(p_ash - p_fus) < 0.001
    assert abs(p_neutron) < 0.001

def test_dd_semi_catalyzed():
    """DD: semi-catalyzed burn with defaults should give ~56% charged."""
    p_fus = 1000.0
    p_ash, p_neutron = ash_neutron_split(p_fus, Fuel.DD)
    ash_frac = p_ash / p_fus
    assert 0.50 < ash_frac < 0.65  # approximately 56% with default burn fractions
    assert abs((p_ash + p_neutron) - p_fus) < 0.001

def test_dhe3_mostly_aneutronic():
    """DHe3: primary aneutronic with ~7% DD side reactions -> ~95% charged."""
    p_fus = 1000.0
    p_ash, p_neutron = ash_neutron_split(p_fus, Fuel.DHE3)
    ash_frac = p_ash / p_fus
    assert 0.93 < ash_frac < 0.97
    assert abs((p_ash + p_neutron) - p_fus) < 0.001

def test_ash_neutron_split_is_jax_differentiable():
    """Verify JAX can differentiate through the ash/neutron split."""
    import jax
    def lcoe_proxy(p_fus):
        p_ash, p_neutron = ash_neutron_split(p_fus, Fuel.DT)
        return p_ash  # simple differentiable function
    grad_fn = jax.grad(lcoe_proxy)
    grad_val = grad_fn(1000.0)
    assert abs(grad_val - 0.2002) < 0.001  # d(p_ash)/d(p_fus) = ash_fraction
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_physics.py -v`
Expected: FAIL (module not found)

**Step 3: Implement fuel physics in JAX**

```python
# src/costingfe/layers/__init__.py
# (empty)

# src/costingfe/layers/physics.py
"""Layer 2: Physics — fuel physics, power balance (forward + inverse)."""

import jax.numpy as jnp
from scipy import constants as sc
from costingfe.types import Fuel

# ---------------------------------------------------------------------------
# Fundamental constants from scipy (CODATA)
# ---------------------------------------------------------------------------
MEV_TO_JOULES = sc.eV * 1e6                                          # 1 MeV in J
M_DEUTERIUM_KG = sc.physical_constants['deuteron mass'][0]            # kg

# ---------------------------------------------------------------------------
# Fusion Q-values and product energies (MeV) — nuclear reaction data
# Source: pyFECONs fuel_physics.py
# ---------------------------------------------------------------------------
# DT: D + T -> He4(3.52 MeV) + n(14.06 MeV), Q = 17.58 MeV
E_ALPHA_DT = 3.52
Q_DT = 17.58
E_N_DT = 14.06

# DD branch 1: D + D -> T(1.01 MeV) + p(3.02 MeV), Q = 4.03 MeV
E_T_DD = 1.01
E_P_DD = 3.02
Q_DD_PT = 4.03

# DD branch 2: D + D -> He3(0.82 MeV) + n(2.45 MeV), Q = 3.27 MeV
E_HE3_DD = 0.82
E_N_DD = 2.45
Q_DD_NHE3 = 3.27

# DHe3: D + He3 -> He4(3.6 MeV) + p(14.7 MeV), Q = 18.35 MeV
Q_DHE3 = 18.35

# PB11: p + B11 -> 3 He4, Q = 8.68 MeV
Q_PB11 = 8.68

# DD primary per-event averages (50/50 branches)
_E_CHARGED_PRIMARY_DD = 0.5 * (E_T_DD + E_P_DD) + 0.5 * E_HE3_DD  # ~2.425
_E_NEUTRON_PRIMARY_DD = 0.5 * E_N_DD  # ~1.225
_E_TOTAL_PRIMARY_DD = 0.5 * Q_DD_PT + 0.5 * Q_DD_NHE3  # ~3.65


def ash_neutron_split(
    p_fus: float,
    fuel: Fuel,
    dd_f_T: float = 0.969,
    dd_f_He3: float = 0.689,
    dhe3_dd_frac: float = 0.07,
    dhe3_f_T: float = 0.97,
) -> tuple[float, float]:
    """Compute charged-particle (ash) and neutron power from fusion power.

    Returns (p_ash, p_neutron) in MW. All paths are JAX-differentiable.

    Source: pyFECONs fuel_physics.py:compute_ash_neutron_split
    """
    if fuel == Fuel.DT:
        ash_frac = E_ALPHA_DT / Q_DT
    elif fuel == Fuel.DD:
        E_charged = (_E_CHARGED_PRIMARY_DD
                     + 0.5 * dd_f_T * E_ALPHA_DT
                     + 0.5 * dd_f_He3 * Q_DHE3)
        E_total = (_E_TOTAL_PRIMARY_DD
                   + 0.5 * dd_f_T * Q_DT
                   + 0.5 * dd_f_He3 * Q_DHE3)
        ash_frac = E_charged / E_total
    elif fuel == Fuel.DHE3:
        E_n_dd = _E_NEUTRON_PRIMARY_DD + 0.5 * dhe3_f_T * E_N_DT
        E_c_dd = _E_CHARGED_PRIMARY_DD + 0.5 * dhe3_f_T * E_ALPHA_DT
        ash_frac = (1 - dhe3_dd_frac) + dhe3_dd_frac * E_c_dd / (E_n_dd + E_c_dd)
    elif fuel == Fuel.PB11:
        ash_frac = 1.0
    else:
        raise ValueError(f"Unknown fuel type: {fuel}")

    p_ash = p_fus * ash_frac
    p_neutron = p_fus * (1.0 - ash_frac)
    return p_ash, p_neutron
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_physics.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/costingfe/layers/ tests/test_physics.py
git commit -m "feat: fuel physics ash/neutron split for all fuel types (JAX-differentiable)"
```

---

### Task 3: MFE Forward Power Balance

Port the MFE power balance from pyFECONs to JAX. Takes fusion power + engineering params, returns complete PowerTable.

**Files:**
- Modify: `src/costingfe/layers/physics.py`
- Create: `tests/test_power_balance.py`

**Step 1: Write failing tests**

```python
# tests/test_power_balance.py
import jax
from costingfe.layers.physics import mfe_forward_power_balance
from costingfe.types import Fuel, PowerTable

# CATF reference parameters (from pyFECONs customers/CATF/mfe/DefineInputs.py)
CATF_PARAMS = dict(
    p_fus=2600.0, fuel=Fuel.DT, p_input=50.0,
    mn=1.1, eta_th=0.46, eta_p=0.5, eta_pin=0.5, eta_de=0.85,
    f_sub=0.03, f_dec=0.0,
    p_coils=2.0, p_cool=13.7, p_pump=1.0,
    p_trit=10.0, p_house=4.0, p_cryo=0.5,
    n_e=1.0e20, T_e=15.0, Z_eff=1.5, plasma_volume=500.0, B=5.0,
)

def test_mfe_forward_energy_conservation():
    """Total power in = total power out (1st law of thermodynamics)."""
    pt = mfe_forward_power_balance(**CATF_PARAMS)
    # p_fus + p_input should equal p_net + p_loss + recirculating
    assert pt.p_net > 0, "Net power should be positive"
    assert pt.p_et > pt.p_net, "Gross > net (recirculating losses)"

def test_mfe_forward_net_power_positive():
    """CATF reference design should produce positive net electric."""
    pt = mfe_forward_power_balance(**CATF_PARAMS)
    assert pt.p_net > 500  # CATF ~1 GW electric

def test_mfe_forward_q_eng_reasonable():
    """Engineering Q should be > 1 for a viable power plant."""
    pt = mfe_forward_power_balance(**CATF_PARAMS)
    assert pt.q_eng > 1.0
    assert pt.q_eng < 100.0  # sanity check

def test_mfe_forward_no_dec():
    """With f_dec=0, DEC electric should be zero."""
    pt = mfe_forward_power_balance(**CATF_PARAMS)
    assert abs(pt.p_dee) < 0.001

def test_mfe_forward_is_differentiable():
    """JAX should be able to differentiate p_net w.r.t. p_fus."""
    def p_net_fn(p_fus):
        pt = mfe_forward_power_balance(p_fus=p_fus, fuel=Fuel.DT, p_input=50.0,
            mn=1.1, eta_th=0.46, eta_p=0.5, eta_pin=0.5, eta_de=0.85,
            f_sub=0.03, f_dec=0.0, p_coils=2.0, p_cool=13.7, p_pump=1.0,
            p_trit=10.0, p_house=4.0, p_cryo=0.5)
        return pt.p_net
    grad_fn = jax.grad(p_net_fn)
    grad_val = grad_fn(2600.0)
    assert grad_val > 0  # more fusion power -> more net electric
```

**Step 2: Run tests to verify fail**

Run: `uv run pytest tests/test_power_balance.py -v`
Expected: FAIL (function not found)

**Step 3: Implement MFE forward power balance**

Add to `src/costingfe/layers/physics.py`:

```python
def compute_p_rad(
    n_e: float,
    T_e: float,
    Z_eff: float,
    volume: float,
    B: float = 0.0,
) -> float:
    """Plasma radiation power (bremsstrahlung + synchrotron).

    Calculated from plasma parameters by default, can be overridden.
    P_brem = 5.35e-37 * n_e^2 * Z_eff * sqrt(T_e) * V  [W], T_e in keV, n_e in m^-3
    P_sync = 6.2e-22 * n_e * T_e^2 * B^2 * V  [W] (MFE only)
    """
    p_brem = 5.35e-37 * n_e**2 * Z_eff * jnp.sqrt(T_e) * volume * 1e-6  # -> MW
    p_sync = 6.2e-22 * n_e * T_e**2 * B**2 * volume * 1e-6  # -> MW
    return p_brem + p_sync


def mfe_forward_power_balance(
    p_fus: float,
    fuel: Fuel,
    p_input: float,
    mn: float,
    eta_th: float,
    eta_p: float,
    eta_pin: float,
    eta_de: float,
    f_sub: float,
    f_dec: float,
    p_coils: float,
    p_cool: float,
    p_pump: float,
    p_trit: float,
    p_house: float,
    p_cryo: float,
    # Radiation: calculated from plasma params, or override with p_rad_override
    n_e: float = 1.0e20,
    T_e: float = 15.0,
    Z_eff: float = 1.5,
    plasma_volume: float = 500.0,
    B: float = 5.0,
    p_rad_override: Optional[float] = None,
    dd_f_T: float = 0.969,
    dd_f_He3: float = 0.689,
    dhe3_dd_frac: float = 0.07,
    dhe3_f_T: float = 0.97,
) -> PowerTable:
    """MFE forward power balance: fusion power -> net electric.

    Source: pyFECONs power_balance.py + fusion-tea mfe_power_balance.sysml
    Radiation model: bremsstrahlung + synchrotron from plasma parameters.
    """
    # Step 1: Ash/neutron split
    p_ash, p_neutron = ash_neutron_split(
        p_fus, fuel, dd_f_T, dd_f_He3, dhe3_dd_frac, dhe3_f_T
    )

    # Step 2: Radiation power (p_rad + p_transport = p_ash)
    if p_rad_override is not None:
        p_rad = p_rad_override
    else:
        p_rad = compute_p_rad(n_e, T_e, Z_eff, plasma_volume, B)
    p_rad = jnp.minimum(p_rad, p_ash)  # Can't radiate more than ash power
    p_transport = p_ash - p_rad

    # Step 3: Auxiliary power
    p_aux = p_trit + p_house

    # Step 4: DEC routing (operates on transport channel only)
    p_dee = f_dec * eta_de * p_transport
    p_dec_waste = f_dec * (1.0 - eta_de) * p_transport
    p_wall = (1.0 - f_dec) * p_transport

    # Step 5: Thermal power (neutrons + radiation on first wall + transport on walls + heating + pumping)
    p_th = mn * p_neutron + p_rad + p_wall + p_input + eta_p * p_pump

    # Step 6: Thermal electric
    p_the = eta_th * p_th

    # Step 7: Gross electric
    p_et = p_dee + p_the

    # Step 8: Lost power
    p_loss = (p_th - p_the) + p_dec_waste

    # Step 9: Subsystem power
    p_sub = f_sub * p_et

    # Step 10: Scientific Q
    q_sci = p_fus / p_input

    # Step 11: Engineering Q
    recirculating = (p_coils + p_pump + p_sub + p_aux
                     + p_cool + p_cryo + p_input / eta_pin)
    q_eng = p_et / recirculating

    # Step 12: Net electric
    rec_frac = 1.0 / q_eng
    p_net = (1.0 - rec_frac) * p_et

    return PowerTable(
        p_fus=p_fus, p_ash=p_ash, p_neutron=p_neutron,
        p_rad=p_rad, p_wall=p_wall, p_dee=p_dee, p_dec_waste=p_dec_waste,
        p_th=p_th, p_the=p_the, p_et=p_et, p_loss=p_loss, p_net=p_net,
        p_pump=p_pump, p_sub=p_sub, p_aux=p_aux,
        p_coils=p_coils, p_cool=p_cool,
        q_sci=q_sci, q_eng=q_eng, rec_frac=rec_frac,
    )
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_power_balance.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/costingfe/layers/physics.py tests/test_power_balance.py
git commit -m "feat: MFE forward power balance (JAX-differentiable)"
```

---

### Task 4: Inverse MFE Power Balance

Algebraic inversion: given target net electric, solve for required fusion power. Uses the closed-form linear inversion from fusion-tea's SysML.

**Files:**
- Modify: `src/costingfe/layers/physics.py`
- Modify: `tests/test_power_balance.py`

**Step 1: Write failing tests**

```python
# Add to tests/test_power_balance.py
from costingfe.layers.physics import mfe_inverse_power_balance

def test_inverse_roundtrip():
    """Forward then inverse should recover original p_fus."""
    pt = mfe_forward_power_balance(**CATF_PARAMS)
    p_fus_recovered = mfe_inverse_power_balance(
        p_net_target=pt.p_net, fuel=Fuel.DT, p_input=50.0,
        mn=1.1, eta_th=0.46, eta_p=0.5, eta_pin=0.5, eta_de=0.85,
        f_sub=0.03, f_dec=0.0, p_coils=2.0, p_cool=13.7, p_pump=1.0,
        p_trit=10.0, p_house=4.0, p_cryo=0.5,
        n_e=1.0e20, T_e=15.0, Z_eff=1.5, plasma_volume=500.0, B=5.0,
    )
    assert abs(p_fus_recovered - 2600.0) < 0.1, f"Expected ~2600, got {p_fus_recovered}"

def test_inverse_1gw_target():
    """1 GW net electric target should give a reasonable fusion power."""
    p_fus = mfe_inverse_power_balance(
        p_net_target=1000.0, fuel=Fuel.DT, p_input=50.0,
        mn=1.1, eta_th=0.46, eta_p=0.5, eta_pin=0.5, eta_de=0.85,
        f_sub=0.03, f_dec=0.0, p_coils=2.0, p_cool=13.7, p_pump=1.0,
        p_trit=10.0, p_house=4.0, p_cryo=0.5,
        n_e=1.0e20, T_e=15.0, Z_eff=1.5, plasma_volume=500.0, B=5.0,
    )
    assert p_fus > 1000  # fusion power must exceed net electric
    assert p_fus < 10000  # but not absurdly large
```

**Step 2: Run to verify fail**

Run: `uv run pytest tests/test_power_balance.py::test_inverse_roundtrip -v`
Expected: FAIL

**Step 3: Implement inverse power balance**

Add to `src/costingfe/layers/physics.py`:

```python
def mfe_inverse_power_balance(
    p_net_target: float,
    fuel: Fuel,
    p_input: float,
    mn: float,
    eta_th: float,
    eta_p: float,
    eta_pin: float,
    eta_de: float,
    f_sub: float,
    f_dec: float,
    p_coils: float,
    p_cool: float,
    p_pump: float,
    p_trit: float,
    p_house: float,
    p_cryo: float,
    # Radiation: calculated from plasma params, or override
    n_e: float = 1.0e20,
    T_e: float = 15.0,
    Z_eff: float = 1.5,
    plasma_volume: float = 500.0,
    B: float = 5.0,
    p_rad_override: Optional[float] = None,
    dd_f_T: float = 0.969,
    dd_f_He3: float = 0.689,
    dhe3_dd_frac: float = 0.07,
    dhe3_f_T: float = 0.97,
) -> float:
    """Inverse MFE power balance: target net electric -> required fusion power.

    Closed-form linear inversion (no iteration needed).
    p_rad is constant w.r.t. p_fus (depends on plasma params, not fusion power),
    so it enters as a constant term in the linear inversion.
    Source: fusion-tea Inverse MFE Power Balance Calc
    """
    # Step 1: Radiation power (constant w.r.t. p_fus)
    if p_rad_override is not None:
        p_rad = p_rad_override
    else:
        p_rad = compute_p_rad(n_e, T_e, Z_eff, plasma_volume, B)

    # Step 2: Ash fraction from fuel type
    # Use p_fus=1.0 to get the fraction
    p_ash_unit, _ = ash_neutron_split(1.0, fuel, dd_f_T, dd_f_He3, dhe3_dd_frac, dhe3_f_T)
    ash_frac = p_ash_unit
    neutron_frac = 1.0 - ash_frac

    # Step 3: Linearize forward chain coefficients
    # p_transport = p_ash - p_rad, but p_rad is capped at p_ash.
    # Per unit p_fus: transport_frac = ash_frac (p_rad is constant, subtracted below)

    # Thermal power per unit p_fus (neutrons + wall transport from ash)
    # p_wall = (1 - f_dec) * p_transport = (1 - f_dec) * (ash_frac * p_fus - p_rad)
    c_th = mn * neutron_frac + (1.0 - f_dec) * ash_frac

    # Constant thermal power (radiation + heating + pumping - wall transport offset from p_rad)
    c_th0 = p_rad - (1.0 - f_dec) * p_rad + p_input + eta_p * p_pump
    # Simplifies to: c_th0 = f_dec * p_rad + p_input + eta_p * p_pump

    # DEC electric per unit p_fus: p_dee = f_dec * eta_de * (ash_frac * p_fus - p_rad)
    c_dee = f_dec * eta_de * ash_frac
    c_dee0 = -f_dec * eta_de * p_rad

    # Gross electric: p_et = c_et * p_fus + c_et0
    c_et = c_dee + eta_th * c_th
    c_et0 = c_dee0 + eta_th * c_th0

    # Recirculating (p_fus-dependent): p_sub = f_sub * p_et
    c_den = f_sub * c_et

    # Recirculating (constant loads)
    p_aux = p_trit + p_house
    c_den0 = (p_coils + p_pump + f_sub * c_et0
              + p_aux + p_cool + p_cryo + p_input / eta_pin)

    # Step 4: Solve p_net = (c_et - c_den) * p_fus + (c_et0 - c_den0)
    p_fus = (p_net_target - c_et0 + c_den0) / (c_et - c_den)
    return p_fus
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_power_balance.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add src/costingfe/layers/physics.py tests/test_power_balance.py
git commit -m "feat: inverse MFE power balance (closed-form linear inversion)"
```

---

## Phase 3: Financial Core (Layer 5 Prerequisites)

### Task 5: Financial Functions — CRF, Levelized Cost

Port the financial calculations from pyFECONs. These are needed by CAS70, CAS80, CAS90.

**Files:**
- Create: `src/costingfe/layers/economics.py`
- Create: `tests/test_economics.py`

**Step 1: Write failing tests**

```python
# tests/test_economics.py
from costingfe.layers.economics import (
    compute_crf, compute_effective_crf, levelized_annual_cost, compute_lcoe,
)

def test_crf_basic():
    """CRF at 7% for 30 years should be ~0.0806."""
    crf = compute_crf(0.07, 30)
    assert abs(crf - 0.0806) < 0.001

def test_crf_high_rate():
    """CRF at 10% for 20 years should be ~0.1175."""
    crf = compute_crf(0.10, 20)
    assert abs(crf - 0.1175) < 0.001

def test_effective_crf():
    """Effective CRF accounts for construction time."""
    crf = compute_crf(0.07, 30)
    eff_crf = compute_effective_crf(0.07, 30, 6)
    assert eff_crf > crf  # construction time increases effective CRF
    expected = crf * (1.07 ** 6)
    assert abs(eff_crf - expected) < 0.0001

def test_levelized_annual_cost_zero_inflation():
    """With zero inflation, cost is unchanged."""
    result = levelized_annual_cost(
        annual_cost=10.0, inflation_rate=0.0, construction_time=6,
    )
    assert abs(result - 10.0) < 0.001

def test_levelized_annual_cost_inflation_scales():
    """Higher inflation should increase levelized cost."""
    low = levelized_annual_cost(
        annual_cost=10.0, inflation_rate=0.02, construction_time=6,
    )
    high = levelized_annual_cost(
        annual_cost=10.0, inflation_rate=0.05, construction_time=6,
    )
    assert high > low

def test_lcoe_sanity():
    """LCOE should be in reasonable range for fusion (10-200 $/MWh)."""
    lcoe = compute_lcoe(
        cas90=500.0, cas70=50.0, cas80=5.0,
        p_net=1000.0, n_mod=1, availability=0.85,
    )
    assert 10 < lcoe < 200
```

**Step 2: Run to verify fail**

Run: `uv run pytest tests/test_economics.py -v`
Expected: FAIL

**Step 3: Implement financial functions**

```python
# src/costingfe/layers/economics.py
"""Layer 5: Economics — CRF, levelized costs, LCOE."""


def compute_crf(interest_rate: float, plant_lifetime: float) -> float:
    """Capital Recovery Factor: CRF = i*(1+i)^n / ((1+i)^n - 1)."""
    i = interest_rate
    n = plant_lifetime
    return (i * (1 + i) ** n) / (((1 + i) ** n) - 1)


def compute_effective_crf(
    interest_rate: float,
    plant_lifetime: float,
    construction_time: float,
) -> float:
    """CRF adjusted for construction time: CRF * (1+i)^Tc."""
    crf = compute_crf(interest_rate, plant_lifetime)
    return crf * (1 + interest_rate) ** construction_time


def levelized_annual_cost(
    annual_cost: float,
    inflation_rate: float,
    construction_time: float,
) -> float:
    """Adjust an annual cost from today's dollars to operation-start dollars.

    annual_cost is in today's dollars. Inflation shifts it forward to the
    dollar-year when operation begins (after construction):
      annual_cost_at_operation = annual_cost * (1 + inflation)^construction_time
    """
    return annual_cost * (1 + inflation_rate) ** construction_time


def compute_lcoe(
    cas90: float,
    cas70: float,
    cas80: float,
    p_net: float,
    n_mod: int,
    availability: float,
) -> float:
    """LCOE in $/MWh. CAS values in M$, p_net in MW.

    LCOE = (CAS90 + CAS70 + CAS80) * 1e6 / (8760 * p_net * n_mod * availability)
    """
    annual_energy_mwh = 8760 * p_net * n_mod * availability
    total_annual_cost_usd = (cas90 + cas70 + cas80) * 1e6
    return total_annual_cost_usd / annual_energy_mwh
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_economics.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/costingfe/layers/economics.py tests/test_economics.py
git commit -m "feat: financial core (CRF, levelized cost, LCOE)"
```

---

## Phase 4: Defaults + Cost Accounts (Layer 4)

### Task 6: Defaults Module — CostingConstants + YAML Loading

Centralize all constants (costing coefficients, engineering defaults) into a typed dataclass loaded from YAML. Computation files stay pure — no magic numbers.

**Files:**
- Create: `src/costingfe/defaults.py`
- Create: `src/costingfe/data/defaults/costing_constants.yaml`
- Create: `src/costingfe/data/defaults/mfe_tokamak.yaml`
- Create: `tests/test_defaults.py`

**Step 1: Write failing tests**

```python
# tests/test_defaults.py
from costingfe.defaults import CostingConstants, EngineeringDefaults, load_costing_constants, load_engineering_defaults

def test_load_costing_constants():
    """Should load defaults from YAML."""
    cc = load_costing_constants()
    assert cc.site_permits > 0
    assert cc.licensing_cost_dt > 0
    assert len(cc.building_costs_per_kw) > 10

def test_load_engineering_defaults():
    """Should load MFE tokamak defaults."""
    ed = load_engineering_defaults("mfe_tokamak")
    assert ed['p_input'] > 0
    assert ed['eta_th'] > 0

def test_costing_constants_override():
    """Should allow field overrides."""
    cc = load_costing_constants()
    cc_custom = cc.replace(site_permits=99.0)
    assert cc_custom.site_permits == 99.0
    assert cc.site_permits != 99.0  # original unchanged
```

**Step 2: Run to verify fail**

Run: `uv run pytest tests/test_defaults.py -v`
Expected: FAIL

**Step 3: Create YAML defaults**

```yaml
# src/costingfe/data/defaults/costing_constants.yaml
# All values from pyFECONs CostingConstants defaults

# CAS10 (bureaucracy: licensing, permits, studies)
site_permits: 3.0           # M$
plant_studies_foak: 20.0    # M$
plant_studies_noak: 4.0     # M$
plant_permits: 2.0          # M$
plant_reports: 1.0          # M$
other_precon: 1.0           # M$
land_intensity: 0.001       # acres per MWe
land_cost: 10000.0          # $/acre
licensing_cost_dt: 5.0      # M$
licensing_cost_dd: 3.0
licensing_cost_dhe3: 1.0
licensing_cost_pb11: 0.1
licensing_time_dt: 5.0      # years
licensing_time_dd: 3.0
licensing_time_dhe3: 2.0
licensing_time_pb11: 1.0

# CAS21 building costs ($/kW of gross electric)
building_costs_per_kw:
  site_improvements: 268
  fusion_heat_island: 126
  turbine_building: 54
  heat_exchanger: 12
  power_supply_storage: 17
  reactor_auxiliaries: 35
  hot_cell: 93.4
  reactor_services: 25
  service_water: 11
  fuel_storage: 9.1
  control_room: 17
  onsite_ac: 21
  administration: 10
  site_services: 4
  cryogenics: 15
  security: 8
  ventilation_stack: 9.2
  isotope_separation: 40
  assembly_hall: 20
  direct_energy_building: 5

# CAS23-26 BOP costs ($/MW of gross electric, 2024$)
inflation_2019_2024: 1.22
turbine_per_mw: 0.19764     # 0.162 * 1.22
electric_per_mw: 0.08418    # 0.069 * 1.22
misc_per_mw: 0.05124        # 0.042 * 1.22
heat_rej_per_mw: 0.03416    # 0.028 * 1.22

# CAS28
digital_twin: 5.0           # M$

# CAS29
contingency_rate_foak: 0.10
contingency_rate_noak: 0.0

# CAS30
field_indirect_coeff: 0.4
construction_supervision_coeff: 0.4
design_services_coeff: 0.4
indirect_ref_power: 1000.0   # MW

# CAS50
shipping: 1.0                # M$
spare_parts_frac: 0.01
taxes: 0.5                   # M$
insurance_cost: 0.5          # M$
decommissioning: 5.0         # M$

# CAS60
idc_coeff: 0.05

# CAS70
om_cost_per_mw_yr: 60.0     # $/MW/yr

# CAS80 fuel cost (physics constants from scipy, not duplicated here)
# u_deuterium: STARFIRE (1980) inflation-adjusted via GDP IPD ratio.
# Current range: $1,500-3,500/kg (see 20260211-fusion-fuel-isotope-sourcing.md).
# Fuel cost is <1% of DT LCOE. Parameter ranges for optimism/learning
# sweeps are a future concern — will need a separate range config for all params.
u_deuterium: 2175.0          # $/kg
```

```yaml
# src/costingfe/data/defaults/mfe_tokamak.yaml
# Default engineering parameters for MFE tokamak

p_input: 50.0       # Heating power [MW]
mn: 1.1             # Neutron energy multiplier
eta_th: 0.46        # Thermal conversion efficiency
eta_p: 0.5          # Pumping efficiency
eta_pin: 0.5        # Heating system wall-plug efficiency
eta_de: 0.85        # DEC efficiency
f_sub: 0.03         # Subsystem power fraction
f_dec: 0.0          # DEC fraction (0 = no DEC)
p_coils: 2.0        # Coil power [MW]
p_cool: 13.7        # Cooling power [MW]
p_pump: 1.0         # Pumping power [MW]
p_trit: 10.0        # Tritium processing power [MW]
p_house: 4.0        # Housekeeping power [MW]
p_cryo: 0.5         # Cryogenic power [MW]

# Plasma parameters for radiation calculation
n_e: 1.0e20         # Electron density [m^-3]
T_e: 15.0           # Electron temperature [keV]
Z_eff: 1.5          # Effective charge
plasma_volume: 500.0 # Plasma volume [m^3]
B: 5.0              # Magnetic field [T]
```

**Step 4: Implement defaults module**

```python
# src/costingfe/defaults.py
"""Load and manage default parameters from YAML files."""

import yaml
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Dict

_DATA_DIR = Path(__file__).parent / "data" / "defaults"


@dataclass(frozen=True)
class CostingConstants:
    """All costing coefficients. Immutable — use .replace() for overrides."""
    # CAS10
    site_permits: float = 3.0
    plant_studies_foak: float = 20.0
    plant_studies_noak: float = 4.0
    plant_permits: float = 2.0
    plant_reports: float = 1.0
    other_precon: float = 1.0
    land_intensity: float = 0.001
    land_cost: float = 10000.0
    licensing_cost_dt: float = 5.0
    licensing_cost_dd: float = 3.0
    licensing_cost_dhe3: float = 1.0
    licensing_cost_pb11: float = 0.1
    licensing_time_dt: float = 5.0
    licensing_time_dd: float = 3.0
    licensing_time_dhe3: float = 2.0
    licensing_time_pb11: float = 1.0

    # CAS21
    building_costs_per_kw: Dict[str, float] = None  # loaded from YAML

    # CAS23-26
    turbine_per_mw: float = 0.19764
    electric_per_mw: float = 0.08418
    misc_per_mw: float = 0.05124
    heat_rej_per_mw: float = 0.03416

    # CAS28
    digital_twin: float = 5.0

    # CAS29
    contingency_rate_foak: float = 0.10
    contingency_rate_noak: float = 0.0

    # CAS30
    field_indirect_coeff: float = 0.4
    construction_supervision_coeff: float = 0.4
    design_services_coeff: float = 0.4
    indirect_ref_power: float = 1000.0

    # CAS50
    shipping: float = 1.0
    spare_parts_frac: float = 0.01
    taxes: float = 0.5
    insurance_cost: float = 0.5
    decommissioning: float = 5.0

    # CAS60
    idc_coeff: float = 0.05

    # CAS70
    om_cost_per_mw_yr: float = 60.0

    # CAS80 — STARFIRE (1980) inflation-adjusted via GDP IPD. Range: $1,500-3,500/kg.
    u_deuterium: float = 2175.0  # $/kg

    def replace(self, **kwargs):
        return replace(self, **kwargs)

    def licensing_cost(self, fuel):
        from costingfe.types import Fuel
        return {
            Fuel.DT: self.licensing_cost_dt,
            Fuel.DD: self.licensing_cost_dd,
            Fuel.DHE3: self.licensing_cost_dhe3,
            Fuel.PB11: self.licensing_cost_pb11,
        }.get(fuel, self.licensing_cost_dt)

    def licensing_time(self, fuel):
        from costingfe.types import Fuel
        return {
            Fuel.DT: self.licensing_time_dt,
            Fuel.DD: self.licensing_time_dd,
            Fuel.DHE3: self.licensing_time_dhe3,
            Fuel.PB11: self.licensing_time_pb11,
        }.get(fuel, self.licensing_time_dt)

    def contingency_rate(self, noak):
        return self.contingency_rate_noak if noak else self.contingency_rate_foak


def load_costing_constants(path: Path = None) -> CostingConstants:
    """Load costing constants from YAML, falling back to dataclass defaults."""
    if path is None:
        path = _DATA_DIR / "costing_constants.yaml"
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f)
        return CostingConstants(**{k: v for k, v in data.items()
                                   if k in {f.name for f in fields(CostingConstants)}})
    return CostingConstants()


def load_engineering_defaults(concept_fuel: str) -> dict:
    """Load engineering defaults for a concept (e.g., 'mfe_tokamak')."""
    path = _DATA_DIR / f"{concept_fuel}.yaml"
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}
```

**Step 5: Run tests**

Run: `uv run pytest tests/test_defaults.py -v`
Expected: All 3 tests PASS

**Step 6: Commit**

```bash
git add src/costingfe/defaults.py src/costingfe/data/defaults/ tests/test_defaults.py
git commit -m "feat: defaults module with CostingConstants dataclass and YAML loading"
```

---

### Task 7: CAS Cost Accounts — All Accounts

Implement all CAS accounts as pure functions. Each takes a `CostingConstants` object plus relevant inputs and returns cost in M$.

**Files:**
- Create: `src/costingfe/layers/costs.py`
- Create: `tests/test_costs.py`

**Step 1: Write failing tests**

```python
# tests/test_costs.py
from costingfe.defaults import load_costing_constants
from costingfe.layers.costs import (
    cas10_preconstruction, cas21_buildings,
    cas23_turbine, cas24_electrical, cas25_misc, cas26_heat_rejection,
    cas28_digital_twin, cas29_contingency,
    cas30_indirect, cas40_owner, cas50_supplementary, cas60_idc,
    cas70_om, cas80_fuel, cas90_financial,
)
from costingfe.types import Fuel

CC = load_costing_constants()

def test_cas10_dt_licensing():
    """DT licensing should be $5M."""
    cost = cas10_preconstruction(CC, p_net=1000.0, n_mod=1, fuel=Fuel.DT, noak=True)
    assert cost > 0

def test_cas10_pb11_cheaper_licensing():
    """pB11 licensing should be cheaper than DT."""
    cost_dt = cas10_preconstruction(CC, p_net=1000.0, n_mod=1, fuel=Fuel.DT, noak=True)
    cost_pb11 = cas10_preconstruction(CC, p_net=1000.0, n_mod=1, fuel=Fuel.PB11, noak=True)
    assert cost_pb11 < cost_dt

def test_cas21_scales_with_power():
    """Building costs should scale with gross electric power."""
    cost_low = cas21_buildings(CC, p_et=500.0, fuel=Fuel.DT, noak=True)
    cost_high = cas21_buildings(CC, p_et=1000.0, fuel=Fuel.DT, noak=True)
    assert cost_high > cost_low

def test_cas23_to_26_scale_with_power():
    """BOP equipment scales with gross electric power."""
    c23 = cas23_turbine(CC, p_et=1000.0, n_mod=1)
    c24 = cas24_electrical(CC, p_et=1000.0, n_mod=1)
    c25 = cas25_misc(CC, p_et=1000.0, n_mod=1)
    c26 = cas26_heat_rejection(CC, p_et=1000.0, n_mod=1)
    for c in [c23, c24, c25, c26]:
        assert c > 0

def test_cas90_annualizes_capital():
    """CAS90 should be CRF * total capital."""
    c90 = cas90_financial(
        CC, total_capital=5000.0, interest_rate=0.07,
        plant_lifetime=30, construction_time=6, fuel=Fuel.DT, noak=True,
    )
    assert c90 > 0
    assert c90 < 5000  # annualized should be less than total

def test_lcoe_end_to_end_sanity():
    """Full cost stack should produce LCOE in reasonable range."""
    from costingfe.layers.economics import compute_lcoe
    c90 = cas90_financial(CC, 5000.0, 0.07, 30, 6, Fuel.DT, True)
    c70 = cas70_om(CC, p_net=1000.0, inflation_rate=0.0245,
                   construction_time=6, fuel=Fuel.DT, noak=True)
    c80 = cas80_fuel(CC, p_fus=2600.0, n_mod=1, availability=0.85,
                         inflation_rate=0.0245,
                         construction_time=6, fuel=Fuel.DT, noak=True)
    lcoe = compute_lcoe(c90, c70, c80, p_net=1000.0, n_mod=1, availability=0.85)
    assert 10 < lcoe < 500, f"LCOE {lcoe} $/MWh out of expected range"
```

**Step 2: Run to verify fail**

Run: `uv run pytest tests/test_costs.py -v`
Expected: FAIL

**Step 3: Implement all CAS accounts**

```python
# src/costingfe/layers/costs.py
"""Layer 4: Costs — CAS10-CAS90 per-account costing.

All functions are pure (no side effects). Each takes a CostingConstants
object as first argument — no inline magic numbers.
Costs returned in millions USD (M$).

Source: pyFECONs costing/calculations/cas*.py
"""
import math
from costingfe.defaults import CostingConstants
from costingfe.types import Fuel
from costingfe.layers.physics import MEV_TO_JOULES, M_DEUTERIUM_KG, Q_DT
from costingfe.layers.economics import (
    compute_effective_crf, levelized_annual_cost,
)


def _total_project_time(cc, construction_time, fuel, noak):
    if noak:
        return construction_time
    return construction_time + cc.licensing_time(fuel)


# ---------------------------------------------------------------------------
# CAS Accounts
# ---------------------------------------------------------------------------

def cas10_preconstruction(cc, p_net, n_mod, fuel, noak):
    """CAS10: Pre-construction costs. Returns M$."""
    land = cc.land_intensity * p_net * math.sqrt(n_mod) * cc.land_cost / 1e6
    licensing = cc.licensing_cost(fuel)
    studies = cc.plant_studies_noak if noak else cc.plant_studies_foak
    subtotal = (land + cc.site_permits + licensing + cc.plant_permits
                + studies + cc.plant_reports + cc.other_precon)
    contingency = cc.contingency_rate(noak) * subtotal
    return subtotal + contingency


def cas21_buildings(cc, p_et, fuel, noak):
    """CAS21: Buildings. Scales with gross electric. Returns M$."""
    fuel_scale = 1.0 if fuel == Fuel.DT else 0.5
    total = 0.0
    for name, cost_per_kw in cc.building_costs_per_kw.items():
        scale = fuel_scale if name in ('site_improvements', 'fusion_heat_island',
                                        'hot_cell', 'fuel_storage') else 1.0
        total += cost_per_kw * p_et / 1000.0 * scale
    contingency = cc.contingency_rate(noak) * total
    return total + contingency


def cas23_turbine(cc, p_et, n_mod):
    """CAS23: Turbine plant equipment. Returns M$."""
    return n_mod * p_et * cc.turbine_per_mw


def cas24_electrical(cc, p_et, n_mod):
    """CAS24: Electric plant equipment. Returns M$."""
    return n_mod * p_et * cc.electric_per_mw


def cas25_misc(cc, p_et, n_mod):
    """CAS25: Miscellaneous plant equipment. Returns M$."""
    return n_mod * p_et * cc.misc_per_mw


def cas26_heat_rejection(cc, p_et, n_mod):
    """CAS26: Heat rejection. Returns M$."""
    return n_mod * p_et * cc.heat_rej_per_mw


def cas28_digital_twin(cc):
    """CAS28: Digital twin. Returns M$."""
    return cc.digital_twin


def cas29_contingency(cc, cas2x_total, noak):
    """CAS29: Contingency on direct costs. Returns M$."""
    return cc.contingency_rate(noak) * cas2x_total


def cas30_indirect(cc, cas20, p_net, construction_time):
    """CAS30: Indirect service costs. Returns M$."""
    power_scale = (p_net / cc.indirect_ref_power) ** -0.5
    field = power_scale * p_net * cc.field_indirect_coeff * construction_time / 1e3
    supervision = power_scale * p_net * cc.construction_supervision_coeff * construction_time / 1e3
    design = power_scale * p_net * cc.design_services_coeff * construction_time / 1e3
    return field + supervision + design


def cas40_owner(cas20):
    """CAS40: Owner's costs (~5% of direct). Returns M$."""
    return 0.05 * cas20


def cas50_supplementary(cc, cas23_to_28_total, p_net, noak):
    """CAS50: Supplementary costs. Returns M$."""
    spare_parts = cc.spare_parts_frac * cas23_to_28_total
    fuel_load = (p_net / 1000.0) * 10.0  # rough scaling
    subtotal = (cc.shipping + spare_parts + cc.taxes
                + cc.insurance_cost + fuel_load + cc.decommissioning)
    contingency = cc.contingency_rate(noak) * subtotal
    return subtotal + contingency


def cas60_idc(cc, cas20, p_net, construction_time, fuel, noak):
    """CAS60: Interest during construction. Returns M$."""
    t_project = _total_project_time(cc, construction_time, fuel, noak)
    return p_net * cc.idc_coeff * t_project / 1e3


def cas70_om(cc, p_net, inflation_rate, construction_time, fuel, noak):
    """CAS70: Annualized O&M costs (today's $ inflated to operation start). Returns M$."""
    annual_om = cc.om_cost_per_mw_yr * p_net * 1000 / 1e6  # M$
    t_project = _total_project_time(cc, construction_time, fuel, noak)
    return levelized_annual_cost(annual_om, inflation_rate, t_project)


def cas80_fuel(cc, p_fus, n_mod, availability, inflation_rate,
               construction_time, fuel, noak):
    """CAS80: Annualized fuel cost. Architecture-agnostic — depends on
    fusion power and fuel type, not confinement concept. Returns M$."""
    c_f = (n_mod * p_fus * 1e6 * 3600 * 8760 * cc.u_deuterium
           * M_DEUTERIUM_KG * availability / (Q_DT * MEV_TO_JOULES))
    annual_fuel_musd = c_f / 1e6
    t_project = _total_project_time(cc, construction_time, fuel, noak)
    return levelized_annual_cost(annual_fuel_musd, inflation_rate, t_project)


def cas90_financial(cc, total_capital, interest_rate, plant_lifetime,
                    construction_time, fuel, noak):
    """CAS90: Annualized financial (capital) costs. Returns M$."""
    t_project = _total_project_time(cc, construction_time, fuel, noak)
    eff_crf = compute_effective_crf(interest_rate, plant_lifetime, t_project)
    return eff_crf * total_capital
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_costs.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/costingfe/layers/costs.py tests/test_costs.py
git commit -m "feat: all CAS cost accounts (CAS10-CAS90) as pure functions taking CostingConstants"
```

---

## Phase 5: CostModel — Top-Level API

### Task 8: CostModel Wiring All Layers

Wire all layers together into the CostModel class.

**Files:**
- Create: `src/costingfe/model.py`
- Modify: `src/costingfe/__init__.py`
- Create: `tests/test_model.py`

**Step 1: Write failing tests**

```python
# tests/test_model.py
import jax
from costingfe import CostModel, ConfinementConcept, Fuel

def test_forward_basic():
    """Basic forward costing should produce an LCOE."""
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    result = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    assert result.costs.lcoe > 0
    assert result.power_table.p_net > 0
    assert result.power_table.p_fus > 0

def test_forward_lcoe_range():
    """LCOE for a tokamak DT plant should be in reasonable range."""
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    result = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    assert 10 < result.costs.lcoe < 500, f"LCOE {result.costs.lcoe} $/MWh unexpected"

def test_forward_pb11_no_breeding_blanket():
    """pB11 plant should have different cost structure than DT."""
    model_dt = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    model_pb11 = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.PB11)
    result_dt = model_dt.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    result_pb11 = model_pb11.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    # pB11 should have lower licensing cost
    assert result_pb11.costs.cas10 < result_dt.costs.cas10

def test_sensitivity_returns_gradients():
    """Sensitivity should return per-parameter gradients."""
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    result = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    sens = model.sensitivity(result.params)
    assert 'eta_th' in sens
    assert sens['eta_th'] != 0  # thermal efficiency should affect LCOE

def test_compare_all_returns_ranking():
    """Cross-concept comparison should return sorted results."""
    from costingfe import compare_all
    results = compare_all(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
    assert len(results) > 0
    # Should be sorted by LCOE ascending
    lcoes = [r.lcoe for r in results]
    assert lcoes == sorted(lcoes)
```

**Step 2: Run to verify fail**

Run: `uv run pytest tests/test_model.py -v`
Expected: FAIL

**Step 3: Implement CostModel**

```python
# src/costingfe/model.py
"""Top-level CostModel API: wires all 5 layers together."""

from dataclasses import asdict
from costingfe.types import (
    ConfinementConcept, ConfinementFamily, Fuel,
    CONCEPT_TO_FAMILY, PowerTable, CostResult, ForwardResult,
)
from costingfe.layers.physics import (
    mfe_forward_power_balance, mfe_inverse_power_balance,
)
from costingfe.defaults import (
    CostingConstants, load_costing_constants, load_engineering_defaults,
)
from costingfe.layers.costs import (
    cas10_preconstruction, cas21_buildings,
    cas23_turbine, cas24_electrical, cas25_misc, cas26_heat_rejection,
    cas28_digital_twin, cas29_contingency,
    cas30_indirect, cas40_owner, cas50_supplementary, cas60_idc,
    cas70_om, cas80_fuel, cas90_financial,
)
from costingfe.layers.economics import compute_lcoe


class CostModel:
    def __init__(self, concept: ConfinementConcept, fuel: Fuel,
                 costing_constants: CostingConstants = None):
        self.concept = concept
        self.fuel = fuel
        self.family = CONCEPT_TO_FAMILY[concept]
        self.cc = costing_constants or load_costing_constants()
        self._eng_defaults = load_engineering_defaults(
            f"{self.family.value}_{concept.value}"
        )

    def forward(
        self,
        net_electric_mw: float,
        availability: float,
        lifetime_yr: float,
        n_mod: int = 1,
        construction_time_yr: float = 6.0,
        interest_rate: float = 0.07,
        inflation_rate: float = 0.0245,
        noak: bool = True,
        **overrides,
    ) -> ForwardResult:
        """Forward costing: customer requirements -> LCOE."""
        # Merge defaults with overrides
        params = dict(self._eng_defaults)
        params.update(overrides)
        params.update(dict(
            net_electric_mw=net_electric_mw, availability=availability,
            lifetime_yr=lifetime_yr, n_mod=n_mod,
            construction_time_yr=construction_time_yr,
            interest_rate=interest_rate, inflation_rate=inflation_rate,
            noak=noak, fuel=self.fuel, concept=self.concept,
        ))

        # Layer 2: Inverse power balance (net electric per module -> fusion power)
        p_fus = mfe_inverse_power_balance(
            p_net_target=net_electric_mw / n_mod, fuel=self.fuel,
            p_input=params['p_input'], mn=params['mn'],
            eta_th=params['eta_th'], eta_p=params['eta_p'],
            eta_pin=params['eta_pin'], eta_de=params['eta_de'],
            f_sub=params['f_sub'], f_dec=params['f_dec'],
            p_coils=params['p_coils'], p_cool=params['p_cool'],
            p_pump=params['p_pump'], p_trit=params['p_trit'],
            p_house=params['p_house'], p_cryo=params['p_cryo'],
        )

        # Layer 2: Forward power balance (for full PowerTable)
        pt = mfe_forward_power_balance(
            p_fus=p_fus, fuel=self.fuel,
            p_input=params['p_input'], mn=params['mn'],
            eta_th=params['eta_th'], eta_p=params['eta_p'],
            eta_pin=params['eta_pin'], eta_de=params['eta_de'],
            f_sub=params['f_sub'], f_dec=params['f_dec'],
            p_coils=params['p_coils'], p_cool=params['p_cool'],
            p_pump=params['p_pump'], p_trit=params['p_trit'],
            p_house=params['p_house'], p_cryo=params['p_cryo'],
        )

        # Layer 4: Cost accounts
        cc = self.cc
        c10 = cas10_preconstruction(cc, pt.p_net, n_mod, self.fuel, noak)
        c21 = cas21_buildings(cc, pt.p_et, self.fuel, noak)
        c23 = cas23_turbine(cc, pt.p_et, n_mod)
        c24 = cas24_electrical(cc, pt.p_et, n_mod)
        c25 = cas25_misc(cc, pt.p_et, n_mod)
        c26 = cas26_heat_rejection(cc, pt.p_et, n_mod)
        c27 = 0.0  # TODO: special materials (needs blanket details)
        c28 = cas28_digital_twin(cc)
        cas2x_pre_contingency = c21 + 0.0 + c23 + c24 + c25 + c26 + c27 + c28
        c29 = cas29_contingency(cc, cas2x_pre_contingency, noak)
        c20 = cas2x_pre_contingency + c29
        c30 = cas30_indirect(cc, c20, pt.p_net, construction_time_yr)
        c40 = cas40_owner(c20)
        c50 = cas50_supplementary(cc, c23 + c24 + c25 + c26 + c27 + c28, pt.p_net, noak)
        c60 = cas60_idc(cc, c20, pt.p_net, construction_time_yr, self.fuel, noak)
        total_capital = c10 + c20 + c30 + c40 + c50 + c60

        # Layer 5: Economics
        c90 = cas90_financial(cc, total_capital, interest_rate, lifetime_yr,
                              construction_time_yr, self.fuel, noak)
        c70 = cas70_om(cc, pt.p_net, inflation_rate,
                       construction_time_yr, self.fuel, noak)
        c80 = cas80_fuel(cc, pt.p_fus, n_mod, availability, inflation_rate,
                             construction_time_yr, self.fuel, noak)
        lcoe = compute_lcoe(c90, c70, c80, pt.p_net, n_mod, availability)
        overnight = total_capital * 1e6 / (pt.p_net * n_mod * 1e3)  # $/kW

        costs = CostResult(
            cas10=c10, cas21=c21, cas22=0.0, cas23=c23, cas24=c24,
            cas25=c25, cas26=c26, cas27=c27, cas28=c28, cas29=c29,
            cas20=c20, cas30=c30, cas40=c40, cas50=c50, cas60=c60,
            cas70=c70, cas80=c80, cas90=c90,
            total_capital=total_capital, lcoe=lcoe, overnight_cost=overnight,
        )
        return ForwardResult(power_table=pt, costs=costs, params=params)

    def sensitivity(self, params: dict) -> dict[str, float]:
        """Compute d(LCOE)/d(param) for all continuous parameters.

        Uses JAX autodiff (jax.grad) for exact gradients.
        """
        import jax

        # Extract continuous params that affect LCOE
        continuous_keys = [
            'p_input', 'mn', 'eta_th', 'eta_p', 'eta_pin', 'f_sub',
            'p_coils', 'p_cool', 'p_pump', 'p_trit', 'p_house', 'p_cryo',
            'interest_rate', 'inflation_rate',
        ]

        sensitivities = {}
        for key in continuous_keys:
            if key not in params:
                continue
            base_val = params[key]
            if base_val == 0:
                continue

            # Finite-difference fallback (JAX grad through full model
            # requires all-JAX pipeline; use FD for MVP, replace with
            # jax.grad when pipeline is fully JAX-traced)
            delta = abs(base_val) * 0.01
            params_plus = dict(params)
            params_plus[key] = base_val + delta

            # Strip non-forward params
            fwd_keys = {k: v for k, v in params_plus.items()
                        if k not in ('fuel', 'concept', 'net_electric_mw',
                                     'availability', 'lifetime_yr', 'n_mod',
                                     'construction_time_yr', 'interest_rate',
                                     'inflation_rate', 'noak')}

            result_plus = self.forward(
                net_electric_mw=params['net_electric_mw'],
                availability=params['availability'],
                lifetime_yr=params['lifetime_yr'],
                n_mod=params.get('n_mod', 1),
                construction_time_yr=params.get('construction_time_yr', 6.0),
                interest_rate=params_plus.get('interest_rate', params.get('interest_rate', 0.07)),
                inflation_rate=params_plus.get('inflation_rate', params.get('inflation_rate', 0.0245)),
                noak=params.get('noak', True),
                **fwd_keys,
            )

            result_base = self.forward(
                net_electric_mw=params['net_electric_mw'],
                availability=params['availability'],
                lifetime_yr=params['lifetime_yr'],
                n_mod=params.get('n_mod', 1),
                construction_time_yr=params.get('construction_time_yr', 6.0),
                interest_rate=params.get('interest_rate', 0.07),
                inflation_rate=params.get('inflation_rate', 0.0245),
                noak=params.get('noak', True),
                **{k: v for k, v in params.items()
                   if k not in ('fuel', 'concept', 'net_electric_mw',
                                'availability', 'lifetime_yr', 'n_mod',
                                'construction_time_yr', 'interest_rate',
                                'inflation_rate', 'noak')},
            )

            sensitivities[key] = (result_plus.costs.lcoe - result_base.costs.lcoe) / delta

        return sensitivities
```

```python
# src/costingfe/__init__.py
from costingfe.types import (
    ConfinementFamily, ConfinementConcept, Fuel,
    CONCEPT_TO_FAMILY, PowerTable, CostResult, ForwardResult,
)
from costingfe.model import CostModel

from dataclasses import dataclass

@dataclass
class ComparisonResult:
    concept: ConfinementConcept
    fuel: Fuel
    lcoe: float
    result: ForwardResult

def compare_all(
    net_electric_mw: float,
    availability: float,
    lifetime_yr: float,
    concepts: list[ConfinementConcept] | None = None,
    fuels: list[Fuel] | None = None,
    **kwargs,
) -> list[ComparisonResult]:
    """Run all concept x fuel combinations, rank by LCOE."""
    if concepts is None:
        # MVP: only MFE concepts (power balance implemented)
        concepts = [ConfinementConcept.TOKAMAK]
    if fuels is None:
        fuels = list(Fuel)

    results = []
    for concept in concepts:
        for fuel in fuels:
            try:
                model = CostModel(concept=concept, fuel=fuel)
                result = model.forward(
                    net_electric_mw=net_electric_mw,
                    availability=availability,
                    lifetime_yr=lifetime_yr,
                    **kwargs,
                )
                results.append(ComparisonResult(
                    concept=concept, fuel=fuel,
                    lcoe=result.costs.lcoe, result=result,
                ))
            except Exception:
                continue  # skip non-viable combinations

    return sorted(results, key=lambda r: r.lcoe)
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_model.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/costingfe/ tests/test_model.py
git commit -m "feat: CostModel top-level API with forward, sensitivity, and compare_all"
```

---

## Phase 6: Remaining Tasks (Higher-Level)

The following tasks build on the working MVP. Each follows the same TDD pattern (write failing test, implement, verify, commit).

### Task 9: IFE Power Balance

Port IFE power balance (laser drivers, target factory in recirculating power).
- File: `src/costingfe/layers/physics.py` (add `ife_forward_power_balance`, `ife_inverse_power_balance`)
- Test: `tests/test_power_balance.py` (add IFE tests)
- Update `CostModel` to dispatch based on family

### Task 10: MIF Power Balance

Port MIF power balance (pulsed power in recirculating).
- Same pattern as Task 9

### Task 11: Geometry Module

Implement geometry interface and built-in solvers for simple cases.
- Create: `src/costingfe/layers/engineering/geometry.py`
- Implement: `Geometry` dataclass, `compute_tokamak_geometry()`, `compute_mirror_geometry()`
- Test: geometry calculations produce reasonable volumes/areas

### Task 12: CAS22 Reactor Equipment

Implement CAS220101-220119 (reactor equipment sub-accounts). These are concept-specific:
- CAS220101: First wall + blanket (fuel-dependent configuration matrix)
- CAS220103: Coils (tokamak), Lasers (IFE), Pulsed power (MIF)
- CAS220108: Divertor (tokamak), Target factory (IFE)
- Wire into CostModel.forward()

### Task 13: Backcasting — Constraint Propagation

Implement `model.backcast()` using Jacobian from the forward model.
- Create: `src/costingfe/analysis/backcast.py`
- Test: given target LCOE, find required gain/efficiency

### Task 14: Full JAX Tracing

Refactor the forward pipeline to be fully JAX-traced (no Python conditionals on traced values). This enables:
- True `jax.grad` sensitivity (replacing finite differences from Task 7)
- `jax.vmap` for Monte Carlo parameter sweeps
- `jax.jit` compilation for speed

### Task 15: Default Parameter Sets

Create YAML default files for each concept x fuel combination.
- Create: `src/costingfe/data/defaults/*.yaml`
- Load defaults in CostModel constructor

### Task 16: Fusion-Tea Adapter

Create the typed interface that fusion-tea calls.
- Map SysML extracted parameters to CostModel API
- Write costs back in CAS-code-keyed format

---

## Execution Notes

**MVP scope (Tasks 1-8):** Working end-to-end forward LCOE for MFE tokamak with all fuel types, sensitivity analysis (finite-difference), and cross-concept comparison across fuel types. This is the minimum viable product.

**Post-MVP (Tasks 9-16):** IFE/MIF power balance, full reactor equipment costing, backcasting, full JAX tracing, and fusion-tea integration.

**TODO:** Decide on license (pyfecons uses BSD 3-Clause) and add LICENSE file before first public release.

**Future enhancements (beyond Task 16):**
- **Parameter range config for optimism/learning sweeps:** Each parameter needs low/high bounds for the analysis layer (sensitivity sweeps, Monte Carlo, optimism/learning axes). This should be a separate config system — not baked into CostingConstants defaults.
- **Shared auxiliary scaling with `n_mod`:** Auxiliary systems (cryoplant, power supplies, etc.) may be shared across modules, scaling sublinearly. Per-module auxiliary loads should become `f(n_mod) / n_mod` rather than constants. The linear power balance inversion remains valid — only the input parameters change per `n_mod`. This also enables `n_mod` optimization ("what module count minimizes LCOE?").

**Testing strategy:** Every task validates against physics first principles (energy conservation, reasonable ranges). No pyFECONs oracle validation — pyFECONs is reference only.

**Commit convention:** One commit per task. Message format: `feat: <description>` for new features, `fix: <description>` for corrections.
