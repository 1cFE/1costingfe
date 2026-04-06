# Pulsed Power Balance and Inductive DEC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure IFE/MIF into a unified PULSED family with per-pulse energy balances and first-class inductive DEC support, eliminating the need for cost overrides on Helion-like concepts.

**Architecture:** Two confinement families (STEADY_STATE, PULSED) replace three (MFE, IFE, MIF). Pulsed concepts get two power balance functions (thermal and DEC) selected by a `PulsedConversion` enum. Costing layer responds to conversion mode for C220107, C220109, CAS23, CAS26, CAS72.

**Tech Stack:** Python, JAX (autodiff), pydantic (validation), pytest

**Spec:** `docs/plans/2026-04-05-pulsed-power-balance-revamp-design.md`

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `src/costingfe/types.py` | Modify | New enums (`ConfinementFamily`, `PulsedConversion`), `PowerTable` fields, update `CONCEPT_TO_FAMILY` |
| `src/costingfe/layers/physics.py` | Modify | Add `pulsed_thermal_forward/inverse`, `pulsed_dec_forward/inverse`; remove old `ife_*`/`mif_*` functions |
| `src/costingfe/defaults.py` | Modify | Add new DEC costing constants to `CostingConstants` dataclass |
| `src/costingfe/data/defaults/costing_constants.yaml` | Modify | Add DEC costing constant values |
| `src/costingfe/data/defaults/pulsed_laser_ife.yaml` | Create | Renamed from `ife_laser_ife.yaml` with new pulsed params |
| `src/costingfe/data/defaults/pulsed_zpinch.yaml` | Create | Renamed from `ife_zpinch.yaml` |
| `src/costingfe/data/defaults/pulsed_heavy_ion.yaml` | Create | Renamed from `ife_heavy_ion.yaml` |
| `src/costingfe/data/defaults/pulsed_mag_target.yaml` | Create | Renamed from `mif_mag_target.yaml` |
| `src/costingfe/data/defaults/pulsed_plasma_jet.yaml` | Create | Renamed from `mif_plasma_jet.yaml` |
| `src/costingfe/layers/cas22.py` | Modify | C220107 $/J_stored branch, C220109 inductive DEC |
| `src/costingfe/layers/costs.py` | Modify | CAS23/CAS26 scaling basis change, CAS72 cap replacement |
| `src/costingfe/model.py` | Modify | PULSED dispatch, `PulsedConversion` parameter, engineering keys |
| `src/costingfe/validation.py` | Modify | Pulsed family validation rules |
| `src/costingfe/__init__.py` | Modify | Export new enums, update `compare_all` |
| `tests/test_pulsed_power_balance.py` | Create | Tests for both pulsed power balance functions |
| `tests/test_pulsed_dec_costing.py` | Create | Tests for DEC costing (C220107, C220109, CAS72, CAS23/26) |
| `tests/test_ife_mif_power_balance.py` | Delete | Replaced by `test_pulsed_power_balance.py` |
| `examples/dhe3_pulsed_frc.py` | Modify | Rewrite to use new API (no cost overrides for DEC) |

---

### Task 1: Update types and enums

**Files:**
- Modify: `src/costingfe/types.py`
- Test: `tests/test_types.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_types.py`, add:

```python
from costingfe.types import (
    ConfinementFamily,
    PulsedConversion,
    CONCEPT_TO_FAMILY,
    ConfinementConcept,
    PowerTable,
)


def test_confinement_family_values():
    assert ConfinementFamily.STEADY_STATE.value == "steady_state"
    assert ConfinementFamily.PULSED.value == "pulsed"
    # Old values should not exist
    assert not hasattr(ConfinementFamily, "MFE")
    assert not hasattr(ConfinementFamily, "IFE")
    assert not hasattr(ConfinementFamily, "MIF")


def test_pulsed_conversion_enum():
    assert PulsedConversion.THERMAL.value == "thermal"
    assert PulsedConversion.INDUCTIVE_DEC.value == "inductive_dec"


def test_concept_to_family_mapping():
    # Steady-state concepts
    assert CONCEPT_TO_FAMILY[ConfinementConcept.TOKAMAK] == ConfinementFamily.STEADY_STATE
    assert CONCEPT_TO_FAMILY[ConfinementConcept.MIRROR] == ConfinementFamily.STEADY_STATE
    # Pulsed concepts
    assert CONCEPT_TO_FAMILY[ConfinementConcept.LASER_IFE] == ConfinementFamily.PULSED
    assert CONCEPT_TO_FAMILY[ConfinementConcept.ZPINCH] == ConfinementFamily.PULSED
    assert CONCEPT_TO_FAMILY[ConfinementConcept.MAG_TARGET] == ConfinementFamily.PULSED


def test_power_table_has_pulsed_fields():
    """PowerTable should have per-pulse energy fields."""
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(PowerTable)}
    assert "e_driver_mj" in field_names
    assert "e_stored_mj" in field_names
    assert "f_rep" in field_names
    assert "f_ch" in field_names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/c/Users/talru/1cfe/1costingfe && python -m pytest tests/test_types.py::test_confinement_family_values tests/test_types.py::test_pulsed_conversion_enum tests/test_types.py::test_concept_to_family_mapping tests/test_types.py::test_power_table_has_pulsed_fields -v`
Expected: FAIL — old enums still exist, new ones don't

- [ ] **Step 3: Update `src/costingfe/types.py`**

Replace `ConfinementFamily` enum:

```python
class ConfinementFamily(Enum):
    STEADY_STATE = "steady_state"
    PULSED = "pulsed"
```

Add `PulsedConversion` enum after `ConfinementConcept`:

```python
class PulsedConversion(Enum):
    THERMAL = "thermal"
    INDUCTIVE_DEC = "inductive_dec"
```

Update `CONCEPT_TO_FAMILY`:

```python
CONCEPT_TO_FAMILY = {
    ConfinementConcept.TOKAMAK: ConfinementFamily.STEADY_STATE,
    ConfinementConcept.STELLARATOR: ConfinementFamily.STEADY_STATE,
    ConfinementConcept.MIRROR: ConfinementFamily.STEADY_STATE,
    ConfinementConcept.LASER_IFE: ConfinementFamily.PULSED,
    ConfinementConcept.ZPINCH: ConfinementFamily.PULSED,
    ConfinementConcept.HEAVY_ION: ConfinementFamily.PULSED,
    ConfinementConcept.MAG_TARGET: ConfinementFamily.PULSED,
    ConfinementConcept.PLASMA_JET: ConfinementFamily.PULSED,
}
```

Default conversion mode per concept (add after `CONCEPT_TO_FAMILY`):

```python
CONCEPT_DEFAULT_CONVERSION = {
    ConfinementConcept.LASER_IFE: PulsedConversion.THERMAL,
    ConfinementConcept.ZPINCH: PulsedConversion.THERMAL,
    ConfinementConcept.HEAVY_ION: PulsedConversion.THERMAL,
    ConfinementConcept.MAG_TARGET: PulsedConversion.THERMAL,
    ConfinementConcept.PLASMA_JET: PulsedConversion.THERMAL,
}
```

Add fields to `PowerTable` (at the end, with defaults for backward compatibility):

```python
    e_driver_mj: float = 0.0   # Per-pulse driver energy [MJ]
    e_stored_mj: float = 0.0   # Per-pulse cap bank energy [MJ]
    f_rep: float = 0.0          # Repetition rate [Hz]
    f_ch: float = 0.0           # Charged-particle fraction
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/c/Users/talru/1cfe/1costingfe && python -m pytest tests/test_types.py -v`
Expected: All new tests PASS

- [ ] **Step 5: Fix references to old family values throughout codebase**

Every file that references `ConfinementFamily.MFE`, `.IFE`, or `.MIF` needs updating. Key locations:
- `src/costingfe/model.py`: `self.family == ConfinementFamily.MFE` → `.STEADY_STATE`; `.IFE` and `.MIF` branches → `.PULSED` (temporary — will be reworked in Task 5)
- `src/costingfe/layers/cas22.py`: `family == ConfinementFamily.MFE` for C220108 divertor/target factory split
- `src/costingfe/__init__.py`: `compare_all` concept list
- `src/costingfe/validation.py`: family-based required parameter checks
- `src/costingfe/defaults.py`: `load_engineering_defaults` uses `self.family.value` for YAML filename prefix

For `load_engineering_defaults`: the YAML filename format changes from `mfe_tokamak` / `ife_laser_ife` / `mif_mag_target` to `steady_state_tokamak` / `pulsed_laser_ife` / `pulsed_mag_target`. Rename YAML files accordingly (see Step 6).

- [ ] **Step 6: Rename YAML default files**

```bash
cd src/costingfe/data/defaults
mv ife_laser_ife.yaml pulsed_laser_ife.yaml
mv ife_zpinch.yaml pulsed_zpinch.yaml
mv ife_heavy_ion.yaml pulsed_heavy_ion.yaml
mv mif_mag_target.yaml pulsed_mag_target.yaml
mv mif_plasma_jet.yaml pulsed_plasma_jet.yaml
mv mfe_tokamak.yaml steady_state_tokamak.yaml
mv mfe_stellarator.yaml steady_state_stellarator.yaml
mv mfe_mirror.yaml steady_state_mirror.yaml
```

- [ ] **Step 7: Run full test suite to verify nothing is broken**

Run: `cd /mnt/c/Users/talru/1cfe/1costingfe && python -m pytest -x -v`
Expected: All existing tests pass (some IFE/MIF tests may need param adjustments in later tasks)

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: restructure ConfinementFamily to STEADY_STATE/PULSED, add PulsedConversion enum"
```

---

### Task 2: Implement pulsed thermal power balance

**Files:**
- Modify: `src/costingfe/layers/physics.py`
- Create: `tests/test_pulsed_power_balance.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_pulsed_power_balance.py`:

```python
from costingfe.layers.physics import (
    pulsed_thermal_forward,
    pulsed_thermal_inverse,
)
from costingfe.types import Fuel

# Reference: Z-pinch-like, DT, 1 Hz, 100 MJ/pulse
THERMAL_PARAMS = dict(
    p_fus=2500.0,
    fuel=Fuel.DT,
    e_driver_mj=100.0,
    f_rep=1.0,
    mn=1.1,
    eta_th=0.40,
    eta_pin=0.15,
    f_rad=0.10,
    f_sub=0.03,
    p_pump=1.0,
    p_trit=10.0,
    p_house=4.0,
    p_cryo=0.5,
    p_target=1.0,
    p_coils=0.0,
)


def test_thermal_forward_positive_net():
    pt = pulsed_thermal_forward(**THERMAL_PARAMS)
    assert pt.p_net > 0


def test_thermal_forward_energy_conservation():
    """Total power in = total power out (no creation/destruction)."""
    pt = pulsed_thermal_forward(**THERMAL_PARAMS)
    p_driver_avg = THERMAL_PARAMS["e_driver_mj"] * THERMAL_PARAMS["f_rep"]
    p_in = pt.p_fus + p_driver_avg + THERMAL_PARAMS["eta_pin"] * THERMAL_PARAMS["p_pump"]  # noqa: E501
    # Thermal includes neutron*mn, so add (mn-1)*p_neutron for multiplication gain
    p_mult_gain = (THERMAL_PARAMS["mn"] - 1.0) * pt.p_neutron
    p_out = pt.p_et + pt.p_loss
    assert abs(p_in + p_mult_gain - p_out) < 1.0, (
        f"Energy not conserved: in={p_in + p_mult_gain:.1f}, out={p_out:.1f}"
    )


def test_thermal_forward_no_dec():
    pt = pulsed_thermal_forward(**THERMAL_PARAMS)
    assert pt.p_dee == 0.0


def test_thermal_forward_pulsed_fields():
    pt = pulsed_thermal_forward(**THERMAL_PARAMS)
    assert pt.e_driver_mj == 100.0
    assert pt.f_rep == 1.0
    assert pt.e_stored_mj > pt.e_driver_mj  # e_stored = e_driver / eta_pin
    assert pt.f_ch > 0  # DT has ~0.20 charged fraction


def test_thermal_forward_driver_thermalizes():
    """Driver energy should appear in thermal pool."""
    pt = pulsed_thermal_forward(**THERMAL_PARAMS)
    p_driver_avg = THERMAL_PARAMS["e_driver_mj"] * THERMAL_PARAMS["f_rep"]
    # p_th should include driver power
    assert pt.p_th > pt.p_fus * THERMAL_PARAMS["mn"]  # must exceed just neutron*mn


def test_thermal_forward_pump_only_with_thermal():
    """p_pump should not be in recirculating when eta_th=0."""
    params_no_th = dict(THERMAL_PARAMS, eta_th=0.0)
    pt = pulsed_thermal_forward(**params_no_th)
    # With eta_th=0, gross electric = 0, net electric < 0 (all recirculating)
    # but p_pump should NOT appear in recirculating
    assert pt.p_et == 0.0


def test_thermal_inverse_roundtrip():
    pt = pulsed_thermal_forward(**THERMAL_PARAMS)
    inv_params = {k: v for k, v in THERMAL_PARAMS.items() if k not in ("p_fus", "fuel")}
    p_fus_recovered = pulsed_thermal_inverse(
        p_net_target=pt.p_net,
        fuel=Fuel.DT,
        **inv_params,
    )
    assert abs(p_fus_recovered - 2500.0) < 0.5, f"Expected ~2500, got {p_fus_recovered}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/c/Users/talru/1cfe/1costingfe && python -m pytest tests/test_pulsed_power_balance.py -v`
Expected: ImportError — functions don't exist yet

- [ ] **Step 3: Implement `pulsed_thermal_forward` in `src/costingfe/layers/physics.py`**

Add after the existing MIF functions (which will be removed in Task 4):

```python
# ---------------------------------------------------------------------------
# Pulsed Power Balances (unified IFE + MIF)
# ---------------------------------------------------------------------------


def _charged_particle_fraction(
    fuel: Fuel,
    dd_f_T: float = DD_F_T_DEFAULT,
    dd_f_He3: float = DD_F_HE3_DEFAULT,
    dhe3_dd_frac: float = 0.07,
    dhe3_f_T: float = 0.97,
    pb11_f_alpha_n: float = 0.0,
    pb11_f_p_n: float = 0.0,
) -> float:
    """Return charged-particle fraction f_ch for a given fuel.

    Uses ash_neutron_split on unit fusion power to get the fraction.
    """
    ash, _ = ash_neutron_split(
        1.0, fuel, dd_f_T, dd_f_He3, dhe3_dd_frac, dhe3_f_T,
        pb11_f_alpha_n, pb11_f_p_n,
    )
    return ash


def pulsed_thermal_forward(
    p_fus: float,
    fuel: Fuel,
    e_driver_mj: float,
    f_rep: float,
    mn: float,
    eta_th: float,
    eta_pin: float,
    f_rad: float,
    f_sub: float,
    p_pump: float,
    p_trit: float,
    p_house: float,
    p_cryo: float,
    p_target: float,
    p_coils: float = 0.0,
    dd_f_T: float = DD_F_T_DEFAULT,
    dd_f_He3: float = DD_F_HE3_DEFAULT,
    dhe3_dd_frac: float = 0.07,
    dhe3_f_T: float = 0.97,
    pb11_f_alpha_n: float = 0.0,
    pb11_f_p_n: float = 0.0,
) -> PowerTable:
    """Pulsed thermal power balance: driver energy thermalizes.

    For laser IFE, Z-pinch, heavy ion, General Fusion MTF.
    Per-pulse energy balance converted to average power via f_rep.
    """
    fuel_frac_kw = dict(
        dd_f_T=dd_f_T, dd_f_He3=dd_f_He3, dhe3_dd_frac=dhe3_dd_frac,
        dhe3_f_T=dhe3_f_T, pb11_f_alpha_n=pb11_f_alpha_n, pb11_f_p_n=pb11_f_p_n,
    )
    f_ch = _charged_particle_fraction(fuel, **fuel_frac_kw)

    # Average powers
    p_driver = e_driver_mj * f_rep  # MW average

    # Ash/neutron split
    p_ash, p_neutron = ash_neutron_split(p_fus, fuel, **fuel_frac_kw)

    # Radiation fraction of charged-particle energy
    p_rad = f_rad * p_ash
    p_charged_net = p_ash - p_rad

    # All energy thermalizes (driver + fusion products + radiation)
    p_th = mn * p_neutron + p_rad + p_charged_net + p_driver
    if eta_th > 0:
        p_th = p_th + eta_th * 0.0  # placeholder for pumping term below

    # Include pumping heat only if thermal BOP exists
    p_pump_heat = jnp.where(eta_th > 0, p_pump, 0.0) if hasattr(jnp, 'where') else (p_pump if eta_th > 0 else 0.0)
    p_th = p_th + p_pump_heat

    # Thermal electric
    p_the = eta_th * p_th

    # Gross electric (no DEC)
    p_et = p_the
    p_dee = 0.0
    p_dec_waste = 0.0
    p_wall = p_charged_net  # charged particles hit walls and thermalize

    # Lost power (thermal not converted)
    p_loss = p_th - p_the

    # Subsystem power
    p_sub = f_sub * p_et

    # Scientific Q
    q_sci = p_fus / p_driver

    # Recirculating power
    p_aux = p_trit + p_house
    p_pump_recirc = jnp.where(eta_th > 0, p_pump, 0.0) if hasattr(jnp, 'where') else (p_pump if eta_th > 0 else 0.0)
    recirculating = (
        p_driver / eta_pin
        + p_pump_recirc
        + p_sub
        + p_aux
        + p_cryo
        + p_target
        + p_coils
    )
    q_eng = p_et / recirculating

    # Net electric
    rec_frac = 1.0 / q_eng
    p_net = (1.0 - rec_frac) * p_et

    # Derived pulsed quantities
    e_stored_mj = e_driver_mj / eta_pin

    return PowerTable(
        p_fus=p_fus,
        p_ash=p_ash,
        p_neutron=p_neutron,
        p_rad=p_rad,
        p_wall=p_wall,
        p_dee=p_dee,
        p_dec_waste=p_dec_waste,
        p_th=p_th,
        p_the=p_the,
        p_et=p_et,
        p_loss=p_loss,
        p_net=p_net,
        p_input=p_driver,
        p_pump=p_pump,
        p_sub=p_sub,
        p_aux=p_aux,
        p_coils=p_coils,
        p_cool=0.0,
        p_cryo=p_cryo,
        p_target=p_target,
        q_sci=q_sci,
        q_eng=q_eng,
        rec_frac=rec_frac,
        e_driver_mj=e_driver_mj,
        e_stored_mj=e_stored_mj,
        f_rep=f_rep,
        f_ch=f_ch,
    )
```

Note on `jnp.where` vs plain `if`: the code must remain JAX-traceable for sensitivity analysis. Use `jnp.where` for conditionals on float parameters. The `eta_th > 0` conditional for `p_pump` must use `jnp.where` since `eta_th` may be a JAX tracer.

- [ ] **Step 4: Implement `pulsed_thermal_inverse`**

Add below `pulsed_thermal_forward`:

```python
def pulsed_thermal_inverse(
    p_net_target: float,
    fuel: Fuel,
    e_driver_mj: float,
    f_rep: float,
    mn: float,
    eta_th: float,
    eta_pin: float,
    f_rad: float,
    f_sub: float,
    p_pump: float,
    p_trit: float,
    p_house: float,
    p_cryo: float,
    p_target: float,
    p_coils: float = 0.0,
    dd_f_T: float = DD_F_T_DEFAULT,
    dd_f_He3: float = DD_F_HE3_DEFAULT,
    dhe3_dd_frac: float = 0.07,
    dhe3_f_T: float = 0.97,
    pb11_f_alpha_n: float = 0.0,
    pb11_f_p_n: float = 0.0,
) -> float:
    """Inverse pulsed thermal: target P_net -> required P_fus.

    Linear inversion: P_net = a * P_fus + b, solve for P_fus.
    """
    fuel_frac_kw = dict(
        dd_f_T=dd_f_T, dd_f_He3=dd_f_He3, dhe3_dd_frac=dhe3_dd_frac,
        dhe3_f_T=dhe3_f_T, pb11_f_alpha_n=pb11_f_alpha_n, pb11_f_p_n=pb11_f_p_n,
    )
    ash_frac, _ = ash_neutron_split(1.0, fuel, **fuel_frac_kw)
    neutron_frac = 1.0 - ash_frac

    p_driver = e_driver_mj * f_rep

    # Thermal: p_th = c_th * p_fus + c_th0
    c_th = mn * neutron_frac + ash_frac  # f_rad doesn't matter: rad + charged_net = ash
    c_th0 = p_driver + jnp.where(eta_th > 0, p_pump, 0.0)

    # Gross electric
    c_et = eta_th * c_th
    c_et0 = eta_th * c_th0

    # Recirculating: c_den * p_fus + c_den0
    c_den = f_sub * c_et
    p_aux = p_trit + p_house
    c_den0 = (
        p_driver / eta_pin
        + jnp.where(eta_th > 0, p_pump, 0.0)
        + f_sub * c_et0
        + p_aux
        + p_cryo
        + p_target
        + p_coils
    )

    # P_net = (c_et - c_den) * p_fus + (c_et0 - c_den0)
    p_fus = (p_net_target - c_et0 + c_den0) / (c_et - c_den)
    return p_fus
```

- [ ] **Step 5: Run tests**

Run: `cd /mnt/c/Users/talru/1cfe/1costingfe && python -m pytest tests/test_pulsed_power_balance.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/costingfe/layers/physics.py tests/test_pulsed_power_balance.py
git commit -m "feat: add pulsed thermal power balance (forward + inverse)"
```

---

### Task 3: Implement pulsed DEC power balance

**Files:**
- Modify: `src/costingfe/layers/physics.py`
- Modify: `tests/test_pulsed_power_balance.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_pulsed_power_balance.py`:

```python
from costingfe.layers.physics import (
    pulsed_dec_forward,
    pulsed_dec_inverse,
)

# Reference: Helion-like FRC, D-He3, 1 Hz, 12 MJ/pulse
DEC_PARAMS = dict(
    p_fus=500.0,
    fuel=Fuel.DHE3,
    e_driver_mj=12.0,
    f_rep=1.0,
    mn=1.0,
    eta_th=0.0,       # no thermal BOP
    eta_pin=0.95,
    eta_dec=0.85,
    f_pdv=0.80,
    f_rad=0.05,
    f_sub=0.03,
    p_pump=0.0,       # no thermal cycle → no pumping
    p_trit=0.5,
    p_house=2.0,
    p_cryo=0.0,
    p_target=0.0,
    p_coils=0.5,
)


def test_dec_forward_positive_net():
    pt = pulsed_dec_forward(**DEC_PARAMS)
    assert pt.p_net > 0


def test_dec_forward_has_dec_output():
    pt = pulsed_dec_forward(**DEC_PARAMS)
    assert pt.p_dee > 0  # DEC should produce electric power


def test_dec_forward_no_thermal_when_eta_th_zero():
    pt = pulsed_dec_forward(**DEC_PARAMS)
    assert pt.p_the == 0.0  # eta_th=0 means no thermal electric


def test_dec_forward_pulsed_fields():
    pt = pulsed_dec_forward(**DEC_PARAMS)
    assert pt.e_driver_mj == 12.0
    assert pt.f_rep == 1.0
    assert abs(pt.e_stored_mj - 12.0 / 0.95) < 0.1
    assert pt.f_ch > 0.9  # D-He3 has ~0.94 charged fraction


def test_dec_forward_driver_recirc_is_losses_only():
    """DEC driver recirculating power should be charging losses, not full draw."""
    pt = pulsed_dec_forward(**DEC_PARAMS)
    p_driver = DEC_PARAMS["e_driver_mj"] * DEC_PARAMS["f_rep"]
    # Full draw would be p_driver / eta_pin = 12.63 MW
    # Charging losses = p_driver * (1/eta_pin - 1) = 0.63 MW
    # Recirculating should be much less than full draw
    expected_losses = p_driver * (1.0 / DEC_PARAMS["eta_pin"] - 1.0)
    # Total recirculating includes other loads, but driver component is small
    assert pt.rec_frac < 0.5  # should not be recirculating-dominated


def test_dec_forward_with_thermal_bop():
    """D-T with DEC + thermal BOP for neutrons."""
    params_dt = dict(
        p_fus=2000.0,
        fuel=Fuel.DT,
        e_driver_mj=50.0,
        f_rep=1.0,
        mn=1.1,
        eta_th=0.40,      # thermal BOP for neutrons
        eta_pin=0.90,
        eta_dec=0.85,
        f_pdv=0.75,
        f_rad=0.10,
        f_sub=0.03,
        p_pump=1.0,        # pumping for thermal cycle
        p_trit=10.0,
        p_house=4.0,
        p_cryo=0.0,
        p_target=0.0,
        p_coils=0.5,
    )
    pt = pulsed_dec_forward(**params_dt)
    assert pt.p_dee > 0      # DEC output
    assert pt.p_the > 0      # thermal output from neutrons
    assert pt.p_et > pt.p_dee  # gross = DEC + thermal
    assert pt.p_net > 0


def test_dec_inverse_roundtrip():
    pt = pulsed_dec_forward(**DEC_PARAMS)
    inv_params = {k: v for k, v in DEC_PARAMS.items() if k not in ("p_fus", "fuel")}
    p_fus_recovered = pulsed_dec_inverse(
        p_net_target=pt.p_net,
        fuel=Fuel.DHE3,
        **inv_params,
    )
    assert abs(p_fus_recovered - 500.0) < 0.5, f"Expected ~500, got {p_fus_recovered}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/c/Users/talru/1cfe/1costingfe && python -m pytest tests/test_pulsed_power_balance.py -k dec -v`
Expected: ImportError

- [ ] **Step 3: Implement `pulsed_dec_forward`**

Add to `src/costingfe/layers/physics.py`:

```python
def pulsed_dec_forward(
    p_fus: float,
    fuel: Fuel,
    e_driver_mj: float,
    f_rep: float,
    mn: float,
    eta_th: float,
    eta_pin: float,
    eta_dec: float,
    f_pdv: float,
    f_rad: float,
    f_sub: float,
    p_pump: float,
    p_trit: float,
    p_house: float,
    p_cryo: float,
    p_target: float,
    p_coils: float = 0.0,
    dd_f_T: float = DD_F_T_DEFAULT,
    dd_f_He3: float = DD_F_HE3_DEFAULT,
    dhe3_dd_frac: float = 0.07,
    dhe3_f_T: float = 0.97,
    pb11_f_alpha_n: float = 0.0,
    pb11_f_p_n: float = 0.0,
) -> PowerTable:
    """Pulsed DEC (inductive) power balance.

    Driver energy is in an electromagnetic loop (cap bank -> coils -> plasma
    -> coils -> cap bank). Charged-particle PdV work is recovered inductively.
    Optional thermal BOP for neutron energy when eta_th > 0.
    """
    fuel_frac_kw = dict(
        dd_f_T=dd_f_T, dd_f_He3=dd_f_He3, dhe3_dd_frac=dhe3_dd_frac,
        dhe3_f_T=dhe3_f_T, pb11_f_alpha_n=pb11_f_alpha_n, pb11_f_p_n=pb11_f_p_n,
    )
    f_ch = _charged_particle_fraction(fuel, **fuel_frac_kw)

    # Average powers
    p_driver = e_driver_mj * f_rep

    # Ash/neutron split
    p_ash, p_neutron = ash_neutron_split(p_fus, fuel, **fuel_frac_kw)

    # Radiation: fraction of charged energy radiated during burn
    p_rad = f_rad * p_ash
    p_charged_net = p_ash - p_rad

    # DEC recovery
    p_pdv = f_pdv * p_charged_net
    p_recovered = eta_dec * (p_driver + p_pdv)
    p_dee = p_recovered - p_driver  # net DEC electric output

    # Waste heat from DEC circuit
    p_dec_waste = (1.0 - eta_dec) * (p_driver + p_pdv)
    p_undirected = p_charged_net - p_pdv  # thermal energy that didn't do PdV work

    # Thermal pool: neutrons + radiation + undirected + DEC waste
    p_pump_heat = jnp.where(eta_th > 0, p_pump, 0.0)
    p_th = mn * p_neutron + p_rad + p_undirected + p_dec_waste + p_pump_heat
    p_the = eta_th * p_th

    # Gross electric: DEC net + thermal
    p_et = p_dee + p_the

    # Wall power (for geometry/costing — what hits the first wall)
    p_wall = p_undirected + p_dec_waste + p_rad

    # Lost power
    p_loss = p_th - p_the

    # Subsystem power
    p_sub = f_sub * p_et

    # Scientific Q
    q_sci = p_fus / p_driver

    # Recirculating: driver charging losses only (cap bank loop recovers E_driver)
    p_aux = p_trit + p_house
    p_pump_recirc = jnp.where(eta_th > 0, p_pump, 0.0)
    recirculating = (
        p_driver * (1.0 / eta_pin - 1.0)  # charging losses only
        + p_pump_recirc
        + p_sub
        + p_aux
        + p_cryo
        + p_target
        + p_coils
    )
    q_eng = p_et / recirculating

    # Net electric
    rec_frac = 1.0 / q_eng
    p_net = (1.0 - rec_frac) * p_et

    # Derived pulsed quantities
    e_stored_mj = e_driver_mj / eta_pin

    return PowerTable(
        p_fus=p_fus,
        p_ash=p_ash,
        p_neutron=p_neutron,
        p_rad=p_rad,
        p_wall=p_wall,
        p_dee=p_dee,
        p_dec_waste=p_dec_waste,
        p_th=p_th,
        p_the=p_the,
        p_et=p_et,
        p_loss=p_loss,
        p_net=p_net,
        p_input=p_driver,
        p_pump=p_pump,
        p_sub=p_sub,
        p_aux=p_aux,
        p_coils=p_coils,
        p_cool=0.0,
        p_cryo=p_cryo,
        p_target=p_target,
        q_sci=q_sci,
        q_eng=q_eng,
        rec_frac=rec_frac,
        e_driver_mj=e_driver_mj,
        e_stored_mj=e_stored_mj,
        f_rep=f_rep,
        f_ch=f_ch,
    )
```

- [ ] **Step 4: Implement `pulsed_dec_inverse`**

```python
def pulsed_dec_inverse(
    p_net_target: float,
    fuel: Fuel,
    e_driver_mj: float,
    f_rep: float,
    mn: float,
    eta_th: float,
    eta_pin: float,
    eta_dec: float,
    f_pdv: float,
    f_rad: float,
    f_sub: float,
    p_pump: float,
    p_trit: float,
    p_house: float,
    p_cryo: float,
    p_target: float,
    p_coils: float = 0.0,
    dd_f_T: float = DD_F_T_DEFAULT,
    dd_f_He3: float = DD_F_HE3_DEFAULT,
    dhe3_dd_frac: float = 0.07,
    dhe3_f_T: float = 0.97,
    pb11_f_alpha_n: float = 0.0,
    pb11_f_p_n: float = 0.0,
) -> float:
    """Inverse pulsed DEC: target P_net -> required P_fus.

    Linear in P_fus: all energy terms scale proportionally with P_fus,
    E_driver is fixed. Solve P_net = a * P_fus + b for P_fus.
    """
    fuel_frac_kw = dict(
        dd_f_T=dd_f_T, dd_f_He3=dd_f_He3, dhe3_dd_frac=dhe3_dd_frac,
        dhe3_f_T=dhe3_f_T, pb11_f_alpha_n=pb11_f_alpha_n, pb11_f_p_n=pb11_f_p_n,
    )
    ash_frac, _ = ash_neutron_split(1.0, fuel, **fuel_frac_kw)
    neutron_frac = 1.0 - ash_frac

    p_driver = e_driver_mj * f_rep

    # Per unit p_fus:
    # p_charged_net = ash_frac * (1 - f_rad)
    # p_pdv = f_pdv * ash_frac * (1 - f_rad)
    # p_dee_per_fus = eta_dec * f_pdv * ash_frac * (1 - f_rad)
    c_ash_net = ash_frac * (1.0 - f_rad)
    c_pdv = f_pdv * c_ash_net
    c_dee = eta_dec * c_pdv  # DEC output per unit p_fus (from fusion only)

    # Thermal per unit p_fus:
    c_undirected = c_ash_net - c_pdv
    c_dec_waste = (1.0 - eta_dec) * c_pdv  # DEC waste from fusion PdV
    c_th = mn * neutron_frac + ash_frac * f_rad + c_undirected + c_dec_waste
    c_the = eta_th * c_th

    # Gross electric per unit p_fus
    c_et = c_dee + c_the

    # Constant terms (independent of p_fus)
    # DEC: p_dee0 = eta_dec * p_driver - p_driver = p_driver * (eta_dec - 1)
    p_dee0 = p_driver * (eta_dec - 1.0)
    # DEC waste from driver: (1 - eta_dec) * p_driver
    p_dec_waste0 = (1.0 - eta_dec) * p_driver
    p_pump_heat = jnp.where(eta_th > 0, p_pump, 0.0)
    p_th0 = p_dec_waste0 + p_pump_heat
    p_the0 = eta_th * p_th0
    p_et0 = p_dee0 + p_the0

    # Recirculating: c_recirc * p_fus + c_recirc0
    c_recirc = f_sub * c_et
    p_aux = p_trit + p_house
    p_pump_recirc = jnp.where(eta_th > 0, p_pump, 0.0)
    c_recirc0 = (
        p_driver * (1.0 / eta_pin - 1.0)
        + p_pump_recirc
        + f_sub * p_et0
        + p_aux
        + p_cryo
        + p_target
        + p_coils
    )

    # P_net = (c_et - c_recirc) * p_fus + (p_et0 - c_recirc0)
    p_fus = (p_net_target - p_et0 + c_recirc0) / (c_et - c_recirc)
    return p_fus
```

- [ ] **Step 5: Run tests**

Run: `cd /mnt/c/Users/talru/1cfe/1costingfe && python -m pytest tests/test_pulsed_power_balance.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/costingfe/layers/physics.py tests/test_pulsed_power_balance.py
git commit -m "feat: add pulsed DEC (inductive) power balance (forward + inverse)"
```

---

### Task 4: Add DEC costing constants and update YAML defaults

**Files:**
- Modify: `src/costingfe/defaults.py`
- Modify: `src/costingfe/data/defaults/costing_constants.yaml`
- Modify: all `pulsed_*.yaml` files (add pulsed params)

- [ ] **Step 1: Add DEC fields to `CostingConstants` in `src/costingfe/defaults.py`**

Add after the existing `dec_grid_lifetime_pb11` field:

```python
    # Pulsed inductive DEC — driver cost basis
    # $/J_stored, NOAK all-in (caps + switches + charging + buswork)
    # Sensitivity range: 1.5-4.0
    c_cap_allin_per_joule: float = 2.0

    # Pulsed inductive DEC — C220109 incremental markups
    markup_switch_bidir: float = 0.06   # Bidirectional switch premium (frac of driver)
    markup_controls: float = 0.04       # FPGA/energy management (frac of driver)
    c_inv_per_kw_net: float = 150.0     # Grid-tie inverter ($/kW_net)

    # Pulsed inductive DEC — CAS72 cap replacement
    cap_shot_lifetime: float = 1.0e8    # Shots, NOAK baseline. Range: 1e7-1e9

    # Pulsed radiation fraction defaults (fraction of charged-particle energy)
    f_rad_dt: float = 0.10
    f_rad_dd: float = 0.08
    f_rad_dhe3: float = 0.05
    f_rad_pb11: float = 0.15   # High Z^2 bremsstrahlung

    # PdV work fraction — fraction of charged-particle energy doing work
    # against the confining field during expansion. For adiabatic expansion:
    # f_pdv = 1 - (1/r)^(gamma-1), gamma=5/3
    # r=10 -> 0.78, r=20 -> 0.86, r=50 -> 0.91
    f_pdv: float = 0.80
```

Add a helper method to `CostingConstants`:

```python
    def f_rad(self, fuel):
        """Default radiation fraction for pulsed concepts."""
        from costingfe.types import Fuel
        return {
            Fuel.DT: self.f_rad_dt,
            Fuel.DD: self.f_rad_dd,
            Fuel.DHE3: self.f_rad_dhe3,
            Fuel.PB11: self.f_rad_pb11,
        }.get(fuel, self.f_rad_dt)
```

- [ ] **Step 2: Add values to `costing_constants.yaml`**

Append to end of file:

```yaml
# Pulsed inductive DEC — driver cost basis
c_cap_allin_per_joule: 2.0   # $/J_stored, NOAK all-in (caps+switches+charging+buswork)
                              # Sensitivity range: 1.5-4.0

# Pulsed inductive DEC — C220109 incremental markups
markup_switch_bidir: 0.06    # Bidirectional switch premium (frac of driver cost)
markup_controls: 0.04        # FPGA/energy management upgrade (frac of driver cost)
c_inv_per_kw_net: 150.0      # Grid-tie inverter ($/kW_net)

# Pulsed inductive DEC — CAS72 cap replacement
cap_shot_lifetime: 1.0e8     # Shots, NOAK baseline. Sensitivity: 1e7-1e9

# Pulsed radiation fraction defaults (frac of charged-particle energy)
f_rad_dt: 0.10
f_rad_dd: 0.08
f_rad_dhe3: 0.05
f_rad_pb11: 0.15             # High Z^2 bremsstrahlung

# PdV work fraction for inductive DEC expansion stroke
# For adiabatic expansion: f_pdv = 1 - (1/r)^(gamma-1), gamma=5/3
# r=10 -> 0.78, r=20 -> 0.86, r=50 -> 0.91
f_pdv: 0.80
```

- [ ] **Step 3: Add pulsed parameters to each `pulsed_*.yaml`**

For each of the five pulsed YAML files, add pulsed parameters. Replace the old `p_implosion`/`p_ignition`/`eta_pin1`/`eta_pin2` (IFE files) or `p_driver`/`eta_pin` (MIF files) with the new parameter set.

Example for `pulsed_zpinch.yaml` (replace entire file):

```yaml
# Default engineering parameters for pulsed Z-pinch IFE

# Pulsed parameters
e_driver_mj: 20.0       # Energy delivered to plasma per pulse [MJ]
f_rep: 0.1              # Repetition rate [Hz]
pulsed_conversion: thermal
f_rad: 0.10             # Radiation fraction of charged-particle energy
eta_pin: 0.15            # Pulsed power wall-plug efficiency

mn: 1.1              # Neutron energy multiplier
f_sub: 0.03          # Subsystem power fraction
p_pump: 1.0          # Pumping power [MW]
p_trit: 10.0         # Tritium processing power [MW]
p_house: 4.0         # Housekeeping power [MW]
p_cryo: 0.5          # Cryogenic power [MW]
p_target: 1.0        # Target factory power [MW]
p_coils: 0.0         # No guide field coils

# Radial build geometry [m] (spherical chamber)
R0: 0.0
plasma_t: 5.0
blanket_t: 1.0
ht_shield_t: 0.30
structure_t: 0.20
vessel_t: 0.10

construction_time_yr: 6.0

# Fuel burn fraction defaults
dd_f_T: 0.969
dd_f_He3: 0.689
dhe3_dd_frac: 0.07
dhe3_f_T: 0.97
pb11_f_alpha_n: 0.0
pb11_f_p_n: 0.0
```

Similarly update the other four files:
- `pulsed_laser_ife.yaml`: `e_driver_mj: 2.0`, `f_rep: 10.0`, `eta_pin: 0.10`, `pulsed_conversion: thermal`
- `pulsed_heavy_ion.yaml`: `e_driver_mj: 5.0`, `f_rep: 5.0`, `eta_pin: 0.25`, `pulsed_conversion: thermal`
- `pulsed_mag_target.yaml`: `e_driver_mj: 50.0`, `f_rep: 1.0`, `eta_pin: 0.30`, `pulsed_conversion: thermal`, `p_coils: 0.5`
- `pulsed_plasma_jet.yaml`: `e_driver_mj: 30.0`, `f_rep: 1.0`, `eta_pin: 0.25`, `pulsed_conversion: thermal`, `p_coils: 0.2`

- [ ] **Step 4: Run tests**

Run: `cd /mnt/c/Users/talru/1cfe/1costingfe && python -m pytest tests/test_defaults.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add DEC costing constants and pulsed YAML defaults"
```

---

### Task 5: Wire pulsed power balance into CostModel

**Files:**
- Modify: `src/costingfe/model.py`
- Modify: `src/costingfe/validation.py`

- [ ] **Step 1: Update `CostModel.__init__` to accept `pulsed_conversion`**

In `src/costingfe/model.py`, add `pulsed_conversion` parameter:

```python
from costingfe.types import (
    CONCEPT_DEFAULT_CONVERSION,
    CONCEPT_TO_FAMILY,
    CoilMaterial,
    ConfinementConcept,
    ConfinementFamily,
    CostResult,
    ForwardResult,
    Fuel,
    PowerCycle,
    PulsedConversion,
    WallMaterial,
)

class CostModel:
    def __init__(
        self,
        concept: ConfinementConcept,
        fuel: Fuel,
        costing_constants: CostingConstants = None,
        power_cycle: PowerCycle = PowerCycle.RANKINE,
        pulsed_conversion: PulsedConversion = None,
    ):
        self.concept = concept
        self.fuel = fuel
        self.family = CONCEPT_TO_FAMILY[concept]
        self.power_cycle = power_cycle
        self.pulsed_conversion = pulsed_conversion or CONCEPT_DEFAULT_CONVERSION.get(concept)
        # ... rest unchanged
```

- [ ] **Step 2: Replace IFE/MIF dispatch in `_power_balance`**

Replace the `elif self.family == ConfinementFamily.IFE:` and `elif self.family == ConfinementFamily.MIF:` blocks with a single `elif self.family == ConfinementFamily.PULSED:` block:

```python
        elif self.family == ConfinementFamily.PULSED:
            fuel_frac_kw = dict(
                dd_f_T=params["dd_f_T"],
                dd_f_He3=params["dd_f_He3"],
                dhe3_dd_frac=params["dhe3_dd_frac"],
                dhe3_f_T=params["dhe3_f_T"],
                pb11_f_alpha_n=params["pb11_f_alpha_n"],
                pb11_f_p_n=params["pb11_f_p_n"],
            )
            common_kw = dict(
                fuel=self.fuel,
                e_driver_mj=params["e_driver_mj"],
                f_rep=params["f_rep"],
                mn=params["mn"],
                eta_th=params["eta_th"],
                eta_pin=params["eta_pin"],
                f_rad=params.get("f_rad", self.cc.f_rad(self.fuel)),
                f_sub=params["f_sub"],
                p_pump=params["p_pump"],
                p_trit=params["p_trit"],
                p_house=params["p_house"],
                p_cryo=params["p_cryo"],
                p_target=params.get("p_target", 0.0),
                p_coils=params.get("p_coils", 0.0),
                **fuel_frac_kw,
            )

            if self.pulsed_conversion == PulsedConversion.INDUCTIVE_DEC:
                p_fus = pulsed_dec_inverse(
                    p_net_target=p_net_per_mod,
                    eta_dec=params["eta_dec"],
                    f_pdv=params.get("f_pdv", self.cc.f_pdv),
                    **common_kw,
                )
                pt = pulsed_dec_forward(
                    p_fus=p_fus,
                    eta_dec=params["eta_dec"],
                    f_pdv=params.get("f_pdv", self.cc.f_pdv),
                    **common_kw,
                )
            else:
                p_fus = pulsed_thermal_inverse(
                    p_net_target=p_net_per_mod,
                    **common_kw,
                )
                pt = pulsed_thermal_forward(
                    p_fus=p_fus,
                    **common_kw,
                )
```

Update imports at top of file: add `pulsed_thermal_forward`, `pulsed_thermal_inverse`, `pulsed_dec_forward`, `pulsed_dec_inverse`; remove `ife_*` and `mif_*` imports.

- [ ] **Step 3: Update `_engineering_keys` for PULSED family**

Replace the `ConfinementFamily.IFE` and `ConfinementFamily.MIF` entries:

```python
        family_specific = {
            ConfinementFamily.STEADY_STATE: [
                "p_input",
                "eta_pin",
                "eta_de",
                "f_dec",
                "p_coils",
                "p_cool",
                "R0",
                "elon",
                "chamber_length",
                "q95",
                "f_GW",
                "B",
                "T_e",
                "n_e",
                "Z_eff",
                "plasma_volume",
                "R_w",
                "T_edge",
                "tau_ratio",
                "b_max",
                "r_coil",
                "p_nbi",
                "p_ecrh",
                "p_icrf",
                "p_lhcd",
                "M_ion",
                "lambda_q",
                "disruption_rate_base",
                "disruption_steepness",
                "disruption_damage",
                "disruption_downtime",
            ],
            ConfinementFamily.PULSED: [
                "e_driver_mj",
                "f_rep",
                "eta_pin",
                "f_rad",
                "p_target",
                "p_coils",
                "eta_dec",
                "f_pdv",
            ],
        }
```

- [ ] **Step 4: Update `forward()` to pass `pulsed_conversion` to params and handle `f_rad` default**

In the `forward()` method, after the `params.update(overrides)` block, add:

```python
        # Default f_rad from fuel if not provided
        if self.family == ConfinementFamily.PULSED and "f_rad" not in overrides:
            params.setdefault("f_rad", cc.f_rad(self.fuel))
```

- [ ] **Step 5: Update CAS23 and CAS26 calls in `forward()`**

Change scaling basis from `p_et` to `p_the`/`p_th`:

```python
        c23 = co.get("CAS23", cas23_turbine(cc, pt.p_the, n_mod))
        # ...
        c26 = co.get("CAS26", cas26_heat_rejection(cc, pt.p_th, n_mod))
```

Update the function signatures in `costs.py` (next task) to accept the new arguments.

- [ ] **Step 6: Update validation in `src/costingfe/validation.py`**

Add pulsed-family required parameters. Replace the IFE/MIF validation blocks with a PULSED block that requires `e_driver_mj`, `f_rep`, `eta_pin`.

- [ ] **Step 7: Run integration test**

Run: `cd /mnt/c/Users/talru/1cfe/1costingfe && python -m pytest tests/test_model.py -v`
Expected: Existing model tests may need updating for new parameter names. Fix as needed.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat: wire pulsed power balance into CostModel dispatch"
```

---

### Task 6: Update CAS22, CAS23, CAS26, CAS72 for DEC costing

**Files:**
- Modify: `src/costingfe/layers/cas22.py`
- Modify: `src/costingfe/layers/costs.py`
- Modify: `src/costingfe/model.py` (pass new params to costing functions)
- Create: `tests/test_pulsed_dec_costing.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_pulsed_dec_costing.py`:

```python
from costingfe import CostModel, Fuel
from costingfe.types import ConfinementConcept, PulsedConversion


def test_dec_c220107_uses_joule_basis():
    """C220107 should use $/J_stored for INDUCTIVE_DEC."""
    model = CostModel(
        concept=ConfinementConcept.MAG_TARGET,
        fuel=Fuel.DHE3,
        pulsed_conversion=PulsedConversion.INDUCTIVE_DEC,
    )
    result = model.forward(
        net_electric_mw=50.0,
        availability=0.85,
        lifetime_yr=30,
        e_driver_mj=12.0,
        f_rep=1.0,
        eta_pin=0.95,
        eta_dec=0.85,
        eta_th=0.0,
        mn=1.0,
        p_cryo=0.0,
        p_target=0.0,
    )
    # e_stored = 12.0 / 0.95 ≈ 12.63 MJ
    # c220107 = 2.0 $/J * 12.63e6 J / 1e6 = 25.26 M$
    c220107 = result.cas22_detail["C220107"]
    expected = 2.0 * 12.0 / 0.95  # M$
    assert abs(c220107 - expected) < 0.5, f"Expected ~{expected:.1f}, got {c220107:.1f}"


def test_dec_c220109_populated():
    """C220109 should be nonzero for INDUCTIVE_DEC."""
    model = CostModel(
        concept=ConfinementConcept.MAG_TARGET,
        fuel=Fuel.DHE3,
        pulsed_conversion=PulsedConversion.INDUCTIVE_DEC,
    )
    result = model.forward(
        net_electric_mw=50.0,
        availability=0.85,
        lifetime_yr=30,
        e_driver_mj=12.0,
        f_rep=1.0,
        eta_pin=0.95,
        eta_dec=0.85,
        eta_th=0.0,
        mn=1.0,
        p_cryo=0.0,
        p_target=0.0,
    )
    assert result.cas22_detail["C220109"] > 0


def test_dec_cas23_zero_when_no_thermal():
    """CAS23 should be zero when eta_th=0 (no turbine plant)."""
    model = CostModel(
        concept=ConfinementConcept.MAG_TARGET,
        fuel=Fuel.DHE3,
        pulsed_conversion=PulsedConversion.INDUCTIVE_DEC,
    )
    result = model.forward(
        net_electric_mw=50.0,
        availability=0.85,
        lifetime_yr=30,
        e_driver_mj=12.0,
        f_rep=1.0,
        eta_pin=0.95,
        eta_dec=0.85,
        eta_th=0.0,
        mn=1.0,
        p_cryo=0.0,
        p_target=0.0,
    )
    assert result.costs.cas23 == 0.0


def test_thermal_pulsed_cas23_nonzero():
    """CAS23 should be nonzero for thermal pulsed concept."""
    model = CostModel(
        concept=ConfinementConcept.ZPINCH,
        fuel=Fuel.DT,
    )
    result = model.forward(
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
        e_driver_mj=20.0,
        f_rep=0.1,
        eta_pin=0.15,
    )
    assert result.costs.cas23 > 0


def test_dec_no_cost_overrides_needed():
    """DEC concept should NOT need cost overrides for CAS23, CAS26, C220107, C220109."""
    model = CostModel(
        concept=ConfinementConcept.MAG_TARGET,
        fuel=Fuel.DHE3,
        pulsed_conversion=PulsedConversion.INDUCTIVE_DEC,
    )
    result = model.forward(
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
        n_mod=20,
        e_driver_mj=12.0,
        f_rep=1.0,
        eta_pin=0.95,
        eta_dec=0.85,
        eta_th=0.0,
        mn=1.0,
        p_cryo=0.0,
        p_target=0.0,
    )
    # Should have zero overrides
    assert len(result.overridden) == 0
    # CAS23 auto-zeroed, C220107 on $/J basis, C220109 populated
    assert result.costs.cas23 == 0.0
    assert result.cas22_detail["C220107"] > 0
    assert result.cas22_detail["C220109"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/c/Users/talru/1cfe/1costingfe && python -m pytest tests/test_pulsed_dec_costing.py -v`
Expected: FAIL

- [ ] **Step 3: Update CAS23 and CAS26 in `src/costingfe/layers/costs.py`**

Change function signatures and scaling:

```python
def cas23_turbine(cc, p_the, n_mod):
    """CAS23: Turbine plant equipment. Returns M$.

    Scales with thermal electric power (steam turbine output).
    When eta_th=0 (no thermal BOP), p_the=0 and CAS23=0 automatically.
    """
    return n_mod * p_the * cc.turbine_per_mw


def cas26_heat_rejection(cc, p_th, n_mod):
    """CAS26: Heat rejection. Returns M$.

    Scales with total thermal power (heat that must be rejected).
    """
    return n_mod * p_th * cc.heat_rej_per_mw
```

- [ ] **Step 4: Update C220107 in `src/costingfe/layers/cas22.py`**

Add `pulsed_conversion`, `e_stored_mj`, `q_sci`, `f_ch`, and `p_net` parameters to `cas22_reactor_plant_equipment`. Branch C220107:

```python
    # -----------------------------------------------------------------------
    # 220107: Power Supplies
    # INDUCTIVE_DEC: $/J_stored basis for pulsed cap bank driver
    # Otherwise: MFE-calibrated power scaling
    # -----------------------------------------------------------------------
    if pulsed_conversion == PulsedConversion.INDUCTIVE_DEC:
        c220107 = cc.c_cap_allin_per_joule * e_stored_mj * 1e6 / 1e6  # $/J * J → M$
    else:
        c220107 = cc.power_supplies_base * (p_et / 1000.0) ** 0.7
```

Note: `e_stored_mj * 1e6` converts MJ to J; dividing by 1e6 converts $ to M$. Net: `c_cap_allin_per_joule * e_stored_mj`.

- [ ] **Step 5: Update C220109 for inductive DEC**

```python
    # -----------------------------------------------------------------------
    # 220109: Direct Energy Converter
    # INDUCTIVE_DEC: circuit-derived markups on pulsed driver
    # STEADY_STATE + f_dec > 0: electrostatic DEC (existing mirror logic)
    # Otherwise: zero
    # -----------------------------------------------------------------------
    if pulsed_conversion == PulsedConversion.INDUCTIVE_DEC:
        markup_cap = eta_dec * (1.0 + q_sci * f_ch) - 1.0
        delta_cap = c220107 * jnp.maximum(markup_cap, 0.0)
        delta_switch = c220107 * cc.markup_switch_bidir
        delta_inv = cc.c_inv_per_kw_net * p_net / 1e3  # $/kW * MW * 1000 / 1e6 = M$
        delta_ctrl = c220107 * cc.markup_controls
        c220109 = delta_cap + delta_switch + delta_inv + delta_ctrl
    elif p_dee > 0:
        # Existing electrostatic DEC for mirrors
        P_DEE_REF = 400.0
        p_dee_safe = jnp.where(p_dee > 0, p_dee, 1.0)
        c220109 = jnp.where(
            p_dee > 0,
            cc.dec_base * (p_dee_safe / P_DEE_REF) ** 0.7,
            0.0,
        )
    else:
        c220109 = 0.0
```

- [ ] **Step 6: Add cap bank replacement to CAS72 in `costs.py`**

In `cas70_om`, after the DEC grid replacement block, add:

```python
    # Cap bank scheduled replacement (INDUCTIVE_DEC only)
    if pulsed_conversion == PulsedConversion.INDUCTIVE_DEC and f_rep > 0:
        n_shots_per_year = f_rep * 8760.0 * 3600.0 * availability
        t_replace_yr = cc.cap_shot_lifetime / n_shots_per_year
        n_rep_cap = jnp.maximum(0.0, jnp.ceil(lifetime_yr / t_replace_yr) - 1.0)
        cap_cost = cas22_detail.get("C220107", 0.0) * n_mod
        pv_cap = 0.0
        for k in range(1, MAX_REP + 1):
            discount = (1 + interest_rate) ** (k * t_replace_yr)
            pv_cap = pv_cap + jnp.where(k <= n_rep_cap, cap_cost / discount, 0.0)
        cas72 = cas72 + pv_cap * crf
```

Pass `pulsed_conversion` and `f_rep` through from `model.py`.

- [ ] **Step 7: Update `model.py` to pass new params to costing functions**

Pass `pulsed_conversion`, `e_stored_mj`, `q_sci`, `f_ch`, `p_net`, `eta_dec` to `cas22_reactor_plant_equipment`. Pass `pulsed_conversion`, `f_rep` to `cas70_om`. Pass `p_the` to `cas23_turbine` and `p_th` to `cas26_heat_rejection` (from Step 5 of Task 5).

- [ ] **Step 8: Run tests**

Run: `cd /mnt/c/Users/talru/1cfe/1costingfe && python -m pytest tests/test_pulsed_dec_costing.py -v`
Expected: All PASS

- [ ] **Step 9: Run full test suite**

Run: `cd /mnt/c/Users/talru/1cfe/1costingfe && python -m pytest -x -v`
Expected: All PASS (fix any breakage from CAS23/26 signature changes)

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "feat: DEC costing — C220107 $/J basis, C220109 markups, CAS72 cap replacement, CAS23/26 thermal scaling"
```

---

### Task 7: Remove old IFE/MIF functions and update exports

**Files:**
- Modify: `src/costingfe/layers/physics.py`
- Modify: `src/costingfe/__init__.py`
- Delete: `tests/test_ife_mif_power_balance.py`

- [ ] **Step 1: Remove old functions from physics.py**

Delete `ife_forward_power_balance`, `ife_inverse_power_balance`, `mif_forward_power_balance`, `mif_inverse_power_balance`.

- [ ] **Step 2: Update `__init__.py`**

Export `PulsedConversion`:

```python
from costingfe.types import (
    PulsedConversion as PulsedConversion,
)
```

- [ ] **Step 3: Delete old test file**

```bash
rm tests/test_ife_mif_power_balance.py
```

- [ ] **Step 4: Run full test suite**

Run: `cd /mnt/c/Users/talru/1cfe/1costingfe && python -m pytest -x -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: remove legacy IFE/MIF power balance functions"
```

---

### Task 8: Rewrite `dhe3_pulsed_frc.py` example

**Files:**
- Modify: `examples/dhe3_pulsed_frc.py`

- [ ] **Step 1: Rewrite the example**

Replace the content to use the new API. Key changes:
- `ConfinementConcept.MAG_TARGET` with `pulsed_conversion=PulsedConversion.INDUCTIVE_DEC`
- `e_driver_mj=12.0`, `f_rep=1.0`, `eta_pin=0.95`, `eta_dec=0.85`, `f_pdv=0.80`
- `eta_th=0.0` (no thermal BOP)
- Remove all cost overrides for DEC-related accounts (CAS23, CAS26, C220107, C220109, C220200)
- Keep minimal overrides only for concept-specific geometry if needed

```python
"""Example: D-He3 Pulsed Colliding FRC — Helion-like concept.

Models a pulsed colliding Field-Reversed Configuration power plant operating
on D-He3 fuel with direct electromagnetic energy recovery.

Architecture:
  - Modular array of factory-built pulsed FRC generators (~50 MWe each)
  - Direct inductive energy recovery (no steam cycle)
  - Normal-conducting copper coils (no superconductors, no cryogenics)
  - D-He3 fuel with ~5% neutron fraction from DD side reactions
"""

from costingfe import CostModel, Fuel
from costingfe.defaults import load_costing_constants
from costingfe.types import ConfinementConcept, PulsedConversion

cc = load_costing_constants().replace(
    burn_fraction=0.10,
    fuel_recovery=0.95,
)
model = CostModel(
    concept=ConfinementConcept.MAG_TARGET,
    fuel=Fuel.DHE3,
    pulsed_conversion=PulsedConversion.INDUCTIVE_DEC,
    costing_constants=cc,
)

N_MODULES = 20
NET_ELECTRIC_MW = 1000.0

result = model.forward(
    net_electric_mw=NET_ELECTRIC_MW,
    availability=0.85,
    lifetime_yr=30,
    n_mod=N_MODULES,
    construction_time_yr=4.0,
    interest_rate=0.07,
    inflation_rate=0.02,
    noak=True,
    # Pulsed parameters
    e_driver_mj=12.0,
    f_rep=1.0,
    eta_pin=0.95,
    eta_dec=0.85,
    f_pdv=0.80,
    eta_th=0.0,         # no thermal BOP (pure DEC, D-He3)
    mn=1.0,             # no breeding blanket
    p_cryo=0.0,         # copper coils
    p_target=0.0,       # in-situ FRC formation
    p_coils=0.5,
    # Geometry (cylindrical, per module)
    R0=0.0,
    plasma_t=0.5,
    blanket_t=0.05,
    ht_shield_t=0.05,
    structure_t=0.10,
    vessel_t=0.10,
)

# Results
c = result.costs
pt = result.power_table

print(
    f"D-He3 Pulsed Colliding FRC — {N_MODULES} modules x "
    f"{NET_ELECTRIC_MW / N_MODULES:.0f} MWe"
)
print(f"  {NET_ELECTRIC_MW:.0f} MWe net, 85% availability, 30 yr lifetime")
print()
lcoe_ckwh = float(c.lcoe) / 10
print(
    f"LCOE: {c.lcoe:.1f} $/MWh ({lcoe_ckwh:.2f} c/kWh)"
    f" | Overnight: {c.overnight_cost:.0f} $/kW"
)
print(f"Fusion: {pt.p_fus:.0f} MW | Net: {pt.p_net:.0f} MW | Q_eng: {pt.q_eng:.1f}")
print(f"Recirculating fraction: {pt.rec_frac:.1%}")
print(f"Scientific Q (P_fus/P_driver): {pt.q_sci:.1f}")
print(f"E_stored: {pt.e_stored_mj:.1f} MJ/pulse | f_ch: {pt.f_ch:.2f}")
print()

# Cost breakdown
cas = [
    ("CAS10", "Preconstruction", c.cas10),
    ("CAS21", "Buildings", c.cas21),
    ("CAS22", "Reactor Plant Equipment", c.cas22),
    ("CAS23", "Turbine Plant", c.cas23),
    ("CAS24", "Electrical Plant", c.cas24),
    ("CAS25", "Miscellaneous", c.cas25),
    ("CAS26", "Heat Rejection", c.cas26),
    ("CAS28", "Digital Twin", c.cas28),
    ("CAS29", "Contingency", c.cas29),
    ("CAS30", "Indirect Costs", c.cas30),
    ("CAS40", "Owner's Costs", c.cas40),
    ("CAS50", "Supplementary", c.cas50),
    ("CAS60", "IDC", c.cas60),
    ("CAS70", "O&M (annualized)", c.cas70),
    ("CAS80", "Fuel (annualized)", c.cas80),
    ("CAS90", "Financial", c.cas90),
]

print(f"{'Code':<8} {'Account':<28} {'M$':>10}")
print("-" * 48)
for code, name, val in cas:
    print(f"{code:<8} {name:<28} {float(val):>10.1f}")
print("-" * 48)
print(f"{'':8} {'Total Capital':<28} {float(c.total_capital):>10.1f}")

print(f"\nOverridden: {', '.join(result.overridden) or 'None'}")
```

- [ ] **Step 2: Run the example**

Run: `cd /mnt/c/Users/talru/1cfe/1costingfe && python examples/dhe3_pulsed_frc.py`
Expected: Produces valid output with CAS23 = 0, no cost overrides

- [ ] **Step 3: Run full test suite**

Run: `cd /mnt/c/Users/talru/1cfe/1costingfe && python -m pytest -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat: rewrite D-He3 pulsed FRC example using inductive DEC API"
```
