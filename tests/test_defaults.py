from pathlib import Path

import pytest
import yaml

from costingfe.defaults import (
    load_costing_constants,
    load_engineering_defaults,
)
from costingfe.types import BlanketFill, BlanketForm


def test_load_costing_constants():
    """Should load defaults from YAML."""
    cc = load_costing_constants()
    assert cc.site_permits > 0
    assert cc.licensing_cost_dt > 0
    assert len(cc.building_costs) > 10


def test_load_engineering_defaults():
    """Should load MFE tokamak defaults."""
    ed = load_engineering_defaults("steady_state_tokamak")
    assert ed["p_input"] > 0
    # eta_th is no longer in the concept YAML — it comes from POWER_CYCLE_DEFAULTS
    assert "eta_th" not in ed


def test_costing_constants_override():
    """Should allow field overrides via replace()."""
    cc = load_costing_constants()
    cc_custom = cc.replace(site_permits=99.0)
    assert cc_custom.site_permits == 99.0
    assert cc.site_permits != 99.0  # original unchanged


def test_missing_concept_returns_empty():
    """Unknown concept should return empty dict, not error."""
    ed = load_engineering_defaults("nonexistent_concept")
    assert ed == {}


def test_dec_constants_exist():
    """DEC add-on constants should be loadable from defaults."""
    from costingfe.defaults import load_costing_constants

    cc = load_costing_constants()
    assert cc.dec_base == 125.0
    assert cc.dec_grid_cost == 12.0
    assert cc.dec_grid_lifetime_dt == 2.0
    assert cc.dec_grid_lifetime_dd == 3.0
    assert cc.dec_grid_lifetime_dhe3 == 4.0
    assert cc.dec_grid_lifetime_pb11 == 3.0


def test_dec_grid_lifetime_accessor():
    """dec_grid_lifetime(fuel) should return fuel-specific values."""
    from costingfe.defaults import load_costing_constants
    from costingfe.types import Fuel

    cc = load_costing_constants()
    assert cc.dec_grid_lifetime(Fuel.DT) == 2.0
    assert cc.dec_grid_lifetime(Fuel.DD) == 3.0
    assert cc.dec_grid_lifetime(Fuel.DHE3) == 4.0
    assert cc.dec_grid_lifetime(Fuel.PB11) == 3.0


_DEFAULTS_DIR = Path(__file__).parent.parent / "src" / "costingfe" / "data" / "defaults"
_CONCEPT_YAMLS = sorted(
    p for p in _DEFAULTS_DIR.glob("*.yaml") if p.name != "costing_constants.yaml"
)


@pytest.mark.parametrize("yaml_path", _CONCEPT_YAMLS, ids=lambda p: p.stem)
def test_every_concept_yaml_declares_blanket(yaml_path):
    """Every concept YAML must declare blanket_form and blanket_fill, and the
    pair must be valid per BlanketForm.valid_fills."""
    data = yaml.safe_load(yaml_path.read_text())
    assert "blanket_form" in data, f"{yaml_path.name} missing blanket_form"
    assert "blanket_fill" in data, f"{yaml_path.name} missing blanket_fill"

    form = BlanketForm(data["blanket_form"])
    fill = BlanketFill(data["blanket_fill"])
    assert fill in form.valid_fills, (
        f"{yaml_path.name}: blanket_fill={fill.value!r} not valid for "
        f"blanket_form={form.value!r}"
    )


@pytest.mark.parametrize("yaml_path", _CONCEPT_YAMLS, ids=lambda p: p.stem)
def test_every_concept_yaml_declares_burn_fraction_and_fuel_recovery(yaml_path):
    """Every concept YAML must declare burn_fraction and fuel_recovery; both
    must lie in (0, 1].

    Background: these two physics knobs used to live in costing_constants.yaml
    as global defaults but are concept-specific (burn_fraction) or declared
    per-concept by policy (fuel_recovery). See
    docs/physics/burn_fraction.md.
    """
    data = yaml.safe_load(yaml_path.read_text())
    for key in ("burn_fraction", "fuel_recovery"):
        assert key in data, f"{yaml_path.name} missing {key}"
        value = data[key]
        assert isinstance(value, int | float), (
            f"{yaml_path.name}: {key}={value!r} is not numeric"
        )
        assert 0 < value <= 1, f"{yaml_path.name}: {key}={value} is outside (0, 1]"
