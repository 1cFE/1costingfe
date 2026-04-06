from costingfe.defaults import (
    load_costing_constants,
    load_engineering_defaults,
)


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
    assert cc.dec_base == 100.0
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
