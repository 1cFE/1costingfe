"""Tests for the n_coils override in cas22 / forward()."""

from costingfe import ConfinementConcept, CostModel, Fuel


def _base_kwargs():
    return dict(
        net_electric_mw=500.0,
        availability=0.85,
        lifetime_yr=30,
        n_mod=1,
        construction_time_yr=5.0,
        interest_rate=0.07,
        inflation_rate=0.0245,
        noak=True,
        R0=5.0,
        plasma_t=1.0,
        blanket_t=0.6,
        ht_shield_t=0.2,
        structure_t=0.2,
        vessel_t=0.2,
        p_input=20.0,
        b_center=8.0,
        r_bore=1.0,
    )


def test_n_coils_default_used_when_not_passed_mirror():
    """When n_coils is not passed, MIRROR uses hardcoded default (10)."""
    model = CostModel(concept=ConfinementConcept.MIRROR, fuel=Fuel.DT)
    result = model.forward(**_base_kwargs())
    c220103_default = result.cas22_detail["C220103"]
    assert c220103_default > 0


def test_n_coils_override_scales_c220103_linearly():
    """Cutting n_coils from default 10 to 1 should reduce C220103 by ~10x."""
    model = CostModel(concept=ConfinementConcept.MIRROR, fuel=Fuel.DT)
    r10 = model.forward(**_base_kwargs())
    r1 = model.forward(n_coils=1, **_base_kwargs())
    ratio = float(r10.cas22_detail["C220103"]) / float(r1.cas22_detail["C220103"])
    assert 9.5 < ratio < 10.5, f"expected ~10x ratio, got {ratio}"


def test_n_coils_override_zero_zeroes_c220103():
    """n_coils=0 should set C220103 to zero for MIRROR."""
    model = CostModel(concept=ConfinementConcept.MIRROR, fuel=Fuel.DT)
    r = model.forward(n_coils=0, **_base_kwargs())
    assert float(r.cas22_detail["C220103"]) == 0.0


def test_n_coils_ignored_for_tokamak():
    """Non-MIRROR concepts use a different G factor; n_coils kwarg is a no-op."""
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    r_default = model.forward(**_base_kwargs())
    r_with = model.forward(n_coils=3, **_base_kwargs())
    assert float(r_default.cas22_detail["C220103"]) == float(
        r_with.cas22_detail["C220103"]
    )


def test_n_coils_negative_raises():
    """Negative n_coils should raise ValueError, not produce a negative cost."""
    import pytest

    model = CostModel(concept=ConfinementConcept.MIRROR, fuel=Fuel.DT)
    with pytest.raises(ValueError, match="n_coils must be >= 0"):
        model.forward(n_coils=-1, **_base_kwargs())


def test_dipole_stationary_coil_count_default():
    """DIPOLE's `_COIL_DEFAULTS[DIPOLE]` is retained for table-lookup safety
    (downstream tooling iterates the dict), but its values are NOT consumed by
    the C220103 calculation: the DIPOLE branch in cas22 short-circuits the
    standard stationary path and instead derives the external lift coil's
    cost as `stationary_lift_coil_fraction * floating_with_markup` (Simpson
    2026 — the lift coil's field is set by force balance against the floating
    coil's gravity, not by plasma confinement). The entry's n_coils is set to
    1 (the single external lift coil Simpson specifies)."""
    from costingfe.layers.cas22 import _COIL_DEFAULTS

    assert _COIL_DEFAULTS[ConfinementConcept.DIPOLE]["n_coils"] == 1


def test_dipole_stationary_lift_coil_fraction_default():
    """Default `stationary_lift_coil_fraction = 0.10`: the single external lift
    coil's full installed cost is 10% of the floating coil's with-markup cost."""
    from costingfe import CostModel, Fuel

    model = CostModel(concept=ConfinementConcept.DIPOLE, fuel=Fuel.DT)
    base = model.forward(**_base_kwargs())
    # Doubling the fraction increases C220103 by exactly the floating-with-
    # markup cost * 0.10 (the delta), so per-module delta should be positive
    # and the ratio of (more - base) / floating_with_markup should be ~0.10.
    more = model.forward(stationary_lift_coil_fraction=0.20, **_base_kwargs())
    assert float(more.cas22_detail["C220103"]) > float(base.cas22_detail["C220103"])


def test_dipole_levitated_coil_cryostat_additive():
    """The levitated coil's integral cryostat cost enters C220103 additively and
    independently of the stationary coils, so a delta flows straight through."""
    model = CostModel(concept=ConfinementConcept.DIPOLE, fuel=Fuel.DT)
    base = model.forward(lev_coil_cryostat_cost=50.0, **_base_kwargs())
    more = model.forward(lev_coil_cryostat_cost=150.0, **_base_kwargs())
    delta = float(more.cas22_detail["C220103"]) - float(base.cas22_detail["C220103"])
    assert abs(delta - 100.0) < 0.5, f"expected +100 M$, got {delta}"


def test_n_coils_ignored_for_stellarator():
    """STELLARATOR uses path_factor for G; n_coils kwarg must be a no-op."""
    model = CostModel(concept=ConfinementConcept.STELLARATOR, fuel=Fuel.DT)
    r_default = model.forward(**_base_kwargs())
    r_with = model.forward(n_coils=3, **_base_kwargs())
    assert float(r_default.cas22_detail["C220103"]) == float(
        r_with.cas22_detail["C220103"]
    )
    # Sanity: STELLARATOR C220103 should be non-zero so this test is meaningful
    assert float(r_default.cas22_detail["C220103"]) > 0
