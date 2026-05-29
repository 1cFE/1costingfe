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
        b_max=8.0,
        r_coil=1.0,
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
    """DIPOLE's _COIL_DEFAULTS n_coils is the STATIONARY levitation coil set,
    which is >=2 (the levitated coil itself is costed as a separate term)."""
    from costingfe.layers.cas22 import _COIL_DEFAULTS

    assert _COIL_DEFAULTS[ConfinementConcept.DIPOLE]["n_coils"] >= 2


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
