from costingfe.defaults import load_costing_constants
from costingfe.layers.cas22 import cas22_reactor_plant_equipment
from costingfe.layers.geometry import RadialBuild, compute_geometry
from costingfe.types import CoilMaterial, ConfinementConcept, ConfinementFamily, Fuel

CC = load_costing_constants()

# Reference tokamak geometry for tests
RB = RadialBuild(R0=6.2, plasma_t=2.0, elon=1.7, blanket_t=0.70)
GEO = compute_geometry(RB, ConfinementConcept.TOKAMAK)
BLANKET_VOL = GEO.firstwall_vol + GEO.blanket_vol + GEO.reflector_vol
SHIELD_VOL = GEO.ht_shield_vol + GEO.lt_shield_vol
STRUCTURE_VOL = GEO.structure_vol
VESSEL_VOL = GEO.vessel_vol


def _make_cas22(fuel=Fuel.DT, n_mod=1, blanket_t=0.70):
    """Helper to compute CAS22 with geometry."""
    rb = RadialBuild(R0=6.2, plasma_t=2.0, elon=1.7, blanket_t=blanket_t)
    geo = compute_geometry(rb, ConfinementConcept.TOKAMAK)
    return cas22_reactor_plant_equipment(
        CC,
        p_net=1000.0,
        p_th=2500.0,
        p_et=1100.0,
        p_fus=2300.0,
        p_cryo=0.5,
        n_mod=n_mod,
        fuel=fuel,
        noak=True,
        blanket_vol=geo.firstwall_vol + geo.blanket_vol + geo.reflector_vol,
        shield_vol=geo.ht_shield_vol + geo.lt_shield_vol,
        structure_vol=geo.structure_vol,
        vessel_vol=geo.vessel_vol,
        family=ConfinementFamily.MFE,
        concept=ConfinementConcept.TOKAMAK,
        b_max=12.0,
        r_coil=1.85,
        coil_material=CoilMaterial.REBCO_HTS,
        p_nbi=50.0,
        p_icrf=0.0,
        p_ecrh=0.0,
        p_lhcd=0.0,
    )


def test_cas22_dt_has_breeding_blanket():
    """DT should include tritium breeding blanket cost."""
    result = _make_cas22(fuel=Fuel.DT)
    assert result["C220101"] > 0  # first wall + blanket
    assert result["C220000"] > 0  # total


def test_cas22_pb11_no_breeding():
    """pB11 should have cheaper blanket (no breeding)."""
    dt = _make_cas22(fuel=Fuel.DT)
    pb11 = _make_cas22(fuel=Fuel.PB11)
    assert pb11["C220101"] < dt["C220101"]  # no breeding blanket


def test_cas22_isotope_separation_zeroed():
    """CAS220112 should be zero — isotope procurement is in CAS80 market prices."""
    for fuel in [Fuel.DT, Fuel.DD, Fuel.DHE3, Fuel.PB11]:
        result = _make_cas22(fuel=fuel)
        assert result["C220112"] == 0.0, (
            f"C220112 should be 0 for {fuel.value} (no on-site separation plant)"
        )


def test_cas22_fuel_handling_tritium_containment():
    """DT should have much higher fuel handling cost (tritium containment)."""
    dt = _make_cas22(fuel=Fuel.DT)
    pb11 = _make_cas22(fuel=Fuel.PB11)
    assert dt["C220500"] > pb11["C220500"]


def test_cas22_scales_with_n_mod():
    """Total CAS22 should scale with number of modules."""
    single = _make_cas22(n_mod=1)
    double = _make_cas22(n_mod=2)
    assert double["C220000"] > single["C220000"]


def test_cas22_all_subaccounts_present():
    """Result should contain all expected sub-account keys (no C220119)."""
    result = _make_cas22()
    expected_keys = [
        "C220101",
        "C220102",
        "C220103",
        "C220104",
        "C220105",
        "C220106",
        "C220107",
        "C220108",
        "C220109",
        "C220110",
        "C220111",
        "C220112",
        "C220200",
        "C220300",
        "C220400",
        "C220500",
        "C220600",
        "C220700",
        "C220000",
    ]
    for key in expected_keys:
        assert key in result, f"Missing sub-account {key}"
        assert result[key] >= 0, f"Sub-account {key} is negative"
    # C220119 removed — replacement is now CAS72 (annualized, not capital)
    assert "C220119" not in result


def test_cas22_blanket_scales_with_thickness():
    """Thicker blanket should cost more (volume-based costing)."""
    thin = _make_cas22(blanket_t=0.50)
    thick = _make_cas22(blanket_t=0.90)
    assert thick["C220101"] > thin["C220101"]
    assert thick["C220000"] > thin["C220000"]


def test_cas22_shield_volume_based():
    """Shield cost should be proportional to volume."""
    result = _make_cas22()
    expected_shield = CC.shield_unit_cost * SHIELD_VOL * 1.0  # DT scale=1.0
    assert abs(result["C220102"] - expected_shield) < 0.1


def test_cas22_structure_volume_based():
    """Structure cost should use volume-based costing."""
    result = _make_cas22()
    expected = CC.structure_unit_cost * STRUCTURE_VOL
    assert abs(result["C220105"] - expected) < 0.1


# ---- CAS220108: Divertor vs Target Factory ----


def _make_cas22_with_family(family=ConfinementFamily.MFE):
    """Helper to compute CAS22 with a specific confinement family."""
    return cas22_reactor_plant_equipment(
        CC,
        p_net=1000.0,
        p_th=2500.0,
        p_et=1100.0,
        p_fus=2300.0,
        p_cryo=0.5,
        n_mod=1,
        fuel=Fuel.DT,
        noak=True,
        blanket_vol=BLANKET_VOL,
        shield_vol=SHIELD_VOL,
        structure_vol=STRUCTURE_VOL,
        vessel_vol=VESSEL_VOL,
        family=family,
        concept=ConfinementConcept.TOKAMAK,
        b_max=12.0,
        r_coil=1.85,
        coil_material=CoilMaterial.REBCO_HTS,
        p_nbi=50.0,
        p_icrf=0.0,
        p_ecrh=0.0,
        p_lhcd=0.0,
    )


def test_cas220108_mfe_uses_divertor():
    """MFE should use divertor_base for CAS220108."""
    result = _make_cas22_with_family(ConfinementFamily.MFE)
    expected = CC.divertor_base * (2500.0 / 1000.0) ** 0.5
    assert abs(result["C220108"] - expected) < 0.01


def test_cas220108_ife_uses_target_factory():
    """IFE should use target_factory_base (larger than divertor)."""
    mfe = _make_cas22_with_family(ConfinementFamily.MFE)
    ife = _make_cas22_with_family(ConfinementFamily.IFE)
    msg = "Target factory should cost more than divertor"
    assert ife["C220108"] > mfe["C220108"], msg
    expected = CC.target_factory_base * (1100.0 / 1000.0) ** 0.7
    assert abs(ife["C220108"] - expected) < 0.01


# ---- Plant-wide accounts must use total plant power for n_mod > 1 ----


def test_plant_wide_accounts_scale_with_n_mod():
    """C220400/500/600/700 should use n_mod * p_th / n_mod * p_net.

    At n_mod=2, these plant-wide accounts should be larger than at n_mod=1
    because the plant handles twice the total power.
    """
    single = _make_cas22(n_mod=1)
    double = _make_cas22(n_mod=2)
    for acct in ["C220400", "C220500", "C220600", "C220700"]:
        assert double[acct] > single[acct], (
            f"{acct} should increase with n_mod (plant-wide system serves all modules)"
        )


def test_plant_wide_c220200_scales_with_n_mod():
    """C220200 (coolant) should increase with n_mod.

    C220201 (primary) scales linearly with n_mod * p_net.
    C220202 (intermediate) scales sub-linearly with n_mod * p_th.
    Both should increase, so total should exceed single-module value.
    """
    single = _make_cas22(n_mod=1)
    double = _make_cas22(n_mod=2)
    # At n_mod=2: C220201 doubles, C220202 increases sub-linearly
    # Total should be between 1x and 2x single
    assert double["C220200"] > single["C220200"]
    # C220201 dominates and doubles, so total should be well above 1.5x
    assert double["C220200"] > 1.5 * single["C220200"]


# ---- CAS220103: Conductor scaling coil model ----


def _make_cas22_coil(
    b_max=12.0,
    r_coil=1.85,
    concept=ConfinementConcept.TOKAMAK,
    coil_material=CoilMaterial.REBCO_HTS,
):
    """Helper for coil model tests."""
    rb = RadialBuild(R0=6.2, plasma_t=2.0, elon=1.7, blanket_t=0.70)
    geo = compute_geometry(rb, ConfinementConcept.TOKAMAK)
    return cas22_reactor_plant_equipment(
        CC,
        p_net=1000.0,
        p_th=2500.0,
        p_et=1100.0,
        p_fus=2300.0,
        p_cryo=0.5,
        n_mod=1,
        fuel=Fuel.DT,
        noak=True,
        blanket_vol=geo.firstwall_vol + geo.blanket_vol + geo.reflector_vol,
        shield_vol=geo.ht_shield_vol + geo.lt_shield_vol,
        structure_vol=geo.structure_vol,
        vessel_vol=geo.vessel_vol,
        family=ConfinementFamily.MFE,
        concept=concept,
        b_max=b_max,
        r_coil=r_coil,
        coil_material=coil_material,
        p_nbi=50.0,
        p_icrf=0.0,
        p_ecrh=0.0,
        p_lhcd=0.0,
    )


def test_cas220103_conductor_scaling_formula():
    """Coil cost should use conductor scaling: G * B * R^2 / (mu0 * 1000)
    * $/kAm * markup."""
    import math

    result = _make_cas22_coil(b_max=12.0, r_coil=1.85)
    mu0 = 4 * math.pi * 1e-7
    G = 4 * math.pi**2  # tokamak
    total_kAm = G * 12.0 * 1.85**2 / (mu0 * 1000)
    conductor_cost = total_kAm * 50.0 / 1e6  # REBCO default
    expected = conductor_cost * 8.0  # tokamak markup
    assert abs(result["C220103"] - expected) < 0.1


def test_cas220103_scales_with_b_field():
    """Higher B-field -> more conductor -> higher cost (linear in B)."""
    low_b = _make_cas22_coil(b_max=8.0)
    high_b = _make_cas22_coil(b_max=16.0)
    assert high_b["C220103"] > low_b["C220103"]
    # Linear in B, so 2x B -> 2x cost
    assert abs(high_b["C220103"] / low_b["C220103"] - 2.0) < 0.01


def test_cas220103_scales_with_r_coil_squared():
    """Larger coil bore -> quadratically more conductor."""
    small = _make_cas22_coil(r_coil=1.0)
    large = _make_cas22_coil(r_coil=2.0)
    assert abs(large["C220103"] / small["C220103"] - 4.0) < 0.01


def test_cas220103_stellarator_higher_than_tokamak():
    """Stellarator: higher markup + path_factor -> more expensive."""
    tok = _make_cas22_coil(concept=ConfinementConcept.TOKAMAK)
    stell = _make_cas22_coil(concept=ConfinementConcept.STELLARATOR)
    assert stell["C220103"] > tok["C220103"]


def test_cas220103_mirror_cheaper_than_tokamak():
    """Mirror: lower markup -> cheaper."""
    tok = _make_cas22_coil(concept=ConfinementConcept.TOKAMAK)
    mir = _make_cas22_coil(concept=ConfinementConcept.MIRROR)
    assert mir["C220103"] < tok["C220103"]


def test_cas220103_material_affects_cost():
    """Different coil materials have different conductor costs."""
    rebco = _make_cas22_coil(coil_material=CoilMaterial.REBCO_HTS)
    nb3sn = _make_cas22_coil(coil_material=CoilMaterial.NB3SN)
    assert rebco["C220103"] > nb3sn["C220103"]  # REBCO $50 vs Nb3Sn $7


# ---- CAS220104: Per-MW heating sub-accounts ----


def _make_cas22_heating(p_nbi=50.0, p_icrf=0.0, p_ecrh=0.0, p_lhcd=0.0):
    """Helper for heating model tests."""
    return cas22_reactor_plant_equipment(
        CC,
        p_net=1000.0,
        p_th=2500.0,
        p_et=1100.0,
        p_fus=2300.0,
        p_cryo=0.5,
        n_mod=1,
        fuel=Fuel.DT,
        noak=True,
        blanket_vol=BLANKET_VOL,
        shield_vol=SHIELD_VOL,
        structure_vol=STRUCTURE_VOL,
        vessel_vol=VESSEL_VOL,
        family=ConfinementFamily.MFE,
        concept=ConfinementConcept.TOKAMAK,
        b_max=12.0,
        r_coil=1.85,
        coil_material=CoilMaterial.REBCO_HTS,
        p_nbi=p_nbi,
        p_icrf=p_icrf,
        p_ecrh=p_ecrh,
        p_lhcd=p_lhcd,
    )


def test_cas220104_per_mw_linear():
    """Heating cost should be linear: cost_per_MW * power for each type."""
    result = _make_cas22_heating(p_nbi=50.0, p_icrf=0.0, p_ecrh=0.0, p_lhcd=0.0)
    expected = CC.heating_nbi_per_mw * 50.0
    assert abs(result["C220104"] - expected) < 0.01


def test_cas220104_multi_type_sum():
    """Multiple heating types should sum linearly."""
    result = _make_cas22_heating(p_nbi=50.0, p_icrf=25.0, p_ecrh=10.0, p_lhcd=15.0)
    expected = (
        CC.heating_nbi_per_mw * 50.0
        + CC.heating_icrf_per_mw * 25.0
        + CC.heating_ecrh_per_mw * 10.0
        + CC.heating_lhcd_per_mw * 15.0
    )
    assert abs(result["C220104"] - expected) < 0.01


def test_cas220104_scales_linearly_with_power():
    """Doubling heating power should exactly double cost."""
    single = _make_cas22_heating(p_nbi=50.0)
    double = _make_cas22_heating(p_nbi=100.0)
    assert abs(double["C220104"] / single["C220104"] - 2.0) < 0.01


def test_cas220104_nbi_most_expensive_per_mw():
    """NBI should be more expensive per MW than ICRF/ECRH/LHCD."""
    nbi = _make_cas22_heating(p_nbi=50.0, p_icrf=0.0)
    icrf = _make_cas22_heating(p_nbi=0.0, p_icrf=50.0)
    assert nbi["C220104"] > icrf["C220104"]  # NBI ~$7/MW vs ICRF ~$4/MW


# ---- CAS220110: Remote Handling & Maintenance Equipment ----


def test_cas220110_dt_has_remote_handling():
    """DT should have substantial remote handling cost."""
    result = _make_cas22(fuel=Fuel.DT)
    assert result["C220110"] > 100  # > $100M for DT tokamak


def test_cas220110_pb11_much_cheaper():
    """pB11 should have much cheaper maintenance equipment than DT."""
    dt = _make_cas22(fuel=Fuel.DT)
    pb11 = _make_cas22(fuel=Fuel.PB11)
    assert pb11["C220110"] < dt["C220110"] * 0.25  # at least 4x cheaper


def test_cas220110_concept_scales():
    """Mirror (linear) should be cheaper than tokamak (toroidal) for same fuel."""
    tok = cas22_reactor_plant_equipment(
        CC,
        p_net=1000.0,
        p_th=2500.0,
        p_et=1100.0,
        p_fus=2300.0,
        p_cryo=0.5,
        n_mod=1,
        fuel=Fuel.DT,
        noak=True,
        blanket_vol=BLANKET_VOL,
        shield_vol=SHIELD_VOL,
        structure_vol=STRUCTURE_VOL,
        vessel_vol=VESSEL_VOL,
        family=ConfinementFamily.MFE,
        concept=ConfinementConcept.TOKAMAK,
        b_max=12.0,
        r_coil=1.85,
        coil_material=CoilMaterial.REBCO_HTS,
        p_nbi=50.0,
        p_icrf=0.0,
        p_ecrh=0.0,
        p_lhcd=0.0,
    )
    mir = cas22_reactor_plant_equipment(
        CC,
        p_net=1000.0,
        p_th=2500.0,
        p_et=1100.0,
        p_fus=2300.0,
        p_cryo=0.5,
        n_mod=1,
        fuel=Fuel.DT,
        noak=True,
        blanket_vol=BLANKET_VOL,
        shield_vol=SHIELD_VOL,
        structure_vol=STRUCTURE_VOL,
        vessel_vol=VESSEL_VOL,
        family=ConfinementFamily.MFE,
        concept=ConfinementConcept.MIRROR,
        b_max=12.0,
        r_coil=1.85,
        coil_material=CoilMaterial.REBCO_HTS,
        p_nbi=50.0,
        p_icrf=0.0,
        p_ecrh=0.0,
        p_lhcd=0.0,
    )
    assert mir["C220110"] < tok["C220110"]
