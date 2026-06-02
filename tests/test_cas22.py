import pytest

from costingfe.defaults import load_costing_constants
from costingfe.layers.cas22 import cas22_reactor_plant_equipment
from costingfe.layers.geometry import RadialBuild, compute_geometry
from costingfe.types import (
    BlanketForm,
    CoilMaterial,
    ConfinementConcept,
    ConfinementFamily,
    Fuel,
)

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
        family=ConfinementFamily.STEADY_STATE,
        concept=ConfinementConcept.TOKAMAK,
        b_center=12.0,
        r_bore=1.85,
        coil_material=CoilMaterial.REBCO_HTS,
        blanket_form=BlanketForm.LIQUID_METAL,
        p_nbi=50.0,
        p_icrf=0.0,
        p_ecrh=0.0,
        p_lhcd=0.0,
        p_driver=0.0,
        f_dec=0.0,
        p_dee=0.0,
        burn_fraction=0.05,
        vac_op_pressure_pa=1.0,
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
    """CAS220112 should be zero - isotope procurement is in CAS80 market prices."""
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


def test_cas22_multi_unit_labor_discount():
    """Installation labor (C220111) for unit 2+ at the same site is discounted
    by cc.multi_unit_labor_factor. Equipment (C220101..C220110) is not
    discounted: each module is still a manufactured copy. The per-module
    sub-account values returned in the dict are unchanged; the discount
    shows up in the C220000 rollup."""
    single = _make_cas22(n_mod=1)
    double = _make_cas22(n_mod=2)

    # Per-module values returned for C220101 and C220111 are unchanged
    assert double["C220101"] == single["C220101"]
    assert double["C220111"] == single["C220111"]

    # Difference between n_mod=1 and n_mod=2 totals = one extra module,
    # equipment at full price + labor at multi_unit_labor_factor of full,
    # plus the increase in plant-wide accounts (which use total power).
    # The labor saving at n_mod=2 vs naive 2x: c220111 * (1 - factor)
    naive_double = 2.0 * (
        single["C220101"]
        + single["C220102"]
        + single["C220103"]
        + single["C220104"]
        + single["C220105"]
        + single["C220106"]
        + single["C220107"]
        + single["C220108"]
        + single["C220109"]
        + single["C220110"]
        + single["C220111"]
        + single["C220112"]
    ) + (
        double["C220200"]
        + double["C220300"]
        + double["C220400"]
        + double["C220500"]
        + double["C220600"]
        + double["C220700"]
    )
    expected_labor_saving = single["C220111"] * (1.0 - CC.multi_unit_labor_factor)
    assert double["C220000"] == pytest.approx(
        naive_double - expected_labor_saving, rel=1e-5
    )


def test_cas22_n_mod_one_unchanged_by_multi_unit_factor():
    """At n_mod=1, the multi-unit labor factor must have no effect: there
    are no 'subsequent units' to discount."""
    cc_aggressive = CC.replace(multi_unit_labor_factor=0.50)
    rb = RadialBuild(R0=6.2, plasma_t=2.0, elon=1.7, blanket_t=0.70)
    geo = compute_geometry(rb, ConfinementConcept.TOKAMAK)
    base_kwargs = dict(
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
        family=ConfinementFamily.STEADY_STATE,
        concept=ConfinementConcept.TOKAMAK,
        b_center=12.0,
        r_bore=1.85,
        coil_material=CoilMaterial.REBCO_HTS,
        blanket_form=BlanketForm.LIQUID_METAL,
        p_nbi=50.0,
        p_icrf=0.0,
        p_ecrh=0.0,
        p_lhcd=0.0,
        p_driver=0.0,
        f_dec=0.0,
        p_dee=0.0,
        burn_fraction=0.05,
        vac_op_pressure_pa=1.0,
    )
    base = cas22_reactor_plant_equipment(CC, **base_kwargs)
    aggr = cas22_reactor_plant_equipment(cc_aggressive, **base_kwargs)
    assert base["C220000"] == aggr["C220000"]


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
    # C220119 removed - replacement is now CAS72 (annualized, not capital)
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


def _make_cas22_with_family(family=ConfinementFamily.STEADY_STATE):
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
        b_center=12.0,
        r_bore=1.85,
        coil_material=CoilMaterial.REBCO_HTS,
        blanket_form=BlanketForm.LIQUID_METAL,
        p_nbi=50.0,
        p_icrf=0.0,
        p_ecrh=0.0,
        p_lhcd=0.0,
        p_driver=0.0,
        f_dec=0.0,
        p_dee=0.0,
        burn_fraction=0.05,
        vac_op_pressure_pa=1.0,
    )


def test_cas220108_mfe_uses_divertor():
    """MFE should use divertor_base for CAS220108."""
    result = _make_cas22_with_family(ConfinementFamily.STEADY_STATE)
    expected = CC.divertor_base * (2500.0 / 1000.0) ** 0.5
    assert abs(result["C220108"] - expected) < 0.01


def test_cas220108_ife_uses_target_factory():
    """IFE should use target_factory_base (larger than divertor)."""
    mfe = _make_cas22_with_family(ConfinementFamily.STEADY_STATE)
    ife = _make_cas22_with_family(ConfinementFamily.PULSED)
    msg = "Target factory should cost more than divertor"
    assert ife["C220108"] > mfe["C220108"], msg
    expected = CC.target_factory_base * (1100.0 / 1000.0) ** 0.7
    assert abs(ife["C220108"] - expected) < 0.01


def test_cas220108_dipole_has_no_divertor():
    """Levitated dipole exhausts through the loss cone at the top/bottom
    openings of the chamber -- no W monoblock divertor cassette to cost.
    C220108 must be zero for DIPOLE even though it is STEADY_STATE."""
    from costingfe import CostModel, Fuel

    r = CostModel(concept=ConfinementConcept.DIPOLE, fuel=Fuel.DT).forward(
        net_electric_mw=208.0,
        availability=0.85,
        lifetime_yr=30,
    )
    assert float(r.cas22_detail["C220108"]) == 0.0


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
    b_center=12.0,
    r_bore=1.85,
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
        family=ConfinementFamily.STEADY_STATE,
        concept=concept,
        b_center=b_center,
        r_bore=r_bore,
        coil_material=coil_material,
        blanket_form=BlanketForm.LIQUID_METAL,
        p_nbi=50.0,
        p_icrf=0.0,
        p_ecrh=0.0,
        p_lhcd=0.0,
        p_driver=0.0,
        f_dec=0.0,
        p_dee=0.0,
        burn_fraction=0.05,
        vac_op_pressure_pa=1.0,
    )


def test_cas220103_conductor_scaling_formula():
    """Coil cost should use conductor scaling: G * B * R^2 / (mu0 * 1000)
    * $/kAm * markup."""
    import math

    result = _make_cas22_coil(b_center=12.0, r_bore=1.85)
    mu0 = 4 * math.pi * 1e-7
    G = 4 * math.pi**2  # tokamak
    total_kAm = G * 12.0 * 1.85**2 / (mu0 * 1000)
    conductor_cost = total_kAm * 50.0 / 1e6  # REBCO default
    expected = conductor_cost * 8.0  # tokamak markup
    assert abs(result["C220103"] - expected) < 0.1


def test_cas220103_biot_savart_consistency_single_loop():
    """For a single circular loop the formula must reduce to the textbook
    Biot-Savart identity:

        NI    = 2 * b_center * R / mu_0          (B at loop center)
        kA*m  = NI * (2 pi R) / 1000             (one turn of length 2 pi R)
              = 4 pi * b_center * R^2 / (mu_0 * 1000)

    A single-loop concept (DIPOLE with n_coils=1) must satisfy this for any
    consistent (b_center, r_bore) triple - independent of cost calibration.
    """
    import math

    from costingfe.layers.cas22 import _COIL_DEFAULTS, _compute_geometry_factor

    for b_center, r_bore in [(6.26, 5.3), (12.0, 1.85), (0.5, 1.85), (23.0, 1.0)]:
        path_factor = _COIL_DEFAULTS[ConfinementConcept.DIPOLE]["path_factor"]
        G = _compute_geometry_factor(ConfinementConcept.DIPOLE, path_factor, n_coils=1)
        mu0 = 4 * math.pi * 1e-7
        kAm_formula = G * b_center * r_bore**2 / (mu0 * 1000)

        # Biot-Savart reference path for one circular loop
        NI = 2 * b_center * r_bore / mu0
        kAm_biot_savart = NI * (2 * math.pi * r_bore) / 1000

        assert math.isclose(kAm_formula, kAm_biot_savart, rel_tol=1e-12), (
            f"kA*m formula {kAm_formula} != Biot-Savart {kAm_biot_savart} "
            f"for (b_center={b_center}, r_bore={r_bore})"
        )


def test_cas220103_scales_with_b_field():
    """Higher B-field -> more conductor -> higher cost (linear in B)."""
    low_b = _make_cas22_coil(b_center=8.0)
    high_b = _make_cas22_coil(b_center=16.0)
    assert high_b["C220103"] > low_b["C220103"]
    # Linear in B, so 2x B -> 2x cost
    assert abs(high_b["C220103"] / low_b["C220103"] - 2.0) < 0.01


def test_cas220103_scales_with_r_bore_squared():
    """Larger coil bore -> quadratically more conductor."""
    small = _make_cas22_coil(r_bore=1.0)
    large = _make_cas22_coil(r_bore=2.0)
    assert abs(large["C220103"] / small["C220103"] - 4.0) < 0.01


def test_cas220103_stellarator_higher_than_tokamak():
    """Stellarator: higher markup + path_factor -> more expensive."""
    tok = _make_cas22_coil(concept=ConfinementConcept.TOKAMAK)
    stell = _make_cas22_coil(concept=ConfinementConcept.STELLARATOR)
    assert stell["C220103"] > tok["C220103"]


def test_cas220103_mirror_comparable_to_tokamak():
    """Mirror: lower markup (2.5x vs 8x) is offset by n_coils=10 worth of
    independent solenoids in a HAMMIR-class tandem layout, so coil cost lands
    within ~20% of tokamak rather than dramatically below it. The mirror's
    overall cost advantage comes from BOP/blanket simplicity, not C220103."""
    tok = _make_cas22_coil(concept=ConfinementConcept.TOKAMAK)
    mir = _make_cas22_coil(concept=ConfinementConcept.MIRROR)
    assert 0.8 < mir["C220103"] / tok["C220103"] < 1.2


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
        family=ConfinementFamily.STEADY_STATE,
        concept=ConfinementConcept.TOKAMAK,
        b_center=12.0,
        r_bore=1.85,
        coil_material=CoilMaterial.REBCO_HTS,
        blanket_form=BlanketForm.LIQUID_METAL,
        p_nbi=p_nbi,
        p_icrf=p_icrf,
        p_ecrh=p_ecrh,
        p_lhcd=p_lhcd,
        p_driver=0.0,
        f_dec=0.0,
        p_dee=0.0,
        burn_fraction=0.05,
        vac_op_pressure_pa=1.0,
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
        family=ConfinementFamily.STEADY_STATE,
        concept=ConfinementConcept.TOKAMAK,
        b_center=12.0,
        r_bore=1.85,
        coil_material=CoilMaterial.REBCO_HTS,
        blanket_form=BlanketForm.LIQUID_METAL,
        p_nbi=50.0,
        p_icrf=0.0,
        p_ecrh=0.0,
        p_lhcd=0.0,
        p_driver=0.0,
        f_dec=0.0,
        p_dee=0.0,
        burn_fraction=0.05,
        vac_op_pressure_pa=1.0,
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
        family=ConfinementFamily.STEADY_STATE,
        concept=ConfinementConcept.MIRROR,
        b_center=12.0,
        r_bore=1.85,
        coil_material=CoilMaterial.REBCO_HTS,
        blanket_form=BlanketForm.LIQUID_METAL,
        p_nbi=50.0,
        p_icrf=0.0,
        p_ecrh=0.0,
        p_lhcd=0.0,
        p_driver=0.0,
        f_dec=0.0,
        p_dee=0.0,
        burn_fraction=0.05,
        vac_op_pressure_pa=1.0,
    )
    assert mir["C220110"] < tok["C220110"]


# ---- CAS220109: Direct Energy Converter ----


def _make_cas22_dec(f_dec=0.3, p_dee=300.0):
    """Helper for DEC tests - mirror with DEC."""
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
        fuel=Fuel.DHE3,
        noak=True,
        blanket_vol=geo.firstwall_vol + geo.blanket_vol + geo.reflector_vol,
        shield_vol=geo.ht_shield_vol + geo.lt_shield_vol,
        structure_vol=geo.structure_vol,
        vessel_vol=geo.vessel_vol,
        family=ConfinementFamily.STEADY_STATE,
        concept=ConfinementConcept.MIRROR,
        b_center=12.0,
        r_bore=1.85,
        coil_material=CoilMaterial.REBCO_HTS,
        blanket_form=BlanketForm.LIQUID_METAL,
        p_nbi=50.0,
        p_icrf=0.0,
        p_ecrh=0.0,
        p_lhcd=0.0,
        p_driver=0.0,
        f_dec=f_dec,
        p_dee=p_dee,
        burn_fraction=0.05,
        vac_op_pressure_pa=1.0,
    )


def test_c220109_nonzero_when_dec_active():
    """C220109 should be nonzero when f_dec > 0 and p_dee > 0."""
    result = _make_cas22_dec(f_dec=0.3, p_dee=300.0)
    assert result["C220109"] > 0


def test_c220109_zero_when_no_dec():
    """C220109 should be zero when f_dec = 0."""
    result = _make_cas22_dec(f_dec=0.0, p_dee=0.0)
    assert result["C220109"] == 0.0


def test_c220109_scales_with_p_dee():
    """Higher DEC output should increase C220109."""
    low = _make_cas22_dec(f_dec=0.3, p_dee=200.0)
    high = _make_cas22_dec(f_dec=0.3, p_dee=600.0)
    assert high["C220109"] > low["C220109"]


def test_c220109_scaling_exponent():
    """C220109 should scale as (p_dee / P_DEE_REF) ** 0.7."""
    result = _make_cas22_dec(f_dec=0.3, p_dee=400.0)
    # At p_dee = P_DEE_REF = 400, scaling factor is 1.0
    expected = CC.dec_base * 1.0
    assert abs(result["C220109"] - expected) < 0.01


def test_c220109_included_in_total():
    """C220109 should be included in C220000 total."""
    with_dec = _make_cas22_dec(f_dec=0.3, p_dee=400.0)
    without_dec = _make_cas22_dec(f_dec=0.0, p_dee=0.0)
    assert with_dec["C220000"] > without_dec["C220000"]
    diff = with_dec["C220000"] - without_dec["C220000"]
    # Difference should include C220109 plus its share of installation labor
    assert diff > with_dec["C220109"]


def test_multi_unit_labor_discount_in_normal_path():
    """Normal path: C220000 at n_mod=2 applies the 0.92 labor discount.

    cas22_reactor_plant_equipment returns per-module equipment values and
    folds the discounted labor into C220000.
    Formula: C220000(n=2) = equip_sum*2 + labor*(1 + 0.92) + plant_wide_n2
    """
    _equipment_keys = (
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
        "C220112",
    )
    _plant_wide_keys = (
        "C220200",
        "C220300",
        "C220400",
        "C220500",
        "C220600",
        "C220700",
    )

    one = _make_cas22(fuel=Fuel.DT, n_mod=1)
    two = _make_cas22(fuel=Fuel.DT, n_mod=2)

    # Equipment keys are per-module values - unchanged by n_mod
    assert float(two["C220101"]) == pytest.approx(float(one["C220101"]), rel=1e-9)

    # Reconstruct expected C220000(n=2) from n=1 per-module values + n=2 plant_wide
    equip_sum = sum(float(one[k]) for k in _equipment_keys)
    labor = float(one["C220111"])
    plant_wide_2 = sum(float(two[k]) for k in _plant_wide_keys)
    expected = equip_sum * 2 + labor * (1.0 + 0.92) + plant_wide_2
    assert float(two["C220000"]) == pytest.approx(expected, rel=1e-6)


def test_multi_unit_labor_discount_in_override_path():
    """Override path applies the same labor discount as the normal path.

    A user overriding C220101 should still see total CAS22 reflect the
    labor discount on units 2+, not full-cost labor per module.
    """
    from costingfe import ConfinementConcept, CostModel, Fuel

    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    common = dict(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30, n_mod=2)
    base = model.forward(**common)
    overridden = model.forward(
        **common, cost_overrides={"C220101": float(base.cas22_detail["C220101"])}
    )

    # Override echoes the same value -> C220000 should match
    assert float(overridden.cas22_detail["C220000"]) == pytest.approx(
        float(base.cas22_detail["C220000"]), rel=1e-6
    )


# ---------------------------------------------------------------------------
# C220104 pulsed driver dispatch: per-MJ (laser/accelerator/EM-gun) vs per-MW
# (mechanical). EM-gun formation hardware (sheared-flow Z-pinch, plasma jet) is
# costed per joule of pulse energy, so cost tracks e_driver_mj, not average power.
# ---------------------------------------------------------------------------
def _make_cas22_pulsed(concept, e_driver_mj=0.0, p_driver=0.0):
    """Helper to compute CAS22 for a pulsed concept (driver dispatch tests)."""
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
        family=ConfinementFamily.PULSED,
        concept=concept,
        b_center=12.0,
        r_bore=1.85,
        coil_material=CoilMaterial.REBCO_HTS,
        blanket_form=BlanketForm.LIQUID_METAL,
        p_nbi=0.0,
        p_icrf=0.0,
        p_ecrh=0.0,
        p_lhcd=0.0,
        p_driver=p_driver,
        e_driver_mj=e_driver_mj,
        f_dec=0.0,
        p_dee=0.0,
        burn_fraction=0.05,
        vac_op_pressure_pa=1.0,
    )


def test_cas22_staged_zpinch_shear_flow_drive_per_mj():
    # Sheared-flow Z-pinch: coaxial gun + gas injection costed per MJ of pulse
    # energy. p_driver deliberately differs from e_driver_mj to prove the cost
    # tracks pulse energy (gun size/current), not rep-rate-scaled average power.
    r = _make_cas22_pulsed(
        ConfinementConcept.STAGED_ZPINCH, e_driver_mj=100.0, p_driver=50.0
    )
    assert r["C220104"] == pytest.approx(1.5 * 100.0)  # driver_staged_zpinch_per_mj


def test_cas22_bare_zpinch_has_no_c220104_driver():
    # Bare Z-pinch is purely electrical (driver in C220107), no C220104 hardware.
    r = _make_cas22_pulsed(ConfinementConcept.ZPINCH, e_driver_mj=100.0, p_driver=50.0)
    assert r["C220104"] == 0.0


def test_cas22_plasma_jet_driver_rep_rate_independent():
    # PLASMA_JET gun costed per MJ of pulse energy: tracks e_driver_mj, not p_driver.
    r = _make_cas22_pulsed(
        ConfinementConcept.PLASMA_JET, e_driver_mj=100.0, p_driver=50.0
    )
    assert r["C220104"] == pytest.approx(4.0 * 100.0)  # driver_plasma_jet_per_mj
