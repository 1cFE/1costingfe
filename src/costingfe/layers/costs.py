"""Layer 4: Costs — CAS10-CAS90 per-account costing.

All functions are pure (no side effects). Each takes a CostingConstants
object as first argument — no inline magic numbers.
Costs returned in millions USD (M$).

Source: pyFECONs costing/calculations/cas*.py
"""

from costingfe._backend import xp as jnp
from costingfe.layers.economics import (
    compute_crf,
    levelized_annual_cost,
    levelized_replacement_cost,
)
from costingfe.layers.physics import (
    DD_F_HE3_DEFAULT,
    DD_F_T_DEFAULT,
    M_B11_KG,
    M_DEUTERIUM_KG,
    M_HE3_KG,
    M_LI6_KG,
    M_PROTON_KG,
    MEV_TO_JOULES,
    Q_DD_NHE3,
    Q_DD_PT,
    Q_DHE3,
    Q_DT,
    Q_PB11,
)
from costingfe.types import (
    BlanketFill,
    ConfinementConcept,
    Fuel,
    LaserDriverType,
    PulsedConversion,
)


def _total_project_time(cc, construction_time, fuel, noak):
    if noak:
        return construction_time
    return construction_time + cc.licensing_time(fuel)


# ---------------------------------------------------------------------------
# CAS Accounts
# ---------------------------------------------------------------------------


def cas10_preconstruction(cc, p_net, n_mod, fuel, noak):
    """CAS10: Pre-construction costs. Returns M$.

    Land scales with the square root of plant-total net electric power
    (p_net * n_mod): a compact, exclusion-zone-free site whose footprint
    grows sublinearly, like a gas combined-cycle station sharing one site
    across blocks. land_intensity (acres/MWe) is anchored at
    ref_net_power_mwe, so a single module at the reference power keeps the
    linear-intensity result; the sqrt scaling applies away from it.
    """
    land = (
        cc.land_intensity
        * jnp.sqrt(p_net * n_mod * cc.ref_net_power_mwe)
        * cc.land_cost
        / 1e6
    )
    licensing = cc.licensing_cost(fuel)
    studies = cc.plant_studies_noak if noak else cc.plant_studies_foak
    subtotal = (
        land
        + cc.site_permits
        + licensing
        + cc.plant_permits
        + studies
        + cc.plant_reports
        + cc.other_precon
    )
    contingency = cc.contingency_rate(noak) * subtotal
    return subtotal + contingency


def cas21_buildings(cc, p_et, p_the, p_th, p_fus, n_mod, fuel, coil_material):
    """CAS21: Buildings. Returns M$.

    Returned raw, without contingency: CAS21 is part of the CAS20 series, so
    its contingency is applied once by CAS29 over the CAS21-28 sum, like the
    other CAS2x accounts.

    Each building is priced per fuel type with its own scaling basis.
    Fuel-dependent buildings have dt/dd/dhe3/pb11 costs (M$ at 1 GWe ref).
    Fuel-independent buildings have an 'all' cost.
    Each building specifies what it scales with (fixed, p_fus, p_et, etc).

    Buildings/site serve the whole plant, not one module: the power-scaling
    buildings are fed plant-total power (n_mod x per-module, the same total the
    BOP accounts CAS23-26 use), so they reach plant scale under module
    replication. Fixed buildings ignore power and are charged once per site.

    See docs/account_justification/CAS21_buildings.md
    """
    # Reference power levels at 1 GWe calibration point
    P_ET_REF = cc.ref_gross_power_mwe  # MW gross electric (~1 GWe net)
    P_THE_REF = cc.ref_gross_power_mwe  # MW thermal-electric (= p_et, no DEC)
    P_TH_REF = 2500.0  # MW thermal
    P_FUS_REF = 2300.0  # MW fusion

    # Plant-total power: the site houses every module's installed equipment.
    p_et_tot = p_et * n_mod
    p_the_tot = p_the * n_mod
    p_th_tot = p_th * n_mod
    p_fus_tot = p_fus * n_mod

    fuel_key = {
        Fuel.DT: "dt",
        Fuel.DD: "dd",
        Fuel.DHE3: "dhe3",
        Fuel.PB11: "pb11",
    }.get(fuel, "dt")

    scale_map = {
        "fixed": 1.0,
        "p_et": p_et_tot / P_ET_REF,
        "p_the": p_the_tot / P_THE_REF,
        "p_th": p_th_tot / P_TH_REF,
        "p_fus": p_fus_tot / P_FUS_REF,
        # Administration is staff-driven; staff (and so the building) scales as
        # P^0.5, matching the staffing accounts CAS40/CAS70.
        "staff": jnp.sqrt(p_et_tot / P_ET_REF),
    }

    total = 0.0
    for _name, entry in cc.building_costs.items():
        # The cryogenics building is the magnet cryoplant (fuel-independent flat
        # cost). Only superconducting-magnet concepts need it; normal-conducting
        # (copper) and magnet-free concepts carry no cryoplant.
        if _name == "cryogenics" and not coil_material.is_superconducting:
            continue
        scales = entry.get("scales", "fixed")
        base_cost = entry.get(fuel_key, entry.get("all", 0.0))
        ratio = scale_map.get(scales, 1.0)
        total += base_cost * ratio

    return total


def cas23_turbine(cc, p_the, n_mod):
    """CAS23: Turbine plant equipment. Returns M$.

    Scales with thermal electric power (steam turbine output).
    When eta_th=0, p_the=0 and CAS23=0 automatically.
    See docs/account_justification/CAS23_26_balance_of_plant.md
    """
    return n_mod * p_the * cc.turbine_per_mw


def cas24_electrical(cc, p_et, n_mod):
    """CAS24: Electric plant equipment. Returns M$.

    See docs/account_justification/CAS23_26_balance_of_plant.md
    """
    return n_mod * p_et * cc.electric_per_mw


def cas25_misc(cc, p_et, n_mod):
    """CAS25: Miscellaneous plant equipment. Returns M$.

    See docs/account_justification/CAS23_26_balance_of_plant.md
    """
    return n_mod * p_et * cc.misc_per_mw


def cas26_heat_rejection(cc, p_th, n_mod):
    """CAS26: Heat rejection. Returns M$.

    Scales with total thermal power (heat to be rejected).
    See docs/account_justification/CAS23_26_balance_of_plant.md
    """
    return n_mod * p_th * cc.heat_rej_per_mw


def cas27_special_materials(cc, blanket_fill: BlanketFill, blanket_vol, n_mod):
    """CAS27: Special materials, initial blanket-fill inventory. Returns M$.

    The blanket fill (PbLi, Li, FLiBe, Be/ceramic pebbles) is a material
    inventory: its cost is set by blanket *volume*, not net electric power.
    CAS220101 covers the blanket *structure*; CAS27 covers the *material fill*.

    Volume-based mass build-up, keyed on blanket_fill, per module then scaled
    by the module count (blanket_vol is per-module, like the CAS22 components):

        CAS27 = n_mod x blanket_vol x vol_frac x density x price / 1e6

    where vol_frac is the fraction of the blanket region occupied by that
    material (liquid fill fraction for liquids; breeder/multiplier-zone x
    pebble-packing for solids). Per-fill density/vol_frac/price live in
    cc.cas27_fill_materials. Aneutronic concepts use blanket_fill = none -> 0.

    See docs/account_justification/CAS27_special_materials.md
    """
    m = cc.cas27_fill_materials[blanket_fill.value]
    return n_mod * blanket_vol * m["vol_frac"] * m["density"] * m["price"] / 1e6


def cas28_digital_twin(cc):
    """CAS28: Digital twin. Returns M$.

    Fixed cost, plant-size independent.
    See docs/account_justification/CAS28_digital_twin.md
    """
    return cc.digital_twin


def cas29_contingency(cc, cas2x_total, noak):
    """CAS29: Contingency on direct costs. Returns M$.

    10% FOAK / 0% NOAK per Gen-IV EMWG convention.
    See docs/account_justification/CAS29_contingency.md
    """
    return cc.contingency_rate(noak) * cas2x_total


def cas30_indirect(cc, cas20, construction_time):
    """CAS30: Indirect service costs. Returns M$.

    Computed as a fraction of total direct cost (CAS20), scaled by
    construction time relative to a reference duration.

    See docs/account_justification/CAS30_indirect_service_costs.md
    for derivation and source analysis.
    """
    return (
        cc.indirect_fraction
        * cas20
        * (construction_time / cc.reference_construction_time)
    )


def cas40_owner(cc, fuel, p_net, n_mod):
    """CAS40: Capitalized owner's costs. Returns M$.

    Pre-operational costs to recruit, train, house, and compensate
    the plant operations staff before COD.  Derived from the CAS71-73
    staffing analysis applied through the INL CAS40 methodology
    (1.5 yr pre-op, 10 % overhire, 25 % recruiting, 58 % benefits).

    Uses the SAME staffing basis as CAS70 annual O&M — CAS40 covers
    pre-COD costs, CAS70 covers post-COD costs.  No double-counting.

    Power-law exponent 0.5 reflects staffing economy of scale
    (INL SFR data: 165 MWe to 3108 MWe, alpha ~ 0.5), applied to
    plant-total net electric (p_net * n_mod) like CAS70.

    See docs/account_justification/CAS40_capitalized_owners_costs.md
    """
    return cc.owner_cost(fuel) * (p_net * n_mod / cc.ref_net_power_mwe) ** 0.5


def cas50_supplementary(cc, fuel, cas20, cas22_to_28, cas30, p_net, n_mod, noak):
    """CAS50: Capitalized supplementary costs. Returns M$.

    Sub-account model with fuel-dependent spare parts, startup
    inventory, and decommissioning provisions.  Shipping, taxes,
    and insurance scale with plant cost (fuel-independent).

    Startup fuel (c55) and the decommissioning provision (c56) are linear in
    plant-total net electric (p_net * n_mod): the startup inventory and the
    end-of-life obligation both accrue once per module.

    See docs/account_justification/CAS50_supplementary_costs.md
    """
    p_net_total = p_net * n_mod
    c51_shipping = cc.shipping_frac * cas20
    c52_spares = cc.spare_parts_frac(fuel) * cas22_to_28
    c53_taxes = cc.tax_frac * cas20
    c54_insurance = cc.construction_insurance_frac * (cas20 + cas30)
    c55_fuel = cc.startup_fuel(fuel) * (p_net_total / cc.ref_net_power_mwe)
    c56_decom = cc.decom_provision(fuel) * (p_net_total / cc.ref_net_power_mwe)
    subtotal = (
        c51_shipping + c52_spares + c53_taxes + c54_insurance + c55_fuel + c56_decom
    )
    c59_contingency = cc.contingency_rate(noak) * subtotal
    return subtotal + c59_contingency


def cas60_idc(interest_rate, overnight_cost, construction_time):
    """CAS60: Interest during construction. Returns M$.

    Assumes uniform capital spending over the construction period.
    f_IDC = ((1+i)^T - 1) / (i*T) - 1

    See docs/account_justification/CAS60_interest_during_construction.md
    """
    i = interest_rate
    T = construction_time
    f_idc = ((1 + i) ** T - 1) / (i * T) - 1
    return f_idc * overnight_cost


# Replaceable laser-driver subsystems per architecture: (replace_frac attr,
# shot_lifetime attr) on CostingConstants. Consumed by the CAS72 dispatch in
# cas70_om. Module-level (pure constant) to avoid rebuilding it per call.
_LASER_SUBSYSTEMS = {
    LaserDriverType.DPSSL: (
        ("dpssl_diode_replace_frac", "dpssl_diode_shot_lifetime"),
        ("dpssl_crystal_replace_frac", "dpssl_crystal_shot_lifetime"),
        ("dpssl_optics_replace_frac", "dpssl_optics_shot_lifetime"),
    ),
    LaserDriverType.KRF: (
        ("krf_foil_replace_frac", "krf_foil_shot_lifetime"),
        ("krf_ebeam_replace_frac", "krf_ebeam_shot_lifetime"),
    ),
    LaserDriverType.NDGLASS: (
        ("ndglass_lamp_replace_frac", "ndglass_lamp_shot_lifetime"),
    ),
}


def cas70_om(
    cc,
    cas22_detail,
    replaceable_accounts,
    n_mod,
    p_net,
    availability,
    inflation_rate,
    interest_rate,
    lifetime_yr,
    core_lifetime,
    construction_time,
    fuel,
    noak,
    p_dee=0.0,
    pulsed_conversion=None,
    f_rep=0.0,
    concept=None,
    laser_driver_type=None,
):
    """CAS70: Annualized O&M + scheduled replacement. Returns (total, cas71, cas72).

    CAS71: Annual O&M (today's $ inflated to operation start).
    CAS72: Annualized scheduled replacement (PV-discounted at interest rate,
           annualized via CRF). core_lifetime is in FPY, converted to calendar
           years via availability.
    """
    # CAS71: Annual O&M — fuel-dependent staffing-based base cost, scaled by
    # plant size. Power-law exponent 0.5: staffing economy of scale (INL SFR
    # data, plant-total basis). Uses plant-total net electric (p_net * n_mod):
    # one operated plant, so staffing scales with total output, not per module.
    # No concept-dependent factor: no concept has a sourced O&M-ergonomics
    # basis, so all concepts share the fuel-driven staffing baseline.
    # Source: docs/account_justification/CAS70_staffing_and_om_costs.md
    annual_om = cc.om_cost(fuel) * (p_net * n_mod / cc.ref_net_power_mwe) ** 0.5  # M$
    t_project = _total_project_time(cc, construction_time, fuel, noak)
    cas71 = levelized_annual_cost(
        annual_om, interest_rate, inflation_rate, lifetime_yr, t_project
    )

    # CAS72: Annualized scheduled replacement. Each term is the level annual
    # cost of replacing an item every t_replace years over the plant life,
    # computed by the shared geometric closed-form helper
    # (levelized_replacement_cost). The first set is capital, so only
    # replacements beyond it are charged.
    core_lifetime_cal = core_lifetime / availability  # FPY → calendar years
    cost_per_event = sum(cas22_detail[k] for k in replaceable_accounts) * n_mod
    cas72 = levelized_replacement_cost(
        cost_per_event, core_lifetime_cal, interest_rate, lifetime_yr
    )

    # DEC grid replacement (additive, independent cycle).
    # jnp.maximum(p_dee, 1e-6) keeps the power-law gradient finite at p_dee=0;
    # the outer jnp.where masks the result to zero when p_dee == 0.
    P_DEE_REF = 400.0
    p_dee_safe = jnp.maximum(p_dee, 1e-6)
    dec_grid = cc.dec_grid_cost * jnp.where(
        p_dee > 0, (p_dee_safe / P_DEE_REF) ** 0.7, 0.0
    )
    dec_grid_life_cal = cc.dec_grid_lifetime(fuel) / availability
    cas72 = cas72 + levelized_replacement_cost(
        dec_grid * n_mod, dec_grid_life_cal, interest_rate, lifetime_yr
    )

    # Cap bank scheduled replacement (INDUCTIVE_DEC only).
    if pulsed_conversion == PulsedConversion.INDUCTIVE_DEC and f_rep > 0:
        n_shots_per_year = f_rep * 8760.0 * 3600.0 * availability
        t_replace_cap = cc.cap_shot_lifetime / n_shots_per_year
        cap_cost = cas22_detail.get("C220107", 0.0) * n_mod
        cas72 = cas72 + levelized_replacement_cost(
            cap_cost, t_replace_cap, interest_rate, lifetime_yr
        )

    # Formation-electrode scheduled replacement (EM-gun concepts).
    # Plasma-facing coaxial-gun electrodes (sheared-flow Z-pinch, plasma jet) erode
    # under high current density and are periodically replaced. The replacement
    # interval can be sub-annual. This stays on levelized_annual_cost (the
    # inflation-escalating convention) rather than the geometric replacement
    # helper, to preserve its existing calibration.
    if (
        concept in (ConfinementConcept.STAGED_ZPINCH, ConfinementConcept.PLASMA_JET)
        and f_rep > 0
        and cc.electrode_shot_lifetime > 0
    ):
        n_shots_per_year = f_rep * 8760.0 * 3600.0 * availability
        annual_electrode = (
            cc.electrode_replace_frac
            * cas22_detail.get("C220104", 0.0)
            * n_mod
            * n_shots_per_year
            / cc.electrode_shot_lifetime
        )
        cas72 = cas72 + levelized_annual_cost(
            annual_electrode, interest_rate, inflation_rate, lifetime_yr, t_project
        )

    # Laser-IFE driver scheduled replacement (DPSSL / KrF / Nd:Glass).
    # Each architecture has its own replaceable subsystems with distinct shot
    # lifetimes; each is summed via the shared geometric helper. Diodes whose
    # NOAK life approaches/exceeds the plant contribute ~0 (at most one
    # heavily-discounted replacement); flashlamps wear sub-annually and
    # dominate. See CAS22_reactor_components.md (CAS72 O&M).
    if (
        concept == ConfinementConcept.LASER_IFE
        and f_rep > 0
        and laser_driver_type is not None
    ):
        n_shots_per_year = f_rep * 8760.0 * 3600.0 * availability
        c220104 = cas22_detail.get("C220104", 0.0) * n_mod
        for frac_attr, life_attr in _LASER_SUBSYSTEMS[laser_driver_type]:
            event_cost = getattr(cc, frac_attr) * c220104
            t_replace = getattr(cc, life_attr) / n_shots_per_year
            cas72 = cas72 + levelized_replacement_cost(
                event_cost, t_replace, interest_rate, lifetime_yr
            )

    return cas71 + cas72, cas71, cas72


def cas80_fuel(
    cc,
    p_fus,
    n_mod,
    availability,
    inflation_rate,
    interest_rate,
    lifetime_yr,
    construction_time,
    fuel,
    noak,
    dd_f_T=DD_F_T_DEFAULT,
    dd_f_He3=DD_F_HE3_DEFAULT,
    dhe3_dd_frac=0.131,
    dhe3_f_T=0.5,
    dhe3_f_He3=0.1,
    *,
    burn_fraction,
    fuel_recovery,
    target_unit_cost,
    n_targets_per_year,
):
    """CAS80: Annualized fuel cost. Fuel-specific consumable costs.

    Each fuel cycle has different consumables, Q-values, and costs per reaction.
    The 1e6 (MW->W) and /1e6 ($->M$) cancel, giving a clean formula.
    Returns M$.

    For inertial/magneto-inertial concepts the fuel-bearing consumable is not
    just the isotope but the fabricated target destroyed each shot (capsule,
    hohlraum, liner, recyclable transmission line). Following the fission
    fuel-assembly convention (CAS80 holds the fabricated assembly, not only the
    raw isotope), target_unit_cost [$ /shot] x n_targets_per_year [shots/yr per
    module] adds this hardware to CAS80. Both are 0 for concepts with no
    manufactured target (MFE, in-situ plasma/liner formation).
    """
    SECONDS_PER_YR = 3600.0 * 8760.0

    if fuel == Fuel.DT:
        cost_per_rxn = M_DEUTERIUM_KG * cc.u_deuterium + M_LI6_KG * cc.u_li6
        q_eff = Q_DT
    elif fuel == Fuel.DD:
        q_eff = (
            0.5 * Q_DD_PT
            + 0.5 * Q_DD_NHE3
            + 0.5 * dd_f_T * Q_DT
            + 0.5 * dd_f_He3 * Q_DHE3
        )
        d_per_event = 2 + 0.5 * dd_f_T + 0.5 * dd_f_He3
        cost_per_rxn = d_per_event * M_DEUTERIUM_KG * cc.u_deuterium
    elif fuel == Fuel.DHE3:
        # Per fusion event in a D-He-3 plasma:
        #   (1 - dhe3_dd_frac) are D-He-3 events:  1 D + 1 He-3 -> Q_DHE3
        #   dhe3_dd_frac are D-D events:           2 D, 50/50 D(d,p)T and D(d,n)He-3
        #     T burnup (dhe3_f_T) consumes another D in D-T -> Q_DT
        #     He-3 burnup (dhe3_f_He3) consumes another D in D-He-3 -> Q_DHE3.
        # Bred He-3 enters the same recovery loop as primary He-3, so its
        # cumulative fusion probability equals that of primary He-3; the
        # external-He-3 credit cancels exactly in the steady-state balance
        # and only the primary need remains in he3_per_event. The cost-side
        # fuel_recovery multiplier below handles all He-3 recovery losses.
        f_dhe3 = 1.0 - dhe3_dd_frac
        q_dd_avg = 0.5 * Q_DD_PT + 0.5 * Q_DD_NHE3
        q_eff = f_dhe3 * Q_DHE3 + dhe3_dd_frac * (
            q_dd_avg + 0.5 * dhe3_f_T * Q_DT + 0.5 * dhe3_f_He3 * Q_DHE3
        )
        d_per_event = (
            f_dhe3 + 2.0 * dhe3_dd_frac + dhe3_dd_frac * 0.5 * (dhe3_f_T + dhe3_f_He3)
        )
        he3_per_event = f_dhe3
        cost_per_rxn = (
            d_per_event * M_DEUTERIUM_KG * cc.u_deuterium
            + he3_per_event * M_HE3_KG * cc.u_he3
        )
    elif fuel == Fuel.PB11:
        b11_price = cc.u_b11_noak if noak else cc.u_b11
        cost_per_rxn = M_PROTON_KG * cc.u_protium + M_B11_KG * b11_price
        q_eff = Q_PB11
    else:
        cost_per_rxn = 0.0
        q_eff = Q_DT

    annual_musd = (
        n_mod
        * p_fus
        * SECONDS_PER_YR
        * availability
        * cost_per_rxn
        / (q_eff * MEV_TO_JOULES)
    )

    # Burn-fraction correction: unburned fuel not recovered must be repurchased.
    # multiplier = 1 + (1 - burn_fraction) / burn_fraction * (1 - fuel_recovery)
    fuel_loss = (1.0 - burn_fraction) / burn_fraction * (1.0 - fuel_recovery)
    annual_musd = annual_musd * (1.0 + fuel_loss)

    # Target consumable (IFE/MIF): fabricated target/liner hardware destroyed
    # each shot. $/shot x shots/yr/module x modules -> $/yr, /1e6 -> M$/yr. The
    # burn-fraction correction above does not apply: the target is consumed
    # whole every shot regardless of fuel burnup.
    annual_target_musd = n_mod * n_targets_per_year * target_unit_cost / 1e6
    annual_musd = annual_musd + annual_target_musd

    t_project = _total_project_time(cc, construction_time, fuel, noak)
    return levelized_annual_cost(
        annual_musd, interest_rate, inflation_rate, lifetime_yr, t_project
    )


def cas90_financial(total_capital, interest_rate, plant_lifetime):
    """CAS90: Annualized financial (capital) costs. Returns M$.

    Plain CRF * total_capital. Construction-period financing is handled
    by CAS60 (IDC), so no effective CRF adjustment here.

    See docs/account_justification/CAS90_annualized_financial_costs.md
    """
    crf = compute_crf(interest_rate, plant_lifetime)
    return crf * total_capital
