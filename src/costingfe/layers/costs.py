"""Layer 4: Costs — CAS10-CAS90 per-account costing.

All functions are pure (no side effects). Each takes a CostingConstants
object as first argument — no inline magic numbers.
Costs returned in millions USD (M$).

Source: pyFECONs costing/calculations/cas*.py
"""

import math

import numpy as jnp

from costingfe.layers.economics import (
    compute_crf,
    levelized_annual_cost,
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
from costingfe.types import Fuel, PulsedConversion


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


def cas21_buildings(cc, p_et, p_the, p_th, p_fus, fuel, noak):
    """CAS21: Buildings. Returns M$.

    Each building is priced per fuel type with its own scaling basis.
    Fuel-dependent buildings have dt/dd/dhe3/pb11 costs (M$ at 1 GWe ref).
    Fuel-independent buildings have an 'all' cost.
    Each building specifies what it scales with (fixed, p_fus, p_et, etc).

    See docs/account_justification/CAS21_buildings.md
    """
    # Reference power levels at 1 GWe calibration point
    P_ET_REF = 1150.0  # MW gross electric (~1 GWe net)
    P_THE_REF = 1150.0  # MW thermal electric (steam-only, = p_et when no DEC)
    P_TH_REF = 2500.0  # MW thermal
    P_FUS_REF = 2300.0  # MW fusion

    fuel_key = {
        Fuel.DT: "dt",
        Fuel.DD: "dd",
        Fuel.DHE3: "dhe3",
        Fuel.PB11: "pb11",
    }.get(fuel, "dt")

    scale_map = {
        "fixed": 1.0,
        "p_et": p_et / P_ET_REF,
        "p_the": p_the / P_THE_REF,
        "p_th": p_th / P_TH_REF,
        "p_fus": p_fus / P_FUS_REF,
        "floor_area": p_et / P_ET_REF,  # proxy
        "staff": p_et / P_ET_REF,  # proxy (staff scales with P^0.5 but
    }  # building area is a weaker function)

    total = 0.0
    for _name, entry in cc.building_costs.items():
        scales = entry.get("scales", "fixed")
        base_cost = entry.get(fuel_key, entry.get("all", 0.0))
        ratio = scale_map.get(scales, 1.0)
        total += base_cost * ratio

    contingency = cc.contingency_rate(noak) * total
    return total + contingency


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


def cas27_special_materials(cc, p_net, fuel):
    """CAS27: Special materials — initial reactor material inventory. Returns M$.

    Covers non-fuel reactor materials: breeding blanket fill (PbLi, Li, FLiBe),
    neutron multiplier (Be if HCPB concept), and other special inventory.
    CAS220101 covers the blanket *structure*; CAS27 covers the *material fill*.

    Default assumes PbLi blanket concept for DT. For HCPB concepts with
    beryllium pebbles (~300 tonnes × $600/kg = $180M), override via
    cost_overrides["CAS27"].

    See docs/account_justification/CAS27_special_materials.md
    """
    base = {
        Fuel.DT: cc.special_materials_dt,
        Fuel.DD: cc.special_materials_dd,
        Fuel.DHE3: cc.special_materials_dhe3,
        Fuel.PB11: cc.special_materials_pb11,
    }
    return base[fuel] * (p_net / 1000.0)


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


def cas40_owner(cc, fuel, p_net):
    """CAS40: Capitalized owner's costs. Returns M$.

    Pre-operational costs to recruit, train, house, and compensate
    the plant operations staff before COD.  Derived from the CAS71-73
    staffing analysis applied through the INL CAS40 methodology
    (1.5 yr pre-op, 10 % overhire, 25 % recruiting, 58 % benefits).

    Uses the SAME staffing basis as CAS70 annual O&M — CAS40 covers
    pre-COD costs, CAS70 covers post-COD costs.  No double-counting.

    Power-law exponent 0.5 reflects staffing economy of scale
    (INL SFR data: 165 MWe to 3108 MWe, alpha ~ 0.5).

    See docs/account_justification/CAS40_capitalized_owners_costs.md
    """
    return cc.owner_cost(fuel) * (p_net / 1000.0) ** 0.5


def cas50_supplementary(cc, fuel, cas20, cas22_to_28, cas30, p_net, noak):
    """CAS50: Capitalized supplementary costs. Returns M$.

    Sub-account model with fuel-dependent spare parts, startup
    inventory, and decommissioning provisions.  Shipping, taxes,
    and insurance scale with plant cost (fuel-independent).

    See docs/account_justification/CAS50_supplementary_costs.md
    """
    c51_shipping = cc.shipping_frac * cas20
    c52_spares = cc.spare_parts_frac(fuel) * cas22_to_28
    c53_taxes = cc.tax_frac * cas20
    c54_insurance = cc.construction_insurance_frac * (cas20 + cas30)
    c55_fuel = cc.startup_fuel(fuel) * (p_net / 1000.0)
    c56_decom = cc.decom_provision(fuel) * (p_net / 1000.0)
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
):
    """CAS70: Annualized O&M + scheduled replacement. Returns (total, cas71, cas72).

    CAS71: Annual O&M (today's $ inflated to operation start).
    CAS72: Annualized scheduled replacement (PV-discounted at interest rate,
           annualized via CRF). core_lifetime is in FPY, converted to calendar
           years via availability.
    """
    # CAS71: Annual O&M — fuel-dependent staffing-based coefficient
    # Power-law exponent 0.5: staffing economy of scale (INL SFR data).
    # Source: docs/account_justification/CAS70_staffing_and_om_costs.md
    annual_om = cc.om_cost(fuel) * (p_net / 1000.0) ** 0.5  # M$
    t_project = _total_project_time(cc, construction_time, fuel, noak)
    cas71 = levelized_annual_cost(
        annual_om, interest_rate, inflation_rate, lifetime_yr, t_project
    )

    # CAS72: Annualized scheduled replacement
    core_lifetime_cal = core_lifetime / availability  # FPY → calendar years
    n_rep = jnp.maximum(0.0, jnp.ceil(lifetime_yr / core_lifetime_cal) - 1.0)
    cost_per_event = sum(cas22_detail[k] for k in replaceable_accounts) * n_mod
    # Sum PV of each replacement event, using jnp.where for JAX traceability.
    # Max replacements bounded by lifetime/core_lifetime; 20 covers all cases
    # (e.g., 60yr plant / 3yr core = 19 replacements).
    MAX_REP = 20
    pv = 0.0
    for k in range(1, MAX_REP + 1):
        discount = (1 + interest_rate) ** (k * core_lifetime_cal)
        pv = pv + jnp.where(k <= n_rep, cost_per_event / discount, 0.0)
    crf = compute_crf(interest_rate, lifetime_yr)
    cas72 = pv * crf

    # DEC grid replacement (additive, independent cycle)
    # Use jnp.maximum(p_dee, 1e-6) as a safe base for the power law to avoid NaN
    # gradients at p_dee=0 (since d/dp_dee of p_dee^0.7 → ∞ there). The outer
    # jnp.where masks the result to zero when p_dee == 0.
    P_DEE_REF = 400.0
    p_dee_safe = jnp.maximum(p_dee, 1e-6)
    dec_grid = cc.dec_grid_cost * jnp.where(
        p_dee > 0, (p_dee_safe / P_DEE_REF) ** 0.7, 0.0
    )
    dec_grid_life_cal = cc.dec_grid_lifetime(fuel) / availability
    n_rep_dec = jnp.maximum(0.0, jnp.ceil(lifetime_yr / dec_grid_life_cal) - 1.0)
    dec_cost = dec_grid * n_mod
    pv_dec = 0.0
    for k in range(1, MAX_REP + 1):
        discount = (1 + interest_rate) ** (k * dec_grid_life_cal)
        pv_dec = pv_dec + jnp.where(k <= n_rep_dec, dec_cost / discount, 0.0)
    cas72 = cas72 + pv_dec * crf

    # Cap bank scheduled replacement (INDUCTIVE_DEC only)
    if pulsed_conversion == PulsedConversion.INDUCTIVE_DEC and f_rep > 0:
        n_shots_per_year = f_rep * 8760.0 * 3600.0 * availability
        t_replace_cap = cc.cap_shot_lifetime / n_shots_per_year
        n_rep_cap = jnp.maximum(0.0, jnp.ceil(lifetime_yr / t_replace_cap) - 1.0)
        cap_cost = cas22_detail.get("C220107", 0.0) * n_mod
        pv_cap = 0.0
        for k in range(1, MAX_REP + 1):
            discount = (1 + interest_rate) ** (k * t_replace_cap)
            pv_cap = pv_cap + jnp.where(k <= n_rep_cap, cap_cost / discount, 0.0)
        cas72 = cas72 + pv_cap * crf

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
    burn_fraction=None,
    fuel_recovery=None,
):
    """CAS80: Annualized fuel cost. Fuel-specific consumable costs.

    Each fuel cycle has different consumables, Q-values, and costs per reaction.
    The 1e6 (MW->W) and /1e6 ($->M$) cancel, giving a clean formula.
    Returns M$.
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
        cost_per_rxn = M_DEUTERIUM_KG * cc.u_deuterium + M_HE3_KG * cc.u_he3
        q_eff = Q_DHE3
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
    bf = burn_fraction if burn_fraction is not None else cc.burn_fraction
    fr = fuel_recovery if fuel_recovery is not None else cc.fuel_recovery
    fuel_loss = (1.0 - bf) / bf * (1.0 - fr)
    annual_musd = annual_musd * (1.0 + fuel_loss)

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
