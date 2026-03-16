"""Layer 4: Costs — CAS10-CAS90 per-account costing.

All functions are pure (no side effects). Each takes a CostingConstants
object as first argument — no inline magic numbers.
Costs returned in millions USD (M$).

Source: pyFECONs costing/calculations/cas*.py
"""

import math

import jax.numpy as jnp

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
from costingfe.types import Fuel


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


def cas21_buildings(cc, p_et, fuel, noak):
    """CAS21: Buildings. Scales with gross electric. Returns M$.

    19 building types, each with per-kW cost ($/kW gross electric).
    Fuel-dependent: 4 tritium-related buildings get 0.5x for non-DT.
    Source: ARIES (Waganer 2013) / NETL Case B12A calibration.
    See docs/account_justification/CAS21_buildings.md
    """
    fuel_scale = 1.0 if fuel == Fuel.DT else 0.5
    tritium_buildings = (
        "site_improvements",
        "fusion_heat_island",
        "hot_cell",
        "fuel_storage",
    )
    total = 0.0
    for name, cost_per_kw in cc.building_costs_per_kw.items():
        scale = fuel_scale if name in tritium_buildings else 1.0
        total += cost_per_kw * p_et / 1000.0 * scale
    contingency = cc.contingency_rate(noak) * total
    return total + contingency


def cas23_turbine(cc, p_et, n_mod):
    """CAS23: Turbine plant equipment. Returns M$.

    Linear scaling with gross electric power. Coefficient from
    ARIES/NETL calibration, inflation-adjusted to 2024$.
    See docs/account_justification/CAS23_26_balance_of_plant.md
    """
    return n_mod * p_et * cc.turbine_per_mw


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


def cas26_heat_rejection(cc, p_et, n_mod):
    """CAS26: Heat rejection. Returns M$.

    See docs/account_justification/CAS23_26_balance_of_plant.md
    """
    return n_mod * p_et * cc.heat_rej_per_mw


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
