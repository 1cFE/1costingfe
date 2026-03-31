"""Layer 2: Physics — fuel physics, power balance (forward + inverse)."""

import numpy as jnp

from costingfe.layers.radiation import (
    compute_impurity_fraction,
    compute_p_rad,
)
from costingfe.types import Fuel, ImpurityMix, PowerTable, WallMaterial

# ---------------------------------------------------------------------------
# Fundamental constants (CODATA 2018 values, formerly from scipy.constants)
# ---------------------------------------------------------------------------
_EV = 1.602176634e-19  # J per eV
_ATOMIC_MASS = 1.66053906892e-27  # kg
MEV_TO_JOULES = _EV * 1e6  # 1 MeV in J
M_DEUTERIUM_KG = 3.3435837768e-27  # deuteron mass, kg
M_PROTON_KG = 1.67262192595e-27  # proton mass, kg
M_HE3_KG = 5.0064127862e-27  # helion mass, kg (He-3 nucleus)
M_LI6_KG = 6.015122795 * _ATOMIC_MASS  # kg
M_B11_KG = 11.0093054 * _ATOMIC_MASS  # kg

# DD burn fraction defaults (T=50 keV, tau_p=5 s)
DD_F_T_DEFAULT = 0.969  # DD tritium burn fraction
DD_F_HE3_DEFAULT = 0.689  # DD He-3 burn fraction

# ---------------------------------------------------------------------------
# Fusion Q-values and product energies (MeV) — nuclear reaction data
# Source: pyFECONs fuel_physics.py
# ---------------------------------------------------------------------------
# DT: D + T -> He4(3.52 MeV) + n(14.06 MeV), Q = 17.58 MeV
E_ALPHA_DT = 3.52
Q_DT = 17.58
E_N_DT = 14.06

# DD branch 1: D + D -> T(1.01 MeV) + p(3.02 MeV), Q = 4.03 MeV
E_T_DD = 1.01
E_P_DD = 3.02
Q_DD_PT = 4.03

# DD branch 2: D + D -> He3(0.82 MeV) + n(2.45 MeV), Q = 3.27 MeV
E_HE3_DD = 0.82
E_N_DD = 2.45
Q_DD_NHE3 = 3.27

# DHe3: D + He3 -> He4(3.6 MeV) + p(14.7 MeV), Q = 18.35 MeV
Q_DHE3 = 18.35

# PB11: p + B11 -> 3 He4, Q = 8.68 MeV
Q_PB11 = 8.68

# PB11 side reactions (neutron-producing)
# 11B(alpha,n)14N: Q = +0.158 MeV, dominant (~1e-3 per alpha)
Q_ALPHA_N_PB11 = 0.158
# 11B(p,n)11C: Q = -2.765 MeV, threshold 3.02 MeV (~1e-5 per proton)
Q_P_N_PB11 = -2.765

# DD primary per-event averages (50/50 branches)
_E_CHARGED_PRIMARY_DD = 0.5 * (E_T_DD + E_P_DD) + 0.5 * E_HE3_DD  # ~2.425
_E_NEUTRON_PRIMARY_DD = 0.5 * E_N_DD  # ~1.225
_E_TOTAL_PRIMARY_DD = 0.5 * Q_DD_PT + 0.5 * Q_DD_NHE3  # ~3.65


def ash_neutron_split(
    p_fus: float,
    fuel: Fuel,
    dd_f_T: float,
    dd_f_He3: float,
    dhe3_dd_frac: float,
    dhe3_f_T: float,
    pb11_f_alpha_n: float,
    pb11_f_p_n: float,
) -> tuple[float, float]:
    """Compute charged-particle (ash) and neutron power from fusion power.

    Returns (p_ash, p_neutron) in MW. All paths are JAX-differentiable.

    Source: pyFECONs fuel_physics.py:compute_ash_neutron_split
    """
    if fuel == Fuel.DT:
        ash_frac = E_ALPHA_DT / Q_DT
    elif fuel == Fuel.DD:
        E_charged = (
            _E_CHARGED_PRIMARY_DD + 0.5 * dd_f_T * E_ALPHA_DT + 0.5 * dd_f_He3 * Q_DHE3
        )
        E_total = _E_TOTAL_PRIMARY_DD + 0.5 * dd_f_T * Q_DT + 0.5 * dd_f_He3 * Q_DHE3
        ash_frac = E_charged / E_total
    elif fuel == Fuel.DHE3:
        E_n_dd = _E_NEUTRON_PRIMARY_DD + 0.5 * dhe3_f_T * E_N_DT
        E_c_dd = _E_CHARGED_PRIMARY_DD + 0.5 * dhe3_f_T * E_ALPHA_DT
        ash_frac = (1 - dhe3_dd_frac) + dhe3_dd_frac * E_c_dd / (E_n_dd + E_c_dd)
    elif fuel == Fuel.PB11:
        # Side reaction 1: 11B(alpha,n)14N
        # Each p+11B produces 3 alphas (~2.89 MeV each). Each alpha has
        # probability pb11_f_alpha_n of undergoing 11B(alpha,n)14N.
        n_alpha_n = 3.0 * pb11_f_alpha_n  # alpha-n reactions per primary event
        E_alpha = Q_PB11 / 3.0  # average alpha energy (~2.89 MeV)
        # CM kinematics: neutron energy in lab frame
        # E_CM = E_alpha * m_B11/(m_alpha + m_B11) = E_alpha * 11/15
        # Products share E_CM + Q among n(1/15) and 14N(14/15)
        E_CM_products = E_alpha * 11.0 / 15.0 + Q_ALPHA_N_PB11
        E_n_alpha = E_CM_products * 14.0 / 15.0 + E_alpha * 4.0 / 225.0  # ~2.18 MeV

        # Side reaction 2: 11B(p,n)11C
        # Endothermic (Q = -2.765 MeV, threshold 3.02 MeV). The primary
        # proton has ~zero kinetic energy in the CM, so this reaction competes
        # with p+11B only at the high-energy tail. pb11_f_p_n is the fraction
        # of primary p+11B events that instead undergo (p,n).
        # CM kinematics: E_CM = E_p * 11/12; products share E_CM + Q
        # At threshold (~3.02 MeV), E_n ~ (E_CM + Q) * 11/12 ~ 0.17 MeV
        E_n_pn = 0.17  # MeV, neutron energy near threshold

        # Total energy and neutron energy per primary event
        E_total = Q_PB11 + n_alpha_n * Q_ALPHA_N_PB11 + pb11_f_p_n * Q_P_N_PB11
        E_neutron = n_alpha_n * E_n_alpha + pb11_f_p_n * E_n_pn
        ash_frac = (E_total - E_neutron) / E_total
    else:
        raise ValueError(f"Unknown fuel type: {fuel}")

    p_ash = p_fus * ash_frac
    p_neutron = p_fus * (1.0 - ash_frac)
    return p_ash, p_neutron


def mfe_forward_power_balance(
    p_fus: float,
    fuel: Fuel,
    p_input: float,
    mn: float,
    eta_th: float,
    eta_p: float,
    eta_pin: float,
    eta_de: float,
    f_sub: float,
    f_dec: float,
    p_coils: float,
    p_cool: float,
    p_pump: float,
    p_trit: float,
    p_house: float,
    p_cryo: float,
    n_e: float,
    T_e: float,
    Z_eff: float,
    plasma_volume: float,
    B: float,
    dd_f_T: float,
    dd_f_He3: float,
    dhe3_dd_frac: float,
    dhe3_f_T: float,
    pb11_f_alpha_n: float,
    pb11_f_p_n: float,
    wall_material: WallMaterial | None,
    T_edge: float,
    tau_ratio: float,
    fw_area: float,
    R_major: float,
    a_minor: float,
    kappa: float,
    R_w: float,
    # Genuine optional overrides (keep defaults)
    p_rad_override: float | None = None,
    seeded_impurities: dict[str, float] | None = None,
) -> PowerTable:
    """MFE forward power balance: fusion power -> net electric.

    Plasma energy balance: P_ash + P_input = P_rad + P_transport.
    If P_rad > P_ash + P_input at the given T_e, heating power is increased
    to sustain the plasma (P_input_eff = P_rad - P_ash).

    Radiation model: bremsstrahlung + synchrotron + impurity line radiation.
    """
    # Step 1: Ash/neutron split
    p_ash, p_neutron = ash_neutron_split(
        p_fus,
        fuel,
        dd_f_T,
        dd_f_He3,
        dhe3_dd_frac,
        dhe3_f_T,
        pb11_f_alpha_n,
        pb11_f_p_n,
    )

    # Step 2: Build impurity mix (if wall_material provided)
    impurities = None
    if wall_material is not None and p_rad_override is None:
        wall_derived = {}
        symbol, f_z = compute_impurity_fraction(
            wall_material,
            T_edge,
            fw_area,
            plasma_volume,
            tau_ratio,
        )
        wall_derived[symbol] = f_z
        impurities = ImpurityMix(
            wall_derived=wall_derived,
            seeded=seeded_impurities or {},
        )
    elif seeded_impurities and p_rad_override is None:
        # Seeded impurities without wall material (wall_material=None explicitly)
        impurities = ImpurityMix(wall_derived={}, seeded=seeded_impurities)

    # Step 3: Radiation power
    if p_rad_override is not None:
        p_rad = p_rad_override
    else:
        p_rad = compute_p_rad(
            n_e,
            T_e,
            Z_eff,
            plasma_volume,
            B,
            impurities,
            R=R_major,
            a=a_minor,
            kappa=kappa,
            R_w=R_w,
        )

    # Step 4: Plasma energy balance — P_ash + P_input = P_rad + P_transport
    # If P_rad > P_ash + P_input, increase heating to sustain T_e
    p_input_eff = jnp.maximum(p_input, p_rad - p_ash)
    p_transport = p_ash + p_input_eff - p_rad

    # Step 5: Auxiliary power
    p_aux = p_trit + p_house

    # Step 6: DEC routing (operates on transport channel only)
    p_dee = f_dec * eta_de * p_transport
    p_dec_waste = f_dec * (1.0 - eta_de) * p_transport
    p_wall = (1.0 - f_dec) * p_transport

    # Step 7: Thermal power (neutrons + radiation + wall transport + pumping)
    # P_input enters via transport (Step 4), not here
    p_th = mn * p_neutron + p_rad + p_wall + eta_p * p_pump

    # Step 8: Thermal electric
    p_the = eta_th * p_th

    # Step 9: Gross electric
    p_et = p_dee + p_the

    # Step 10: Lost power
    p_loss = (p_th - p_the) + p_dec_waste

    # Step 11: Subsystem power
    p_sub = f_sub * p_et

    # Step 12: Scientific Q
    q_sci = p_fus / p_input_eff

    # Step 13: Engineering Q (uses effective heating power)
    recirculating = (
        p_coils + p_pump + p_sub + p_aux + p_cool + p_cryo + p_input_eff / eta_pin
    )
    q_eng = p_et / recirculating

    # Step 14: Net electric
    rec_frac = 1.0 / q_eng
    p_net = (1.0 - rec_frac) * p_et

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
        p_input=p_input_eff,
        p_pump=p_pump,
        p_sub=p_sub,
        p_aux=p_aux,
        p_coils=p_coils,
        p_cool=p_cool,
        p_cryo=p_cryo,
        p_target=0.0,
        q_sci=q_sci,
        q_eng=q_eng,
        rec_frac=rec_frac,
    )


def mfe_inverse_power_balance(
    p_net_target: float,
    fuel: Fuel,
    p_input: float,
    mn: float,
    eta_th: float,
    eta_p: float,
    eta_pin: float,
    eta_de: float,
    f_sub: float,
    f_dec: float,
    p_coils: float,
    p_cool: float,
    p_pump: float,
    p_trit: float,
    p_house: float,
    p_cryo: float,
    n_e: float,
    T_e: float,
    Z_eff: float,
    plasma_volume: float,
    B: float,
    dd_f_T: float,
    dd_f_He3: float,
    dhe3_dd_frac: float,
    dhe3_f_T: float,
    pb11_f_alpha_n: float,
    pb11_f_p_n: float,
    wall_material: WallMaterial | None,
    T_edge: float,
    tau_ratio: float,
    fw_area: float,
    R_major: float,
    a_minor: float,
    kappa: float,
    R_w: float,
    # Genuine optional overrides (keep defaults)
    p_rad_override: float | None = None,
    seeded_impurities: dict[str, float] | None = None,
) -> float:
    """Inverse MFE power balance: target net electric -> required fusion power.

    Uses Newton iteration to invert the forward power balance.  The forward
    balance is piecewise linear in P_fus due to the plasma energy balance:

        P_ash + P_input_eff = P_rad + P_transport

    where P_input_eff = max(P_input, P_rad - P_ash).  When P_rad exceeds
    P_ash + P_input, the effective heating increases to sustain T_e, which
    raises recirculating power and penalises LCOE.

    Newton converges in ≤2 iterations (piecewise-linear function).
    """
    # Step 1: Build impurity mix (if wall_material provided)
    impurities = None
    if wall_material is not None and p_rad_override is None:
        wall_derived = {}
        symbol, f_z = compute_impurity_fraction(
            wall_material,
            T_edge,
            fw_area,
            plasma_volume,
            tau_ratio,
        )
        wall_derived[symbol] = f_z
        impurities = ImpurityMix(
            wall_derived=wall_derived,
            seeded=seeded_impurities or {},
        )
    elif seeded_impurities and p_rad_override is None:
        # Seeded impurities without wall material (wall_material=None explicitly)
        impurities = ImpurityMix(wall_derived={}, seeded=seeded_impurities)

    # Step 2: Radiation power (constant w.r.t. p_fus)
    if p_rad_override is not None:
        p_rad_raw = p_rad_override
    else:
        p_rad_raw = compute_p_rad(
            n_e,
            T_e,
            Z_eff,
            plasma_volume,
            B,
            impurities,
            R=R_major,
            a=a_minor,
            kappa=kappa,
            R_w=R_w,
        )

    # Step 3: Ash fraction from fuel type (use p_fus=1.0 to get the fraction)
    p_ash_unit, _ = ash_neutron_split(
        1.0, fuel, dd_f_T, dd_f_He3, dhe3_dd_frac, dhe3_f_T, pb11_f_alpha_n, pb11_f_p_n
    )
    ash_frac = p_ash_unit
    neutron_frac = 1.0 - ash_frac

    # Constant recirculating loads (independent of p_fus and p_input_eff)
    p_aux = p_trit + p_house
    p_recirc_base = p_coils + p_pump + p_aux + p_cool + p_cryo

    # Step 4: Forward P_net as a function of P_fus (matches forward balance)
    def _p_net(pf):
        pa = ash_frac * pf
        pn = neutron_frac * pf
        # Plasma energy balance: effective heating covers radiation deficit
        pi_eff = jnp.maximum(p_input, p_rad_raw - pa)
        p_transport = pa + pi_eff - p_rad_raw
        p_dee = f_dec * eta_de * p_transport
        p_wall = (1.0 - f_dec) * p_transport
        # P_input enters via transport, not directly into P_th
        p_th = mn * pn + p_rad_raw + p_wall + eta_p * p_pump
        p_et = eta_th * p_th + p_dee
        p_sub = f_sub * p_et
        return p_et - p_sub - p_recirc_base - pi_eff / eta_pin

    # Analytical dP_net/dP_fus (piecewise constant — function is piecewise linear)
    def _dp_net(pf):
        pa = ash_frac * pf
        capped = p_rad_raw >= pa + p_input
        # Uncapped: p_input_eff = p_input (constant), transport active
        dp_uncapped = (1.0 - f_sub) * (
            eta_th * (mn * neutron_frac + (1.0 - f_dec) * ash_frac)
            + f_dec * eta_de * ash_frac
        )
        # Capped: p_input_eff = p_rad - p_ash → dp_input_eff/dp_fus = -ash_frac
        # transport = 0, p_th = mn*nf*pf + p_rad + eta_p*p_pump
        dp_capped = (1.0 - f_sub) * eta_th * mn * neutron_frac + ash_frac / eta_pin
        return jnp.where(capped, dp_capped, dp_uncapped)

    # Step 5: Linear initial guess (uncapped assumption — P_input in transport)
    c_et = (
        eta_th * (mn * neutron_frac + (1.0 - f_dec) * ash_frac)
        + f_dec * eta_de * ash_frac
    )
    c_et0 = eta_th * (
        f_dec * p_rad_raw + (1.0 - f_dec) * p_input + eta_p * p_pump
    ) + f_dec * eta_de * (p_input - p_rad_raw)
    p_recirc_init = p_recirc_base + p_input / eta_pin
    p_fus = (p_net_target + p_recirc_init - (1.0 - f_sub) * c_et0) / (
        (1.0 - f_sub) * c_et
    )

    # Step 6: Newton refinement (converges in ≤2 for piecewise-linear)
    for _ in range(4):
        residual = _p_net(p_fus) - p_net_target
        dp = _dp_net(p_fus)
        p_fus = p_fus - residual / dp

    return p_fus


# ---------------------------------------------------------------------------
# IFE Power Balance
# ---------------------------------------------------------------------------


def ife_forward_power_balance(
    p_fus: float,
    fuel: Fuel,
    p_implosion: float,
    p_ignition: float,
    mn: float,
    eta_th: float,
    eta_p: float,
    eta_pin1: float,
    eta_pin2: float,
    f_sub: float,
    p_pump: float,
    p_trit: float,
    p_house: float,
    p_cryo: float,
    p_target: float,
    dd_f_T: float = DD_F_T_DEFAULT,
    dd_f_He3: float = DD_F_HE3_DEFAULT,
    dhe3_dd_frac: float = 0.07,
    dhe3_f_T: float = 0.97,
    pb11_f_alpha_n: float = 0.0,
    pb11_f_p_n: float = 0.0,
) -> PowerTable:
    """IFE forward power balance: fusion power -> net electric.

    Key differences from MFE:
    - No DEC (all ash thermalizes)
    - Split driver: implosion + ignition, each with own wall-plug efficiency
    - Target factory power in recirculating
    - No coil power or coil cooling
    - No plasma confinement radiation (p_rad=0)
    Source: pyFECONs power_balance.py (IFE branch)
    """
    p_input = p_implosion + p_ignition

    # Step 1: Ash/neutron split
    p_ash, p_neutron = ash_neutron_split(
        p_fus,
        fuel,
        dd_f_T,
        dd_f_He3,
        dhe3_dd_frac,
        dhe3_f_T,
        pb11_f_alpha_n,
        pb11_f_p_n,
    )

    # Step 2: No DEC in IFE — all ash thermalizes
    p_rad = 0.0
    p_wall = p_ash
    p_dee = 0.0
    p_dec_waste = 0.0

    # Step 3: Auxiliary power
    p_aux = p_trit + p_house

    # Step 4: Thermal power (all ash thermalizes + neutrons + heating + pumping)
    p_th = mn * p_neutron + p_ash + p_input + eta_p * p_pump

    # Step 5: Thermal electric
    p_the = eta_th * p_th

    # Step 6: Gross electric (no DEC)
    p_et = p_the

    # Step 7: Lost power
    p_loss = p_th - p_the

    # Step 8: Subsystem power
    p_sub = f_sub * p_et

    # Step 9: Scientific Q
    q_sci = p_fus / p_input

    # Step 10: Engineering Q — split driver efficiencies
    recirculating = (
        p_target
        + p_pump
        + p_sub
        + p_aux
        + p_cryo
        + p_implosion / eta_pin1
        + p_ignition / eta_pin2
    )
    q_eng = p_et / recirculating

    # Step 11: Net electric
    rec_frac = 1.0 / q_eng
    p_net = (1.0 - rec_frac) * p_et

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
        p_input=0.0,
        p_pump=p_pump,
        p_sub=p_sub,
        p_aux=p_aux,
        p_coils=0.0,
        p_cool=0.0,
        p_cryo=p_cryo,
        p_target=p_target,
        q_sci=q_sci,
        q_eng=q_eng,
        rec_frac=rec_frac,
    )


def ife_inverse_power_balance(
    p_net_target: float,
    fuel: Fuel,
    p_implosion: float,
    p_ignition: float,
    mn: float,
    eta_th: float,
    eta_p: float,
    eta_pin1: float,
    eta_pin2: float,
    f_sub: float,
    p_pump: float,
    p_trit: float,
    p_house: float,
    p_cryo: float,
    p_target: float,
    dd_f_T: float = DD_F_T_DEFAULT,
    dd_f_He3: float = DD_F_HE3_DEFAULT,
    dhe3_dd_frac: float = 0.07,
    dhe3_f_T: float = 0.97,
    pb11_f_alpha_n: float = 0.0,
    pb11_f_p_n: float = 0.0,
) -> float:
    """Inverse IFE power balance: target net electric -> required fusion power.

    Closed-form linear inversion (same approach as MFE inverse).
    """
    p_input = p_implosion + p_ignition

    # Ash fraction
    ash_frac, _ = ash_neutron_split(
        1.0,
        fuel,
        dd_f_T,
        dd_f_He3,
        dhe3_dd_frac,
        dhe3_f_T,
        pb11_f_alpha_n,
        pb11_f_p_n,
    )
    neutron_frac = 1.0 - ash_frac

    # All ash thermalizes (no DEC in IFE)
    c_th = mn * neutron_frac + ash_frac
    c_th0 = p_input + eta_p * p_pump

    # Gross electric = thermal electric (no DEC)
    c_et = eta_th * c_th
    c_et0 = eta_th * c_th0

    # Recirculating (p_fus-dependent part): p_sub = f_sub * p_et
    c_den = f_sub * c_et

    # Recirculating (constant loads)
    p_aux = p_trit + p_house
    c_den0 = (
        p_target
        + p_pump
        + f_sub * c_et0
        + p_aux
        + p_cryo
        + p_implosion / eta_pin1
        + p_ignition / eta_pin2
    )

    p_fus = (p_net_target - c_et0 + c_den0) / (c_et - c_den)
    return p_fus


# ---------------------------------------------------------------------------
# MIF Power Balance
# ---------------------------------------------------------------------------


def mif_forward_power_balance(
    p_fus: float,
    fuel: Fuel,
    p_driver: float,
    mn: float,
    eta_th: float,
    eta_p: float,
    eta_pin: float,
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
    """MIF forward power balance: fusion power -> net electric.

    MIF uses pulsed magnetic compression. Key differences:
    - Single driver (pulsed power) with one wall-plug efficiency
    - Target/liner factory power in recirculating
    - Optional small guide-field coils
    - No DEC (all ash thermalizes)
    - No plasma confinement radiation
    """
    # Step 1: Ash/neutron split
    p_ash, p_neutron = ash_neutron_split(
        p_fus,
        fuel,
        dd_f_T,
        dd_f_He3,
        dhe3_dd_frac,
        dhe3_f_T,
        pb11_f_alpha_n,
        pb11_f_p_n,
    )

    # Step 2: No DEC — all ash thermalizes
    p_rad = 0.0
    p_wall = p_ash
    p_dee = 0.0
    p_dec_waste = 0.0

    # Step 3: Auxiliary power
    p_aux = p_trit + p_house

    # Step 4: Thermal power
    p_th = mn * p_neutron + p_ash + p_driver + eta_p * p_pump

    # Step 5: Thermal electric
    p_the = eta_th * p_th

    # Step 6: Gross electric (no DEC)
    p_et = p_the

    # Step 7: Lost power
    p_loss = p_th - p_the

    # Step 8: Subsystem power
    p_sub = f_sub * p_et

    # Step 9: Scientific Q
    q_sci = p_fus / p_driver

    # Step 10: Engineering Q
    recirculating = (
        p_target + p_coils + p_pump + p_sub + p_aux + p_cryo + p_driver / eta_pin
    )
    q_eng = p_et / recirculating

    # Step 11: Net electric
    rec_frac = 1.0 / q_eng
    p_net = (1.0 - rec_frac) * p_et

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
        p_input=0.0,
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
    )


def mif_inverse_power_balance(
    p_net_target: float,
    fuel: Fuel,
    p_driver: float,
    mn: float,
    eta_th: float,
    eta_p: float,
    eta_pin: float,
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
    """Inverse MIF power balance: target net electric -> required fusion power.

    Closed-form linear inversion.
    """
    # Ash fraction
    ash_frac, _ = ash_neutron_split(
        1.0,
        fuel,
        dd_f_T,
        dd_f_He3,
        dhe3_dd_frac,
        dhe3_f_T,
        pb11_f_alpha_n,
        pb11_f_p_n,
    )
    neutron_frac = 1.0 - ash_frac

    # All ash thermalizes
    c_th = mn * neutron_frac + ash_frac
    c_th0 = p_driver + eta_p * p_pump

    # Gross electric = thermal electric
    c_et = eta_th * c_th
    c_et0 = eta_th * c_th0

    # Recirculating
    c_den = f_sub * c_et

    p_aux = p_trit + p_house
    c_den0 = (
        p_target
        + p_coils
        + p_pump
        + f_sub * c_et0
        + p_aux
        + p_cryo
        + p_driver / eta_pin
    )

    p_fus = (p_net_target - c_et0 + c_den0) / (c_et - c_den)
    return p_fus
