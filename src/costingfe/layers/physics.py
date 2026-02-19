"""Layer 2: Physics — fuel physics, power balance (forward + inverse)."""

from typing import Optional

import jax.numpy as jnp
from scipy import constants as sc

from costingfe.types import Fuel, PowerTable

# ---------------------------------------------------------------------------
# Fundamental constants from scipy (CODATA)
# ---------------------------------------------------------------------------
MEV_TO_JOULES = sc.eV * 1e6  # 1 MeV in J
M_DEUTERIUM_KG = sc.physical_constants["deuteron mass"][0]  # kg

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

# DD primary per-event averages (50/50 branches)
_E_CHARGED_PRIMARY_DD = 0.5 * (E_T_DD + E_P_DD) + 0.5 * E_HE3_DD  # ~2.425
_E_NEUTRON_PRIMARY_DD = 0.5 * E_N_DD  # ~1.225
_E_TOTAL_PRIMARY_DD = 0.5 * Q_DD_PT + 0.5 * Q_DD_NHE3  # ~3.65


def ash_neutron_split(
    p_fus: float,
    fuel: Fuel,
    dd_f_T: float = 0.969,
    dd_f_He3: float = 0.689,
    dhe3_dd_frac: float = 0.07,
    dhe3_f_T: float = 0.97,
) -> tuple[float, float]:
    """Compute charged-particle (ash) and neutron power from fusion power.

    Returns (p_ash, p_neutron) in MW. All paths are JAX-differentiable.

    Source: pyFECONs fuel_physics.py:compute_ash_neutron_split
    """
    if fuel == Fuel.DT:
        ash_frac = E_ALPHA_DT / Q_DT
    elif fuel == Fuel.DD:
        E_charged = (
            _E_CHARGED_PRIMARY_DD
            + 0.5 * dd_f_T * E_ALPHA_DT
            + 0.5 * dd_f_He3 * Q_DHE3
        )
        E_total = (
            _E_TOTAL_PRIMARY_DD + 0.5 * dd_f_T * Q_DT + 0.5 * dd_f_He3 * Q_DHE3
        )
        ash_frac = E_charged / E_total
    elif fuel == Fuel.DHE3:
        E_n_dd = _E_NEUTRON_PRIMARY_DD + 0.5 * dhe3_f_T * E_N_DT
        E_c_dd = _E_CHARGED_PRIMARY_DD + 0.5 * dhe3_f_T * E_ALPHA_DT
        ash_frac = (1 - dhe3_dd_frac) + dhe3_dd_frac * E_c_dd / (E_n_dd + E_c_dd)
    elif fuel == Fuel.PB11:
        ash_frac = 1.0
    else:
        raise ValueError(f"Unknown fuel type: {fuel}")

    p_ash = p_fus * ash_frac
    p_neutron = p_fus * (1.0 - ash_frac)
    return p_ash, p_neutron


def compute_p_rad(
    n_e: float,
    T_e: float,
    Z_eff: float,
    volume: float,
    B: float = 0.0,
) -> float:
    """Plasma radiation power (bremsstrahlung + synchrotron).

    Calculated from plasma parameters by default, can be overridden.
    P_brem = 5.35e-37 * n_e^2 * Z_eff * sqrt(T_e) * V  [W], T_e in keV, n_e in m^-3
    P_sync = 6.2e-22 * n_e * T_e^2 * B^2 * V  [W] (MFE only)
    """
    p_brem = 5.35e-37 * n_e**2 * Z_eff * jnp.sqrt(T_e) * volume * 1e-6  # -> MW
    p_sync = 6.2e-22 * n_e * T_e**2 * B**2 * volume * 1e-6  # -> MW
    return p_brem + p_sync


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
    # Radiation: calculated from plasma params, or override with p_rad_override
    n_e: float = 1.0e20,
    T_e: float = 15.0,
    Z_eff: float = 1.5,
    plasma_volume: float = 500.0,
    B: float = 5.0,
    p_rad_override: Optional[float] = None,
    dd_f_T: float = 0.969,
    dd_f_He3: float = 0.689,
    dhe3_dd_frac: float = 0.07,
    dhe3_f_T: float = 0.97,
) -> PowerTable:
    """MFE forward power balance: fusion power -> net electric.

    Source: pyFECONs power_balance.py + fusion-tea mfe_power_balance.sysml
    Radiation model: bremsstrahlung + synchrotron from plasma parameters.
    """
    # Step 1: Ash/neutron split
    p_ash, p_neutron = ash_neutron_split(
        p_fus, fuel, dd_f_T, dd_f_He3, dhe3_dd_frac, dhe3_f_T
    )

    # Step 2: Radiation power (p_rad + p_transport = p_ash)
    if p_rad_override is not None:
        p_rad = p_rad_override
    else:
        p_rad = compute_p_rad(n_e, T_e, Z_eff, plasma_volume, B)
    p_rad = jnp.minimum(p_rad, p_ash)  # Can't radiate more than ash power
    p_transport = p_ash - p_rad

    # Step 3: Auxiliary power
    p_aux = p_trit + p_house

    # Step 4: DEC routing (operates on transport channel only)
    p_dee = f_dec * eta_de * p_transport
    p_dec_waste = f_dec * (1.0 - eta_de) * p_transport
    p_wall = (1.0 - f_dec) * p_transport

    # Step 5: Thermal power (neutrons + radiation + wall transport + heating + pumping)
    p_th = mn * p_neutron + p_rad + p_wall + p_input + eta_p * p_pump

    # Step 6: Thermal electric
    p_the = eta_th * p_th

    # Step 7: Gross electric
    p_et = p_dee + p_the

    # Step 8: Lost power
    p_loss = (p_th - p_the) + p_dec_waste

    # Step 9: Subsystem power
    p_sub = f_sub * p_et

    # Step 10: Scientific Q
    q_sci = p_fus / p_input

    # Step 11: Engineering Q
    recirculating = (
        p_coils + p_pump + p_sub + p_aux + p_cool + p_cryo + p_input / eta_pin
    )
    q_eng = p_et / recirculating

    # Step 12: Net electric
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
    # Radiation: calculated from plasma params, or override
    n_e: float = 1.0e20,
    T_e: float = 15.0,
    Z_eff: float = 1.5,
    plasma_volume: float = 500.0,
    B: float = 5.0,
    p_rad_override: Optional[float] = None,
    dd_f_T: float = 0.969,
    dd_f_He3: float = 0.689,
    dhe3_dd_frac: float = 0.07,
    dhe3_f_T: float = 0.97,
) -> float:
    """Inverse MFE power balance: target net electric -> required fusion power.

    Closed-form linear inversion (no iteration needed).
    p_rad is constant w.r.t. p_fus (depends on plasma params, not fusion power),
    so it enters as a constant term in the linear inversion.
    Source: fusion-tea Inverse MFE Power Balance Calc
    """
    # Step 1: Radiation power (constant w.r.t. p_fus)
    if p_rad_override is not None:
        p_rad = p_rad_override
    else:
        p_rad = compute_p_rad(n_e, T_e, Z_eff, plasma_volume, B)

    # Step 2: Ash fraction from fuel type (use p_fus=1.0 to get the fraction)
    p_ash_unit, _ = ash_neutron_split(
        1.0, fuel, dd_f_T, dd_f_He3, dhe3_dd_frac, dhe3_f_T
    )
    ash_frac = p_ash_unit
    neutron_frac = 1.0 - ash_frac

    # Step 3: Linearize forward chain coefficients
    # Thermal power per unit p_fus
    c_th = mn * neutron_frac + (1.0 - f_dec) * ash_frac

    # Constant thermal power (radiation + heating + pumping)
    # p_rad - (1-f_dec)*p_rad simplifies to f_dec*p_rad
    c_th0 = f_dec * p_rad + p_input + eta_p * p_pump

    # DEC electric per unit p_fus
    c_dee = f_dec * eta_de * ash_frac
    c_dee0 = -f_dec * eta_de * p_rad

    # Gross electric: p_et = c_et * p_fus + c_et0
    c_et = c_dee + eta_th * c_th
    c_et0 = c_dee0 + eta_th * c_th0

    # Recirculating (p_fus-dependent): p_sub = f_sub * p_et
    c_den = f_sub * c_et

    # Recirculating (constant loads)
    p_aux = p_trit + p_house
    c_den0 = (
        p_coils + p_pump + f_sub * c_et0 + p_aux + p_cool + p_cryo + p_input / eta_pin
    )

    # Step 4: Solve p_net = (c_et - c_den) * p_fus + (c_et0 - c_den0)
    p_fus = (p_net_target - c_et0 + c_den0) / (c_et - c_den)
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
    dd_f_T: float = 0.969,
    dd_f_He3: float = 0.689,
    dhe3_dd_frac: float = 0.07,
    dhe3_f_T: float = 0.97,
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
        p_fus, fuel, dd_f_T, dd_f_He3, dhe3_dd_frac, dhe3_f_T
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
        p_target + p_pump + p_sub + p_aux + p_cryo
        + p_implosion / eta_pin1 + p_ignition / eta_pin2
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
    dd_f_T: float = 0.969,
    dd_f_He3: float = 0.689,
    dhe3_dd_frac: float = 0.07,
    dhe3_f_T: float = 0.97,
) -> float:
    """Inverse IFE power balance: target net electric -> required fusion power.

    Closed-form linear inversion (same approach as MFE inverse).
    """
    p_input = p_implosion + p_ignition

    # Ash fraction
    ash_frac, _ = ash_neutron_split(1.0, fuel, dd_f_T, dd_f_He3, dhe3_dd_frac, dhe3_f_T)
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
        p_target + p_pump + f_sub * c_et0 + p_aux + p_cryo
        + p_implosion / eta_pin1 + p_ignition / eta_pin2
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
    dd_f_T: float = 0.969,
    dd_f_He3: float = 0.689,
    dhe3_dd_frac: float = 0.07,
    dhe3_f_T: float = 0.97,
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
        p_fus, fuel, dd_f_T, dd_f_He3, dhe3_dd_frac, dhe3_f_T
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
        p_target + p_coils + p_pump + p_sub + p_aux + p_cryo
        + p_driver / eta_pin
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
    dd_f_T: float = 0.969,
    dd_f_He3: float = 0.689,
    dhe3_dd_frac: float = 0.07,
    dhe3_f_T: float = 0.97,
) -> float:
    """Inverse MIF power balance: target net electric -> required fusion power.

    Closed-form linear inversion.
    """
    # Ash fraction
    ash_frac, _ = ash_neutron_split(1.0, fuel, dd_f_T, dd_f_He3, dhe3_dd_frac, dhe3_f_T)
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
        p_target + p_coils + p_pump + f_sub * c_et0 + p_aux + p_cryo
        + p_driver / eta_pin
    )

    p_fus = (p_net_target - c_et0 + c_den0) / (c_et - c_den)
    return p_fus
