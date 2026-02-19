"""Layer 2: Physics — fuel physics, power balance (forward + inverse)."""

import jax.numpy as jnp
from scipy import constants as sc
from costingfe.types import Fuel

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
