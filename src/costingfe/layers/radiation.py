"""Plasma radiation model: bremsstrahlung, synchrotron, impurity line radiation."""

import jax.numpy as jnp

from costingfe.types import ImpurityMix, WallMaterial

# ---------------------------------------------------------------------------
# Sputtering yield model (Bohdansky/Eckstein physical sputtering)
# ---------------------------------------------------------------------------
# Fit params: (E_th_eV, Q_eff)
# Simplified Bohdansky: Y(E) = Q_eff * (1 - (E_th/E)^(2/3)) * (1 - E_th/E)^2
# Q_eff absorbs the Thomas-Fermi nuclear stopping cross-section S_n(ε).
# Source: Eckstein 2007, Bohdansky 1984
_SPUT_PARAMS: dict[str, tuple[float, float]] = {
    "W": (220.0, 0.042),
    "C": (28.0, 0.060),
    "Be": (15.0, 0.100),
    "Mo": (100.0, 0.050),
    "SiC": (40.0, 0.055),  # Si-like effective values
    "Li": (10.0, 0.020),  # Low yield from flowing surface renewal
}

# Default SOL screening factors by atomic number category
_SOL_SCREENING = {
    "W": 0.01,
    "Mo": 0.01,  # high-Z: strong screening
    "C": 0.10,
    "Be": 0.10,
    "SiC": 0.10,  # low-Z: weaker screening
    "Li": 0.005,  # liquid Li: flowing surface, very low penetration
}


def compute_sputtering_yield(T_edge_keV: float, wall_material: WallMaterial) -> float:
    """Physical sputtering yield Y (atoms/ion) at incident energy ~3*T_edge.

    Uses simplified Bohdansky formula. T_edge_keV is the edge ion temperature.
    Incident ion energy is approximated as 3*T_edge (sheath-accelerated D ions).
    """
    symbol = wall_material.value
    E_th, Q = _SPUT_PARAMS[symbol]
    E_ion = 3.0 * T_edge_keV * 1e3  # keV -> eV

    # Below threshold: zero yield
    ratio = E_th / jnp.maximum(E_ion, 1.0)  # avoid division by zero
    below_threshold = E_ion <= E_th

    # Simplified Bohdansky: Y = Q * (1 - (E_th/E)^(2/3)) * (1 - E_th/E)^2
    Y = Q * (1.0 - ratio ** (2.0 / 3.0)) * (1.0 - ratio) ** 2
    Y = jnp.where(below_threshold, 0.0, Y)
    return Y


def compute_impurity_fraction(
    wall_material: WallMaterial,
    T_edge_keV: float,
    fw_area: float,
    plasma_volume: float,
    tau_ratio: float = 3.0,
) -> tuple[str, float]:
    """Steady-state impurity fraction from wall sputtering.

    Returns (species_symbol, f_z) where f_z = n_z/n_e.

    f_z = Y(3*T_edge) * (A_wall / V_plasma) * tau_ratio * f_screen

    f_screen is looked up from _SOL_SCREENING by wall material.
    """
    symbol = wall_material.value
    f_screen = _SOL_SCREENING[symbol]
    Y = compute_sputtering_yield(T_edge_keV, wall_material)
    f_z = Y * (fw_area / jnp.maximum(plasma_volume, 1.0)) * tau_ratio * f_screen
    return symbol, f_z


# ---------------------------------------------------------------------------
# Coronal equilibrium cooling curves L_z(T_e) [W·m³]
# ---------------------------------------------------------------------------
# Piecewise power-law fits to Post-Jensen / Mavrin data.
# Each species: list of (T_min_keV, T_max_keV, coefficient, exponent)
# such that L_z = coeff * T_e^exponent in that range.
# Units: L_z in W·m³, T_e in keV.
_COOLING_CURVES: dict[str, list[tuple[float, float, float, float]]] = {
    "W": [
        (0.01, 0.1, 5.0e-31, -1.0),  # strong radiation at low T
        (0.1, 1.0, 1.5e-31, 0.5),  # rising through ionization stages
        (1.0, 10.0, 5.0e-31, 0.0),  # plateau
        (10.0, 100.0, 5.0e-31, -0.5),  # slow decline
    ],
    "C": [
        (0.01, 0.01, 1.0e-31, 0.0),  # fully ionized above ~0.3 keV
        (0.01, 0.1, 3.0e-32, -0.5),
        (0.1, 0.5, 1.0e-32, 0.5),
        (0.5, 100.0, 1.0e-36, 0.0),  # fully stripped, negligible
    ],
    "Be": [
        (0.01, 0.05, 2.0e-32, -0.5),
        (0.05, 0.3, 5.0e-33, 0.5),
        (0.3, 100.0, 1.0e-36, 0.0),  # fully stripped
    ],
    "Mo": [
        (0.01, 0.1, 2.0e-31, -1.0),
        (0.1, 1.0, 8.0e-32, 0.5),
        (1.0, 10.0, 2.0e-31, 0.0),
        (10.0, 100.0, 2.0e-31, -0.5),
    ],
    "Si": [
        (0.01, 0.1, 1.0e-31, -0.5),
        (0.1, 1.0, 3.0e-32, 0.5),
        (1.0, 100.0, 5.0e-33, 0.0),  # mostly stripped
    ],
    "Li": [
        # Li (Z=3): fully ionized above ~0.1 keV, negligible line radiation
        (0.01, 0.05, 1.0e-33, -0.5),
        (0.05, 0.1, 5.0e-34, 0.0),
        (0.1, 100.0, 1.0e-38, 0.0),  # ~zero at reactor temperatures
    ],
    "Ne": [
        (0.01, 0.1, 5.0e-32, -0.5),
        (0.1, 0.5, 2.0e-32, 0.5),
        (0.5, 2.0, 4.0e-32, 0.0),  # Ne-like shell peak
        (2.0, 100.0, 4.0e-32, -1.0),  # decline
    ],
    "Ar": [
        (0.01, 0.1, 1.0e-31, -0.5),
        (0.1, 1.0, 5.0e-32, 0.5),
        (1.0, 5.0, 1.0e-31, 0.0),  # broad peak
        (5.0, 100.0, 1.0e-31, -0.5),  # decline
    ],
}


def cooling_rate(species: str, T_e_keV: float) -> float:
    """Coronal equilibrium radiative loss rate L_z [W·m³].

    Piecewise power-law fits for each species. Returns L_z at given T_e.
    For species not in the table, returns 0.
    """
    # Map SiC wall material to Si cooling curve
    lookup = "Si" if species == "SiC" else species
    if lookup not in _COOLING_CURVES:
        return 0.0

    segments = _COOLING_CURVES[lookup]
    T_e = jnp.maximum(T_e_keV, 0.01)

    # Evaluate all segments and select the matching one
    result = 0.0
    for T_min, T_max, coeff, exponent in segments:
        in_range = (T_e >= T_min) & (T_e < T_max)
        L_z = coeff * T_e**exponent
        result = jnp.where(in_range, L_z, result)

    return result


def compute_p_line(
    n_e: float,
    T_e: float,
    impurities: ImpurityMix | None,
    volume: float,
) -> float:
    """Impurity line radiation power [MW].

    P_line = n_e² × Σ(f_z × L_z(T_e)) × V
    """
    if impurities is None:
        return 0.0
    total_Lz = 0.0
    all_species = {**impurities.wall_derived, **impurities.seeded}
    for species, f_z in all_species.items():
        total_Lz = total_Lz + f_z * cooling_rate(species, T_e)
    # Scale n_e to avoid float32 overflow (n_e~1e20, n_e^2~1e40 > float32 max)
    n_e_20 = n_e * 1e-20
    return total_Lz * n_e_20**2 * volume * 1e34  # 1e40 * 1e-6 = 1e34


_BETA_T = 2.0  # Temperature profile shape exponent (fixed; see Appendix A)


def compute_p_sync_albajar(
    T_e0: float,
    n_e0_20: float,
    B: float,
    R: float,
    a: float,
    kappa: float = 1.7,
    R_w: float = 0.6,
    alpha_n: float = 0.5,
    alpha_T: float = 1.0,
) -> float:
    """Synchrotron radiation power [MW] — Albajar et al. (2001).

    Accounts for optical thickness and wall reflection.

    Ref: Albajar, Johner, Granata, Nucl. Fusion 41 665 (2001), eqs 13, 15.
         Fidone, Giruzzi, Granata, Nucl. Fusion 41 1755 (2001).

    Parameters
    ----------
    T_e0 : central electron temperature [keV]
    n_e0_20 : central electron density [10^20 m^-3]
    B : toroidal magnetic field on axis [T]
    R : major radius [m]
    a : minor radius [m]
    kappa : elongation (default 1.7)
    R_w : wall reflectivity (default 0.6; metallic walls ~0.6-0.8)
    alpha_n, alpha_T : density and temperature profile peaking exponents
    """
    A = R / a
    p_a0 = 6.04e3 * a * n_e0_20 / B

    # Wall reflection / optical thickness correction
    correction = (1 - R_w) ** 0.62 / (
        1 + 0.12 * T_e0 / p_a0**0.41 * (1 - R_w) ** 0.41
    ) ** 1.51

    # Profile shape factor K (beta_T fixed at 2)
    K = (
        (alpha_n + 3.87 * alpha_T + 1.46) ** (-0.79)
        * (1.98 + alpha_T) ** 1.36
        * _BETA_T**2.14
        * (_BETA_T**1.53 + 1.87 * alpha_T - 0.16) ** (-1.33)
    )

    # Aspect ratio correction G
    G = 0.93 * (1 + 0.85 * jnp.exp(-0.82 * A))

    return (
        3.84e-8
        * correction
        * R
        * a**1.38
        * kappa**0.79
        * B**2.62
        * n_e0_20**0.38
        * T_e0
        * (16 + T_e0) ** 2.61
        * K
        * G
    )


def compute_p_rad(
    n_e: float,
    T_e: float,
    Z_eff: float,
    volume: float,
    B: float = 0.0,
    impurities: ImpurityMix | None = None,
    R: float = 0.0,
    a: float = 0.0,
    kappa: float = 1.7,
    R_w: float = 0.6,
    alpha_n: float = 0.5,
    alpha_T: float = 1.0,
) -> float:
    """Plasma radiation power (bremsstrahlung + synchrotron + line) [MW].

    P_brem = 5.35e-37 * n_e^2 * Z_eff * sqrt(T_e) * V  [W], T_e in keV
    P_sync: Albajar et al. (2001) when R and a > 0; zero otherwise
    P_line = n_e^2 * Σ(f_z * L_z(T_e)) * V  [W]

    Volume-averaged n_e, T_e are converted to central values using
    the profile exponents: T_e0 = T_e * (1 + alpha_T),
    n_e0 = n_e * (1 + alpha_n).  Assumes beta_T = 2 (parabolic).
    """
    n_e_20 = n_e * 1e-20
    p_brem = 5.35e3 * n_e_20**2 * Z_eff * jnp.sqrt(T_e) * volume * 1e-6  # -> MW

    # Synchrotron: Albajar when geometry available, else zero
    use_albajar = jnp.logical_and(R > 0, a > 0)
    # Convert volume-averaged to central: <f> = f_0 / (1 + alpha) for beta_T=2
    T_e0 = T_e * (1 + alpha_T)
    n_e0_20 = n_e_20 * (1 + alpha_n)
    # Clamp a > 0 to prevent R/a division by zero: in plain Python
    # (validation path) this raises ZeroDivisionError; in JAX it
    # produces inf/nan that poison gradients through jnp.where.
    a_safe = jnp.maximum(a, 1e-10)
    p_sync_albajar = compute_p_sync_albajar(
        T_e0,
        n_e0_20,
        B,
        R,
        a_safe,
        kappa,
        R_w,
        alpha_n,
        alpha_T,
    )
    p_sync = jnp.where(use_albajar, p_sync_albajar, 0.0)

    p_line = compute_p_line(n_e, T_e, impurities, volume)
    return p_brem + p_sync + p_line
