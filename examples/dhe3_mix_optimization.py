"""D-He3 fuel mix optimization for direct energy conversion.

Sweeps the D/He3 density ratio r = n_He3/n_D to find the mix that
minimizes the unrecoverable energy fraction (neutrons + bremsstrahlung)
in a thermal D-He3 plasma. Used by the "Direct Energy Conversion and
the Cost Floor" post to compute the DEC-optimal fuel mix and to set
self-consistent 1costingfe defaults.

Physics:
- Cross-sections from Bosch-Hale (Nucl. Fusion 32, 611, 1992)
- Bremsstrahlung from NRL Plasma Formulary, with relativistic + e-e
  corrections after Rider (1995)
- Energy bookkeeping: D-He3 primary (18.35 MeV, all charged) +
  D-D side reactions (with tritium burnup chain)

Caveats:
- Assumes thermal equilibrium (T_e = T_i)
- No alpha-ash buildup in Z_eff
- Synchrotron / impurity-line radiation not included
"""

import numpy as np
from scipy.optimize import minimize_scalar


# ============================================================
# Bosch-Hale (1992) thermal reactivities
# ============================================================
def _bh(T, BG, mrc2, C1, C2, C3, C4, C5, C6, C7):
    """Bosch-Hale parametrization. T in keV, returns <sigma*v> in m^3/s."""
    num = T * (C2 + T * (C4 + T * C6))
    den = 1 + T * (C3 + T * (C5 + T * C7))
    theta = T / (1 - num / den)
    xi = (BG**2 / (4 * theta)) ** (1.0 / 3)
    return C1 * theta * np.sqrt(xi / (mrc2 * T**3)) * np.exp(-3 * xi) * 1e-6


def sigv_dhe3(T):
    """D + 3He -> p + 4He."""
    return _bh(
        T,
        68.7508,
        1124572,
        5.51036e-10,
        6.41918e-3,
        -2.02896e-3,
        -1.91080e-5,
        1.35776e-4,
        0,
        0,
    )


def sigv_dd_n(T):
    """D + D -> n + 3He branch."""
    return _bh(
        T, 31.3970, 937814, 5.43360e-12, 5.85778e-3, 7.68222e-3, 0, -2.96400e-6, 0, 0
    )


def sigv_dd_p(T):
    """D + D -> p + T branch."""
    return _bh(
        T, 31.3970, 937814, 5.65718e-12, 3.41267e-3, 1.99167e-3, 0, 1.05060e-5, 0, 0
    )


def sigv_dd_total(T):
    return sigv_dd_n(T) + sigv_dd_p(T)


# ============================================================
# Per-event energies (MeV)
# ============================================================
E_DHE3 = 18.35  # all charged (4He + p)

# D-D events (mean over the two equally probable branches), with
# tritium-burnup contribution from T+D -> 4He + n
F_T_BURNUP = 0.97
E_N_DD = 0.5 * 2.45 + 0.5 * F_T_BURNUP * 14.06
E_C_DD = 0.5 * 0.82 + 0.5 * 4.03 + 0.5 * F_T_BURNUP * 3.52
E_DD_TOT = E_N_DD + E_C_DD


# ============================================================
# Bremsstrahlung — non-relativistic plus relativistic + e-e corrections
# ============================================================
def brem_factor_rel(T_keV, Zeff):
    """Relativistic e-i + e-e correction to non-relativistic NRL brem.

    From Rider (1995) / Wesson Tokamaks. m_e c^2 = 511 keV.
    """
    tau = T_keV / 511.0
    rel_ei = 1 + 0.7936 * tau + 1.874 * tau**2
    ee_correction = 1.5 * tau / Zeff
    return rel_ei + ee_correction


def fractions(r, T_keV, n_e=1e20, relativistic=True):
    """Compute neutron / bremsstrahlung / extractable fractions of fusion power.

    Inputs:
        r = n_He3 / n_D
        T_keV = ion/electron temperature (assumed equal)
        n_e = electron density (m^-3); cancels out for fractions
        relativistic = include relativistic + e-e brem corrections

    Returns dict with f_n, f_brem, f_ext, f_DD (reaction fraction), Zeff.
    """
    Zeff = (1 + 4 * r) / (1 + 2 * r)
    n_D = n_e / (1 + 2 * r)
    n_He3 = r * n_e / (1 + 2 * r)

    R_DHe3 = n_D * n_He3 * sigv_dhe3(T_keV)
    R_DD = 0.5 * n_D**2 * sigv_dd_total(T_keV)

    MeV_to_J = 1.602e-13
    P_fus = (R_DHe3 * E_DHE3 + R_DD * E_DD_TOT) * MeV_to_J
    P_n = R_DD * E_N_DD * MeV_to_J
    P_c = (R_DHe3 * E_DHE3 + R_DD * E_C_DD) * MeV_to_J

    P_brem_NR = 1.69e-38 * Zeff * n_e**2 * np.sqrt(T_keV * 1000)
    P_brem = P_brem_NR * (brem_factor_rel(T_keV, Zeff) if relativistic else 1.0)

    f_n = P_n / P_fus
    f_brem = P_brem / P_fus
    f_ext = max(P_c / P_fus - f_brem, 0.0)
    f_DD = R_DD / (R_DHe3 + R_DD)

    return dict(f_n=f_n, f_brem=f_brem, f_ext=f_ext, f_DD=f_DD, Zeff=Zeff)


def find_optimum(T_keV, r_bounds=(0.05, 20.0), relativistic=True):
    """Return (r_opt, fractions_dict) minimizing f_n + f_brem at fixed T."""
    res = minimize_scalar(
        lambda r: fractions(r, T_keV, relativistic=relativistic)["f_n"]
        + fractions(r, T_keV, relativistic=relativistic)["f_brem"],
        bounds=r_bounds,
        method="bounded",
    )
    r = float(res.x)
    return r, fractions(r, T_keV, relativistic=relativistic)


# ============================================================
# Total fusion power at fixed plasma pressure
# ============================================================
# Constraint: 2 n_D + 3 n_He3 = const (proportional to plasma pressure
# at T_e = T_i, charge-balanced D-He3 plasma).
# Total fusion power (D-He3 + D-D side) per unit volume:
#   P_fus(r) ∝ [r * sigv_DHe3 * E_DHe3 + 0.5 * sigv_DD * E_DD] / (2+3r)^2
# Maximize w.r.t. r.


def total_fusion_power_density(r, T_keV, N=1.0):
    """Total P_fus per unit volume at fixed pressure parameter N = 2 n_D + 3 n_He3.

    Returns P_fus in arbitrary units (proportional to N^2 * T-dependent terms).
    Use the relative magnitude to compare different mixes at the same pressure.
    """
    A = sigv_dhe3(T_keV) * E_DHE3  # MeV * m^3/s
    B = sigv_dd_total(T_keV) * E_DD_TOT
    return (r * A + 0.5 * B) / (2 + 3 * r) ** 2 * N**2


def find_max_fusion_power(T_keV, r_bounds=(0.05, 20.0)):
    """Find r maximizing total fusion power at fixed plasma pressure.

    Analytical optimum: r = 2/3 - B/A  where A = sigv_DHe3 * E_DHe3,
    B = sigv_DD * E_DD_total. Returns (r_opt, fractions_dict, relative_power).
    """
    A = sigv_dhe3(T_keV) * E_DHE3
    B = sigv_dd_total(T_keV) * E_DD_TOT
    r_opt = max(2.0 / 3 - B / A, r_bounds[0])
    return r_opt, fractions(r_opt, T_keV)


def relative_reactivity(r, T_keV):
    """Total fusion power at mix r, normalized to peak at fixed pressure (=1.0)."""
    r_peak, _ = find_max_fusion_power(T_keV)
    return total_fusion_power_density(r, T_keV) / total_fusion_power_density(
        r_peak, T_keV
    )


# ============================================================
# Reporting
# ============================================================
if __name__ == "__main__":
    print("=" * 75)
    print("D-He3 mix optimization (Bosch-Hale + relativistic brem)")
    print("=" * 75)

    T = 70
    print(f"\nSweep at T = {T} keV (n_e = 1e20 m^-3):")
    print(
        f"  <sv>_DHe3 = {sigv_dhe3(T):.3e}  <sv>_DD = {sigv_dd_total(T):.3e}  "
        f"ratio = {sigv_dd_total(T) / sigv_dhe3(T):.3f}"
    )
    print()
    print(f"  {'r':>5} {'f_DD':>7} {'f_n':>7} {'f_brem':>9} {'f_ext':>9} {'sum_NB':>9}")
    print("  " + "-" * 55)
    for r in [0.2, 0.3, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]:
        f = fractions(r, T)
        print(
            f"  {r:>5.2f} {f['f_DD'] * 100:>6.1f}% {f['f_n'] * 100:>6.1f}% "
            f"{f['f_brem'] * 100:>8.1f}% {f['f_ext'] * 100:>8.1f}% "
            f"{(f['f_n'] + f['f_brem']) * 100:>8.1f}%"
        )

    print("\nLoss-minimizing mix (minimum N + B) vs operating temperature:")
    print(
        f"  {'T_keV':>6} {'r_opt':>7} {'f_n%':>7} {'f_brem%':>9} "
        f"{'f_ext%':>8} {'sum%':>7}"
    )
    print("  " + "-" * 50)
    for T in [50, 70, 100, 150, 200, 300]:
        r, f = find_optimum(T)
        print(
            f"  {T:>6} {r:>7.2f} {f['f_n'] * 100:>7.1f} {f['f_brem'] * 100:>9.1f} "
            f"{f['f_ext'] * 100:>8.1f} {(f['f_n'] + f['f_brem']) * 100:>7.1f}"
        )

    print("\nReactivity-optimal mix (max total fusion power at fixed plasma pressure):")
    print(
        f"  {'T_keV':>6} {'r_opt':>7} {'n_D/n_He3':>10} {'f_n%':>7} "
        f"{'f_brem%':>9} {'f_ext%':>8}"
    )
    print("  " + "-" * 55)
    for T in [50, 70, 100, 150, 200, 300]:
        r, f = find_max_fusion_power(T)
        print(
            f"  {T:>6} {r:>7.2f} {1 / r:>10.2f} {f['f_n'] * 100:>7.1f} "
            f"{f['f_brem'] * 100:>9.1f} {f['f_ext'] * 100:>8.1f}"
        )

    print(
        "\nRelative total fusion power vs reactivity-optimal mix (at fixed pressure):"
    )
    print(f"  {'r':>5} {'label':<22} {'P_fus/P_peak':>14}")
    print("  " + "-" * 50)
    T = 70
    for r, label in [
        (find_max_fusion_power(T)[0], "reactivity-peak"),
        (find_optimum(T)[0], "DEC-optimal"),
        (1.0, "50/50 (textbook)"),
        (0.5, "D-rich (1:2 He3:D)"),
        (2.0, "3He-rich (2:1 He3:D)"),
    ]:
        rel = relative_reactivity(r, T)
        print(f"  {r:>5.2f} {label:<22} {rel * 100:>12.1f}%")

    print("\n" + "=" * 75)
    print("Comparison to literature consensus")
    print("=" * 75)
    print()
    print(
        f"{'Operating point':<28} {'T':>5} {'r':>5} {'f_DD':>7} "
        f"{'f_n':>7} {'f_brem':>9} {'f_ext':>8}"
    )
    print("-" * 75)
    cases = [
        ("Wesson 50/50 @ 70 keV", 70, 1.0),
        ("Wesson 50/50 @ 100 keV", 100, 1.0),
        ("DEC-optimal @ 70 keV", 70, 0.71),
        ("DEC-optimal @ 100 keV", 100, 0.80),
        ("D-rich (1:2 He3:D)", 70, 0.5),
        ("3He-rich (2:1 He3:D)", 70, 2.0),
    ]
    for label, T, r in cases:
        f = fractions(r, T)
        print(
            f"{label:<28} {T:>5} {r:>5.2f} {f['f_DD'] * 100:>6.1f}% "
            f"{f['f_n'] * 100:>6.1f}% {f['f_brem'] * 100:>8.1f}% "
            f"{f['f_ext'] * 100:>7.1f}%"
        )

    print()
    print("Literature reference values for D-He3:")
    print("  Wesson 'Tokamaks' (Fig 1.5.x):    P_brem/P_fus ~ 20% at 50/50, T~100 keV")
    print("  Santarius & Kulcinski:            P_brem/P_fus ~ 25% (used in 1costingfe)")
    print("  Rider 1995 (with extra losses):   P_brem/P_fus ~ 30%")

    print("\n" + "=" * 75)
    print("Self-consistent 1costingfe defaults (compute both params at chosen point)")
    print("=" * 75)
    for label, T, r in [
        ("Wesson 50/50 @ 70 keV", 70, 1.0),
        ("Wesson 50/50 @ 100 keV", 100, 1.0),
        ("DEC-optimal @ 100 keV", 100, 0.80),
    ]:
        f = fractions(r, T)
        print(f"\n{label}  (r={r}, T={T} keV):")
        print(f"  dhe3_dd_frac     = {f['f_DD']:.3f}   (current default 0.070)")
        print(f"  f_rad_fus_dhe3   = {f['f_brem']:.3f}   (current default 0.250)")
        print(
            f"  -> {f['f_n'] * 100:.1f}% N, "
            f"{f['f_brem'] * 100:.1f}% B, "
            f"{f['f_ext'] * 100:.1f}% E"
        )
