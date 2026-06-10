"""Fusion reactivity fits and fuel-mix algebra (concept-agnostic).

Thermal reactivities <sigma*v>(T_i) for the four supported fuels, plus the
quasineutrality dilution and effective-charge algebra they imply. All
functions are pure and JAX-differentiable, so any concept layer (tokamak
today; mirror/FRC sizing later) can import them.

Sources:
- D-T, D-D (both branches), D-He3: Bosch & Hale, Nucl. Fusion 32 (1992) 611.
  Identical coefficients to the float64 verification in
  examples/dhe3_mix_optimization.py, which stays independent on purpose.
- p-B11: Nevins & Swain, Nucl. Fusion 40 (2000) 865, high-temperature branch
  (valid 50-500 keV; below ~50 keV the 148 keV resonance contribution is
  underestimated, acceptable because the p-B11 operating bracket starts at
  50 keV). Coefficients as tabulated by Tentori & Belloni, Nucl. Fusion 63
  (2023). Tentori & Belloni's own updated fit and Putvinski et al., Nucl.
  Fusion 59 (2019) 076018 give higher reactivity (up to +50% at the tail)
  and are the optimistic alternatives; Nevins-Swain is the default.
"""

import jax.numpy as jnp


def _bosch_hale(T_keV, BG, mrc2, C1, C2, C3, C4, C5, C6, C7):
    """Bosch-Hale reactivity parameterization. T in keV, returns cm^3/s
    (callers convert). BG is the Gamow constant [keV^0.5], mrc2 the reduced
    mass energy [keV]."""
    theta = T_keV / (
        1.0
        - T_keV
        * (C2 + T_keV * (C4 + T_keV * C6))
        / (1.0 + T_keV * (C3 + T_keV * (C5 + T_keV * C7)))
    )
    xi = (BG**2 / (4.0 * theta)) ** (1.0 / 3.0)
    return C1 * theta * jnp.sqrt(xi / (mrc2 * T_keV**3)) * jnp.exp(-3.0 * xi)


def sigv_dt(T_keV):
    """D + T -> n + 4He reactivity [m^3/s], Bosch-Hale, valid 0.2-100 keV."""
    return (
        _bosch_hale(
            T_keV,
            34.3827,
            1124656.0,
            1.17302e-9,
            1.51361e-2,
            7.51886e-2,
            4.60643e-3,
            1.35000e-2,
            -1.06750e-4,
            1.36600e-5,
        )
        * 1e-6
    )


def sigv_dhe3(T_keV):
    """D + 3He -> p + 4He reactivity [m^3/s], Bosch-Hale, valid 0.5-190 keV."""
    return (
        _bosch_hale(
            T_keV,
            68.7508,
            1124572.0,
            5.51036e-10,
            6.41918e-3,
            -2.02896e-3,
            -1.91080e-5,
            1.35776e-4,
            0.0,
            0.0,
        )
        * 1e-6
    )


def sigv_dd_n(T_keV):
    """D + D -> n + 3He branch reactivity [m^3/s], Bosch-Hale."""
    return (
        _bosch_hale(
            T_keV,
            31.3970,
            937814.0,
            5.43360e-12,
            5.85778e-3,
            7.68222e-3,
            0.0,
            -2.96400e-6,
            0.0,
            0.0,
        )
        * 1e-6
    )


def sigv_dd_p(T_keV):
    """D + D -> p + T branch reactivity [m^3/s], Bosch-Hale."""
    return (
        _bosch_hale(
            T_keV,
            31.3970,
            937814.0,
            5.65718e-12,
            3.41267e-3,
            1.99167e-3,
            0.0,
            1.05060e-5,
            0.0,
            0.0,
        )
        * 1e-6
    )


# p-11B Gamow energy E_G = B_G^2 [keV] and reduced mass energy [keV]
# (Tentori & Belloni 2023, after Nevins & Swain 2000).
_PB11_EG = 22589.0
_PB11_MRC2 = 859526.0


def sigv_pb11(T_keV):
    """p + 11B -> 3 alpha reactivity [m^3/s], Nevins-Swain HT branch.

    Valid 50-500 keV. The C1 coefficient is in keV m^3/s (no cm^3 -> m^3
    conversion, unlike the Bosch-Hale fits above).
    """
    return _bosch_hale(
        T_keV,
        _PB11_EG**0.5,
        _PB11_MRC2,
        4.4467e-14,
        -5.9357e-2,
        2.0165e-1,
        1.0404e-3,
        2.7621e-3,
        -9.1653e-6,
        9.8305e-7,
    )
