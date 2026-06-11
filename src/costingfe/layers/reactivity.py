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

import jax.lax
import jax.numpy as jnp

from costingfe.layers.physics import MEV_TO_JOULES, event_energies
from costingfe.types import Fuel


def _density_1e10(n_e):
    """Density scaled by 1e-10, behind an optimization barrier.

    The barrier keeps XLA from reassociating the product back into
    n_e * n_e (~1e40, inf in float32) by pulling the 1e-10 constants out;
    without it the DD/DHE3/PB11 rates silently become inf (or zero,
    depending on how the constants gather) under jit. Differentiable and
    an identity in eager mode.
    """
    return jax.lax.optimization_barrier(n_e * 1e-10)


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


# ---------------------------------------------------------------------------
# Quasineutrality mix algebra: n_e = sum(Z_j n_j)
# ---------------------------------------------------------------------------
def n_i_over_n_e(fuel, dhe3_fuel_ratio, pb11_fuel_ratio):
    """Total fuel-ion to electron density ratio. Plain arithmetic (works on
    floats and JAX tracers alike)."""
    if fuel == Fuel.DT or fuel == Fuel.DD:
        return 1.0
    if fuel == Fuel.DHE3:
        r = dhe3_fuel_ratio
        return (1.0 + r) / (1.0 + 2.0 * r)
    if fuel == Fuel.PB11:
        r = pb11_fuel_ratio
        return (1.0 + r) / (1.0 + 5.0 * r)
    raise ValueError(f"Unknown fuel type: {fuel}")


def z_eff_fuel(fuel, dhe3_fuel_ratio, pb11_fuel_ratio):
    """Fuel-ion contribution to Z_eff = sum(n_j Z_j^2)/n_e (fully stripped)."""
    if fuel == Fuel.DT or fuel == Fuel.DD:
        return 1.0
    if fuel == Fuel.DHE3:
        r = dhe3_fuel_ratio
        return (1.0 + 4.0 * r) / (1.0 + 2.0 * r)
    if fuel == Fuel.PB11:
        r = pb11_fuel_ratio
        return (1.0 + 25.0 * r) / (1.0 + 5.0 * r)
    raise ValueError(f"Unknown fuel type: {fuel}")


# ---------------------------------------------------------------------------
# Rate -> fusion power dispatch
# ---------------------------------------------------------------------------


def fusion_power(
    fuel,
    n_e,
    T_i,
    V_plasma,
    *,
    dhe3_fuel_ratio,
    pb11_fuel_ratio,
    dhe3_dd_frac_pin,
    dd_f_T,
    dd_f_He3,
    dhe3_f_T,
    dhe3_f_He3,
    pb11_f_alpha_n,
    pb11_f_p_n,
):
    """Fusion power [MW] and effective D-D side-channel fraction for a
    thermal plasma at ion temperature T_i [keV] and volume V_plasma [m^3].

    Reactant densities follow quasineutrality with the fuel-mix ratio knobs.
    Per-event energies come from physics.event_energies, so the result is
    consistent with ash_neutron_split's partition. dhe3_dd_frac_pin, when not
    None, overrides the rate-derived side-channel fraction (and is what the
    partition will see). Returns (p_fus_MW, dhe3_dd_frac_eff); the fraction
    is 0.0 for fuels without a side channel.

    float32 safety in the DD/DHE3/PB11 branches: XLA's algebraic simplifier
    reassociates multiplicative chains freely, which can either gather the
    tiny scaling constants ((1e-20)^2 * 1e-6 ~ 1e-47 folds to zero) or pull
    them out and form n_e * n_e ~ 1e40 = inf; both silently corrupt the rate
    under jit. Defenses: density enters via _density_1e10 (scaled once,
    behind an optimization barrier so XLA cannot reassociate across it),
    reactivities are used in 1e-22 units so the derived-fraction quotient is
    O(1) (whose VJP would otherwise square a ~1e-23 denominator into
    underflow and emit NaN gradients), and the residual exponent folds into
    one benign constant per branch. The DT branch keeps the legacy
    multiplication order bit-pinned to compute_fusion_power.
    """

    if fuel == Fuel.DT:
        # Bit-identical to compute_fusion_power in tokamak.py:
        #   rate = n_e * sv; return 0.25 * rate * n_e * E_fus_J * V * 1e-6
        E_total, _ = event_energies(
            fuel,
            dd_f_T=dd_f_T,
            dd_f_He3=dd_f_He3,
            dhe3_dd_frac=0.0,
            dhe3_f_T=dhe3_f_T,
            dhe3_f_He3=dhe3_f_He3,
            pb11_f_alpha_n=pb11_f_alpha_n,
            pb11_f_p_n=pb11_f_p_n,
        )
        E_fus_J = E_total * MEV_TO_JOULES
        sv = sigv_dt(T_i)
        rate = n_e * sv
        return 0.25 * rate * n_e * E_fus_J * V_plasma * 1e-6, 0.0

    if fuel == Fuel.DD:
        E_total, _ = event_energies(
            fuel,
            dd_f_T=dd_f_T,
            dd_f_He3=dd_f_He3,
            dhe3_dd_frac=0.0,
            dhe3_f_T=dhe3_f_T,
            dhe3_f_He3=dhe3_f_He3,
            pb11_f_alpha_n=pb11_f_alpha_n,
            pb11_f_p_n=pb11_f_p_n,
        )
        # m10^2 carries n^2 * 1e-20, sv22 the reactivity in 1e-22 units; the
        # residual exponent 1e20 * 1e-22 * 1e-6 = 1e-8 folds into the constant.
        m10_D = _density_1e10(n_e)
        sv22 = (sigv_dd_n(T_i) + sigv_dd_p(T_i)) * 1e22
        const = 0.5 * E_total * (MEV_TO_JOULES * 1e-8) * V_plasma
        return m10_D * m10_D * sv22 * const, 0.0

    if fuel == Fuel.DHE3:
        r = dhe3_fuel_ratio
        sv3_22 = sigv_dhe3(T_i) * 1e22
        svt_22 = (sigv_dd_n(T_i) + sigv_dd_p(T_i)) * 1e22
        # Rate ratio R_dd/R_dhe3: densities cancel (n_D^2 vs r*n_D^2), leaving
        # a pure O(1) reactivity ratio whose quotient VJP cannot underflow.
        x = 0.5 * svt_22 / (r * sv3_22)
        derived = x / (1.0 + x)
        frac = derived if dhe3_dd_frac_pin is None else dhe3_dd_frac_pin
        E_total, _ = event_energies(
            fuel,
            dd_f_T=dd_f_T,
            dd_f_He3=dd_f_He3,
            dhe3_dd_frac=frac,
            dhe3_f_T=dhe3_f_T,
            dhe3_f_He3=dhe3_f_He3,
            pb11_f_alpha_n=pb11_f_alpha_n,
            pb11_f_p_n=pb11_f_p_n,
        )
        # Total rate = R_dhe3*(1+x), physical regardless of any pin (the pin
        # moves the partition via E_total, not the reaction rates).
        m10 = _density_1e10(n_e)
        mix = r / ((1.0 + 2.0 * r) ** 2)
        const = (mix * (MEV_TO_JOULES * 1e-8)) * E_total * V_plasma
        return m10 * m10 * sv3_22 * (1.0 + x) * const, frac

    if fuel == Fuel.PB11:
        r = pb11_fuel_ratio
        E_total, _ = event_energies(
            fuel,
            dd_f_T=dd_f_T,
            dd_f_He3=dd_f_He3,
            dhe3_dd_frac=0.0,
            dhe3_f_T=dhe3_f_T,
            dhe3_f_He3=dhe3_f_He3,
            pb11_f_alpha_n=pb11_f_alpha_n,
            pb11_f_p_n=pb11_f_p_n,
        )
        m10 = _density_1e10(n_e)
        sv22 = sigv_pb11(T_i) * 1e22
        mix = r / ((1.0 + 5.0 * r) ** 2)
        const = (mix * (MEV_TO_JOULES * 1e-8)) * E_total * V_plasma
        return m10 * m10 * sv22 * const, 0.0

    raise ValueError(f"Unknown fuel type: {fuel}")
