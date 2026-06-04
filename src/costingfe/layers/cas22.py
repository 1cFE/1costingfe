"""CAS22: Reactor Plant Equipment sub-accounts.

Hybrid volume + thermal-intensity costing for geometry-dependent items:
  cost = unit_cost * volume * (p_th / p_th_ref)^alpha

Volume captures reactor size (geometry dimensions). Thermal intensity
captures the fact that components handling more power need better
cooling, thicker walls, and higher-grade materials per unit volume.

Power-scaled for remaining items (coils, heating, power supplies, divertor).
Fuel-dependent config for blanket, isotope sep, fuel handling.

All costs in M$. Source: pyFECONs costing/calculations/cas22/
"""

import math

import jax
import jax.numpy as jnp

from costingfe.defaults import CostingConstants
from costingfe.types import (
    BlanketFill,
    BlanketForm,
    CoilMaterial,
    ConfinementConcept,
    ConfinementFamily,
    Fuel,
    LaserDriverType,
    PulsedConversion,
)

# Concept-dependent coil defaults. Markups are engineering build-ups
# (conductor is ~10-15% of finished magnet cost; per-concept winding,
# structure, and quench-protection complexity), validated against CFS
# SPARC REBCO usage. See docs/account_justification/CAS22_reactor_components.md
# markup: manufacturing complexity multiplier over raw conductor cost
# path_factor: extra coil path length for 3D geometries (stellarator)
# n_coils: number of discrete coils — only used for mirror (G = n_coils * 4*pi);
#         ignored by tokamak/stellarator branches whose G is empirical total-system
# None → no confinement magnets (IFE drivers, magnet-free pulsed concepts)
_COIL_DEFAULTS = {
    # MFE / electrostatic — full confinement magnets
    # Tokamak/stellarator use the bilinear toroidal conductor model
    # (total_kAm = G * B * R0 * r_coil), so markup is a pure fabrication
    # multiplier, NOT a blended geometry+complexity factor. markup=3.09 is
    # calibrated so the SPARC-class reference tokamak (B=12T, R0=3.0, r_coil=
    # vessel_or=2.95m) coil system = $516M, matching the prior r_bore^2
    # calibration at its reference point. The stellarator's 5.87 is 1.9x the
    # tokamak value, the NCSX modular-coil production cost overrun (90%,
    # Neilson 2010 PPPL-4455) — the documented penalty for non-planar 3D coil
    # fabrication. The longer 3D winding path is handled separately by
    # path_factor=2 in G, so the 1.9x is fabrication complexity only.
    ConfinementConcept.TOKAMAK: {"markup": 3.09, "path_factor": 1.0, "n_coils": 0},
    ConfinementConcept.STELLARATOR: {"markup": 5.87, "path_factor": 2.0, "n_coils": 0},
    # Mirror n_coils calibrated to Realta HAMMIR-class tandem mirror:
    # 4 end-plug HTS coils (2 per end, Hammer evolution) + ~6 LTS central-cell
    # solenoid coils discretizing the 50 m central cell. Simple-mirror devices
    # (WHAM/BEAM/Anvil) would use n_coils ≈ 4.
    ConfinementConcept.MIRROR: {"markup": 2.5, "path_factor": 1.0, "n_coils": 10},
    # DIPOLE: the DIPOLE branch in C220103 does NOT consume these defaults
    # (markup / path_factor / n_coils). It computes the floating coil's kA*m
    # from b_center / r_bore directly, then approximates the single external
    # stationary lift coil as `stationary_lift_coil_fraction * floating_with_
    # markup` (default 0.10). This dict entry is kept so concept-level table
    # lookups (e.g. test_cas22_n_coils, downstream tooling) don't KeyError on
    # DIPOLE; the values are not used by the cost calculation.
    ConfinementConcept.DIPOLE: {"markup": 3.0, "path_factor": 1.0, "n_coils": 1},
    # Steady-state FRC (beam-driven, e.g. TAE; RMF-driven PFRC) is a linear,
    # open-field-line device like a mirror: external formation + mirror/end-plug
    # coils discretized as independent solenoids (G = n_coils * 4*pi). The
    # self-organized internal FRC field carries no coil cost. n_coils counts only
    # the external coil set; resistive-magnet markup is lower than HTS mirrors.
    ConfinementConcept.STEADY_FRC: {"markup": 2.0, "path_factor": 1.0, "n_coils": 4},
    ConfinementConcept.PULSED_FRC: {"markup": 1.5, "path_factor": 1.0, "n_coils": 0},
    ConfinementConcept.THETA_PINCH: {"markup": 1.5, "path_factor": 1.0, "n_coils": 0},
    ConfinementConcept.ORBITRON: {"markup": 1.5, "path_factor": 1.0, "n_coils": 0},
    ConfinementConcept.POLYWELL: {"markup": 2.0, "path_factor": 1.0, "n_coils": 0},
    # MIF — no plant-scale confinement magnets. MagLIF compresses
    # with a Z-pinch liner; MAG_TARGET (General Fusion, NearStar) compresses
    # magnetized plasma mechanically/kinetically; PLASMA_JET merges plasma guns.
    # The MFE conductor-scaling model (tokamak geometry factor + multi-tesla
    # field) does not apply. Any small seed/guide coil is opt-in per concept via
    # cost_overrides on C220103.
    ConfinementConcept.MAG_TARGET: None,
    ConfinementConcept.PLASMA_JET: None,
    ConfinementConcept.MAGLIF: None,
    # IFE / magnet-free pulsed — no confinement magnets
    ConfinementConcept.LASER_IFE: None,
    ConfinementConcept.ZPINCH: None,
    ConfinementConcept.HEAVY_ION: None,
    ConfinementConcept.DENSE_PLASMA_FOCUS: None,
    ConfinementConcept.STAGED_ZPINCH: None,
}

_MU0 = 4 * math.pi * 1e-7  # Vacuum permeability (T·m/A)
_RHO_CU = 8960.0  # Copper density (kg/m³), for resistive-coil mass build-up


def _compute_geometry_factor(
    concept: ConfinementConcept,
    path_factor: float,
    n_coils: int,
) -> float:
    """Geometry factor G for conductor quantity scaling.

    total_kAm = G * B_center * R^2 / (mu_0 * 1000)

    Derived from Biot-Savart for a thin circular loop:
      B_center = mu_0 * N*I / (2 R)  ->  N*I = 2 * B_center * R / mu_0
      kA*m     = N*I * (2 pi R) / 1000 = 4 pi * B_center * R^2 / (mu_0 * 1000)
    so a single circular loop has G = 4 pi. Multi-coil and 3D-path systems
    multiply by per-concept factors below. The B input is the field at the
    center of the loop (axis), NOT the peak field on the conductor.

    Tokamak: G = 4pi^2 — empirical total-system (TF+CS+PF) scaling.
    Mirror:  G = n_coils * 4*pi — sum over independent solenoid coils.
    Steady FRC: G = n_coils * 4*pi — linear device, same as mirror.
    Dipole:  G = n_coils * 4*pi — sum over the stationary levitation coils
             (the levitated coil is costed separately, not via this factor).
    Stellarator: G = 4*pi^2 * path_factor — 3D coil paths ~2x longer.
    """
    if concept in (
        ConfinementConcept.MIRROR,
        ConfinementConcept.STEADY_FRC,
        ConfinementConcept.DIPOLE,
    ):
        return n_coils * 4 * math.pi
    elif concept == ConfinementConcept.STELLARATOR:
        return 4 * math.pi**2 * path_factor
    else:  # tokamak (default)
        return 4 * math.pi**2


def cas22_reactor_plant_equipment(
    cc: CostingConstants,
    p_net: float,
    p_th: float,
    p_et: float,
    p_fus: float,
    p_cryo: float,
    n_mod: int,
    fuel: Fuel,
    noak: bool,
    blanket_vol: float,
    shield_vol: float,
    structure_vol: float,
    vessel_vol: float,
    family: ConfinementFamily,
    concept: ConfinementConcept,
    b_center: float,
    r_bore: float,
    R0: float,
    r_coil: float,
    coil_material: CoilMaterial,
    blanket_form: BlanketForm,
    p_nbi: float,
    p_icrf: float,
    p_ecrh: float,
    p_lhcd: float,
    p_driver: float,
    f_dec: float,
    p_dee: float,
    burn_fraction: float,
    vac_op_pressure_pa: float,
    # True iff the concept fabricates a consumed target/liner each shot
    # (drives the C220108 target factory).
    manufactured_target: bool,
    e_driver_mj: float = 0.0,  # Per-pulse driver energy [MJ]; $/J laser/accel basis
    e_preheat_mj: float = 0.0,  # Per-pulse preheat laser energy [MJ]; 0 = no preheat
    laser_driver_type=None,  # LaserDriverType for LASER_IFE; selects C220104 $/MJ
    # Pulsed DEC parameters
    pulsed_conversion=None,
    e_stored_mj: float = 0.0,
    q_sci: float = 0.0,
    f_ch: float = 0.0,
    eta_dec: float = 0.0,
    n_coils: int | None = None,
    lev_coil_markup: float | None = None,
    lev_coil_cryostat_cost: float | None = None,
    stationary_lift_coil_fraction: float = 0.10,
    blanket_fill: BlanketFill | None = None,
) -> dict[str, float]:
    """Compute all CAS22 sub-accounts. Returns dict of account_code -> M$.

    Volume-based accounts use hybrid formula:
      cost = unit_cost * volume * (power / power_ref)^alpha
    This captures both reactor size (volume) and thermal intensity (power).

    Sub-line convention: where one account aggregates components with distinct
    cost bases, informational sub-lines keyed `<CODE>_<component>` are emitted
    alongside the canonical total (e.g. C220106_vessel, C220106_pump). They are
    excluded from all aggregation and exist for visibility / sensitivity only.
    """
    # Reference power levels at calibration geometry (1 GWe DT tokamak)
    P_TH_REF = 2500.0  # MW thermal
    P_ET_REF = 1100.0  # MW gross electric

    # -----------------------------------------------------------------------
    # 220101: First Wall + Blanket + Neutron Multiplier
    # DT: breeding blanket (TBR>1.05) + neutron multiplier (RAFM steel +
    #   PbLi/Li breeder + Be multiplier + W FW armor). Complex assembly.
    # DD: energy-capture blanket (no breeding). Simpler RAFM steel + coolant.
    # DHe3/pB11: minimal (X-ray shielding only)
    # See docs/account_justification/CAS22_reactor_components.md
    # -----------------------------------------------------------------------
    blanket_unit = {
        Fuel.DT: cc.blanket_unit_cost_dt,
        Fuel.DD: cc.blanket_unit_cost_dd,
        Fuel.DHE3: cc.blanket_unit_cost_dhe3,
        Fuel.PB11: cc.blanket_unit_cost_pb11,
    }
    # Chemistry override: the fuel-keyed table above is calibrated against PbLi
    # liquid-metal blankets (density ~9.4 t/m^3, flow loop + MHD ducts + online
    # tritium extraction). Solid Li2O (~2 t/m^3, no flow loop) is ~3-5x cheaper
    # per m^3. Branch on blanket_fill so DIPOLE-class machines using Li2O do not
    # pay the PbLi premium volumetrically.
    if blanket_fill == BlanketFill.LI2O:
        unit = cc.blanket_unit_cost_li2o
    else:
        unit = blanket_unit[fuel]
    # TODO: incorporate wall_material cost multiplier into C220101
    # (W tiles vs flowing Li systems vs SiC composites have very different
    # fabrication costs — requires dedicated research)
    c220101 = (
        unit * blanket_form.structure_factor * blanket_vol * (p_th / P_TH_REF) ** 0.6
    )

    # -----------------------------------------------------------------------
    # 220102: Shield (HT + LT + Bioshield)
    # Full shield for DT (14.1 MeV), reduced for lower-neutron fuels.
    # See docs/account_justification/CAS22_reactor_components.md
    # -----------------------------------------------------------------------
    shield_scale = {
        Fuel.DT: 1.0,  # Heavy shield (14.1 MeV neutrons)
        Fuel.DD: 0.7,  # Mixed (2.45 MeV neutrons)
        Fuel.DHE3: 0.3,  # Light (~5% neutron fraction)
        Fuel.PB11: 0.1,  # Minimal (aneutronic)
    }
    c220102 = (
        cc.shield_unit_cost * shield_vol * shield_scale[fuel] * (p_th / P_TH_REF) ** 0.6
    )

    # -----------------------------------------------------------------------
    # 220103: Coils. Conductor quantity (ampere-meters) by device topology:
    #   - TOROIDAL (tokamak, stellarator): total_kAm = G * b_center * R0 *
    #     r_coil / mu0. This is BILINEAR, not r^2: the toroidal field needs
    #     ampere-turns ~ B*R0 (Ampere's law around the torus), and the
    #     conductor length per turn ~ the coil bore r_coil. R0 is the major
    #     radius; r_coil is the coil-bore radius = vessel_or from the radial
    #     build (the TF/modular coils sit just outside the vessel). Using a
    #     single r^2 (the prior model, calibrated at SPARC's R0=1.85m) made
    #     coil cost grow as R0^2 and exploded for large machines; real toroidal
    #     conductor grows ~linearly in R0.
    #   - LINEAR/LOOP (mirror, FRC, dipole, pulsed): total_kAm = G * b_center *
    #     r_bore^2 / mu0. For a real solenoid/ring loop, B = mu0*N*I/(2R) and
    #     length = 2*pi*R, so r^2 is correct; r_bore is the loop radius.
    # b_center is the field at the geometric center of the loop (axis), NOT the
    # peak field on the conductor; peak-on-conductor is a factor of ~2-4 higher
    # for high-field SC, but that ratio does not enter the ampere-meter quantity
    # (it would only matter for a J_c-vs-field conductor derating or a B_max^2
    # structure term, neither of which is modeled here). See
    # _compute_geometry_factor docstring and
    # docs/account_justification/CAS22_reactor_components.md.
    #   - Superconducting (HTS/LTS): cost = total_kAm * $/kAm * markup. The
    #     expensive conductor dominates; REBCO $50/kAm (NOAK), markup captures
    #     winding, quench protection, cryostat, testing.
    #   - Resistive (copper): the conductor is cheap per kAm, so cost is set by
    #     the bulk copper MASS plus structure and fabrication, not by $/kAm.
    #     Re-price the same ampere-meters by mass: m_Cu = (rho_Cu / J) *
    #     ampere_meters, add steel support, apply fabrication markups. This is
    #     the right basis for low-field FRC/linear copper coils, where the
    #     $/kAm path collapses to a near-zero, unphysical number.
    # -----------------------------------------------------------------------
    if concept == ConfinementConcept.DIPOLE:
        # ---------------------------------------------------------------
        # Levitated dipole — two distinct coil populations:
        #   1) the FLOATING field coil (the "core magnet") — a single HTS
        #      ring carrying persistent current at the design point's
        #      b_center / r_bore. Its kA*m follows Biot-Savart with G=4*pi
        #      and gets the float / no-access lev_coil_markup, plus the
        #      flat lev_coil_cryostat_cost (integral neon-slush cryoplant
        #      + flux pump + levitation control).
        #   2) ONE external stationary lift coil — sits outside the inner
        #      VV (Simpson 2026, arXiv:2602.20564, §2.2 Reactor A: the
        #      floating coil is held up by external lift coils whose field
        #      is set by force balance against the coil's gravity, NOT by
        #      plasma confinement). The lift coil operates at a much lower
        #      center field (only force balance is required) at typically
        #      larger R (sits outside the chamber), so its kA*m and
        #      conductor budget are much smaller than the floating coil.
        #      It also does NOT carry the float / no-access engineering
        #      markup (it's a standard external SC coil with full access
        #      for assembly and maintenance).
        # We approximate the stationary lift coil's full installed cost as
        # a small fraction of the floating coil's with-markup cost. The
        # default (stationary_lift_coil_fraction = 0.10) is a first-cut
        # estimate; a more rigorous version would parameterize the lift
        # coil's own b_center_stationary / r_bore_stationary.
        # ---------------------------------------------------------------
        lev_kAm = 4 * math.pi * b_center * r_bore**2 / (_MU0 * 1000)
        lev_conductor = lev_kAm * coil_material.default_cost_per_kAm / 1e6
        floating_with_markup = lev_conductor * lev_coil_markup
        stationary_lift_cost = stationary_lift_coil_fraction * floating_with_markup
        c220103 = floating_with_markup + stationary_lift_cost + lev_coil_cryostat_cost
    else:
        defaults = _COIL_DEFAULTS.get(concept)
        if defaults is None:
            # No confinement magnets (IFE drivers, magnet-free pulsed)
            c220103 = 0.0
        else:
            coil_markup = defaults["markup"]
            path_factor = defaults["path_factor"]
            # Honor per-call override; fall back to concept default
            n_coils_eff = n_coils if n_coils is not None else defaults["n_coils"]
            if n_coils_eff < 0:
                raise ValueError(f"n_coils must be >= 0, got {n_coils_eff}")
            G = _compute_geometry_factor(concept, path_factor, n_coils_eff)
            if concept in (
                ConfinementConcept.TOKAMAK,
                ConfinementConcept.STELLARATOR,
            ):
                # Toroidal: bilinear in major radius and coil-bore radius.
                # Guard against a silent zero-cost coil from a missing R0/r_coil,
                # but only on concrete values: under JAX tracing (sensitivity /
                # uncertainty) these are positive-valued tracers and a Python
                # comparison would raise TracerBoolConversionError.
                _concrete = not (
                    isinstance(R0, jax.core.Tracer)
                    or isinstance(r_coil, jax.core.Tracer)
                )
                if _concrete and (R0 <= 0 or r_coil <= 0):
                    raise ValueError(
                        f"{concept.value} coil cost needs R0>0 and r_coil>0, "
                        f"got R0={R0}, r_coil={r_coil}"
                    )
                total_kAm = G * b_center * R0 * r_coil / (_MU0 * 1000)
            else:
                # Linear/loop device: r^2 is correct for a real solenoid/ring.
                total_kAm = G * b_center * r_bore**2 / (_MU0 * 1000)
            if coil_material == CoilMaterial.COPPER:
                ampere_meters = total_kAm * 1000.0
                j_a_m2 = cc.coil_cu_current_density_a_mm2 * 1e6
                m_cu = (_RHO_CU / j_a_m2) * ampere_meters
                m_steel = cc.coil_struct_fraction * m_cu
                c220103 = (
                    m_cu * cc.coil_cu_price_per_kg * cc.coil_cu_fab_markup
                    + m_steel * cc.coil_steel_price_per_kg * cc.coil_steel_fab_markup
                ) / 1e6
            else:
                conductor_cost = total_kAm * coil_material.default_cost_per_kAm / 1e6
                c220103 = conductor_cost * coil_markup

    # -----------------------------------------------------------------------
    # 220104: Supplementary Heating (MFE) or Primary Driver (pulsed)
    # MFE: per-MW linear costs calibrated to ITER procurement (FOAK→NOAK)
    # Pulsed: concept-specific driver capital (laser, accelerator, mechanical)
    # Concepts whose driver is purely electrical use C220107 instead.
    # See docs/account_justification/CAS22_reactor_components.md
    # -----------------------------------------------------------------------
    if family == ConfinementFamily.STEADY_STATE:
        c220104 = (
            cc.heating_nbi_per_mw * p_nbi
            + cc.heating_icrf_per_mw * p_icrf
            + cc.heating_ecrh_per_mw * p_ecrh
            + cc.heating_lhcd_per_mw * p_lhcd
        )
    else:
        # Lasers, accelerators, and electromagnetic guns: capital is set by
        # per-pulse energy (aperture/diode count, ring charge, coaxial-gun size and
        # peak current), not rep rate, so cost on $/J of E_driver. STAGED_ZPINCH is
        # the sheared-flow coaxial gun + gas injection; its cap bank is in C220107.
        _DRIVER_COST_PER_MJ = {
            ConfinementConcept.LASER_IFE: cc.driver_laser_per_mj,
            ConfinementConcept.HEAVY_ION: cc.driver_heavy_ion_per_mj,
            ConfinementConcept.PLASMA_JET: cc.driver_plasma_jet_per_mj,
            ConfinementConcept.STAGED_ZPINCH: cc.driver_staged_zpinch_per_mj,
        }
        # Pneumatic/mechanical injectors accelerate mass each shot, so average
        # power (throughput) is the defensible basis.
        _DRIVER_COST_PER_MW = {
            ConfinementConcept.MAG_TARGET: cc.driver_mag_target_per_mw,
        }
        # MAGLIF is in neither map: its main driver is electrical (C220107). Only
        # laser preheat lands here, costed per joule of preheat pulse energy, so a
        # magnetized-compression concept with e_preheat_mj=0 pays nothing.
        # For LASER_IFE the per-MJ coefficient is set by the driver architecture
        # (DPSSL/KrF/Nd:Glass). The MagLIF preheat line stays DPSSL-class.
        _LASER_DRIVER_PER_MJ = {
            LaserDriverType.DPSSL: cc.driver_laser_per_mj,
            LaserDriverType.KRF: cc.driver_krf_per_mj,
            LaserDriverType.NDGLASS: cc.driver_ndglass_per_mj,
        }
        if concept == ConfinementConcept.LASER_IFE and laser_driver_type is not None:
            drv_per_mj = _LASER_DRIVER_PER_MJ[laser_driver_type]
        else:
            drv_per_mj = _DRIVER_COST_PER_MJ.get(concept, 0.0)
        c220104 = (
            drv_per_mj * e_driver_mj
            + _DRIVER_COST_PER_MW.get(concept, 0.0) * p_driver
            + cc.laser_preheat_per_mj * e_preheat_mj
        )

    # -----------------------------------------------------------------------
    # 220105: Primary Structure — gravity supports, thermal shields,
    # inter-coil structure, machine base.
    # See docs/account_justification/CAS22_reactor_components.md
    # -----------------------------------------------------------------------
    c220105 = cc.structure_unit_cost * structure_vol * (p_et / P_ET_REF) ** 0.5

    # -----------------------------------------------------------------------
    # 220106: Vacuum System = vessel shell (volume-based) + gas-load pumping.
    # Vessel: double-walled SS chamber, port extensions, gauges, leak detection.
    # Pumping: installed speed S_req = Q_gas / P_op set by the gas throughput,
    # NOT vessel volume. Q_gas = NBI neutral-gas load (beam particle rate / E_b)
    # + fueling/exhaust throughput (fusion rate / burn_fraction). Outgassing
    # (surface-area term) is negligible for a baked UHV system and is omitted.
    # Applied uniformly to every concept; P_op (the plenum pressure at the pump
    # throat) is the concept-specific knob that makes low-pressure devices
    # (FRC, mirror) pump far more than a high-pressure tokamak divertor.
    # See docs/account_justification/CAS220106_vacuum_pumping.md
    # -----------------------------------------------------------------------
    c220106_vessel = cc.vessel_unit_cost * vessel_vol * (p_et / P_ET_REF) ** 0.6

    _KB = 1.380649e-23  # Boltzmann constant [J/K]
    _QE = 1.602176634e-19  # elementary charge [J/eV]
    kt_gas = _KB * cc.pump_gas_temp_k  # Pa·m³ per pumped particle (Q = N*kT)
    e_fus_mev = {
        Fuel.DT: cc.e_fus_mev_dt,
        Fuel.DD: cc.e_fus_mev_dd,
        Fuel.DHE3: cc.e_fus_mev_dhe3,
        Fuel.PB11: cc.e_fus_mev_pb11,
    }[fuel]
    # NBI neutral-gas throughput [Pa·m³/s]: beam particle rate = p_nbi / E_b.
    e_b_j = cc.nbi_beam_energy_kev * 1e3 * _QE
    q_nbi = cc.pump_nbi_gas_amplification * (p_nbi * 1e6 / e_b_j) * kt_gas
    # Fueling/exhaust throughput [Pa·m³/s]: fuel feed = reaction_rate / burn_fraction;
    # the unburned fraction circulates through the pump each pass.
    e_fus_j = e_fus_mev * 1e6 * _QE
    reaction_rate = p_fus * 1e6 / e_fus_j  # reactions/s
    q_fuel = (
        cc.pump_ion_per_reaction
        * (1.0 - burn_fraction)
        / burn_fraction
        * reaction_rate
        * kt_gas
    )
    s_req = (q_nbi + q_fuel) / vac_op_pressure_pa  # required pumping speed [m³/s]
    c220106_pump = cc.pump_unit_cost * s_req
    c220106 = c220106_vessel + c220106_pump

    # -----------------------------------------------------------------------
    # 220107: Power Supplies — vendor-purchased (ABB, GE, Siemens)
    # Steady-state: high-current DC for superconducting magnets, switchgear.
    # Pulsed: cap bank + switches + charging + buswork on $/J_stored basis.
    # See docs/account_justification/CAS22_plant_systems.md
    # -----------------------------------------------------------------------
    if family == ConfinementFamily.PULSED:
        # $/J_stored basis: pulsed driver (cap bank, laser, accelerator)
        c220107 = cc.c_cap_allin_per_joule * e_stored_mj  # $/J * MJ = M$
    else:
        c220107 = cc.power_supplies_base * (p_et / 1000.0) ** 0.7

    # -----------------------------------------------------------------------
    # 220108: Divertor (MFE) or Target Factory (IFE/MIF)
    # MFE: W monoblock cassettes on CuCrZr heat sinks
    # IFE/MIF: high-rep-rate target manufacturing infrastructure
    # See docs/account_justification/CAS22_plant_systems.md
    # -----------------------------------------------------------------------
    if family == ConfinementFamily.STEADY_STATE:
        if concept == ConfinementConcept.DIPOLE:
            # Closed-field-line dipole: particles exhaust through the loss cone
            # at the top/bottom openings of the chamber, so there is no W
            # monoblock divertor cassette to manufacture (Simpson 2026 §2.2.6).
            c220108 = 0.0
        else:
            c220108 = cc.divertor_base * (p_th / 1000.0) ** 0.5
    elif manufactured_target:
        # Pulsed concept that consumes a fabricated target/liner each shot
        # (laser/heavy-ion capsule, MagLIF/Z-pinch liner): high-rep-rate
        # target manufacturing infrastructure. Gated on manufactured_target
        # rather than the concept enum so in-situ-formation concepts
        # (plasma-jet, FRC/theta/DPF/staged-Z, liquid-liner MTF) carry no
        # phantom target factory by default — no cost override required.
        c220108 = cc.target_factory_base * (p_et / 1000.0) ** 0.7
    else:
        # In-situ plasma/liner formation: no fabricated target, no factory.
        c220108 = 0.0

    # -----------------------------------------------------------------------
    # 220109: Direct Energy Converter
    # Inductive DEC: circuit-derived markups on pulsed driver cost.
    # Electrostatic DEC: for mirrors/FRCs with directed axial exhaust.
    # See docs/account_justification/CAS220109_direct_energy_converter.md
    # -----------------------------------------------------------------------
    if pulsed_conversion == PulsedConversion.INDUCTIVE_DEC:
        # Inductive DEC: circuit-derived markups on pulsed driver cost
        markup_cap = eta_dec * (1.0 + q_sci * f_ch) - 1.0
        delta_cap = c220107 * jnp.maximum(markup_cap, 0.0)
        delta_switch = c220107 * cc.markup_switch_bidir
        delta_inv = cc.c_inv_per_kw_net * p_net / 1e3  # $/kW * MW -> M$
        delta_ctrl = c220107 * cc.markup_controls
        c220109 = delta_cap + delta_switch + delta_inv + delta_ctrl
    else:
        # Electrostatic DEC for mirrors (existing logic) — JAX-safe
        P_DEE_REF = 400.0
        p_dee_safe = jnp.where(p_dee > 0, p_dee, 1.0)
        c220109 = jnp.where(
            p_dee > 0,
            cc.dec_base * (p_dee_safe / P_DEE_REF) ** 0.7,
            0.0,
        )

    # -----------------------------------------------------------------------
    # 220110: Remote Handling & Maintenance Equipment
    # Fuel-dependent (rad-hardening tier) x concept-dependent (vessel geometry).
    # Base costs calibrated to toroidal geometry (tokamak/stellarator).
    # Linear concepts (mirror) have simpler end-access → lower cost.
    # See docs/account_justification/CAS220110_remote_handling.md
    # -----------------------------------------------------------------------
    rh_base = {
        Fuel.DT: cc.remote_handling_dt_base,
        Fuel.DD: cc.remote_handling_dd_base,
        Fuel.DHE3: cc.remote_handling_dhe3_base,
        Fuel.PB11: cc.remote_handling_pb11_base,
    }
    # Toroidal vessels (narrow ports) vs linear (end-access)
    rh_concept_scale = {
        ConfinementConcept.TOKAMAK: 1.0,
        ConfinementConcept.STELLARATOR: 1.0,
        ConfinementConcept.MIRROR: 0.55,
    }
    concept_scale = rh_concept_scale.get(concept, 0.5)
    c220110 = rh_base[fuel] * concept_scale * (p_et / 1000.0) ** 0.5

    # -----------------------------------------------------------------------
    # 220111: Installation Labor — 14% of reactor subtotal
    # Industry norm: 10-20% (nuclear 15-25%, conventional 10-15%)
    # See docs/account_justification/CAS22_plant_systems.md
    # -----------------------------------------------------------------------
    reactor_subtotal = (
        c220101
        + c220102
        + c220103
        + c220104
        + c220105
        + c220106
        + c220107
        + c220108
        + c220109
        + c220110
    )
    c220111 = cc.installation_frac * reactor_subtotal

    # -----------------------------------------------------------------------
    # 220112: Isotope Separation Plant — zeroed
    # No on-site separation plant. All isotope procurement is modeled as
    # market purchase in CAS80 (enriched $/kg prices). The separation
    # plant capital is embedded in the market price.
    # See: docs/account_justification/CAS220112_isotope_separation.md
    # -----------------------------------------------------------------------
    c220112 = 0.0

    # -----------------------------------------------------------------------
    # 220200: Main & Secondary Coolant
    # Primary loops + intermediate HX + secondary to steam generators
    # See docs/account_justification/CAS22_plant_systems.md
    # -----------------------------------------------------------------------
    # Plant-wide accounts use total plant power (n_mod * per-module)
    p_th_total = n_mod * p_th
    p_net_total = n_mod * p_net

    c220201 = 166.0 * (p_net_total / 1000.0)  # Primary coolant
    c220202 = 40.6 * (p_th_total / 3500.0) ** 0.55  # Intermediate coolant
    c220200 = c220201 + c220202

    # -----------------------------------------------------------------------
    # 220300: Auxiliary Cooling + Cryoplant
    # Cryoplant calibrated to ITER: EUR 148M for 75kW @ 4.5K (Air Liquide)
    # See docs/account_justification/CAS22_plant_systems.md
    # -----------------------------------------------------------------------
    c220301 = 1.10e-3 * p_th_total  # Aux coolant
    c220302 = 200.0 * (p_cryo / 30.0) ** 0.7  # Cryoplant (ref: $200M @ 30MW)
    c220300 = c220301 + c220302

    # -----------------------------------------------------------------------
    # 220400: Radioactive Waste Management
    # Low-level activated waste (no fission products)
    # See docs/account_justification/CAS22_plant_systems.md
    # -----------------------------------------------------------------------
    c220400 = 1.96 * (p_th_total / 1000.0)

    # -----------------------------------------------------------------------
    # 220500: Fuel Handling & Storage — fuel-dependent
    # DT: full tritium processing + containment ($120M @ 1 GWe)
    # DD: small-scale tritium + deuterium ($60M)
    # DHe3: He-3 recovery/recycling ($40M)
    # pB11: boron powder injection ($15M)
    # See docs/account_justification/CAS22_plant_systems.md
    # -----------------------------------------------------------------------
    fuel_handling_base = {
        Fuel.DT: cc.fuel_handling_dt_base,
        Fuel.DD: cc.fuel_handling_dd_base,
        Fuel.DHE3: cc.fuel_handling_dhe3_base,
        Fuel.PB11: cc.fuel_handling_pb11_base,
    }
    c220500 = fuel_handling_base[fuel] * (p_net_total / 1000.0) ** 0.7

    # -----------------------------------------------------------------------
    # 220600: Other Reactor Plant Equipment — catch-all
    # See docs/account_justification/CAS22_plant_systems.md
    # -----------------------------------------------------------------------
    c220600 = 11.5 * (p_net_total / 1000.0) ** 0.8

    # -----------------------------------------------------------------------
    # 220700: Instrumentation & Control — plasma control, diagnostics,
    # safety interlocks, data acquisition, plant computer
    # See docs/account_justification/CAS22_plant_systems.md
    # -----------------------------------------------------------------------
    c220700 = 85.0 * (p_th_total / 3500.0) ** 0.65

    # -----------------------------------------------------------------------
    # Total CAS22 (per module, then multiply)
    # -----------------------------------------------------------------------
    per_module_equipment = (
        c220101
        + c220102
        + c220103
        + c220104
        + c220105
        + c220106
        + c220107
        + c220108
        + c220109
        + c220110
        + c220112
    )
    # Multi-unit labor: unit 1 at full c220111, each subsequent unit
    # discounted by cc.multi_unit_labor_factor (default 0.92, fission norm).
    # Equipment is not discounted — each module is still a manufactured copy.
    total_labor = c220111 * (1.0 + (n_mod - 1) * cc.multi_unit_labor_factor)
    plant_wide = c220200 + c220300 + c220400 + c220500 + c220600 + c220700
    c220000 = per_module_equipment * n_mod + total_labor + plant_wide

    return {
        "C220101": c220101,
        "C220102": c220102,
        "C220103": c220103,
        "C220104": c220104,
        "C220105": c220105,
        "C220106": c220106,
        # Informational split of C220106 (vessel shell vs gas-load pumping).
        # Not aggregated: the canonical C220106 already carries the sum. These
        # are for visibility and sensitivity tracing only.
        "C220106_vessel": c220106_vessel,
        "C220106_pump": c220106_pump,
        "C220107": c220107,
        "C220108": c220108,
        "C220109": c220109,
        "C220110": c220110,
        "C220111": c220111,
        "C220112": c220112,
        "C220200": c220200,
        "C220300": c220300,
        "C220400": c220400,
        "C220500": c220500,
        "C220600": c220600,
        "C220700": c220700,
        "C220000": c220000,
    }
