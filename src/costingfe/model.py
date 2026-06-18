"""Top-level CostModel API: wires all 5 layers together."""

import difflib
import math
import numbers

import jax
import jax.numpy as jnp

from costingfe.defaults import (
    POWER_CYCLE_DEFAULTS,
    CostingConstants,
    cc_float_fields,
    get_magnet_properties,
    load_costing_constants,
    load_engineering_defaults,
)
from costingfe.layers.cas22 import cas22_reactor_plant_equipment
from costingfe.layers.costs import (
    cas10_preconstruction,
    cas21_buildings,
    cas23_turbine,
    cas24_electrical,
    cas25_misc,
    cas26_heat_rejection,
    cas27_special_materials,
    cas28_digital_twin,
    cas29_contingency,
    cas30_indirect,
    cas40_owner,
    cas50_supplementary,
    cas60_idc,
    cas70_om,
    cas80_fuel,
    cas90_financial,
)
from costingfe.layers.economics import compute_lcoe
from costingfe.layers.geometry import RadialBuild, compute_geometry
from costingfe.layers.mirror import (
    mirror_0d_inverse,
    mirror_size_from_power,
    net_electric_at_L,
)
from costingfe.layers.physics import (
    OperatingPointInfeasible,
    SizingInfeasible,
    mfe_forward_power_balance,
    mfe_inverse_power_balance,
    pulsed_dec_forward,
    pulsed_dec_inverse,
    pulsed_thermal_forward,
    pulsed_thermal_inverse,
)
from costingfe.layers.reactivity import (
    n_i_over_n_e,
    z_eff_fuel,
)
from costingfe.layers.tokamak import (
    _T_BRACKET_DEFAULTS,
    DisruptionModel,
    apply_disruption_penalty,
    aux_heating_from_confinement,
    derive_radial_build,
    resistive_recirc_power,
    tokamak_0d_forward,
    tokamak_0d_inverse,
    tokamak_size_from_power,
)
from costingfe.types import (
    CONCEPT_DEFAULT_CONVERSION,
    CONCEPT_TO_FAMILY,
    N_MOD_SIZED_CONCEPTS,
    BlanketFill,
    BlanketForm,
    CoilMaterial,
    ConfinementConcept,
    ConfinementFamily,
    CostResult,
    ForwardResult,
    Fuel,
    LaserDriverType,
    PowerCycle,
    PulsedConversion,
    WallMaterial,
)
from costingfe.validation import CostingInput


def _core_lifetime_fpy(cc, fuel, q_n, lifetime_yr, availability):
    """Fluence-based core lifetime [FPY]: Phi_max / q_n, clamped to [0.5,
    lifetime_yr * availability]. Floor of 0.5 FPY keeps the 1/q_n gradient
    finite at extreme wall loadings; cap at lifetime_yr * availability because
    nothing is replaced beyond plant life."""
    return jnp.clip(
        cc.fluence_limit(fuel) / jnp.maximum(q_n, 1e-6),
        0.5,
        lifetime_yr * availability,
    )


class CostModel:
    def __init__(
        self,
        concept: ConfinementConcept,
        fuel: Fuel,
        costing_constants: CostingConstants = None,
        power_cycle: PowerCycle = PowerCycle.RANKINE,
        pulsed_conversion: PulsedConversion = None,
        laser_driver_type: LaserDriverType = None,
    ):
        self.concept = concept
        self.fuel = fuel
        self.family = CONCEPT_TO_FAMILY[concept]
        self.power_cycle = power_cycle
        self.pulsed_conversion = pulsed_conversion or CONCEPT_DEFAULT_CONVERSION.get(
            concept
        )
        self._cc_user_provided = costing_constants is not None
        self.cc = costing_constants or load_costing_constants()
        self._eng_defaults = load_engineering_defaults(
            f"{self.family.value}_{concept.value}"
        )
        # Driver architecture for LASER_IFE. Default comes from the concept YAML
        # (laser_driver_type: dpssl), not hardcoded; explicit arg overrides it.
        if laser_driver_type is None:
            _ld = self._eng_defaults.get("laser_driver_type")
            if _ld is not None:
                laser_driver_type = LaserDriverType(_ld)
        self.laser_driver_type = laser_driver_type

    @staticmethod
    def _dhe3_f_He3_eff(params):
        """Cumulative bred-He3 fusion fraction given burn-per-pass and recovery.

        Bred He-3 from D-D side reactions enters the same exhaust/recovery
        loop as primary He-3, so its cumulative fusion probability equals
        bf / (1 - fr (1 - bf)) where bf = burn_fraction and fr = fuel_recovery.
        """
        bf = params["burn_fraction"]
        fr = params["fuel_recovery"]
        return bf / (1.0 - fr * (1.0 - bf))

    def _effective_eta_pin(self, params):
        """Heating wall-plug efficiency eta_pin = eta_source(method) x eta_couple.

        For NBI/RF-heated concepts (those that set eta_couple), the effective
        eta_pin is the mix-weighted product
            eta_pin = sum_i p_i / sum_i p_i / (eta_source_i * eta_couple).
        Concepts without an NBI/RF method (electrostatic orbitron/polywell;
        pulsed drivers) keep their explicit eta_pin. For an NBI/RF-heated
        concept eta_pin is DERIVED, so passing it directly is rejected (override
        eta_couple instead). Dict-key checks are static (JAX-safe under AD
        tracing); the arithmetic traces cleanly.
        """
        if "eta_couple" not in params:
            # Electrostatic / pulsed concepts: eta_pin is the direct input.
            return params["eta_pin"]
        if "eta_pin" in params:
            raise ValueError(
                "eta_pin cannot be set for an NBI/RF-heated concept (it is "
                "derived from eta_source x eta_couple). Override eta_couple or "
                "the eta_source_* constants instead."
            )
        ec = params["eta_couple"]
        cc = self.cc
        p_input = params.get("p_input", 0.0)
        p_nbi = params.get("p_nbi", p_input)
        p_icrf = params.get("p_icrf", 0.0)
        p_ecrh = params.get("p_ecrh", 0.0)
        p_lhcd = params.get("p_lhcd", 0.0)
        num = p_nbi + p_icrf + p_ecrh + p_lhcd
        den = (
            p_nbi / (cc.eta_source_nbi * ec)
            + p_icrf / (cc.eta_source_icrf * ec)
            + p_ecrh / (cc.eta_source_ecrh * ec)
            + p_lhcd / (cc.eta_source_lhcd * ec)
        )
        return num / den

    def _power_balance(self, params, n_mod):
        """Dispatch power balance based on confinement family."""
        # Derive eta_pin from per-method source x per-concept coupling, so all
        # downstream power-balance sites (and the 0D branch) use the same value.
        params = {**params, "eta_pin": self._effective_eta_pin(params)}
        p_net_per_mod = params["net_electric_mw"] / n_mod

        # 0D dispatch: tokamak or mirror branch
        use_0d = params.get("use_0d_model", False)
        if use_0d and self.concept == ConfinementConcept.TOKAMAK:
            return self._power_balance_0d(params, n_mod)
        if use_0d and self.concept == ConfinementConcept.MIRROR:
            return self._power_balance_mirror_0d(params, n_mod)

        if self.family == ConfinementFamily.STEADY_STATE:
            # Parse impurity model params
            wm_raw = params.get("wall_material")
            wall_mat = None
            if wm_raw is not None:
                wall_mat = WallMaterial(wm_raw) if isinstance(wm_raw, str) else wm_raw
            impurity_kw = dict(
                wall_material=wall_mat,
                seeded_impurities=params.get("seeded_impurities") or None,
                T_edge=params["T_edge"],
                tau_ratio=params["tau_ratio"],
                fw_area=params.get("fw_area", 0.0),
            )

            # Synchrotron geometry: for mirrors (R0=0), use L/(2*pi)
            R_major = params.get("R0", 0.0)
            L = params.get("chamber_length", 0.0)
            R_major = jnp.where(R_major > 0, R_major, L / (2 * math.pi))
            a_minor = params.get("plasma_t", 0.0)
            sync_kw = dict(
                R_major=R_major,
                a_minor=a_minor,
                kappa=params.get("elon", 1.0),
                R_w=params["R_w"],
            )

            # Plasma radiation parameters (from YAML defaults or user overrides)
            def _to_num(v):
                """str→float (PyYAML parses '5e19' as str); pass Tracers."""
                return float(v) if isinstance(v, str) else v

            rad_kw = dict(
                n_e=_to_num(params["n_e"]),
                T_e=_to_num(params["T_e"]),
                Z_eff=_to_num(params["Z_eff"]),
                plasma_volume=_to_num(params["plasma_volume"]),
                B=_to_num(params["B"]),
                radiation_peaking_factor=_to_num(params["radiation_peaking_factor"]),
                f_rad_fus=params.get("f_rad_fus", self.cc.f_rad_fus(self.fuel)),
            )

            fuel_frac_kw = dict(
                dd_f_T=params["dd_f_T"],
                dd_f_He3=params["dd_f_He3"],
                dhe3_dd_frac=params["dhe3_dd_frac"],
                dhe3_f_T=params["dhe3_f_T"],
                dhe3_f_He3=self._dhe3_f_He3_eff(params),
                pb11_f_alpha_n=params["pb11_f_alpha_n"],
                pb11_f_p_n=params["pb11_f_p_n"],
            )

            p_fus = mfe_inverse_power_balance(
                p_net_target=p_net_per_mod,
                fuel=self.fuel,
                p_input=params["p_input"],
                mn=params["mn"],
                eta_th=params["eta_th"],
                eta_p=params["eta_p"],
                eta_pin=params["eta_pin"],
                eta_de=params["eta_de"],
                f_sub=params["f_sub"],
                f_dec=params["f_dec"],
                p_coils=params["p_coils"],
                p_cool=params["p_cool"],
                p_pump=params["p_pump"],
                p_trit=params["p_trit"],
                p_house=params["p_house"],
                p_cryo=params["p_cryo"],
                **rad_kw,
                **fuel_frac_kw,
                **impurity_kw,
                **sync_kw,
            )
            pt = mfe_forward_power_balance(
                p_fus=p_fus,
                fuel=self.fuel,
                p_input=params["p_input"],
                mn=params["mn"],
                eta_th=params["eta_th"],
                eta_p=params["eta_p"],
                eta_pin=params["eta_pin"],
                eta_de=params["eta_de"],
                f_sub=params["f_sub"],
                f_dec=params["f_dec"],
                p_coils=params["p_coils"],
                p_cool=params["p_cool"],
                p_pump=params["p_pump"],
                p_trit=params["p_trit"],
                p_house=params["p_house"],
                p_cryo=params["p_cryo"],
                **rad_kw,
                **fuel_frac_kw,
                **impurity_kw,
                **sync_kw,
            )

        elif self.family == ConfinementFamily.PULSED:
            fuel_frac_kw = dict(
                dd_f_T=params["dd_f_T"],
                dd_f_He3=params["dd_f_He3"],
                dhe3_dd_frac=params["dhe3_dd_frac"],
                dhe3_f_T=params["dhe3_f_T"],
                dhe3_f_He3=self._dhe3_f_He3_eff(params),
                pb11_f_alpha_n=params["pb11_f_alpha_n"],
                pb11_f_p_n=params["pb11_f_p_n"],
            )
            common_kw = dict(
                fuel=self.fuel,
                f_rep=params["f_rep"],
                mn=params["mn"],
                eta_th=params["eta_th"],
                eta_pin=params["eta_pin"],
                f_rad=params.get("f_rad", self.cc.f_rad(self.fuel)),
                f_sub=params["f_sub"],
                p_pump=params["p_pump"],
                p_trit=params["p_trit"],
                p_house=params["p_house"],
                p_cryo=params["p_cryo"],
                p_target=params.get("p_target", 0.0),
                p_coils=params.get("p_coils", 0.0),
                **fuel_frac_kw,
            )
            # Hybrid-thermal parameters: only threaded for pure-thermal path.
            # INDUCTIVE_DEC has its own driver-side DEC and shouldn't receive
            # ash-side f_dec/eta_de.
            thermal_kw = dict(
                f_dec=params.get("f_dec", 0.0),
                eta_de=params.get("eta_de", 0.6),
            )

            if self.pulsed_conversion == PulsedConversion.INDUCTIVE_DEC:
                dec_kw = dict(
                    eta_dec=params["eta_dec"],
                    f_pdv=params.get("f_pdv", self.cc.f_pdv),
                )
                # DEC inverse: solve for e_driver_mj from P_net and Q_eng
                q_eng = params.get("q_eng", 5.0)
                p_fus, e_driver_solved = pulsed_dec_inverse(
                    p_net_target=p_net_per_mod,
                    q_eng=q_eng,
                    **common_kw,
                    **dec_kw,
                )
                # Use solved e_driver_mj for forward pass
                common_kw["e_driver_mj"] = e_driver_solved
                pt = pulsed_dec_forward(
                    p_fus=p_fus,
                    **common_kw,
                    **dec_kw,
                )
            else:
                q_eng = params.get("q_eng", 5.0)
                p_fus, e_driver_solved = pulsed_thermal_inverse(
                    p_net_target=p_net_per_mod,
                    q_eng=q_eng,
                    **common_kw,
                    **thermal_kw,
                )
                common_kw["e_driver_mj"] = e_driver_solved
                pt = pulsed_thermal_forward(
                    p_fus=p_fus,
                    **common_kw,
                    **thermal_kw,
                )

        else:
            raise ValueError(f"Unknown confinement family: {self.family}")

        return pt

    def _power_balance_0d(self, params, n_mod):
        """0D tokamak power balance: derives p_fus from plasma physics."""
        mode = params.get("0d_mode", "inverse")
        R = params["R0"]
        a = params["plasma_t"]
        kappa = params["elon"]
        B = params["B"]
        q95 = params["q95"]
        f_GW = params["f_GW"]

        # Parse impurity model params
        wm_raw = params.get("wall_material")
        wall_mat = None
        if wm_raw is not None:
            wall_mat = WallMaterial(wm_raw) if isinstance(wm_raw, str) else wm_raw
        impurity_kw = dict(
            wall_material=wall_mat,
            seeded_impurities=params.get("seeded_impurities") or None,
            T_edge=params["T_edge"],
            tau_ratio=params["tau_ratio"],
            fw_area=params.get("fw_area", 0.0),
        )

        base_frac_kw = dict(
            dd_f_T=params["dd_f_T"],
            dd_f_He3=params["dd_f_He3"],
            dhe3_f_T=params["dhe3_f_T"],
            dhe3_f_He3=self._dhe3_f_He3_eff(params),
            pb11_f_alpha_n=params["pb11_f_alpha_n"],
            pb11_f_p_n=params["pb11_f_p_n"],
        )

        # Kernel-only knobs: the 0D plasma model consumes these; the
        # mfe_*_power_balance functions must never receive them.
        kernel_kw = dict(
            T_i_over_T_e=params["T_i_over_T_e"],
            dhe3_fuel_ratio=params["dhe3_fuel_ratio"],
            pb11_fuel_ratio=params["pb11_fuel_ratio"],
        )

        if mode == "forward":
            plasma_state = tokamak_0d_forward(
                R=R,
                a=a,
                kappa=kappa,
                B=B,
                q95=q95,
                f_GW=f_GW,
                T_e=params["T_e"],
                p_input=params["p_input"],
                fuel=self.fuel,
                M_ion=params.get("M_ion", 2.5),
                Z_eff=params.get("Z_eff", 1.5),
                lambda_q=params.get("lambda_q", 0.002),
                dhe3_dd_frac_pin=params["dhe3_dd_frac_pin"],
                **base_frac_kw,
                **kernel_kw,
            )
            pb_frac = (
                plasma_state.dhe3_dd_frac_eff
                if self.fuel == Fuel.DHE3
                else params["dhe3_dd_frac"]
            )
            pt = mfe_forward_power_balance(
                p_fus=plasma_state.p_fus,
                fuel=self.fuel,
                p_input=params["p_input"],
                mn=params["mn"],
                eta_th=params["eta_th"],
                eta_p=params["eta_p"],
                eta_pin=params["eta_pin"],
                eta_de=params["eta_de"],
                f_sub=params["f_sub"],
                f_dec=params["f_dec"],
                p_coils=params["p_coils"],
                p_cool=params["p_cool"],
                p_pump=params["p_pump"],
                p_trit=params["p_trit"],
                p_house=params["p_house"],
                p_cryo=params["p_cryo"],
                n_e=plasma_state.n_e,
                T_e=plasma_state.T_e,
                Z_eff=params.get("Z_eff", 1.5),
                plasma_volume=plasma_state.V_plasma,
                B=B,
                R_major=R,
                a_minor=a,
                kappa=kappa,
                R_w=params["R_w"],
                dhe3_dd_frac=pb_frac,
                f_rad_fus=params.get("f_rad_fus"),
                **base_frac_kw,
                **impurity_kw,
            )
        else:
            # Inverse mode (default): find T_e that produces required p_fus
            plasma_state, pt = tokamak_0d_inverse(
                p_net_target=params["net_electric_mw"],
                R=R,
                a=a,
                kappa=kappa,
                B=B,
                q95=q95,
                f_GW=f_GW,
                fuel=self.fuel,
                M_ion=params.get("M_ion", 2.5),
                Z_eff=params.get("Z_eff", 1.5),
                lambda_q=params.get("lambda_q", 0.002),
                p_input=params["p_input"],
                mn=params["mn"],
                eta_th=params["eta_th"],
                eta_p=params["eta_p"],
                eta_pin=params["eta_pin"],
                eta_de=params["eta_de"],
                f_sub=params["f_sub"],
                f_dec=params["f_dec"],
                p_coils=params["p_coils"],
                p_cool=params["p_cool"],
                p_pump=params["p_pump"],
                p_trit=params["p_trit"],
                p_house=params["p_house"],
                p_cryo=params["p_cryo"],
                n_mod=n_mod,
                dhe3_dd_frac=params["dhe3_dd_frac"],
                dhe3_dd_frac_pin=params["dhe3_dd_frac_pin"],
                f_rad_fus=params.get("f_rad_fus"),
                enforce_plasma_limits=params["enforce_plasma_limits"],
                **base_frac_kw,
                **kernel_kw,
            )

        self._plasma_state = plasma_state
        return pt

    def _power_balance_mirror_0d(self, params, n_mod):
        """0D mirror power balance: derives p_fus from mirror plasma physics.

        Maps YAML params to mirror_0d_inverse, stores the resulting
        MirrorPlasmaState on self._plasma_state, and returns the PowerTable.
        Parameter mapping per the spec:
          chamber_length -> L, plasma_t -> a, B -> B_min, R_m -> R_m.
        R_w comes from YAML (0.4 for mirrors per the open-ended radiation
        geometry). Synchrotron geometry (R_eff = L / 2pi) is handled inside
        mirror_0d_inverse / mirror_0d_forward; do not remap here.
        """
        L = params["chamber_length"]
        a = params["plasma_t"]
        B_min = params["B"]
        R_m = params["R_m"]
        T_e = params["T_e"]

        def _to_num(v):
            return float(v) if isinstance(v, str) else v

        # Parse impurity model for the mirror power balance
        wm_raw = params.get("wall_material")
        wall_mat = None
        if wm_raw is not None:
            wall_mat = WallMaterial(wm_raw) if isinstance(wm_raw, str) else wm_raw

        plasma_state, pt = mirror_0d_inverse(
            p_net_target=params["net_electric_mw"],
            L=L,
            a=a,
            B_min=B_min,
            R_m=R_m,
            n_e=_to_num(params["n_e"]),
            T_e=_to_num(T_e),
            fuel=self.fuel,
            M_ion=params.get("M_ion", 2.5),
            Z_eff=_to_num(params["Z_eff"]),
            R_w=params["R_w"],
            p_input=params["p_input"],
            mn=params["mn"],
            eta_th=params["eta_th"],
            eta_p=params["eta_p"],
            eta_pin=params["eta_pin"],
            eta_de=params["eta_de"],
            f_sub=params["f_sub"],
            f_dec=params["f_dec"],
            p_coils=params["p_coils"],
            p_cool=params["p_cool"],
            p_pump=params["p_pump"],
            p_trit=params["p_trit"],
            p_house=params["p_house"],
            p_cryo=params["p_cryo"],
            n_mod=n_mod,
            dd_f_T=params["dd_f_T"],
            dd_f_He3=params["dd_f_He3"],
            dhe3_dd_frac=params["dhe3_dd_frac"],
            dhe3_f_T=params["dhe3_f_T"],
            dhe3_f_He3=self._dhe3_f_He3_eff(params),
            pb11_f_alpha_n=params["pb11_f_alpha_n"],
            pb11_f_p_n=params["pb11_f_p_n"],
            dhe3_fuel_ratio=params["dhe3_fuel_ratio"],
            pb11_fuel_ratio=params["pb11_fuel_ratio"],
            dhe3_dd_frac_pin=params["dhe3_dd_frac_pin"],
            vacuum_t=params["vacuum_t"],
            wall_material=wall_mat,
            T_edge=params["T_edge"],
            tau_ratio=params["tau_ratio"],
            f_rad_fus=params.get("f_rad_fus"),
            beta_max=params["beta_max"],
            q_wall_max=params["q_wall_max"],
            q_surface_max=params["q_surface_max"],
            p_aux_floor=params["p_aux_floor"],
            plug_density_ratio=params["plug_density_ratio"],
            collisionality_min=params["collisionality_min"],
            f_alpha_heat=params["f_alpha_heat"],
            T_e_plug=_to_num(params["T_e_plug"]),
            p_plug=params["p_plug"],
            enforce_plasma_limits=params["enforce_plasma_limits"],
        )

        self._plasma_state = plasma_state
        return pt

    def _size_tokamak(self, params, n_mod):
        """Solve geometry from the power target, inject it into params, and
        return the power table. Mutates params with the solved geometry and the
        derived operating point: R0, plasma_t (a), B, b_center, T_e, eta_pin, and
        p_coils (recirc term added). Also sets 0d_mode="forward", which forces the
        forward control-flow branch of the 0D power balance."""
        props = get_magnet_properties(params["coil_material"])
        # Derive eta_pin (heating wall-plug efficiency) exactly as _power_balance
        # does, so both the sizing solve and the final 0D forward balance use it.
        params["eta_pin"] = self._effective_eta_pin(params)
        solve_params = dict(params)
        solve_params["b_max"] = props.b_max
        solve_params["recirc_power_factor"] = props.recirc_power_factor
        # The sizing solver reads dhe3_f_He3 raw (it is derived, not in YAML).
        solve_params["dhe3_f_He3"] = self._dhe3_f_He3_eff(params)
        # Size one module to the per-module net power.
        solve_params["net_electric_mw"] = params["net_electric_mw"] / n_mod

        result = tokamak_size_from_power(solve_params, self.fuel)

        # Inject solved geometry so downstream geometry and coil cost use it.
        params["R0"] = result.R0
        params["plasma_t"] = result.a
        params["B"] = result.B0
        params["T_e"] = result.T_e
        self._last_R0 = result.R0  # exposed for inspection and integration tests
        self._last_B0 = result.B0  # solved on-axis field, exposed for inspection

        # Keep the final power balance consistent with the sizing solve: add the
        # resistive-coil recirculation term (zero for superconductors) to p_coils
        # using the solved geometry, matching net_electric_at_R0.
        params["p_coils"] = params["p_coils"] + float(
            resistive_recirc_power(
                props.recirc_power_factor,
                result.B0,
                result.R0,
                result.a,
                params["elon"],
            )
        )

        # Solve the auxiliary heating from the confinement requirement at the
        # solved operating point, so H_factor drives recirculating power and the
        # C220104 driver cost. Run one forward at the solved point to get I_p,
        # n_e, V_plasma, p_alpha, then close the form via aux_heating_from_confinement.
        ps = tokamak_0d_forward(
            R=result.R0,
            a=result.a,
            kappa=params["elon"],
            B=result.B0,
            q95=params["q95"],
            f_GW=params["f_GW"],
            T_e=result.T_e,
            p_input=params["p_input"],
            fuel=self.fuel,
            M_ion=params["M_ion"],
            Z_eff=params["Z_eff"],
            lambda_q=params["lambda_q"],
            dd_f_T=params["dd_f_T"],
            dd_f_He3=params["dd_f_He3"],
            dhe3_dd_frac_pin=params["dhe3_dd_frac_pin"],
            dhe3_f_T=params["dhe3_f_T"],
            dhe3_f_He3=self._dhe3_f_He3_eff(params),
            pb11_f_alpha_n=params["pb11_f_alpha_n"],
            pb11_f_p_n=params["pb11_f_p_n"],
            T_i_over_T_e=params["T_i_over_T_e"],
            dhe3_fuel_ratio=params["dhe3_fuel_ratio"],
            pb11_fuel_ratio=params["pb11_fuel_ratio"],
        )
        p_aux = float(
            aux_heating_from_confinement(
                params["H_factor"],
                ps.I_p,
                result.B0,
                ps.n_e,
                ps.T_e,
                ps.V_plasma,
                ps.p_alpha,
                result.R0,
                result.a,
                params["elon"],
                params["M_ion"],
                T_i_over_T_e=params["T_i_over_T_e"],
                n_i_frac=n_i_over_n_e(
                    self.fuel, params["dhe3_fuel_ratio"], params["pb11_fuel_ratio"]
                ),
            )
        )
        old_input = params["p_input"]
        params["p_input"] = p_aux
        # Rescale the heating mix so the driver cost reflects the solved aux power.
        # Mirror the eta_pin defaulting convention: p_nbi defaults to the old
        # p_input when absent, the others to 0.
        old_mix = (
            params.get("p_nbi", old_input)
            + params.get("p_ecrh", 0.0)
            + params.get("p_icrf", 0.0)
            + params.get("p_lhcd", 0.0)
        )
        if old_mix > 0:
            scale = p_aux / old_mix
            params["p_nbi"] = params.get("p_nbi", old_input) * scale
            for k in ("p_ecrh", "p_icrf", "p_lhcd"):
                params[k] = params.get(k, 0.0) * scale

        # Re-run the forward 0D power balance at the solved point to produce the
        # full PowerTable and plasma state for the rest of the pipeline.
        params["0d_mode"] = "forward"
        return self._power_balance_0d(params, n_mod)

    def _size_mirror(self, params, n_mod):
        """Solve chamber_length from the power target, inject it into params, and
        return the power table built DIRECTLY from the GSS-optimum operating point.

        The sizing solve uses mirror_size_from_power (bisection over L at the
        constraint-boundary operating point with GSS over T_i). The final plasma
        state and power table are taken from the same GSS forward path that the
        bisection optimizes (net_electric_at_L with return_full=True), so the
        reported operating point is exactly (T_star, n_e_star). This keeps beta
        bounded by the f_beta / wall / surface construction (beta <= beta_max by
        construction). The earlier approach re-solved T_i through the inverse at
        fixed n_e_star, which discarded T_star and let beta drift above beta_max.
        """
        # Derive eta_pin (heating wall-plug efficiency) exactly as _power_balance
        # does, so the sizing solve and the final forward state use the same value.
        params["eta_pin"] = self._effective_eta_pin(params)

        # Build the solve_params dict that mirror_size_from_power expects.
        # It needs B_min (= params["B"]) and the sizing knobs from the YAML.
        solve_params = dict(params)
        solve_params["B_min"] = params["B"]
        solve_params["a"] = params["plasma_t"]
        solve_params["dhe3_f_He3"] = self._dhe3_f_He3_eff(params)
        # Per-module target for the bisection
        solve_params["net_electric_mw"] = params["net_electric_mw"] / n_mod
        solve_params["n_mod"] = 1

        L_solved = mirror_size_from_power(solve_params, self.fuel)

        # Build the full forward state and power table at the GSS optimum. This
        # surfaces the (MirrorPlasmaState, PowerTable) already computed inside the
        # bisection's net_electric_at_L, so the operating point is consistent by
        # construction and beta is bounded by the f_beta / wall / surface caps.
        _pn, ps, pt = net_electric_at_L(
            L_solved, solve_params, self.fuel, return_full=True
        )

        # Write solved geometry and density back so all downstream layers see them.
        params["chamber_length"] = L_solved
        params["n_e"] = float(ps.n_e)
        self._last_L = L_solved  # exposed for inspection
        self._plasma_state = ps

        return pt

    def _size_modular(self, params, n_mod):
        """Solve the integer module count for a module-replication concept and
        return the per-module power table.

        Unlike the tokamak/mirror geometry solves, this is a division: a plant
        target is met by replicating a fixed module of design net power
        `module_net_mwe`. n_mod = ceil(target / module_net_mwe), each module runs
        at its design power, and the realized plant net (n_mod * module_net_mwe)
        overshoots the target by less than one module (a fractional module cannot
        be built). The solved count is exposed via self._last_n_mod and threads
        into the cost aggregation as the effective n_mod. n_mod is a physical
        integer; pinning it (caller n_mod != 1) is rejected.
        """
        if n_mod != 1:
            raise ValueError(
                "n_mod cannot be pinned in size_from_power mode for "
                "module-replication concepts; it is solved from module_net_mwe. "
                "Remove n_mod or set size_from_power=False."
            )
        module_net_mwe = params["module_net_mwe"]
        target = params["net_electric_mw"]
        n_solved = max(1, math.ceil(target / module_net_mwe))
        # Each module at its design power; the plant overshoots target by < 1 module.
        params["net_electric_mw"] = n_solved * module_net_mwe
        self._last_n_mod = n_solved
        return self._power_balance(params, n_solved)

    def _optimize_gss(self, lcoe_of_param, param_lo, param_hi):
        """Golden-section minimize LCOE over a scalar parameter in [param_lo, param_hi].

        lcoe_of_param is a callable mapping a trial parameter value to the
        resulting LCOE (it runs the full sizing+cost pipeline). Returns the
        minimizing parameter value. Used for both tokamak f_GW and mirror f_beta.
        """
        invphi = (5**0.5 - 1) / 2
        invphi2 = (3 - 5**0.5) / 2
        lo, hi = param_lo, param_hi
        h = hi - lo
        c = lo + invphi2 * h
        d = lo + invphi * h
        fc = lcoe_of_param(c)
        fd = lcoe_of_param(d)
        for _ in range(self._GSS_OPT_ITERS):
            if fc < fd:  # minimizing: keep [lo, d]
                hi, d, fd = d, c, fc
                h = hi - lo
                c = lo + invphi2 * h
                fc = lcoe_of_param(c)
            else:
                lo, c, fc = c, d, fd
                h = hi - lo
                d = lo + invphi * h
                fd = lcoe_of_param(d)
        return 0.5 * (lo + hi)

    def forward(
        self,
        net_electric_mw: float,
        availability: float,
        lifetime_yr: float,
        n_mod: float = 1.0,
        interest_rate: float = 0.07,
        inflation_rate: float = 0.02,
        noak: bool = True,
        cost_overrides: dict[str, float] | None = None,
        override_reference_mw: float | None = None,
        **overrides,
    ) -> ForwardResult:
        """Forward costing: customer requirements -> LCOE.

        If override_reference_mw is set, cost_overrides are interpreted as
        absolute M$ values valid at that reference power.  The framework
        scales them to net_electric_mw by computing the ratio of each
        account at the target vs. reference power and applying that as a
        multiplier.  This preserves the model's internal scaling laws
        (which depend on different power quantities per account) while
        respecting the user's empirical data.
        """
        # construction_time_yr is concept config: it lives in the concept YAML,
        # not a signature default (which would silently mask the YAML value). A
        # customer override arrives as a keyword and lands in **overrides; pull
        # it out here, before the unknown-kwarg check and the merge below.
        construction_time_yr = overrides.pop(
            "construction_time_yr", self._eng_defaults["construction_time_yr"]
        )
        if override_reference_mw is not None and cost_overrides:
            cost_overrides = self._scale_overrides(
                cost_overrides,
                override_reference_mw,
                net_electric_mw,
                availability=availability,
                lifetime_yr=lifetime_yr,
                n_mod=n_mod,
                construction_time_yr=construction_time_yr,
                interest_rate=interest_rate,
                inflation_rate=inflation_rate,
                noak=noak,
                **overrides,
            )
        # Reject unknown override kwargs. params.update(overrides) below
        # silently swallows any key, so a stale/misspelled parameter (e.g.
        # r_coil for r_bore, b_max for b_center) would leave the YAML default
        # in force and produce a wrong cost with no signal. Validate against
        # the concept's engineering parameters (its YAML), the costing-constant
        # fields, and the cross-cutting optional/derived knobs that forward()
        # and cas22 read via params.get() but that a given concept's YAML need
        # not declare (see _OPTIONAL_OVERRIDE_KEYS).
        allowed = (
            set(self._eng_defaults)
            | set(cc_float_fields())
            | self._OPTIONAL_OVERRIDE_KEYS
        )
        unknown = [k for k in overrides if k not in allowed]
        if unknown:
            hints = []
            for k in sorted(unknown):
                close = difflib.get_close_matches(k, allowed, n=1)
                hints.append(f"{k} (did you mean {close[0]}?)" if close else k)
            raise ValueError(
                f"forward() got unknown parameter(s) for concept "
                f"{self.concept.value}: {', '.join(hints)}. Unknown kwargs are "
                "not silently ignored, as that would hide a costing error."
            )

        # Merge defaults with overrides
        params = dict(self._eng_defaults)
        # Inject CostingConstants float fields into params so they are
        # JAX-traceable for sensitivity analysis.  User overrides take
        # precedence (via params.update(overrides) below).
        for name in cc_float_fields():
            params.setdefault(name, getattr(self.cc, name))
        params.update(overrides)
        # Apply power cycle preset: inject eta_th and BOP coefficients.
        # eta_th is no longer in concept YAMLs — the preset is the source
        # of truth. User explicit kwargs (in `overrides`) always win.
        cycle_preset = POWER_CYCLE_DEFAULTS[self.power_cycle]
        if "eta_th" not in overrides:
            if self.pulsed_conversion == PulsedConversion.INDUCTIVE_DEC:
                # Pure DEC: default to no thermal BOP
                params["eta_th"] = 0.0
            else:
                params["eta_th"] = cycle_preset["eta_th"]
        # BOP coefficients: apply preset unless user provided custom
        # CostingConstants (which means they want full control).
        if not self._cc_user_provided:
            for cc_key in ("turbine_per_mw", "heat_rej_per_mw"):
                params[cc_key] = cycle_preset[cc_key]
        params.update(
            dict(
                net_electric_mw=net_electric_mw,
                availability=availability,
                lifetime_yr=lifetime_yr,
                n_mod=n_mod,
                construction_time_yr=construction_time_yr,
                interest_rate=interest_rate,
                inflation_rate=inflation_rate,
                noak=noak,
                fuel=self.fuel,
                concept=self.concept,
            )
        )

        # Zero tritium processing power for non-DT fuels (no breeding loop)
        if self.fuel != Fuel.DT and "p_trit" not in overrides:
            params["p_trit"] = 0.0

        # Fuel-dependent f_rad default for pulsed concepts
        if self.family == ConfinementFamily.PULSED and "f_rad" not in overrides:
            params.setdefault("f_rad", self.cc.f_rad(self.fuel))

        # Validate merged parameters (skip under JAX tracing)
        _tracing = any(isinstance(v, jax.core.Tracer) for v in params.values())
        if not _tracing:
            CostingInput(
                concept=self.concept,
                fuel=self.fuel,
                net_electric_mw=net_electric_mw,
                availability=availability,
                lifetime_yr=lifetime_yr,
                n_mod=n_mod,
                construction_time_yr=construction_time_yr,
                interest_rate=interest_rate,
                inflation_rate=inflation_rate,
                noak=noak,
                cost_overrides=cost_overrides or {},
                **{
                    k: v
                    for k, v in params.items()
                    if k in CostingInput.model_fields
                    and k
                    not in {
                        "concept",
                        "fuel",
                        "net_electric_mw",
                        "availability",
                        "lifetime_yr",
                        "n_mod",
                        "construction_time_yr",
                        "interest_rate",
                        "inflation_rate",
                        "noak",
                        "cost_overrides",
                    }
                },
            )

        # optimize_lcoe implies size_from_power (you can only optimize the
        # sized machine's operating point).
        if params.get("optimize_lcoe", False):
            params["size_from_power"] = True

        # Note: sizing mode does NOT force use_0d_model. The disruption penalty
        # keys off self._plasma_state (which _size_tokamak sets), so it is active
        # for sized machines regardless. Leaving use_0d_model alone keeps the
        # design's radial build (YAML or overrides) instead of clobbering it with
        # the generic fuel-derived build, which would suppress on-axis field on a
        # compact high-field machine.

        # 0D radial build: derive thicknesses from fuel before geometry
        self._plasma_state = None
        use_0d = params.get("use_0d_model", False)
        if use_0d and self.concept == ConfinementConcept.TOKAMAK:
            rb_derived = derive_radial_build(
                self.fuel,
                blanket_t=overrides.get("blanket_t"),
                ht_shield_t=overrides.get("ht_shield_t"),
                structure_t=overrides.get("structure_t"),
                vessel_t=overrides.get("vessel_t"),
            )
            for k, v in rb_derived.items():
                if k not in overrides:
                    params[k] = v

        # Multi-fuel kernel inputs for the 0D/sizing paths: effective Z_eff
        # (fuel-ion contribution + impurity excess over hydrogenic), the
        # dhe3_dd_frac pin (explicit user override -> pinned; otherwise derived
        # at the operating point), non-DT operating-temperature brackets, and
        # the per-fuel radiation proxy (pb11/dhe3 default to cc.f_rad_fus,
        # DT/DD to None -> full radiation model).
        # T brackets are concept-specific: tokamak uses _T_BRACKET_DEFAULTS;
        # the mirror bracket lives inside mirror_0d_inverse (_T_BRACKET_MIRROR).
        _0d_concepts = {ConfinementConcept.TOKAMAK, ConfinementConcept.MIRROR}
        if (use_0d or params.get("size_from_power", False)) and (
            self.concept in _0d_concepts
        ):
            if self.fuel != Fuel.DT:
                params["Z_eff"] = z_eff_fuel(
                    self.fuel, params["dhe3_fuel_ratio"], params["pb11_fuel_ratio"]
                ) + (params["Z_eff"] - 1.0)
                if self.concept == ConfinementConcept.TOKAMAK:
                    if "T_min" not in overrides:
                        params["T_min"] = _T_BRACKET_DEFAULTS[self.fuel][0]
                    if "T_max" not in overrides:
                        params["T_max"] = _T_BRACKET_DEFAULTS[self.fuel][1]
            params["dhe3_dd_frac_pin"] = overrides.get("dhe3_dd_frac")
            if "f_rad_fus" not in params:
                params["f_rad_fus"] = self.cc.f_rad_fus(self.fuel)

        # Layer 2: Power balance (dispatched by family), or sizing solve.
        solved_n_mod = None
        if params.get("size_from_power", False):
            if self.concept == ConfinementConcept.TOKAMAK:
                pinned = [
                    k for k in ("R0", "plasma_t", "b_center", "B") if k in overrides
                ]
                if pinned:
                    raise ValueError(
                        f"{pinned} cannot be pinned in size_from_power mode; they are "
                        "solved. Remove them or set size_from_power=False."
                    )
                if params.get("optimize_lcoe", False):
                    # Re-run the full sizing+cost pipeline per trial f_GW. A
                    # low-density (low f_GW) trial may not reach the power target
                    # within R0_max; treat that as a worst-case point (+inf) so
                    # the golden-section search steers toward the feasible,
                    # higher-density region instead of crashing on it.
                    def _lcoe_at(fgw):
                        try:
                            return self.forward(
                                net_electric_mw=net_electric_mw,
                                availability=availability,
                                lifetime_yr=lifetime_yr,
                                n_mod=n_mod,
                                construction_time_yr=construction_time_yr,
                                interest_rate=interest_rate,
                                inflation_rate=inflation_rate,
                                noak=noak,
                                cost_overrides=cost_overrides,
                                override_reference_mw=override_reference_mw,
                                **{
                                    **overrides,
                                    "size_from_power": True,
                                    "optimize_lcoe": False,
                                    "f_GW": fgw,
                                },
                            ).costs.lcoe
                        except SizingInfeasible:
                            return float("inf")

                    best_fgw = self._optimize_gss(
                        _lcoe_at, params["f_GW_min"], params["f_GW_max"]
                    )
                    self._sizing_fgw = best_fgw
                    return self.forward(
                        net_electric_mw=net_electric_mw,
                        availability=availability,
                        lifetime_yr=lifetime_yr,
                        n_mod=n_mod,
                        construction_time_yr=construction_time_yr,
                        interest_rate=interest_rate,
                        inflation_rate=inflation_rate,
                        noak=noak,
                        cost_overrides=cost_overrides,
                        override_reference_mw=override_reference_mw,
                        **{
                            **overrides,
                            "size_from_power": True,
                            "optimize_lcoe": False,
                            "f_GW": best_fgw,
                        },
                    )
                self._sizing_fgw = params["f_GW"]
                pt = self._size_tokamak(params, n_mod)
            elif self.concept == ConfinementConcept.MIRROR:
                if "chamber_length" in overrides:
                    raise ValueError(
                        "['chamber_length'] cannot be pinned in size_from_power mode; "
                        "it is solved. Remove it or set size_from_power=False."
                    )
                if params.get("optimize_lcoe", False):
                    # Outer GSS over f_beta minimizing LCOE.
                    # Solve-cost profile: _GSS_OPT_ITERS (12) outer f_beta
                    # evaluations, each triggering a full mirror sizing run
                    # (_L_BISECT_ITERS=60 bisection steps x _GSS_ITERS=40 T_i
                    # GSS steps), yielding roughly 30k eager forward/power-balance
                    # calls per optimize_lcoe run.  Tune _GSS_OPT_ITERS (outer)
                    # and the sizing knobs in mirror.py to trade off precision
                    # against wall-clock time.
                    def _lcoe_at_fb(fb):
                        # Two exception types signal an infeasible f_beta:
                        #   OperatingPointInfeasible — raised by mirror_0d_inverse
                        #     when the wall-cap density violates beta_max.
                        #   SizingInfeasible — raised by mirror_size_from_power
                        #     (via _size_mirror) when the wall cap prevents the
                        #     machine from reaching the power target at any L.
                        # Return a large sentinel LCOE so GSS steers away.
                        try:
                            return self.forward(
                                net_electric_mw=net_electric_mw,
                                availability=availability,
                                lifetime_yr=lifetime_yr,
                                n_mod=n_mod,
                                construction_time_yr=construction_time_yr,
                                interest_rate=interest_rate,
                                inflation_rate=inflation_rate,
                                noak=noak,
                                cost_overrides=cost_overrides,
                                override_reference_mw=override_reference_mw,
                                **{
                                    **overrides,
                                    "size_from_power": True,
                                    "optimize_lcoe": False,
                                    "f_beta": fb,
                                },
                            ).costs.lcoe
                        except (OperatingPointInfeasible, SizingInfeasible):
                            # infeasible f_beta (beta gate or sizing cap)
                            return 1e8  # steer GSS away

                    best_fb = self._optimize_gss(
                        _lcoe_at_fb, params["f_beta_min"], params["f_beta_max"]
                    )
                    self._sizing_fbeta = best_fb
                    return self.forward(
                        net_electric_mw=net_electric_mw,
                        availability=availability,
                        lifetime_yr=lifetime_yr,
                        n_mod=n_mod,
                        construction_time_yr=construction_time_yr,
                        interest_rate=interest_rate,
                        inflation_rate=inflation_rate,
                        noak=noak,
                        cost_overrides=cost_overrides,
                        override_reference_mw=override_reference_mw,
                        **{
                            **overrides,
                            "size_from_power": True,
                            "optimize_lcoe": False,
                            "f_beta": best_fb,
                        },
                    )
                self._sizing_fbeta = params["f_beta"]
                pt = self._size_mirror(params, n_mod)
            elif self.concept in N_MOD_SIZED_CONCEPTS:
                if params.get("optimize_lcoe", False):
                    raise ValueError(
                        "optimize_lcoe is not applicable to module-replication "
                        "concepts; there is no continuous design variable to "
                        "optimize (n_mod is a discrete count)."
                    )
                pt = self._size_modular(params, n_mod)
                # The solved module count becomes the effective n_mod for all
                # downstream cost aggregation.
                n_mod = self._last_n_mod
                solved_n_mod = n_mod
            else:
                supported = (
                    "TOKAMAK, MIRROR (geometry); "
                    + ", ".join(sorted(c.value for c in N_MOD_SIZED_CONCEPTS))
                    + " (module count)"
                )
                raise ValueError(
                    f"size_from_power is implemented for {supported}; "
                    f"got concept={self.concept.value}"
                )
        else:
            pt = self._power_balance(params, n_mod)

        # Layer 3: Geometry (radial build -> component volumes)
        from dataclasses import fields as dc_fields

        rb_field_names = {f.name for f in dc_fields(RadialBuild)}
        rb_params = {k: params[k] for k in rb_field_names if k in params}
        rb = RadialBuild(**rb_params)
        geo = compute_geometry(rb, self.concept)

        # Combined volumes for CAS22 accounts
        blanket_vol = geo.firstwall_vol + geo.blanket_vol + geo.reflector_vol
        shield_vol = geo.ht_shield_vol + geo.lt_shield_vol
        structure_vol = geo.structure_vol
        vessel_vol = geo.vessel_vol

        # Layer 4: Cost accounts
        # Reconstruct CC from params (which may contain JAX tracers
        # from sensitivity analysis) so gradients flow through.
        cc_kwargs = {k: params[k] for k in cc_float_fields() if k in params}
        cc_kwargs["building_costs"] = self.cc.building_costs
        cc_kwargs["replaceable_accounts"] = self.cc.replaceable_accounts
        cc_kwargs["coil_markup"] = self.cc.coil_markup
        cc_kwargs["cas27_fill_materials"] = self.cc.cas27_fill_materials
        cc = CostingConstants(**cc_kwargs)
        co = cost_overrides or {}
        overridden = []

        # Compute defaults, apply CAS-level overrides
        c10 = co.get(
            "CAS10", cas10_preconstruction(cc, pt.p_net, n_mod, self.fuel, noak)
        )
        if "CAS10" in co:
            overridden.append("CAS10")

        # coil_material drives the cryogenics building (magnet cryoplant) in
        # CAS21 and the coil cost in CAS22. Every confinement-magnet concept
        # declares it in YAML; COPPER is an inert sentinel for magnet-free
        # concepts, whose _COIL_DEFAULTS is None so the material is never read
        # for coil cost (and, being normal-conducting, carries no cryoplant).
        coil_material = CoilMaterial(params.get("coil_material", "copper"))

        c21 = co.get(
            "CAS21",
            cas21_buildings(
                cc,
                pt.p_et,
                pt.p_the,
                pt.p_th,
                pt.p_fus,
                n_mod,
                self.fuel,
                noak,
                coil_material,
            ),
        )
        if "CAS21" in co:
            overridden.append("CAS21")

        # CAS22: compute detail, apply sub-account overrides, recompute totals
        # Coil parameters: every confinement-magnet concept declares b_center /
        # r_bore in its concept YAML (the source of truth). The 0.0 fallback is
        # a "no field" sentinel for magnet-free concepts (IFE drivers, MIF), whose
        # _COIL_DEFAULTS is None so the coil model never reads these. A nonzero
        # code default here would silently impose a tokamak-class field on any
        # concept that omits the parameter.
        # r_bore = loop radius for LINEAR/loop devices (mirror, FRC, dipole,
        # pulsed), which use the r^2 coil model. For TOROIDAL devices (tokamak,
        # stellarator) the coil model is bilinear in R0 and the coil-bore radius
        # (= geo.vessel_or, passed below), and r_bore is unused there.
        # b_center = field at the center of the loop (axis), NOT peak-on-conductor.
        r_bore = params.get("r_bore", 0.0)
        # A tokamak's coil-cost center field IS the plasma field B (same physical
        # on-axis toroidal field). Derive it here, at the point of consumption,
        # rather than storing a second b_center: that keeps it from drifting from
        # B and from becoming a spurious free parameter in sensitivity. Mirrors
        # and other loop devices keep an explicit b_center (central vs plug field
        # genuinely differ).
        if self.concept == ConfinementConcept.TOKAMAK:
            b_center = params["B"]
        else:
            b_center = params.get("b_center", 0.0)
        n_coils = params.get("n_coils", None)
        blanket_form = BlanketForm(params["blanket_form"])
        blanket_fill = BlanketFill(params["blanket_fill"])

        # Heating mix: use explicit breakdown if provided, else default
        # all p_input to NBI (backward-compatible).
        p_input = params.get("p_input", params.get("p_driver", 0.0))
        p_nbi = params.get("p_nbi", p_input)
        p_ecrh = params.get("p_ecrh", 0.0)
        p_icrf = params.get("p_icrf", 0.0)
        p_lhcd = params.get("p_lhcd", 0.0)
        # If any explicit heating breakdown is provided, don't auto-fill p_nbi
        if any(k in overrides for k in ("p_nbi", "p_ecrh", "p_icrf", "p_lhcd")):
            p_nbi = params.get("p_nbi", 0.0)

        # Integrity by construction: the heating split is what C220104 costs
        # ($/MW per heating type), and p_input is what the power balance uses
        # (q_sci = p_fus/p_input, recirc = p_input/eta_pin). They must reference
        # the SAME injected MW. The split defines the MIX; we normalize its total
        # to p_input so the costed heating always equals the power-balance
        # heating, even when p_input is overridden without the split. A fully
        # zero split (e.g. electrostatic orbitron/polywell, where input power is
        # not NBI/RF heating) stays zero, so its NBI/RF capital is correctly $0.
        # Concrete values only; under the uncertainty/AD path these are JAX
        # tracers and the default p_nbi=p_input already tracks p_input.
        _split_vals = (p_nbi, p_ecrh, p_icrf, p_lhcd, p_input)
        if all(isinstance(v, numbers.Real) for v in _split_vals):
            heat_split = p_nbi + p_ecrh + p_icrf + p_lhcd
            if heat_split > 1e-12:
                k = p_input / heat_split
                p_nbi, p_ecrh, p_icrf, p_lhcd = (
                    p_nbi * k,
                    p_ecrh * k,
                    p_icrf * k,
                    p_lhcd * k,
                )

        # C220104 pulsed-driver inputs. p_driver (avg power) costs mechanical
        # injectors; pt.e_driver_mj (per-pulse energy) costs lasers/accelerators on
        # a rep-rate-independent $/J basis; e_preheat_mj costs the laser preheat
        # add-on (0 unless the concept sets it).
        if self.family == ConfinementFamily.PULSED:
            p_driver = pt.e_driver_mj * params["f_rep"]
            e_preheat_mj = params.get("e_preheat_mj", 0.0)
            # Structural: a pulsed concept manufactures a consumed target iff its
            # per-shot cost is positive (laser/heavy-ion capsule, MagLIF/Z-pinch
            # liner). In-situ formation concepts set 0. Read the merged params
            # value so a forward() override turns the target factory on/off in
            # lockstep with the CAS80 consumable (e.g. a pellet-fed MTF setting
            # target_unit_cost > 0). target_unit_cost is not a differentiable
            # key, so this stays a concrete bool under jax.grad/vmap.
            manufactured_target = params["target_unit_cost"] > 0.0
        else:
            p_driver = 0.0
            e_preheat_mj = 0.0
            manufactured_target = False  # steady-state always uses a divertor

        c22_detail = cas22_reactor_plant_equipment(
            cc,
            pt.p_net,
            pt.p_th,
            pt.p_et,
            pt.p_fus,
            params["p_cryo"],
            n_mod,
            self.fuel,
            noak,
            blanket_vol=blanket_vol,
            shield_vol=shield_vol,
            structure_vol=structure_vol,
            vessel_vol=vessel_vol,
            family=self.family,
            concept=self.concept,
            manufactured_target=manufactured_target,
            f_rep=params.get("f_rep") or 0.0,
            target_factory_capex_fixed=params.get("target_factory_capex_fixed") or 0.0,
            target_factory_capex_per_hz=params.get("target_factory_capex_per_hz")
            or 0.0,
            target_factory_capex_per_gwfus=(
                params.get("target_factory_capex_per_gwfus") or 0.0
            ),
            b_center=b_center,
            r_bore=r_bore,
            R0=params["R0"],
            r_coil=geo.vessel_or,
            coil_material=coil_material,
            blanket_form=blanket_form,
            blanket_fill=blanket_fill,
            n_coils=n_coils,
            lev_coil_markup=params.get("lev_coil_markup"),
            lev_coil_cryostat_cost=params.get("lev_coil_cryostat_cost"),
            stationary_lift_coil_fraction=params.get(
                "stationary_lift_coil_fraction", 0.10
            ),
            # Mirror two-class coil params (MIRROR only; ignored for other concepts
            # because the two-class branch in cas22 is guarded by concept==MIRROR).
            chamber_length=params.get("chamber_length", 0.0),
            coil_spacing=params.get("coil_spacing", 0.0),
            n_plug_coils=int(params.get("n_plug_coils", 0)),
            R_m=params.get("R_m", 1.0),
            B=params.get("B", 0.0),
            r_bore_central=(
                geo.vessel_or + params["coil_standoff"]
                if self.concept == ConfinementConcept.MIRROR
                else 0.0
            ),
            r_bore_plug=(
                params["plasma_t"] / math.sqrt(params["R_m"]) + params["plug_standoff"]
                if self.concept == ConfinementConcept.MIRROR
                else 0.0
            ),
            p_nbi=p_nbi,
            p_ecrh=p_ecrh,
            p_icrf=p_icrf,
            p_lhcd=p_lhcd,
            p_driver=p_driver,
            e_driver_mj=pt.e_driver_mj,
            e_preheat_mj=e_preheat_mj,
            laser_driver_type=self.laser_driver_type,
            f_dec=params.get("f_dec", 0.0),
            p_dee=pt.p_dee,
            burn_fraction=params["burn_fraction"],
            vac_op_pressure_pa=params["vac_op_pressure_pa"],
            # Pulsed DEC params
            pulsed_conversion=self.pulsed_conversion,
            e_stored_mj=getattr(pt, "e_stored_mj", 0.0),
            q_sci=pt.q_sci,
            f_ch=getattr(pt, "f_ch", 0.0),
            eta_dec=params.get("eta_dec", 0.0),
        )
        # Informational sub-lines in cas22_detail (C220103_central/_plug,
        # C220106_vessel/_pump, r_bore_*) are deliberately absent from these
        # key sets; the parent account carries the aggregated total.
        _PER_MODULE_KEYS = {
            "C220101",
            "C220102",
            "C220103",
            "C220104",
            "C220105",
            "C220106",
            "C220107",
            "C220108",
            "C220109",
            "C220110",
            "C220111",
            "C220112",
        }
        _PLANT_WIDE_KEYS = {
            "C220200",
            "C220300",
            "C220400",
            "C220500",
            "C220600",
            "C220700",
        }
        for key in co:
            if key in c22_detail and key != "C220000":
                c22_detail[key] = co[key]
                overridden.append(key)
        if any(k in co for k in (_PER_MODULE_KEYS | _PLANT_WIDE_KEYS)):
            # C220111 (labor) gets the multi-unit discount; equipment keys do not.
            equipment_keys = _PER_MODULE_KEYS - {"C220111"}
            per_module_equipment = sum(c22_detail[k] for k in equipment_keys)
            labor = c22_detail["C220111"] * (
                1.0 + (n_mod - 1) * cc.multi_unit_labor_factor
            )
            plant_wide = sum(c22_detail[k] for k in _PLANT_WIDE_KEYS)
            c22_detail["C220000"] = per_module_equipment * n_mod + labor + plant_wide

        c22 = co.get("CAS22", c22_detail["C220000"])
        if "CAS22" in co:
            overridden.append("CAS22")
            # Scale sub-account detail proportionally so downstream
            # consumers (e.g. CAS72 scheduled replacement) reflect the
            # override.  Without this, zeroing CAS22 still leaves
            # non-zero sub-accounts that produce phantom replacement costs.
            computed = c22_detail["C220000"]
            scale = c22 / computed if computed > 0 else 0.0
            for k in c22_detail:
                c22_detail[k] = c22_detail[k] * scale

        c23 = co.get("CAS23", cas23_turbine(cc, pt.p_the, n_mod))
        if "CAS23" in co:
            overridden.append("CAS23")

        c24 = co.get("CAS24", cas24_electrical(cc, pt.p_et, n_mod))
        if "CAS24" in co:
            overridden.append("CAS24")

        c25 = co.get("CAS25", cas25_misc(cc, pt.p_et, n_mod))
        if "CAS25" in co:
            overridden.append("CAS25")

        c26 = co.get("CAS26", cas26_heat_rejection(cc, pt.p_th, n_mod))
        if "CAS26" in co:
            overridden.append("CAS26")

        c27 = co.get(
            "CAS27",
            cas27_special_materials(cc, blanket_fill, blanket_vol),
        )
        if "CAS27" in co:
            overridden.append("CAS27")

        c28 = co.get("CAS28", cas28_digital_twin(cc))
        if "CAS28" in co:
            overridden.append("CAS28")

        cas2x_pre_contingency = c21 + c22 + c23 + c24 + c25 + c26 + c27 + c28
        c29 = cas29_contingency(cc, cas2x_pre_contingency, noak)
        c20 = cas2x_pre_contingency + c29
        c30 = cas30_indirect(cc, c20, construction_time_yr)
        c40 = cas40_owner(cc, self.fuel, pt.p_net)
        c50 = cas50_supplementary(
            cc, self.fuel, c20, c23 + c24 + c25 + c26 + c27 + c28, c30, pt.p_net, noak
        )
        overnight_cost = c10 + c20 + c30 + c40 + c50
        c60 = cas60_idc(interest_rate, overnight_cost, construction_time_yr)
        total_capital = overnight_cost + c60

        # Layer 5: Economics
        # Apply disruption penalty for tokamak 0D/sizing runs only.
        # MirrorPlasmaState has no disruption_rate field; guard by concept so
        # a mirror 0D run never reaches the tokamak-specific penalty path.
        # Steady-state MFE families derive core lifetime from the neutron wall
        # loading (fluence basis); IFE/MIF keep the fixed per-fuel constant.
        # pt.p_neutron and geo.firstwall_area are both per-module, so q_n needs
        # no n_mod scaling.
        if self.family == ConfinementFamily.STEADY_STATE:
            # jnp.maximum on the area is a defensive guard against any concept
            # reporting firstwall_area=0 (none currently do; all steady-state
            # concepts get a strictly positive area in geometry.py). Note that
            # a near-zero area would drive q_n up, shortening lifetime -- the
            # opposite of clamping. Aneutronic steady-state fuels (p-B11,
            # D-He3) clamp to plant life via vanishing p_neutron (q_n -> 0),
            # not via the area.
            q_n = pt.p_neutron / jnp.maximum(geo.firstwall_area, 1e-6)
            core_lt = _core_lifetime_fpy(cc, self.fuel, q_n, lifetime_yr, availability)
        else:
            core_lt = cc.core_lifetime(self.fuel)
        avail_eff = availability
        if (
            self._plasma_state is not None
            and self.concept == ConfinementConcept.TOKAMAK
        ):
            dm = DisruptionModel(
                rate_base=params["disruption_rate_base"],
                steepness=params["disruption_steepness"],
                damage_per_disruption=params["disruption_damage"],
                downtime_per_disruption=params["disruption_downtime"],
            )
            core_lt, avail_eff = apply_disruption_penalty(
                core_lt,
                availability,
                self._plasma_state.disruption_rate,
                dm,
            )

        c90 = cas90_financial(total_capital, interest_rate, lifetime_yr)
        # For IFE/MIF, C220108 is the target factory (capital equipment),
        # not the divertor — it does not need periodic replacement.
        repl_accounts = cc.replaceable_accounts
        if self.family != ConfinementFamily.STEADY_STATE:
            repl_accounts = tuple(a for a in repl_accounts if a != "C220108")

        c70, c71, c72 = cas70_om(
            cc,
            cas22_detail=c22_detail,
            replaceable_accounts=repl_accounts,
            n_mod=n_mod,
            p_net=pt.p_net,
            availability=avail_eff,
            inflation_rate=inflation_rate,
            interest_rate=interest_rate,
            lifetime_yr=lifetime_yr,
            core_lifetime=core_lt,
            construction_time=construction_time_yr,
            fuel=self.fuel,
            noak=noak,
            p_dee=pt.p_dee,
            pulsed_conversion=self.pulsed_conversion,
            f_rep=params.get("f_rep", 0.0),
            concept=self.concept,
            laser_driver_type=self.laser_driver_type,
        )
        # Per-shot target consumable (IFE/MIF): one fabricated target/liner per
        # shot per module. Structural zero for steady-state (continuous
        # operation, no shots), mirroring p_driver above.
        if self.family == ConfinementFamily.PULSED:
            target_unit_cost = params["target_unit_cost"]
            n_targets_per_year = params["f_rep"] * 3600.0 * 8760.0 * avail_eff
        else:
            target_unit_cost = 0.0
            n_targets_per_year = 0.0
        # The fuel bill must price the same plasma as the power partition:
        # 0D/sized D-He3 runs use the kernel's effective side-channel fraction
        # (derived or pinned); non-0D runs keep the YAML value, matching their
        # partition.
        dhe3_frac_cost = params["dhe3_dd_frac"]
        if self.fuel == Fuel.DHE3 and self._plasma_state is not None:
            dhe3_frac_cost = self._plasma_state.dhe3_dd_frac_eff
        c80 = cas80_fuel(
            cc,
            pt.p_fus,
            n_mod,
            avail_eff,
            inflation_rate,
            interest_rate,
            lifetime_yr,
            construction_time_yr,
            self.fuel,
            noak,
            dd_f_T=params["dd_f_T"],
            dd_f_He3=params["dd_f_He3"],
            dhe3_dd_frac=dhe3_frac_cost,
            dhe3_f_T=params["dhe3_f_T"],
            dhe3_f_He3=self._dhe3_f_He3_eff(params),
            burn_fraction=params["burn_fraction"],
            fuel_recovery=params["fuel_recovery"],
            target_unit_cost=target_unit_cost,
            n_targets_per_year=n_targets_per_year,
        )
        lcoe = compute_lcoe(c90, c70, c80, pt.p_net, n_mod, avail_eff)
        capital_per_kw = total_capital * 1e6 / (pt.p_net * n_mod * 1e3)  # $/kW

        costs = CostResult(
            cas10=c10,
            cas21=c21,
            cas22=c22,
            cas23=c23,
            cas24=c24,
            cas25=c25,
            cas26=c26,
            cas27=c27,
            cas28=c28,
            cas29=c29,
            cas20=c20,
            cas30=c30,
            cas40=c40,
            cas50=c50,
            cas60=c60,
            cas70=c70,
            cas71=c71,
            cas72=c72,
            cas80=c80,
            cas90=c90,
            total_capital=total_capital,
            lcoe=lcoe,
            overnight_cost=overnight_cost,
            capital_per_kw=capital_per_kw,
        )
        return ForwardResult(
            power_table=pt,
            costs=costs,
            params=params,
            overridden=overridden,
            cas22_detail=c22_detail,
            plasma_state=self._plasma_state,
            solved_n_mod=solved_n_mod,
        )

    # Map from cost_overrides keys to CostResult attribute names, used by
    # _scale_overrides() to scale an override by the account's value ratio
    # across plant sizes. Restricted to the direct-capital accounts forward()
    # actually applies (CAS10, CAS21-CAS28); CAS22 sub-accounts are scaled
    # separately via cas22_detail. Downstream accounts (CAS29/30/60 aggregates,
    # CAS70/71/72 O&M, CAS80 fuel, CAS90 financial) are computed, not
    # overridable, so they must not appear here — scaling a value forward()
    # never reads would be dead.
    _OVERRIDE_TO_ATTR = {
        "CAS10": "cas10",
        "CAS21": "cas21",
        "CAS22": "cas22",
        "CAS23": "cas23",
        "CAS24": "cas24",
        "CAS25": "cas25",
        "CAS26": "cas26",
        "CAS27": "cas27",
        "CAS28": "cas28",
    }

    def _scale_overrides(
        self,
        cost_overrides: dict[str, float],
        reference_mw: float,
        target_mw: float,
        **forward_kwargs,
    ) -> dict[str, float]:
        """Scale cost overrides from reference_mw to target_mw.

        Runs the model at both power levels (without overrides) to get
        the computed cost for each overridden account, then applies the
        ratio as a multiplier to the user's override values.

        FR-2 of costingfe-library-preconditions: the reference-side
        forward runs at n_mod=1 so per-module overrides are framed at one
        module at native per-module power (which is the frame the analyst
        writes them in). The target-side forward uses the caller's n_mod
        so plant-aggregate overrides scale to the target plant total.

        CAS22 sub-account overrides (C220101, etc.) are scaled using the
        CAS22 sub-account detail from the reference and target runs.
        """
        ref_result = self.forward(
            net_electric_mw=reference_mw,
            cost_overrides=None,
            **dict(forward_kwargs, n_mod=1),
        )
        target_result = self.forward(
            net_electric_mw=target_mw, cost_overrides=None, **forward_kwargs
        )

        scaled = {}
        for key, value in cost_overrides.items():
            # CAS22 sub-accounts
            if key.startswith("C22") and key in ref_result.cas22_detail:
                ref_val = ref_result.cas22_detail[key]
                tgt_val = target_result.cas22_detail[key]
                is_cas22_subaccount = True
            # Top-level CAS accounts
            elif key in self._OVERRIDE_TO_ATTR:
                attr = self._OVERRIDE_TO_ATTR[key]
                ref_val = getattr(ref_result.costs, attr)
                tgt_val = getattr(target_result.costs, attr)
                is_cas22_subaccount = False
            else:
                # Unknown key: pass through unscaled
                scaled[key] = value
                continue

            if ref_val > 0:
                scaled[key] = value * (tgt_val / ref_val)
            elif is_cas22_subaccount:
                # The library has no value for this CAS22 sub-account in this
                # config (e.g. a driver account that is $0 for this concept),
                # so no per-account ratio is available. The override IS the
                # cost basis, and reactor-island hardware grows with plant
                # size, so scale it linearly with net power rather than
                # freezing it at the reference-power dollars. A genuinely
                # scale-invariant CAS22 line item should use an explicit
                # fixed-scaling hint instead.
                scaled[key] = value * (target_mw / reference_mw)
            else:
                # Top-level account that is zero at both scales (e.g. CAS28
                # digital twin): treated as a fixed/absent cost, so pass
                # through unscaled rather than inflating it with plant size.
                scaled[key] = value

        return scaled

    # Financial parameters — given by cost of capital, not engineering levers
    _FINANCIAL_KEYS = ["interest_rate", "inflation_rate"]

    def _engineering_keys(self) -> list[str]:
        """Return engineering parameter names (things you can actually improve)."""
        common = [
            "availability",
            "construction_time_yr",
            "lifetime_yr",
            "mn",
            "eta_th",
            "eta_p",
            "f_sub",
            "p_pump",
            "p_trit",
            "p_house",
            "p_cryo",
            # Geometry — radial build dimensions
            "blanket_t",
            "ht_shield_t",
            "structure_t",
            "vessel_t",
            "plasma_t",
            # Fuel burn fractions (physics model)
            "burn_fraction",
            "fuel_recovery",
            "dd_f_T",
            "dd_f_He3",
            "dhe3_dd_frac",
            "dhe3_f_T",
            "pb11_f_alpha_n",
            "pb11_f_p_n",
        ]
        family_specific = {
            ConfinementFamily.STEADY_STATE: [
                "p_input",
                "eta_de",
                "f_dec",
                "p_coils",
                "p_cool",
                "R0",
                "elon",  # torus geometry
                "chamber_length",  # mirror cylinder length
                "q95",
                "f_GW",
                "B",
                "T_e",
                # Radiation model parameters
                "n_e",
                "Z_eff",
                "plasma_volume",
                "R_w",
                # Impurity model parameters
                "T_edge",
                "tau_ratio",
                # Magnet costing
                "b_center",
                "r_bore",
                # NOTE: p_nbi/p_ecrh/p_icrf/p_lhcd are deliberately not sliders.
                # forward() renormalizes the heating mix to p_input on concrete
                # values, but the JAX-traced sensitivity path skips that branch,
                # so their elasticity would not match production. The improvable
                # hardware question is captured by eta_source_* (costing category).
                # 0D model / disruption parameters
                "M_ion",
                "lambda_q",
                "disruption_rate_base",
                "disruption_steepness",
                "disruption_damage",
                "disruption_downtime",
            ],
            ConfinementFamily.PULSED: [
                "q_eng",
                "f_rep",
                "eta_pin",
                "f_rad",
                "p_target",
                "p_coils",
                "eta_dec",
                "f_pdv",
                # Per-shot consumed-target cost (CAS80): a dominant IFE/MIF LCOE
                # driver at high rep-rate. Zero for in-situ-formation concepts,
                # which the params[k] != 0 filter excludes automatically.
                "target_unit_cost",
            ],
        }
        keys = common + family_specific.get(self.family, [])
        if self.family == ConfinementFamily.STEADY_STATE:
            # Heated concepts tune eta_couple (eta_pin is derived); electrostatic
            # concepts tune eta_pin directly.
            keys.append(
                "eta_couple" if "eta_couple" in self._eng_defaults else "eta_pin"
            )
        return keys

    # Cross-cutting optional/derived overrides that forward() and cas22 read
    # via params.get() but that a concept's YAML need not declare. forward()
    # validates override kwargs against this set (plus the concept's YAML keys
    # and the costing-constant fields). Keep in sync when a new params.get()
    # input is added; an omission surfaces as a spurious "unknown parameter".
    # Golden-section iterations for LCOE optimization (optimize_lcoe).
    _GSS_OPT_ITERS = 12

    _OPTIONAL_OVERRIDE_KEYS = frozenset(
        {
            # Power-cycle / coupling knobs injected or derived by forward()
            "eta_th",
            "eta_pin",
            "eta_couple",
            "f_rad",
            "f_rad_fus",
            # 0D plasma model (TOKAMAK)
            "use_0d_model",
            "0d_mode",
            "fw_area",
            # Per-concept D-He3 side-channel fraction the 0D path injects into
            # params (None unless overridden); listed so a 0D forward's params
            # round-trip back through forward() (e.g. sensitivity() replay).
            "dhe3_dd_frac_pin",
            # Power-to-geometry sizing (TOKAMAK)
            "size_from_power",
            "optimize_lcoe",
            "aspect_ratio",
            "beta_N_max",
            "H_factor",
            "R0_min",
            "R0_max",
            "T_min",
            "T_max",
            "f_GW_min",
            "f_GW_max",
            # Mirror sizing knobs (not in tokamak YAML)
            "f_beta",
            "L_min",
            "L_max",
            "f_beta_min",
            "f_beta_max",
            # Coil / magnet knobs not in every concept YAML
            "n_coils",
            "lev_coil_markup",
            "lev_coil_cryostat_cost",
            "stationary_lift_coil_fraction",
            # Pulsed driver
            "p_driver",
            "e_preheat_mj",
            # IFE/MIF target factory capital (CAS22.01.08), three-term build-up;
            # in-situ concepts leave them unset (0) and carry no factory.
            "target_factory_capex_fixed",
            "target_factory_capex_per_hz",
            "target_factory_capex_per_gwfus",
        }
    )

    # CostingConstants float fields — cost model calibration parameters
    # Exclude reference/normalization constants that aren't real levers:
    # they exist only to make other parameters dimensionally correct.
    _CC_EXCLUDE = {
        "reference_construction_time",  # normalization for indirect_fraction
    }
    _COSTING_KEYS = set(cc_float_fields()) - _CC_EXCLUDE

    def _continuous_keys(self) -> list[str]:
        """All differentiable continuous parameter names."""
        return (
            self._engineering_keys() + self._FINANCIAL_KEYS + list(self._COSTING_KEYS)
        )

    def _build_lcoe_fn(
        self, params: dict, cost_overrides: dict[str, float] | None = None
    ):
        """Build a JAX-differentiable function: param_vector -> LCOE.

        The param vector includes engineering, financial, AND costing
        constants (all CC float fields are injected into params by
        forward()).  Returns (lcoe_fn, keys, base_values).

        cost_overrides are closed over as constants — gradients through
        overridden accounts are naturally zero.
        """
        keys = [k for k in self._continuous_keys() if k in params and params[k] != 0]
        base_vals = jnp.array([float(params[k]) for k in keys])

        # Named args passed to forward() directly
        named_args = {
            "net_electric_mw",
            "availability",
            "lifetime_yr",
            "n_mod",
            "construction_time_yr",
            "interest_rate",
            "inflation_rate",
            "noak",
            "fuel",
            "concept",
        }

        # Static params (closed over, not traced)
        static_eng = {
            k: v for k, v in params.items() if k not in named_args and k not in keys
        }
        net_mw = params["net_electric_mw"]
        avail = params["availability"]
        life = params["lifetime_yr"]
        n_mod = params.get("n_mod", 1)
        ct = params.get("construction_time_yr", 6.0)
        noak = params.get("noak", True)

        def lcoe_fn(x):
            # Unpack traced params into a dict
            eng = dict(static_eng)
            for i, k in enumerate(keys):
                eng[k] = x[i]

            # Extract named args from traced vector if present
            ir = eng.pop("interest_rate", params.get("interest_rate", 0.07))
            inf = eng.pop("inflation_rate", params.get("inflation_rate", 0.02))
            av = eng.pop("availability", avail)
            ct_val = eng.pop("construction_time_yr", ct)
            lf = eng.pop("lifetime_yr", life)

            result = self.forward(
                net_electric_mw=net_mw,
                availability=av,
                lifetime_yr=lf,
                n_mod=n_mod,
                construction_time_yr=ct_val,
                interest_rate=ir,
                inflation_rate=inf,
                noak=noak,
                cost_overrides=cost_overrides,
                **eng,
            )
            return result.costs.lcoe

        return lcoe_fn, keys, base_vals

    def sensitivity(
        self,
        params: dict,
        cost_overrides: dict[str, float] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Compute elasticity of LCOE w.r.t. each continuous parameter.

        Elasticity = (dLCOE/dp) * (p / LCOE) = %ΔLCOE / %Δparam.
        Dimensionless, allowing fair comparison across parameters.

        Returns {"engineering": {...}, "financial": {...}, "costing": {...}}
        where engineering levers are things you can improve, financial are
        cost-of-capital givens, and costing are CostingConstants calibration
        parameters (unit costs, fractions, base costs).

        Uses jax.grad for exact autodiff gradients, EXCEPT on the 0D /
        power-sizing paths, which solve the operating point with a bisection
        (tokamak_0d_inverse / mirror_0d_inverse). jax.grad cannot see through
        that root-find -- every parameter enters the loop through the
        non-differentiable comparison -- so it disagrees with the concrete
        forward(). Those paths fall back to finite differences on forward().
        """
        if params.get("use_0d_model", False) or params.get("size_from_power", False):
            return self._sensitivity_fd(params, cost_overrides)

        lcoe_fn, keys, base_vals = self._build_lcoe_fn(params, cost_overrides)
        base_lcoe = float(lcoe_fn(base_vals))

        grad_fn = jax.grad(lcoe_fn)
        grads = grad_fn(base_vals)

        engineering = {}
        financial = {}
        costing = {}
        for i, key in enumerate(keys):
            p = float(base_vals[i])
            dLCOE_dp = float(grads[i])
            elasticity = dLCOE_dp * p / base_lcoe
            if key in self._FINANCIAL_KEYS:
                financial[key] = elasticity
            elif key in self._COSTING_KEYS:
                costing[key] = elasticity
            else:
                engineering[key] = elasticity

        return {
            "engineering": engineering,
            "financial": financial,
            "costing": costing,
        }

    def _sensitivity_fd(
        self,
        params: dict,
        cost_overrides: dict[str, float] | None = None,
        h: float = 1e-3,
    ) -> dict[str, dict[str, float]]:
        """Finite-difference elasticities by re-running forward() concretely.

        The fallback for the 0D / sizing paths (see sensitivity()): central
        differences on forward() with plain Python floats reproduce the
        concrete execution the slider runs, including the bisection solve and
        the heating-mix renormalization that JAX tracing skips. Same key
        selection and categorization as the jax.grad path.
        """
        keys = [k for k in self._continuous_keys() if k in params and params[k] != 0]

        # Split params into forward()'s named args and its **overrides exactly
        # as _build_lcoe_fn does, but holding Python floats so forward() takes
        # its concrete path.
        named = {
            "net_electric_mw": float(params["net_electric_mw"]),
            "availability": float(params["availability"]),
            "lifetime_yr": float(params["lifetime_yr"]),
            "n_mod": params.get("n_mod", 1.0),
            "interest_rate": params.get("interest_rate", 0.07),
            "inflation_rate": params.get("inflation_rate", 0.02),
            "noak": params.get("noak", True),
        }
        skip = set(named) | {"fuel", "concept"}
        eng = {k: v for k, v in params.items() if k not in skip}

        def run(key, value):
            n = dict(named)
            e = dict(eng)
            (n if key in n else e)[key] = value
            return float(
                self.forward(cost_overrides=cost_overrides, **n, **e).costs.lcoe
            )

        base_lcoe = float(
            self.forward(cost_overrides=cost_overrides, **named, **eng).costs.lcoe
        )

        engineering: dict[str, float] = {}
        financial: dict[str, float] = {}
        costing: dict[str, float] = {}
        for key in keys:
            p = float(params[key])
            dp = abs(p) * h
            dLCOE_dp = (run(key, p + dp) - run(key, p - dp)) / (2 * dp)
            elasticity = dLCOE_dp * p / base_lcoe
            if key in self._FINANCIAL_KEYS:
                financial[key] = elasticity
            elif key in self._COSTING_KEYS:
                costing[key] = elasticity
            else:
                engineering[key] = elasticity

        return {
            "engineering": engineering,
            "financial": financial,
            "costing": costing,
        }

    def batch_lcoe(
        self,
        param_sets: dict[str, list[float]],
        params: dict,
        cost_overrides: dict[str, float] | None = None,
    ) -> list[float]:
        """Evaluate LCOE for many parameter sets using jax.vmap.

        Args:
            param_sets: Dict of param_name -> list of values (all same length).
                Only the listed params vary; others held at base values.
            params: Base parameter dict (from a forward() result).
            cost_overrides: CAS account overrides (closed over as constants).

        Returns:
            List of LCOE values, one per parameter set.
        """
        lcoe_fn, keys, base_vals = self._build_lcoe_fn(params, cost_overrides)

        n = len(next(iter(param_sets.values())))
        # Build matrix: each row is a param vector
        rows = []
        for _ in range(n):
            rows.append(base_vals)
        batch = jnp.stack(rows)

        # Override the varying params
        for param_name, values in param_sets.items():
            if param_name in keys:
                idx = keys.index(param_name)
                batch = batch.at[:, idx].set(jnp.array(values))

        vmapped = jax.vmap(lcoe_fn)
        results = vmapped(batch)
        return [float(r) for r in results]
