"""Top-level CostModel API: wires all 5 layers together."""

import math

import jax
import jax.numpy as jnp

from costingfe.defaults import (
    POWER_CYCLE_DEFAULTS,
    CostingConstants,
    cc_float_fields,
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
from costingfe.layers.physics import (
    ife_forward_power_balance,
    ife_inverse_power_balance,
    mfe_forward_power_balance,
    mfe_inverse_power_balance,
    mif_forward_power_balance,
    mif_inverse_power_balance,
    pulsed_dec_forward,
    pulsed_dec_inverse,
    pulsed_thermal_forward,
    pulsed_thermal_inverse,
)
from costingfe.layers.tokamak import (
    DisruptionModel,
    apply_disruption_penalty,
    derive_radial_build,
    tokamak_0d_forward,
    tokamak_0d_inverse,
)
from costingfe.types import (
    CONCEPT_DEFAULT_CONVERSION,
    CONCEPT_TO_FAMILY,
    CoilMaterial,
    ConfinementConcept,
    ConfinementFamily,
    CostResult,
    ForwardResult,
    Fuel,
    PowerCycle,
    PulsedConversion,
    WallMaterial,
)
from costingfe.validation import CostingInput


class CostModel:
    def __init__(
        self,
        concept: ConfinementConcept,
        fuel: Fuel,
        costing_constants: CostingConstants = None,
        power_cycle: PowerCycle = PowerCycle.RANKINE,
        pulsed_conversion: PulsedConversion = None,
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

    def _power_balance(self, params, n_mod):
        """Dispatch power balance based on confinement family."""
        p_net_per_mod = params["net_electric_mw"] / n_mod

        # 0D tokamak branch
        use_0d = params.get("use_0d_model", False)
        if use_0d and self.concept == ConfinementConcept.TOKAMAK:
            return self._power_balance_0d(params, n_mod)

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
            )

            fuel_frac_kw = dict(
                dd_f_T=params["dd_f_T"],
                dd_f_He3=params["dd_f_He3"],
                dhe3_dd_frac=params["dhe3_dd_frac"],
                dhe3_f_T=params["dhe3_f_T"],
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
            # Unified pulsed balance (e_driver_mj present) or legacy dispatch
            if "e_driver_mj" in params and params.get("e_driver_mj"):
                fuel_frac_kw = dict(
                    dd_f_T=params["dd_f_T"],
                    dd_f_He3=params["dd_f_He3"],
                    dhe3_dd_frac=params["dhe3_dd_frac"],
                    dhe3_f_T=params["dhe3_f_T"],
                    pb11_f_alpha_n=params["pb11_f_alpha_n"],
                    pb11_f_p_n=params["pb11_f_p_n"],
                )
                common_kw = dict(
                    fuel=self.fuel,
                    e_driver_mj=params["e_driver_mj"],
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

                if self.pulsed_conversion == PulsedConversion.INDUCTIVE_DEC:
                    dec_kw = dict(
                        eta_dec=params["eta_dec"],
                        f_pdv=params.get("f_pdv", self.cc.f_pdv),
                    )
                    p_fus = pulsed_dec_inverse(
                        p_net_target=p_net_per_mod,
                        **common_kw,
                        **dec_kw,
                    )
                    pt = pulsed_dec_forward(
                        p_fus=p_fus,
                        **common_kw,
                        **dec_kw,
                    )
                else:
                    p_fus = pulsed_thermal_inverse(
                        p_net_target=p_net_per_mod,
                        **common_kw,
                    )
                    pt = pulsed_thermal_forward(
                        p_fus=p_fus,
                        **common_kw,
                    )
            elif "p_driver" in params and params["p_driver"]:
                p_fus = mif_inverse_power_balance(
                    p_net_target=p_net_per_mod,
                    fuel=self.fuel,
                    p_driver=params["p_driver"],
                    mn=params["mn"],
                    eta_th=params["eta_th"],
                    eta_p=params["eta_p"],
                    eta_pin=params["eta_pin"],
                    f_sub=params["f_sub"],
                    p_pump=params["p_pump"],
                    p_trit=params["p_trit"],
                    p_house=params["p_house"],
                    p_cryo=params["p_cryo"],
                    p_target=params["p_target"],
                    p_coils=params.get("p_coils", 0.0),
                )
                pt = mif_forward_power_balance(
                    p_fus=p_fus,
                    fuel=self.fuel,
                    p_driver=params["p_driver"],
                    mn=params["mn"],
                    eta_th=params["eta_th"],
                    eta_p=params["eta_p"],
                    eta_pin=params["eta_pin"],
                    f_sub=params["f_sub"],
                    p_pump=params["p_pump"],
                    p_trit=params["p_trit"],
                    p_house=params["p_house"],
                    p_cryo=params["p_cryo"],
                    p_target=params["p_target"],
                    p_coils=params.get("p_coils", 0.0),
                )
            else:
                p_fus = ife_inverse_power_balance(
                    p_net_target=p_net_per_mod,
                    fuel=self.fuel,
                    p_implosion=params["p_implosion"],
                    p_ignition=params["p_ignition"],
                    mn=params["mn"],
                    eta_th=params["eta_th"],
                    eta_p=params["eta_p"],
                    eta_pin1=params["eta_pin1"],
                    eta_pin2=params["eta_pin2"],
                    f_sub=params["f_sub"],
                    p_pump=params["p_pump"],
                    p_trit=params["p_trit"],
                    p_house=params["p_house"],
                    p_cryo=params["p_cryo"],
                    p_target=params["p_target"],
                )
                pt = ife_forward_power_balance(
                    p_fus=p_fus,
                    fuel=self.fuel,
                    p_implosion=params["p_implosion"],
                    p_ignition=params["p_ignition"],
                    mn=params["mn"],
                    eta_th=params["eta_th"],
                    eta_p=params["eta_p"],
                    eta_pin1=params["eta_pin1"],
                    eta_pin2=params["eta_pin2"],
                    f_sub=params["f_sub"],
                    p_pump=params["p_pump"],
                    p_trit=params["p_trit"],
                    p_house=params["p_house"],
                    p_cryo=params["p_cryo"],
                    p_target=params["p_target"],
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

        fuel_frac_kw = dict(
            dd_f_T=params["dd_f_T"],
            dd_f_He3=params["dd_f_He3"],
            dhe3_dd_frac=params["dhe3_dd_frac"],
            dhe3_f_T=params["dhe3_f_T"],
            pb11_f_alpha_n=params["pb11_f_alpha_n"],
            pb11_f_p_n=params["pb11_f_p_n"],
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
                **fuel_frac_kw,
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
                **fuel_frac_kw,
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
                **fuel_frac_kw,
            )

        self._plasma_state = plasma_state
        return pt

    def forward(
        self,
        net_electric_mw: float,
        availability: float,
        lifetime_yr: float,
        n_mod: int = 1,
        construction_time_yr: float = 6.0,
        interest_rate: float = 0.07,
        inflation_rate: float = 0.02,
        noak: bool = True,
        cost_overrides: dict[str, float] | None = None,
        **overrides,
    ) -> ForwardResult:
        """Forward costing: customer requirements -> LCOE."""
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

        # Layer 2: Power balance (dispatched by family)
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
        cc = CostingConstants(**cc_kwargs)
        co = cost_overrides or {}
        overridden = []

        # Compute defaults, apply CAS-level overrides
        c10 = co.get(
            "CAS10", cas10_preconstruction(cc, pt.p_net, n_mod, self.fuel, noak)
        )
        if "CAS10" in co:
            overridden.append("CAS10")

        c21 = co.get(
            "CAS21",
            cas21_buildings(cc, pt.p_et, pt.p_the, pt.p_th, pt.p_fus, self.fuel, noak),
        )
        if "CAS21" in co:
            overridden.append("CAS21")

        # CAS22: compute detail, apply sub-account overrides, recompute totals
        # Coil parameters: from YAML defaults or user overrides.
        # r_coil = effective winding bore radius (calibration parameter,
        # not necessarily equal to the radial build vessel_or).
        r_coil = params.get("r_coil", 1.85)
        b_max = params.get("b_max", 12.0)
        coil_material = CoilMaterial(params.get("coil_material", "rebco_hts"))

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
            b_max=b_max,
            r_coil=r_coil,
            coil_material=coil_material,
            p_nbi=p_nbi,
            p_ecrh=p_ecrh,
            p_icrf=p_icrf,
            p_lhcd=p_lhcd,
            f_dec=params.get("f_dec", 0.0),
            p_dee=pt.p_dee,
            # Pulsed DEC params
            pulsed_conversion=self.pulsed_conversion,
            e_stored_mj=getattr(pt, "e_stored_mj", 0.0),
            q_sci=pt.q_sci,
            f_ch=getattr(pt, "f_ch", 0.0),
            eta_dec=params.get("eta_dec", 0.0),
        )
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
            per_module = sum(c22_detail[k] for k in _PER_MODULE_KEYS)
            plant_wide = sum(c22_detail[k] for k in _PLANT_WIDE_KEYS)
            c22_detail["C220000"] = per_module * n_mod + plant_wide

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

        c27 = co.get("CAS27", cas27_special_materials(cc, pt.p_net, self.fuel))
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
        # Apply disruption penalty when 0D model is active
        core_lt = cc.core_lifetime(self.fuel)
        avail_eff = availability
        if self._plasma_state is not None:
            dm = DisruptionModel(
                rate_base=params.get("disruption_rate_base", 0.1),
                steepness=params.get("disruption_steepness", 15.0),
                damage_per_disruption=params.get("disruption_damage", 0.02),
                downtime_per_disruption=params.get("disruption_downtime", 72.0),
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
        )
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
            burn_fraction=params.get("burn_fraction"),
            fuel_recovery=params.get("fuel_recovery"),
        )
        lcoe = compute_lcoe(c90, c70, c80, pt.p_net, n_mod, avail_eff)
        overnight = total_capital * 1e6 / (pt.p_net * n_mod * 1e3)  # $/kW

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
            overnight_cost=overnight,
        )
        return ForwardResult(
            power_table=pt,
            costs=costs,
            params=params,
            overridden=overridden,
            cas22_detail=c22_detail,
            plasma_state=self._plasma_state,
        )

    # Financial parameters — given by cost of capital, not engineering levers
    _FINANCIAL_KEYS = ["interest_rate", "inflation_rate"]

    def _engineering_keys(self) -> list[str]:
        """Return engineering parameter names (things you can actually improve)."""
        common = [
            "availability",
            "construction_time_yr",
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
                "eta_pin",
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
                "b_max",
                "r_coil",
                # Heating mix (CAS22 costing)
                "p_nbi",
                "p_ecrh",
                "p_icrf",
                "p_lhcd",
                # 0D model / disruption parameters
                "M_ion",
                "lambda_q",
                "disruption_rate_base",
                "disruption_steepness",
                "disruption_damage",
                "disruption_downtime",
            ],
            ConfinementFamily.PULSED: [
                "e_driver_mj",
                "f_rep",
                "eta_pin",
                "f_rad",
                "p_target",
                "p_coils",
                "eta_dec",
                "f_pdv",
                # Legacy params (kept for backward compatibility)
                "p_implosion",
                "p_ignition",
                "eta_pin1",
                "eta_pin2",
                "p_driver",
            ],
        }
        return common + family_specific.get(self.family, [])

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

            result = self.forward(
                net_electric_mw=net_mw,
                availability=av,
                lifetime_yr=life,
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

        Uses jax.grad for exact autodiff gradients.
        """
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
