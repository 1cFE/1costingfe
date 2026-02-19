"""Top-level CostModel API: wires all 5 layers together."""

import jax
import jax.numpy as jnp

from costingfe.defaults import (
    CostingConstants,
    load_costing_constants,
    load_engineering_defaults,
)
from costingfe.layers.cas22 import cas22_reactor_plant_equipment
from costingfe.layers.geometry import RadialBuild, compute_geometry
from costingfe.layers.costs import (
    cas10_preconstruction,
    cas21_buildings,
    cas23_turbine,
    cas24_electrical,
    cas25_misc,
    cas26_heat_rejection,
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
from costingfe.layers.physics import (
    ife_forward_power_balance,
    ife_inverse_power_balance,
    mfe_forward_power_balance,
    mfe_inverse_power_balance,
    mif_forward_power_balance,
    mif_inverse_power_balance,
)
from costingfe.types import (
    CONCEPT_TO_FAMILY,
    ConfinementConcept,
    ConfinementFamily,
    CostResult,
    ForwardResult,
    Fuel,
)


class CostModel:
    def __init__(
        self,
        concept: ConfinementConcept,
        fuel: Fuel,
        costing_constants: CostingConstants = None,
    ):
        self.concept = concept
        self.fuel = fuel
        self.family = CONCEPT_TO_FAMILY[concept]
        self.cc = costing_constants or load_costing_constants()
        self._eng_defaults = load_engineering_defaults(
            f"{self.family.value}_{concept.value}"
        )

    def _power_balance(self, params, n_mod):
        """Dispatch power balance based on confinement family."""
        p_net_per_mod = params["net_electric_mw"] / n_mod

        if self.family == ConfinementFamily.MFE:
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
            )

        elif self.family == ConfinementFamily.IFE:
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

        elif self.family == ConfinementFamily.MIF:
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
            raise ValueError(f"Unknown confinement family: {self.family}")

        return pt

    def forward(
        self,
        net_electric_mw: float,
        availability: float,
        lifetime_yr: float,
        n_mod: int = 1,
        construction_time_yr: float = 6.0,
        interest_rate: float = 0.07,
        inflation_rate: float = 0.0245,
        noak: bool = True,
        cost_overrides: dict[str, float] | None = None,
        **overrides,
    ) -> ForwardResult:
        """Forward costing: customer requirements -> LCOE."""
        # Merge defaults with overrides
        params = dict(self._eng_defaults)
        params.update(overrides)
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
        cc = self.cc
        co = cost_overrides or {}
        overridden = []

        # Compute defaults, apply CAS-level overrides
        c10 = co.get("CAS10", cas10_preconstruction(cc, pt.p_net, n_mod, self.fuel, noak))
        if "CAS10" in co:
            overridden.append("CAS10")

        c21 = co.get("CAS21", cas21_buildings(cc, pt.p_et, self.fuel, noak))
        if "CAS21" in co:
            overridden.append("CAS21")

        # CAS22: compute detail, apply sub-account overrides, recompute totals
        c22_detail = cas22_reactor_plant_equipment(
            cc, pt.p_net, pt.p_th, pt.p_et, pt.p_fus,
            params["p_cryo"], n_mod, self.fuel, noak,
            blanket_vol=blanket_vol,
            shield_vol=shield_vol,
            structure_vol=structure_vol,
            vessel_vol=vessel_vol,
        )
        _PER_MODULE_KEYS = {
            "C220101", "C220102", "C220103", "C220104", "C220105",
            "C220106", "C220107", "C220108", "C220109", "C220111",
            "C220112", "C220119",
        }
        _PLANT_WIDE_KEYS = {
            "C220200", "C220300", "C220400", "C220500", "C220600", "C220700",
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

        c23 = co.get("CAS23", cas23_turbine(cc, pt.p_et, n_mod))
        if "CAS23" in co:
            overridden.append("CAS23")

        c24 = co.get("CAS24", cas24_electrical(cc, pt.p_et, n_mod))
        if "CAS24" in co:
            overridden.append("CAS24")

        c25 = co.get("CAS25", cas25_misc(cc, pt.p_et, n_mod))
        if "CAS25" in co:
            overridden.append("CAS25")

        c26 = co.get("CAS26", cas26_heat_rejection(cc, pt.p_et, n_mod))
        if "CAS26" in co:
            overridden.append("CAS26")

        c27 = co.get("CAS27", 0.0)  # TODO: special materials
        if "CAS27" in co:
            overridden.append("CAS27")

        c28 = co.get("CAS28", cas28_digital_twin(cc))
        if "CAS28" in co:
            overridden.append("CAS28")

        cas2x_pre_contingency = c21 + c22 + c23 + c24 + c25 + c26 + c27 + c28
        c29 = cas29_contingency(cc, cas2x_pre_contingency, noak)
        c20 = cas2x_pre_contingency + c29
        c30 = cas30_indirect(cc, c20, pt.p_net, construction_time_yr)
        c40 = cas40_owner(c20)
        c50 = cas50_supplementary(
            cc, c23 + c24 + c25 + c26 + c27 + c28, pt.p_net, noak
        )
        c60 = cas60_idc(cc, c20, pt.p_net, construction_time_yr, self.fuel, noak)
        total_capital = c10 + c20 + c30 + c40 + c50 + c60

        # Layer 5: Economics
        c90 = cas90_financial(
            cc, total_capital, interest_rate, lifetime_yr,
            construction_time_yr, self.fuel, noak,
        )
        c70 = cas70_om(
            cc, pt.p_net, inflation_rate,
            construction_time_yr, self.fuel, noak,
        )
        c80 = cas80_fuel(
            cc, pt.p_fus, n_mod, availability, inflation_rate,
            construction_time_yr, self.fuel, noak,
        )
        lcoe = compute_lcoe(c90, c70, c80, pt.p_net, n_mod, availability)
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
            cas80=c80,
            cas90=c90,
            total_capital=total_capital,
            lcoe=lcoe,
            overnight_cost=overnight,
        )
        return ForwardResult(
            power_table=pt, costs=costs, params=params,
            overridden=overridden,
            cas22_detail=c22_detail,
        )

    # Financial parameters — given by cost of capital, not engineering levers
    _FINANCIAL_KEYS = ["interest_rate", "inflation_rate"]

    def _engineering_keys(self) -> list[str]:
        """Return engineering parameter names (things you can actually improve)."""
        common = [
            "availability", "construction_time_yr",
            "mn", "eta_th", "eta_p", "f_sub",
            "p_pump", "p_trit", "p_house", "p_cryo",
            # Geometry — radial build dimensions
            "blanket_t", "ht_shield_t", "structure_t", "vessel_t",
            "plasma_t",
        ]
        family_specific = {
            ConfinementFamily.MFE: [
                "p_input", "eta_pin", "eta_de", "f_dec",
                "p_coils", "p_cool",
                "axis_t", "elon",  # torus geometry
            ],
            ConfinementFamily.IFE: [
                "p_implosion", "p_ignition", "eta_pin1", "eta_pin2",
                "p_target",
            ],
            ConfinementFamily.MIF: [
                "p_driver", "eta_pin", "p_target", "p_coils",
            ],
        }
        return common + family_specific.get(self.family, [])

    def _continuous_keys(self) -> list[str]:
        """All differentiable continuous parameter names (engineering + financial)."""
        return self._engineering_keys() + self._FINANCIAL_KEYS

    def _build_lcoe_fn(self, params: dict, cost_overrides: dict[str, float] | None = None):
        """Build a JAX-differentiable function: param_vector -> LCOE.

        Fuel, concept, CostingConstants, and discrete params are closed
        over. Only continuous engineering/financial params are traced.
        Returns (lcoe_fn, keys, base_values) where lcoe_fn takes a 1D
        JAX array and returns a scalar LCOE.

        cost_overrides are closed over as constants — gradients through
        overridden accounts are naturally zero.
        """
        keys = [k for k in self._continuous_keys() if k in params and params[k] != 0]
        base_vals = jnp.array([float(params[k]) for k in keys])

        # Named args passed to forward() directly
        named_args = {
            "net_electric_mw", "availability", "lifetime_yr", "n_mod",
            "construction_time_yr", "interest_rate", "inflation_rate",
            "noak", "fuel", "concept",
        }

        # Static params (closed over, not traced)
        static_eng = {k: v for k, v in params.items()
                      if k not in named_args and k not in keys}
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
            inf = eng.pop("inflation_rate", params.get("inflation_rate", 0.0245))
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
        self, params: dict, cost_overrides: dict[str, float] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Compute elasticity of LCOE w.r.t. each continuous parameter.

        Elasticity = (dLCOE/dp) * (p / LCOE) = %ΔLCOE / %Δparam.
        Dimensionless, allowing fair comparison across parameters.

        Returns {"engineering": {...}, "financial": {...}} so that
        engineering levers (things you can improve) are separated from
        financial givens (cost of capital).

        Uses jax.grad for exact autodiff gradients.
        cost_overrides are closed over as constants — overridden accounts
        have zero gradient automatically.
        """
        lcoe_fn, keys, base_vals = self._build_lcoe_fn(params, cost_overrides)
        base_lcoe = float(lcoe_fn(base_vals))

        grad_fn = jax.grad(lcoe_fn)
        grads = grad_fn(base_vals)

        engineering = {}
        financial = {}
        for i, key in enumerate(keys):
            p = float(base_vals[i])
            dLCOE_dp = float(grads[i])
            elasticity = dLCOE_dp * p / base_lcoe
            if key in self._FINANCIAL_KEYS:
                financial[key] = elasticity
            else:
                engineering[key] = elasticity

        return {"engineering": engineering, "financial": financial}

    def batch_lcoe(
        self, param_sets: dict[str, list[float]], params: dict,
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
