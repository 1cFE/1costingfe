"""Top-level CostModel API: wires all 5 layers together."""

from costingfe.defaults import (
    CostingConstants,
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

        # Layer 4: Cost accounts
        cc = self.cc
        c10 = cas10_preconstruction(cc, pt.p_net, n_mod, self.fuel, noak)
        c21 = cas21_buildings(cc, pt.p_et, self.fuel, noak)
        c22_detail = cas22_reactor_plant_equipment(
            cc, pt.p_net, pt.p_th, pt.p_et, pt.p_fus,
            params["p_cryo"], n_mod, self.fuel, noak,
        )
        c22 = c22_detail["C220000"]
        c23 = cas23_turbine(cc, pt.p_et, n_mod)
        c24 = cas24_electrical(cc, pt.p_et, n_mod)
        c25 = cas25_misc(cc, pt.p_et, n_mod)
        c26 = cas26_heat_rejection(cc, pt.p_et, n_mod)
        c27 = 0.0  # TODO: special materials (needs blanket details)
        c28 = cas28_digital_twin(cc)
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
        return ForwardResult(power_table=pt, costs=costs, params=params)

    def sensitivity(self, params: dict) -> dict[str, float]:
        """Compute d(LCOE)/d(param) for all continuous parameters.

        Uses finite-difference for MVP. Will be replaced with jax.grad
        when the pipeline is fully JAX-traced (Task 14).
        """
        # Family-specific continuous keys
        common_keys = [
            "mn", "eta_th", "eta_p", "f_sub",
            "p_pump", "p_trit", "p_house", "p_cryo",
            "interest_rate", "inflation_rate",
        ]
        family_keys = {
            ConfinementFamily.MFE: [
                "p_input", "eta_pin", "eta_de", "f_dec",
                "p_coils", "p_cool",
            ],
            ConfinementFamily.IFE: [
                "p_implosion", "p_ignition", "eta_pin1", "eta_pin2",
                "p_target",
            ],
            ConfinementFamily.MIF: [
                "p_driver", "eta_pin", "p_target", "p_coils",
            ],
        }
        continuous_keys = common_keys + family_keys.get(self.family, [])

        # Params that are passed as named forward() args, not **overrides
        named_args = {
            "net_electric_mw", "availability", "lifetime_yr", "n_mod",
            "construction_time_yr", "interest_rate", "inflation_rate",
            "noak", "fuel", "concept",
        }

        def _run(override_key=None, override_val=None):
            eng = {k: v for k, v in params.items() if k not in named_args}
            if override_key and override_key in eng:
                eng[override_key] = override_val
            return self.forward(
                net_electric_mw=params["net_electric_mw"],
                availability=params["availability"],
                lifetime_yr=params["lifetime_yr"],
                n_mod=params.get("n_mod", 1),
                construction_time_yr=params.get("construction_time_yr", 6.0),
                interest_rate=(
                    override_val
                    if override_key == "interest_rate"
                    else params.get("interest_rate", 0.07)
                ),
                inflation_rate=(
                    override_val
                    if override_key == "inflation_rate"
                    else params.get("inflation_rate", 0.0245)
                ),
                noak=params.get("noak", True),
                **eng,
            )

        base_lcoe = _run().costs.lcoe
        sensitivities = {}

        for key in continuous_keys:
            if key not in params:
                continue
            base_val = params[key]
            if base_val == 0:
                continue
            delta = abs(base_val) * 0.01
            result_plus = _run(key, base_val + delta)
            sensitivities[key] = (result_plus.costs.lcoe - base_lcoe) / delta

        return sensitivities
