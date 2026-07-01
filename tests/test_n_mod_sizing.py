"""Tests for simple n_mod-from-power sizing (module-replication concepts).

These concepts scale a plant by replicating a fixed module rather than growing a
single device (tokamak/mirror) or increasing rep-rate/target yield. size_from_power
solves an integer module count from a per-module design net power (module_net_mwe).
"""

import math

import pytest

from costingfe.model import CostModel
from costingfe.types import ConfinementConcept, Fuel

# Feasible (concept, fuel, native module_net_mwe) operating points.
_ORBITRON = (ConfinementConcept.ORBITRON, Fuel.PB11, 0.005)
_DPF = (ConfinementConcept.DENSE_PLASMA_FOCUS, Fuel.PB11, 5.0)
_ZAP = (ConfinementConcept.STAGED_ZPINCH, Fuel.DT, 50.0)  # Zap sheared-flow
_FRC = (ConfinementConcept.STEADY_FRC, Fuel.PB11, 50.0)
_ALL = [_ORBITRON, _DPF, _ZAP, _FRC]


class TestNModSizing:
    def test_solved_n_mod_exact(self):
        # Orbitron 5 kWe cells; a 1 GWe plant is exactly 200,000 modules.
        m = CostModel(concept=_ORBITRON[0], fuel=_ORBITRON[1])
        r = m.forward(
            net_electric_mw=1000.0,
            availability=0.85,
            lifetime_yr=30.0,
            size_from_power=True,
        )
        assert r.solved_n_mod == 200000
        assert isinstance(r.solved_n_mod, int)
        assert r.costs.lcoe > 0

    def test_integer_ceil_and_overshoot(self):
        # DPF native module 5 MWe; target 10.1 -> ceil(10.1/5)=3 modules,
        # realized plant net = 15.0 MWe >= target (fractional modules unbuildable).
        m = CostModel(concept=_DPF[0], fuel=_DPF[1])
        r = m.forward(
            net_electric_mw=10.1,
            availability=0.85,
            lifetime_yr=30.0,
            size_from_power=True,
        )
        assert r.solved_n_mod == 3
        assert float(r.power_table.p_net) * r.solved_n_mod == pytest.approx(
            15.0, rel=1e-6
        )

    def test_per_module_at_design_power(self):
        # Each module runs at its design power; plant net = n_mod * module_net_mwe.
        m = CostModel(concept=_ZAP[0], fuel=_ZAP[1])
        r = m.forward(
            net_electric_mw=300.0,
            availability=0.85,
            lifetime_yr=30.0,
            size_from_power=True,
        )
        assert r.solved_n_mod == 6  # ceil(300/50)
        assert float(r.power_table.p_net) == pytest.approx(_ZAP[2], rel=1e-6)

    def test_sizing_equivalent_to_manual_n_mod(self):
        # The sizing branch must thread the solved n_mod through ALL cost
        # accounts: a sized run equals a manual run at the same module count and
        # per-module power. Guards the n_mod_effective wiring.
        concept, fuel, mod = _ZAP
        sized = CostModel(concept=concept, fuel=fuel).forward(
            net_electric_mw=300.0,
            availability=0.85,
            lifetime_yr=30.0,
            size_from_power=True,
        )
        manual = CostModel(concept=concept, fuel=fuel).forward(
            net_electric_mw=sized.solved_n_mod * mod,
            availability=0.85,
            lifetime_yr=30.0,
            n_mod=sized.solved_n_mod,
        )
        assert float(sized.costs.lcoe) == pytest.approx(
            float(manual.costs.lcoe), rel=1e-9
        )

    def test_cost_scales_with_solved_n_mod(self):
        # Doubling the target doubles the module count and scales total capital
        # up substantially but SUB-linearly: land scales as sqrt(n_mod), labor
        # carries the multi-unit discount, and plant-wide accounts do not double.
        m = CostModel(concept=_ZAP[0], fuel=_ZAP[1])
        lo = m.forward(
            net_electric_mw=300.0,
            availability=0.85,
            lifetime_yr=30.0,
            size_from_power=True,
        )
        hi = m.forward(
            net_electric_mw=600.0,
            availability=0.85,
            lifetime_yr=30.0,
            size_from_power=True,
        )
        assert hi.solved_n_mod == 2 * lo.solved_n_mod
        ratio = float(hi.costs.total_capital) / float(lo.costs.total_capital)
        assert 1.6 < ratio < 2.0  # measured ~1.77: co-siting economies of scale

    def test_forbidden_pin_n_mod(self):
        m = CostModel(concept=_ZAP[0], fuel=_ZAP[1])
        with pytest.raises(ValueError, match="n_mod"):
            m.forward(
                net_electric_mw=300.0,
                availability=0.85,
                lifetime_yr=30.0,
                size_from_power=True,
                n_mod=5,
            )

    def test_optimize_lcoe_rejected(self):
        m = CostModel(concept=_ZAP[0], fuel=_ZAP[1])
        with pytest.raises(ValueError, match="optimize_lcoe"):
            m.forward(
                net_electric_mw=300.0,
                availability=0.85,
                lifetime_yr=30.0,
                optimize_lcoe=True,
            )

    @pytest.mark.parametrize("concept,fuel,mod", _ALL)
    def test_all_n_mod_concepts_size(self, concept, fuel, mod):
        m = CostModel(concept=concept, fuel=fuel)
        target = 7.0 * mod + 0.3 * mod  # non-integer multiple to exercise ceil
        r = m.forward(
            net_electric_mw=target,
            availability=0.85,
            lifetime_yr=30.0,
            size_from_power=True,
        )
        assert r.solved_n_mod == math.ceil(target / mod)
        assert r.solved_n_mod >= 1
        assert float(r.costs.lcoe) > 0
        assert float(r.power_table.p_net) * r.solved_n_mod >= target - 1e-6

    def test_volume_concept_still_sizes_by_geometry(self):
        # Tokamak size_from_power must still solve R0. n_mod is also solved
        # (from the R0_max unit ceiling, same helper module-replication
        # concepts use), but a target well within one unit's capacity solves
        # to a single unit.
        m = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
        r = m.forward(
            net_electric_mw=400.0,
            availability=0.85,
            lifetime_yr=30.0,
            size_from_power=True,
        )
        assert r.solved_n_mod == 1  # fits one unit, geometry-sized
        assert m._last_R0 > 0

    @pytest.mark.parametrize(
        "concept,fuel",
        [
            (ConfinementConcept.POLYWELL, Fuel.PB11),
        ],
    )
    def test_unsupported_concept_still_raises(self, concept, fuel):
        # Polywell (volume, no geometry solver) is still unsupported. PULSED_FRC
        # moved to REP_RATE_SIZED_CONCEPTS (rep-rate power sizing, model.py
        # _size_reprate) and is no longer in this unsupported set.
        m = CostModel(concept=concept, fuel=fuel)
        with pytest.raises(ValueError, match="size_from_power"):
            m.forward(
                net_electric_mw=400.0,
                availability=0.85,
                lifetime_yr=30.0,
                size_from_power=True,
            )
