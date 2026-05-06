"""Benchmark 1costingfe against ARIES-AT (Najmabadi et al. 2006).

Inputs follow the ARIES-AT reference design: 1 GWe net, R0=5.2 m,
a=1.3 m (A=4.0), kappa=2.2, B0=5.86 T, P_fus=1755 MW, eta_th=0.59,
D-T fuel, LTS Nb3Sn coils, He-cooled SiC composite blanket with
Brayton power cycle.

Inputs that 1costingfe requires but the Najmabadi 2006 paper does not
publish are defaulted; each such default is annotated `# default:`.

Run as a script:
    python docs/papers/1costingfe_paper/scripts/benchmark_aries_at.py
"""

from __future__ import annotations

import json
from pathlib import Path

from costingfe import ConfinementConcept, CostModel, Fuel


def run(output_dir: Path) -> dict:
    """Run the ARIES-AT benchmark; write aries_at.json; return payload."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = {
        # ARIES-AT published geometry and power balance
        "R0": 5.2,
        "elon": 2.2,
        "eta_th": 0.59,
        "net_electric_mw": 1000.0,
        # Framework requirements not directly published, plant-level defaults
        "availability": 0.85,  # default
        "lifetime_yr": 30,  # default
        "n_mod": 1,
        "construction_time_yr": 6.0,  # default
        "interest_rate": 0.07,  # default
        "inflation_rate": 0.0245,  # default
        "noak": True,
        # Geometry components
        "plasma_t": 1.3,
        "blanket_t": 0.5,  # default: SiC blanket thickness
        "ht_shield_t": 0.2,  # default
        "structure_t": 0.2,  # default
        "vessel_t": 0.2,  # default
        # Power balance auxiliaries
        "p_input": 50.0,  # default
        "mn": 1.1,  # default
        "eta_p": 0.5,  # default
        "eta_pin": 0.5,  # default
        "eta_de": 0.85,  # default
        "f_sub": 0.03,  # default
        "f_dec": 0.0,
        "p_coils": 2.0,  # default
        "p_cool": 13.7,  # default
        "p_pump": 1.0,  # default
        "p_trit": 10.0,  # default
        "p_house": 4.0,  # default
        "p_cryo": 0.5,  # default
    }

    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    result = model.forward(**inputs)

    c = result.costs
    pt = result.power_table

    payload = {
        "reactor": "ARIES-AT",
        "fuel": "DT",
        "concept": "TOKAMAK",
        "inputs": inputs,
        "predicted_overnight_musd": float(c.total_capital),
        "predicted_overnight_per_kwe_usd": float(c.overnight_cost),
        "predicted_lcoe_usd_per_mwh": float(c.lcoe),
        "fusion_power_mw": float(pt.p_fus),
        "net_electric_mw": float(pt.p_net),
        "cas": {
            "cas10": float(c.cas10),
            "cas21": float(c.cas21),
            "cas22": float(c.cas22),
            "cas23": float(c.cas23),
            "cas24": float(c.cas24),
            "cas25": float(c.cas25),
            "cas26": float(c.cas26),
            "cas27": float(c.cas27),
            "cas28": float(c.cas28),
            "cas29": float(c.cas29),
            "cas30": float(c.cas30),
            "cas40": float(c.cas40),
            "cas50": float(c.cas50),
            "cas60": float(c.cas60),
            "cas70": float(c.cas70),
            "cas80": float(c.cas80),
            "cas90": float(c.cas90),
        },
        "cas22_detail": {k: float(v) for k, v in result.cas22_detail.items()},
    }

    (output_dir / "aries_at.json").write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    out = Path(__file__).parent / "_outputs"
    payload = run(out)
    print(
        f"ARIES-AT -- overnight: {payload['predicted_overnight_musd']:.0f} M$ (2025); "
        f"{payload['predicted_overnight_per_kwe_usd']:.0f} $/kWe; "
        f"LCOE: {payload['predicted_lcoe_usd_per_mwh']:.1f} $/MWh"
    )
