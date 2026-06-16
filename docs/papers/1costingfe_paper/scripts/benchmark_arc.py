"""Benchmark 1costingfe against ARC (Sorbom et al. 2015).

Inputs follow the ARC reference design: 270 MWe net, R0=3.3 m,
a=1.13 m, kappa=1.84, B0=9.2 T, P_fus=525 MW, eta_th=0.40, D-T fuel,
demountable REBCO HTS coils.

Inputs that 1costingfe requires but Sorbom 2015 does not publish are
defaulted to the framework's tokamak defaults; each such default is
annotated `# default:` in line.

Run as a script:
    python docs/papers/1costingfe_paper/scripts/benchmark_arc.py
"""

from __future__ import annotations

import json
from pathlib import Path

from costingfe import ConfinementConcept, CostModel, Fuel


def run(output_dir: Path) -> dict:
    """Run the ARC benchmark; write arc.json into output_dir; return payload."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = {
        # ARC published geometry and power balance (Sorbom et al. 2015, Table 1)
        "R0": 3.3,
        "elon": 1.84,
        # On-axis field from the same source as the geometry (NOT the YAML
        # default). For a tokamak the coil-cost center field is derived from B,
        # so setting B here drives both the physics and the bilinear coil cost
        # from one self-consistent design point.
        "B": 9.2,  # B0 on-axis [T]; Sorbom et al. 2015, Table 1
        "p_input": 25.0,  # default: ICRF auxiliary heating
        "eta_th": 0.40,
        # Framework requirements not directly published for ARC
        "net_electric_mw": 270.0,
        "availability": 0.85,  # default
        "lifetime_yr": 30,  # default
        "n_mod": 1,
        "construction_time_yr": 6.0,  # default
        "interest_rate": 0.07,  # default
        "inflation_rate": 0.0245,  # default
        "noak": True,
        # ARC uses a FLiBe liquid-immersion blanket (Sorbom et al. 2015), not the
        # generic tokamak PbLi default — material drives the volume-based CAS27.
        "blanket_form": "molten_salt",
        "blanket_fill": "flibe",
        # Geometry components (defaults; ARC paper does not enumerate)
        "plasma_t": 1.13,
        "blanket_t": 0.5,  # ARC FLiBe blanket nominal thickness
        "ht_shield_t": 0.2,  # default
        "structure_t": 0.2,  # default
        "vessel_t": 0.2,  # default
        # Power balance auxiliaries (defaults)
        "mn": 1.1,  # default
        "eta_p": 0.5,  # default
        "eta_couple": 0.8333,  # NBI coupling; eta_pin = 0.60 x 0.8333 = 0.50
        "eta_de": 0.85,  # default
        "f_sub": 0.03,  # default
        "f_dec": 0.0,
        "p_coils": 1.0,  # default: HTS coils, low cryogenic load
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
        "reactor": "ARC",
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

    (output_dir / "arc.json").write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    out = Path(__file__).parent / "_outputs"
    payload = run(out)
    print(
        f"ARC -- overnight: {payload['predicted_overnight_musd']:.0f} M$ (2025); "
        f"{payload['predicted_overnight_per_kwe_usd']:.0f} $/kWe; "
        f"LCOE: {payload['predicted_lcoe_usd_per_mwh']:.1f} $/MWh"
    )
