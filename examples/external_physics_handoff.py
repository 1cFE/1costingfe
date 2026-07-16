"""Canonical 1 GWe D-3He steady-state mirror reference for the 1costingfe paper.

This is the example cited in Section 2.2: physics outputs from an external
model (central-cell geometry, field, temperatures, densities, DEC fractions,
secondary burn fractions) hand off to the 1costingfe forward call, which
produces a complete CAS-account rollup and an LCOE figure.

The forward call uses an inverse power-balance solve: given the engineering
parameters (geometry, efficiencies, burn fractions) and the net electric
target, the model finds the required fusion power and derives all costs.

Numbers are illustrative; the point is the handoff pattern, not the design.
"""

from costingfe import ConfinementConcept, CostModel, Fuel, PowerCycle


def main() -> None:
    model = CostModel(
        concept=ConfinementConcept.MIRROR,
        fuel=Fuel.DHE3,
        power_cycle=PowerCycle.RANKINE,
    )

    result = model.forward(
        # Customer parameters
        net_electric_mw=1000.0,
        availability=0.85,  # default plant availability (ARIES heritage)
        lifetime_yr=30,
        construction_time_yr=6.0,
        interest_rate=0.07,
        inflation_rate=0.02,
        noak=True,
        # Geometry (Section 2.2 Table: central-cell length, plasma radius)
        R0=0.0,  # cylinder axis (no major-radius offset)
        chamber_length=80.0,  # m, central-cell length
        plasma_t=0.4,  # m, plasma radius at midplane
        elon=1.0,  # circular cross-section
        blanket_t=0.30,  # m, thin blanket (D-3He: low neutron fluence)
        ht_shield_t=0.20,  # m
        structure_t=0.15,  # m
        vessel_t=0.10,  # m
        # Magnets: the coil-cost field is derived from the central-cell field
        # B (radiation block below): central solenoid priced at B = 3 T, plug
        # at R_m * B = 30 T with the YAML mirror ratio R_m = 10. (This example
        # previously handed off b_center = 12 T alongside B = 3 T, the exact
        # two-field disagreement the derivation now forbids; at this n_e/T_e,
        # 3 T is beta ~ 0.2, mirror-sensible, while 12 T would be beta 0.013.)
        r_bore=1.85,  # m, effective winding bore radius
        # Power balance (Section 2.2 Table)
        p_input=50.0,  # MW, NBI heating (neutral beams sustain mirror)
        eta_couple=1.0,  # NBI coupling; eta_pin = 0.60 x 1.0 = 0.60
        eta_p=0.50,  # pumping efficiency
        p_coils=5.0,  # MW, solenoid + mirror coils
        p_cool=25.0,  # MW, first-wall cooling
        p_pump=1.5,  # MW
        p_house=4.0,  # MW
        p_cryo=1.0,  # MW
        f_sub=0.03,  # BOP subsystem fraction
        # Blanket / neutronics: D-3He is aneutronic primary, no blanket needed
        blanket_form="none",
        blanket_fill="none",
        mn=1.0,  # no neutron multiplication without a breeding blanket
        # Conversion: venetian-blind DEC on end-loss ions + thermal
        # (eta_th supplied by the RANKINE preset on CostModel)
        eta_de=0.70,  # venetian-blind DEC efficiency on end-loss ions
        f_dec=0.90,  # fraction of transport power routed to DEC
        # Plasma parameters for radiation calculation (Section 2.2 Table)
        n_e=3.3e19,  # m^-3, electron density
        T_e=70.0,  # keV, electron temperature
        Z_eff=1.3,  # effective ion charge
        B=3.0,  # T, central-cell field
        T_edge=0.20,  # keV, edge temperature (open field lines)
        tau_ratio=3.0,  # impurity confinement time / energy confinement time
        R_w=0.4,  # wall reflectivity (lower for open ends)
        # D-3He secondary burn fractions (Section 2.2 Table)
        dhe3_f_T=0.5,  # secondary D-T burn fraction
        dhe3_dd_frac=0.131,  # D-D side-reaction fraction
        dd_f_T=0.969,
        dd_f_He3=0.689,
    )

    costs = result.costs
    pt = result.power_table

    # Group-level rollup (matches paper Section 2.2 table)
    # CAS20 = sum of CAS21-29 (direct capital); CAS30 is indirect, shown separately.
    cas23_29 = (
        costs.cas23
        + costs.cas24
        + costs.cas25
        + costs.cas26
        + costs.cas27
        + costs.cas28
        + costs.cas29
    )
    rows = [
        ("CAS10", costs.cas10),
        ("CAS21", costs.cas21),
        ("CAS22", costs.cas22),
        ("CAS23-29", cas23_29),
        ("CAS30", costs.cas30),
        ("CAS40", costs.cas40),
        ("CAS50", costs.cas50),
        ("Overnight (CAS10-50)", costs.overnight_cost),
        ("CAS60", costs.cas60),
        ("Total capital (incl. IDC)", costs.total_capital),
        ("CAS70 (M$/yr)", costs.cas70),
        ("CAS80 (M$/yr)", costs.cas80),
        ("CAS90 (M$/yr)", costs.cas90),
        ("LCOE", costs.lcoe),
    ]
    print("1 GWe D-3He mirror with venetian-blind DEC, NOAK reference\n")
    for label, value in rows:
        print(f"  {label:<25s} {float(value):>10.1f}")
    print()
    print("Power balance:")
    print(f"  P_fus:   {pt.p_fus:.0f} MW")
    print(f"  P_net:   {pt.p_net:.0f} MW")
    print(f"  Q_eng:   {pt.q_eng:.2f}")
    print(f"  Recirc:  {pt.rec_frac:.1%}")
    print()
    print(f"Overnight specific cost: {costs.overnight_cost * 1e3 / pt.p_net:.0f} $/kW")
    print(f"Total specific cost (incl. IDC): {costs.capital_per_kw:.0f} $/kW")
    print(f"LCOE: {costs.lcoe:.1f} $/MWh")


if __name__ == "__main__":
    main()
