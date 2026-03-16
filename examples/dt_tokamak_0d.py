"""Example: DT Tokamak with 0D plasma model.

Demonstrates the 0D plasma physics layer, which derives fusion power,
plasma current, confinement time, and stability limits from machine
parameters — rather than asserting p_net and working backwards.

Two modes are shown:
  1. Inverse: "I want 1 GWe net — what plasma does this machine produce?"
  2. Forward: "Given these machine params and T_e, what comes out?"
"""

from costingfe import ConfinementConcept, CostModel, Fuel
from costingfe.layers.tokamak import (
    apply_disruption_penalty,
    check_plasma_limits,
)

# ── Machine definition ────────────────────────────────────────────────
# CATF-like spherical tokamak geometry
MACHINE = dict(
    R0=3.0,  # Major radius R [m]
    plasma_t=1.1,  # Minor radius a [m]
    elon=3.0,  # Elongation kappa
    B=5.0,  # Toroidal field on axis [T]
    q95=3.5,  # Safety factor
    f_GW=0.85,  # Greenwald fraction
    M_ion=2.5,  # Average ion mass [AMU]
    lambda_q=0.002,  # SOL power width [m]
    # Power balance engineering
    p_input=50.0,
    mn=1.1,
    eta_th=0.46,
    eta_p=0.5,
    eta_pin=0.5,
    eta_de=0.85,
    f_sub=0.03,
    f_dec=0.0,
    p_coils=2.0,
    p_cool=13.7,
    p_pump=1.0,
    p_trit=10.0,
    p_house=4.0,
    p_cryo=0.5,
)

model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)

# ── Inverse mode: target 1 GWe net ───────────────────────────────────
print("=" * 64)
print("  INVERSE MODE: 1 GWe target -> find required T_e")
print("=" * 64)

r_inv = model.forward(
    net_electric_mw=1000.0,
    availability=0.85,
    lifetime_yr=30,
    use_0d_model=True,
    **MACHINE,
)

ps = r_inv.plasma_state
pt = r_inv.power_table

print("\nPlasma State")
print(f"  I_p:          {float(ps.I_p):8.1f} MA")
print(f"  n_e:          {float(ps.n_e):8.2e} m^-3")
print(f"  T_e:          {float(ps.T_e):8.1f} keV")
print(f"  beta_N:       {float(ps.beta_N):8.2f} %*m*T/MA")
print(f"  tau_E:        {float(ps.tau_E):8.2f} s")
print(f"  H_factor:     {float(ps.H_factor):8.2f}")
print(f"  f_GW:         {float(ps.f_GW):8.2f}")
print(f"  q95:          {float(ps.q95):8.1f}")

print("\nPower")
print(f"  P_fus:        {float(ps.p_fus):8.0f} MW")
print(f"  P_alpha:      {float(ps.p_alpha):8.0f} MW")
print(f"  P_rad:        {float(ps.p_rad):8.0f} MW")
print(f"  P_net:        {float(pt.p_net):8.0f} MW")
print(f"  Q_sci:        {float(pt.q_sci):8.1f}")
print(f"  Q_eng:        {float(pt.q_eng):8.1f}")

print("\nEngineering Limits")
print(f"  Wall loading: {float(ps.wall_loading):8.2f} MW/m^2")
print(f"  Div heat flux:{float(ps.div_heat_flux):8.1f} MW/m^2")
print(f"  V_plasma:     {float(ps.V_plasma):8.0f} m^3")
print(f"  FW area:      {float(ps.fw_area):8.0f} m^2")

# Physics limit check
issues = check_plasma_limits(ps)
if issues:
    print("\nPhysics Limit Warnings/Errors:")
    for severity, msg in issues:
        print(f"  [{severity.upper()}] {msg}")
else:
    print("\n  All physics limits satisfied.")

print("\nCost")
print(f"  LCOE:         {float(r_inv.costs.lcoe):8.1f} $/MWh")
print(f"  Overnight:    {float(r_inv.costs.overnight_cost):8.0f} $/kW")

# ── Forward mode: specify T_e, see what comes out ─────────────────────
print()
print("=" * 64)
print("  FORWARD MODE: specify T_e=15 keV, see what the machine produces")
print("=" * 64)

r_fwd = model.forward(
    net_electric_mw=1000.0,
    availability=0.85,
    lifetime_yr=30,
    use_0d_model=True,
    **{"0d_mode": "forward"},
    T_e=15.0,
    **MACHINE,
)

ps_fwd = r_fwd.plasma_state
pt_fwd = r_fwd.power_table

print("\n  T_e = 15.0 keV (specified)")
print(f"  P_fus:        {float(ps_fwd.p_fus):8.0f} MW")
print(f"  P_net:        {float(pt_fwd.p_net):8.0f} MW")
print(f"  beta_N:       {float(ps_fwd.beta_N):8.2f} %*m*T/MA")
print(f"  Wall loading: {float(ps_fwd.wall_loading):8.2f} MW/m^2")
print(f"  LCOE:         {float(r_fwd.costs.lcoe):8.1f} $/MWh")

# ── Compare with energy-balance-only mode ─────────────────────────────
print()
print("=" * 64)
print("  COMPARISON: same target, no 0D model (energy balance only)")
print("=" * 64)

r_base = model.forward(
    net_electric_mw=1000.0,
    availability=0.85,
    lifetime_yr=30,
    **MACHINE,
)

print(f"\n  P_fus:        {float(r_base.power_table.p_fus):8.0f} MW")
print(f"  P_net:        {float(r_base.power_table.p_net):8.0f} MW")
print(f"  LCOE:         {float(r_base.costs.lcoe):8.1f} $/MWh")
print(f"  Plasma state: {r_base.plasma_state}")
print("\n  The energy-balance mode does not check whether the machine")
print("  can physically produce the required fusion power.")

# ── Disruption penalty comparison (forward mode) ──────────────────
# Forward mode lets us control T_e directly, so we can isolate
# the effect of f_GW and q95 on disruption rate without the solver
# compensating by pushing T_e (and beta_N) to extreme values.
print()
print("=" * 64)
print("  DISRUPTION PENALTY: safe vs aggressive (forward mode, T_e=15)")
print("=" * 64)

operating_points = [
    ("Safe", dict(f_GW=0.70, q95=4.0)),
    ("Moderate", dict(f_GW=0.85, q95=3.5)),
    ("Aggressive", dict(f_GW=0.95, q95=2.5)),
]

for label, overrides in operating_points:
    machine = dict(MACHINE, **overrides)
    r = model.forward(
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
        use_0d_model=True,
        **{"0d_mode": "forward"},
        T_e=15.0,
        **machine,
    )
    ps_op = r.plasma_state
    dr = float(ps_op.disruption_rate)
    lt, av = apply_disruption_penalty(5.0, 0.85, dr)
    print(f"\n  {label} (f_GW={overrides['f_GW']}, q95={overrides['q95']})")
    print(f"    beta_N:           {float(ps_op.beta_N):.2f}")
    print(f"    Disruption rate:  {dr:.4f} /FPY")
    print(f"    Eff. lifetime:    {float(lt):.3f} FPY  (from 5.0)")
    print(f"    Eff. availability:{float(av):.4f}  (from 0.85)")
    print(f"    LCOE:             {float(r.costs.lcoe):.1f} $/MWh")
