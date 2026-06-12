"""Example: Axisymmetric Mirror with 0D plasma model.

Demonstrates the 0D mirror physics layer, which derives fusion power,
confinement times, ambipolar potential, and beta from machine parameters
rather than asserting p_net and working backwards.

Five sections:
  1. Forward mode: given geometry and T_i, what comes out?
  2. Inverse mode: target 30 MWe net, find the required T_i.
  3. Gate refusal: a configuration hitting beta > beta_max.
  4. Sizing: solve chamber length from net power target.
  5. Fuel comparison: D-T vs D-He3 at the same power target.

D-He3 is the physically interesting mirror fuel: the hot-ion regime that
mirrors can sustain makes the D-He3 reactivity peak more accessible, and
the entirely charged-particle primary products are ideal for end-loss
direct energy conversion. The tradeoff is that D-He3 requires a larger
machine at comparable net power.

Mirror confinement note: at mirror-relevant parameters the gas-dynamic
and Pastukhov times are milliseconds to seconds (depending on regime).
The end-loss power reported by MirrorPlasmaState is the rate at which
stored thermal energy drains from the plasma, which is large when tau_E
is small. The plant power balance (mfe_forward/inverse_power_balance) is
separate: it uses f_dec * p_transport where p_transport = p_ash +
p_input_eff - p_rad, so the LCOE answer does not depend directly on the
MirrorPlasmaState end-loss fields.
"""

from costingfe import ConfinementConcept, CostModel, Fuel
from costingfe.layers.mirror import (
    OperatingPointInfeasible,
    SizingInfeasible,
    mirror_0d_forward,
)

# ── Machine definition ────────────────────────────────────────────────
# WHAM/Realta-class tandem mirror geometry.
# B = B_min (midplane field); R_m = 10 sets the throat field at 30 T.
MACHINE = dict(
    plasma_t=0.4,  # Plasma radius at midplane [m]
    chamber_length=50.0,  # Central cell length [m]
    B=3.0,  # Midplane field B_min [T]
    R_m=10.0,  # Mirror ratio B_throat / B_min [-]
    n_e=1.0e20,  # Electron density [m^-3]
    T_e=20.0,  # Electron temperature [keV]
    # Power balance engineering
    p_input=20.0,  # NBI heating [MW]
    p_nbi=20.0,  # NBI power [MW] (must match p_input)
    eta_couple=0.8333,  # NBI coupling (eta_pin = eta_source x eta_couple)
    mn=1.1,
    eta_th=0.46,
    eta_p=0.5,
    eta_de=0.60,  # DEC efficiency on axial end-loss ions
    f_sub=0.03,
    # f_dec is a YAML input (engineering judgment), not derived from physics:
    f_dec=0.30,
    p_coils=2.0,
    p_cool=5.0,
    p_pump=1.0,
    p_trit=5.0,
    p_house=2.0,
    p_cryo=0.5,
    # Blanket geometry
    blanket_t=0.80,
    ht_shield_t=0.20,
    structure_t=0.15,
    vessel_t=0.10,
    # Mirror stability limit (MHD; user constraint)
    beta_max=0.5,
)

model_dt = CostModel(concept=ConfinementConcept.MIRROR, fuel=Fuel.DT)

# Fuel-fraction kwargs used by mirror_0d_forward
_FRACS = dict(
    dd_f_T=0.969,
    dd_f_He3=0.689,
    dhe3_f_T=0.5,
    dhe3_f_He3=0.05,
    pb11_f_alpha_n=0.0,
    pb11_f_p_n=0.0,
)

# ── Forward mode: specify T_i = 20 keV, see what the machine produces ──
print("=" * 64)
print("  FORWARD MODE: T_i = 20 keV (specified)")
print("=" * 64)

ps = mirror_0d_forward(
    L=MACHINE["chamber_length"],
    a=MACHINE["plasma_t"],
    B_min=MACHINE["B"],
    R_m=MACHINE["R_m"],
    T_i=20.0,
    T_e=MACHINE["T_e"],
    n_e=MACHINE["n_e"],
    p_input=MACHINE["p_input"],
    fuel=Fuel.DT,
    **_FRACS,
    dhe3_fuel_ratio=1.0,
    pb11_fuel_ratio=0.15,
    dhe3_dd_frac_pin=None,
    vacuum_t=MACHINE.get("vacuum_t", 0.10),
)

print("\nPlasma State")
print(f"  T_i:          {float(ps.T_i):8.1f} keV")
print(f"  T_e:          {float(ps.T_e):8.1f} keV")
print(f"  n_e:          {float(ps.n_e):8.2e} m^-3")
print(f"  beta:         {float(ps.beta):8.3f}")
print(f"  phi (e*phi):  {float(ps.phi):8.1f} keV")

print("\nConfinement Times (plasma physics diagnostics)")
print(f"  tau_Pastukhov:{float(ps.tau_Pastukhov):8.3f} s  (electrostatic plugging)")
print(f"  tau_GD:       {float(ps.tau_GD) * 1e3:8.3f} ms (gas-dynamic)")
# At this parameter set, collisionality << 1: collisionless regime.
# tau_GD < tau_Pastukhov so gas-dynamic losses dominate.
print(f"  tau_p:        {float(ps.tau_p) * 1e3:8.3f} ms (combined; tau_GD-dominated)")
print(f"  tau_E:        {float(ps.tau_E) * 1e3:8.3f} ms (energy; < tau_p, ambipolar)")
tau_cl = float(ps.tau_classical)
print(f"  tau_classical:{tau_cl:8.3f} s  (classical mirror, diagnostic)")
coll = float(ps.collisionality)
print(f"  collisionality:{coll:.4f}   L / mean-free-path (<<1: collisionless)")

print("\nFusion Power")
print(f"  P_fus:         {float(ps.p_fus):7.0f} MW")
print(f"  P_alpha:       {float(ps.p_alpha):7.0f} MW  (charged-particle fraction)")
print(f"  P_rad:         {float(ps.p_rad):7.0f} MW")

print("\nGeometry and Wall Loading")
print(f"  V_plasma:      {float(ps.V_plasma):7.0f} m^3")
print(f"  FW area:       {float(ps.fw_area):7.0f} m^2")
print(f"  Wall loading:  {float(ps.wall_loading):7.2f} MW/m^2")
print(f"  R_m:           {float(ps.R_m):7.1f}")
f_ax = float(ps.f_axial_derived)
print(f"  f_axial_diag:  {f_ax:7.3f}   (axial end-loss share, diagnostic only)")

# ── Inverse mode: target 30 MWe net ──────────────────────────────────
# In inverse mode CostModel dispatches to mirror_0d_inverse, which bisects
# on T_i to match the required fusion power from the energy balance.
# T_e is held fixed; only T_i is found by the solver.
print()
print("=" * 64)
print("  INVERSE MODE: 30 MWe target -> find required T_i")
print("=" * 64)

r_inv = model_dt.forward(
    net_electric_mw=30.0,
    availability=0.85,
    lifetime_yr=30,
    use_0d_model=True,
    **MACHINE,
)

ps_inv = model_dt._plasma_state
pt_inv = r_inv.power_table

print("\nSolved Plasma State")
print(f"  T_i:          {float(ps_inv.T_i):8.1f} keV  (bisected to match target)")
bm = MACHINE["beta_max"]
print(f"  beta:         {float(ps_inv.beta):8.3f}  (must be < beta_max = {bm})")
print(f"  tau_Pastukhov:{float(ps_inv.tau_Pastukhov):8.3f} s")
print(f"  tau_GD:       {float(ps_inv.tau_GD) * 1e3:8.3f} ms")
print(f"  phi (e*phi):  {float(ps_inv.phi):8.1f} keV")
print(f"  P_fus:        {float(ps_inv.p_fus):8.0f} MW")
print(f"  P_net:        {float(pt_inv.p_net):8.0f} MW  (target: 30 MW)")
print(f"  Q_sci:        {float(pt_inv.q_sci):8.1f}")
print(f"  Q_eng:        {float(pt_inv.q_eng):8.1f}")

print("\nCost")
print(f"  LCOE:         {float(r_inv.costs.lcoe):8.1f} $/MWh")
print(f"  Overnight:    {float(r_inv.costs.overnight_cost):8.0f} $/kW")

# ── Gate refusal: beta > beta_max ─────────────────────────────────────
# A very compact machine (plasma_t = 0.15 m, chamber_length = 5 m) at high
# density would need implausibly high beta to reach 30 MWe, exceeding
# beta_max. The model refuses rather than returning a cost for a plasma that
# cannot exist. The enforce_plasma_limits flag disables the check for
# exploratory runs.
print()
print("=" * 64)
print("  GATE REFUSAL: compact machine, beta > beta_max")
print("=" * 64)

MACHINE_SMALL = dict(MACHINE, plasma_t=0.15, chamber_length=5.0, n_e=5.0e20)
try:
    model_dt.forward(
        net_electric_mw=30.0,
        availability=0.85,
        lifetime_yr=30,
        use_0d_model=True,
        **MACHINE_SMALL,
    )
except OperatingPointInfeasible as exc:
    print(f"\n  Compact machine (a=0.15 m, L=5 m):\n  {exc}")

# Same 30 MWe target passes at the reference machine (already shown above).
beta_ref = float(ps_inv.beta)
print(f"\n  Reference machine (a=0.4 m, L=50 m): beta = {beta_ref:.3f}")
print(f"  below beta_max = {bm}, gate passes.")

# ── Sizing: solve chamber length from net power target ────────────────
# size_from_power bisects on chamber_length to match the net power target.
# Density is derived from the beta boundary rather than specified directly.
# The operating T_i is found by golden-section search, maximizing net power.
print()
print("=" * 64)
print("  SIZING: solve chamber length from net power target")
print("=" * 64)

# Strip the pinned chamber_length; size_from_power solves it.
SIZING_MACHINE = {k: v for k, v in MACHINE.items() if k != "chamber_length"}

r_sz = model_dt.forward(
    net_electric_mw=200.0,
    availability=0.85,
    lifetime_yr=30,
    size_from_power=True,
    f_beta=0.85,
    L_min=1.0,
    L_max=200.0,
    **SIZING_MACHINE,
)

L_dt = model_dt._last_L
ps_sz = model_dt._plasma_state
pt_sz = r_sz.power_table

print("\n  D-T mirror, 200 MWe target:")
print(f"  Solved L:     {L_dt:8.1f} m")
print(f"  T_i (GSS):    {float(ps_sz.T_i):8.1f} keV")
print(f"  beta:         {float(ps_sz.beta):8.3f}")
print(f"  n_e:          {float(ps_sz.n_e):8.2e} m^-3")
print(f"  P_fus:        {float(ps_sz.p_fus):8.0f} MW")
print(f"  P_net:        {float(pt_sz.p_net):8.0f} MW")
print(f"  LCOE:         {float(r_sz.costs.lcoe):8.1f} $/MWh")
print(f"  Overnight:    {float(r_sz.costs.overnight_cost):8.0f} $/kW")

# ── Fuel comparison: D-T vs D-He3 at 200 MWe ─────────────────────────
# D-He3 is the flagship mirror fuel. All primary D-He3 fusion products are
# charged (proton + He4), ideal for end-loss DEC. However, D-He3 requires
# much higher temperatures (peak reactivity near 200 keV vs. 65 keV for D-T)
# and a larger machine at equal net power. The fuel-appropriate build uses
# a larger plasma radius and field, consistent with the D-He3 hot-ion regime.
# At 200 MWe the YAML default L_max = 200 m is not sufficient; 500 m is used.
print()
print("=" * 64)
print("  FUEL COMPARISON: D-T vs D-He3 (200 MWe, sizing)")
print("=" * 64)

# D-He3 build: higher f_dec (charged products; excellent for DEC),
# higher eta_de, no tritium processing, higher electron temperature,
# larger plasma radius and field (needed for D-He3 energy density).
DHE3_MACHINE = dict(
    SIZING_MACHINE,
    plasma_t=1.5,  # Larger plasma radius [m]
    B=5.0,  # Higher midplane field [T]
    T_e=70.0,  # Hot electron temperature [keV]
    p_trit=0.0,  # No tritium processing for D-He3
    f_dec=0.60,  # Higher DEC fraction (mostly charged products)
    eta_de=0.80,  # Higher DEC efficiency (charged-particle spectrum)
)

fuel_configs = [
    (Fuel.DT, "D-T", SIZING_MACHINE, 200.0),
    (Fuel.DHE3, "D-He3", DHE3_MACHINE, 500.0),
]

header = f"  {'Fuel':<8} {'L [m]':>8} {'T_i [keV]':>10} {'beta':>8} {'LCOE':>14}"
print(f"\n{header}")
print("  " + "-" * 52)

for fuel, label, machine_kw, l_max in fuel_configs:
    m = CostModel(concept=ConfinementConcept.MIRROR, fuel=fuel)
    try:
        r = m.forward(
            net_electric_mw=200.0,
            availability=0.85,
            lifetime_yr=30,
            size_from_power=True,
            f_beta=0.85,
            L_min=1.0,
            L_max=l_max,
            **machine_kw,
        )
        L_val = m._last_L
        ps_val = m._plasma_state
        lcoe_val = float(r.costs.lcoe)
        beta_val = float(ps_val.beta)
        ti_val = float(ps_val.T_i)
        row = (
            f"  {label:<8} {L_val:>8.1f} {ti_val:>10.1f}"
            f" {beta_val:>8.3f} {lcoe_val:>14.1f}"
        )
        print(row)
    except SizingInfeasible as exc:
        row = f"  {label:<8} {'--':>8} {'--':>10} {'--':>8} {'INFEASIBLE':>14}  {exc}"
        print(row)

print()
print("  Key D-He3 physics points:")
print("  - No neutrons from D + He3 -> p + He4; very low activation")
print("  - All primary products charged: ideal for end-loss DEC")
print("  - D-D side reactions produce some neutrons")
print("  - Higher T_i raises tau_ii and Pastukhov confinement time")
print("  - Size ordering vs. D-T depends on the power-balance build: with")
print("    heavy DEC recovery (f_dec=0.6, eta_de=0.85) and no blanket, the")
print("    aneutronic plant sizes smaller here despite lower reactivity;")
print("    at a matched thermal-cycle build D-He3 sizes longer than D-T")
