"""Canonical 1000 MWe tokamak default: solve the operating point and print it.

The steady_state_tokamak.yaml default is the self-consistent machine that
size_from_power lands on for a 1000 MWe net D-T tokamak with REBCO coils
(B_max 23 T) at the boundary operating point (Greenwald-fraction density,
beta-limited temperature). This script is the source of truth for those
numbers: run it and copy the printed YAML block into the concept default
(and the numpy explorer's copy of it).
"""

from costingfe import ConfinementConcept, CostModel, Fuel

CUSTOMER = dict(availability=0.85, lifetime_yr=30)

model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
r = model.forward(net_electric_mw=1000.0, size_from_power=True, **CUSTOMER)

ps = r.plasma_state
R0 = float(model._last_R0)
B0 = float(model._last_B0)
a = R0 / 3.0  # aspect_ratio = 3.0 (conventional)

print("=" * 60)
print("  Solved 1000 MWe D-T tokamak (REBCO, H=1.0, A=3.0)")
print("=" * 60)
print(f"  R0           = {R0:.3f} m")
print(f"  a (plasma_t) = {a:.3f} m")
print(f"  B  (on-axis) = {B0:.3f} T   (b_center derived from this)")
print(f"  T_e          = {float(ps.T_e):.2f} keV")
print(f"  n_e          = {float(ps.n_e):.4e} m^-3")
print(f"  plasma_vol   = {float(ps.V_plasma):.1f} m^3")
print(f"  beta_N       = {float(ps.beta_N):.2f}  (cap 3.5)")
print(f"  P_fus        = {float(ps.p_fus):.0f} MW")
print(f"  LCOE         = {float(r.costs.lcoe):.1f} $/MWh")
print(f"  Overnight    = {float(r.costs.capital_per_kw):.0f} $/kW")

print()
print("  YAML-ready pinned operating point:")
print(f"    R0: {R0:.3f}")
print(f"    plasma_t: {a:.3f}")
print(f"    B: {B0:.3f}")
print(f"    T_e: {float(ps.T_e):.2f}")
print(f"    n_e: {float(ps.n_e):.4e}")
print(f"    plasma_volume: {float(ps.V_plasma):.1f}")
