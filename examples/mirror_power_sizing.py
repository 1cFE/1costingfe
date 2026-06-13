"""Example: mirror power-to-geometry sizing.

The mirror framework solves the chamber length from the net electric power ask
instead of costing a stated machine. With size_from_power=True, you give only
the net power; the solver bisects on L, running a golden-section search over T_i
at each trial L to find the operating temperature that maximises net power at the
f_beta density boundary. Geometry-driven costs (central-cell solenoids, end-plug
coils, blanket, vessel) then scale with the solved L.

Shown here:
  1. Size mode across power targets   - L(P), economies of scale
  2. f_beta sensitivity at 400 MWe   - L / LCOE trade at fixed power
  3. Optimize mode at 400 MWe        - LCOE-optimal f_beta via outer GSS
  4. Infeasibility is loud            - SizingInfeasible, not garbage

L, T_i, n_e and beta cannot be pinned in sizing mode; they are outputs.
Pin mode (stating the machine) is the default used in dt_mirror_0d.py.
"""

from costingfe import ConfinementConcept, CostModel, Fuel
from costingfe.layers.physics import OperatingPointInfeasible, SizingInfeasible

CUSTOMER = dict(availability=0.85, lifetime_yr=30)

# Base sizing overrides. chamber_length is omitted so the solver fills it.
# B, plasma_t, R_m and the power-balance parameters stay at YAML defaults;
# only the sizing-search bounds are pinned here.
BASE = dict(
    L_min=1.0,
    L_max=400.0,
)


def show(model, result):
    """Print solved plasma state and economics."""
    ps = model._plasma_state
    c22 = result.cas22_detail
    print(f"  L (solved):   {model._last_L:8.1f} m")
    print(f"  T_i (GSS):    {float(ps.T_i):8.1f} keV")
    print(f"  n_e (f_beta): {float(ps.n_e):8.2e} m^-3")
    print(f"  beta:         {float(ps.beta):8.3f}")
    print(f"  P_fus:        {float(ps.p_fus):8.0f} MW")
    print(f"  C220103 coil: {float(c22['C220103']):8.1f} M$")
    print(f"  LCOE:         {float(result.costs.lcoe):8.1f} $/MWh")
    print(f"  Overnight:    {float(result.costs.overnight_cost):8.0f} $/kW")


# ── 1. Size mode: chamber length follows the power ask ────────────────
print("=" * 64)
print("  SIZE MODE: solve L from net electric power (f_beta = 0.85)")
print("=" * 64)
print()

header = (
    f"  {'Target':>8}  {'L [m]':>8}  {'T_i [keV]':>10}"
    f"  {'n_e [m^-3]':>12}  {'beta':>6}  {'P_fus [MW]':>11}"
    f"  {'C220103 [M$]':>13}  {'LCOE [$/MWh]':>13}  {'$/kW':>8}"
)
print(header)
print("  " + "-" * 107)

for p_net in (100.0, 200.0, 400.0, 600.0):
    model = CostModel(concept=ConfinementConcept.MIRROR, fuel=Fuel.DT)
    r = model.forward(
        net_electric_mw=p_net,
        size_from_power=True,
        f_beta=0.85,
        **BASE,
        **CUSTOMER,
    )
    ps = model._plasma_state
    c220103 = float(r.cas22_detail["C220103"])
    lcoe = float(r.costs.lcoe)
    oc = float(r.costs.overnight_cost)
    row = (
        f"  {p_net:>8.0f}  {model._last_L:>8.1f}  {float(ps.T_i):>10.1f}"
        f"  {float(ps.n_e):>12.2e}  {float(ps.beta):>6.3f}  {float(ps.p_fus):>11.0f}"
        f"  {c220103:>13.1f}  {lcoe:>13.1f}  {oc:>8.0f}"
    )
    print(row)

# ── 2. f_beta sensitivity at 400 MWe ─────────────────────────────────
# Higher f_beta packs the plasma denser (n_e proportional to f_beta).
# That raises fusion power per unit volume, so a shorter L closes the
# power balance. The coil cost tracks L (central-cell solenoids scale with
# chamber_length / coil_spacing), so LCOE falls. The optimizer in section 3
# should confirm the direction and pin the floor.
print()
print("=" * 64)
print("  f_beta SWEEP at 400 MWe: density-pressure trade vs. L and LCOE")
print("=" * 64)
print()

print(
    f"  {'f_beta':>7}  {'L [m]':>8}  {'T_i [keV]':>10}"
    f"  {'beta':>6}  {'C220103 [M$]':>13}  {'LCOE [$/MWh]':>13}"
)
print("  " + "-" * 70)

for fb in (0.5, 0.7, 0.85, 0.95):
    model = CostModel(concept=ConfinementConcept.MIRROR, fuel=Fuel.DT)
    try:
        r = model.forward(
            net_electric_mw=400.0,
            size_from_power=True,
            f_beta=fb,
            **BASE,
            **CUSTOMER,
        )
    except OperatingPointInfeasible as exc:
        # Wall cap + beta limit simultaneously infeasible at high f_beta: the
        # density is wall-capped below n_beta, the required T_i then exceeds
        # beta_max. Report the condition; narration sync is Task 7.
        print(f"  {fb:>7.2f}  infeasible: {exc}")
        continue
    ps = model._plasma_state
    c220103 = float(r.cas22_detail["C220103"])
    row = (
        f"  {fb:>7.2f}  {model._last_L:>8.1f}  {float(ps.T_i):>10.1f}"
        f"  {float(ps.beta):>6.3f}  {c220103:>13.1f}  {float(r.costs.lcoe):>13.1f}"
    )
    print(row)

# ── 3. Optimize mode: LCOE-optimal f_beta ────────────────────────────
# The outer GSS searches f_beta in [f_beta_min, f_beta_max] (from the
# YAML: 0.3 to 1.0). Each trial f_beta re-runs the full sizing+cost
# pipeline. This is the slowest section: about 12 outer f_beta evaluations,
# each triggering 60 L-bisection steps x 40 T_i GSS steps (about 30k
# forward calls total). One optimize run only.
print()
print("=" * 64)
print("  OPTIMIZE MODE at 400 MWe: LCOE-optimal f_beta (slow run)")
print("=" * 64)

model = CostModel(concept=ConfinementConcept.MIRROR, fuel=Fuel.DT)
r_opt = model.forward(
    net_electric_mw=400.0,
    optimize_lcoe=True,
    **BASE,
    **CUSTOMER,
)
fb_opt = model._sizing_fbeta
print("\n  Optimizer result:")
print(f"  optimal f_beta:  {fb_opt:.3f}")
show(model, r_opt)

# ── 4. Infeasibility demo: target beyond L_max reach ─────────────────
# At L_max=5 m and a=1.5 m, the machine can only deliver about 190 MWe.
# A 400 MWe ask with L_max=5 m cannot converge; the model raises
# SizingInfeasible with the p_net at L_max so the caller knows how
# far short the machine falls.
print()
print("=" * 64)
print("  INFEASIBLE ASK: 400 MWe with L_max = 5 m")
print("=" * 64)

model = CostModel(concept=ConfinementConcept.MIRROR, fuel=Fuel.DT)
try:
    model.forward(
        net_electric_mw=400.0,
        size_from_power=True,
        f_beta=0.85,
        L_min=1.0,
        L_max=5.0,
        **CUSTOMER,
    )
except SizingInfeasible as e:
    print(f"\n  SizingInfeasible: {e}")
