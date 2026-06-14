"""Example: mirror power-to-geometry sizing.

The mirror framework solves the chamber length from the net electric power ask
instead of costing a stated machine. With size_from_power=True, you give only
the net power; the solver bisects on L, running a golden-section search over T_i
at each trial L to find the operating temperature that maximises net power. The
operating density is the minimum of three caps -- the f_beta beta-pressure
ceiling, the neutron wall-load cap (q_wall_max), and the surface heat-flux cap
(q_surface_max) -- so the wall caps can bind below the beta boundary. Geometry-
driven costs (central-cell solenoids, end-plug coils, blanket, vessel) then
scale with the solved L.

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
    print(f"  q_wall:       {float(ps.wall_loading):8.2f} MW/m^2 (cap 5.0)")
    print(f"  q_surface:    {float(ps.q_surface):8.2f} MW/m^2 (cap 1.0)")
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
# Higher f_beta raises the beta-pressure density ceiling, but the operating
# density is min(n_beta, n_wall, n_surf). At D-T defaults the neutron wall
# cap binds first, so above the f_beta where n_beta crosses n_wall the
# density (and hence L and LCOE) stops moving: the 0.70 and 0.85 rows land
# at the same L and LCOE because both are wall-capped. At low f_beta (0.50)
# n_beta is the binding cap, giving a longer L and higher LCOE. At high
# f_beta (0.95) the wall cap still binds: the operating density is wall-capped,
# beta sits below beta_max, and the row sizes feasibly (the sized state is built
# from the GSS optimum, so beta is bounded by construction). Above the wall-cap
# crossover the LCOE is flat in f_beta (0.70, 0.85, 0.95 share the same L and
# LCOE), so the optimizer in section 3 is indifferent across that plateau.
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
        # Retained for robustness: a geometry whose GSS optimum still violates
        # beta_max would surface here. At the default machine every f_beta in
        # this sweep sizes feasibly (beta is bounded by the GSS construction).
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
# The neutron wall cap holds q_wall at its 5.0 MW/m^2 ceiling, so once f_beta
# is high enough for the wall cap to bind, raising it further leaves the
# wall-capped density (and thus L and LCOE) unchanged. The LCOE is flat across
# that plateau, so the reported optimal f_beta is anywhere on it.

# ── 4. Infeasibility demo: target beyond L_max reach ─────────────────
# At L_max=5 m and a=1.5 m (the YAML default plasma radius) the wall-capped
# density yields only about 30 MWe net. A 400 MWe ask with L_max=5 m cannot
# converge; the model raises SizingInfeasible with the p_net at L_max so the
# caller knows how far short the machine falls.
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
