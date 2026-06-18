"""Example: tokamak power-to-geometry sizing.

The framework can solve the machine FROM the power ask instead of costing a
stated machine. With size_from_power=True, you state only the net electric
power; the solver bisects on major radius R0, putting each trial machine at
its constraint-boundary operating point (density at the Greenwald fraction,
operating temperature maximizing net power subject to the Troyon beta limit),
with the on-axis field derived from the magnet's B_max through the inboard
radial build. Geometry-driven costs (coils, blanket, shield, vessel) then
scale with the power target.

Shown here:
  1. Size mode across power targets   - economies of scale, R0(P)
  2. Magnet technology choice         - REBCO vs Nb3Sn vs copper
  3. Confinement quality (H_factor)   - what better confinement buys
  4. Infeasibility is loud            - SizingInfeasible, not garbage
  5. Optimize mode                    - LCOE-optimal Greenwald fraction
  6. Fuel matters                     - D-He3 needs a bigger machine than D-T

R0, B, b_center and plasma_t cannot be pinned in sizing mode; they are
outputs. Pin-mode (stating the machine) remains the default elsewhere.
"""

from costingfe import ConfinementConcept, CostModel, Fuel
from costingfe.layers.tokamak import SizingInfeasible

CUSTOMER = dict(availability=0.85, lifetime_yr=30)


def show(model, result):
    ps = result.plasma_state
    print(f"  R0 (solved):  {model._last_R0:8.2f} m")
    print(f"  T_e (solved): {float(ps.T_e):8.1f} keV")
    print(f"  beta_N:       {float(ps.beta_N):8.2f} %*m*T/MA (cap 3.5)")
    print(f"  P_fus:        {float(ps.p_fus):8.0f} MW")
    print(f"  LCOE:         {float(result.costs.lcoe):8.1f} $/MWh")
    print(f"  Overnight:    {float(result.costs.capital_per_kw):8.0f} $/kW")


# ── 1. Size mode: the machine follows the power ask ───────────────────
print("=" * 64)
print("  SIZE MODE: solve R0 from net electric power (REBCO, H=1.0)")
print("=" * 64)

for p_net in (200.0, 400.0, 1000.0):
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    r = model.forward(net_electric_mw=p_net, size_from_power=True, **CUSTOMER)
    print(f"\n  Target {p_net:.0f} MWe:")
    show(model, r)

# ── 2. Magnet technology ──────────────────────────────────────────────
print()
print("=" * 64)
print("  MAGNET CHOICE at 400 MWe: B_max, recirculation, cryo via table")
print("=" * 64)

for material in ("rebco_hts", "nb3sn", "copper"):
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    try:
        r = model.forward(
            net_electric_mw=400.0,
            size_from_power=True,
            coil_material=material,
            **CUSTOMER,
        )
        print(f"\n  {material}:")
        show(model, r)
    except SizingInfeasible as e:
        print(f"\n  {material}: INFEASIBLE - {e}")

# ── 3. Confinement quality ────────────────────────────────────────────
# H_factor sets the auxiliary heating required to sustain the operating
# point (solved from IPB98y2), hence recirculating power and driver cost.
print()
print("=" * 64)
print("  CONFINEMENT QUALITY at 400 MWe: H = 1.0 vs 1.8")
print("=" * 64)

for h in (1.0, 1.8):
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
    r = model.forward(
        net_electric_mw=400.0, size_from_power=True, H_factor=h, **CUSTOMER
    )
    print(f"\n  H_factor = {h}:")
    show(model, r)

# ── 4. Infeasibility is a result, not an error to work around ─────────
print()
print("=" * 64)
print("  INFEASIBLE ASK: 3 GWe on copper coils")
print("=" * 64)

model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
try:
    model.forward(
        net_electric_mw=3000.0,
        size_from_power=True,
        coil_material="copper",
        **CUSTOMER,
    )
except SizingInfeasible as e:
    print(f"\n  SizingInfeasible: {e}")

# ── 5. Optimize mode: LCOE over the Greenwald fraction ────────────────
# Pushing density raises fusion power per unit machine but raises the
# stored energy, and with it the IPB98 heating requirement (~W^(1/0.31)),
# so Q_eng degrades sharply toward high f_GW; the disruption penalty
# (grounded values, docs/account_justification/disruption_severity.md)
# is second-order against that. The optimum currently sits at the
# f_GW_min bound; whether that low-density preference is physical or a
# missing-physics artifact (no current-drive or L-H-threshold density
# coupling) is an open model question. Slowest section (re-runs the
# full sizing+cost pipeline per trial f_GW).
print()
print("=" * 64)
print("  OPTIMIZE MODE at 400 MWe: LCOE-optimal f_GW")
print("=" * 64)

model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
r = model.forward(net_electric_mw=400.0, optimize_lcoe=True, **CUSTOMER)
print(f"\n  optimal f_GW: {model._sizing_fgw:8.3f}")
show(model, r)

# ── 6. Fuel matters: D-He3 sizes to a bigger machine ──────────────────
# The kernel is fuel-aware: weaker D-He3 reactivity plus dilution means a
# larger machine at equal net power (radiation enters via the f_rad_fus
# proxy, as in the non-0D path). The D-He3 case gets a fuel-appropriate
# build: no breeding blanket (sizing keeps the concept YAML's radial build
# unless overridden, and the tokamak YAML's 0.80 m PbLi blanket is a D-T
# design choice). p-B11 does not close at any R0 <= R0_max and would
# raise SizingInfeasible.
print()
print("=" * 64)
print("  FUEL CHOICE at 200 MWe: D-T vs D-He3 (H = 1.8)")
print("=" * 64)

DHE3_BUILD = dict(
    blanket_form="none",
    blanket_fill="none",
    blanket_t=0.0,
    ht_shield_t=0.1,
    structure_t=0.15,
    vessel_t=0.10,
)
for fuel, extra in ((Fuel.DT, {}), (Fuel.DHE3, DHE3_BUILD)):
    print(f"\n  {fuel.value}:")
    model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=fuel)
    r = model.forward(
        net_electric_mw=200.0,
        size_from_power=True,
        H_factor=1.8,
        **extra,
        **CUSTOMER,
    )
    show(model, r)
