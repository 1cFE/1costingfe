# burn_fraction account justification

Per-concept single-pass burn fraction `burn_fraction` values, with sourcing
and provenance notes.

## Method

`burn_fraction` is the fraction of injected fuel atoms that undergo fusion
before being exhausted from the reaction volume. Combined with
`fuel_recovery` (the fraction of unburned fuel recovered and recycled), it
gives the cumulative fusion probability of an injected atom as
`burn_fraction / (1 - fuel_recovery * (1 - burn_fraction))`, which drives
CAS80 fuel-cost accounting.

The value is bounded above by physics (kinematic and confinement) and
chosen by design (operating-point selection). Values below are
reactor-target (NOAK) design points, not experimental records.

## Steady-state MFE (regime default 0.05)

The MFE concepts (tokamak, stellarator, mirror, orbitron, polywell) all
operate at moderate n·τ confinement and share the same single-pass burnup
regime. Range across designs: 0.02-0.10 (including legacy mirror studies
at the low end). Standard reactor-target value: 0.05.

- **tokamak: 0.05** - ARIES-AT, ITER-class operating point. Sourcing
  pending detailed citation.
- **stellarator: 0.05** - HSX/W7-X reactor projections, same n·τ regime as
  tokamak. Sourcing pending.
- **mirror: 0.05** - Modern axisymmetric tandem mirror (Realta-class).
  Legacy MFTF/TARA studies quote 1-3%; high-mirror-ratio designs catch up.
  Sourcing pending.
- **orbitron: 0.05** - No public reactor-scale study. **MFE-class
  placeholder** pending design-specific data.
- **polywell: 0.05** - No public reactor-scale study. **MFE-class
  placeholder** pending design-specific data.

## Inertial confinement (ICF)

Hot-spot ρR sets the burn fraction; reactor-target designs assume
substantially higher values than NIF ignition shots (which achieve 2-4%).

- **laser_ife: 0.25** - Reactor-target value from direct-drive ICF reactor
  studies (LIFE, HYLIFE, HAPL NRL). Range: 0.10-0.35. Sourcing pending.
- **heavy_ion: 0.30** - Indirect-drive heavy-ion fusion (HIBALL, HIBLIC).
  Higher uniformity than direct-drive laser. Range: 0.20-0.40. Sourcing
  pending.

## Magneto-inertial / pulsed magnetic

A heterogeneous bucket where burn fraction depends strongly on
compression dwell time, peak ρ, and disassembly mechanism. Each concept
gets an individual value.

### pulsed_frc: 0.15 (Helion-class staged-compression FRC)

**Vendor does not publish a burn fraction.** Helion and the broader
compressed-FRC literature use **fusion gain Q** as the figure of merit
(Q of 6-11 at peak compression for D-T staged-compression FRC).

The 0.15 value is a regime midpoint from the magneto-inertial fusion
literature, which generically targets 10-25% single-pass burnup for
economic operation. This will need updating if Helion or a peer
publishes a concept-specific burnup figure.

Sources reviewed (none quote a burn fraction directly):

- [Helion FAQ](https://www.helionenergy.com/faq/)
- [More on Helion's pulsed approach](https://www.helionenergy.com/articles/more-on-helions-pulsed-approach-to-fusion/)
- [Slough et al., "A compact fusion reactor based on staged compression of an FRC" (Nucl. Fusion 2024)](https://iopscience.iop.org/article/10.1088/1741-4326/ae034d) - Q-based metrics, no burn fraction
- [Wurden, "Magneto-Inertial Fusion" 2-pager (PPPL)](https://fire.pppl.gov/IFE_NAS_MTF_Wurden_2pager.pdf) - generic MIF regime statement
- [Hybrid simulations of FRC merging and compression (arxiv 2501.03425)](https://arxiv.org/pdf/2501.03425)
- [Quasi-static magnetic compression of FRC (arxiv 2204.07978)](https://arxiv.org/pdf/2204.07978)

### Other magneto-inertial / pulsed concepts (sourcing pending)

- **maglif: 0.15** - Slutz 2010 / Knapp 2019 2D LASNEX projections at full
  Z-driver. Range: 0.08-0.25. Sourcing pending.
- **mag_target: 0.10** - General Fusion plasma-compression projections,
  MIF concept literature. Range: 0.05-0.15. Sourcing pending.
- **plasma_jet: 0.10** - PJMIF MHD sims (Witherspoon, Hsu). Range:
  0.05-0.15. Sourcing pending.
- **staged_zpinch: 0.10** - LANL Rahman/Wessel staged-pinch designs.
  Range: 0.05-0.15. Sourcing pending.
- **zpinch: 0.10** - Standard Z-pinch reactor concepts; Rayleigh-Taylor
  disassembly limit. Range: 0.05-0.15. Sourcing pending.
- **theta_pinch: 0.05** - Faster expansion than Z-pinch; closer to MFE
  bound. Range: 0.03-0.08. Sourcing pending.
- **dense_plasma_focus: 0.01** - LPP Focus Fusion projections;
  sub-microsecond pinch lifetime is the binding constraint. Range:
  0.005-0.02. **Genuinely low**, drives LCOE materially for DPF
  comparisons. Sourcing pending.

## fuel_recovery: 0.99 (uniform across all concepts)

NOAK fuel-cycle recycling efficiency. ITER targets 99% for tritium and
serves as the mature reference. Within-concept architectural spread
(gas-phase exhaust vs. target-factory residue) is at most 5 percentage
points at NOAK; sensitivity to `fuel_recovery` is highest at low
`burn_fraction`, exactly where concepts are most architecturally similar
(all gas-phase MFE). A single uniform value is therefore defensible.

A future refinement would treat `fuel_recovery` as FOAK/NOAK-toggled
(e.g., FOAK 0.95, NOAK 0.99). Out of scope for this iteration.
