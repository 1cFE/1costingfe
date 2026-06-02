# CAS28: Digital Twin

**Date:** 2026-03-16
**Status:** Justified — value retained

## Overview

CAS28 covers the capital cost of a plant-wide digital twin: a high-fidelity
simulation model of the as-built plant used for operator training, predictive
maintenance, and operational optimization.

**Adopted value:** \$5M flat (plant-size independent).

## Rationale

The digital twin is a software-dominated cost: physics models, sensor
integration, visualization, and data infrastructure.  It does not scale
meaningfully with plant thermal or electric capacity — the computational
complexity is driven by the number of subsystems modeled, not by their
physical size.

The \$5M figure is anchored to disclosed advanced-reactor digital-twin
budgets and industrial platform pricing:

- **DOE ARPA-E GEMINA program:** The Generating Electricity Managed by
  Intelligent Nuclear Assets program funded nine advanced-reactor digital
  twin projects totaling \$27M.  The University of Michigan received \$5.2M
  to build a scalable reactor digital twin (validated on a campus molten-salt
  loop, then applied to the Kairos Power design); Argonne received \$2.2M for
  a complementary Kairos O&M automation twin.  A single scalable plant-wide
  reactor digital twin therefore lands right at \$5M.
- **Industrial digital twin platforms:** Siemens MindSphere, GE Predix, and
  AVEVA deployments for large thermal plants typically cost \$2-10M for
  initial setup including physics models, sensor integration, and operator
  training scenarios.
- **Fusion context:** A fusion digital twin is more complex than a fission
  one (plasma physics, magnet quench modeling, tritium transport) but less
  complex than a full-scope nuclear simulator, placing it mid-range of the
  GEMINA per-project budgets.

The same \$5M appears as an internal pyFECONS placeholder, but the value is
adopted here on the independent GEMINA / industrial-platform basis above.

## Scaling

Fixed cost — does not scale with plant size, fuel type, or number of modules.
For multi-module plants, a single digital twin covers all modules (the
incremental cost of adding a module to an existing model is negligible).

## References

- DOE / ARPA-E GEMINA program awards (2020): nine advanced-reactor digital
  twin projects, \$27M total; U. Michigan scalable reactor digital twin
  \$5.2M, Argonne Kairos O&M twin \$2.2M.
- NRC endorsement of digital twin technology for advanced reactor licensing
  (NUREG/CR-7321, 2023).
- Woodruff, S., "A Costing Framework for Fusion Power Plants,"
  arXiv:2601.21724, January 2026.
