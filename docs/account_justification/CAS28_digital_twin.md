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

The \$5M estimate was provided by NtTau Digital LTD as an in-house cost
estimate for a fusion plant digital twin platform (pyFECONS source code
documentation).  This is consistent with:

- **Industrial digital twin platforms:** Siemens MindSphere, GE Predix, and
  AVEVA deployments for large thermal plants typically cost \$2-10M for
  initial setup including physics models, sensor integration, and operator
  training scenarios.
- **Nuclear industry:** The NRC has endorsed digital twin technology for
  advanced reactor licensing (NUREG/CR-7321, 2023).  Kairos Power and
  TerraPower have disclosed digital twin development budgets in the \$3-8M
  range for their advanced reactor programs.
- **Fusion context:** A fusion digital twin is more complex than a fission
  one (plasma physics, magnet quench modeling, tritium transport) but less
  complex than a full-scope nuclear simulator.

## Scaling

Fixed cost — does not scale with plant size, fuel type, or number of modules.
For multi-module plants, a single digital twin covers all modules (the
incremental cost of adding a module to an existing model is negligible).

## References

- pyFECONS source: `cas28_digital_twin.py` — "In-house cost estimate
  provided by NtTau Digital LTD."
- Woodruff, S., "A Costing Framework for Fusion Power Plants,"
  arXiv:2601.21724, January 2026.
