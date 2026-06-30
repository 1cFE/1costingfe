# Unified Pulsed Inverse via Q_eng

**Date:** 2026-04-10
**Status:** Design approved

## Problem

The pulsed power balance has inconsistent parameterization:
- DEC inverse: fixes `q_sci`, derives `e_driver_mj`
- Thermal inverse: fixes `e_driver_mj`, `q_sci` floats

This is inconsistent with the steady-state approach (fix P_net and a performance
target, derive everything else). Users must specify different variables depending
on conversion type.

## Design

Use `q_eng` (engineering gain = P_et / P_recirc) as the universal pulsed input,
replacing both `e_driver_mj` and `q_sci`. The user specifies three things:

- `p_net` — net electric target (already the top-level input)
- `q_eng` — plant performance target
- `f_rep` — repetition rate (drives hardware sizing)

Everything else is derived:
- `p_et = p_net * q_eng / (q_eng - 1)`
- `p_recirc = p_et / q_eng`
- `p_driver` from `p_recirc` minus auxiliary loads (different formula for thermal vs DEC)
- `e_driver_mj = p_driver / f_rep`
- `p_fus` from forward pass with derived `e_driver_mj`
- `q_sci = p_fus / p_driver` (derived output)

## Recirculating power difference

- **Thermal**: `p_recirc = p_driver/eta_pin + p_aux + f_sub*p_et + p_pump + p_cryo + p_target + p_coils`
- **DEC**: `p_recirc = p_driver*(1/eta_pin - 1) + p_aux + f_sub*p_et + p_pump + p_cryo + p_target + p_coils`

Solving for `p_driver`:
- **Thermal**: `p_driver = (p_recirc - fixed_loads) * eta_pin`
- **DEC**: `p_driver = (p_recirc - fixed_loads) / (1/eta_pin - 1)`

Where `fixed_loads = p_aux + f_sub*p_et + p_pump_term + p_cryo + p_target + p_coils`.

## Changes

### physics.py

Two inverse functions, both accepting `q_eng`:

```
pulsed_thermal_inverse(p_net_target, q_eng, f_rep, ...) -> (p_fus, e_driver_mj)
pulsed_dec_inverse(p_net_target, q_eng, f_rep, ...) -> (p_fus, e_driver_mj)
```

Both return `(p_fus, e_driver_mj)` — same shape. Internal algebra differs
(recirculating definition). Forward passes unchanged.

### model.py

Pulsed branch reads `q_eng` from params, passes to both inverses uniformly.
The `if INDUCTIVE_DEC` / `else` split remains for the forward pass (physics
differs) but the inverse call pattern is the same.

### pulsed_*.yaml (all 10 files)

- Remove `e_driver_mj`
- Remove `q_sci` (where present)
- Add `q_eng` with concept-appropriate defaults

Suggested defaults:
- IFE (laser, heavy ion, z-pinch): `q_eng: 4.0`
- MIF (mag target, plasma jet, maglif): `q_eng: 3.0`
- Pulsed MCF (pulsed FRC, theta pinch, DPF, staged z-pinch): `q_eng: 3.0`

### Tests

Update all pulsed tests to use `q_eng` instead of `e_driver_mj`/`q_sci`.

### paper.tex

Update pulsed power balance section to describe Q_eng as the primary input.
Update inverse formula derivation.

### fusion-backcasting (lcoe-dashboard branch)

1. Copy updated `costingfe/` from numpy-only
2. Frontend `EngineeringPanel.tsx`: Replace `e_driver_mj` slider with `q_eng`
   slider (min: 1.5, max: 10, step: 0.1, format: `Qe = X.X`)
3. Types already have `q_eng` on PowerTable — just needs to be an input too
4. Backend passes params through, no changes expected
