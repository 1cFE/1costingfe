# Levelized Annual Cost — Growing Annuity PV

**Date:** 2026-02-20
**Status:** Implemented
**Affects:** CAS71 (Annual O&M), CAS80 (Annual fuel cost)

## Problem

`levelized_annual_cost` converts an annual cost (stated in today's dollars)
into a level annual payment for LCOE. The old implementation:

```python
annual_cost * (1 + inflation_rate) ** construction_time
```

This only inflates the cost to the first year of operation. It ignores the
continued escalation of costs over the 30-year operating lifetime.

## Why this matters

The model uses nominal financial parameters:

- `interest_rate = 0.07` is the nominal WACC (the opportunity cost of capital —
  what the money could earn in an alternative investment of comparable risk)
- O&M and fuel costs, stated in 2024$, grow at the inflation rate in nominal
  terms

When discounting a nominally-growing cost stream at a nominal rate, the
growing-annuity PV formula is required. The old formula underestimates
levelized O&M + fuel by ~23% (at 2% inflation, 30yr lifetime).

## Formula

### Step 1: Inflate to first-year-of-operation dollars

```
A_1 = annual_cost * (1 + g)^Tc
```

where `g` is the inflation rate and `Tc` is the construction time (or total
project time for FOAK).

### Step 2: Present value of growing annuity

For `i != g`:
```
PV = A_1 * (1 - ((1+g)/(1+i))^n) / (i - g)
```

For `i == g` (L'Hopital limit):
```
PV = A_1 * n / (1 + i)
```

where `i` is the nominal interest rate and `n` is the plant lifetime.

### Step 3: Annualize with CRF

```
levelized = CRF(i, n) * PV
```

Plain CRF, not effective CRF — construction-period financing is handled by
CAS60 (IDC).

## Numerical example

| Parameter | Value |
|-----------|-------|
| annual_cost | $100M (2024$) |
| interest_rate | 7% |
| inflation_rate | 2% |
| plant_lifetime | 30 yr |
| construction_time | 6 yr |

```
A_1 = 100 * 1.02^6 = 112.616 M$
PV  = 112.616 * (1 - (1.02/1.07)^30) / (0.07 - 0.02) = 1716.5 M$
CRF = 0.07 * 1.07^30 / (1.07^30 - 1) = 0.08059
levelized = 0.08059 * 1716.5 = 138.35 M$
```

The old formula would give `100 * 1.02^6 = 112.6 M$` — 23% too low.

## Special cases

- **Zero inflation**: `PV = A_1 * (1 - (1/1.07)^30) / 0.07 = A_1 / CRF`,
  so `levelized = CRF * A_1 / CRF = A_1 = annual_cost` (since `(1+0)^Tc = 1`).
  Correct: no inflation means the levelized cost equals the stated cost.

- **i == g**: The general formula has `(i - g)` in the denominator. The
  L'Hopital limit gives `PV = A_1 * n / (1 + i)`, avoiding division by zero.
  Implementation uses `jnp.where` for JAX traceability.

## Comparison with pyfecons

pyfecons uses `effective_crf * PV` where `effective_crf = CRF * (1+i)^Tc`.
We use `CRF * PV` because our IDC is in CAS60, not baked into CRF.
The economic result is equivalent — the IDC just lives in a different account.

## Implementation

- `economics.py:levelized_annual_cost(annual_cost, interest_rate, inflation_rate, plant_lifetime, construction_time)`
- Called by `cas70_om` (CAS71 O&M) and `cas80_fuel` (CAS80 fuel cost)
- JAX-compatible via `jnp.where` for the `i == g` edge case

## References

- Brealey, Myers & Allen, "Principles of Corporate Finance" — growing annuity PV
- EMWG, "Cost Estimating Guidelines for Generation IV Nuclear Energy Systems"
