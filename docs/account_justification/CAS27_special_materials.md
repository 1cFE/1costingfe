# CAS27: Special Materials

**Date:** 2026-03-16
**Status:** Implemented — new account

## Overview

CAS27 covers the **initial inventory** of non-fuel reactor materials:
one-time capital costs for the material that fills the blanket structure,
neutron multiplier inventory, and other special reactor materials.

**Key boundary:**
- CAS220101 covers the blanket *structure* (steel, fabrication).
- CAS27 covers the *material that fills it* (PbLi, Li, Be, FLiBe).
- CAS80 covers *fuel* (deuterium, tritium breeding feedstock as Li-6, etc.).
- Annual replenishment of consumed materials is an operating cost (CAS70).

## Costing Model

    CAS27 = special_materials_base(fuel) × (P_net / 1000)

Linear scaling with net electric power.

| Fuel | Base cost (M$ at 1 GWe) | Materials |
|------|------------------------:|-----------|
| DT | 15.0 | PbLi eutectic fill + enriched Li |
| DD | 2.0 | Conventional coolant fills |
| DHe3 | 1.0 | Minimal special materials |
| pB11 | 0.0 | No special materials |

## DT Blanket Material Analysis

The default DT cost assumes a **PbLi (lead-lithium eutectic) blanket
concept** — the dominant design for EU-DEMO (HCLL, DCLL, WCLL concepts).

### PbLi Eutectic Fill

PbLi (Pb-17Li) is ~99.3% lead by mass, 0.7% lithium.

- **Inventory estimate:** For a 1 GWe plant with ~650 m³ blanket volume
  and ~50% PbLi fill fraction: 650 × 0.5 × 9,800 kg/m³ ≈ 3,200 tonnes.
  EU-DEMO estimates range from 3,000–5,000 tonnes.
- **Unit cost:** Lead: ~$2/kg; natural lithium: ~$20–40/kg.
  PbLi eutectic (dominated by Pb): ~$2.5–3/kg.
- **Fill cost:** 4,000 tonnes × $3/kg = **$12M**.

### Enriched Lithium Premium

For PbLi blankets using enriched Li-6 (to boost TBR):
- Li-6 enrichment from 7.5% (natural) to 30–90%: ~$100–1,000/kg
- Lithium mass in 4,000 tonnes PbLi: ~28 tonnes
- At $500/kg enrichment premium: ~$14M additional
- Most PbLi concepts use natural or low-enrichment Li (TBR achieved
  through blanket geometry), so this is a moderate estimate.

### Total DT default: $15M

Covers PbLi fill ($12M) + modest enriched-Li premium ($3M).
This is a **small** cost relative to the $2.7B CAS22 total.

### Important: HCPB Beryllium Override

The Helium-Cooled Pebble Bed (HCPB) blanket concept uses **beryllium
pebbles** as neutron multiplier instead of lead:

- EU-DEMO HCPB requires ~300–490 tonnes of Be pebbles
- Beryllium: $500–800/kg (resource-constrained, strategic material)
- **Cost: $180–300M** — 10–20× higher than PbLi concept

If using an HCPB blanket, CAS27 should be overridden:
```python
result = model.forward(..., cost_overrides={"CAS27": 200.0})
```

The HCPB beryllium cost is a major design discriminator.  Global Be
production is ~300 tonnes/yr; a single DEMO-scale reactor would consume
the entire annual supply.  This is a known concern in the EUROfusion
blanket selection process.

## Non-DT Fuels

**DD ($2M):** No breeding materials needed (DD does not require TBR > 1).
Covers conventional coolant system fills (water treatment chemicals,
gas inventory for helium-cooled systems).

**DHe3 ($1M):** Minimal.  ~5% neutron fraction means a very thin
neutron shield but no breeding blanket or multiplier.

**pB11 ($0):** Aneutronic.  No breeding blanket, no neutron multiplier,
no special coolants.  All materials are conventional industrial.

## FLiBe Alternative

For molten-salt blanket concepts (e.g., ARC-class with FLiBe coolant):
- FLiBe (2LiF-BeF2): ~$50–150/kg depending on Li-7 enrichment
- Inventory: ~500–2,000 tonnes for 1 GWe
- **Cost: $25–300M** depending on isotopic requirements

FLiBe concepts should override CAS27 similarly to HCPB.  The cost is
driven by Li-7 enrichment (to suppress parasitic neutron absorption
by Li-6 in non-breeding applications) and beryllium fluoride content.

## References

- EUROfusion, "Progress in EU Breeding Blanket design," WPPMI-CPR(17)
  17709, Cismondi et al.
- OSTI, "Tritium Breeding Blanket for a Commercial Fusion Power Plant,"
  LLNL-TR-652984, 2014.
- ITER Organization, "Blanket," https://www.iter.org/machine/blanket
- DOE OSTI, "Fuels for Fusion," Pearson (2022),
  https://science.osti.gov/-/media/fes/pdf/fes-presentations/2022/
  Pearson_resource-availability-and-supply_presentation.pdf
- INEEL, "FLiBe Use in Fusion Reactors — An Initial Safety Assessment,"
  INEEL/EXT-99-00331, 1999.
