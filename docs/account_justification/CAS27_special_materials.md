# CAS27: Special Materials

**Date:** 2026-03-16
**Status:** Implemented — new account

## Overview

CAS27 covers the **initial inventory** of non-fuel reactor materials:
one-time capital costs for the material that fills the blanket structure,
neutron multiplier inventory, and other special reactor materials.

**Key boundary:**
- CAS220101 covers the blanket *structure* (steel, W armor, fabrication).
- CAS27 covers the *material that fills it* (PbLi, Li, Be, FLiBe, ceramic).
- CAS80 covers *fuel* (deuterium, tritium breeding feedstock as Li-6, etc.).
- Annual replenishment of consumed materials is an operating cost (CAS70).

For this split to avoid double-counting, the CAS220101 unit costs must be
**structure only**. The D-T unit cost was re-anchored from 0.60 to 0.35 M$/m³
for exactly this reason: the 0.60 value had folded the PbLi fill mass into the
structure account, so every breeding blanket (PbLi, Li, FLiBe, Be-ceramic) paid
for the fill twice — once in CAS220101 and again here. See
`CAS22_reactor_components.md`.

The Li₂O breeding blanket carries its own CAS220101 base
(`blanket_unit_cost_li2o`) so its structure cost is decoupled from the fuel key
(a solid-breeder shell on a non-D-T-keyed machine such as the levitated dipole
would otherwise be priced off the aneutronic fuel unit cost). That base is set
to the same structure-only 0.35 M$/m³ as D-T, with the `solid_breeder`
structure_factor adding the pebble-canister premium on top — the identical
basis used for `be_ceramic` and `ceramic_only` — and the Li₂O fill itself is
priced here in CAS27. For the levitated dipole the volume-based CAS220101 is
only a first-cut template; faithful capital costing of its large thin shell is
mass-based (Simpson et al. Table 5) and belongs in per-instance
`cost_overrides`.

## Costing Model — volume-based, keyed on blanket fill

A blanket-fill inventory is a **material quantity set by blanket volume, not by
net electric power**. CAS27 is therefore a volume-based mass build-up, keyed on
`blanket_fill` (it does not depend on fuel or `P_net`):

    CAS27 = blanket_vol × vol_frac × density × price / 1e6

where `blanket_vol` is the model's first-wall + blanket + reflector volume,
and `vol_frac` is the fraction of that region occupied by the costed material
(liquid fill fraction for liquids; breeder/multiplier-zone × pebble-packing
≈ 0.25 for solid pebble beds). Fuel-appropriateness is captured by which fill a
concept selects. The fill is also normalized by fuel, so a concept's D-T-flavored
YAML default is not carried onto a fuel that cannot use it (explicit overrides
win): aneutronic fuels (D-He³, p-B¹¹) force `blanket_fill = none` → CAS27 = 0;
D-D forces `water` (it breeds its own tritium via D+D→T+p, so it needs no lithium
breeder, but is neutronic and so keeps a low-Z moderating energy-capture blanket)
→ CAS27 ≈ 0; only D-T carries a breeder inventory.

This replaced the earlier `special_materials_base(fuel) × fill_factor ×
(P_net/1000)` model. Power-scaling was a proxy that mis-fit compact / high-
power-density designs (e.g. ARC's FLiBe immersion blanket) and under-counted
large-blanket designs (e.g. the levitated dipole's thin-but-huge Li₂O shell,
whose fill was costed at ~$0.6M against a several-hundred-M$ blanket structure).
Scaling by
the modelled `blanket_vol` instead is consistent with how the blanket
*structure* (C220101) and the other reactor-component accounts are already costed.

| `blanket_fill` | density (kg/m³) | vol_frac | price ($/kg) | basis |
|---|---:|---:|---:|---|
| `pbli` | 9400 | 0.50 | 5.0 | Pb-17Li (99.3% Pb @ ~$3/kg) + enriched Li-6 premium |
| `li` | 490 | 0.80 | 200 | liquid Li; Li-6 enrichment ($80 natural → $1000 high; $200 mid) |
| `flibe` | 1940 | 0.80 | 150 | 2LiF·BeF₂, Be-dominated (Araiinejad & Shirvan 2025 NOAK ~$154/kg); ρ from Sohal 2010 |
| `be_ceramic` | 1850 | 0.25 | 700 | HCPB Be multiplier (dominant cost); Li-ceramic breeder folded in |
| `ceramic_only` | 2400 | 0.25 | 150 | Li₄SiO₄/Li₂TiO₃ breeder pebbles |
| `li2o` | 2013 | 0.25 | 150 | Li₂O ceramic |
| `water` | 1000 | 0.50 | 0.01 | light-water moderator/coolant, no breeder (D-D); ~$10/t treated → negligible |
| `none` | — | — | — | aneutronic / no breeder → 0 |

At a ~650 m³ 1 GWe blanket this lands PbLi ≈ $15M (unchanged baseline), FLiBe
≈ $150M, HCPB ≈ $210M. The two most uncertain inputs are the `li` enrichment
price and the solid-pebble `vol_frac` (~0.25); for HCPB the dominant Be is
costed and the Li-ceramic breeder folded into that number.

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

## Blanket configuration multipliers (added 2026-05-17)

CAS27 is now multiplied by `BlanketFill.fill_factor`, and CAS22.01 is multiplied
by `BlanketForm.structure_factor`. Both factors are relative to the existing
DT/PbLi baseline (which keeps factor 1.0 x 1.0 = unchanged).

### `BlanketFill.fill_factor` (CAS27)

| BlanketFill | fill_factor | Source / rationale |
|---|---:|---|
| `pbli` | 1.0 | Baseline. $12M PbLi + $3M Li-6 premium, as documented above. |
| `li` | 2.0 | Self-cooled Li, ~300 t inventory at $300-1000/kg enriched. Center ~$30M. |
| `flibe` | 5.0 | 2LiF-BeF2 melt at $50-150/kg with Li-7 enrichment. ~$75M. See FLiBe section above. |
| `be_ceramic` | 13.0 | HCPB. ~300 t Be at $600/kg + Li-ceramic pebbles. ~$200M. See "HCPB Beryllium Override" section. |
| `ceramic_only` | 3.0 | WCCB. Li-ceramic pebbles without Be multiplier. ~$45M for 300 t at $150/kg synthesis. |
| `none` | 0.0 | Aneutronic, no breeder. |

### `BlanketForm.structure_factor` (CAS22.01)

| BlanketForm | structure_factor | Source / rationale |
|---|---:|---|
| `liquid_metal` | 1.0 | Baseline. RAFM steel flow channels + W FW armor. |
| `molten_salt` | 1.3 | Hastelloy-N corrosion liner on FLiBe-wetted surfaces. Source: INEEL/EXT-99-00331. |
| `solid_breeder` | 1.2 | Pebble-bed canisters: separate breeder/multiplier zones, He coolant manifolds. Source: EUROfusion HCPB cost basis. |
| `none` | 0.0 | No blanket structure. |

### Calibration caveat

These are first-pass values calibrated against limited public data. Spread is
wide for some entries (e.g., `li` 0.8-10x, `flibe` 2-20x) driven by enrichment
choices and supply assumptions. Users should override the multipliers via
`cost_overrides={"CAS27": ...}` for project-specific sensitivity analyses, the
same way they would for any other coefficient with a documented uncertainty
band.
