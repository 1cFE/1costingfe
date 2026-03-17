# pB11 Side Reactions and NRC Regulatory Risk Assessment

**Date:** 2026-03-07
**Status:** Analysis (updates DI-016 in light of Feb 2026 proposed rule)
**Related research:** `fusion-tea/knowledge/research/approved/20260203-100000_fusion-regulatory-framework-dt-pb11.md`

---

## Background

Our previous analysis (performed February 03 2026) concluded that pB11 fusion "likely falls outside NRC jurisdiction" because the primary reaction (p + 11B -> 3 He-4 + 8.7 MeV) produces only stable helium and no byproduct material under the pre-2026 definition in 10 CFR 30.4. This assessment needs revision in light the NRC's proposed rule on fusion machines (Federal Register, Feb 26, 2026)

---

## The Side Reactions

The primary pB11 reaction is aneutronic. However, two secondary reactions produce neutrons:

### 1. 11B(alpha,n)14N (dominant)

- **Q = +0.158 MeV** (exothermic)
- Each p+11B event produces 3 alpha particles at ~2.89 MeV each
- Each alpha has probability ~10^-3 of undergoing 11B(alpha,n)14N while still in the plasma
- Net: ~3 x 10^-3 neutron-producing reactions per primary fusion event
- Neutron energy: ~2.18 MeV (lab frame, from CM kinematics)

This reaction is **unavoidable by design**: the alphas and boron coexist in the plasma. As Ochs, Kolmes, and Fisch (2025) show, alpha particles must be managed carefully in any pB11 reactor — if confined too long, they poison the plasma through pressure buildup and bremsstrahlung losses. But while they remain confined, they undergo side reactions with boron.

The key constraint from Ochs et al. is that any working pB11 reactor must remove alpha ash faster than it accumulates, meaning alpha confinement time must be shorter than fuel confinement time. This limits but does not eliminate the side reaction window — the alphas exist at fusion-relevant energies for long enough to produce some neutrons before extraction.

### 2. 11B(p,n)11C (minor)

- **Q = -2.765 MeV** (endothermic, threshold 3.02 MeV)
- Probability ~10^-5 per proton — negligible compared to alpha-n channel
- Neutron energy: ~0.17 MeV near threshold

### Net Neutron Production

Literature consensus (multiple sources):
- **~0.1% of reactions produce neutrons**
- Neutron energy carries **<0.2% of total fusion power**
- This is ~10,000x fewer neutrons per unit fusion power than DT

These neutrons are low-energy (~2 MeV) compared to DT (14.1 MeV), producing significantly less structural activation per neutron. However, they are not zero, and they will activate structural materials over the lifetime of a power plant.

---

## The Regulatory Landscape (as of March 2026)

### What Changed: The Feb 2026 Proposed Rule

On February 26, 2026, the NRC published a proposed rule titled ["Regulatory Framework for Fusion Machines"](https://www.federalregister.gov/documents/2026/02/26/2026-03865/regulatory-framework-for-fusion-machines). Key provisions relevant to pB11:

**1. Broad definition of "fusion machine":**

> A machine that is capable of (1) transforming atomic nuclei, through fusion processes, into different elements, isotopes, or other particles; and (2) directly capturing and using the resultant products, including particles, heat, or other electromagnetic radiation.

This is fuel-agnostic. A pB11 machine unambiguously meets this definition.

**2. pB11 explicitly named as in-scope:**

The proposed rule states it is "intended to address" configurations with fuels including "deuterium-tritium, deuterium-helium-3, and proton-boron-11." This directly contradicts the our previous work's assumption that pB11 would be outside NRC jurisdiction.

**3. Amended byproduct material definition:**

The ADVANCE Act of 2024 (Section 205) amended the Atomic Energy Act to include radioactive material produced by fusion machines as byproduct material. The proposed rule implements this by amending the definition across multiple parts of Title 10. Even small quantities of neutron-activated structural material from pB11 side reactions now formally qualify as byproduct material.

**4. No tiered approach by neutron yield:**

The proposed rule does not differentiate requirements based on neutron production levels. All "near-term" fusion machines face the same Part 30 licensing framework, characterized as "performance-based, technology-inclusive, risk-informed."

**5. No explicit exemption for aneutronic or low-neutron approaches:**

Despite the rule claiming to be "risk-informed," no de minimis threshold or categorical exclusion exists for fusion machines that produce very few neutrons.

### Industry Response

[TAE Technologies](https://www.nrc.gov/docs/ML2415/ML24157A328.pdf) (the leading pB11 fusion company) submitted comments arguing that:
- pB11 creates "significantly less activated material than other approaches"
- Requirements referencing tritium should not apply to systems that do not use or produce tritium
- Failure to differentiate would create "a perverse market disincentive" against lower-risk approaches
- They urged a "risk-informed, technology neutral approach" that recognizes the radiological differences

The comment period for the proposed rule runs through **May 27, 2026**, with the final rule deadline of **December 31, 2027** (per NEIMA/ADVANCE Act).

---

## Likelihood Assessment

| Scenario | Likelihood | Description |
|----------|-----------|-------------|
| **A. pB11 fully exempt from NRC** | **15-25%** | NRC carves out an exemption for fusion machines below some activation threshold, or narrows the definition to exclude machines that don't produce tritium. Requires significant rule modification during comment period. |
| **B. Part 30 with graded/reduced requirements** | **50-60%** | pB11 is formally under Part 30 but the "risk-informed" approach leads to minimal practical requirements — perhaps a general license rather than specific license, simplified safety analysis, no tritium-specific requirements. The activation products from ~0.1% neutron fraction may fall below Schedule C thresholds for most radionuclides. |
| **C. Part 30 with same requirements as DT** | **15-25%** | The final rule applies uniform requirements regardless of fuel type. pB11 operators must demonstrate compliance with all Part 30 provisions, though many (tritium handling, breeding blanket safety) would be trivially satisfied. |

### Rationale

**Against full exemption (Scenario A):**
- The proposed rule text explicitly names pB11 as in-scope
- The ADVANCE Act definition is broad — any radioactive material from a fusion machine is byproduct material
- The 11B(alpha,n)14N reaction genuinely produces neutrons; "aneutronic" is a misnomer in the strict sense
- NRC has institutional incentives to maintain jurisdiction

**Against uniform treatment (Scenario C):**
- The rule claims to be "risk-informed" and "performance-based"
- The radiological difference is genuinely enormous (~10,000x fewer neutrons than DT)
- TAE and other stakeholders will push hard during comment period
- NRC guidance documents (NUREG-1556 Vol. 22) can implement graded requirements even if the rule text doesn't explicitly create tiers
- Activation products from 2 MeV neutrons at 0.1% reaction fraction may literally fall below exempt quantities for many radionuclides

**For the middle ground (Scenario B):**
- This is the NRC's standard approach — uniform framework with risk-informed implementation
- Part 30 already has mechanisms for general licenses and exempt quantities
- The activation product inventory from a pB11 plant would be orders of magnitude below DT
- A general license or simplified specific license would satisfy both NRC jurisdiction and proportionality

---

## Implications for Cost Modeling

The current 1costingfe model uses:
- `licensing_time_pb11 = 0.0 yr` (no NRC licensing)
- `licensing_cost_pb11 = 0.1 M$` (nominal)

Under the most likely scenario (B), more defensible values would be:
- `licensing_time_pb11 = 0.25-0.5 yr` (streamlined Part 30)
- `licensing_cost_pb11 = 0.5-1.0 M$` (reduced but nonzero)

However, the impact on LCOE is small because:
1. CAS10 is typically <1% of total capital cost
2. The licensing time difference (0 vs 0.5 yr) affects IDC marginally
3. The dominant pB11 cost drivers are plasma physics feasibility (Lawson criterion at ~300 keV), not regulatory burden

---

## Open Questions

1. **Will the final rule include a graded approach?** The comment period (through May 27, 2026) and subsequent rulemaking will determine this. TAE's comments and others arguing for risk-informed differentiation may succeed.

2. **What are the actual activation product inventories?** A quantitative assessment of structural activation from ~0.1% neutron fraction at ~2 MeV, over a 30-year plant lifetime, would determine whether exempt quantity thresholds in 10 CFR 30.71 Schedule B are exceeded. This is a calculable number but depends on structural materials and neutron flux.

3. **General license vs specific license?** If activation products fall below certain thresholds, a pB11 fusion machine might qualify for a general license under 10 CFR 31 rather than a specific license under Part 30 — dramatically reducing regulatory burden while maintaining NRC jurisdiction.

4. **State vs federal jurisdiction?** Some states are Agreement States that regulate byproduct material under their own programs. The interaction between the new fusion rule and state programs adds complexity.

---

## Recommendation

DI-016's conclusion that pB11 is "likely outside NRC jurisdiction" should be updated to reflect the Feb 2026 proposed rule. The most defensible position is now:

> **pB11 fusion will likely fall under NRC Part 30 jurisdiction due to the ADVANCE Act's broad byproduct material definition and the proposed rule's explicit inclusion of pB11. However, the practical regulatory burden will be minimal — significantly less than DT or even DHe3 — because the risk-informed framework should result in streamlined requirements proportional to the very low radiological hazard.**

The 11B(alpha,n)14N side reaction is the physical basis for NRC jurisdiction. It is unavoidable (alphas and boron coexist in the plasma) and produces real, if small, quantities of activated material. The question is not whether NRC has jurisdiction — the proposed rule settles that — but how much practical burden that jurisdiction entails.

---

## Sources

- [NRC Proposed Rule: Regulatory Framework for Fusion Machines (Feb 26, 2026)](https://www.federalregister.gov/documents/2026/02/26/2026-03865/regulatory-framework-for-fusion-machines)
- [NRC Fusion Machine Rulemaking Status](https://www.nrc.gov/materials/fusion/rulemaking-status)
- [NRC Fusion FAQ](https://www.nrc.gov/materials/fusion/faq)
- [Morgan Lewis: NRC Proposes Regulatory Framework for Fusion (March 2026)](https://www.morganlewis.com/blogs/upandatom/2026/03/nrc-proposes-regulatory-framework-for-fusion)
- [Pillsbury: NRC Proposed Rule Establishes Licensing Framework](https://www.pillsburylaw.com/en/news-and-insights/nrc-licensing-framework-fusion-machines.html)
- [POWER Magazine: NRC Proposes First Dedicated Regulatory Framework](https://www.powermag.com/nrc-proposes-first-dedicated-regulatory-framework-for-commercial-fusion-machines/)
- [ANS Nuclear Newswire: NRC Opens Comment Period (March 2026)](https://www.ans.org/news/2026-03-03/article-7812/nrc-opens-comment-period-for-fusion-regulatory-changes/)
- [TAE Technologies Letter to NRC (May 2024)](https://www.nrc.gov/docs/ML2415/ML24157A328.pdf)
- [Ochs, Kolmes, Fisch — "Preventing ash from poisoning proton-boron 11 fusion plasmas," Phys. Plasmas 32, 052506 (2025)](https://pubs.aip.org/aip/pop/article/32/5/052506/3347125/Preventing-ash-from-poisoning-proton-boron-11)
- [CRCPD Technical White Paper: State Regulation of Fusion Machines (2025)](https://crcpd.org/wp-content/uploads/2025/08/25-2-Technical-White-Paper-State-Regulation-on-Fusion-Machines.pdf)
