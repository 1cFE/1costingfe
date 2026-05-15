# Direct Energy Conversion for Fusion: Fuel, Confinement, and the Cost Question

Every fusion power plant has to turn plasma energy into electricity. The default path is thermal: fusion power heats a working fluid that runs through a turbine to spin a generator. Turbines and generators convert heat to power in virtually every coal, gas, and fission plant on the grid. These are mature technology with known capital costs.

The thermal conversion default has two drawbacks. Physics sets a low efficiency ceiling, and its hardware is expensive. Direct energy conversion (DEC) is used loosely here for any architecture that skips the heat cycle, the turbine, or both. Solar photovoltaics (PV) skip both, turning photons directly into current. Wind and hydro skip just the heat cycle, converting kinetic energy through a turbine to electricity. MHD generators skip just the turbine, extracting current from a conducting fluid in a magnetic field. Charged-particle DEC skips both, turning the kinetic energy of plasma particles directly into current by decelerating them against an electric potential, or by inducing current in surrounding coils as the plasma expands.

![Power flow in a 1 GWe p-B11 thermal-only mirror: 83% of fusion power (1,990 MW) is radiated as bremsstrahlung, 17% (448 MW) escapes as charged-particle transport, and both feed a single sCO2 turbine at 47%.](<p-B11 Thermal Cycle.svg>)

The thermodynamic motivation for DEC is clear. Charged-particle DEC interacts directly with the plasma, whose temperature is 10⁸-10⁹ K, so the Carnot ceiling on efficiency η ≤ 1 − T_cold/T_hot is essentially 100%. The thermal default is bounded by what first-wall and blanket materials tolerate (500-700°C for RAFM steel and SiC composites), capping the Carnot limit around 60-65% against a 300 K cold sink. In practice, real thermal cycles lose another 10-15 percentage points to heat-exchanger and turbine irreversibilities, which is why sCO2 Brayton sits at 47% and combined cycle at 53%. Similar considerations inevitably reduce the efficiency of any DEC scheme as well.

Different confinements need different capture hardware: venetian-blind grids for mirrors, inductive coils for pulsed concepts. This post walks through two charged-particle DEC paths, what each would need to deliver, and where the room to run actually is. The costing framework builds on our [prior analysis](https://1cf.energy/fusions-cost-floor-what-if-the-core-were-free/) of the industrial floor on fusion electricity cost (the LCOE remaining when the fusion core is treated as free); we reference it when free-core numbers appear, but this piece stands alone.

## Fusion Fuel and Direct Conversion

Fusion DEC only works on charged particles. Neutrons carry no charge, and photons at keV bremsstrahlung energies have no demonstrated DEC path; TAE's proposed X-ray capture scheme via cascading Auger emission ([Binderbauer & Tajima, 2018](https://patents.google.com/patent/US9893226B2/)) remains untested (see "p-B11's Bremsstrahlung Constraint" below). DEC therefore favors fusion reactions that emit a small fraction of the energy in neutrons.

Photon emission from a fusion plasma comes from plasma dynamics, not nuclear physics: electrons radiate bremsstrahlung as they scatter off ions. At fusion temperatures it becomes a debilitating loss channel. For p-B11, the high plasma temperature (>100 keV) and Z=5 boron drive intense bremsstrahlung: roughly 97% of fusion power in a thermonuclear plasma, leaving a 3% margin ([Putvinski et al., 2019](https://doi.org/10.1088/1741-4326/ab1a60)). Driving the plasma out of equilibrium widens the margin to about 17% ([Ochs et al., 2022](https://doi.org/10.1103/PhysRevE.106.055215)), but photons still carry 83% of the fusion power, and they hit the walls as heat. A thermal plant is the clear candidate to convert this heat to electricity.

D-T puts 20% of its energy in 3.5 MeV alphas, a charged fraction that nominally exceeds p-B11's 17% non-equilibrium margin. From the DEC standpoint D-T resembles p-B11: the bulk of the energy (80% in 14.1 MeV neutrons rather than bremsstrahlung) forces a full thermal plant, so DEC could at best be a small bottoming overlay on the alpha channel. D-T also carries the radioactivity burden p-B11 avoids: neutron activation of structural materials, lithium breeding blankets, and tritium inventory handling all raise plant cost without changing the DEC arithmetic. D-T is fundamentally a thermal fuel, and we do not consider it further here.

D-He3 is the best DEC fuel by these criteria. The primary D-He3 reaction is aneutronic, but D-D side reactions produce neutrons and tritium. The output split depends on temperature, D/³He ratio, and confinement time. In a [steady-state plasma](https://global.oup.com/academic/product/tokamaks-9780199592234) at 50/50 mix and T = 70 keV, the bred tritium fuses with D and adds 14 MeV neutrons. A magneto-inertial plasmoid (Helion-style) holds the plasma too briefly for the secondary D-T reaction; the unburned tritium is exhausted, eliminating the 14 MeV neutron channel. The D/³He mix can then be chosen to minimize bremsstrahlung: D-rich (n_D/n_³He = 3.2) at T = 100 keV is the Helion-likely operating point. The figure below contrasts the two cases; see the [companion script](https://github.com/1cfe/1costingfe/blob/master/examples/dhe3_mix_optimization.py) for the Bosch-Hale + relativistic-bremsstrahlung calculation.

![Fractional split of fusion energy by output channel, by fuel and operating point.](<energy_channel_distribution.svg>)

## Two Paths to Fusion DEC

### Venetian Blind

The [venetian blind](https://iopscience.iop.org/article/10.1088/0029-5515/13/1/005) is the most mature DEC concept (TRL 4-5). It consists of angled metal ribbon grids at successively higher retarding potentials that sort ions by energy and collect them on high-potential electrodes. It mounts on linear magnetic confinement devices such as magnetic mirrors, as an add-on to the expander tank, collecting the particle stream leaving along field lines. It has no moving parts, and was [demonstrated at 48% efficiency](https://doi.org/10.13182/FST83-A20820) on real end-loss plasma at LLNL's TMX (Barr & Moir, 1983), with a [theoretical maximum of 70%](https://iopscience.iop.org/article/10.1088/0029-5515/13/1/005). The 70% cap is set by grid geometry and the ion energy spread, not by the wall-material Carnot limit. [Realta Fusion](https://www.realtafusion.com/) is the sole active developer, with a [patented](https://patents.google.com/patent/US12166398B2) axisymmetric variant.

### Pulsed Inductive

Pulsed inductive DEC is a less-mature DEC concept (TRL 3 for fusion use). It consists of coils surrounding a compression chamber that compress a pulsed plasma and then recover energy as it re-expands, inducing current via Faraday's law, requiring no plasma-surface contact; the same coils drive both halves of the cycle. No published demonstration on fusion plasma yet. [Helion Energy](https://www.helionenergy.com/) is the most prominent [active developer](https://doi.org/10.1088/0029-5515/51/5/053008), using D-He3 FRC plasmoids that collide and merge in the compression chamber. They [claim 95%](https://www.helionenergy.com/faq/) electrical round-trip efficiency on the capacitor → magnetic field → recovery-capacitor loop, measured on early machines without plasma present.

### Honorable Mention: MHD Generator

A [magnetohydrodynamic generator](https://inis.iaea.org/records/4ss3z-rqd83) passes a conducting fluid through a transverse magnetic field, driving current through electrodes via the Lorentz force. No moving parts. From the 1960s through 1993, the US, Soviet, Japanese, and Australian governments [spent hundreds of millions](https://www.gao.gov/products/emd-80-14) developing MHD generators on hot coal combustion gas, demonstrating up to [32 MW gross output](https://www.osti.gov/biblio/6380343) at TRL 5-6. Liquid-metal MHD (LMMHD) has separately shown [>71% efficiency](https://doi.org/10.1016/0196-8904(81)90006-6) at small scale; projected efficiency for LMMHD on fusion coolants is >60%.

MHD generator is not a charged-particle DEC; it replaces the Brayton turbine, not the heat cycle, and is still Carnot-limited. Any fusion plant could route its coolant through an MHD channel instead of a turbine. For D-T, LMMHD integrated into the lithium breeding blanket handles neutron and thermal energy. For aneutronic fuels, it handles the bremsstrahlung fraction that charged-particle DEC cannot touch.

Fusion MHD has costed design heritage. The Compact Fusion Advanced Rankine (CFAR) concept, developed in the late 1980s by an LLNL/Kyoto/TIT collaboration ([Hoffman, Campbell, & Logan, 1989](https://www.osti.gov/biblio/6276900); [CFAR-II inertial variant](https://www.sciencedirect.com/science/article/abs/pii/092037969390114W)), pairs a high-temperature mercury Rankine loop and microwave superheater with a nonequilibrium MHD disk generator, packaged to fit almost entirely inside the reactor vault. The MFE variant reported 24-30% net plant efficiency at 2500-3250 K generator inlet; CFAR-II, with a vaporized blanket at 10⁴ K, projected 38%. Both require bespoke blankets at or past current materials limits, and neither was built. CFAR-II carries a costed 22-subsystem BoP model with direct cost of 30-40 M$/GWe (1992 dollars), the most explicit fusion-MHD BoP cost reference on record, though that figure does not price the blanket development gap, and the projected efficiencies sit below the >60% modern LMMHD target.

Folded into this post's framework, in-vault integration could trim on the order of $200M of BoP and buildings capex at 1 GWe. CFAR-II's 30-40 M$/GWe BoP (1992$) inflates to about $70-90M in 2024 dollars, against the blog's $168M sCO2 turbine island; net conversion-swap saving is $80-100M, with the rest from turbine-hall and piping, for a total LCOE saving of roughly $3/MWh, assuming blanket costs are the same. This may understate CFAR's intended impact. The original motivation was that a fusion core paired with a conventional BoP cannot reach the cost per MWh of fission, gas, or coal because the core is intrinsically expensive, so drastically shrinking the BoP is the path to competitiveness, not a marginal optimization on an already-competitive plant.

The remaining obstacle is that no fusion-MHD system has ever been built. The electrode-erosion problem that killed the coal programs doesn't apply to clean fusion coolants. For LMMHD integrated into a breeding blanket (the modern [ORNL and KIT line of work](https://doi.org/10.1080/15361055.2022.2116905)), the pressure drop of liquid metal flowing through fusion-scale magnetic fields (5-15 T) may exceed the generated power, and no reactor-scale demonstration exists. CFAR avoids this by topology: its disk generator sits outside the toroidal field coils in a region where the field has dropped off rapidly, with dedicated ring magnets, and the 1989 design study reported cycle efficiency insensitive to blanket pressure drops up to 50%. MHD therefore remains a what-if at the hardware level (no operational fusion demonstration).

## p-B11's Bremsstrahlung Constraint

p-B11 is famous for its bremsstrahlung losses, and DEC can't collect them. Even in the most favorable scenario, bremsstrahlung is 83% of fusion power and the charged-particle margin is only 17% (3% in thermonuclear operation). The thermal plant is not a small bottoming cycle in a p-B11 plant; it is the primary power conversion pathway, and DEC captures only the margin.

The options for the radiated fraction are:

1. **Thermal cycle**: a full turbine island converts bremsstrahlung heat at 47–53%. This is what the "hybrid" configurations modeled below assume. It works but preserves essentially the full thermal BOP.
2. **MHD on the coolant**: if the bremsstrahlung-heated wall coolant is liquid metal, an MHD channel could convert at >60% with no turbine. Undemonstrated at reactor scale.
3. **Reject as waste**: accept the efficiency loss and dump the bremsstrahlung heat to cooling towers. At 83% bremsstrahlung fraction, this means wasting most of the fusion energy. The overall plant efficiency collapses: a DEC at 85% efficiency on the 17% net charged fraction yields only 14% of fusion power as electricity, far worse than thermal-only (47%).
4. **X-ray photovoltaic DEC**: convert bremsstrahlung directly to electricity via cascading Auger emission in nanometric high-Z/low-Z layers ([Binderbauer & Tajima, 2018](https://patents.google.com/patent/US9893226B2/)). If it works, this would capture the bremsstrahlung fraction without a thermal cycle, but no prototype has been built, no efficiency has been measured, and the radiation damage environment (keV photons at GW/m² flux for years) is extreme. This is a TRL 1 concept.

Options 3 and 4 do not have a demonstrated path. Today, the thermal plant is the only demonstrated path for the radiated fraction. DEC, where it applies, captures only the charged-particle margin.

Synchrotron is the other photon channel, and it does not impose a comparable constraint. The fundamental electron cyclotron frequency at 10 T is 280 GHz (1.2 × 10⁻³ eV), with harmonics extending into the THz band at fusion electron temperatures, six orders of magnitude below bremsstrahlung. Mm-wave synchrotron reflects at >99% from polished metallic first walls and is typically recycled into the plasma rather than captured; the few-percent escaping fraction could in principle feed a rectenna using mature GHz-band hardware, but the upside is bounded.

## Modeled Approaches

We modeled two DEC architectures (venetian blind hybrid and pulsed inductive) with a free fusion core, using the same methodology as the [previous dispatch](https://1cf.energy/fusions-cost-floor-what-if-the-core-were-free/). The "floor" is the LCOE that remains after zeroing out the fusion core: the irreducible balance-of-plant, buildings, O&M, and financing cost a fusion plant carries even if the heat source were free. Baseline conditions throughout this post are 1 GWe net, 85% availability, 7% WACC, 30-year life, and 6-year construction; aggressive conditions push these to 2 GWe, 95%, 3% WACC, 50 years, and 3 years. Numbers can be reproduced with the [companion script](https://github.com/1cfe/1costingfe/blob/master/examples/dec_blog_numbers.py).

One caveat up front: the thermal capture hardware in the baseline (turbine island, generator, sCO2 Brayton loop) draws on a century of manufacturing cost-down, thousands of GW deployed, and well-understood NOAK pricing. The DEC estimates rest on much thinner ground: 1977 EPRI/LLNL studies for the venetian blind, and a single developer's capacitor build-up for the pulsed inductive. Both sit closer to FOAK than the turbine they displace. The cost tables below compare them head-to-head today; the Learning Rates section returns to what DEC could look like once that gap closes.

### Venetian Blind + Thermal Hybrid  (p-B11)

We carry the p-B11 mirror baseline forward from the [prior dispatch](https://1cf.energy/fusions-cost-floor-what-if-the-core-were-free/) and add a venetian blind as an overlay. The thermal cycle is kept because 83% of p-B11 energy radiates as bremsstrahlung; the DEC captures only the charged particles leaving the mirror axially.

Sizing the VB hardware to 400 MWe DEC output (2024 NOAK) lands at roughly $125M all-in, range $90-175M, comparable to the turbine island it partially replaces ($168M for sCO2 at 1 GWe). The breakdown:

- **High-voltage power conditioning**: $45M, $100-120/kW, 60-70% of a [MISO 2024 VSC HVDC station](https://cdn.misoenergy.org/20240501%20PSC%20Item%2004%20MISO%20Transmission%20Cost%20Estimation%20Guide%20for%20MTEP24632680.pdf) since the AC interconnect isn't duplicated
- **Cryopumps**: $15M, fusion procurement scale, scaled from the [ITER torus and cryostat cryopumps](https://fusionforenergy.europa.eu/news/europe-delivers-eight-iter-cryopumps/) (F4E delivered 8 units, EUR 21M F4E investment)
- **Thermal panels**: $15M, hypervapotron-cooled tungsten panels for the 30-90 MW grid heat load, design analog to the [ITER NBI residual ion dump and calorimeter](https://iopscience.iop.org/article/10.1088/1367-2630/19/2/025005) (Hemsworth et al. 2017)
- **Grid and collector hardware**: $15M, geometry from [Hoffman 1977](https://iopscience.iop.org/article/10.1088/0029-5515/13/1/005) escalated by CPI, cross-checked against the TMX integrated 48% demonstration ([Barr & Moir 1983](https://doi.org/10.13182/FST83-A20820))
- **HV bushings, valves, controls**: $12M
- **Vacuum tank**: $7M, 304L stainless, 300-500 tonnes (10-15 m diameter, 20-40 m long, 10-15 mm wall + ring stiffeners). Same dimensional class as commercial refinery hydrocracker reactors ([ExxonMobil Rotterdam](https://www.exxonmobil.com/en/basestocks/news-insights-and-resources/exxonmobil-rotterdam-refinery-hydrocracker-reactors), 4.5 m diameter, 25-30 m tall), where ASME Section VIII Div 2 vessels at 200 bar internal pressure run $26-30M; the VB tank only needs to hold vacuum (1 atm external collapse load), so it sits well below that price.
- Plus 15% installation

We model the hybrid with the DEC capturing 90% of the charged-particle transport. This is a geometric assumption that 10% of particles do not reach the venetian blind, e.g. lost radially.

![Floor LCOE and overnight capital for a 1 GWe p-B11 plant with a free fusion core: thermal only vs venetian-blind hybrids at 48% and 60% DEC efficiency.](<p-B11 free-core table.svg>)

The venetian blind doesn't lower the floor. With 83% of fusion energy radiated as bremsstrahlung, the DEC captures 90% of the 17% charged-particle margin. At the demonstrated 48% efficiency, the DEC hardware adds cost without meaningfully improving conversion. Even at 60% (never demonstrated on real plasma), the hybrid floor sits at $18/MWh, $1/MWh above thermal-only at $17/MWh, and adds about $25-30/kW on overnight capital. Eliminating the turbine entirely and wasting the bremsstrahlung is not viable for p-B11: without a thermal cycle, the plant needs 16 GW of fusion power for 1 GWe net, driving the fully costed LCOE far above the thermal path ($42/MWh).

The numbers above use the mirror baseline heating power of 40 MW. If the heating power is larger, the picture shifts. The heating power leaves the plasma in the charged-particle transport channel, and is mostly captured by the VB. But at a heating wall-plug efficiency of 50% (the model's NBI default) the injection chain draws 2 MW of gross electric per 1 MW delivered to the plasma, while the transport channel returns only 0.58 MW in the VB 60% hybrid. Each MW of heating therefore costs roughly 1.4 MW of gross electric output, and the fusion core upsizes to cover the shortfall. Holding net output at 1 GWe, LCOE rises by $2-6/MWh as heating goes from 150 to 400 MW; beyond 600 MW the recirculating fraction exceeds 50%. Thermal-only stays cheaper than the thermal+VB hybrid by about $0.9/MWh at every heating level: the VB hardware cost scales with DEC throughput alongside the recovery it provides, so the gap stays roughly flat across the sweep.

Flipping the analysis: at fixed VB hardware cost, what efficiency would justify adding the DEC? At the 1 GWe p-B11 baseline the VB outputs roughly 232 MWe, from 90% capture of the 429 MW charged-particle margin at 60% conversion, and the model scales the VB hardware to about $85M at this plant scale. Holding that fixed, no VB efficiency between 30% and 100% breaks even with thermal-only; the VB hardware exceeds turbine-side savings across the entire physically accessible band. Break-even enters the physically accessible window only if the VB capex at this plant scale falls by roughly 3x, to around $30M, at which point a 54% VB suffices. The leverage is narrow: only 15% of fusion power flows through the DEC, so each percentage point of VB efficiency moves only 0.15 percentage points of plant-level conversion.

![Power flow in a 1 GWe p-B11 mirror with a venetian-blind DEC at 60% hybrid with a thermal cycle: 83% of fusion power (1,901 MW) radiates as bremsstrahlung and feeds the turbine, while the DEC captures only the 17% charged-particle margin (386 MW).](<p-B11 VB DEC.svg>)

### Pulsed Inductive (D-He3)

Pulsed inductive DEC using D-He3 FRC plasmoids is Helion's concept; Polaris is currently under construction to test it on plasma. At the Helion-likely operating point (D-rich mix at T = 100 keV, no T burnup), 81% of the fusion energy is in charged particles that drive the inductive coils. Unlike the VB hybrid, no main turbine is retained and no thermal bottoming cycle is added; the pulsed-power chain is the only conversion stage, and the bremsstrahlung and neutron fractions are dumped to cooling as waste heat.

With no turbine, the BOP becomes a pulsed-power chain: pulse-rated switchgear, capacitor or inductive storage to smooth the duty cycle, DC-DC converters to recharge the compression coils, and a grid-tie inverter at the back. Sizing the pulsed chain downstream of the driver to 1 GWe net (NOAK) lands at roughly $593M ($593/kW). The breakdown:

- **Recovery cap bank**: $351M, sized to the fusion power balance (703 MJ at $0.50/J)
- **VSC HVDC valve hall + IGBT grid-tie**: $150M, $150/kW net, set by industrial [HVDC station pricing](https://cdn.misoenergy.org/20240501%20PSC%20Item%2004%20MISO%20Transmission%20Cost%20Estimation%20Guide%20for%20MTEP24632680.pdf) given pulse-rated topology and high-voltage interconnect
- **DC-DC links**: $78M, $75/kW of gross output
- **Controls**: $14M, 4% of the driver bank

The compression bank itself ($351M) is accounted for as part of the driver (core).

![BOP floor, He-3 fuel, and total LCOE for a 1 GWe D-He3 plant with a free fusion core: thermal only vs pulsed inductive at 85% round-trip efficiency.](<D-He3 free-core table.svg>)

The BOP floor comes in $6/MWh above thermal: $24/MWh vs $18/MWh. The turbine island is not the dominant cost. Buildings ($354M), the electrical plant ($96M), O&M ($37M/yr), and financing charges together dwarf the turbine, and the pulsed-power chain that replaces it costs more than the turbine. Eliminating the turbine and its synchronous generator (including the GSU transformer and sync/protection gear) removes about $186M of overnight capital, but the $593M of pulsed-power hardware needed to shape, store, and grid-tie the recovered energy more than offsets those savings.

The He-3 fuel cost is a different problem. At $2M/kg (optimistic), He-3 adds $36/MWh, roughly four times the 1-cent target by itself. At the [DOE-allocated price](https://www.everycrsreport.com/reports/R41419.html) of $4.5M/kg, it roughly doubles. Helion's answer is to run a D-rich mix that maximizes D-D-bred He-3 and recover the He-3 between shots at assumed 99% efficiency; tritium from D-D is exhausted (or held until decay) rather than re-burned. If the fuel cost drops to zero, the D-He3 pulsed DEC floor sits at $24/MWh, still above thermal ($18/MWh) and well above the 1-cent target.

Running the Helion-likely D-rich mix saves about $20/MWh of LCOE for pulsed inductive over the textbook 50/50 reference, almost entirely through reduced He-3 burn: more D-D side reactions produce more internal He-3 that the recovery loop re-injects, lowering external He-3 demand by about a quarter. Mix selection moves less than architecture choice (which moves $38/MWh between thermal and pulsed inductive), but it is a substantial lever at He-3 prices of $2M/kg.

The plasma-present round-trip efficiency is the other open question. The [theoretical framework](https://youtu.be/5nHmqk1cI2E?t=505) (Kirtley et al., APS DPP 2024) suggests that 85% efficiency is a realistic target for Helion's Diesel-cycle plan. At that level, the reduction of required fusion power by about a third and external He-3 burn by more than half is what carries the architecture, not the BOP hardware, which already costs more than a turbine.

How much cheaper does the pulsed-power hardware need to be for Helion to compete on price? Assuming Helion's stated goal of self-bred He-3 (fuel cost dropping to roughly zero), the Helion-likely LCOE falls from $81/MWh to about $57/MWh, in the upper range of gas and nuclear ($50-70/MWh). The pulsed-power chain at the modeled $593M contributes about $10/MWh of that. Making it free saves $10/MWh and brings Helion to $47/MWh. The floor below that is set by buildings, electrical plant, O&M, and financing, not the cap bank. Closing to p-B11 thermal ($42/MWh) requires squeezing those non-pulsed-power line items as well; closing to the 1-cent target is out of reach by hardware learning alone.

![Power flow in a 1 GWe D-He3 plant at the Helion-likely operating point (D-rich mix, T = 100 keV, T exhausted, 99% inter-shot He-3 recovery, no thermal bottoming): 1,348 MW of charged-particle transport drives the pulsed inductive DEC at 85%; bremsstrahlung (276 MW), neutrons (39 MW), and DEC waste are dumped to cooling. p_fus = 1,695 MW. The textbook 50/50 reference (24% brem, 6% neutrons) appears as the D-He3 column in the energy-channel figure above.](<Helion Pulsed Inductive.svg>)

## The Floor Is Industrial, Not Architectural

Pushing every parameter to aggressive limits (2 GWe, 95% availability, 3% WACC, 50-year life, 3-year construction) collapses the gap between thermal and DEC, for both fuels:

![LCOE under aggressive conditions (2 GWe, 95% availability, 3% WACC, 50-year life, 3-year construction) for p-B11, D-He3 with purchased He-3, and D-He3 with self-bred He-3.](<Aggressive 2 GWe table.svg>)

For p-B11, hybrid DEC and thermal land within $0.2/MWh: the DEC overlay costs about as much as the thermal hardware it partially replaces. (Pulsed inductive is not viable for p-B11: wasting 83% of fusion energy as bremsstrahlung without a thermal cycle requires an enormous core that more than offsets the turbine savings.) For D-He3, He-3 fuel at $2M/kg dominates everything else; self-breeding closes that gap and pulsed inductive then ends up $6/MWh higher than thermal because the pulsed-power chain is more expensive than the turbine.

The floor is set by the industrial cost of building, staffing, and financing a large power plant, not by the power conversion choice. DEC's value, if any, has to show up on the costed core or the fuel bill.

DEC has a separate role that does not appear in the floor analysis: enabling breakeven. Where the recovery stage is intrinsic to the cycle (Helion-style pulsed inductive, with no thermal-only fallback), DEC efficiency is the gating condition for net electric output, not a cost knob. A plant below some round-trip threshold produces no net electricity at all; above it, the plant works but operates at high recirculation by construction, and LCOE tracks the small net output rather than the large gross. Enabling and economic are different bars; clearing the first does not clear the second.

## Where DEC Pays Off

The floor represents the LCOE of a fusion plant with a free fusion core. Since the core is not free, DEC's value emerges through two mechanisms: (1) higher conversion efficiency reduces the fusion power required for a given net electric output, lowering the costed-core capital; and (2) for expensive fuels, lower fusion power also reduces the fuel cost. The two fuels respond very differently to these mechanisms.

### D-He3: a fuel lever

In the fully costed D-He3 plant at 1 GWe baseline conditions:

![Fully-costed D-He3 at 1 GWe baseline: LCOE, core cost, and He-3 fuel cost for thermal-only magnetic mirror, venetian-blind hybrid, and pulsed inductive. Pulsed inductive cuts LCOE primarily through reduced He-3 demand.](<D-He3 fully-costed table.svg>)

For the venetian blind, the core cost rises modestly ($22M): the added DEC hardware partially offsets the savings from a smaller heating system. For pulsed inductive, the core jumps about $380M because the pulsed-power chain that replaces the turbine (dominated by the recovery cap bank) is substantial. LCOE drops $10/MWh with the venetian blind and $38/MWh with pulsed inductive, but the pulsed savings come mostly from fuel: higher efficiency, the D-rich operating mix, and active He-3 recovery between shots all reduce external He-3 demand. For D-He3, DEC is primarily a fuel-saving technology.

### p-B11: no fuel lever

For p-B11 (negligible fuel cost), the second channel disappears, and only the core-shrinkage channel remains. Because bremsstrahlung dominates the fusion energy split, even an efficient DEC captures only the 17% charged-particle margin and does not meaningfully reduce the required fusion power. Thermal LCOE is $42/MWh; VB DEC 60% is $42/MWh (core $1,553M vs $1,585M). The DEC overlay neither lowers the floor nor shrinks the core enough to pay for itself.

## Learning Rates

The numbers above are snapshots. Deployed technologies get cheaper. Turbines have been manufactured for over a century with thousands of GW deployed; there is little room for costs to fall. The DEC concepts are much lower TRL, with larger cost uncertainties but more room for learning.

The venetian blind is the most mature DEC. Its dominant costs (vacuum vessels, cryopumps) are mature industrial hardware, 75% of total cost. The grids are 2%. Little room to learn down.

The pulsed inductive path is different. Capacitor banks and power electronics dominate, and both are early in their deployment curves. The [NOAK assumption](https://arxiv.org/abs/2602.19389) in the 1costingfe model is $0.50/J stored (all-in: capacitors, switches, charging, buswork). Today's lab pricing is $20–50/J, a 40–100x gap, comparable to what solar PV achieved over two decades. Grid-tie inverters have already followed that curve ($1,000/kW in the early 2000s to $100–150/kW today). But the fusion-specific components (capacitors rated for 100 million cycles at high energy density) do not yet exist as commercial products. The $0.50/J figure is a target, not a price anyone can buy today.

The pulsed floors in this post already assume the 40–100x reduction has happened. If capacitors stay at $5–20/J, those floors rise substantially and the economics favor a turbine. Capacitor lifetime compounds the risk: the model assumes 10⁸ shots (NOAK), but current high-energy-density capacitors achieve roughly 10⁷ cycles, which at a 1 Hz rep rate is four months of operation between bank replacements. The thermal and venetian blind floors carry no comparable risk.

![Efficiency and capital cost ranges for the four conversion architectures. MHD is dashed because no fusion-relevant cost basis exists; pulsed inductive assumes the $0.50/J NOAK capacitor target.](<efficiency-cost chart.svg>)

## Conclusions

**1. DEC's potential value is efficiency, not BOP savings.** No DEC has demonstrated efficiency above 48% on real plasma, comparable to sCO2 Brayton. If higher efficiencies are achieved, the payoff is a smaller fusion core for the same net output. This shows up in the fully costed plant, not the free-core floor.

**2. Enabling and economic are different bars.** DEC's value can be cost-reducing or breakeven-enabling. Pulsed inductive concepts have no thermal-only fallback, so DEC efficiency gates whether the plant produces net electricity at all. But an enabling-only plant operates at high recirculation by construction, and LCOE reflects the small net rather than the large gross; clearing breakeven does not clear economics.

**3. Power electronics ride a learning curve; turbines don't.** Turbines have had a century of optimization and sit near their asymptote. Capacitors, pulse-rated switchgear, and grid-tie conversion sit on the same semiconductor and manufacturing-volume curves that took solar PV and inverters from lab to commodity. Capacitor pricing has roughly 40-100x to fall to hit the NOAK target used here, a reduction comparable in magnitude to what PV achieved over two decades. DEC's cost model has two levers that move in the right direction, efficiency and unit cost; the thermal path has neither.

**4. The venetian blind has headroom.** At the demonstrated 48% efficiency, the DEC hardware costs about as much as the turbine it replaces, with comparable conversion. The theoretical ceiling is 70%, with commercial efforts to close the gap underway at companies like Realta. A sustained >60% efficiency on real plasma is the concrete milestone that would flip the venetian blind from even trade to clear win.

**5. Pulsed inductive hinges on unverified efficiency; Polaris will measure it.** If the round-trip efficiency is 85%+, the pulsed architecture cuts the required fusion power by about a third and external He-3 demand by more than half (combining higher conversion efficiency, the D-rich operating mix, and inter-shot He-3 recovery). The pulsed-power chain that replaces the turbine costs more than a turbine, so BOP is more expensive than thermal; the savings come from fuel and core size, not conversion hardware. If efficiency is below 70%, the fuel savings shrink and the higher BOP cost favors building a turbine. The gap between viable and not sits in roughly 15 percentage points of unverified efficiency, which Polaris is expected to measure on plasma.

**6. MHD generators could close the gap, someday.** An MHD generator on the bremsstrahlung-heated coolant could convert the radiated fraction at >60% with no moving parts. The physics is demonstrated; the engineering for fusion environments is not, with liquid-metal MHD pressure drop at fusion field strengths the outstanding problem.

**7. D-He3 is the better DEC fuel, if you can get the He-3.** D-He3 has manageable bremsstrahlung (16-24% of fusion power) and a large charged fraction, making DEC a significant power path rather than a margin add-on. But purchased He-3 costs $36-74/MWh depending on architecture, far more than the entire BOP floor. Self-breeding eliminates this, but whether a D-He3 device can breed enough He-3 to sustain itself is undemonstrated.

**8. The p-B11 floor is industrial, not conversion-limited.** At baseline conditions, the p-B11 cost floor is $18/MWh, and a venetian blind DEC overlay raises it slightly rather than lowering it. With 83% of fusion energy radiated as bremsstrahlung, the thermal plant handles the majority of conversion and cannot be eliminated without an enormous increase in fusion power. At aggressive conditions, thermal and hybrid DEC converge to $7.6-7.9/MWh. The floor is dominated by buildings, electrical systems, O&M, and financing, so DEC's leverage, if any, has to come from elsewhere.

The path to 1-cent fusion energy does not run through direct energy conversion alone. DEC does not remove the industrial cost structure that creates the floor, and reaching $10/MWh still requires the conditions identified in the previous dispatch: large plants, high availability, low-cost financing, fast construction, and long plant life. But DEC has room to run in ways the thermal path does not. Three tractable development targets would turn that room into numbers: venetian blind efficiency sustained >60% on real plasma, Helion's round-trip demonstrated at 85%+ on Polaris, and MHD pressure drop solved at fusion field strengths. Each is an experimental program, not a research miracle.

## Acknowledgements

Thanks to Ian Ochs, Elijah Kolmes, and Ryan Weed for their comments.
This piece was updated on 13 May adding to MHD section. Thanks to Robert Campbell for the additional input.

## References

1. Moir, R.W. & Barr, W.L., "Venetian-blind direct energy converter for fusion reactors," *Nuclear Fusion* 13, 35–45 (1973). [Link](https://iopscience.iop.org/article/10.1088/0029-5515/13/1/005)
2. Hoffman, M.A., "Electrostatic Direct Energy Converter Performance and Cost Scaling Laws," UCID-17560, LLNL (1977). [Link](https://www.osti.gov/biblio/7218298)
3. Slough, J. et al., "Creation of a high-temperature plasma through merging and compression of supersonic field reversed configuration plasmoids," *Nuclear Fusion* 51(5), 053008 (2011). [Link](https://doi.org/10.1088/0029-5515/51/5/053008)
4. Putvinski, S.V., Ryutov, D.D. & Yushmanov, P.N., "Fusion reactivity of the p-B11 plasma revisited," *Nuclear Fusion* 59(7), 076018 (2019). [Link](https://doi.org/10.1088/1741-4326/ab1a60)
5. Ochs, I.E. et al., "Improving the feasibility of economical proton-boron-11 fusion via alpha channeling with a hybrid fast and thermal proton scheme," *Physical Review E* 106, 055215 (2022). [Link](https://doi.org/10.1103/PhysRevE.106.055215)
6. Kirtley, D. et al., "Generalized burn cycle efficiency framework," APS DPP 2024, Abstract GO05.8. [Presentation](https://youtu.be/5nHmqk1cI2E?t=505)
7. CATF IWG, "Extension of the Fusion Power Plant Costing Standard," arXiv:2602.19389 (2026). [Link](https://arxiv.org/abs/2602.19389)
8. Rosa, R.J., *Magnetohydrodynamic Energy Conversion*, Hemisphere Publishing (1987). [Link](https://inis.iaea.org/records/4ss3z-rqd83)
9. GAO, "Magnetohydrodynamics: A Promising Technology for Efficiently Generating Electricity From Coal," EMD-80-14 (1980). [Link](https://www.gao.gov/products/emd-80-14)
10. Shea, D.A. & Morgan, D., "The Helium-3 Shortage: Supply, Demand, and Options for Congress," Congressional Research Service, R41419 (2010). [Link](https://www.everycrsreport.com/reports/R41419.html)
11. Barr, W.L. & Moir, R.W., "Test Results on Plasma Direct Converters," *Fusion Technology* 3(1), 98-111 (1983). [Link](https://doi.org/10.13182/FST83-A20820)
12. Fabris, G. & Hantman, R.G., "Interaction of fluid dynamics phenomena and generator efficiency in two-phase liquid-metal gas magnetohydrodynamic power generators," *Energy Conversion and Management* 21(1), 49-60 (1981). [Link](https://doi.org/10.1016/0196-8904(81)90006-6)
13. Pinkhasov, D. et al., "CDIF 32-MWt MHD Program Results," *Proc. 31st Symposium on Engineering Aspects of MHD* (1993). [Link](https://www.osti.gov/biblio/6380343)
14. Smolentsev, S., "Modeling of Transport Processes in Liquid-Metal Fusion Blankets: Past, Present, and Future," *Fusion Science and Technology* (Oak Ridge National Laboratory) (2022). [Link](https://doi.org/10.1080/15361055.2022.2116905)
15. Binderbauer, M.W. & Tajima, T., US Patent 9,893,226 B2, "Photon-to-electric direct conversion" (2018). [Link](https://patents.google.com/patent/US9893226B2/)
16. Ramasamy, V. et al., "Documenting 15 Years of Reductions in U.S. Solar Photovoltaic System Costs," NREL/TP-7A40-92536 (2025). [Link](https://docs.nrel.gov/docs/fy25osti/92536.pdf)
17. Realta Fusion, US Patent 12,166,398 B2, "Axisymmetric ferromagnetic venetian blinds" (2025). [Link](https://patents.google.com/patent/US12166398B2)
18. 1cFE, "1costingfe: Open-source fusion techno-economic model." [GitHub](https://github.com/1cfe/1costingfe)
19. Hoffman, M.A., Campbell, R., & Logan, B.G., "Advanced fusion MHD power conversion using the CFAR cycle concept," *Fusion Technology* 15(2), 1254-1269 (1989). [OSTI 6276900](https://www.osti.gov/biblio/6276900)
20. Hoffman, M.A. et al., "Inertial fusion reactors using CFARII MHD conversion," *Fusion Engineering and Design* 22(1), 21-37 (1993). [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/092037969390114W)