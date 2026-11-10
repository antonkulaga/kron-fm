# How cheap is it to build our own VitalDB-like PPG cohort?

Research date: 2026-11-10. Prices were seen on that date in search results and retailer pages. Many listings are undated, so treat every price as indicative. **[V]** = verified from a cited source. **[E]** = my estimate. **[C]** = community or vendor-marketing claim, not independently verified.

## Bottom line

- **The hardware is cheap. People are what cost money.** A recording station that captures finger PPG and Galaxy Watch PPG from the same person at the same time costs **about €500** in a budget build and **about €2,500–3,500** with a used hospital monitor. Participants come to the lab, so a cohort of 300 still needs only 1–2 stations. **[E]**
- **The total cost is driven by staff time, blood labs and participant pay.** Rough totals **[E]**:
  - pilot of 30–50 people: **€5–15k**;
  - 300 people: **€75–115k**;
  - 1,000+ people: **€0.4–0.5M**, or **€30–60k** if the PPG recording is added onto an existing cohort that already pays for the labs and the staff.
- **The VitalDB device was not a Philips monitor.** PLETH came from a **GE Solar 8000(M) monitor with a TramRac-4A module**. The authors digitised its analog output with **their own Arduino-based ADC (8 channels, ±5 V, 10-bit), sampled at 500 Hz**, and recorded it with Vital Recorder. **[V]** Today's closest equivalent is a used **Philips IntelliVue** monitor read over RS-232. Vital Recorder records that at 125 Hz with no ADC needed. **[V]**
- **Sampling rate is not the limit.** kron-ppg reads 25 Hz after a 0.5–8 Hz band-pass filter. Any finger device at 50 Hz or faster supplies the information the model actually uses. What matters is the sensor type, which differs on four points:

  | | VitalDB finger probe | Galaxy Watch wrist sensor |
  |---|---|---|
  | Light path | transmissive (light shines through the finger) | reflective (light bounces back) |
  | Wavelength | red/IR | green |
  | Processing | filtered and gain-adjusted inside the monitor | raw |
  | Conditions | anaesthetised, still | awake, moving |

  Recording the same people with both devices at once lets us measure exactly that gap.

---

## 1. What hardware produced VitalDB's `SNUADC/PLETH`

| Item | Detail | Status |
|---|---|---|
| Patient monitor | **GE Solar 8000M** (GE Healthcare). Numeric values every 2 s. | [V] Lee et al. 2022 |
| Waveform source | **TramRac-4A** (GE). Analog ECG, plethysmogram, arterial and venous pressures, CO2 at 1/500 s. | [V] Lee et al. 2022, Table 2 |
| ADC ("SNUADC") | "analog-to-digital converter (SNUADC) developed by authors". The Vital Recorder paper describes it as **Arduino-based, 8-channel, ±5 V, 10-bit**. | [V] Lee et al. 2022; Lee & Jung 2018 |
| Sampling | 500 Hz. The 2018 paper notes the rate is "dependent on the performance of analog-to-digital converter". | [V] |
| SpO2 sensor | **Not named in either paper.** Tram modules came in Masimo SET (Tram 451M) and Nellcor (451N) versions. Which one SNUH used is unknown. | [V] not named; versions [V] from dealer listings |
| Recorder | **Vital Recorder 1.7.4**, free Windows software. A laptop takes serial cables plus the ADC through a USB hub. Ran in 10 of 31 operating rooms. | [V] |
| Cohort | 6,388 non-cardiac surgery cases, SNUH, Aug 2016 – Jun 2017. IRB H-1408-101-605. Consent waived (retrospective, de-identified). | [V] |
| Track counts | PLETH in 6,157 cases, ECG_II 6,355, ART 3,645, ECG_V5 3,390, CVP 1,586, FEM 127. | [V] from `api.vitaldb.net/trks` |

**Signal character:**
- The PLETH signal was **already processed inside the monitor** (filtered and gain-adjusted) before it reached the analog output.
- It was then quantised at 10 bits and recorded from patients who were anaesthetised, ventilated and mostly not moving.
- **This is not raw photodiode output.**

**Prices (2026-11):**
- Used GE Solar 8000M: scrap listings from about **$23**. [V, relinkonline]
- Tram 451M module: **$15–240 used** at auction, or **$897–1,437 refurbished** at PartsSource. [V]
- No dealer offers a current new price. The platform is discontinued, so I assume none exists. [E]
- A working GE Solar + TramRac + Tram 451 + DIY ADC replica: about **€300–1,500** used, plus a biomedical-engineering safety check. [E]

**Modern alternative:**
- A refurbished **Philips IntelliVue MP50** costs **$1,420–2,395** from US dealers, and one German listing asked **€3,400** (June 2025). [V]
- Vital Recorder reads IntelliVue waveforms over RS-232, logging ECG every 1/500 s and "pressure waves" every 1/125 s. [V]
- I could not confirm the PLETH rate directly. A 125 Hz pleth is consistent with Philips' usual waveform rate. [E]
- That gives monitor-processed transmission finger PPG with no ADC to build.

Sources: [Lee et al., Sci Data 2022 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9178032/) · [Lee & Jung, Sci Rep 2018, Vital Recorder (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5784161/) · [vitaldb.net Vital Recorder](https://vitaldb.net/vital-recorder) · [PartsSource Tram 451M](https://partssource.com/parts/ge-medical-systems-information-technology-gemsit/TRAM451M) · [relinkonline Solar 8000M](https://relinkonline.com/products/ge-healthcare-solar-8000m-patient-monitor-inv-00627792) · [PartsSource IntelliVue MP50](https://www.partssource.com/parts/philips/UQPH21506WSWANESANOBATT) · [dotmed MP50 (EUR)](https://www.dotmed.com/listing/anesthesia-monitor/philips/intellivue-mp-50-anesthesia-tragbarer-narkosemonitor/3727980)

---

## 2. Device options for a low-cost cohort

"VitalDB-likeness" in the last column means finger site, transmissive red/IR light and monitor-style processing.

| Option | Unit price | Sampling / raw access | CE / medical-device status | Similarity to VitalDB |
|---|---|---|---|---|
| **Contec CMS50D+** (finger oximeter with USB) | about $70 older list price; plain CMS50D from $9 (needs the D+ for recording) [V] | Community reader on GitHub outputs live pleth. About 60 Hz, 7-bit, scaled. [C] | CE medical device as an oximeter. The raw stream is unofficial. | Finger, transmissive, processed. **High** at 25 Hz model input; low resolution. |
| **Berry BM1000/BM1000C** (BLE oximeter) | about €25–40 [E] | BLE protocol documented by Adafruit, which exposes pleth scaled 0–100. "200 Hz" is a marketing claim. [V]/[C] | CE oximeter; raw stream unofficial. | As CMS50D+. Verify the real rate by timestamping packets. |
| **Nonin Xpod 3012LP** (OEM oximeter module, USB/serial) | **€635 incl. VAT** (Greek retailer) [V] | "High resolution pulse waveform, ideal for PPG signal analysis". Rate needs the OEM manual. [V]/[E] | Medical-grade OEM module. | Finger, transmissive, clinical processing. **Highest of the cheap options.** |
| **PLUX biosignalsplux SpO2 sensor** | **€180** sensor (PLUX) / €210 excl. VAT (exp-tech) [V]; hub about €1–2k [E] | Unprocessed red and IR, kHz-range sampling. [V]/[E] | Research device, not a medical device. | Finger with raw red/IR, before any monitor processing. Good for studying the processing gap. |
| **BITalino PPG sensor** | **€50** [V] plus board about €150–250 [E] | Raw, up to 1 kHz. [E] | Research or teaching device. | Finger or ear, reflective, raw. |
| **DIY MAX30102 + ESP32** | module about €3–10, ESP32 about €5–10 [E] | 50–3,200 samples/s by datasheet, raw red/IR, 18-bit. [E from datasheet] | Not a medical device. Needs an electrical-safety argument (battery only). | Reflective finger contact sensor, sensitive to pressure. Medium. |
| **DIY ProtoCentral MAX86150** (PPG + single-lead ECG) | **about €45** (₹3,995); one listing says discontinued [V] | Raw PPG plus ECG on one chip, so the two are perfectly synchronised (pulse arrival time). | Not a medical device. | As MAX30102, plus ECG. |
| **Shimmer3(R) GSR+ with optical pulse probe** | **€650 + €54 probe** (ear clip €38) [V] | Raw PPG through the GSR+ analog channel, rate configurable. [E] | Research device. | Finger or ear. Medium. |
| **Polar Verity Sense** | **€88–90** [V] | Raw PPG over the Polar SDK: 28/44/**55**/135/176 Hz streaming, 22-bit; accelerometer up to 416 Hz. [V] | Consumer device. | Arm or temple, green reflective. A good **second wrist or arm** reference. |
| **Empatica EmbracePlus** | **$2,160** for a 3-year bundle (device plus cloud) [V] | Raw blood-volume pulse (BVP) in Avro files from the cloud. 64 Hz per third parties, not confirmed by Empatica. [V]/[C] | Medical device (EU) for some functions. | Wrist, green, raw. Expensive, cloud-locked. |
| **Corsano CardioWatch 287-2** | quote only [V] | Raw PPG green/red/IR, accelerometer and bio-impedance up to 128 Hz via portal or API. [V] | EU-MDR certified. [V] | Wrist, multi-wavelength. Good, but price unknown. |
| **Biostrap** | not checked | | | |
| **Samsung Galaxy Watch7 / Watch8** | Watch7: €319 at launch, later €139–189 deals. Watch8 40 mm: **€221–307** at German retailers. [V] | Sensor SDK v1.4.1 (details below). [V] | Consumer device. The ECG app is CE-marked, but SDK use is research. | **The deployment target.** Wrist, green reflective, raw. |
| **Used hospital monitor + Vital Recorder** | IntelliVue MP50 $1.4–2.4k (US), €3.4k (EU); GE Solar replica about €0.3–1.5k [V]/[E] | IntelliVue over RS-232 at 125 Hz; GE through analog output and ADC at 500 Hz. [V] | Medical device, but secondhand. Under the German MPBetreibV the operator needs safety and technical checks (STK). [E] | **Exactly VitalDB's type of signal.** |

**Galaxy Watch through the Samsung Health Sensor SDK** [V]:
- **Continuous raw PPG:** green, IR and red at **25 Hz**.
- **On-demand raw PPG:** green, IR and red at **100 Hz**. Measurements are about 30 s long, in a foreground app, one tracker at a time.
- **On-demand ECG:** 500 Hz.
- **Accelerometer:** 25 Hz.
- **EDA:** 1 Hz, Watch8 and later.
- **Supported watches:** Watch4 and later.
- **Partner approval:** an app needs a Samsung partner request (package name plus release-key SHA-256) **only to distribute it**. For an in-house study, the watch's **Health Platform developer mode** lets a sideloaded app read the data without approval.
- A Samsung forum post mentions partner applications being temporarily closed. That post is undated.

Sources: [Samsung Sensor SDK data specs](https://developer.samsung.com/health/sensor/guide/data-specifications.html) · [SDK overview](https://developer.samsung.com/health/sensor/overview.html) · [App creation / partner process](https://developer.samsung.com/health/sensor/process.html) · [Developer mode](https://developer.samsung.com/health/sensor/guide/developer-mode.html) · [Polar Verity Sense SDK specs](https://github.com/polarofficial/polar-ble-sdk/blob/master/documentation/products/PolarVeritySense.md) · [Contec CMS50D+ reader (GitHub, via ecosyste.ms)](https://awesome.ecosyste.ms/projects/github.com%2Fmarkmll%2Fcontec_cms50dplus) · [Adafruit BM1000](https://www.adafruit.com/product/4582) · [Adafruit BerryMed library](https://docs.circuitpython.org/projects/ble_berrymed_pulse_oximeter/en/latest/api.html) · [Nonin Xpod 3012LP price](https://www.medi-shop.gr/en/handheld-pulse-oximeters/nonin-xpod-3012lp-external-oem-pulse-oximeter) · [PLUX SpO2 sensor](https://www.pluxbiosignals.com/products/blood-oxygen-saturation-spo2-sensor) · [BITalino PPG](https://pluxbiosignals.com/en-gb/products/photoplethysmography-ppg-sensor) · [ProtoCentral MAX86150](https://protocentral.com/product/protocentral-max86150-ppg-and-ecg-breakout-with-qwiic-v2/) · [Shimmer optical pulse probe](https://www.shimmersensing.com/product/optical-pulse-probe/) · [Empatica store](https://empatica.com/store/platform-professional) · [Corsano research](https://corsano.com/solutions/academic-research) · [Galaxy Watch7 prices (Vodafone)](https://www.vodafone.de/featured/article/samsung-galaxy-watch7-galaxy-watch-ultra-preise-so-viel-kosten-die-modelle-203079) · [Galaxy Watch8 (Bechtle)](https://www.bechtle.com/de-en/finder/product-family/samsung-galaxy-watch7--5896--f)

---

## 3. Reference labels for validating an aging clock

| Label | Device or method | Cost | Status |
|---|---|---|---|
| Cuff blood pressure | **Omron M7 Intelli IT** (clinically validated, Bluetooth) | €118–215, typically about €135 | [V] |
| Single-lead ECG / R-peak timing reference | **Polar H10** chest strap; raw ECG at 130 Hz via the Polar SDK | €75–90 | price [V], ECG rate [E from Polar SDK docs] |
| | KardiaMobile / 6L | about €90–150 | [E] |
| Pulse-wave velocity, gold standard | **SphygmoCor XCEL** (carotid tonometry plus femoral cuff) | No public price; quote likely about €15–30k | [E] |
| Pulse-wave velocity, cheaper | **Mobil-O-Graph** (brachial cuff, PWV estimated by algorithm) | about €3–6k | [E] |
| | It **reads systematically lower** than SphygmoCor: 5.9 vs 7.3 m/s in one study | | [V] |
| Pulse-wave velocity, free proxy | Pulse arrival time from Polar H10 ECG plus finger PPG | €0 extra | method only |
| Blood panel (self-pay prices, German private-fee schedule GOÄ) | HbA1c | €11.66–15.96 | [V] |
| | Total cholesterol, HDL, LDL, triglycerides, creatinine | €2.33–3.19 each | [V] |
| | **Basic panel total** | **about €25–30**; about €35–50 with CRP, blood-draw consumables and courier | total [E] |
| | Rate negotiated with the university hospital lab | often lower | [E] |
| Grip strength | Jamar-type hydraulic dynamometer | about €300–400 | [E] |
| | Camry-type digital dynamometer | about €30 | [E] |
| Frailty | Fried phenotype (grip, gait speed, exhaustion, activity, weight loss); Clinical Frailty Scale | stopwatch plus tape | [E] |
| Questionnaires | IPAQ, PSQI, PROMIS (free); SF-12/SF-36v2 (may need a licence) | €0–small | [E] |
| Anthropometry | Scale, height measure (stadiometer), waist tape | about €200 | [E] |

Sources: [Omron M7 prices (PriceSpy)](https://pricespy.ie/product.php?p=14116657) · [Polar H10 prices (Galaxus)](https://www.galaxus.de/en/s1/product/polar-h10-heart-rate-belts-6191018) · [SphygmoCor vs Mobil-O-Graph (Birmingham)](https://research.birmingham.ac.uk/en/publications/pulse-wave-velocity-in-south-african-women-and-children-compariso) · [Labor Düsseldorf IGeL 2026](https://www.labor-duesseldorf.de/assets/Sonic/IGEL/0b50acbe30/IGeL_Profile_2026_web.pdf) · [Labor Leipzig single tests](https://www.labor-leipzig.de/fileadmin/user_upload/Leipzig/09_Service/Flyer_Individuelle_Gesundheitsleistungen.pdf) · [Mein Labor Berlin 2025](https://mein.laborberlin.com/wp-content/uploads/2025/03/Preisliste-Mein-Labor-2025.pdf)

---

## 4. Budgets [all estimates unless marked]

### Assumptions
- **One 60–90 min lab visit** with:
  - 10 min seated rest;
  - paced breathing;
  - 5 min standing (orthostatic challenge);
  - optional light step test;
  - cuff blood pressure;
  - ECG;
  - blood draw;
  - grip strength;
  - questionnaires.
- Recruitment is stratified by decade from age 20 to 80+, with roughly equal numbers of women and men.
- **Participant pay:** €20–30 per visit (about €10–15/h, the usual German range [E]). Payments below **€256 per person per year** are tax-free for participants. [V, Ärzte Zeitung]
- **Direct cost per participant:** €70–100. That covers pay, the blood panel with consumables, recruitment advertising, and electrode and probe consumables.
- **Staff rates:**
  - student assistant: about €15/h;
  - study nurse or medical lab technician: about €55–65k/yr fully loaded;
  - coordinator on the E13 public-sector pay grade (TV-L): about €75–80k/yr fully loaded.
- **Ethics:** the University Medicine Rostock fee schedule (2017) charges **€300** for an advisory vote under §15 of the physicians' professional code, and **€300** for research on humans not run by physicians. [V] The schedule is reviewed every 2 years.

### Pilot, 30–50 people (about 40)
| Item | € |
|---|---|
| 1 station with a used IntelliVue monitor (budget build: CMS50D+ instead) | 2,500–3,500 (budget 500) |
| Labels kit: Omron, Polar H10, cheap dynamometer, scale | about 400 |
| Direct costs: 40 × €70–100 | 2,800–4,000 |
| Staff: about 200 h student assistant or technician; PI and PhD time in-kind | 3,000–4,000 |
| Ethics | 300 |
| **Total** | **about €9–12k** (budget build without blood: **about €4–6k**) |

### Small cohort, 300 people
| Item | € |
|---|---|
| 2 stations plus a spare watch and sensors | 6,000–8,000 |
| Mobil-O-Graph-class pulse-wave velocity device | 3,000–6,000 |
| Direct costs: 300 × €70–100 | 21,000–30,000 |
| Study nurse or lab technician 0.5–1 FTE × 12 months, plus a student assistant | 40,000–70,000 |
| Data management and storage, insurance, ethics | 2,000–5,000 |
| **Total** | **about €75–115k** |

### Larger cohort, 1,000+ people
| Item | € |
|---|---|
| 3–4 stations plus a SphygmoCor-class pulse-wave velocity reference | 40,000–50,000 |
| Direct costs: 1,000 × €70–100 (more with extra labs, e.g. cystatin C, CRP, epigenetic clock samples) | 70,000–100,000+ |
| Staff: coordinator E13 1 FTE plus 2 study nurses × about 24 months | 250,000–350,000 |
| IT, data management, external data-protection review, insurance | 15,000–30,000 |
| **Total** | **about €0.4–0.5M** |

**Much cheaper route at scale:** add a 10–15 min PPG module to a cohort that already exists in Mecklenburg-Western Pomerania.
- Example: SHIP (Greifswald). I have not checked its access terms or whether it would agree. [E]
- Its labs, blood pressure and staff are already paid for.
- The added cost is about **€20–40 per person**, mainly staff minutes and pay. For 1,000 people that is about **€30–60k**.

### Ethics, data protection (GDPR) and the medical-device regulation (MDR)
- **Ethics vote:** the faculty ethics committee at University Medicine Rostock must give one. Physicians involved need an advisory vote under §15 of their professional code; the fee is €300. [V]
- **Does the study fall under the MDR?** A study is a clinical investigation under the MDR when it systematically tests a device's safety or performance. This study does not, provided that:
  - devices are used only to record data;
  - CE devices (monitor, Omron) are used as intended;
  - consumer and DIY devices are used purely for data collection;
  - the model makes no diagnostic decisions about participants.

  Under that reading, the study should count as observational research needing only the ethics vote.

  **Caveat:** German law (§3 No. 4 MPDG) excludes studies that are part of "planned product development" from the lighter "other clinical investigation" category. If the protocol frames the work as developing a medical-device algorithm, the committee could classify it differently. **So frame it as an academic question:** how finger and wrist PPG relate, and PPG markers of ageing. Then ask the committee for a written determination. [V for the definition; the conclusion is my interpretation, not legal advice]
- **Electrical safety for DIY ECG:** battery powered only, never connected to mains while on a person.
- **GDPR:**
  - Health data fall under Art. 9, so we need explicit consent.
  - A data-protection impact assessment (DPIA) is likely required.
  - Prefer devices that keep the data local: Galaxy Watch with our own app, Polar SDK, Vital Recorder.
  - Cloud-locked devices (Empatica, Corsano) need a data-processing agreement and a check of transfers outside the EU.

Sources: [Rostock ethics fee schedule 2017](https://www.uni-rostock.de/storages/uni-rostock/UniHome/Gremien/Rechtsgrundlagen/Amtliche_Bekanntmachungen/2017/NR_41_2017.pdf) · [BfArM FAQ on MDR/MPDG](https://bfarm.de/EN/Medical-devices/_FAQ/Clinical-investigations-performance-studies/MDR-MPDG/faq-liste.html) · [Ärzte Zeitung on study payments and tax](https://www.aerztezeitung.de/Wirtschaft/Honorar-fuer-Studienteilnahme-ist-meist-steuerpflichtig-421807.html)

---

## 5. Cheapest credible finger-plus-wrist bridging plan

**Recommended station** [prices V unless marked]:

| Role | Device | € |
|---|---|---|
| Finger, VitalDB-type signal | **Used Philips IntelliVue MP-series monitor** (MP30/40/50 or MX450) with an SpO2 module (MMS), reusable finger clip, RS-232 interface cable, and a Windows laptop running **Vital Recorder**. Pleth at 125 Hz. | 1,500–3,500 [E from listings] |
| Finger, raw (optional) | **ProtoCentral MAX86150 + ESP32** on another finger, giving raw red/IR plus ECG at 200–400 Hz. Shows what the monitor's processing removes. | about 60 |
| Wrist, deployment target | **Galaxy Watch8 40 mm** (or a discounted Watch7). Our own Wear OS app in Health Platform developer mode: PPG green/IR/red at 25 Hz continuously, plus 30 s bursts at 100 Hz on demand, plus ECG at 500 Hz. | 220–300 |
| Timing reference | **Polar H10**: ECG at 130 Hz gives an R-peak time base for aligning all three PPG streams and for pulse arrival time. | 75–90 |
| Second wearable (optional) | **Polar Verity Sense** on the other forearm: raw PPG at 55–176 Hz. Tests whether the bridge holds across devices. | about 90 |
| Labels | Omron M7, digital dynamometer, scale and tape | about 250 |
| **Station total** | | **about €2.2–4.3k** |

**Budget station, about €500:**
- Swap the monitor for a **Contec CMS50D+** or **Berry BM1000C**: transmissive finger, roughly 60–100 Hz, 7-bit, read through the community protocol.
- That is enough for kron-ppg's 25 Hz, 0.5–8 Hz input.
- It is not a faithful copy of the monitor's processing.

**Protocol notes:**
- Put the finger probe on the hand **opposite** the watch, so pressure and cuff inflation on one arm do not affect the other.
- Never inflate the blood-pressure cuff on the arm wearing the watch during PPG blocks.
- **Clock synchronisation:**
  - log everything to one laptop or phone clock with NTP;
  - start with a shared event (three hand taps, which show up in both the watch accelerometer and the PPG);
  - align after the fact by cross-correlating beat times against the H10 R-peaks.
- **Plan the 100 Hz on-demand bursts.** They run only about 30 s each and one tracker at a time, so schedule them at rest, during paced breathing and while standing. Keep the 25 Hz continuous stream running otherwise.
- **Later replica option:** a GE Solar 8000M + TramRac-4A + Tram 451M/N with a modern 16-bit USB ADC at 500 Hz would reproduce VitalDB's exact chain for about €0.3–1.5k used.
  - Worth doing only if a used unit is available from the university hospital's medical engineering department.
  - It must pass a technical safety check before use.

**Open items to verify before buying:**
1. Which SpO2 version (Masimo or Nellcor) SNUH's Tram modules used. Ask the VitalDB team.
2. The actual sample rate and bit depth of the CMS50D+ and BM1000C streams, by timestamping packets.
3. The current status of Samsung partner applications, if the app is ever to be distributed.
4. Current dealer quotes for an IntelliVue monitor in the EU, and for a SphygmoCor or Mobil-O-Graph.
5. Whether the Rostock ethics committee has updated its fees since 2017.
