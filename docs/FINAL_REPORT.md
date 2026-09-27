# Risk Horizon: a road-risk signal for the car's electronic horizon, built from Assam's police crash records

**Granica × IIT Guwahati Hackathon: "Bring the Physical World to AI"** · Team DirectorsInc · 27 Sep 2026

> **In one line:** we turned 44,928 police crash PDFs, OpenStreetMap, NASA night lights, hourly weather and live traffic into a structured, validated record of Assam's roads. From it, a transparent model tells a driver-assist car which 500 m stretches ahead are dangerous, and when. Trained on the past and tested on months it never saw, it finds **49.7% of future serious crashes on just 5% of the road**, against 39.5% for today's blackspot-list approach, and it flags **184 of 236 new hotspots** before they happen.

---

## Contents

1. Problem, user and the decision we support
2. Why this is Physical AI: the gap between the car's sensors and the road's history
3. From research to features: what we read and what we took from it
4. Data collection: sources, how, when, and how much
5. EDA: what the data told us, and how each finding changed the plan
6. Data problems we found and how we tackled them (our main strength)
7. The model: architecture, dimensions and exact inputs
8. Results and what they mean
9. The signal given to the car
10. Limitations: what the data cannot tell
11. Future work
12. How this answers the brief and the judging criteria
13. Reproducibility, folder map and privacy
14. References

---

## 1. Problem, user and decision

**The problem.** Assam recorded **3,296 road deaths in 2023** (MoRTH). Road agencies manage crashes through **blackspot lists**. A stretch becomes a blackspot only after it has repeatedly killed or maimed people (MoRTH definition: about 500 m with 5 fatal or grievous crashes, or 10 deaths, in 3 years). The list is backward-looking by design: a road must kill before it is flagged.

**The users.**

1. **Driver-assist (ADAS) and self-driving systems.** They read an *electronic horizon* (ADASIS): map attributes of the road ahead such as curvature, slope and speed limit. They have no attribute for *how dangerous* the next kilometre is.
2. **Road-safety engineers** (PWD, NHAI, district road-safety committees). They must choose which stretches to fix before they turn into blackspots.

**The decision we improve.** *Which road stretches ahead deserve caution, and under what conditions?* For the car, the output is a band from 0 to 3 for every 500 m segment, with an action such as an earlier alert, a lower cruise speed or a larger following gap. For engineers, it is a ranked list of segments that are likely to become blackspots.

**The workflow, worked backward (as the brief asks):**

> real problem (crashes cluster, lists lag) → physical workflow (police report every crash in iRAD; the road network, lighting and weather are observable) → useful observations (where, when and under what conditions serious crashes happen) → AI-assisted decision (risk per segment × condition multiplier) → action (car slows or alerts; engineers fix the top segments)

## 2. Why this is Physical AI

A car's cameras and radar see about 200 m, in the present. They cannot see that the bend 2 km ahead killed people last winter, or that darkness makes this unlit stretch deadlier. That knowledge exists only as a *physical record*: police reports, road geometry, lights at night, weather at the crash hour. Nobody had captured it in a learnable form for Assam. It sat in 44,928 PDFs, a map, a satellite archive and a weather archive.

Risk Horizon does what the brief describes for self-driving: it **captures a slice of the physical world for a specific decision**. The novelty lies in three places:

- **The unit:** 500 m segments that line up with the MoRTH blackspot unit.
- **The linkage:** each crash is tied to its road, its light and its weather.
- **The honesty layer:** every field is labelled observed, inferred or computed, and checked against an independent source.

The model is deliberately simple. The data work is the contribution.

## 3. From research to features

We read how road-safety science predicts crashes, then built only the features the data could support.

| What we read | What it says | What we took |
|---|---|---|
| Highway Safety Manual network screening; EB hotspot studies [1][2][3] | Crash counts are over-dispersed. Rank sites with a **safety performance function (SPF, negative binomial)** blended with each site's own history by **Empirical Bayes (EB)**. EB beats raw crash counts at finding true hotspots. | Model spine: SPF + EB. Crash-count ranking is our baseline. |
| SPF vs machine-learning comparisons [4][5] | Gradient-boosted trees (XGBoost/RF) beat NB on accuracy; traffic volume, segment length and road class dominate. | A boosted Poisson tree model next to the SPF, blended. Length and road class are core features. Traffic volume is our main gap (§10). |
| Time-stratified case-crossover weather studies [6] | Compare each crash with the same place at the same weekday and clock time on other days (conditional logistic regression). Weather effects differ by road type. | The WHEN table: 44,112 crashes, each with about 3.4 matched control hours. A rain × road-type test. |
| Night-time severity studies [7] | Low light raises crash severity. | Severity model: light (computed sun position) × street-lighting proxy (VIIRS). |
| MoRTH blackspot definition | ≥5 fatal or grievous crashes or ≥10 deaths per ~500 m in 3 years. | Segment unit ≤500 m; blackspot and "new hotspot" labels. |

## 4. Data collection

**Principle: collect only what the decision needs.** Downloads were scoped to crash places and dates, not "all of Assam, all hours". The weather plan fell from a full-period pull to **674 requests covering 580,850 cell-hours**, because we fetched only each crash hour, its control hours and the 3 hours before.

| # | Source | What it gives | How collected | Volume | Kind |
|---|---|---|---|---|---|
| 1 | **iRAD (MoRTH) police crash reports**, Assam | Location, time, severity, vehicles, persons, road and conditions | Our PDF parser (`irad_to_excel.py`), 16 tables | 44,928 PDFs → **44,114 crashes**, Jan 2023 – May 2026 | Observed |
| 2 | **OpenStreetMap** (Geofabrik NE India) | Road network, class, bridges, junctions, rail crossings, signals, POIs | `fetch_osm.py` (pyosmium filter) | 190,077 ways (133,205 in Assam), 8,137 POIs in Assam | Observed |
| 3 | **ERA5 reanalysis** via Open-Meteo | Hourly rain, humidity, dew point, wind, temperature per 25 km cell | `fetch_weather.py`, limited to needed hours, quota-aware retries | 150 cells, 3,791,040 cell-hours | Inferred |
| 4 | **METAR airport reports** (Iowa Mesonet) | Observed visibility and fog | `fetch_metar.py` | Guwahati VEGT: 50,618 reports (5 airports pending) | Observed |
| 5 | **NASA Black Marble VNP46A4** (2024) | Night-time light: a street-lighting and activity proxy | `fetch_viirs.py` (Earthdata token, resumable download) | 1,445,970 pixels (~460 m) | Observed |
| 6 | **TomTom Traffic API (live)** | Speed vs free-flow on the Guwahati corridor | `tomtom_live_collector.py`: 19 tiles every 15 min + 24 points hourly, **running since 26 Sep 2026** | Growing | Observed |
| 7 | **CE323 field studies** (IIT Guwahati lab reports) | Hand-counted volumes, headways, spot speeds, parking, OD | Digitised into 23 tables, 21/21 checks | 19–644 rows per table | Observed |
| 8 | **Computed** | Sun elevation for every crash and control hour; segment geometry; neighbourhood features | pvlib, shapely, our feature code | — | Computed |

**Recorded how and when.** Every downloader writes a manifest (time, source, success and error counts, file hashes; see `08_Logs/`). All tables are Parquet, with CSV copies for reading.

**Collected more than once.** TomTom polls every 15 minutes and was still collecting during the pitch.

## 5. EDA: findings, and how each changed the plan

**A. Can the crash data be trusted?**

- **Deduplication and parse check:** 655 duplicate PDFs removed and 159 unreadable ones logged. An independent re-check of 300 random reports against the PDF text found **0 of 44,768 cells missing**.
- **Coverage:** iRAD holds **3,237 of MoRTH's 3,296 deaths for 2023 (98%)**, so deaths are well covered. iRAD has 32% more crashes than MoRTH because it includes minor ones.
- **Report depth:** 47% of reports are full, 37% medium and **15.5% skeletal**, with road, weather and light left blank.

→ *Decision:* use **deaths and serious injuries (KSI)** as the target, where coverage is strongest, and handle missing sections explicitly.

**B. Which fields are reliable?**

- **Reliable (≥85% usable):** light, weather, road class and collision type.
- **Unreliable:** behaviour and mechanical fields.
  - cell-phone use: 37% usable
  - helmet/seatbelt: 43%
  - drunk driving: 47%
  - "accident due to": 29%
- **Dropdown defaults:** "Sunny/Clear" appears in **60% of crashes at night**.

→ *Decision:* never use behaviour fields as labels. **Measure conditions independently**: sun position is computed from time and place, and weather comes from ERA5 and METAR.

**C. Who and what dies?**

- **Two-wheelers** are in 46% of crashes and **57% of fatal crashes**.
- Vehicle-hits-pedestrian crashes are 38% fatal. Rear impacts by trucks on two-wheelers are 45% fatal.
- Darkness without street lights is 35% fatal, against 23% in daylight.

→ *Decision:* include light and street lighting (VIIRS) and pedestrian places. Make severity, not raw crash count, the target of the condition multiplier.

**D. Where?**

- About 570 hotspot cells of ~100 m hold ~9% of all crashes.
- Five districts hold 38.5% of crashes.

→ *Decision:* a **segment-level** model (500 m, the MoRTH unit). The live-collection site is the **Guwahati corridor** (Jalukbari–Saraighat–Amingaon), chosen by a region scorecard (`04_EDA/region_selection`).

**E. After the crash:**

- 42% of deaths happen within 1 hour, and 58% after it.
- 21% of reports are filed more than 6 hours after the crash.

→ This supports warning *before* the crash, and it motivates the post-crash alert as future work.

**F. Checking police records against measurements:**

- When police wrote "Rain", ERA5 shows rain in the previous 4 h **70% of the time vs 25% otherwise**. The rain field is credible.
- Police light condition matches the computed sun position **85.4%** of the time.
- Police "Mist/Fog" agrees only weakly: ERA5 flags fog 11% vs 5%, and the airport saw fog 1.4% vs 0.6%.

→ *Decision:* **drop the police fog field.** Use measured fog only.

## 6. Data problems we found and how we tackled them

This is where most of our effort went, and we think it is the project's main strength. Each row is a real problem in real data, how we found it, and what we did.

| # | Problem | How we found it | What we did |
|---|---|---|---|
| 1 | 655 duplicate PDFs, 159 unreadable | Hashing and ID checks during extraction | Deduplicated; logged failures; verified 44,768 cells with 0 missing |
| 2 | **No crashes at all in Jan 2024 and Jul 2024** | Monthly counts | Disclosed; rates computed per *data month*, not calendar month |
| 3 | **iRAD reporting grew 1.5× from early 2023 to mid 2024, unevenly by area** (0.8× to 3.8× across 0.5° regions, IQR 1.17–1.74); stable after (1.04×) | Region-by-period growth analysis (`05_Data_quality/reporting_drift_by_region.csv`) | Early crash history partly maps *where iRAD was adopted*, not danger. Added **drift-corrected** history features (nearby rate ÷ regional rate). The test period falls in the stable era. Absolute levels are reported with this caveat. |
| 4 | Behaviour fields ≤47% usable; dropdown defaults | Field-reliability audit | Excluded as labels or features |
| 5 | Police weather and light entries are unmeasured | Cross-check vs ERA5, METAR and sun position | Conditions measured independently; police fog dropped |
| 6 | Crash times rounded to :00/:30 | Minute-level histogram | Flagged `time_precision`; weather includes the 3 h before |
| 7 | Crash points are off the road | Snap distance analysis | Snapped within 100 m: **90.3% matched, median 5 m** |
| 8 | OSM tags are thin: lanes on 1% of roads, maxspeed 0.4%, 28 speed breakers in all Assam | Tag-coverage audit | Rely on geometry and network features; don't pretend lanes or speed are known |
| 9 | Lab reports had inconsistent numbers: edited video times implying **2,571 km/h**, swapped free/combined labels, impossible percentiles | 21 automated consistency checks | Corrected or flagged in a Validation_Log; the data is usable with provenance |
| 10 | Weather over-collection risk and API quotas | Request-weight planning | Scoped to needed hours; automatic pause-and-retry on quota |
| 11 | Truncated satellite downloads and lost auth headers on redirect | File-size and HDF5 integrity checks | Resumable downloads, length verification, corrupt-file detection |
| 12 | **Hidden leak 1 in our own feature code:** float residue from fast grid sums (~1e-6, negative only when the segment itself had crashes) encoded the segment's own crash count | A **replication check**: rebuilding the previous model gave 43% instead of its known 48.7% | Replaced the grid sums with an **exact great-circle neighbour search**. The replication then matched exactly (48.70%). The feature study was re-run from scratch. |
| 13 | **Hidden leak 2:** subtracting a segment's own crashes from its *regional* total makes that total a copy of its own count | A sudden 12-point drop in validation after the change | Kept the regional rate identical for all segments in a region |
| 14 | Calibration drift: 2025–26 reports 46% more serious crashes per month than 2023–24 | Predicted vs observed by decile | The model is judged on **ranking** (what the signal uses); levels are shown with the caveat |

**Lesson for judges:** leaks 12 and 13 would have produced an impressive but false improvement (+6 points). We caught them because we always re-ran the previous model under the new code and demanded an exact match. We would rather present a smaller, real gain.

## 7. The model

### 7.1 Architecture (three stages)

```
DATA                                   MODELS (fit on the past only)                     SIGNAL
Road segments (OSM) + night light  ─┬─ A1 NB safety performance function (20%) ─┐
+ crash history around segment      └─ A2 boosted Poisson trees (80%)          ─┴─ blend ─ Empirical Bayes with own history ─┐
                                                                                                                         ├─ risk now = static × multipliers ─ bands 0-3 per 500 m
Crash hours vs control hours       ── B1 conditional logit (does rain/fog raise crash odds?) ─┐                          │   + reasons + pedestrian flag
(case-crossover) + ERA5 + sun      ── B2 logistic severity (does dark/lighting raise fatality?) ─┴─ multipliers (never <1) ─┘
Live TomTom speed ─────────────────────────────────────────────────────────── decides WHEN to warn (NOW layer)
```

### 7.2 Dimensions

| Stage | Rows | Target | Inputs |
|---|---|---|---|
| A (static risk) | **246,789 segments** (81,169 km, ≤500 m each) | KSI crashes on the segment in the training window | **33 features** for trees; 25 terms for the SPF (§7.3) |
| B1 (likelihood) | **193,904 place-hours** = 44,112 crashes + 149,792 controls | crash hour vs control hour, within each crash's set | rain, heavy rain, rain in last 4 h, fog-likely, rain × main road |
| B2 (severity) | 44,112 crashes (21,108 train / 23,004 test) | fatal vs non-fatal | light (computed), lighting class (VIIRS), road class, rain, fog, season, time of day, weekend, road context (bends, junctions, pedestrian places, distance to town, light contrast) |
| C (signal) | 246,789 segments × {day, twilight, dark, rain} | — | A score × B multipliers → band |

### 7.3 Exact model input (Stage A)

52 candidate features were engineered from the raw sources. We then:

1. removed redundant ones (|Spearman| ≥ 0.85 clusters)
2. removed near-empty ones (>99.8% zeros)
3. dropped groups that did not help validation
4. ran a backward elimination on the validation year

That left **33 features**. Full definitions are in `06_Model/v2_final/feature_study/feature_catalog.csv` and `03_Data/schema/where_model_input_final_dictionary.csv`.

| Group | Final features |
|---|---|
| Road class | `hw_rank` (0–6), `is_main_road`, `is_national_highway`, `is_divided` |
| Geometry | `length_m`, `turn_deg_per_km`, `turn_contrast` (bend vs its neighbours: "a bend after a straight"), `is_bridge`, `is_unpaved` |
| Junctions | `junctions`, `junctions_per_km`, `junctions_1km` |
| Pedestrian places | `ped_places_x_main` (places on a main road), `poi200_eatery`, `poi200_worship`, `poi_2km`, `school_500m` |
| Lighting (VIIRS) | `night_light`, `night_light_10km`, `light_contrast` (lit strip in a dark area), `dist_to_town_km` |
| Network | `main_road_km_1km`, `dist_to_main_road_km`, `road_km_5km` |
| Crash history (other segments only) | `nearby_ksi_500m`, `nearby_ksi_3km`, `nearby_ksi_1km_recent`, `nearby_fatal_1km`, `nearby_severity_ratio_1km` |
| Drift-corrected history | `nearby_ksi_1km_rel`, `nearby_ksi_3km_rel` |
| Location | `mid_lat`, `mid_lon` |

**Trees (A2):** sklearn HistGradientBoosting, Poisson loss, 500 trees, learning rate 0.05, 63 leaves, min 400 segments per leaf. Chosen from an 8-configuration search on validation.

**Blend and EB:** 0.2 × SPF + 0.8 × trees (weight chosen on validation), then Empirical Bayes with the segment's own KSI history.

**Training time:** Stage A takes about 2–3 minutes on a laptop CPU; the one-off feature study takes about 15 minutes.

### 7.4 Time-safe protocol

No choice ever looked at the test period.

| Role | Window | Used for |
|---|---|---|
| Selection-train | Jan 2023 – Jun 2024 (17 data months) | fitting candidates during the feature study |
| Selection-validation | Jul 2024 – Jun 2025 (11 data months) | choosing features, parameters and blend weight |
| **Final training** | **Jan 2023 – Jun 2025 (28 data months, ~72% of crashes)** | fitting the final model |
| **Held-out test** | **Jul 2025 – May 2026 (11 months)** | reporting results, once |

Scores are always **out-of-fold over 5 spatial folds (0.2° blocks)**, so no segment is scored by a model that saw its area.

## 8. Results and what they mean

**Metrics (all measured on the held-out period):**

- **capN** = the share of serious (KSI) crashes that fall inside the riskiest N% of road length.
- **Capture-curve score** = the average of capN over 0–20% of road.
- **New hotspots** = segments with ≥3 KSI crashes in the test period that had fewer than 3 in the whole 28-month training window, i.e. places a crash-history list could not have flagged.

### 8.1 Held-out test (Jan 2023–Jun 2025 → Jul 2025–May 2026)

| Model | cap1 | cap5 | cap10 | Capture-curve score | New hotspots in top 5% (of 236) |
|---|---|---|---|---|---|
| Blackspot list (past crashes on the segment) | 15.2% | 39.5% | 53.9% | 0.479 | 124 |
| Previous model v1 | 16.5% | 49.3% | 67.4% | 0.605 | 185 |
| A1 SPF alone | 15.0% | 45.8% | 64.6% | 0.579 | 168 |
| A2 trees alone | 15.1% | 48.7% | 67.5% | 0.604 | 189 |
| Blend | 16.1% | 48.8% | 67.4% | 0.606 | 184 |
| **FINAL (blend + EB)** | **17.2%** | **49.7%** | **68.2%** | **0.611** | **184** |

**Uncertainty** (spatial block bootstrap, 300 resamples):

| Comparison | cap5 gain (95% CI) | Capture-curve gain (95% CI) |
|---|---|---|
| FINAL vs blackspot list | **+10.6 points** (8.1 to 12.2) | +0.141 (0.115 to 0.168) |
| FINAL vs v1 | +0.45 points (−0.25 to +1.1) | **+0.006** (0.003 to 0.009) |

**Risk bands (final model):**

| Band | Road length | Share of future serious crashes | Serious crashes per 100 km |
|---|---|---|---|
| 3 very high | 1% | 17.7% | 202 |
| 2 high | 4% | 32.5% | 93 |
| 1 elevated | 15% | 32.7% | 25 |
| 0 low | 80% | 17.1% | 2.4 |

**What this means:**

1. **Against the current practice.** On the same 5% of road (about 4,060 km), Risk Horizon covers **a quarter more of the future serious crashes** than a list built from past crashes. It flags **60 more of the new hotspots**, which is exactly where lists fail.
2. **Risk is concentrated and predictable.** The very-high band has **83 times** the serious-crash rate of the low band. Warnings on 5% of road would be rare enough for drivers to trust.
3. **Feature engineering gave a small, statistically clear gain in overall ranking** (+0.006 capture-curve score, CI above zero) but no significant gain at the 5% cut. The previous model was already close to what these inputs allow. The next leap needs **traffic exposure data** (§10), not more tuning.
4. **The same result holds on a second split** (train 2023–24, test Jan 2025 – May 2026, MoRTH blackspots): 49.2% in the top 5% vs 37.2% for the list; 85 of 91 new blackspots vs 64.

### 8.2 What carries the information (feature study, validation year)

| Finding | Evidence | Meaning |
|---|---|---|
| **Road class is the most informative source** | Removing it costs the most (−0.012); alone it reaches 0.475 | Busy, high-speed roads dominate serious crashes (partly exposure) |
| Geometry and crash history are next | −0.004 and −0.003 when removed | Bends and the neighbourhood's record add information beyond class |
| Road network alone is strong (0.481) | Single-group test | Settlement and road density stand in for traffic we can't measure |
| Lighting helps a little (−0.0008) | Group ablation | It works better as a severity factor (Stage B2) |
| Pedestrian POIs add almost nothing independently | −0.0001 | Captured already by network and light density; kept only as a pedestrian flag for the car |
| **Low redundancy across sources** | PCA: 90% of the variance of the 33 static features needs **19 components**; the first explains only 24% | The sources carry different information. Compressing with PCA **lowered** the validation score (0.5945 vs 0.6041), so PCA was rejected |
| Training on all crashes instead of KSI | 0.6040 vs 0.6044 | No gain, so the KSI target is kept |
| Top features on the test period | `hw_rank`, `nearby_ksi_3km`, `length_m`, `turn_deg_per_km`, `nearby_ksi_500m`, `nearby_ksi_1km_rel`, `dist_to_main_road_km` | Road class, local crash environment, length and bends |

### 8.3 Conditions (Stage B)

- **Rain slightly lowers crash likelihood:** matched OR 0.955 (0.922–0.989). Rain in the last 4 h: 0.938 (0.913–0.964).
  - Fog-likely has no clear effect (0.977, 0.927–1.030). Rain × main road also shows none.
  - This most likely reflects **fewer trips and slower driving in rain** (exposure).
  - **Signal rule:** conditions never lower a warning, so rain and fog get a multiplier of 1.0.
- **Darkness makes crashes deadlier:** fatal odds × **1.39** (1.19–1.61), holding time of day, road type, weather and season fixed. Twilight: × 1.18 (0.97–1.43, not significant).
  - Early morning (4–7 am) adds × 1.24 (1.07–1.45). A lit strip inside a dark area lowers fatal odds (× 0.73 per unit of light contrast).
  - Once road context is included, the street-lighting class adds no significant effect, so the dark effect is similar everywhere.
  - Dark multipliers used in the signal: 1.17 to 1.19 across lighting classes.
  - The severity model separates single crashes only weakly (AUC 0.571 on 2025–26; boosted trees reach only 0.573, so the interpretable logistic model is kept), so it is used only for **average multipliers**, not per-crash predictions.

## 9. The signal given to the car

| Band | Definition | Suggested vehicle action |
|---|---|---|
| 3 very high | top 1% of road length | Chime + spoken warning ~500 m before; cruise control slows before the segment; maximum pedestrian-detection sensitivity |
| 2 high | next 4% | Chime + icon ~300 m before; cap speed near the limit; larger gap; pedestrian sensitivity up if `ped_flag` |
| 1 elevated | next 15% | No alert (logged); slightly larger following gap |
| 0 low | remaining 80% | Normal |

**Fields per segment:**

- `segment_id`, position and length
- `static_band`, `twilight_band`, `dark_band`, `rain_band`
- `main_reasons` (from the SPF), `ped_flag`, `confidence`

**Behaviour after dark:** darkness raises the share of road in band 2 or higher from 5.0% to 5.9%.

**Live layer:** TomTom speed ratios decide *when* to warn. Examples: fast free-flow into band 3, or a sudden slowdown ahead of band 2+.

**Delivery:** the signal is a small lookup file (segment ID, shape, bands, multipliers) that fits the ADASIS electronic-horizon model. WHERE updates yearly, WHEN hourly.

## 10. Limitations

- **No traffic volumes.** There are no public counts for Assam, and TomTom history is paid. Risk is per km, not per vehicle, so busy roads rank high partly for being busy. This is the single biggest gap.
- **Reporting drift.** iRAD adoption grew unevenly in 2023–24. We corrected the history features and kept the test in the stable era, but early labels still carry some of it. Absolute predicted levels run about 26–49% low in the top deciles.
- **Missing months:** there are no crash records for Jan 2024 and Jul 2024.
- **Police data quality:**
  - crash times are often rounded
  - 9.7% of crashes could not be placed on a road within 100 m
  - behaviour and cause fields are mostly unusable
- **ERA5 weather is a 25 km estimate.** Airport observations exist only for Guwahati so far.
- **OSM is thin** on lanes, speed limits and speed breakers.
- **Night light is a proxy** for street lighting, not an inventory of lamps.
- **Severity is hard to predict** for single crashes (AUC 0.57).
- **Live traffic covers one corridor** and started on 26 Sep 2026. It is a working demo, not a statewide feed.
- **Known open item:** the v2 SPF (25 terms) showed a convergence warning. The driver-facing "reasons" should use the stable v1 SPF (14 terms) until the SPF term set is re-selected. The signal table in `RiskHorizon_Dataset` is currently generated from v1 scores; `run_all` regenerates it.

## 11. Future work

1. **Exposure:** keep TomTom collecting and add the CE323-style counts to turn speed into approximate flow. Model risk *per vehicle-km*.
2. **More measured conditions:** download the 5 pending airports; add IMD rainfall stations.
3. **Road inventory:** add speed breakers, lanes and speed limits from field surveys or dash-cam imagery. Our segment table is ready to receive them.
4. **Post-crash alerting:** 58% of deaths happen after the first hour and 21% of reports are filed more than 6 h late. Phone or vehicle crash detection plus our road context could speed up response.
5. **Scaling:** the pipeline is state-agnostic. Any state with iRAD and OSM can be scored.
6. **Closing the loop:** agencies fix top segments, and the model measures the before/after change (EB before-after is the standard method).

## 12. How this answers the brief

| Brief asks | What we did |
|---|---|
| Real problem, named user, a decision | Serious crashes; ADAS/AV makers and road engineers; "which stretch ahead needs caution" |
| Understand the physical workflow | Police reporting, road geometry, lighting, weather at the crash hour |
| Collect evidence, cheaply but for real; collect more than once; record how and when | 8 sources, manifests, live TomTom every 15 min (running at pitch time) |
| Open format (Parquet preferred) | All tables in Parquet with schema, data catalog, samples and CSV copies |
| AI where it earns its place | Interpretable count models + boosted trees, validated on unseen months |
| **Understand** (sufficient to win) | EDA, data-quality audit, drift discovery, condition effects |
| **Predict** (bonus) | 49.7% of future serious crashes on 5% of road |
| **Recommend** (bonus) | Bands with vehicle actions; ranked fix-list for engineers |
| Actuate (optional) | Suggested cruise-control and alert actions per band |

**Judging criteria (25% each):**

- **Idea:** a road-risk attribute for the electronic horizon, built from local police data.
- **Data collection process:** scoped, logged, quota-aware and resumable, with a live collector.
- **Data collected:** 44k crashes linked to 247k segments, weather and light; every field labelled; data quality audited.
- **Impact:** a quarter more future serious crashes caught on the same road length; new hotspots flagged before they appear.

## 13. Reproducibility, folder map and privacy

**Folder map:** see `00_START_HERE.md`.

**Re-running:**

- **v1 end-to-end signal:** `07_Code/model_v1_signal/run_all.bat`
- **v2 final static model:** `07_Code/model_v2_final/`. Run `00_export_model_inputs.py`, then `01_feature_study.py` (optional, ~15 min), then `02_static_risk.py`.

**Privacy:**

- Raw iRAD records, crash-level tables and the crash-to-segment link are **classified and not included**.
- Everything shared is aggregated to segments, regions or summary tables.
- Lab data appears at group level only.
- API keys are typed at run time and never stored.

**Licences:**

- © OpenStreetMap contributors (ODbL)
- ERA5 / Open-Meteo (CC-BY 4.0)
- NASA open data
- TomTom readings shared as aggregates

## 14. References

1. Highway Safety Manual network screening and Empirical Bayes; comparison of EB vs PSI hotspot methods. *Sustainability* 2024: https://www.mdpi.com/2071-1050/16/4/1537
2. Generalized criteria for evaluating hotspot identification methods: https://www.sciencedirect.com/science/article/abs/pii/S0001457520303511
3. EB vs Bayesian hierarchical models in hotspot identification: https://journals.sagepub.com/doi/abs/10.1177/0361198119849899
4. GLMM and XGBoost for safety performance functions, *Scientific Reports*: https://pmc.ncbi.nlm.nih.gov/articles/PMC13469119/
5. Boosting techniques for calibrating freeway SPFs: https://pubmed.ncbi.nlm.nih.gov/34172259/
6. Time-stratified case-crossover of weather and road crashes, Singapore: https://www.sciencedirect.com/science/article/abs/pii/S2212095524004541
7. Severity of night-time crashes under low illumination: https://journals.sagepub.com/doi/full/10.1177/1687814019840940
8. MoRTH, *Road Accidents in India 2023* (Assam: 3,296 deaths); MoRTH blackspot definition.
