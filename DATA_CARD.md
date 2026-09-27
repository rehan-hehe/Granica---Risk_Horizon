# Data card

| Dataset | Rows | Window | Cadence | Source | Kind | In repo |
|---|---|---|---|---|---|---|
| iRAD police crash reports (Assam) | 44,928 PDFs → 44,114 crashes (16 tables) | Jan 2023 – May 2026 | per crash, as filed by police | MoRTH iRAD | **Observed** | cleaned release only (`02_data/irad_cleaned/`) |
| Road segments (WHERE table) | 246,789 segments, 81,169 km | OSM snapshot Sep 2026; crash counts 2023–May 2026 | static | derived from OSM + VIIRS + iRAD counts | **Computed** (features) / observed (counts) | `04_model_inputs/data/where_segments.csv` |
| OpenStreetMap roads, road points, POIs | 190,077 ways; 4,319 points; 10,551 POIs | snapshot Sep 2026 | one-off | Geofabrik NE India (ODbL) | Observed | POIs, points, boundary in `02_data/public/openstreetmap` (roads via segments) |
| ERA5 weather | 580,850 cell-hours (150 cells × crash and control hours) | Dec 2022 – May 2026 | hourly | Copernicus ERA5 via Open-Meteo (CC-BY 4.0) | **Inferred** (reanalysis) | `02_data/public/era5_weather` |
| Airport weather (METAR) | 50,618 reports (Guwahati VEGT) | Dec 2022 – May 2026 | 30 min | Iowa Environmental Mesonet | Observed | `02_data/public/metar_airports` |
| NASA Black Marble night lights | 1,445,970 pixels (~460 m) | 2024 annual | one-off | NASA VNP46A4 | Observed | `02_data/public/nasa_night_lights` |
| TomTom live traffic | 19 map tiles + 24 points per poll | from 26 Sep 2026, **still running** | tiles every 15 min, points hourly | TomTom Traffic API (collected by us) | Observed | `02_data/public/tomtom_live` |
| CE323 field studies | 23 tables (19–644 rows) | course field work | manual counts | IIT Guwahati CE323 lab groups | Observed | `02_data/ce323_field_studies` |
| Sun position, case-crossover controls | 193,904 place-hours | = crash window | per crash | computed (pvlib) | **Computed** | built by code |

**Synthetic data:** none.

**Known gaps and bias:**
- No crashes in Jan 2024 or Jul 2024.
- iRAD reporting grew 1.5× unevenly across regions between 2023 and 2024.
- 15.5% of police reports are skeletal (summary only).
- Behaviour fields are under half filled.
- 9.7% of crashes could not be placed on a road within 100 m.
- There are no traffic volumes.

See `03_eda/06_data_quality`.

**How AI was used:**
1. **Models** (the AI that makes decisions):
   - negative-binomial regression and gradient-boosted Poisson trees for road risk, with Empirical Bayes
   - conditional logistic regression for condition effects
   - logistic regression for severity

   All trained on past months and tested on unseen months.
2. **Parsing** the PDFs is rule-based (no AI).
3. **An AI assistant (Claude)** helped write code, run analyses and draft documents. Every number was produced by the code in this repository, and the team can explain each step.
