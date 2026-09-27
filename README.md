# Risk Horizon — a road-risk signal for the car's electronic horizon

**Granica × IIT Guwahati Hackathon: "Bring the Physical World to AI"** · Team DirectorsInc

> A road has to kill before anyone flags it. We turned **44,928 Assam police crash reports**, OpenStreetMap, NASA night lights, hourly weather and **live traffic (still collecting)** into a tested record of Assam's roads. A transparent model then gives driver-assist cars and road engineers a risk band for every 500 m of road.

| | |
|---|---|
| **User** | Driver-assist / self-driving systems (a risk attribute in the map ahead) and road-safety engineers (what to fix first) |
| **Question** | Which 500 m of road ahead is dangerous, and when does it get worse? |
| **Headline** | Trained Jan 2023–Jun 2025, judged once on Jul 2025–May 2026: **49.7%** of future serious crashes fall in the riskiest **5%** of road, vs **39.5%** for today's blackspot-list approach (+10.6 pts, 95% CI 8.1–12.2). **184 of 236** new hotspots flagged before they appeared. |
| **Deck / report** | [`deck/RiskHorizon_Deck.pdf`](deck/RiskHorizon_Deck.pdf) · [`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md) |
| **Data card** | [`DATA_CARD.md`](DATA_CARD.md): row counts, windows, cadence, sources, observed / inferred / computed, how AI was used |

## Repository map (read in this order)

| Folder | What it shows | Why it exists |
|---|---|---|
| [`01_data_collection/`](01_data_collection) | PDF parser, scoping script, 4 downloaders, live TomTom collector, download logs | How the physical world was captured, and what we did when collection failed |
| [`02_data/`](02_data) | All collected data as CSV (public layers, live traffic, CE323 field counts), schema and catalog | Anyone can load and inspect the evidence |
| [`03_eda/`](03_eda) | Crash EDA, region selection, public-data summaries, condition analysis, data-quality proofs | What we learned and which problems we found |
| [`04_model_inputs/`](04_model_inputs) | Code that turns raw data into model tables; the segment table (246,789 rows); feature dictionary | Exactly what the model sees |
| [`05_model/`](05_model) | Final pipeline (feature study → WHERE → WHEN → signal bands) and `run_pipeline.ipynb` | How the model is built and why this split |
| [`06_results/`](06_results) | Held-out test results, bootstrap CIs, feature catalog, condition effects, signal bands, figures | What the results mean |

## Pipeline

```
police PDFs ─▶ irad_to_excel.py ─▶ crash tables ─▶ EDA ─▶ build_scope.py (fetch only crash places/hours)
      ▶ fetch_weather / fetch_metar / fetch_osm / fetch_viirs  +  tomtom_live_collector (every 15 min, running)
      ▶ build_when_table.py (crash hour vs control hours)   build_where_table.py (500 m segments)
      ▶ rh_features.py + 01_feature_study.py (select on 2023–mid 2024 → validate mid 2024–mid 2025)
      ▶ 02_static_risk.py (WHERE)  03/04 (WHEN)  ▶ bands 0–3 per 500 m ▶ what the car / engineer does
```

## Run it

```bash
pip install -r requirements.txt
jupyter notebook 05_model/run_pipeline.ipynb   # runs every step; see 05_model/README.md for the folder layout
```
Keys (TomTom, NASA Earthdata) are typed at run time and never stored.

## Privacy
Raw police records are **not** in this repository. `02_data/irad_cleaned/` explains the cleaned release (personal fields removed). Everything else is public data or aggregated to road segments.
