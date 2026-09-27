# 03 · Exploratory analysis and data quality

**Why:** before modelling, we had to prove the data could be trusted and learn what drives serious crashes. Each finding changed the plan.

| Folder | What it shows | Finding → decision |
|---|---|---|
| `01_irad_crash_eda/` | Notebook + rendered HTML, 8 figures, reliability and hotspot tables | iRAD holds 98% of MoRTH's 2023 deaths → target = fatal + grievous crashes. Behaviour fields ≤ 47% usable → never used as labels. "Sunny" on 60% of night crashes → conditions measured, not copied from forms |
| `02_region_selection/` | Region scorecard and corridor hotspots | Picked the Guwahati corridor for live traffic collection |
| `03_public_data/` | OSM summary and map, weather by month, night-light map, airport coverage | OSM tags are thin (lanes on 1% of roads) → use geometry and network features |
| `04_conditions/` | Matched odds ratios, severity by condition, police vs measured | Rain slightly *lowers* crash odds (0.955), so a condition never lowers a warning. Darkness raises the fatal share (26.7% vs 22.4%) |
| `05_live_traffic/` | Live snapshot map and collection status | Collection running |
| `06_data_quality/` | `reporting_drift_by_region.csv`, `crashes_per_month.csv`, `police_vs_measured.csv`, `light_police_vs_computed.csv`, `field_reliability.csv`, `CE323_lab_validation_log.csv`, `metar_coverage.csv` | iRAD reporting grew 1.5× unevenly (0.8×–3.8×) → drift-corrected features. Jan and Jul 2024 missing → rates per real data month. Police fog agrees weakly → dropped |
