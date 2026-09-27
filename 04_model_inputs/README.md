# 04 · From raw data to model inputs

| Code (in `code/`) | What it builds | Key choices |
|---|---|---|
| `build_when_table.py` | **WHEN table**: 193,904 place-hours (44,112 crashes + 149,792 control hours) with weather, 3-h rain build-up, airport observations (±30 min), computed sun position, police conditions | Case-crossover: each crash compared with the *same place and clock time* on other days, so only the conditions differ |
| `make_when_readable.py` | `summaries/WHEN_dataset_readable.xlsx` | Odds ratios, severity, police-vs-measured checks |
| `build_where_table.py` | **WHERE table**: Assam roads cut into 246,789 segments of ≤ 500 m (the MoRTH blackspot unit), with geometry, junctions, crossings, pedestrian places, night light; crashes snapped within 100 m (90.3%, median 5 m) | Segment = the unit a car receives |
| `00_export_model_inputs.py` | Converts model tables to readable CSV | From here on, models read CSV only |
| `rh_features.py` | **Feature engineering**, 52 candidates: road-class rank, bends and "bend after a straight", junction density, pedestrian places at 200 m / 500 m / 2 km, night light at 3 and 10 km, distance to town and main road, road density, crash history around each segment (exact search, own segment excluded) and drift-corrected versions | Two leaks were found and fixed here (see `06_results`) |

| Data (in `data/`) | Rows | Notes |
|---|---|---|
| `where_segments.csv` | 246,789 | Features + crash counts per segment (aggregated, no personal data) |
| `where_segments_geometry.csv` | 246,789 | Segment shapes (WKT, lon/lat), join on `segment_id` |
| `where_model_input_final_dictionary.csv` | — | **The exact final input**: 33 features for the trees, 25 terms for the regression, with definitions |
| `model_inputs_dictionary.csv` | — | Column meanings |
