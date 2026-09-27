# 02 · Data collected (all CSV)

| Folder | Contents | Kind |
|---|---|---|
| `public/era5_weather/` | Hourly weather for 150 × 25 km cells at every crash hour and control hour (580,850 rows) | Inferred |
| `public/metar_airports/` | Guwahati airport reports (50,618) | Observed |
| `public/nasa_night_lights/` | 2024 night-light radiance per ~460 m pixel (1.45 M) | Observed |
| `public/openstreetmap/` | Pedestrian places, road points (crossings, signals, speed breakers), Assam boundary | Observed |
| `public/tomtom_live/` | Every live poll so far: corridor road speeds (`flow_tiles_all_polls.csv`, join `geom_hash` → `road_pieces.csv`) and 24 speed points | Observed, **still collecting** |
| `irad_cleaned/` | Cleaned police crash records (personal fields removed), uploaded separately | Observed |
| `schema/` | `DATA_CATALOG.csv` and column dictionaries | — |
| `explored_not_used/` | Data we collected and examined but did not put into the model (CE323 field studies) | Observed |

**Collected but not in the model yet:** the airport data is only used to check fog, and TomTom history starts on 26 Sep 2026 (live warnings only). The CE323 counts are in `explored_not_used/`.
