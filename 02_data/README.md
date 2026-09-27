# 02 · Data collected (all CSV)

| Folder | Contents | Kind |
|---|---|---|
| `public/era5_weather/` | Hourly weather for 150 × 25 km cells at every crash hour and control hour (580,850 rows) | Inferred |
| `public/metar_airports/` | Guwahati airport reports (50,618) | Observed |
| `public/nasa_night_lights/` | 2024 night-light radiance per ~460 m pixel (1.45 M) | Observed |
| `public/openstreetmap/` | Pedestrian places (POIs), road points (crossings, signals, speed breakers), Assam boundary | Observed |
| `public/tomtom_live/` | Every live poll so far: speed on corridor roads (`flow_tiles_all_polls.csv`, join `geom_hash` → `road_pieces.csv`) and 24 speed points | Observed, **still collecting** |
| `ce323_field_studies/` | Our digitised CE323 lab data: volumes, headways, spot speeds, parking, OD; `Validation_Log.csv` lists problems found (e.g. a 2,571 km/h speed from edited video times) | Observed |
| `irad_cleaned/` | Cleaned police crash records (personal fields removed): **uploaded separately**, see its README | Observed |
| `schema/` | `DATA_CATALOG.csv` (every file: stage, rows, source, kind, use) and column dictionaries | — |

**Collected but not in the model yet:**
- **Airport data:** used to check fog.
- **CE323 counts:** kept to calibrate traffic exposure later.
- **TomTom history:** starts on 26 Sep 2026; live warnings only for now.

We kept all three because they make the next step possible.
