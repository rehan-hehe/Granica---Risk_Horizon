# Risk Horizon: data inventory and fetch plan

**Rule: we fetch only what the crash footprint needs.** `build_scope.py` reads the 44,112 geocoded iRAD crashes (Jan 2023 – May 2026) and writes the `scope/` files. Each downloader is only allowed to fetch the places and dates listed there.

## Status

| # | Dataset | Aim it serves | Scope (why this much and no more) | Status |
|---|---|---|---|---|
| 0 | **iRAD Assam** | Labels (where and when crashes happen) | 44,112 crashes with coordinates and time | ✅ have |
| 1 | **Case-crossover skeleton + sun position** | WHEN model: each crash compared with the same place and clock time on other days | Time-stratified design: controls are the same weekday and time in the same calendar month (3.4 per crash). 193,904 rows; sun elevation computed per row | ✅ built (`derived/`) |
| 2 | **Open-Meteo ERA5 hourly weather** | WHEN: rain, humidity, low cloud, wind | Only the 150 cells (25 km) containing crashes. Only the dates of each crash and its controls. **Only the case/control hour plus the 3 hours before it is kept**: 580,850 cell-hours, 12% of all hours. 674 requests, about 9.4k weighted calls (fits in one day's free quota); 7 variables | ⏳ run `1_fetch_public_data.bat` |
| 3 | **METAR (airport weather reports)** | WHEN: measured visibility and fog, which ERA5 lacks | 6 airports, each with ≥1,100 crashes within 30 km (together 32% of crashes). Only reports in the needed hours are kept (106,856 station-hours) | ⏳ same script |
| 4 | **OpenStreetMap** | WHERE: lanes, class, divided, junctions, speed breakers, signals, crossings, markets/schools/bus stops | Assam only (crash bounding box + 5 km); drivable roads; 14 road tags; 8 node types; pedestrian-generating places. The 104 MB source file is deleted after extraction | ⏳ same script |
| 5 | **NASA Black Marble VNP46A4 2024** | WHERE: street-lighting proxy | 1 year, 2 tiles, cropped to Assam | ⏳ needs a free Earthdata token |
| 6 | **TomTom live traffic** | NOW: live congestion, plus a typical-by-hour traffic profile | 19 map tiles (97% of hotspot crashes in the 3 corridor micro-sites) every 15 min; 24 hotspot points hourly. That's ~3.6k + 1.2k requests over 48 h, against free limits of 200k and 20k/month | ⏳ `2_start_live_traffic.bat` at the event (needs a free TomTom key) |
| 7 | **CE323 lab data** | Checking TomTom speeds against measured speeds (Amingaon) | Already in `../Lab_Dataset` | ✅ have |

## Why weather can't be trimmed further

The weather API only serves continuous date ranges and bills per location per started 2 weeks, so we request the needed dates in runs and throw away the unneeded hours immediately. Sparse rural cells download only a few days each.

The Guwahati cell is the exception. It has a crash on almost every day, so its crashes and controls need about 95% of its hours anyway. It is also where most of the data is.

## Why the downloads must run on your PC

This session's network policy blocks the data hosts (open-meteo.com, mesonet.agron.iastate.edu, geofabrik.de, nasa.gov, tomtom.com). You can either:

- run the two `.bat` files on your own PC, or
- ask your Claude org admin to allow those domains (Admin settings → Capabilities).

Everything else (scoping, joins, 500 m road segments, checks) is done here once the files land in `data/`.

## How to run (one downloader per dataset)

| Dataset | Run | Script | Time | Needs from `scope/` |
|---|---|---|---|---|
| Weather | `run_weather.bat` | `fetch_weather.py` | ~2 h | scope.json, era5_cells.csv, weather_requests.csv, weather_needed_hours.parquet |
| Airport weather (METAR) | `run_metar.bat` | `fetch_metar.py` | ~5 min | scope.json, metar_stations.csv, metar_needed_hours.parquet |
| Roads (OSM) | `run_osm.bat` | `fetch_osm.py` | ~10 min | scope.json |
| Night lights | `run_viirs.bat` | `fetch_viirs.py` | ~5 min + free Earthdata token | scope.json |
| All four at once on one PC | `run_all_parallel.bat` | — | ~2 h (weather is the long one) | — |
| Live traffic (at the event) | `2_start_live_traffic.bat` | `tomtom_live_collector.py` | 48 h | tomtom_*.csv |

**How the downloaders behave**
- They are independent. Each can run on a different device and at the same time; a limit on one service doesn't affect the others.
- Each is resumable: re-running skips anything already saved.
- Each writes its own log (`manifest_weather.jsonl`, `manifest_metar.jsonl`, ...), so they never clash.

**Running on another device**
1. Copy the whole `Risk_Horizon_Data` folder to that device.
2. Run its `.bat`.
3. Copy back the new files in `data/` (plus its `manifest_*.jsonl`) into this folder's `data/`.

**Only one weather download at a time.** Open-Meteo's free limit is per internet connection, so two PCs on the same Wi-Fi share it.

**Legacy files.** `1_fetch_public_data.bat` and `fetch_public_data.py` are the older all-in-one versions. If that window is still running weather, let it finish. When it moves on to `[2/4]`, close it if another device already did METAR/OSM/night lights. Don't start `run_weather.bat` while it's running.

**Live traffic.** At the hackathon, run `2_start_live_traffic.bat` with a free TomTom key. First check TomTom's terms on storing traffic data, and publish aggregates if unsure.

## Expected sizes

| Output | Rows | Size |
|---|---|---|
| `weather_era5_hourly.parquet` | ~0.58M rows | ~6 MB |
| `metar_assam_airports.parquet` | ~0.2M reports | ~3 MB |
| `osm_roads` / `osm_road_nodes` / `osm_pois` | | a few tens of MB |
| `viirs_vnp46a4_2024_assam.parquet` | ~1.5M pixels | ~15 MB |
| Live traffic | readings store a hash, not geometry (each road piece's geometry is saved once) | a few MB/day |

## Lineage

- Every call is logged in `manifest.jsonl` with URL, time, bytes, SHA-256 and row count.
- Observed / inferred / computed labels:
  - **Observed:** METAR, OSM, VIIRS, TomTom, iRAD
  - **Inferred:** ERA5 (reanalysis); TomTom "typical hour" profiles applied to historical crashes
  - **Computed:** sun elevation

## Finding already made (from the skeleton, no download needed)

We compared the police-recorded light condition with the sun elevation calculated from each crash's time and place. They **agree 85.4% of the time**. Most disagreements are "Twilight" or "Dawn" crashes that happened after full dark (1,841 of 3,214).

Also, **47% of crash times are rounded to :00 or :30**. So there is up to 30 minutes of time uncertainty; hourly weather is fine, but twilight labels are soft.
