# 05 · Live traffic (TomTom), collected by us and still running

**Why:** Assam publishes no traffic counts. Without them, crash risk can only be expressed per km of road, not per vehicle. So on 26 Sep 2026 we started our own collection on the Guwahati corridor (Jalukbari – Saraighat – Amingaon, NH-27 and G.S. Road). We chose the corridor with the region scorecard in `../02_region_selection`.

**How:** `01_data_collection/public_data/tomtom_live_collector.py` polls the TomTom Traffic API:
- **19 flow tiles (zoom 13) every 15 minutes:** every road piece TomTom has live data for, with its speed relative to free flow (`traffic_level`, where 1 = free flow)
- **24 hotspot points every hour:** current speed, free-flow speed and travel time at fixed points on our crash hotspots

Every poll is written to `manifest.jsonl` with its time and ok/error counts. The API key is passed on the command line at run time and is never stored.

## Snapshot in this folder (rebuilt 28 Sep 2026)

| | |
|---|---|
| Window | 26 Sep 18:24 → 28 Sep 03:26 IST (33 h) |
| Flow polls | 92 in the data (104 logged; 10 failed) |
| Flow readings (road piece × poll) | 47,071 |
| Distinct road pieces seen | 17,750 |
| Hotspot-point readings | 552 (23 polls × 24 points) |
| Failed polls | 10, each followed by a successful poll |
| Gaps over 25 min | 5; the longest was 4 h 51 min (27 Sep 20:18 → 28 Sep 01:09), when nothing was logged at all, most likely because the laptop slept or lost network |

The collector writes one data file per hour, so the most recent partial hour is not included yet.

## Files

| File | What it is |
|---|---|
| `collection_status.csv` | the numbers above |
| `collection_log_tomtom.csv` | every poll from the manifest: time (IST), source, ok / error counts |
| `collection_gaps.csv` | every pause longer than 25 min, with when it started and when collection resumed |
| `flow_polls_timeseries.csv` | per poll: road pieces reported, cumulative readings, mean speed ÷ free-flow, share of pieces below half of free-flow speed, closures |
| `segment_points_summary.csv` | per hotspot point: polls, mean speed, free-flow speed, lowest and mean ratio, hours below 0.7 |
| `hour_of_day_profile.csv` / `.png` | average by hour of day (IST) |
| `live_traffic_timeseries.png` | road pieces per poll and mean speed ratio over time; gaps are left blank, not joined |
| `live_dataset_growth.png` | cumulative readings over time |
| `hotspot_points_heatmap.png` | 24 points × hour; grey = no poll |
| `live_flow_snapshot.png` | the very first poll (26 Sep), kept as the starting point |

The raw data for every poll is in `02_data/public/tomtom_live/` (`flow_tiles_all_polls.csv`, `segment_points_all_polls.csv`, and `road_pieces.csv` with the road shapes).

## What it shows so far

- **A clear daily rhythm.** The evening peak is the slowest time of day: at 18:00 IST, mean speed is 63% of free flow and about 1,400 road pieces are reported per poll. From midnight to 4 am, speeds are about 90% of free flow and only about 150 pieces are reported. The hotspot points show the same pattern (ratio 0.78 at 18:00, about 0.98 at night).
- **The number of pieces reported rises with congestion.** TomTom reports mostly roads where it has live probe data and where traffic departs from free flow. So "pieces reported" is a congestion signal, not a road count.

## What it can't tell yet

It covers one corridor and about 1.5 days. That is enough to prove the collection works and to see daily patterns, but not yet enough to estimate traffic volume. The next step is turning speed patterns plus field counts (CE323-style) into exposure, so risk can be expressed per vehicle.

Rebuild this folder with `build_live_eda.py` (reads the hourly collector files and `manifest.jsonl`, rewrites every table and figure here and the raw CSVs in `02_data/public/tomtom_live`).
