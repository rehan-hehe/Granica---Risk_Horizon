# 01 · Data collection

**Why:** the brief asks for real evidence gathered for a real decision. Ours is where and when serious crashes happen, plus the physical conditions around them.

| File | What it does | How |
|---|---|---|
| `irad_extraction/irad_to_excel.py` | Turns 44,928 police crash PDFs into 16 tables | Rule-based parser; removes 655 duplicates; logs 159 unreadable files; a 300-report re-check found 0 of 44,768 cells missing |
| `public_data/build_scope.py` | Decides **what to download**: the 150 weather cells, 6 airports, 19 traffic tiles and exact hours | Uses crash places and times. Builds the case-crossover design (each crash + control hours at the same place, weekday and clock time) |
| `public_data/fetch_weather.py` | ERA5 hourly weather for crash and control hours only (674 requests) | Pauses and retries on API quota |
| `public_data/fetch_metar.py` | Airport observations (visibility, fog) | Iowa Mesonet |
| `public_data/fetch_osm.py` | Roads, crossings, signals, speed breakers, pedestrian places | Geofabrik extract filtered with pyosmium |
| `public_data/fetch_viirs.py` | NASA night-time lights 2024 | Earthdata token typed at run time; resumable download with size and integrity checks (added after 401 errors and truncated files) |
| `public_data/tomtom_live_collector.py` | **Live traffic every 15 min** on the Guwahati corridor | Key typed at run time; hourly files; every poll logged |
| `scope/` | Scoping outputs (cells, airports, tiles, points, request list) | — |
| `logs/manifest*.jsonl` | Every download and poll: time, source, success/error counts, file hashes | Proof of process |

**Run:** `python public_data/build_scope.py <data_root>`, then each `fetch_*.py` (they can run in parallel), then `python public_data/tomtom_live_collector.py --key <KEY> --hours 50`.
