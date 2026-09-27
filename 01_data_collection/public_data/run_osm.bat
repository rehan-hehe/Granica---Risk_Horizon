@echo off
REM OpenStreetMap roads/junctions/speed breakers/POIs for Assam: ~10 min, downloads 104 MB then keeps only the needed parts.
cd /d "%~dp0"
python -m pip install --upgrade pandas pyarrow osmium
python fetch_osm.py
pause
