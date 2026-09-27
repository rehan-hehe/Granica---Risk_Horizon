@echo off
REM ---- Start the live TomTom collector for the Guwahati corridor. Leave this window open for the 48 h. ----
cd /d "%~dp0"
python -m pip install --upgrade pandas pyarrow mapbox-vector-tile
set /p KEY=Paste TomTom API key:
python tomtom_live_collector.py --key %KEY% --hours 50
pause
