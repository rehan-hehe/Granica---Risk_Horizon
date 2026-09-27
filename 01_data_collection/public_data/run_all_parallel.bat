@echo off
REM Opens one window per dataset so they all run at the same time on this PC.
cd /d "%~dp0"
start "Weather" cmd /k run_weather.bat
start "METAR" cmd /k run_metar.bat
start "OSM" cmd /k run_osm.bat
start "Night lights" cmd /k run_viirs.bat
