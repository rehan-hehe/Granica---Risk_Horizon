@echo off
REM Weather (Open-Meteo ERA5): ~2 h. Run on ONE device only (quota is per internet connection).
cd /d "%~dp0"
python -m pip install --upgrade pandas pyarrow
python fetch_weather.py
python fetch_weather.py --status
pause
