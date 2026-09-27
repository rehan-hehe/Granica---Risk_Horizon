@echo off
REM Airport weather reports (visibility/fog), 6 Assam airports: ~5 min.
cd /d "%~dp0"
python -m pip install --upgrade pandas pyarrow
python fetch_metar.py
pause
