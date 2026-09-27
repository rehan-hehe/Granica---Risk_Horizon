@echo off
REM NASA night lights 2024 (street-lighting proxy): ~5 min. Needs a free token from https://urs.earthdata.nasa.gov (Generate Token).
cd /d "%~dp0"
python -m pip install --upgrade pandas pyarrow h5py requests
set /p TOKEN=Paste Earthdata token: 
python fetch_viirs.py --token %TOKEN%
pause
