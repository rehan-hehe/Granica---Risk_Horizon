@echo off
REM Risk Horizon model pipeline (~3 min). Needs: pip install pandas pyarrow shapely scikit-learn statsmodels matplotlib openpyxl
cd /d "%~dp0"
for %%s in (00_export_model_inputs 01_static_risk 02_condition_multipliers 03_signal_bands 04_report) do (
  echo === %%s
  python %%s.py "%~dp0..\.."
  if errorlevel 1 exit /b 1
)
echo All done. Results in 2_readable\11_model_results
pause
