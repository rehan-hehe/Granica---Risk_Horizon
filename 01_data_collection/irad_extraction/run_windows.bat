@echo off
REM ---- iRAD PDFs -> Excel. Double-click, or run from a terminal in this folder. ----
REM Edit these two paths if your folders are different:
set INPUT=D:\DataSet
set OUTPUT=D:\iRAD_output\iRAD_Assam_all.xlsx

python -m pip install --upgrade pdfplumber openpyxl pandas pyarrow
python "%~dp0irad_to_excel.py" "%INPUT%" "%OUTPUT%" --parquet --verify 300
echo.
echo Finished. Output: %OUTPUT%
pause
