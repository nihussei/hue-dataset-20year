@echo off
setlocal
if "%~1"=="" (
  echo Usage: RUN_FULL_WINDOWS.bat C:\path\to\HUE [output_folder]
  exit /b 1
)
set HUE_DATA=%~1
set RESULT_DIR=%~2
if "%RESULT_DIR%"=="" set RESULT_DIR=HUE_CLIENT_RESULTS
python run_experiment.py --hue-root "%HUE_DATA%" --output "%RESULT_DIR%" --prosumers 10 --episodes 30 --training-window-days 90 --outage-scenario moderate --full
if errorlevel 1 exit /b 1
python scripts\audit_outputs.py "%RESULT_DIR%"
endlocal
