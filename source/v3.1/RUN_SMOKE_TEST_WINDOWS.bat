@echo off
setlocal
cd /d "%~dp0"
py -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_experiment.py --synthetic-smoke --output SMOKE_RESULTS --prosumers 3 --episodes 1 --training-window-days 2 --validation-days 2 --test-days 3 --max-test-hours 48 --skip-plots
echo.
echo Finished. Open the SMOKE_RESULTS folder.
pause
