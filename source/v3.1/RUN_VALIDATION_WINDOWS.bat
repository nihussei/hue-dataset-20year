@echo off
python scripts\run_self_checks.py
if errorlevel 1 exit /b 1
python run_experiment.py --synthetic-smoke --output FULL_PATH_VALIDATION --prosumers 3 --episodes 1 --training-window-days 2 --validation-days 2 --test-days 3 --max-test-hours 48 --full --skip-plots
