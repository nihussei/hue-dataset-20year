# Kaggle Run Guide

## Attach inputs

Create a Kaggle notebook and attach:

1. this project ZIP/folder;
2. `Hourly Energy Usage: Buildings in British Columbia`
   (`noahjanes/hourly-energy-usage-buildings-in-british-columbia`).

Kaggle may unpack inputs several levels deep. The supplied notebook discovers
`run_experiment.py` and `Solar.csv` recursively.

## Run

Open `notebooks/00_KAGGLE_COMPLETE_20Y_RUN.ipynb`, choose:

```python
RUN_MODE = "full"         # final evidence
# RUN_MODE = "preliminary"  # 48-hour integration check
```

Then run all cells. Full mode generates the 20-year data, six RL models, every
held-out comparison, figures, tables, reports, audit, and a downloadable ZIP in
`/kaggle/working`.

Equivalent command:

```bash
python run_experiment.py \
  --hue-root /kaggle/input/.../HUE-folder \
  --output /kaggle/working/HUE_AUGMENTED_20Y_FINAL \
  --prosumers 10 --augment-years 20 --augmentation-start-year 2021 \
  --validation-years 2 --test-years 1 \
  --episodes 30 --rl-seeds 42,52,62 --training-window-days 90 \
  --outage-scenario moderate --full --save-augmented-data --resume
python scripts/audit_outputs.py /kaggle/working/HUE_AUGMENTED_20Y_FINAL
```

If Kaggle interrupts the session, rerun with `--resume`. Completed augmented
data, seed models, histories, and hourly trajectories are reused. Finally,
download `/kaggle/working/HUE_AUGMENTED_20Y_FINAL.zip` from the Output panel.
