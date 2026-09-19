# Forecast Integration

The supplied `LSTM_forecast_20yr.py` is retained for provenance but is not a
complete executable community forecast input: it references external HUE files
and checkpoints, and the active export path covers only one household.

This milestone therefore creates reproducible, causal forecasts directly from
the selected HUE histories:

```text
forecast(t) = 0.65 * actual(t-24h) + 0.35 * actual(t-168h)
```

Warm-up uses only observations available before the target hour. The first 168
hours are excluded from the HUE study horizon. Forecast and realised arrays are
stored separately throughout the environment.

- Rule-Based uses current measured load/PV/grid/price.
- MPC uses current status plus forecast load, PV, RTP and availability.
- Policy switching clones the live BESS/billing state, substitutes forecast
  streams, and compares candidate next-hour rewards.
- Reported cost, PV, reliability, degradation and reward use realised held-out
  data.

`forecast_accuracy_test.csv` reports per-prosumer MAE, RMSE, MAPE (where actual
values are non-negligible), and RTP MAE/RMSE. A future LSTM can replace these
columns without changing any controller or evaluation code by supplying the
canonical long-format schema in `sample_forecast_schema.csv`.
