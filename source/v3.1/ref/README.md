- `PlotRewardFunction_original.py`: original reward, tariff and allocation work.
- `prosumer_original.py`: prosumer data model.
- `LSTM_forecast_20yr_original.py`: weather/load forecast-generation script.
- `Weather_YVR_extended.csv`: supplied weather series.

The production implementation is in `microgrid_opt/`. The default final
experiment now ingests the public HUE files directly and constructs an aligned,
causal long-format study table. `sample_forecast_schema.csv` remains available
when the client later supplies externally generated forecasts.
