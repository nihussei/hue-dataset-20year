# Augmented 20-Year Hourly Usage of Energy (HUE) Dataset for Buildings in British Columbia

The original HUE dataset (https://github.com/smakonin/HUE.dataset) consists of hourly energy data from 28 residential households that are customers of power utility BC Hydro, along with corresponding weather data from the nearest weather station.

As of 2026, each household has at most 3 years of data. Household and weather data are augmented to provide 20 observed years. Every generated row is marked `is\_augmented = 1`; a reproducible scenario generator.

The 20-year series (2021-2040) is a labelled multivariate seasonal block-bootstrap scenario. `Solar.csv` supplies the PV shape. Load/PV forecasts are causal. Outages are reproducible research scenarios. RTP is anchored to historical EIA/ICE Mid-C daily prices converted with Bank of Canada FX; its intraday residential component is modeled and is not a historical BC Hydro residential RTP product.

Data is anonymized to protect donor identities.

## Loading

1. Select HUE homes with the longest simultaneous hourly overlap.
2. Fill only short gaps; remaining gaps use each home's hour-of-week pattern.
3. Estimate a month × day-of-week × hour template per home.
4. Obtain a multivariate residual matrix by subtracting the templates.
5. Resample 168-hour residual blocks jointly across homes, preferring the same
month. This retains serial and cross-house dependence.
6. Add a shared AR(1) innovation (`phi=0.92`, SD `0.025`) and 0.7% annual load growth.

## PV and forecasts

HUE Solar watt-scale AC output is converted to hourly kWh. Household capacity multipliers (0.70-1.30), community-correlated daily cloud factors, and 0.5%
annual PV degradation are applied. Forecasts use only lag-24, lag-168, and expanding past history. Validation and test consist of untouched later years.

## Validation

`hue\_augmentation\_validation.csv` compares observed and first-generated-year mean, p95, lag-1, and lag-24 statistics. `hue\_augmentation\_yearly\_energy.csv`
contains every prosumer/year total. Defaults are 2021–2040, seed 42.

# References
Original downloadable dataset: https://doi.org/10.7910/DVN/N3HGRN

Original dataset documentation: https://doi.org/10.1016/j.dib.2019.103744
