# HUE 20-Year Augmentation Methodology

The HUE files contain at most about three years per building and do not provide
20 observed years. Every generated row is marked `is\_augmented = 1`; a
reproducible scenario generator, not a claim of future measurements.

## Load

1. Select HUE homes with the longest simultaneous hourly overlap.
2. Fill only short gaps; remaining gaps use each home's hour-of-week pattern.
3. Estimate a month × day-of-week × hour template per home.
4. Obtain a multivariate residual matrix by subtracting the templates.
5. Resample 168-hour residual blocks jointly across homes, preferring the same
month. This retains serial and cross-house dependence.
6. Add a shared AR(1) innovation (`phi=0.92`, SD `0.025`) and 0.7% annual load growth.

## PV and forecasts

HUE Solar watt-scale AC output is converted to hourly kWh. Household capacity
multipliers (0.70-1.30), community-correlated daily cloud factors, and 0.5%
annual PV degradation are applied. Forecasts use only lag-24, lag-168, and
expanding past history. Validation and test consist of untouched later years.

## Evidence

`hue\_augmentation\_validation.csv` compares observed and first-generated-year
mean, p95, lag-1, and lag-24 statistics. `hue\_augmentation\_yearly\_energy.csv`
contains every prosumer/year total. Defaults are 2021–2040, seed 42.

