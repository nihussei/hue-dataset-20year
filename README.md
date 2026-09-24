# Augmented 20-Year Hourly Usage of Energy (HUE) Dataset for Buildings in British Columbia with Real-Time Prices & Grid-Outage Events

The original HUE dataset (https://github.com/smakonin/HUE.dataset) consists of hourly energy data from 28 residential households that are customers of power utility BC Hydro, along with corresponding weather data from the nearest weather station. As of 2026, each household has at most 3 years of data.

The augmented 20-year series (2021-2040) is a labelled multivariate seasonal block-bootstrap scenario for 10 prosumers. `Solar.csv` supplies the PV shape. Load/PV forecasts are causal. Outages are reproducible research scenarios. Real-time pricing (RTP) is anchored to historical EIA/ICE Mid-C daily prices converted with Bank of Canada FX; its intraday residential component is modeled and is not a historical BC Hydro residential RTP product. Every generated row is marked `is_augmented = 1`; a reproducible scenario generator.

Data is anonymized to protect donor identities.

## Repository Structure

```text
hue-dataset-20year-main/
├── LICENSE                          MIT license
├── README.md                        Dataset overview, augmentation summary, references
├── raw/                             Original HUE seed data
│   ├── Residential_1.csv ... Residential_28.csv
│   │                                28 household hourly load files
│   │                                (date, hour, energy_kWh)
│   ├── Solar.csv                    1-year hourly PV output template
│   │                                (date, hour, dc_output, ac_output)
│   ├── Weather_YVR.csv              Hourly weather, Vancouver Int'l station
│   │                                (2012-01-01 to 2020-05-13)
│   ├── Weather_YYJ.csv              Hourly weather, Victoria Int'l station
│   │                                (2015-01-01 to 2019-12-31)
│   └── Holidays.csv                 Daily calendar covariates
│                                    (day, weekend, holiday, dst; 2012-2018)
├── source/
│   └── v3.1/                        Augmentation & control pipeline code 
│       ├── microgrid_opt/            Core Python package
│       │   ├── augmentation.py       20-year block-bootstrap load/PV generator
│       │   ├── core.py               Canonical schema definition & validation
│       │   ├── config.py             Experiment configuration dataclasses
│       │   ├── data.py               HUE ingestion, outage simulation, RTP construction
│       │   ├── evaluation.py         Forecast/controller evaluation metrics
│       │   ├── market.py             Tariff/pricing logic
│       │   ├── plots.py              Figure generation
│       │   └── tariffs.py             Flat/tiered/ToU/RTP tariff definitions
│       ├── run_experiment.py          CLI entry point for the full pipeline
│       ├── notebooks/
│       │   └── 00_KAGGLE_COMPLETE_20Y_RUN.ipynb
│       │                                End-to-end Kaggle runner
│       ├── scripts/                  Audit, price-baseline builder,
│       │                              input validation, self-checks
│       ├── ref/                      Legacy/reference implementations for provenance
│       ├── tests/                    Pytest suite (core logic, augmentation/market)
│       ├── data/
│       │   └── market/
│       │       └── midc_daily_2017_2025.csv
│       │                                2,170 daily EIA/ICE Mid-C wholesale prices
│       │                                anchoring the RTP series
│       ├── *.md                      Methodology, provenance, assumptions,
│       │                              forecast-integration, Kaggle-run guide
│       ├── *.bat                     Windows launchers (full run, smoke test,
│       │                              validation run)
│       ├── requirements.txt
│       └── pytest.ini
└── results/                          Outputs of one pipeline run 
    ├── RESULTS_SUMMARY.md             Narrative summary of the run
    ├── VALIDATION_CHECKS.json         Energy-balance / SoC self-consistency checks
    ├── figures/
    │   └── *.png                      4 figures: augmentation validation,
    │                                    20-year energy, per-prosumer energy,
    │                                    RTP/outage overview
    └── tables/                        8 CSV files
        ├── hue_prosumer_mapping.csv
        │                                Prosumer-to-HUE household map, PV multiplier,
        │                                critical fraction
        ├── hue_augmentation_config.csv
        │                                Augmentation parameters (seed, horizon,
        │                                AR(1) terms)
        ├── hue_augmentation_validation.csv
        │                                Historical vs. augmented-year mean/p95/
        │                                autocorrelation per prosumer
        ├── hue_augmentation_yearly_energy.csv
        │                                200 rows: annual load/PV totals
        │                                per prosumer × year
        ├── hue_outage_events.csv        120 simulated outage events
        │                                under the moderate scenario
        ├── data_split_manifest.csv      Train/validation/test date boundaries
        │                                and hour counts
        ├── balanced_tariff_assignments_from_training.csv
        │                                Per-prosumer cost under each tariff
        │                                and selected tariff
        └── forecast_accuracy_test.csv   Held-out test-set forecast error
                                         per prosumer
```
## Loading

1. Select HUE homes with the longest simultaneous hourly overlap.
2. Fill only short gaps; remaining gaps use each home's hour-of-week pattern.
3. Estimate a month × day-of-week × hour template per home.
4. Obtain a multivariate residual matrix by subtracting the templates.
5. Resample 168-hour residual blocks jointly across homes, preferring the same
month to retain serial and cross-house dependence.
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
