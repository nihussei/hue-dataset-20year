# Data Provenance and Electricity Rates

## Energy data

Harvard Dataverse DOI:`10.7910/DVN/N3HGRN`; the project accepts the Kaggle mirror
`noahjanes/hourly-energy-usage-buildings-in-british-columbia`. Loads come from
`Residential\_\*.csv`, and `Solar.csv` is converted from watt scale to hourly kWh.

## BC Hydro defaults

* Flat: CAD 0.1270/kWh and CAD 0.2500/day.
* Tiered: CAD 0.1187/kWh then CAD 0.1408/kWh, 22.1918 kWh/day threshold,
CAD 0.2344/day.
* ToU: flat rate plus CAD 0.05/kWh at 16:00–21:00 and minus CAD 0.05/kWh
at 23:00–07:00.

## RTP baseline

Because no historical hourly BC Hydro residential RTP was supplied, the project
uses actual EIA-republished ICE `Mid C Peak` daily wholesale observations
(2017–2025), converted from USD/MWh to CAD/kWh using Bank of Canada FXUSDCAD.
The 2,000+ source rows are bundled at
`data/market/midc\_daily\_2017\_2025.csv`; the downloader is
`scripts/build\_midc\_price\_baseline.py`.

The hourly experimental retail signal adds a retail component, intraday profile,
net-load stress, PV-surplus relief, outage scarcity, and causal disturbance.
Realized RTP settles cost; only forecast RTP is visible to decision algorithms.

## Outages

No measured availability profile was supplied. Normal/moderate/stress profiles
are seeded scenarios, not historical reliability claims. Event tables are saved.

