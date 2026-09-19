# Results Summary

**Data source:** `HUE\_observed\_profiles\_multivariate\_20year\_augmentation`

HUE residential files provide the observed source profiles. The 20-year series is a labelled multivariate seasonal block-bootstrap scenario, not 20 measured years. `Solar.csv` supplies the PV shape. Load/PV forecasts are causal. Outages are reproducible research scenarios. RTP is anchored to historical EIA/ICE Mid-C daily prices converted with Bank of Canada FX; its intraday residential component is modeled and is not a historical BC Hydro residential RTP product.

## Simulated outage scenario

Generated events across the complete data horizon: **120**; total unavailable hours: **510**. Future unplanned outage status is hidden from MPC and policy switching; only current grid status is observable.


## Forecast diagnostics on the test set

These errors quantify the lag-based decision forecasts used by MPC. 

|prosumer\_id|load\_mae\_kwh|load\_rmse\_kwh|load\_mape\_percent|pv\_mae\_kwh|pv\_rmse\_kwh|pv\_mape\_percent|rtp\_mae\_cad\_per\_kwh|rtp\_rmse\_cad\_per\_kwh|
|-|-|-|-|-|-|-|-|-|
|P1|0.397383|0.585364|53.2183|0.226183|0.464298|104.042|0.0130766|0.0222354|
|P10|0.552447|0.784916|42.0043|0.2932|0.601868|105.858|0.0130766|0.0222354|
|P2|0.435432|0.600719|44.0671|0.242937|0.498691|104.284|0.0130766|0.0222354|
|P3|0.448009|0.656433|73.9883|0.309954|0.63626|105.891|0.0130766|0.0222354|
|P4|0.166638|0.24976|49.2953|0.17592|0.361121|100.632|0.0130766|0.0222354|
|P5|0.349311|0.530775|60.2374|0.209428|0.429906|102.972|0.0130766|0.0222354|
|P6|0.383803|0.563981|71.4306|0.276445|0.567475|105.052|0.0130766|0.0222354|
|P7|0.277106|0.406173|61.9447|0.259691|0.533083|104.457|0.0130766|0.0222354|
|P8|0.192742|0.286128|39.6707|0.326708|0.670653|105.863|0.0130766|0.0222354|
|P9|0.467749|0.688814|67.1966|0.192674|0.395513|102.217|0.0130766|0.0222354|

## Augmentation validation

Source vs. generated statistics are shown per prosumer; see the methodology file for the complete algorithm.

|prosumer\_id|source\_house\_id|historical\_mean\_kwh|augmented\_first\_year\_mean\_kwh|historical\_p95\_kwh|augmented\_first\_year\_p95\_kwh|historical\_acf1|augmented\_first\_year\_acf1|historical\_acf24|augmented\_first\_year\_acf24|
|-|-|-|-|-|-|-|-|-|-|
|P1|3|0.923452|0.927749|2.39|2.05927|0.437825|0.488776|0.471498|0.569876|
|P2|4|1.21508|1.2227|2.63|2.38|0.717898|0.731313|0.545276|0.572096|
|P3|5|0.752874|0.736347|2.35|1.92855|0.583286|0.60099|0.369203|0.423781|
|P4|6|0.362918|0.363032|0.91|0.832213|0.689121|0.720368|0.45791|0.530816|
|P5|8|0.66633|0.667495|1.57|1.46513|0.597267|0.606271|0.260306|0.312067|
|P6|10|0.642446|0.637646|1.84|1.57072|0.51728|0.580705|0.24697|0.288335|
|P7|11|0.539262|0.546468|1.6|1.45207|0.674715|0.714576|0.455126|0.529518|
|P8|12|0.523643|0.505185|1.1|1.01774|0.741051|0.759189|0.46487|0.52245|
|P9|13|0.900264|0.944893|2.59|2.28652|0.602981|0.645304|0.425565|0.508564|
|P10|14|1.57345|1.53809|3.28|2.83695|0.720805|0.733677|0.453512|0.498988|
