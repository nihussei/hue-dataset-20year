"""Build the bundled 2017--2025 historical Mid-C CAD/kWh baseline.

Sources
-------
* U.S. EIA republished ICE daily wholesale electricity workbooks.
* Bank of Canada Valet daily FXUSDCAD observations.

The resulting CSV contains source rows only; non-trading days are interpolated
inside the RTP module at simulation time.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


EIA = "https://www.eia.gov/electricity/wholesale/xls/archive/ice_electric-{year}final.xlsx"
BOC = ("https://www.bankofcanada.ca/valet/observations/FXUSDCAD/csv"
       "?start_date=2017-01-01&end_date=2025-12-31")


def _download(url: str) -> bytes:
    with urlopen(url, timeout=120) as response:
        return response.read()


def _fx() -> pd.Series:
    text = _download(BOC).decode("utf-8-sig")
    start = text.index('"date","FXUSDCAD"')
    frame = pd.read_csv(io.StringIO(text[start:]))
    frame["date"] = pd.to_datetime(frame["date"])
    frame["FXUSDCAD"] = pd.to_numeric(frame["FXUSDCAD"], errors="coerce")
    series = frame.dropna().set_index("date")["FXUSDCAD"].sort_index()
    daily = series.reindex(pd.date_range(series.index.min(), series.index.max(), freq="D"))
    return daily.interpolate().ffill().bfill()


def build(output: Path) -> pd.DataFrame:
    fx = _fx()
    rows = []
    for year in range(2017, 2026):
        raw = pd.read_excel(io.BytesIO(_download(EIA.format(year=year))))
        hub = raw[raw["Price hub"].astype(str).str.strip().eq("Mid C Peak")].copy()
        hub["date"] = pd.to_datetime(hub["Delivery start date"], errors="coerce").dt.normalize()
        hub["midc_usd_per_mwh"] = pd.to_numeric(
            hub["Wtd avg price $/MWh"], errors="coerce")
        hub = hub.dropna(subset=["date", "midc_usd_per_mwh"])
        hub = hub.groupby("date", as_index=False)["midc_usd_per_mwh"].mean()
        hub["fx_usd_cad"] = hub["date"].map(fx)
        hub["midc_cad_per_kwh"] = hub["midc_usd_per_mwh"] * hub["fx_usd_cad"] / 1000.0
        hub["source"] = "EIA/ICE Mid C Peak + Bank of Canada FXUSDCAD"
        rows.append(hub)
        print(f"Downloaded {year}: {len(hub)} Mid-C observations", flush=True)
    result = pd.concat(rows, ignore_index=True).sort_values("date")
    result = result[result["midc_cad_per_kwh"] > 0].drop_duplicates("date", keep="last")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("data/market/midc_daily_2017_2025.csv"))
    args = parser.parse_args()
    result = build(args.output)
    print(f"Wrote {len(result)} rows to {args.output}")
