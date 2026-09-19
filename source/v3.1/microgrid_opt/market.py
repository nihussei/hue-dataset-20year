from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MARKET_FILE = Path(__file__).resolve().parents[1] / "data" / "market" / "midc_daily_2017_2025.csv"


def load_midc_daily(path: str | Path | None = None) -> pd.DataFrame:
    source = Path(path) if path else DEFAULT_MARKET_FILE
    if not source.exists():
        raise FileNotFoundError(
            f"Historical Mid-C baseline missing: {source}. "
            "Run scripts/build_midc_price_baseline.py first."
        )
    data = pd.read_csv(source, parse_dates=["date"])
    needed = {"date", "midc_cad_per_kwh"}
    if not needed.issubset(data):
        raise ValueError(f"Market file must contain {sorted(needed)}")
    data = data.dropna(subset=["date", "midc_cad_per_kwh"]).sort_values("date")
    if data.empty or (data["midc_cad_per_kwh"] <= 0).any():
        raise ValueError("Historical market baseline is empty or non-positive")
    return data


def _daily_anchor(timestamps: pd.DatetimeIndex, market: pd.DataFrame,
                  seed: int) -> np.ndarray:
    series = market.set_index("date")["midc_cad_per_kwh"].sort_index()
    calendar = series.reindex(pd.date_range(series.index.min(), series.index.max(), freq="D"))
    calendar = calendar.interpolate().ffill().bfill()
    lookup = calendar.groupby([calendar.index.month, calendar.index.day]).mean()
    rng = np.random.default_rng(seed + 101)
    years = sorted(set(timestamps.year))
    year_factor = {year: (1.015 ** max(year - int(market["date"].dt.year.max()), 0))
                   * float(rng.lognormal(0.0, 0.035)) for year in years}
    anchors = []
    for ts in timestamps.normalize():
        if ts in calendar.index:
            anchors.append(float(calendar.loc[ts]))
            continue
        day = 28 if ts.month == 2 and ts.day == 29 else ts.day
        anchors.append(float(lookup.loc[(ts.month, day)]) * year_factor[ts.year])
    return np.asarray(anchors)


def add_historical_midc_rtp(data: pd.DataFrame, market_price_path: str | Path | None = None,
                            seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add an hourly retail RTP signal anchored to historical Mid-C prices.

    EIA/ICE Mid-C is daily wholesale data.  The hourly component is therefore
    explicitly modeled from time-of-day, community net demand, renewable
    surplus and grid scarcity.  Realized settlement and causal forecast prices
    are kept separate to prevent future leakage.
    """

    out = data.copy()
    market = load_midc_daily(market_price_path)
    agg = out.groupby("timestamp", sort=True).agg(
        load=("load_kwh", "sum"), pv=("pv_kwh", "sum"),
        load_forecast=("load_forecast_kwh", "sum"),
        pv_forecast=("pv_forecast_kwh", "sum"),
        grid=("grid_available", "first"),
        grid_forecast=("grid_available_forecast", "first"),
    )
    timestamps = pd.DatetimeIndex(agg.index)
    baseline = _daily_anchor(timestamps, market, seed)
    net = agg["load"].to_numpy() - agg["pv"].to_numpy()
    forecast_net = agg["load_forecast"].to_numpy() - agg["pv_forecast"].to_numpy()
    reference = net[:max(min(len(net), 3 * 8760), 168)]
    q10, q90 = np.quantile(reference, [0.10, 0.90])
    span = max(float(q90 - q10), 1e-6)
    stress = np.clip((net - q10) / span, 0.0, 1.5)
    stress_f = np.clip((forecast_net - q10) / span, 0.0, 1.5)
    surplus_scale = max(float(np.quantile(np.maximum(-reference, 0.0), 0.90)), 1e-6)
    relief = np.clip(np.maximum(-net, 0.0) / surplus_scale, 0.0, 1.0)
    relief_f = np.clip(np.maximum(-forecast_net, 0.0) / surplus_scale, 0.0, 1.0)
    hour = timestamps.hour.to_numpy()
    intraday = 0.75 + 0.20 * np.exp(-0.5 * ((hour - 8) / 2.8) ** 2) \
        + 0.55 * np.exp(-0.5 * ((hour - 18) / 3.0) ** 2)
    rng = np.random.default_rng(seed + 211)
    shock = np.zeros(len(agg))
    innovations = rng.normal(0.0, 0.0045, len(agg))
    for i in range(1, len(shock)):
        shock[i] = 0.86 * shock[i - 1] + innovations[i]
    shock_forecast = np.r_[0.0, shock[:-1]]
    retail_adder = 0.055
    realized = np.clip(
        retail_adder + baseline * intraday + 0.075 * stress - 0.025 * relief
        + 0.20 * (1.0 - agg["grid"].to_numpy()) + shock,
        0.02, 0.80,
    )
    forecast = np.clip(
        retail_adder + baseline * intraday + 0.075 * stress_f - 0.025 * relief_f
        + 0.20 * (1.0 - agg["grid_forecast"].to_numpy()) + shock_forecast,
        0.02, 0.80,
    )
    out["rtp_cad_per_kwh"] = out["timestamp"].map(pd.Series(realized, index=agg.index))
    out["rtp_forecast_cad_per_kwh"] = out["timestamp"].map(pd.Series(forecast, index=agg.index))
    diagnostics = agg.assign(
        historical_midc_anchor_cad_per_kwh=baseline,
        intraday_multiplier=intraday,
        demand_stress=stress,
        renewable_relief=relief,
        rtp_cad_per_kwh=realized,
        rtp_forecast_cad_per_kwh=forecast,
    ).reset_index()
    return out, diagnostics
