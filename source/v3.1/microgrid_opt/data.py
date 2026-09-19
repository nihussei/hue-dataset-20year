from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import warnings

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"timestamp", "prosumer_id", "load_kwh", "pv_kwh"}


def validate_long_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the canonical hourly, long-format community dataset."""
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="raise")
    out["prosumer_id"] = out["prosumer_id"].astype(str)
    out = out.sort_values(["timestamp", "prosumer_id"]).reset_index(drop=True)

    for col in ("load_kwh", "pv_kwh"):
        out[col] = pd.to_numeric(out[col], errors="raise").clip(lower=0.0)
        if not np.isfinite(out[col]).all():
            raise ValueError(f"Column {col} contains missing or non-finite values")

    for forecast, realised in (("load_forecast_kwh", "load_kwh"),
                               ("pv_forecast_kwh", "pv_kwh")):
        if forecast not in out:
            warnings.warn(f"{forecast} missing; falling back to realised {realised}", RuntimeWarning)
            out[forecast] = out[realised]
        out[forecast] = pd.to_numeric(out[forecast], errors="raise").clip(lower=0.0)
        if not np.isfinite(out[forecast]).all():
            raise ValueError(f"Column {forecast} contains missing or non-finite values")

    if "critical_fraction" not in out:
        out["critical_fraction"] = 0.65
    out["critical_fraction"] = pd.to_numeric(
        out["critical_fraction"], errors="raise").clip(0.0, 1.0)

    if "grid_available" not in out:
        out["grid_available"] = 1
    out["grid_available"] = pd.to_numeric(
        out["grid_available"], errors="raise").astype(int).clip(0, 1)
    if "grid_available_forecast" not in out:
        out["grid_available_forecast"] = out["grid_available"]
    out["grid_available_forecast"] = pd.to_numeric(
        out["grid_available_forecast"], errors="raise").astype(int).clip(0, 1)

    for col in ("rtp_cad_per_kwh", "rtp_forecast_cad_per_kwh"):
        if col not in out:
            out[col] = np.nan
        else:
            out[col] = pd.to_numeric(out[col], errors="raise")
            if (out[col].dropna() < 0).any():
                raise ValueError(f"{col} cannot contain negative prices")

    if out.duplicated(["timestamp", "prosumer_id"]).any():
        raise ValueError("Duplicate timestamp/prosumer rows found")
    expected_ids = frozenset(out["prosumer_id"].unique())
    id_sets = out.groupby("timestamp")["prosumer_id"].agg(lambda x: frozenset(x))
    if not id_sets.map(lambda x: x == expected_ids).all():
        raise ValueError("Every timestamp must contain the identical prosumer set")

    timestamps = pd.DatetimeIndex(out["timestamp"].drop_duplicates().sort_values())
    if len(timestamps) > 1:
        gaps = timestamps.to_series().diff().dropna()
        if not (gaps == pd.Timedelta(hours=1)).all():
            raise ValueError("The simulation input must be a continuous hourly time series")

    for col in ("grid_available", "grid_available_forecast"):
        grid_counts = out.groupby("timestamp")[col].nunique()
        if (grid_counts > 1).any():
            raise ValueError(f"{col} must be identical for all prosumers at each timestamp")

    if out["rtp_cad_per_kwh"].notna().any() and out["rtp_cad_per_kwh"].isna().any():
        warnings.warn("RTP contains gaps; the RTP experiment will be skipped", RuntimeWarning)
    return out


def load_forecast_csv(path: str | Path) -> pd.DataFrame:
    return validate_long_data(pd.read_csv(path))


def find_hue_root(path: str | Path) -> Path:
    """Resolve a HUE root regardless of Kaggle's nesting convention."""
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"HUE input path does not exist: {root}")
    if (root / "Solar.csv").exists() and list(root.glob("Residential_*.csv")):
        return root
    for candidate in root.rglob("Solar.csv"):
        if list(candidate.parent.glob("Residential_*.csv")):
            return candidate.parent
    raise FileNotFoundError(
        f"Could not find Solar.csv and Residential_*.csv below {root}. "
        "Attach the Kaggle HUE dataset first."
    )


def _read_house(path: Path) -> pd.Series:
    raw = pd.read_csv(path, usecols=["date", "hour", "energy_kWh"])
    raw["timestamp"] = (pd.to_datetime(raw["date"], errors="coerce")
                        + pd.to_timedelta(pd.to_numeric(raw["hour"], errors="coerce"), unit="h"))
    raw["energy_kWh"] = pd.to_numeric(raw["energy_kWh"], errors="coerce")
    raw = raw.dropna(subset=["timestamp"]).groupby("timestamp")["energy_kWh"].mean().sort_index()
    idx = pd.date_range(raw.index.min(), raw.index.max(), freq="h")
    series = raw.reindex(idx).clip(lower=0.0)
    series = series.interpolate(limit=6, limit_direction="both")
    if series.isna().any():
        how = pd.Series(series.index.dayofweek * 24 + series.index.hour, index=series.index)
        medians = series.groupby(how).median()
        series = series.fillna(how.map(medians))
    series = series.ffill().bfill()
    series.name = path.stem
    return series


def _choose_common_houses(root: Path, n_prosumers: int,
                          requested: Optional[Iterable[int]] = None) -> tuple[list[int], pd.Timestamp, pd.Timestamp]:
    files = {int(p.stem.split("_")[-1]): p for p in root.glob("Residential_*.csv")}
    if requested is not None:
        chosen = [int(x) for x in requested]
        absent = sorted(set(chosen).difference(files))
        if absent:
            raise ValueError(f"Requested HUE houses are absent: {absent}")
    else:
        coverage = {}
        for house, path in files.items():
            d = pd.read_csv(path, usecols=["date"])
            coverage[house] = (pd.to_datetime(d["date"].iloc[0]), pd.to_datetime(d["date"].iloc[-1]))
        best = None
        for candidate_start in sorted({v[0] for v in coverage.values()}):
            eligible = [(h, end) for h, (start, end) in coverage.items()
                        if start <= candidate_start < end]
            eligible.sort(key=lambda x: x[1], reverse=True)
            if len(eligible) < n_prosumers:
                continue
            selection = eligible[:n_prosumers]
            end = min(v for _, v in selection)
            duration = end - candidate_start
            if best is None or duration > best[0]:
                best = (duration, sorted(h for h, _ in selection), candidate_start, end)
        if best is None:
            raise ValueError(f"No {n_prosumers} HUE profiles share an hourly overlap")
        _, chosen, _, _ = best

    series = {h: _read_house(files[h]) for h in chosen}
    start = max(s.index.min() for s in series.values())
    end = min(s.index.max() for s in series.values())
    if end <= start:
        raise ValueError("Selected HUE houses have no common chronological overlap")
    return chosen, pd.Timestamp(start), pd.Timestamp(end)


def _solar_reference(root: Path) -> pd.Series:
    solar = pd.read_csv(root / "Solar.csv")
    solar["month"] = solar["date"].astype(str).str.extract(r"-(\d{2})-")[0].astype(int)
    solar["day"] = solar["date"].astype(str).str.extract(r"-(\d{2})$")[0].astype(int)
    solar["hour"] = pd.to_numeric(solar["hour"], errors="raise").astype(int)
    ac = pd.to_numeric(solar["ac_output"], errors="coerce").fillna(0.0).clip(lower=0.0)
    # HUE Solar.csv is watt-scale; W / 1000 over one hour is numerically kWh.
    # This fixes the original load-kWh/PV-W mismatch.
    if ac.quantile(0.99) > 50.0:
        ac = ac / 1000.0
    key = pd.MultiIndex.from_arrays([solar["month"], solar["day"], solar["hour"]])
    return pd.Series(ac.to_numpy(), index=key).groupby(level=[0, 1, 2]).mean()


def _causal_forecast(values: pd.Series) -> pd.Series:
    """Day/week seasonal-naive forecast using only information known before t."""
    daily = values.shift(24)
    weekly = values.shift(168)
    forecast = (0.65 * daily + 0.35 * weekly).where(weekly.notna(), daily)
    # Warm-up values use only the expanding history available before the target
    # hour. The first observation is an explicit initial-condition estimate.
    history_mean = values.expanding(min_periods=1).mean().shift(1)
    initial = float(values.iloc[0]) if len(values) else 0.0
    return forecast.fillna(history_mean).fillna(initial)


def simulate_outages(data: pd.DataFrame, scenario: str = "moderate", seed: int = 42
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create reproducible scenario outages, explicitly not historical claims."""
    scenario_parameters = {
        "none": (0.0, 0.0),
        "normal": (3.0, 2.0),
        "moderate": (6.0, 4.0),
        "stress": (12.0, 8.0),
    }
    if scenario not in scenario_parameters:
        raise ValueError(f"Unknown outage scenario {scenario!r}")
    events_per_year, mean_duration = scenario_parameters[scenario]
    out = data.copy()
    timestamps = pd.DatetimeIndex(out["timestamp"].drop_duplicates().sort_values())
    grid = np.ones(len(timestamps), dtype=int)
    rng = np.random.default_rng(seed)
    years = max(len(timestamps) / 8760.0, 1 / 12)
    event_count = int(round(events_per_year * years))
    rows = []
    if event_count:
        weather = (out.groupby("timestamp")["weather"].first().reindex(timestamps)
                   if "weather" in out else pd.Series("", index=timestamps))
        storm = weather.fillna("").str.contains(
            "rain|snow|storm|freez|fog|drizzle", case=False, regex=True).to_numpy(dtype=float)
        winter = np.isin(timestamps.month, [11, 12, 1, 2]).astype(float)
        weights = 1.0 + 2.0 * storm + 0.5 * winter
        if len(weights) > 48:
            weights[:24] = 0.0
            weights[-24:] = 0.0
        if weights.sum() <= 0:
            weights[:] = 1.0
        weights /= weights.sum()
        candidates = np.arange(len(timestamps))
        starts = rng.choice(candidates, size=min(event_count, len(candidates)), replace=False, p=weights)
        starts.sort()
        for event_id, start in enumerate(starts, 1):
            duration = max(1, int(round(rng.lognormal(np.log(max(mean_duration, 1.0)), 0.35))))
            stop = min(start + duration, len(grid))
            grid[start:stop] = 0
            rows.append({
                "event_id": event_id, "scenario": scenario,
                "start": timestamps[start],
                "end_exclusive": timestamps[stop - 1] + pd.Timedelta(hours=1),
                "duration_hours": stop - start,
            })
    mapping = pd.Series(grid, index=timestamps)
    out["grid_available"] = out["timestamp"].map(mapping).astype(int)
    # The generated events represent unplanned outages.  Their future start and
    # duration are therefore not disclosed to the controllers.  Current grid
    # status is observed directly by the environment; the day-ahead availability
    # forecast remains "available" and may be wrong during an event.
    out["grid_available_forecast"] = 1
    return out, pd.DataFrame(rows, columns=[
        "event_id", "scenario", "start", "end_exclusive", "duration_hours"])


def add_dynamic_rtp(data: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add grid-responsive realised and forecast RTP without future leakage."""
    out = data.copy()
    agg = out.groupby("timestamp", sort=True).agg(
        load=("load_kwh", "sum"), pv=("pv_kwh", "sum"),
        load_forecast=("load_forecast_kwh", "sum"),
        pv_forecast=("pv_forecast_kwh", "sum"),
        grid=("grid_available", "first"),
        grid_forecast=("grid_available_forecast", "first"),
    )
    net = agg["load"] - agg["pv"]
    forecast_net = agg["load_forecast"] - agg["pv_forecast"]
    calibration = net.iloc[:max(int(0.40 * len(net)), 168)]
    q10, q90 = calibration.quantile([0.10, 0.90])
    span = max(float(q90 - q10), 1e-6)
    demand_stress = np.clip((net - q10) / span, 0.0, 1.5)
    forecast_stress = np.clip((forecast_net - q10) / span, 0.0, 1.5)
    surplus_scale = max(float(np.maximum(-calibration, 0.0).quantile(0.90)), 1e-6)
    renewable_relief = np.clip(np.maximum(-net, 0.0) / surplus_scale, 0.0, 1.0)
    renewable_relief_f = np.clip(np.maximum(-forecast_net, 0.0) / surplus_scale, 0.0, 1.0)

    rng = np.random.default_rng(seed)
    shock = np.zeros(len(agg), dtype=float)
    innovations = rng.normal(0.0, 0.008, len(agg))
    for i in range(1, len(shock)):
        shock[i] = 0.85 * shock[i - 1] + innovations[i]
    shock_forecast = np.r_[0.0, shock[:-1]]
    scarcity = 1.0 - agg["grid"].to_numpy(dtype=float)
    scarcity_f = 1.0 - agg["grid_forecast"].to_numpy(dtype=float)
    realised = np.clip(
        0.065 + 0.19 * demand_stress.to_numpy() - 0.035 * renewable_relief.to_numpy()
        + 0.12 * scarcity + shock, 0.03, 0.60)
    forecast = np.clip(
        0.065 + 0.19 * forecast_stress.to_numpy() - 0.035 * renewable_relief_f.to_numpy()
        + 0.12 * scarcity_f + shock_forecast, 0.03, 0.60)
    realised_map = pd.Series(realised, index=agg.index)
    forecast_map = pd.Series(forecast, index=agg.index)
    out["rtp_cad_per_kwh"] = out["timestamp"].map(realised_map)
    out["rtp_forecast_cad_per_kwh"] = out["timestamp"].map(forecast_map)
    diagnostics = agg.assign(
        demand_stress=demand_stress, renewable_relief=renewable_relief,
        rtp_cad_per_kwh=realised, rtp_forecast_cad_per_kwh=forecast,
    ).reset_index()
    return out, diagnostics


def prepare_hue_community(
    hue_root: str | Path,
    n_prosumers: int = 10,
    house_ids: Optional[Iterable[int]] = None,
    seed: int = 42,
    outage_scenario: str = "moderate",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build a simultaneous, hourly community directly from the HUE files."""
    root = find_hue_root(hue_root)
    chosen, overlap_start, overlap_end = _choose_common_houses(root, n_prosumers, house_ids)
    house_series = {h: _read_house(root / f"Residential_{h}.csv") for h in chosen}
    timestamps = pd.date_range(overlap_start, overlap_end, freq="h")
    solar_lookup = _solar_reference(root)
    solar_key = pd.MultiIndex.from_arrays([timestamps.month, timestamps.day, timestamps.hour])
    pv_reference = solar_lookup.reindex(solar_key).to_numpy(dtype=float)
    if np.isnan(pv_reference).any():
        fallback_key = pd.MultiIndex.from_arrays([
            timestamps.month,
            np.where((timestamps.month == 2) & (timestamps.day == 29), 28, timestamps.day),
            timestamps.hour,
        ])
        pv_reference = solar_lookup.reindex(fallback_key).to_numpy(dtype=float)

    weather_path = root / "Weather_YVR.csv"
    weather_map = None
    if weather_path.exists():
        weather = pd.read_csv(weather_path)
        weather_hour = pd.to_numeric(weather["hour"], errors="coerce")
        # HUE weather files use 1..24 for hour-ending labels; residential and
        # simulation timestamps use 0..23 hour-start labels.
        if weather_hour.min() >= 1 and weather_hour.max() <= 24:
            weather_hour = weather_hour - 1
        weather["timestamp"] = (pd.to_datetime(weather["date"], errors="coerce")
                                + pd.to_timedelta(weather_hour, unit="h"))
        weather_map = weather.dropna(subset=["timestamp"]).groupby("timestamp").first()

    rng = np.random.default_rng(seed)
    multipliers = np.linspace(0.70, 1.30, len(chosen))
    rng.shuffle(multipliers)
    critical = rng.uniform(0.50, 0.75, len(chosen))
    rows, mapping_rows = [], []
    for i, (source_house, multiplier) in enumerate(zip(chosen, multipliers), 1):
        load = house_series[source_house].reindex(timestamps)
        load = load.interpolate(limit=6).ffill().bfill().clip(lower=0.01)
        pv = pd.Series(pv_reference * multiplier, index=timestamps).clip(lower=0.0)
        frame = pd.DataFrame({
            "timestamp": timestamps, "prosumer_id": f"P{i}",
            "source_house_id": source_house, "load_kwh": load.to_numpy(),
            "pv_kwh": pv.to_numpy(),
            "load_forecast_kwh": _causal_forecast(load).to_numpy(),
            "pv_forecast_kwh": _causal_forecast(pv).to_numpy(),
            "critical_fraction": critical[i - 1],
        })
        if weather_map is not None:
            for col in ("temperature", "humidity", "pressure", "weather"):
                if col in weather_map:
                    frame[col] = frame["timestamp"].map(weather_map[col])
        rows.append(frame)
        mapping_rows.append({
            "prosumer_id": f"P{i}", "hue_source_house": source_house,
            "pv_capacity_multiplier": multiplier,
            "critical_fraction": critical[i - 1],
        })
    data = pd.concat(rows, ignore_index=True)
    data = data[data["timestamp"] >= timestamps[168]].copy()
    data, outage_events = simulate_outages(data, outage_scenario, seed)
    data, rtp_diagnostics = add_dynamic_rtp(data, seed)
    data = validate_long_data(data)
    return data, {
        "prosumer_mapping": pd.DataFrame(mapping_rows),
        "outage_events": outage_events,
        "rtp_diagnostics": rtp_diagnostics,
    }


def chronological_split(data: pd.DataFrame, validation_days: int = 180,
                        test_days: int = 365) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split, preferring a complete calendar-year test set."""
    timestamps = pd.DatetimeIndex(data["timestamp"].drop_duplicates().sort_values())
    test_ts = None
    if test_days == 365:
        # Monthly, seasonal and annual client figures are easier to interpret
        # when the held-out year is Jan-Dec instead of two partial calendar years.
        for year in sorted(timestamps.year.unique(), reverse=True):
            candidate = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
            if len(candidate) != 8760 or not candidate.isin(timestamps).all():
                continue
            first = int(timestamps.get_indexer([candidate[0]])[0])
            if first >= (validation_days + 180) * 24:
                test_ts = candidate
                break

    if test_ts is not None:
        test_start = int(timestamps.get_indexer([test_ts[0]])[0])
        val_start = test_start - validation_days * 24
        train_ts = timestamps[:val_start]
        val_ts = timestamps[val_start:test_start]
    elif len(timestamps) >= (validation_days + test_days + 180) * 24:
        test_start = len(timestamps) - test_days * 24
        val_start = test_start - validation_days * 24
        train_ts = timestamps[:val_start]
        val_ts = timestamps[val_start:test_start]
        test_ts = timestamps[test_start:]
    else:
        val_start = max(int(0.60 * len(timestamps)), 1)
        test_start = max(int(0.80 * len(timestamps)), val_start + 1)
        warnings.warn(
            "HUE overlap is too short for fixed-day split; using 60/20/20",
            RuntimeWarning)
        train_ts = timestamps[:val_start]
        val_ts = timestamps[val_start:test_start]
        test_ts = timestamps[test_start:]
    train = data[data["timestamp"].isin(train_ts)].copy()
    validation = data[data["timestamp"].isin(val_ts)].copy()
    test = data[data["timestamp"].isin(test_ts)].copy()
    manifest = pd.DataFrame([
        {"split": "train", "start": train_ts.min(), "end": train_ts.max(), "hours": len(train_ts)},
        {"split": "validation", "start": val_ts.min(), "end": val_ts.max(), "hours": len(val_ts)},
        {"split": "test", "start": test_ts.min(), "end": test_ts.max(), "hours": len(test_ts)},
    ])
    return train, validation, test, manifest


def chronological_year_split(
    data: pd.DataFrame,
    validation_years: int = 2,
    test_years: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reserve complete, non-overlapping calendar years for validation/test."""
    timestamps = pd.DatetimeIndex(data["timestamp"].drop_duplicates().sort_values())
    years = sorted(int(y) for y in timestamps.year.unique())
    if len(years) <= validation_years + test_years:
        raise ValueError("Not enough complete years for the requested chronological split")
    test_set = set(years[-test_years:])
    validation_set = set(years[-(validation_years + test_years):-test_years])
    train_set = set(years).difference(test_set | validation_set)
    train = data[data["timestamp"].dt.year.isin(train_set)].copy()
    validation = data[data["timestamp"].dt.year.isin(validation_set)].copy()
    test = data[data["timestamp"].dt.year.isin(test_set)].copy()
    frames = (("train", train), ("validation", validation), ("test", test))
    manifest = pd.DataFrame([{
        "split": name,
        "start": frame["timestamp"].min(),
        "end": frame["timestamp"].max(),
        "hours": frame["timestamp"].nunique(),
        "calendar_years": ",".join(map(str, sorted(frame["timestamp"].dt.year.unique()))),
    } for name, frame in frames])
    if not (train["timestamp"].max() < validation["timestamp"].min()
            < test["timestamp"].min()):
        raise AssertionError("Chronological split overlap detected")
    return train, validation, test, manifest


def generate_synthetic_community(
    n_prosumers: int = 10, days: int = 120, seed: int = 42,
    weather_csv: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Deterministic unit-test data; never presented as final client results."""
    rng = np.random.default_rng(seed)
    n = days * 24
    timestamps = pd.date_range("2025-01-01", periods=n, freq="h")
    hour = timestamps.hour.to_numpy()
    day = np.arange(n) / 24.0
    if weather_csv and Path(weather_csv).exists():
        weather = pd.read_csv(weather_csv, usecols=["temperature"]).iloc[:n]
        temperature = np.resize(weather["temperature"].to_numpy(), n)
    else:
        temperature = 8 + 8 * np.sin(2 * np.pi * (day - 80) / 365)
    solar_shape = np.maximum(np.sin(np.pi * (hour - 6) / 12), 0.0)
    seasonal = np.clip(0.65 + 0.015 * temperature, 0.35, 1.15)
    rows = []
    for p in range(1, n_prosumers + 1):
        load_scale, pv_scale = rng.uniform(0.65, 1.55), rng.uniform(0.6, 1.8)
        morning = np.exp(-0.5 * ((hour - 7) / 2.2) ** 2)
        evening = np.exp(-0.5 * ((hour - 19) / 2.8) ** 2)
        load = np.maximum(load_scale * (0.35 + 0.45 * morning + 0.9 * evening)
                          + rng.normal(0, 0.08, n), 0.05)
        pv = np.maximum(pv_scale * 2.2 * solar_shape * seasonal
                        * np.clip(rng.normal(0.85, 0.18, n), 0.15, 1.1), 0.0)
        load_s, pv_s = pd.Series(load), pd.Series(pv)
        rows.append(pd.DataFrame({
            "timestamp": timestamps, "prosumer_id": f"P{p}",
            "load_kwh": load, "pv_kwh": pv,
            "load_forecast_kwh": _causal_forecast(load_s).bfill().to_numpy(),
            "pv_forecast_kwh": _causal_forecast(pv_s).bfill().to_numpy(),
            "critical_fraction": rng.uniform(0.50, 0.75),
        }))
    out = pd.concat(rows, ignore_index=True)
    out, _ = simulate_outages(out, "moderate", seed)
    out, _ = add_dynamic_rtp(out, seed)
    return validate_long_data(out)
