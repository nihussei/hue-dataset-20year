from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .data import (
    _causal_forecast,
    _choose_common_houses,
    _read_house,
    _solar_reference,
    find_hue_root,
    simulate_outages,
    validate_long_data,
)
from .market import add_historical_midc_rtp


@dataclass(frozen=True)
class AugmentationConfig:
    """Reproducible assumptions for extending the short HUE record.

    The generator is deliberately statistical, not a claim that 2021--2040
    measurements exist.  It keeps HUE month/day-of-week/hour structure and
    resamples *multivariate* 168-hour residual blocks so cross-house and serial
    dependence are retained.
    """

    start_year: int = 2021
    years: int = 20
    block_hours: int = 168
    residual_weight: float = 0.80
    annual_load_growth: float = 0.007
    annual_pv_degradation: float = 0.005
    community_ar_phi: float = 0.92
    community_ar_sigma: float = 0.025
    seed: int = 42


def _seasonal_template(frame: pd.DataFrame) -> pd.Series:
    key = [frame.index.month, frame.index.dayofweek, frame.index.hour]
    return frame.groupby(key).mean()


def _template_on_index(template: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    keys = pd.MultiIndex.from_arrays([index.month, index.dayofweek, index.hour])
    values = template.reindex(keys).to_numpy(dtype=float)
    if np.isnan(values).any():
        hourly = template.groupby(level=2).mean()
        values[np.isnan(values)] = hourly.reindex(index.hour[np.isnan(values)]).to_numpy()
    return values


def _bootstrap_multivariate_residuals(
    residuals: pd.DataFrame,
    target: pd.DatetimeIndex,
    cfg: AugmentationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    source = residuals.to_numpy(dtype=float)
    source_month = residuals.index.month.to_numpy()
    block = max(24, min(cfg.block_hours, len(source)))
    valid_starts = np.arange(max(len(source) - block + 1, 1))
    pieces: list[np.ndarray] = []
    produced = 0
    while produced < len(target):
        wanted_month = int(target[produced].month)
        candidates = valid_starts[source_month[valid_starts] == wanted_month]
        if not len(candidates):
            candidates = valid_starts
        start = int(rng.choice(candidates))
        take = min(block, len(target) - produced, len(source) - start)
        pieces.append(source[start:start + take])
        produced += take
    return np.vstack(pieces)[:len(target)]


def _acf(values: np.ndarray, lag: int) -> float:
    x = np.asarray(values, dtype=float)
    if len(x) <= lag or np.std(x) < 1e-12:
        return 0.0
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def prepare_augmented_hue_community(
    hue_root: str | Path,
    n_prosumers: int = 10,
    house_ids: Optional[Iterable[int]] = None,
    augmentation: AugmentationConfig | None = None,
    outage_scenario: str = "moderate",
    market_price_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build a 20-year hourly community from HUE without cloning years.

    Each target load is a HUE seasonal template plus a sampled multivariate
    residual block and a small shared AR innovation. PV follows the HUE solar
    shape with household capacity diversity and daily cloud factors. Forecasts
    are causal (lag-24/lag-168) and the final frame is explicitly labelled as
    augmented data.
    """

    cfg = augmentation or AugmentationConfig()
    if cfg.years < 4:
        raise ValueError("At least four augmented years are required for train/validation/test")
    root = find_hue_root(hue_root)
    selected, common_start, common_end = _choose_common_houses(
        root, n_prosumers, house_ids)
    historical_index = pd.date_range(common_start, common_end, freq="h")
    histories = pd.DataFrame({
        house: _read_house(root / f"Residential_{house}.csv").reindex(historical_index)
        for house in selected
    }).interpolate(limit=6).ffill().bfill().clip(lower=0.01)

    templates = {house: _seasonal_template(histories[house]) for house in selected}
    fitted = pd.DataFrame({
        house: _template_on_index(templates[house], historical_index)
        for house in selected
    }, index=historical_index)
    residuals = histories - fitted

    target = pd.date_range(
        f"{cfg.start_year}-01-01 00:00",
        f"{cfg.start_year + cfg.years - 1}-12-31 23:00",
        freq="h",
    )
    rng = np.random.default_rng(cfg.seed)
    sampled = _bootstrap_multivariate_residuals(residuals, target, cfg, rng)
    shared = np.zeros(len(target), dtype=float)
    innovations = rng.normal(0.0, cfg.community_ar_sigma, len(target))
    for i in range(1, len(shared)):
        shared[i] = cfg.community_ar_phi * shared[i - 1] + innovations[i]

    solar = _solar_reference(root)
    solar_keys = pd.MultiIndex.from_arrays([target.month, target.day, target.hour])
    pv_reference = solar.reindex(solar_keys).to_numpy(dtype=float)
    if np.isnan(pv_reference).any():
        fallback = pd.MultiIndex.from_arrays([
            target.month,
            np.where((target.month == 2) & (target.day == 29), 28, target.day),
            target.hour,
        ])
        pv_reference = solar.reindex(fallback).to_numpy(dtype=float)
    pv_reference = np.nan_to_num(pv_reference, nan=0.0)

    day_codes = pd.factorize(target.normalize())[0]
    daily_cloud = np.clip(rng.lognormal(mean=-0.035, sigma=0.20,
                                        size=day_codes.max() + 1), 0.35, 1.20)
    pv_multipliers = np.linspace(0.70, 1.30, len(selected))
    rng.shuffle(pv_multipliers)
    critical_fractions = rng.uniform(0.50, 0.75, len(selected))
    years_since_start = target.year.to_numpy() - cfg.start_year

    rows: list[pd.DataFrame] = []
    mapping_rows: list[dict] = []
    for j, source_house in enumerate(selected):
        expected = _template_on_index(templates[source_house], target)
        house_scale = max(float(histories[source_house].mean()), 0.05)
        load = expected + cfg.residual_weight * sampled[:, j] + shared * house_scale
        load *= (1.0 + cfg.annual_load_growth) ** years_since_start
        load = np.clip(load, 0.01, None)

        pv = (pv_reference * pv_multipliers[j] * daily_cloud[day_codes]
              * (1.0 - cfg.annual_pv_degradation) ** years_since_start)
        pv = np.clip(pv, 0.0, None)
        pid = f"P{j + 1}"
        frame = pd.DataFrame({
            "timestamp": target,
            "prosumer_id": pid,
            "source_house_id": source_house,
            "load_kwh": load,
            "pv_kwh": pv,
            "critical_fraction": critical_fractions[j],
            "is_augmented": 1,
            "augmentation_seed": cfg.seed,
        })
        frame["load_forecast_kwh"] = _causal_forecast(frame["load_kwh"])
        frame["pv_forecast_kwh"] = _causal_forecast(frame["pv_kwh"])
        rows.append(frame)
        mapping_rows.append({
            "prosumer_id": pid,
            "hue_source_house": source_house,
            "pv_capacity_multiplier": pv_multipliers[j],
            "critical_fraction": critical_fractions[j],
        })

    data = pd.concat(rows, ignore_index=True)
    data, outage_events = simulate_outages(data, outage_scenario, cfg.seed)
    data, rtp_diagnostics = add_historical_midc_rtp(
        data, market_price_path=market_price_path, seed=cfg.seed)
    data = validate_long_data(data)

    validation_rows = []
    for j, source_house in enumerate(selected):
        observed = histories[source_house].to_numpy(dtype=float)
        generated = data.loc[data["prosumer_id"] == f"P{j + 1}", "load_kwh"].to_numpy()
        validation_rows.append({
            "prosumer_id": f"P{j + 1}",
            "source_house_id": source_house,
            "historical_mean_kwh": float(np.mean(observed)),
            "augmented_first_year_mean_kwh": float(np.mean(generated[:8760])),
            "historical_p95_kwh": float(np.quantile(observed, 0.95)),
            "augmented_first_year_p95_kwh": float(np.quantile(generated[:8760], 0.95)),
            "historical_acf1": _acf(observed, 1),
            "augmented_first_year_acf1": _acf(generated[:8760], 1),
            "historical_acf24": _acf(observed, 24),
            "augmented_first_year_acf24": _acf(generated[:8760], 24),
        })

    yearly = (data.groupby([data["timestamp"].dt.year.rename("year"), "prosumer_id"])
              .agg(load_kwh=("load_kwh", "sum"), pv_kwh=("pv_kwh", "sum"))
              .reset_index())
    config_table = pd.DataFrame([asdict(cfg)])
    return data, {
        "prosumer_mapping": pd.DataFrame(mapping_rows),
        "augmentation_config": config_table,
        "augmentation_validation": pd.DataFrame(validation_rows),
        "augmentation_yearly_energy": yearly,
        "outage_events": outage_events,
        "rtp_diagnostics": rtp_diagnostics,
    }
