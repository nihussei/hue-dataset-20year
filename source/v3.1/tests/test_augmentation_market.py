import numpy as np
import pandas as pd

from microgrid_opt.augmentation import AugmentationConfig, _bootstrap_multivariate_residuals
from microgrid_opt.data import chronological_year_split
from microgrid_opt.market import add_historical_midc_rtp, load_midc_daily


def test_bundled_market_baseline_is_real_and_valid():
    data = load_midc_daily()
    assert len(data) > 2000
    assert data["date"].dt.year.min() == 2017
    assert data["date"].dt.year.max() == 2025
    # Wholesale scarcity events can legitimately exceed CAD 1/kWh.
    assert data["midc_cad_per_kwh"].between(0, 5).all()


def test_multivariate_block_bootstrap_is_deterministic_and_keeps_shape():
    index = pd.date_range("2017-01-01", periods=24 * 365, freq="h")
    residuals = pd.DataFrame({"P1": np.sin(np.arange(len(index)) / 8),
                              "P2": np.sin(np.arange(len(index)) / 8) * .7}, index=index)
    target = pd.date_range("2021-01-01", periods=1000, freq="h")
    cfg = AugmentationConfig(years=4, seed=11)
    a = _bootstrap_multivariate_residuals(residuals, target, cfg, np.random.default_rng(11))
    b = _bootstrap_multivariate_residuals(residuals, target, cfg, np.random.default_rng(11))
    assert a.shape == (1000, 2)
    assert np.array_equal(a, b)
    assert np.corrcoef(a.T)[0, 1] > .99


def test_historical_rtp_is_bounded_and_forecast_is_separate():
    ts = pd.date_range("2039-01-01", periods=400, freq="h")
    load = 1.0 + .4 * np.sin(np.arange(len(ts)) * 2 * np.pi / 24)
    frame = pd.DataFrame({
        "timestamp": ts, "prosumer_id": "P1", "load_kwh": load, "pv_kwh": .2,
        "load_forecast_kwh": np.r_[load[0], load[:-1]], "pv_forecast_kwh": .2,
        "grid_available": 1, "grid_available_forecast": 1,
    })
    priced, diagnostic = add_historical_midc_rtp(frame, seed=7)
    assert priced["rtp_cad_per_kwh"].between(.02, .80).all()
    assert priced["rtp_forecast_cad_per_kwh"].between(.02, .80).all()
    assert priced["rtp_cad_per_kwh"].nunique() > 30
    assert "historical_midc_anchor_cad_per_kwh" in diagnostic


def test_whole_year_split_has_no_overlap():
    ts = pd.date_range("2021-01-01", "2026-12-31 23:00", freq="h")
    data = pd.DataFrame({"timestamp": np.repeat(ts, 2),
                         "prosumer_id": np.tile(["P1", "P2"], len(ts))})
    train, validation, test, manifest = chronological_year_split(data, 2, 1)
    assert train["timestamp"].max() < validation["timestamp"].min() < test["timestamp"].min()
    assert set(validation["timestamp"].dt.year) == {2024, 2025}
    assert set(test["timestamp"].dt.year) == {2026}
    assert manifest.loc[manifest["split"] == "test", "hours"].iat[0] == 8760
