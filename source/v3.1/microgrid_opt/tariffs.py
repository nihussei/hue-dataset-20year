from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .config import TariffConfig


def tariff_prices(
    timestamps: pd.DatetimeIndex,
    tariff: str,
    cfg: TariffConfig,
    imports_kwh: np.ndarray | None = None,
    rtp: np.ndarray | None = None,
) -> np.ndarray:
    tariff = tariff.lower()
    n = len(timestamps)
    if tariff == "flat":
        return np.full(n, cfg.flat_cad_per_kwh)
    if tariff == "rtp":
        if rtp is None or np.isnan(rtp).any():
            raise ValueError("RTP selected but rtp_cad_per_kwh is missing")
        return np.asarray(rtp, dtype=float)
    if tariff == "tou":
        # The study treats ToU as one of four mutually-exclusive pricing cases.
        # Use the BC Hydro flat energy charge as the base plus the official
        # +/- 5 cents/kWh TOD differential.
        prices = np.full(n, cfg.flat_cad_per_kwh)
        hours = timestamps.hour.to_numpy()
        prices[np.isin(hours, cfg.overnight_hours)] -= cfg.tou_discount_cad_per_kwh
        prices[np.isin(hours, cfg.on_peak_hours)] += cfg.tou_surcharge_cad_per_kwh
        return np.maximum(prices, 0.0)
    if tariff == "tiered":
        if imports_kwh is None:
            raise ValueError("Tiered pricing requires imports_kwh")
        imports = np.asarray(imports_kwh, dtype=float)
        prices = np.empty(n, dtype=float)
        periods = pd.PeriodIndex(timestamps, freq="M")
        for period in periods.unique():
            idx = np.flatnonzero(periods == period)
            threshold = cfg.tier1_daily_kwh * period.days_in_month
            cumulative_before = np.r_[0.0, np.cumsum(imports[idx][:-1])]
            prices[idx] = np.where(cumulative_before < threshold, cfg.tier1_cad_per_kwh, cfg.tier2_cad_per_kwh)
        return prices
    raise ValueError(f"Unknown tariff: {tariff}")


def tariff_basic_charge_per_day(tariff: str, cfg: TariffConfig) -> float:
    tariff = tariff.lower()
    if tariff == "tiered":
        return cfg.tiered_basic_charge_cad_per_day
    if tariff in ("flat", "tou"):
        return cfg.flat_basic_charge_cad_per_day
    # RTP is a study input rather than a current BC Hydro residential schedule;
    # no fixed charge is invented unless the client supplies one.
    return 0.0


def select_best_fixed_tariff(
    timestamps: pd.DatetimeIndex,
    imports_kwh: np.ndarray,
    cfg: TariffConfig,
    rtp: np.ndarray | None = None,
) -> tuple[str, dict[str, float]]:
    costs: dict[str, float] = {}
    n_days = max(len(pd.DatetimeIndex(timestamps).normalize().unique()), 1)
    for name in ("flat", "tou", "tiered", "rtp"):
        try:
            price = tariff_prices(timestamps, name, cfg, imports_kwh, rtp)
        except ValueError:
            continue
        costs[name] = float(np.sum(imports_kwh * price) + n_days * tariff_basic_charge_per_day(name, cfg))
    return min(costs, key=costs.get), costs


def fit_best_tariff_assignments(train_data: pd.DataFrame, cfg: TariffConfig
                                ) -> tuple[dict[str, str], pd.DataFrame]:
    """Choose each prosumer's fixed best tariff using training data only."""
    d = train_data.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    assignments: dict[str, str] = {}
    rows = []
    for pid, group in d.sort_values("timestamp").groupby("prosumer_id", sort=True):
        timestamps = pd.DatetimeIndex(group["timestamp"])
        imports = np.maximum(group["load_kwh"].to_numpy()
                             - group["pv_kwh"].to_numpy(), 0.0)
        rtp = group["rtp_cad_per_kwh"].to_numpy() if "rtp_cad_per_kwh" in group else None
        selected, costs = select_best_fixed_tariff(timestamps, imports, cfg, rtp)
        assignments[str(pid)] = selected
        rows.append({
            "prosumer_id": str(pid), "selected_tariff": selected,
            **{f"training_cost_{name}_cad": value for name, value in costs.items()},
        })
    return assignments, pd.DataFrame(rows)



def fit_balanced_tariff_assignments(train_data: pd.DataFrame, cfg: TariffConfig
                                    ) -> tuple[dict[str, str], pd.DataFrame]:
    """Assign one fixed tariff per prosumer with balanced tariff representation.

    Scenario 2 requires each prosumer to keep a single tariff for the full horizon,
    while remaining genuinely distinct from the four homogeneous Scenario-1 cases.
    Assignment is fitted on training data only.  For N households and four tariffs,
    tariff counts differ by at most one (for N=10: 3,3,2,2 in the minimum-cost
    arrangement).  Among every feasible placement of the larger quotas, the global
    minimum training-cost assignment is selected using the Hungarian algorithm.
    """
    d = train_data.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    tariffs = ("flat", "tou", "tiered", "rtp")
    pids = sorted(map(str, d["prosumer_id"].unique()))
    rows = []
    cost_matrix = np.zeros((len(pids), len(tariffs)), dtype=float)
    for i, pid in enumerate(pids):
        group = d[d["prosumer_id"].astype(str).eq(pid)].sort_values("timestamp")
        timestamps = pd.DatetimeIndex(group["timestamp"])
        imports = np.maximum(group["load_kwh"].to_numpy() - group["pv_kwh"].to_numpy(), 0.0)
        rtp = group["rtp_cad_per_kwh"].to_numpy() if "rtp_cad_per_kwh" in group else None
        _, costs = select_best_fixed_tariff(timestamps, imports, cfg, rtp)
        for j, name in enumerate(tariffs):
            cost_matrix[i, j] = costs[name]
        rows.append({"prosumer_id": pid, **{f"training_cost_{name}_cad": costs[name] for name in tariffs}})

    n = len(pids)
    q, r = divmod(n, len(tariffs))
    # Choose which r tariffs receive q+1 slots; optimize over all combinations.
    import itertools
    best = None
    for larger in itertools.combinations(range(len(tariffs)), r):
        quotas = [q + (j in larger) for j in range(len(tariffs))]
        slots = [j for j, count in enumerate(quotas) for _ in range(count)]
        expanded = cost_matrix[:, slots]
        rr, cc = linear_sum_assignment(expanded)
        total = float(expanded[rr, cc].sum())
        if best is None or total < best[0]:
            best = (total, quotas, slots, rr, cc)
    assert best is not None
    total, quotas, slots, rr, cc = best
    assignments = {pids[int(i)]: tariffs[slots[int(j)]] for i, j in zip(rr, cc)}
    out = pd.DataFrame(rows)
    out["selected_tariff"] = out["prosumer_id"].map(assignments)
    out["balanced_assignment_training_cost_cad"] = out.apply(
        lambda x: x[f"training_cost_{x['selected_tariff']}_cad"], axis=1)
    out["scenario"] = "balanced_fixed_training_only"
    return assignments, out[["prosumer_id", "selected_tariff",
                             "training_cost_flat_cad", "training_cost_tou_cad",
                             "training_cost_tiered_cad", "training_cost_rtp_cad",
                             "balanced_assignment_training_cost_cad", "scenario"]]
