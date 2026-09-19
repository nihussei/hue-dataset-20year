from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .config import BatteryConfig, ExperimentConfig
from .equity import atkinson_index, ede, normalize_row
from .tariffs import select_best_fixed_tariff, tariff_basic_charge_per_day, tariff_prices


AllocationMethod = Literal["proportional", "priority", "contribution"]


@dataclass
class BatteryState:
    energy_kwh: float
    soh: float = 1.0
    cumulative_fade: float = 0.0


class Battery:
    def __init__(self, cfg: BatteryConfig):
        self.cfg = cfg
        self.state = BatteryState(cfg.initial_soc * cfg.capacity_kwh)

    @property
    def usable_capacity(self) -> float:
        return self.cfg.capacity_kwh * self.state.soh

    @property
    def soc(self) -> float:
        return self.state.energy_kwh / max(self.usable_capacity, 1e-9)

    def clone(self) -> "Battery":
        other = Battery(self.cfg)
        other.state = BatteryState(**vars(self.state))
        return other

    def dispatch(self, requested_kw: float, surplus_kwh: float, deficit_kwh: float, dt_h: float = 1.0):
        cfg = self.cfg
        cap = self.usable_capacity
        e_min, e_max = cfg.min_soc * cap, cfg.max_soc * cap
        charge_input = discharge_output = 0.0
        if requested_kw < 0:
            max_input_by_headroom = max(e_max - self.state.energy_kwh, 0.0) / cfg.charge_efficiency
            charge_input = min(-requested_kw * dt_h, cfg.max_charge_kw * dt_h, surplus_kwh, max_input_by_headroom)
            self.state.energy_kwh += charge_input * cfg.charge_efficiency
        elif requested_kw > 0:
            max_output_by_energy = max(self.state.energy_kwh - e_min, 0.0) * cfg.discharge_efficiency
            discharge_output = min(requested_kw * dt_h, cfg.max_discharge_kw * dt_h, deficit_kwh, max_output_by_energy)
            self.state.energy_kwh -= discharge_output / cfg.discharge_efficiency

        throughput = charge_input * cfg.charge_efficiency + discharge_output / cfg.discharge_efficiency
        cycle_fade = cfg.end_of_life_fade * throughput / (2 * cfg.capacity_kwh * cfg.cycle_life)
        calendar_fade = cfg.calendar_fade_per_year * dt_h / 8760.0
        incremental_fade = cycle_fade + calendar_fade
        self.state.cumulative_fade += incremental_fade
        self.state.soh = max(1.0 - self.state.cumulative_fade, 1.0 - cfg.end_of_life_fade)
        self.state.energy_kwh = np.clip(self.state.energy_kwh, cfg.min_soc * self.usable_capacity, cfg.max_soc * self.usable_capacity)
        cycle_degradation_cost = cycle_fade / cfg.end_of_life_fade * cfg.replacement_cost_cad
        calendar_degradation_cost = calendar_fade / cfg.end_of_life_fade * cfg.replacement_cost_cad
        degradation_cost = cycle_degradation_cost + calendar_degradation_cost
        return (charge_input, discharge_output, degradation_cost,
                cycle_degradation_cost, calendar_degradation_cost)


def allocate(values: np.ndarray, available: float, method: AllocationMethod, weights=None, credits=None):
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    if available <= 0 or values.sum() <= 0:
        return np.zeros_like(values)
    if method == "priority":
        score = values * np.asarray(weights if weights is not None else np.ones_like(values))
    elif method == "contribution":
        score = values * np.asarray(credits if credits is not None else np.ones_like(values))
    else:
        score = values
    if score.sum() <= 0:
        score = values
    tentative = available * score / score.sum()
    allocation = np.minimum(tentative, values)
    remaining = available - allocation.sum()
    # Water-fill remaining energy without depending on prosumer ordering.
    for _ in range(len(values)):
        unmet = np.maximum(values - allocation, 0.0)
        if remaining <= 1e-10 or unmet.sum() <= 1e-10:
            break
        add = np.minimum(remaining * unmet / unmet.sum(), unmet)
        allocation += add
        remaining -= add.sum()
    return allocation


class MicrogridEnvironment:
    """Hourly shared-BESS environment; positive action discharges, negative charges."""
    def __init__(self, data: pd.DataFrame, cfg: ExperimentConfig, tariff="tou",
                 fixed_tariffs: dict[str, str] | None = None):
        self.cfg = cfg
        self.tariff_name = tariff
        self.timestamps = pd.DatetimeIndex(data["timestamp"].drop_duplicates().sort_values())
        self.ids = sorted(data["prosumer_id"].unique())
        self.np = len(self.ids)
        idx = pd.MultiIndex.from_product([self.timestamps, self.ids], names=["timestamp", "prosumer_id"])
        d = data.set_index(["timestamp", "prosumer_id"]).reindex(idx)
        # Realised values are used for physical execution/evaluation. Forecast values
        # are kept separately and are used by MPC/D3QN switching look-ahead.
        self.load = d["load_kwh"].to_numpy().reshape(len(self.timestamps), self.np)
        self.pv = d["pv_kwh"].to_numpy().reshape(len(self.timestamps), self.np)
        self.load_forecast = d["load_forecast_kwh"].to_numpy().reshape(len(self.timestamps), self.np)
        self.pv_forecast = d["pv_forecast_kwh"].to_numpy().reshape(len(self.timestamps), self.np)
        self.critical_fraction = d["critical_fraction"].to_numpy().reshape(len(self.timestamps), self.np)
        self.grid = d["grid_available"].to_numpy().reshape(len(self.timestamps), self.np).min(axis=1).astype(int)
        grid_forecast = (d["grid_available_forecast"]
                         if "grid_available_forecast" in d.columns
                         else d["grid_available"])
        self.grid_forecast = grid_forecast.to_numpy().reshape(
            len(self.timestamps), self.np).min(axis=1).astype(int)
        self.rtp = d["rtp_cad_per_kwh"].to_numpy().reshape(len(self.timestamps), self.np)[:, 0]
        self.rtp_forecast = d["rtp_forecast_cad_per_kwh"].to_numpy().reshape(len(self.timestamps), self.np)[:, 0]
        self.priority = np.linspace(1.0, 2.0, self.np)
        baseline_imports = np.maximum(self.load - self.pv, 0.0)
        self.best_tariffs = {}
        self.price_matrix = np.zeros_like(self.load)
        for i, pid in enumerate(self.ids):
            if tariff == "balanced":
                if fixed_tariffs is not None and pid in fixed_tariffs:
                    selected = fixed_tariffs[pid]
                else:
                    selected, _ = select_best_fixed_tariff(
                        self.timestamps, baseline_imports[:, i], cfg.tariff, self.rtp)
                self.best_tariffs[pid] = selected
            else:
                selected = tariff
                self.best_tariffs[pid] = tariff
            self.price_matrix[:, i] = tariff_prices(
                self.timestamps, selected, cfg.tariff, baseline_imports[:, i], self.rtp)
        # Build the forecast-only decision price matrix once. Policy switching
        # creates thousands of short rollouts; rebuilding an entire T x N price
        # array inside every candidate rollout is both unnecessary and O(T^2).
        forecast_imports = np.maximum(self.load_forecast - self.pv_forecast, 0.0)
        self.forecast_price_matrix = np.zeros_like(self.load_forecast)
        for i, pid in enumerate(self.ids):
            selected = self.best_tariffs[pid]
            self.forecast_price_matrix[:, i] = tariff_prices(
                self.timestamps, selected, cfg.tariff,
                forecast_imports[:, i], self.rtp_forecast)
        self.reset()

    def reset(self):
        self.t = 0
        self.battery = Battery(self.cfg.battery)
        self.credits = np.ones(self.np)
        self.billing_cumulative = np.zeros(self.np)
        self.billing_period = None
        self.baseline_billing_cumulative = np.zeros(self.np)
        self.baseline_billing_period = None
        self.records = []
        return self.observation()

    def clone(self):
        other = object.__new__(MicrogridEnvironment)
        other.__dict__ = self.__dict__.copy()
        other.battery = self.battery.clone()
        other.credits = self.credits.copy()
        other.billing_cumulative = self.billing_cumulative.copy()
        other.baseline_billing_cumulative = self.baseline_billing_cumulative.copy()
        other.records = list(self.records)
        return other

    def forecast_view(self) -> "MicrogridEnvironment":
        """Clone state and substitute forecast series for decision-only rollouts.

        The original environment remains untouched, so realised outcomes are always
        evaluated against the realised load/PV/RTP streams.
        """
        other = self.clone()
        # Candidate value needs only new rollout records, never the realized
        # history. Copying a growing history at every switch causes O(T^2)
        # memory traffic on annual runs.
        other.records = []
        other.load = self.load_forecast
        other.pv = self.pv_forecast
        other.rtp = self.rtp_forecast
        other.grid = self.grid_forecast
        # Current availability is measured, but future simulated outages are
        # unplanned and must not leak into the switching-policy look-ahead.
        if other.t < len(other.grid) and self.grid[self.t] != self.grid_forecast[self.t]:
            other.grid = self.grid_forecast.copy()
            other.grid[other.t] = self.grid[self.t]
        # Rebuild decision prices from forecast information. Without this step,
        # Rule-Based/D3QN policy ranking under RTP could accidentally observe the
        # realised future price matrix created by the original environment.
        other.price_matrix = self.forecast_price_matrix
        return other

    def decision_load(self, start: int, stop: int) -> np.ndarray:
        return self.load_forecast[start:stop]

    def decision_pv(self, start: int, stop: int) -> np.ndarray:
        return self.pv_forecast[start:stop]

    def decision_rtp(self, start: int, stop: int) -> np.ndarray:
        return self.rtp_forecast[start:stop]

    def _basic_charge_vector(self) -> np.ndarray:
        return np.array([tariff_basic_charge_per_day(self.best_tariffs[pid], self.cfg.tariff) / 24.0
                         for pid in self.ids], dtype=float)

    def current_price(self, mutate: bool = True) -> float:
        deficit = np.maximum(self.load[self.t] - self.pv[self.t], 0.0)
        prices = self._price_vector(self.t, mutate=mutate)
        if deficit.sum() > 0:
            return float(np.average(prices, weights=deficit))
        return float(prices.mean())

    def _price_vector(self, t: int, mutate: bool = True) -> np.ndarray:
        period = self.timestamps[t].to_period("M")
        cumulative = self.billing_cumulative
        if period != self.billing_period and mutate:
            self.billing_cumulative[:] = 0.0
            self.billing_period = period
        elif period != self.billing_period:
            cumulative = np.zeros_like(self.billing_cumulative)
        prices = self.price_matrix[t].copy()
        threshold = self.cfg.tariff.tier1_daily_kwh * period.days_in_month
        for i, pid in enumerate(self.ids):
            if self.best_tariffs[pid] == "tiered":
                prices[i] = (self.cfg.tariff.tier1_cad_per_kwh
                             if cumulative[i] < threshold
                             else self.cfg.tariff.tier2_cad_per_kwh)
        return prices

    def _baseline_price_vector(self, t: int, baseline_import: np.ndarray) -> np.ndarray:
        """No-BESS counterfactual prices with an independent tier accumulator."""
        period = self.timestamps[t].to_period("M")
        if period != self.baseline_billing_period:
            self.baseline_billing_cumulative[:] = 0.0
            self.baseline_billing_period = period
        prices = self.price_matrix[t].copy()
        threshold = self.cfg.tariff.tier1_daily_kwh * period.days_in_month
        for i, pid in enumerate(self.ids):
            if self.best_tariffs[pid] == "tiered":
                prices[i] = (self.cfg.tariff.tier1_cad_per_kwh
                             if self.baseline_billing_cumulative[i] < threshold
                             else self.cfg.tariff.tier2_cad_per_kwh)
        return prices

    def observation(self):
        if self.t >= len(self.timestamps):
            return np.zeros(10, dtype=float)
        load, pv = self.load[self.t].sum(), self.pv[self.t].sum()
        surplus, deficit = max(pv - load, 0), max(load - pv, 0)
        h = self.timestamps[self.t].hour
        next_end = min(self.t + 6, len(self.timestamps))
        next_net = (self.load_forecast[self.t:next_end].sum(axis=1) - self.pv_forecast[self.t:next_end].sum(axis=1)).mean()
        scale = max(self.cfg.battery.max_discharge_kw, 1.0)
        return np.array([
            np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), self.battery.soc,
            np.clip((load - pv) / scale, -2, 2), surplus / scale, deficit / scale,
            self.current_price(mutate=False) / 0.30, float(self.grid[self.t]), np.clip(next_net / scale, -2, 2),
            self.battery.state.soh,
        ], dtype=float)

    def step(self, action_kw: float):
        t = self.t
        load, pv = self.load[t], self.pv[t]
        self_use = np.minimum(load, pv)
        surplus, deficit = np.maximum(pv - load, 0), np.maximum(load - pv, 0)
        c, d, deg_cost, cycle_deg_cost, calendar_deg_cost = self.battery.dispatch(action_kw, surplus.sum(), deficit.sum())
        charge_alloc = allocate(surplus, c, "proportional")

        if self.grid[t]:
            # Grid-connected operation: allocate BESS discharge against the full
            # household deficits according to the selected sharing algorithm.
            discharge_alloc = allocate(deficit, d, self.cfg.allocation_method, self.priority, self.credits)
            residual = np.maximum(deficit - discharge_alloc, 0.0)
            grid_import = residual
            unserved = np.zeros_like(residual)
            critical_unserved = np.zeros_like(residual)
        else:
            # Islanded operation: critical demand is served first, independent
            # of the chosen sharing rule. Any BESS output remaining after all
            # critical deficits are covered may then serve non-critical demand.
            # This implements the client-approved industry rule: critical unmet
            # demand has first claim on the shared BESS during a grid outage.
            critical_load = load * self.critical_fraction[t]
            critical_deficit = np.maximum(critical_load - pv, 0.0)
            d_critical = min(d, float(critical_deficit.sum()))
            discharge_critical = allocate(critical_deficit, d_critical, self.cfg.allocation_method, self.priority, self.credits)
            remaining_discharge = max(d - float(discharge_critical.sum()), 0.0)
            noncritical_deficit = np.maximum(deficit - critical_deficit, 0.0)
            discharge_noncritical = allocate(noncritical_deficit, remaining_discharge, self.cfg.allocation_method, self.priority, self.credits)
            discharge_alloc = discharge_critical + discharge_noncritical
            grid_import = np.zeros_like(deficit)
            critical_unserved = np.maximum(critical_deficit - discharge_critical, 0.0)
            # Track all interrupted energy for physical balance and IEEE-style
            # interruption events, while retaining the critical-only quantity
            # for the controller's reliability objective.
            unserved = np.maximum(deficit - discharge_alloc, 0.0)

        self.credits = np.maximum(self.credits + charge_alloc / np.maximum(load, 1e-6) - discharge_alloc / np.maximum(load.max(), 1e-6), 0.05)
        exports = np.maximum(surplus - charge_alloc, 0.0)
        prices = self._price_vector(t)
        individual_cost = grid_import * prices
        threshold = self.cfg.tariff.tier1_daily_kwh * self.timestamps[t].days_in_month
        for i, pid in enumerate(self.ids):
            if self.best_tariffs[pid] == "tiered":
                tier1_energy = min(grid_import[i], max(threshold - self.billing_cumulative[i], 0.0))
                tier2_energy = grid_import[i] - tier1_energy
                individual_cost[i] = (tier1_energy * self.cfg.tariff.tier1_cad_per_kwh
                                      + tier2_energy * self.cfg.tariff.tier2_cad_per_kwh)
        basic_charge = self._basic_charge_vector()
        individual_cost = individual_cost + basic_charge
        effective_prices = prices.copy()
        grid_cost = float(individual_cost.sum() - exports.sum() * self.cfg.tariff.export_credit_cad_per_kwh)
        self.billing_cumulative += grid_import

        # Correct no-BESS counterfactual. The tiered baseline must accumulate its
        # own imports; reusing the BESS-controlled price path biases cost savings.
        baseline_import = deficit.copy() if self.grid[t] else np.zeros_like(deficit)
        baseline_prices = self._baseline_price_vector(t, baseline_import)
        individual_baseline_cost = baseline_import * baseline_prices
        baseline_threshold = self.cfg.tariff.tier1_daily_kwh * self.timestamps[t].days_in_month
        for i, pid in enumerate(self.ids):
            if self.best_tariffs[pid] == "tiered":
                tier1_energy = min(
                    baseline_import[i],
                    max(baseline_threshold - self.baseline_billing_cumulative[i], 0.0),
                )
                tier2_energy = baseline_import[i] - tier1_energy
                individual_baseline_cost[i] = (
                    tier1_energy * self.cfg.tariff.tier1_cad_per_kwh
                    + tier2_energy * self.cfg.tariff.tier2_cad_per_kwh
                )
        individual_baseline_cost = individual_baseline_cost + basic_charge
        self.baseline_billing_cumulative += baseline_import
        pv_util = float((self_use.sum() + charge_alloc.sum()) / max(pv.sum(), 1e-9)) if pv.sum() > 0 else 1.0
        reliability = float(1.0 - critical_unserved.sum()
                            / max((load * self.critical_fraction[t]).sum(), 1e-9))

        # --- Per-prosumer utility -> Atkinson EDE welfare (efficiency + equity), Eq. per
        # function_definitions.py: normalize each metric across prosumers this hour, take the
        # Atkinson EDE of each, combine via reward weights. epsilon<1 leans efficiency, >1 equity.
        rw = self.cfg.reward
        pv_i = np.where(pv > 1e-9, (self_use + charge_alloc) / np.maximum(pv, 1e-9), 1.0)
        critical_load_i = load * self.critical_fraction[t]
        reliability_i = np.where(
            critical_load_i > 1e-9,
            1.0 - critical_unserved / np.maximum(critical_load_i, 1e-9),
            1.0,
        )
        cost_norm_i = normalize_row(individual_cost, maximize=False)  # lower cost -> higher score
        pv_norm_i = normalize_row(pv_i, maximize=True)
        reliability_norm_i = normalize_row(reliability_i, maximize=True)
        ede_components = {
            "cost": ede(cost_norm_i, rw.atkinson_epsilon),
            "pv": ede(pv_norm_i, rw.atkinson_epsilon),
            "reliability": ede(reliability_norm_i, rw.atkinson_epsilon),
        }
        welfare_t = sum(rw.weights[k] * ede_components[k] for k in rw.weights)

        # Battery aging cost allocation requested for the shared asset:
        # calendar aging is time-driven and therefore shared equally; cyclic aging
        # is action-driven and allocated according to each prosumer's share of BESS
        # charge/discharge throughput during the interval.
        throughput_i = charge_alloc + discharge_alloc
        cycle_gamma_i = (throughput_i / throughput_i.sum()
                         if throughput_i.sum() > 1e-9 else np.zeros(self.np))
        calendar_deg_cost_i = np.full(self.np, calendar_deg_cost / self.np)
        cycle_deg_cost_i = cycle_gamma_i * cycle_deg_cost
        deg_cost_i = calendar_deg_cost_i + cycle_deg_cost_i

        performance_i = (
            rw.weights["cost"] * cost_norm_i
            + rw.weights["pv"] * pv_norm_i
            + rw.weights["reliability"] * reliability_norm_i
        )
        utility_i = performance_i - rw.kappa * deg_cost_i / rw.cost_scale_cad
        reward = welfare_t - rw.kappa * deg_cost / rw.cost_scale_cad
        atkinson_t = atkinson_index(performance_i, rw.atkinson_epsilon)

        record = {
            "timestamp": self.timestamps[t], "requested_action_kw": action_kw,
            "action_kw": d - c, "charge_kwh": c,
            "discharge_kwh": d, "soc": self.battery.soc, "soh": self.battery.state.soh,
            "battery_energy_kwh": self.battery.state.energy_kwh,
            "cumulative_fade": self.battery.state.cumulative_fade,
            "grid_import_kwh": grid_import.sum(), "export_kwh": exports.sum(),
            "unserved_kwh": unserved.sum(),
            "critical_unserved_kwh": critical_unserved.sum(),
            "pv_utilization": pv_util, "grid_cost_cad": grid_cost,
            "degradation_cost_cad": deg_cost,
            "cycle_degradation_cost_cad": cycle_deg_cost,
            "calendar_degradation_cost_cad": calendar_deg_cost,
            "reward": reward, "grid_available": self.grid[t],
            "grid_available_forecast": self.grid_forecast[t],
            "rtp_cad_per_kwh": self.rtp[t],
            "rtp_forecast_cad_per_kwh": self.rtp_forecast[t],
            "community_load_kwh": load.sum(), "community_pv_kwh": pv.sum(),
            "peak_net_load_kwh": max(load.sum() - pv.sum(), 0.0),
            "welfare_cost_ede": ede_components["cost"], "welfare_pv_ede": ede_components["pv"],
            "welfare_reliability_ede": ede_components["reliability"],
            "atkinson_index": atkinson_t,
            "atkinson_epsilon": rw.atkinson_epsilon, "kappa": rw.kappa, "allocation_method": self.cfg.allocation_method,
        }
        for i, pid in enumerate(self.ids):
            record[f"unserved_{pid}"] = unserved[i]
            record[f"critical_unserved_{pid}"] = critical_unserved[i]
            record[f"import_{pid}"] = grid_import[i]
            record[f"load_{pid}"] = load[i]
            record[f"pv_{pid}"] = pv[i]
            record[f"pv_utilization_{pid}"] = pv_i[i]
            record[f"charge_{pid}"] = charge_alloc[i]
            record[f"discharge_{pid}"] = discharge_alloc[i]
            record[f"price_{pid}"] = effective_prices[i]
            record[f"cost_{pid}"] = individual_cost[i]
            record[f"basic_charge_{pid}"] = basic_charge[i]
            record[f"baseline_cost_{pid}"] = individual_baseline_cost[i]
            record[f"degradation_cost_{pid}"] = deg_cost_i[i]
            record[f"cycle_degradation_cost_{pid}"] = cycle_deg_cost_i[i]
            record[f"calendar_degradation_cost_{pid}"] = calendar_deg_cost_i[i]
            record[f"performance_{pid}"] = performance_i[i]
            record[f"utility_{pid}"] = utility_i[i]
            record[f"cost_score_{pid}"] = cost_norm_i[i]
            record[f"pv_score_{pid}"] = pv_norm_i[i]
            record[f"reliability_score_{pid}"] = reliability_norm_i[i]
        self.records.append(record)
        self.t += 1
        done = self.t >= len(self.timestamps)
        return self.observation(), reward, done, record

    def results(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)
