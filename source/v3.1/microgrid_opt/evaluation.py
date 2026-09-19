from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from .core import MicrogridEnvironment
from .equity import EPSILON_GRID, KAPPA_GRID
from .metrics import equity_summary


def run_controller(env: MicrogridEnvironment, controller) -> pd.DataFrame:
    obs = env.reset(); done = False
    while not done:
        action = controller.act(obs, env, explore=False)
        obs, _, done, _ = env.step(action)
    return env.results()


def _rollout_value_forecast(env: MicrogridEnvironment, controller, steps: int) -> float:
    """Evaluate candidate policy on forecast data only, from the current state.

    This prevents oracle leakage: the selector never sees realised future load/PV
    when deciding which controller to activate.
    """
    trial = env.forecast_view()
    obs = trial.observation(); total = 0.0
    stop = min(trial.t + steps, len(trial.timestamps))
    while trial.t < stop:
        action = controller.act(obs, trial, explore=False)
        obs, reward, done, _ = trial.step(action)
        total += reward
        if done:
            break
    return float(total)


def run_equity_sweep(data, cfg, controllers: list, tariff: str = "tou",
                      epsilon_grid=EPSILON_GRID, kappa_grid=KAPPA_GRID,
                      allocation_methods=None, capture_reward_curves: bool = False):
    """Run the client's epsilon x kappa efficiency-equity sweep.

    When ``allocation_methods`` is supplied, each of the three original
    allocation algorithms is tested independently under the same controller,
    tariff, epsilon and kappa settings.
    """
    allocation_methods = tuple(allocation_methods or (cfg.allocation_method,))
    rows = []
    reward_curves = {}
    for allocation_method in allocation_methods:
        for epsilon in epsilon_grid:
            for kappa in kappa_grid:
                run_cfg = deepcopy(cfg)
                run_cfg.reward.atkinson_epsilon = epsilon
                run_cfg.reward.kappa = kappa
                run_cfg.allocation_method = allocation_method
                for controller in controllers:
                    env = MicrogridEnvironment(data, run_cfg, tariff=tariff)
                    result = run_controller(env, controller)
                    summary = equity_summary(result, env.ids, epsilon, kappa)
                    summary["controller"] = controller.name
                    summary["allocation_method"] = allocation_method
                    summary["tariff"] = tariff
                    rows.append(summary)
                    if capture_reward_curves:
                        reward_curves[(float(epsilon), float(kappa), controller.name,
                                       allocation_method)] = (
                            result[["timestamp", "reward"]].copy())
    summary_df = pd.DataFrame(rows)
    if capture_reward_curves:
        return summary_df, reward_curves
    return summary_df


def run_allocation_method_sweep(data, cfg, controllers: list,
                                tariffs=("tou", "balanced"),
                                allocation_methods=("proportional", "priority", "contribution"),
                                fixed_tariffs: dict[str, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Directly compare the three client-provided allocation algorithms.

    Returns a compact summary and all hourly runs. The aggregate controller is
    held explicit in the output so allocation-method effects are not conflated
    with D3QN/MPC/rule-based effects.
    """
    summary_rows = []
    hourly = []
    for tariff in tariffs:
        for allocation_method in allocation_methods:
            for controller in controllers:
                run_cfg = deepcopy(cfg)
                run_cfg.allocation_method = allocation_method
                env = MicrogridEnvironment(data, run_cfg, tariff=tariff,
                                           fixed_tariffs=fixed_tariffs)
                result = run_controller(env, controller)
                result = result.assign(controller=controller.name, tariff=tariff,
                                       allocation_method=allocation_method)
                hourly.append(result)
                reliability_cols = [f"unserved_{pid}" for pid in env.ids]
                affected = np.array([(result[c] > 1e-9).any() for c in reliability_cols])
                interruption_events = 0
                interruption_hours = 0
                for c in reliability_cols:
                    mask = result[c].to_numpy() > 1e-9
                    interruption_events += int(np.sum(mask & ~np.r_[False, mask[:-1]]))
                    interruption_hours += int(mask.sum())
                n = max(len(env.ids), 1)
                n_affected = int(affected.sum())
                summary_rows.append({
                    "controller": controller.name,
                    "tariff": tariff,
                    "allocation_method": allocation_method,
                    "cumulative_reward": float(result["reward"].sum()),
                    "net_cost_cad": float((result["grid_cost_cad"] + result["degradation_cost_cad"]).sum()),
                    "pv_utilization": float(np.average(result["pv_utilization"], weights=np.maximum(result["community_pv_kwh"], 1e-9))),
                    "unserved_kwh": float(result["unserved_kwh"].sum()),
                    "SAIDI_hours": float(interruption_hours / n),
                    "SAIFI_interruptions": float(interruption_events / n),
                    "CAIFI_interruptions_affected_customer": float(interruption_events / n_affected) if n_affected else 0.0,
                })
    return pd.DataFrame(summary_rows), pd.concat(hourly, ignore_index=True)


def run_policy_switching(env: MicrogridEnvironment, controllers: list, switch_interval_hours: int,
                         evaluation_horizon_hours: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Switch every ``switch_interval_hours`` using next-hour forecast reward.

    This implements the client's wording literally: e.g. with a 6 h switching
    window, a controller is selected every 6 h based on the highest expected
    cumulative reward for the *next forecasted hour*. The two horizons are kept
    separate so the study can sweep the switching interval without changing the
    selection objective.
    """
    if switch_interval_hours < 1 or evaluation_horizon_hours < 1:
        raise ValueError("Switch interval and evaluation horizon must be >= 1 hour")
    obs = env.reset(); done = False; selected = controllers[0]; decisions = []
    while not done:
        if env.t % switch_interval_hours == 0:
            values = [_rollout_value_forecast(env, c, evaluation_horizon_hours) for c in controllers]
            selected = controllers[int(np.argmax(values))]
            decisions.append({
                "timestamp": env.timestamps[env.t],
                "switch_interval_hours": switch_interval_hours,
                "evaluation_horizon_hours": evaluation_horizon_hours,
                "selected_controller": selected.name,
                "expected_next_horizon_reward": float(max(values)),
                **{f"expected_{c.name}_reward": float(v) for c, v in zip(controllers, values)},
            })
        action = selected.act(obs, env, explore=False)
        obs, _, done, _ = env.step(action)
    return env.results(), pd.DataFrame(decisions)


def _rowwise_ede(values: np.ndarray, epsilon: float) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 1e-9, None)
    if abs(epsilon - 1.0) < 1e-9:
        return np.exp(np.mean(np.log(values), axis=1))
    return np.mean(values ** (1.0 - epsilon), axis=1) ** (1.0 / (1.0 - epsilon))


def posthoc_equity_sweep(
    trajectories: dict[tuple[str, str], pd.DataFrame],
    cfg,
    epsilon_grid=EPSILON_GRID,
    kappa_grid=KAPPA_GRID,
    capture_reward_curves: bool = False,
):
    """Re-score fixed physical trajectories over the client's 18-point grid.

    This matches the original PlotRewardFunction workflow: dispatch is run once
    per controller/allocation pair, then epsilon and kappa re-score the same
    physical outcomes.  It avoids pretending that 216 separate dispatch runs
    are independent controller-training experiments.
    """
    rows, curves = [], {}
    weights = cfg.reward.weights
    for (controller_name, allocation_method), result in trajectories.items():
        ids = sorted(c[len("cost_score_"):] for c in result.columns
                     if c.startswith("cost_score_"))
        if not ids:
            raise ValueError("Trajectory lacks per-prosumer component score columns")
        cost = result[[f"cost_score_{pid}" for pid in ids]].to_numpy()
        pv = result[[f"pv_score_{pid}" for pid in ids]].to_numpy()
        reliability = result[[f"reliability_score_{pid}" for pid in ids]].to_numpy()
        degradation = result["degradation_cost_cad"].to_numpy(dtype=float)
        degradation_i = result[[f"degradation_cost_{pid}" for pid in ids]].to_numpy()
        performance_i = (weights["cost"] * cost + weights["pv"] * pv
                         + weights["reliability"] * reliability)
        mean_perf = np.maximum(performance_i.mean(axis=1), 1e-9)
        for epsilon in epsilon_grid:
            welfare = (weights["cost"] * _rowwise_ede(cost, epsilon)
                       + weights["pv"] * _rowwise_ede(pv, epsilon)
                       + weights["reliability"] * _rowwise_ede(reliability, epsilon))
            inequality = 1.0 - _rowwise_ede(performance_i, epsilon) / mean_perf
            for kappa in kappa_grid:
                reward = welfare - kappa * degradation / cfg.reward.cost_scale_cad
                utility_i = performance_i - kappa * degradation_i / cfg.reward.cost_scale_cad
                cumulative_utility_i = utility_i.sum(axis=0)
                rows.append({
                    "atkinson_epsilon": float(epsilon), "kappa": float(kappa),
                    "mean_prosumer_reward": float(cumulative_utility_i.mean()),
                    "prosumer_reward_variance": float(cumulative_utility_i.var()),
                    "atkinson_index": float(np.mean(inequality)),
                    "equity_adjusted_welfare": float(reward.sum()),
                    "cumulative_reward": float(reward.sum()),
                    "controller": controller_name,
                    "allocation_method": allocation_method,
                    "tariff": "tou",
                })
                if capture_reward_curves:
                    curves[(float(epsilon), float(kappa), controller_name,
                            allocation_method)] = pd.DataFrame({
                                "timestamp": result["timestamp"].to_numpy(),
                                "reward": reward,
                            })
    summary = pd.DataFrame(rows)
    return (summary, curves) if capture_reward_curves else summary
