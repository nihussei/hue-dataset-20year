from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from microgrid_opt.augmentation import AugmentationConfig, prepare_augmented_hue_community
from microgrid_opt.config import ExperimentConfig
from microgrid_opt.controllers import (
    D3QNController, D3QNTrainingConfig, MPCController, RuleBasedController,
    train_d3qn_per,
)
from microgrid_opt.core import MicrogridEnvironment
from microgrid_opt.data import (
    chronological_split, chronological_year_split, find_hue_root,
    generate_synthetic_community, load_forecast_csv, simulate_outages,
)
from microgrid_opt.equity import EPSILON_GRID, KAPPA_GRID
from microgrid_opt.evaluation import posthoc_equity_sweep, run_controller, run_policy_switching
from microgrid_opt.financial import battery_financial_projection
from microgrid_opt.metrics import (
    disaggregated_metrics, forecast_accuracy, prosumer_metrics,
    reliability_by_prosumer, temporal_metrics, variation_metrics,
)
from microgrid_opt.plots import (
    plot_allocation_method_comparison, plot_annual_tariffs,
    plot_atkinson_allocation_combinations, plot_atkinson_reward_combinations,
    plot_augmented_yearly_energy, plot_augmentation_validation,
    plot_controller_comparison, plot_disaggregated_boxplots, plot_equity_sweep,
    plot_hue_data_overview, plot_operation, plot_outage_sensitivity,
    plot_reliability, plot_switching_sweep, plot_temporal_metric_comparisons,
    plot_training, plot_variation,
)
from microgrid_opt.reporting import write_results_summary
from microgrid_opt.tariffs import fit_balanced_tariff_assignments
from microgrid_opt.validation import validate_hourly_result, write_validation_report


def parse_args():
    p = argparse.ArgumentParser(description="Complete HUE-augmented D3QN/MPC/RBC shared-BESS study")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--hue-root", help="HUE directory or Kaggle dataset root")
    source.add_argument("--data", help="Canonical hourly actual+forecast long CSV")
    source.add_argument("--synthetic-smoke", action="store_true")
    p.add_argument("--output", default="results_hue_augmented_20y")
    p.add_argument("--prosumers", type=int, default=10)
    p.add_argument("--house-ids", help="Optional comma-separated HUE house IDs")
    p.add_argument("--augment-years", type=int, default=20)
    p.add_argument("--augmentation-start-year", type=int, default=2021)
    p.add_argument("--validation-years", type=int, default=2)
    p.add_argument("--test-years", type=int, default=1)
    p.add_argument("--market-price-data")
    p.add_argument("--outage-scenario", choices=("none", "normal", "moderate", "stress"), default="moderate")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--training-window-days", type=int, default=90)
    p.add_argument("--rl-seeds", default="42,52,62")
    p.add_argument("--full", action="store_true")
    p.add_argument("--max-test-hours", type=int)
    p.add_argument("--save-augmented-data", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-outage-sensitivity", action="store_true")
    p.add_argument("--skip-plots", action="store_true")
    return p.parse_args()


def _auto_hue_root() -> Path | None:
    root = Path("/kaggle/input")
    if root.exists():
        try:
            return find_hue_root(root)
        except FileNotFoundError:
            pass
    return None


def _subset_hours(data: pd.DataFrame, hours: int | None) -> pd.DataFrame:
    if not hours:
        return data.copy()
    keep = data["timestamp"].drop_duplicates().sort_values().iloc[:hours]
    return data[data["timestamp"].isin(keep)].copy()


def _atomic_csv(frame: pd.DataFrame, path: Path, compression: str | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, compression=compression)
    temporary.replace(path)


def _read_hourly(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, parse_dates=["timestamp"])
    except Exception:
        path.unlink(missing_ok=True)
        return None


def _hourly_path(out: Path, controller: str, tariff: str, allocation: str) -> Path:
    return out / "hourly" / f"{controller}_{tariff}_{allocation}.csv.gz"


class TrainingWindowFactory:
    def __init__(self, data: pd.DataFrame, cfg: ExperimentConfig, hours: int,
                 tariffs: tuple[str, ...], fixed: dict[str, str], seed: int):
        self.data = data
        self.cfg = cfg
        self.ts = pd.DatetimeIndex(data["timestamp"].drop_duplicates().sort_values())
        self.hours = min(max(int(hours), 24), len(self.ts))
        self.tariffs = tariffs
        self.fixed = fixed
        self.rng = np.random.default_rng(seed)

    def __call__(self):
        maximum = len(self.ts) - self.hours
        start = int(self.rng.integers(0, maximum + 1)) if maximum > 0 else 0
        selected = self.ts[start:start + self.hours]
        frame = self.data[self.data["timestamp"].isin(selected)].copy()
        tariff = self.tariffs[int(self.rng.integers(0, len(self.tariffs)))]
        run_cfg = deepcopy(self.cfg); run_cfg.allocation_method = "proportional"
        return MicrogridEnvironment(frame, run_cfg, tariff=tariff, fixed_tariffs=self.fixed)


def _summary_row(result: pd.DataFrame, ids: list[str], controller: str,
                 tariff: str, allocation: str) -> dict:
    rel = reliability_by_prosumer(result, ids)
    affected = int((rel["SAIFI_interruptions"] > 0).sum())
    return {
        "controller": controller, "tariff": tariff, "allocation_method": allocation,
        "cumulative_reward": float(result["reward"].sum()),
        "net_cost_cad": float((result["grid_cost_cad"] + result["degradation_cost_cad"]).sum()),
        "pv_utilization": float(np.average(result["pv_utilization"],
                                      weights=np.maximum(result["community_pv_kwh"], 1e-9))),
        "unserved_kwh": float(result["unserved_kwh"].sum()),
        "SAIDI_hours": float(rel["SAIDI_hours"].sum() / max(len(ids), 1)),
        "SAIFI_interruptions": float(rel["SAIFI_interruptions"].sum() / max(len(ids), 1)),
        "CAIFI_interruptions_affected_customer": (
            float(rel["SAIFI_interruptions"].sum() / affected) if affected else 0.0),
    }


def _run_or_resume(out: Path, env: MicrogridEnvironment, controller, tariff: str,
                   allocation: str, source: str, resume: bool,
                   custom_path: Path | None = None) -> pd.DataFrame:
    path = custom_path or _hourly_path(out, controller.name, tariff, allocation)
    result = _read_hourly(path) if resume else None
    if result is None:
        result = run_controller(env, controller).assign(
            controller=controller.name, tariff=tariff,
            allocation_method=allocation, data_source=source)
        _atomic_csv(result, path, "gzip")
    return result


def _paired_ablation(test_rows: pd.DataFrame) -> pd.DataFrame:
    pivot_r = test_rows.pivot(index="seed", columns="replay", values="cumulative_reward")
    pivot_c = test_rows.pivot(index="seed", columns="replay", values="net_cost_cad")
    diff_r = pivot_r["prioritized"] - pivot_r["uniform"]
    diff_c = pivot_c["prioritized"] - pivot_c["uniform"]
    return pd.DataFrame([{
        "seeds": len(pivot_r),
        "per_reward_mean": pivot_r["prioritized"].mean(),
        "uniform_reward_mean": pivot_r["uniform"].mean(),
        "paired_reward_difference_mean": diff_r.mean(),
        "paired_reward_difference_std": diff_r.std(ddof=1),
        "per_net_cost_mean_cad": pivot_c["prioritized"].mean(),
        "uniform_net_cost_mean_cad": pivot_c["uniform"].mean(),
        "paired_net_cost_difference_mean_cad": diff_c.mean(),
        "per_reward_wins": int((diff_r > 0).sum()),
        "uniform_reward_wins": int((diff_r < 0).sum()),
        "ties": int(np.isclose(diff_r, 0.0).sum()),
    }])


def main():
    args = parse_args(); started = time.time()
    out = Path(args.output)
    for folder in ("hourly", "figures", "tables", "models"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(v.strip()) for v in args.rl_seeds.split(",") if v.strip())
    if not seeds: raise ValueError("--rl-seeds cannot be empty")

    metadata: dict[str, pd.DataFrame] = {}
    augmented_cache = out / "augmented_hue_canonical_20y.csv.gz"
    hue_root = Path(args.hue_root) if args.hue_root else _auto_hue_root()
    if args.data:
        full_data = load_forecast_csv(args.data); source = "client_canonical_data"
    elif hue_root is not None:
        if args.resume and augmented_cache.exists():
            full_data = load_forecast_csv(augmented_cache)
            source = "HUE_observed_profiles_multivariate_20year_augmentation"
            for path in (out / "tables").glob("hue_*.csv"):
                try: metadata[path.stem.removeprefix("hue_")] = pd.read_csv(path)
                except Exception: pass
        else:
            requested = [int(v) for v in args.house_ids.split(",")] if args.house_ids else None
            augmentation = AugmentationConfig(start_year=args.augmentation_start_year,
                                              years=args.augment_years, seed=42)
            full_data, metadata = prepare_augmented_hue_community(
                hue_root, n_prosumers=args.prosumers, house_ids=requested,
                augmentation=augmentation, outage_scenario=args.outage_scenario,
                market_price_path=args.market_price_data)
            source = "HUE_observed_profiles_multivariate_20year_augmentation"
            if args.save_augmented_data: _atomic_csv(full_data, augmented_cache, "gzip")
    elif args.synthetic_smoke:
        full_data = generate_synthetic_community(args.prosumers, 120, seed=42)
        source = "synthetic_smoke_test_NOT_CLIENT_RESULT"
    else:
        raise FileNotFoundError("Attach the HUE Kaggle dataset or pass --hue-root")

    if "is_augmented" in full_data and full_data["timestamp"].dt.year.nunique() >= 4:
        train, validation, test, split_manifest = chronological_year_split(
            full_data, args.validation_years, args.test_years)
    else:
        train, validation, test, split_manifest = chronological_split(full_data, 180, 365)
    test = _subset_hours(test, args.max_test_hours)
    cfg = ExperimentConfig()
    fixed_tariffs, tariff_fit = fit_balanced_tariff_assignments(train, cfg.tariff)
    split_manifest.to_csv(out / "tables/data_split_manifest.csv", index=False)
    tariff_fit.to_csv(out / "tables/balanced_tariff_assignments_from_training.csv", index=False)
    forecast_accuracy(test).to_csv(out / "tables/forecast_accuracy_test.csv", index=False)
    for name, table in metadata.items(): table.to_csv(out / "tables" / f"hue_{name}.csv", index=False)
    if not args.skip_plots:
        plot_hue_data_overview(test, out / "figures")
        if "augmentation_validation" in metadata:
            plot_augmentation_validation(metadata["augmentation_validation"], out / "figures")
        if "augmentation_yearly_energy" in metadata:
            plot_augmented_yearly_energy(metadata["augmentation_yearly_energy"], out / "figures")

    all_tariffs = ("flat", "tiered", "tou", "rtp", "balanced")
    window_hours = min(args.training_window_days * 24, train["timestamp"].nunique())
    train_cfg = D3QNTrainingConfig(
        episodes=args.episodes, warmup_steps=min(1000, max(128, window_hours // 4)),
        target_update_steps=500, epsilon_decay_steps=max(window_hours * args.episodes, 1),
        replay_capacity=min(100_000, max(20_000, window_hours * args.episodes)))
    print(f"Dataset split: train={train['timestamp'].nunique()}h; "
          f"validation={validation['timestamp'].nunique()}h; test={test['timestamp'].nunique()}h", flush=True)
    agents: dict[tuple[str, int], D3QNController] = {}; history_frames = []
    for replay, alpha, agent_name in (("prioritized", 0.6, "d3qn_per"),
                                       ("uniform", 0.0, "d3qn_uniform_replay")):
        for seed in seeds:
            model_path = out / "models" / f"{agent_name}_seed_{seed}.npz"
            history_path = out / "tables" / f"{agent_name}_seed_{seed}_history.csv"
            if args.resume and model_path.exists() and history_path.exists():
                print(f"Resuming {agent_name} seed={seed} from checkpoint", flush=True)
                agent = D3QNController.load(str(model_path), name=agent_name)
                history = pd.read_csv(history_path)
            else:
                print(f"Training {agent_name} seed={seed} ({args.episodes} episodes)", flush=True)
                agent, raw_history = train_d3qn_per(
                    TrainingWindowFactory(train, cfg, window_hours, all_tariffs,
                                          fixed_tariffs, seed),
                    train_cfg, seed=seed, replay_alpha=alpha, agent_name=agent_name)
                history = pd.DataFrame(raw_history).assign(replay=replay, seed=seed)
                agent.save(str(model_path)); history.to_csv(history_path, index=False)
            if "replay" not in history: history["replay"] = replay
            if "seed" not in history: history["seed"] = seed
            agents[(replay, seed)] = agent; history_frames.append(history)
    history = pd.concat(history_frames, ignore_index=True)
    history.to_csv(out / "tables/d3qn_training_history.csv", index=False)
    if not args.skip_plots: plot_training(history, out / "figures")

    validation_rows = []
    for (replay, seed), agent in agents.items():
        env = MicrogridEnvironment(validation, cfg, tariff="tou", fixed_tariffs=fixed_tariffs)
        result = run_controller(env, agent)
        validation_rows.append({"replay": replay, "seed": seed,
                                **_summary_row(result, env.ids, agent.name, "tou", "proportional")})
    validation_summary = pd.DataFrame(validation_rows)
    validation_summary.to_csv(out / "tables/d3qn_seed_validation_summary.csv", index=False)
    selected_seed = {replay: int(group.loc[group["cumulative_reward"].idxmax(), "seed"])
                     for replay, group in validation_summary.groupby("replay")}
    selected_per = agents[("prioritized", selected_seed["prioritized"])]
    selected_uniform = agents[("uniform", selected_seed["uniform"])]
    shutil.copy2(out / "models" / f"d3qn_per_seed_{selected_seed['prioritized']}.npz",
                 out / "models/d3qn_per_model.npz")
    shutil.copy2(out / "models" / f"d3qn_uniform_replay_seed_{selected_seed['uniform']}.npz",
                 out / "models/d3qn_uniform_replay_model.npz")
    validation_summary[validation_summary["seed"].eq(
        validation_summary["replay"].map(selected_seed))].to_csv(
            out / "tables/d3qn_validation_summary.csv", index=False)

    seed_test_rows = []
    for (replay, seed), agent in agents.items():
        env = MicrogridEnvironment(test, cfg, tariff="tou", fixed_tariffs=fixed_tariffs)
        result = run_controller(env, agent)
        seed_test_rows.append({"replay": replay, "seed": seed,
                               **_summary_row(result, env.ids, agent.name, "tou", "proportional")})
    seed_test = pd.DataFrame(seed_test_rows)
    seed_test.to_csv(out / "tables/d3qn_seed_test_summary.csv", index=False)
    _paired_ablation(seed_test).to_csv(out / "tables/d3qn_per_ablation_statistics.csv", index=False)

    controllers = [RuleBasedController(), MPCController(cfg.mpc_horizon_hours), selected_per, selected_uniform]
    tariffs = all_tariffs if args.full else ("tou", "balanced")
    temporal_all, prosumer_all, disagg_all, variation_all = [], [], [], []
    runtime_checks, financial_rows, allocation_rows, controller_rows = [], [], [], []
    representative = {}; equity_trajectories = {}; proportional_cache = {}
    print("Evaluating four controllers across all pricing cases...", flush=True)
    for tariff in tariffs:
        for controller in controllers:
            run_cfg = deepcopy(cfg); run_cfg.allocation_method = "proportional"
            env = MicrogridEnvironment(test, run_cfg, tariff=tariff, fixed_tariffs=fixed_tariffs)
            result = _run_or_resume(out, env, controller, tariff, "proportional", source, args.resume)
            proportional_cache[(controller.name, tariff)] = result
            representative[(controller.name, tariff)] = (result, reliability_by_prosumer(result, env.ids), env.ids)
            if tariff == "tou": equity_trajectories[(controller.name, "proportional")] = result
            controller_rows.append(_summary_row(result, env.ids, controller.name, tariff, "proportional"))
            for freq in ("monthly", "seasonal", "annual"):
                tm = temporal_metrics(result, freq).assign(controller=controller.name,
                                                           tariff=tariff, allocation_method="proportional")
                dm = disaggregated_metrics(result, env.ids, freq).assign(controller=controller.name,
                                                                         tariff=tariff, allocation_method="proportional")
                temporal_all.append(tm); disagg_all.append(dm)
                variation_all.append(variation_metrics(dm).assign(controller=controller.name,
                                                                    tariff=tariff, allocation_method="proportional"))
            prosumer_all.append(prosumer_metrics(result, env.ids).assign(
                controller=controller.name, tariff=tariff, allocation_method="proportional"))
            runtime_checks.append({"run_type": "controller_tariff", "controller": controller.name,
                                   "tariff": tariff, "allocation_method": "proportional",
                                   **validate_hourly_result(result, run_cfg)})
            financial_rows.append({"analysis_scope": "controller_tariff", "controller": controller.name,
                                   "tariff": tariff, "allocation_method": "proportional",
                                   **battery_financial_projection(result, run_cfg.battery)})

    print("Evaluating proportional, priority and contribution allocation...", flush=True)
    for tariff in ("tou", "balanced"):
        for allocation in cfg.allocation_methods:
            for controller in controllers:
                run_cfg = deepcopy(cfg); run_cfg.allocation_method = allocation
                env = MicrogridEnvironment(test, run_cfg, tariff=tariff, fixed_tariffs=fixed_tariffs)
                if allocation == "proportional": result = proportional_cache[(controller.name, tariff)]
                else:
                    result = _run_or_resume(out, env, controller, tariff, allocation, source, args.resume)
                    runtime_checks.append({"run_type": "allocation_comparison", "controller": controller.name,
                                           "tariff": tariff, "allocation_method": allocation,
                                           **validate_hourly_result(result, run_cfg)})
                allocation_rows.append(_summary_row(result, env.ids, controller.name, tariff, allocation))
                if tariff == "tou": equity_trajectories[(controller.name, allocation)] = result
                financial_rows.append({"analysis_scope": "allocation_comparison", "controller": controller.name,
                                       "tariff": tariff, "allocation_method": allocation,
                                       **battery_financial_projection(result, cfg.battery)})

    temporal = pd.concat(temporal_all, ignore_index=True); prosumer = pd.concat(prosumer_all, ignore_index=True)
    disaggregated = pd.concat(disagg_all, ignore_index=True); variation = pd.concat(variation_all, ignore_index=True)
    allocation_summary = pd.DataFrame(allocation_rows); financial = pd.DataFrame(financial_rows)
    pd.DataFrame(controller_rows).to_csv(out / "tables/controller_tariff_summary.csv", index=False)
    temporal.to_csv(out / "tables/temporal_metrics.csv", index=False)
    prosumer.to_csv(out / "tables/prosumer_metrics.csv", index=False)
    disaggregated.to_csv(out / "tables/disaggregated_prosumer_metrics.csv", index=False)
    variation.to_csv(out / "tables/variation_metrics.csv", index=False)
    allocation_summary.to_csv(out / "tables/allocation_method_comparison.csv", index=False)
    financial.to_csv(out / "tables/battery_npv_replacement_summary.csv", index=False)

    print("Sweeping 1, 3, 6, 12 and 24 h policy-switch intervals...", flush=True)
    switching_rows = []; windows = cfg.switching_windows if args.full else (3, 6)
    for window in windows:
        hourly_path = out / "hourly" / f"policy_switch_{window}h.csv.gz"
        decision_path = out / "tables" / f"policy_switch_decisions_{window}h.csv"
        switched = _read_hourly(hourly_path) if args.resume else None
        if switched is None or not decision_path.exists():
            env = MicrogridEnvironment(test, cfg, tariff="tou", fixed_tariffs=fixed_tariffs)
            switched, decisions = run_policy_switching(env, controllers[:3],
                switch_interval_hours=window,
                evaluation_horizon_hours=cfg.switching_evaluation_horizon_hours)
            _atomic_csv(switched, hourly_path, "gzip"); decisions.to_csv(decision_path, index=False)
        ids = sorted(c.removeprefix("load_") for c in switched if c.startswith("load_"))
        switching_rows.append({"switch_interval_hours": window,
                               "evaluation_horizon_hours": cfg.switching_evaluation_horizon_hours,
                               **{k: v for k, v in _summary_row(switched, ids, "policy_switch", "tou", "proportional").items()
                                  if k not in ("controller", "tariff", "allocation_method")}})
    switching_summary = pd.DataFrame(switching_rows)
    switching_summary.to_csv(out / "tables/switching_window_summary.csv", index=False)

    outage_rows = []
    if not args.skip_outage_sensitivity:
        print("Running normal/moderate/stress reliability sensitivity...", flush=True)
        for scenario in ("normal", "moderate", "stress"):
            scenario_data, events = simulate_outages(test, scenario, seed=42)
            events.to_csv(out / "tables" / f"outage_events_{scenario}.csv", index=False)
            for controller in controllers:
                run_cfg = deepcopy(cfg); run_cfg.allocation_method = "proportional"
                env = MicrogridEnvironment(scenario_data, run_cfg, tariff="tou", fixed_tariffs=fixed_tariffs)
                path = out / "hourly" / f"outage_{scenario}_{controller.name}.csv.gz"
                result = _run_or_resume(out, env, controller, "tou", "proportional", source,
                                        args.resume, custom_path=path)
                row = _summary_row(result, env.ids, controller.name, "tou", "proportional")
                row["outage_scenario"] = scenario; outage_rows.append(row)
                runtime_checks.append({"run_type": f"outage_{scenario}", "controller": controller.name,
                                       "tariff": "tou", "allocation_method": "proportional",
                                       **validate_hourly_result(result, run_cfg)})
    outage_summary = pd.DataFrame(outage_rows)
    outage_summary.to_csv(out / "tables/outage_sensitivity_summary.csv", index=False)

    print("Re-scoring exact 18-point epsilon-kappa grid...", flush=True)
    epsilon_grid = EPSILON_GRID if args.full else EPSILON_GRID[:2]
    kappa_grid = KAPPA_GRID if args.full else KAPPA_GRID[:3]
    equity_result = posthoc_equity_sweep(equity_trajectories, cfg, epsilon_grid, kappa_grid,
                                         capture_reward_curves=not args.skip_plots)
    if args.skip_plots: equity_df, reward_curves = equity_result, {}
    else: equity_df, reward_curves = equity_result
    equity_df.to_csv(out / "tables/equity_epsilon_kappa_sweep.csv", index=False)
    pd.DataFrame(runtime_checks).to_csv(out / "tables/runtime_validation_checks.csv", index=False)
    write_validation_report(out, runtime_checks, equity_df)

    years = full_data["timestamp"].dt.year.nunique()
    manifest = pd.DataFrame([{
        "project_version": "6.1.0-client-revision", "data_source": source,
        "n_prosumers": test["prosumer_id"].nunique(),
        "augmentation_years": years if "is_augmented" in full_data else 0,
        "augmentation_start": full_data["timestamp"].min(), "augmentation_end": full_data["timestamp"].max(),
        "train_hours": train["timestamp"].nunique(), "validation_hours": validation["timestamp"].nunique(),
        "test_hours": test["timestamp"].nunique(), "tariffs": ",".join(tariffs),
        "allocation_methods": ",".join(cfg.allocation_methods), "switching_intervals": ",".join(map(str, windows)),
        "switch_evaluation_horizon_hours": cfg.switching_evaluation_horizon_hours,
        "epsilon_values": ",".join(map(str, epsilon_grid)), "kappa_values": ",".join(map(str, kappa_grid)),
        "d3qn_training_episodes_per_seed": args.episodes, "d3qn_seeds": ",".join(map(str, seeds)),
        "selected_per_seed": selected_seed["prioritized"], "selected_uniform_seed": selected_seed["uniform"],
        "outage_scenario": args.outage_scenario, "outage_sensitivity": not args.skip_outage_sensitivity,
        "historical_midc_rtp_anchor": True, "full_mode": bool(args.full),
        "preliminary_test_cap_hours": args.max_test_hours or "", "elapsed_minutes": (time.time() - started) / 60,
    }])
    manifest.to_csv(out / "RUN_MANIFEST.csv", index=False)
    write_results_summary(out, temporal, switching_summary, allocation_summary, equity_df, financial, source)

    if not args.skip_plots:
        monthly = temporal[(temporal["frequency"] == "monthly") & (temporal["tariff"] == "tou")]
        seasonal = temporal[(temporal["frequency"] == "seasonal") & (temporal["tariff"] == "tou")].copy()
        seasonal = seasonal[seasonal["period"].str.endswith(("Summer", "Winter"))]
        annual = temporal[(temporal["frequency"] == "annual") & (temporal["tariff"] == "tou")]
        plot_controller_comparison(monthly, out / "figures", frequency_label="Monthly",
                                   filename="01_monthly_controller_boxplots.png")
        plot_controller_comparison(seasonal, out / "figures", frequency_label="Seasonal (Summer/Winter)",
                                   filename="01_seasonal_summer_winter_controller_boxplots.png")
        plot_controller_comparison(annual, out / "figures", frequency_label="Annual",
                                   filename="01_annual_controller_boxplots.png")
        plot_switching_sweep(switching_summary, out / "figures")
        plot_annual_tariffs(temporal, out / "figures"); plot_temporal_metric_comparisons(temporal, out / "figures/temporal")
        plot_allocation_method_comparison(allocation_summary, out / "figures"); plot_equity_sweep(equity_df, out / "figures")
        if not outage_summary.empty: plot_outage_sensitivity(outage_summary, out / "figures")
        plot_atkinson_reward_combinations(reward_curves,
            out / "figures/atkinson_combinations/controller_comparison_proportional",
            tariff="tou", allocation_method="proportional")
        plot_atkinson_allocation_combinations(reward_curves,
            out / "figures/atkinson_combinations/original_allocation_comparison_d3qn_per",
            tariff="tou", controller="d3qn_per")
        for (controller_name, tariff), (result, reliability, ids) in representative.items():
            folder = out / "figures/disaggregated" / controller_name / tariff
            plot_operation(result, folder, suffix=f"_{tariff}_{controller_name}")
            plot_reliability(reliability, folder, suffix=f"_{tariff}_{controller_name}")
            for freq in ("monthly", "seasonal", "annual"):
                dm = disaggregated[(disaggregated["tariff"] == tariff) & (disaggregated["controller"] == controller_name)
                                   & (disaggregated["frequency"] == freq)]
                vm = variation[(variation["tariff"] == tariff) & (variation["controller"] == controller_name)
                               & (variation["frequency"] == freq)]
                plot_disaggregated_boxplots(dm, freq, folder, filename_prefix="07")
                plot_variation(vm, freq, folder, filename_prefix="08")
    print(f"COMPLETED: {source}", flush=True); print(f"Results: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
