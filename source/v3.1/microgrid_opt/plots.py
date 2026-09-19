from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RELIABILITY_LABELS = {
    "SAIDI_hours": ("SAIDI", "Interruption duration (h/customer)"),
    "SAIFI_interruptions": ("SAIFI", "Interruptions/customer"),
    "CAIFI_interruptions_affected_customer": (
        "CAIFI", "Interruptions/affected customer"
    ),
}


def _safe_name(value: str) -> str:
    return str(value).replace("/", "-").replace(" ", "_")


def plot_controller_comparison(summary: pd.DataFrame, output: str | Path,
                               frequency_label: str = "Monthly",
                               filename: str = "01_monthly_controller_boxplots.png"):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    metrics = ["net_cost_cad", "pv_utilization", "peak_load_reduction_kw", "cumulative_reward"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric in zip(axes.flat, metrics):
        summary.boxplot(column=metric, by="controller", ax=ax, grid=False)
        ax.set_title(metric.replace("_", " ").title()); ax.set_xlabel("")
    fig.suptitle(f"{frequency_label} Controller Comparison")
    fig.tight_layout(); fig.savefig(output / filename, dpi=180); plt.close(fig)


def plot_reliability(reliability: pd.DataFrame, output: str | Path, suffix: str = ""):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    d = reliability.set_index("prosumer_id")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    cols = ["SAIDI_hours", "SAIFI_interruptions", "CAIFI_interruptions_affected_customer"]
    for ax, col in zip(axes, cols):
        title, ylabel = RELIABILITY_LABELS[col]
        d[col].plot.bar(ax=ax, color="#2f6f9f")
        ax.set_title(title); ax.set_xlabel("Prosumer"); ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.2)
        if np.isclose(d[col].fillna(0).abs().max(), 0):
            ax.set_ylim(0, 1)
            ax.text(.5, .5, "No interruption events\nin this evaluation horizon",
                    ha="center", va="center", transform=ax.transAxes, color="#555555")
    fig.suptitle("IEEE Reliability Metrics by Prosumer")
    fig.tight_layout(); fig.savefig(output / f"02_reliability_by_prosumer{suffix}.png", dpi=180); plt.close(fig)


def plot_operation(results: pd.DataFrame, output: str | Path, hours=168, suffix: str = ""):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    d = results.iloc[:hours]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(d["timestamp"], d["community_load_kwh"], label="Load")
    axes[0].plot(d["timestamp"], d["community_pv_kwh"], label="PV"); axes[0].legend(); axes[0].set_ylabel("kWh")
    axes[1].plot(d["timestamp"], d["action_kw"], label="BESS action"); axes[1].axhline(0, lw=.7); axes[1].set_ylabel("kW")
    axes[2].plot(d["timestamp"], 100*d["soc"], label="SoC"); axes[2].set_ylabel("SoC (%)")
    fig.suptitle("One-Week BESS Operation")
    fig.tight_layout(); fig.savefig(output / f"03_operation{suffix}.png", dpi=180); plt.close(fig)


def plot_training(history: pd.DataFrame, output: str | Path):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for replay, g in history.groupby("replay"):
        if "seed" in g and g["seed"].nunique() > 1:
            summary = g.groupby("episode").agg(
                reward_mean=("reward", "mean"), reward_std=("reward", "std"),
                loss_mean=("mean_loss", "mean"), loss_std=("mean_loss", "std"),
            ).reset_index().fillna(0.0)
            x = summary["episode"].to_numpy()
            for ax, mean, std in ((axes[0], "reward_mean", "reward_std"),
                                  (axes[1], "loss_mean", "loss_std")):
                y = summary[mean].to_numpy(); s = summary[std].to_numpy()
                line = ax.plot(x, y, linewidth=2, label=f"{replay} mean")[0]
                ax.fill_between(x, y - s, y + s, alpha=.18, color=line.get_color())
        else:
            axes[0].plot(g["episode"], g["reward"], marker="o", label=replay)
            axes[1].plot(g["episode"], g["mean_loss"], marker="o", label=replay)
    axes[0].set_title("D3QN Training Reward"); axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Reward")
    axes[1].set_title("D3QN Training Loss"); axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Huber loss")
    for ax in axes: ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(output / "04_d3qn_per_vs_uniform_training.png", dpi=180); plt.close(fig)


def plot_augmentation_validation(validation: pd.DataFrame, output: str | Path):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    pairs = [
        ("historical_mean_kwh", "augmented_first_year_mean_kwh", "Mean hourly load"),
        ("historical_p95_kwh", "augmented_first_year_p95_kwh", "95th percentile load"),
        ("historical_acf24", "augmented_first_year_acf24", "24-hour autocorrelation"),
    ]
    for ax, (xcol, ycol, title) in zip(axes, pairs):
        ax.scatter(validation[xcol], validation[ycol], color="#1768ac", s=45)
        lo = min(validation[xcol].min(), validation[ycol].min())
        hi = max(validation[xcol].max(), validation[ycol].max())
        ax.plot([lo, hi], [lo, hi], "--", color="#555555")
        ax.set_xlabel("Observed HUE"); ax.set_ylabel("Augmented year 1")
        ax.set_title(title); ax.grid(alpha=.25)
    fig.suptitle("HUE Augmentation Validation by Prosumer")
    fig.tight_layout(); fig.savefig(output / "00_augmentation_validation.png", dpi=200); plt.close(fig)


def plot_augmented_yearly_energy(yearly: pd.DataFrame, output: str | Path):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    community = yearly.groupby("year")[["load_kwh", "pv_kwh"]].sum()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(community.index, community["load_kwh"] / 1000, marker="o", label="Load")
    ax.plot(community.index, community["pv_kwh"] / 1000, marker="o", label="PV")
    ax.set_xlabel("Year"); ax.set_ylabel("Community energy (MWh/year)")
    ax.set_title("Twenty-Year Augmented HUE Energy Trajectory")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(output / "00_augmented_20year_energy.png", dpi=200); plt.close(fig)


def plot_outage_sensitivity(summary: pd.DataFrame, output: str | Path):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [("SAIDI_hours", "SAIDI (h/customer)"),
               ("SAIFI_interruptions", "SAIFI (interruptions/customer)"),
               ("unserved_kwh", "Unserved energy (kWh)")]
    for ax, (metric, title) in zip(axes, metrics):
        pivot = summary.pivot(index="outage_scenario", columns="controller", values=metric)
        order = [x for x in ("normal", "moderate", "stress") if x in pivot.index]
        pivot.loc[order].plot.bar(ax=ax)
        ax.set_title(title); ax.set_xlabel("Outage scenario"); ax.grid(axis="y", alpha=.25)
    fig.suptitle("Reliability Sensitivity to Simulated Outage Severity")
    fig.tight_layout(); fig.savefig(output / "10_outage_sensitivity.png", dpi=200); plt.close(fig)


def plot_switching_sweep(summary: pd.DataFrame, output: str | Path):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    xcol = "switch_interval_hours" if "switch_interval_hours" in summary else "window_hours"
    fig, ax1 = plt.subplots(figsize=(8, 4.5)); ax2 = ax1.twinx()
    reward_line = ax1.plot(
        summary[xcol], summary["cumulative_reward"], marker="o", linewidth=2,
        color="#1768ac", label="Cumulative reward"
    )
    cost_line = ax2.plot(
        summary[xcol], summary["net_cost_cad"], marker="s", linewidth=2,
        color="#d1495b", label="Net cost"
    )
    ax1.set_xlabel("Policy switching interval (hours)")
    ax1.set_ylabel("Cumulative reward", color="#1768ac")
    ax2.set_ylabel("Net cost (CAD)", color="#d1495b")
    ax1.tick_params(axis="y", colors="#1768ac"); ax2.tick_params(axis="y", colors="#d1495b")
    ax1.set_xticks(sorted(summary[xcol].unique()))
    ax1.grid(alpha=.25)
    ax1.legend(reward_line + cost_line,
               [line.get_label() for line in reward_line + cost_line],
               loc="best")
    fig.suptitle("Policy-Switching Interval Sensitivity")
    fig.tight_layout(); fig.savefig(output / "05_switching_interval_sweep.png", dpi=180); plt.close(fig)


def plot_hue_data_overview(data: pd.DataFrame, output: str | Path, hours: int = 336):
    """Realised/forecast HUE profiles, dynamic RTP, and simulated outages."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    agg = data.groupby("timestamp", sort=True).agg(
        load=("load_kwh", "sum"), pv=("pv_kwh", "sum"),
        load_forecast=("load_forecast_kwh", "sum"),
        pv_forecast=("pv_forecast_kwh", "sum"),
        rtp=("rtp_cad_per_kwh", "first"),
        rtp_forecast=("rtp_forecast_cad_per_kwh", "first"),
        grid_available=("grid_available", "first"),
    ).reset_index()
    # Select a window containing an outage when possible.
    outage_idx = np.flatnonzero(agg["grid_available"].to_numpy() == 0)
    start = max(int(outage_idx[0]) - 72, 0) if len(outage_idx) else 0
    d = agg.iloc[start:start + min(hours, len(agg))]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(d["timestamp"], d["load"], label="Realised load", color="#1f77b4")
    axes[0].plot(d["timestamp"], d["load_forecast"], label="1-h decision forecast",
                 color="#1f77b4", linestyle="--", alpha=.75)
    axes[0].plot(d["timestamp"], d["pv"], label="Realised PV", color="#e6a700")
    axes[0].plot(d["timestamp"], d["pv_forecast"], label="PV forecast",
                 color="#e6a700", linestyle="--", alpha=.75)
    axes[0].set_ylabel("Community energy (kWh/h)"); axes[0].legend(ncol=2)
    axes[0].set_title("HUE Community: Realised and Forecast Energy")
    axes[1].plot(d["timestamp"], d["rtp"], label="Realised RTP", color="#b22222")
    axes[1].plot(d["timestamp"], d["rtp_forecast"], label="Forecast RTP",
                 color="#b22222", linestyle="--", alpha=.75)
    axes[1].set_ylabel("CAD/kWh"); axes[1].legend(); axes[1].set_title("Dynamic Grid-Responsive RTP")
    axes[2].step(d["timestamp"], d["grid_available"], where="post", color="#333333")
    axes[2].fill_between(d["timestamp"], 0, 1 - d["grid_available"], step="post",
                         color="#d62728", alpha=.35, label="Simulated outage")
    axes[2].set_ylim(-.05, 1.05); axes[2].set_yticks([0, 1]); axes[2].set_yticklabels(["Outage", "Available"])
    axes[2].set_title("Reproducible Grid-Availability Scenario"); axes[2].legend()
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output / "00_hue_data_rtp_outage_overview.png", dpi=200); plt.close(fig)

    energy = data.groupby("prosumer_id").agg(
        annualized_load_kwh=("load_kwh", "sum"),
        annualized_pv_kwh=("pv_kwh", "sum"),
    )
    scale = 8760 / data["timestamp"].nunique()
    energy *= scale
    ax = energy.plot.bar(figsize=(11, 5), color=["#1f77b4", "#e6a700"])
    ax.set_title("Annualized HUE Load and Assigned PV by Prosumer")
    ax.set_ylabel("Energy (kWh/year)"); ax.set_xlabel("Prosumer"); ax.grid(axis="y", alpha=.2)
    plt.tight_layout(); plt.savefig(output / "00_hue_annualized_energy_by_prosumer.png", dpi=200); plt.close()


def plot_disaggregated_boxplots(disaggregated: pd.DataFrame, frequency: str, output: str | Path,
                                  filename_prefix: str = "07"):
    """All client-requested disaggregated metrics at one time resolution."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    d = disaggregated[disaggregated["frequency"] == frequency].copy()
    if d.empty:
        return
    pids = sorted(d["prosumer_id"].astype(str).unique(), key=lambda p: int(p[1:]) if p.startswith("P") and p[1:].isdigit() else p)
    d["prosumer_id"] = pd.Categorical(d["prosumer_id"], categories=pids, ordered=True)

    fig, axes = plt.subplots(1, 3, figsize=(max(12, 1.2 * len(pids)), 4.5))
    rel_cols = ["SAIDI_hours", "SAIFI_interruptions", "CAIFI_interruptions_affected_customer"]
    for ax, col in zip(axes, rel_cols):
        d.boxplot(column=col, by="prosumer_id", ax=ax, grid=False, positions=range(len(pids)))
        title, ylabel = RELIABILITY_LABELS[col]
        ax.set_xticks(range(len(pids))); ax.set_xticklabels(pids, rotation=45)
        ax.set_title(title); ax.set_xlabel("Prosumer"); ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.2)
        if np.isclose(d[col].fillna(0).abs().max(), 0):
            ax.set_ylim(0, 1)
            ax.text(.5, .5, "No interruption events",
                    ha="center", va="center", transform=ax.transAxes, color="#555555")
    fig.texts.clear(); fig.suptitle(f"Disaggregated Reliability ({frequency})")
    fig.tight_layout(); fig.savefig(output / f"{filename_prefix}_a_reliability_{frequency}.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(max(12, 1.2 * len(pids)), 4.5))
    for ax, col in zip(axes, ["net_cost_cad", "net_cost_savings_cad", "peak_load_reduction_kw"]):
        d.boxplot(column=col, by="prosumer_id", ax=ax, grid=False, positions=range(len(pids)))
        ax.set_xticks(range(len(pids))); ax.set_xticklabels(pids, rotation=45); ax.set_title(col); ax.set_xlabel("Prosumer")
    fig.texts.clear(); fig.suptitle(f"Disaggregated Cost/Savings/Peak Reduction ({frequency})")
    fig.tight_layout(); fig.savefig(output / f"{filename_prefix}_b_cost_peak_{frequency}.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(pids)), 4.5))
    d.boxplot(column="pv_utilization", by="prosumer_id", ax=ax, grid=False, positions=range(len(pids)))
    ax.set_xticks(range(len(pids))); ax.set_xticklabels(pids, rotation=45)
    fig.texts.clear(); ax.set_title(f"Disaggregated PV Utilization ({frequency})"); ax.set_xlabel("Prosumer")
    fig.tight_layout(); fig.savefig(output / f"{filename_prefix}_c_pv_utilization_{frequency}.png", dpi=180); plt.close(fig)


def plot_variation(variation: pd.DataFrame, frequency: str, output: str | Path, filename_prefix: str = "08"):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    d = variation[variation["frequency"] == frequency].sort_values("period")
    if d.empty:
        return
    metrics = [
        "net_cost_cad_std", "net_cost_savings_cad_std", "pv_utilization_std",
        "SAIDI_hours_std", "SAIFI_interruptions_std",
        "CAIFI_interruptions_affected_customer_std", "peak_load_reduction_kw_std",
    ]
    metrics = [m for m in metrics if m in d]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(11, max(8, 2.0*len(metrics))), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, metrics):
        ax.plot(d["period"], d[col], marker="o")
        ax.set_ylabel(col.replace("_std", "")); ax.grid(alpha=.25)
    axes[-1].tick_params(axis="x", rotation=45)
    fig.suptitle(f"Cross-Prosumer Metric Variation ({frequency})")
    fig.tight_layout(); fig.savefig(output / f"{filename_prefix}_variation_{frequency}.png", dpi=180); plt.close(fig)


def plot_equity_sweep(equity_summary: pd.DataFrame, output: str | Path, filename_prefix: str = "09"):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    allocation_methods = equity_summary.get("allocation_method", pd.Series(["proportional"])).unique()
    for allocation_method in allocation_methods:
        base = equity_summary[equity_summary.get("allocation_method", allocation_method) == allocation_method] if "allocation_method" in equity_summary else equity_summary
        controllers = base["controller"].unique()
        fig, axes = plt.subplots(len(controllers), 2, figsize=(11, 3.2 * len(controllers)), squeeze=False)
        for row, controller in enumerate(controllers):
            d = base[base["controller"] == controller]
            for col, metric in enumerate(["atkinson_index", "equity_adjusted_welfare"]):
                pivot = d.pivot_table(index="kappa", columns="atkinson_epsilon", values=metric)
                ax = axes[row][col]
                im = ax.imshow(pivot.values, aspect="auto")
                ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
                ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
                ax.set_xlabel("Atkinson epsilon"); ax.set_ylabel("kappa")
                ax.set_title(f"{controller}: {metric}")
                fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"18-Combination Efficiency-Equity Sweep: {allocation_method}")
        fig.tight_layout(); fig.savefig(output / f"{filename_prefix}_equity_sweep_{_safe_name(allocation_method)}.png", dpi=180); plt.close(fig)


def plot_atkinson_reward_combinations(reward_curves: dict, output: str | Path,
                                      tariff: str = "tou", allocation_method: str = "proportional"):
    """Reproduce the client's exact 18-combination reward figure family.

    For every epsilon-kappa pair this writes two figures, matching the supplied
    ``PlotRewardFunction.py`` convention:

    * cumulative reward over the complete evaluated horizon;
    * hourly reward over the first 48 hours.

    The four controller/replay cases are overlaid while the allocation method is
    held fixed, so controller effects remain readable. With the requested 3 x 6
    grid the function always creates 36 figures.
    """
    output = Path(output)
    cumulative_dir = output / "cumulative"
    hourly_dir = output / "hourly_first_48h"
    cumulative_dir.mkdir(parents=True, exist_ok=True)
    hourly_dir.mkdir(parents=True, exist_ok=True)
    pairs = sorted({(key[0], key[1]) for key in reward_curves})
    keyed_by_allocation = bool(reward_curves and len(next(iter(reward_curves))) == 4)

    def curve(epsilon, kappa, controller):
        key = ((epsilon, kappa, controller, allocation_method)
               if keyed_by_allocation else (epsilon, kappa, controller))
        return reward_curves.get(key)

    preferred = ["rule_based", "mpc", "d3qn_uniform_replay", "d3qn_per"]
    labels = {
        "rule_based": "Rule-Based",
        "mpc": "MPC",
        "d3qn_uniform_replay": "D3QN (Uniform Replay)",
        "d3qn_per": "D3QN + PER",
    }
    for epsilon, kappa in pairs:
        available = {key[2] for key in reward_curves
                     if key[:2] == (epsilon, kappa)
                     and (not keyed_by_allocation or key[3] == allocation_method)}
        controllers = [c for c in preferred if c in available]
        controllers += sorted(available - set(controllers))
        stem = f"epsilon_{str(epsilon).replace('.', 'p')}_kappa_{str(kappa).replace('.', 'p')}"

        fig, ax = plt.subplots(figsize=(10, 5.5))
        for controller in controllers:
            d = curve(epsilon, kappa, controller)
            ax.plot(np.arange(len(d)), d["reward"].to_numpy().cumsum(),
                    label=labels.get(controller, controller))
        ax.set_title(f"Cumulative Reward - epsilon={epsilon}, kappa={kappa}")
        ax.set_xlabel("Hour"); ax.set_ylabel("Cumulative reward")
        ax.grid(alpha=.25); ax.legend()
        fig.text(.5, .01, f"Tariff: {tariff} | Allocation: {allocation_method}",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .035, 1, 1))
        fig.savefig(cumulative_dir / f"cumulative_{stem}.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5.5))
        for controller in controllers:
            d = curve(epsilon, kappa, controller).iloc[:48]
            ax.plot(np.arange(len(d)), d["reward"].to_numpy(),
                    label=labels.get(controller, controller))
        ax.set_title(f"Hourly Reward (First 48 h) - epsilon={epsilon}, kappa={kappa}")
        ax.set_xlabel("Hour"); ax.set_ylabel("Reward")
        ax.grid(alpha=.25); ax.legend()
        fig.text(.5, .01, f"Tariff: {tariff} | Allocation: {allocation_method}",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .035, 1, 1))
        fig.savefig(hourly_dir / f"hourly_first48_{stem}.png", dpi=180)
        plt.close(fig)


def plot_atkinson_allocation_combinations(reward_curves: dict, output: str | Path,
                                          tariff: str = "tou",
                                          controller: str = "d3qn_per"):
    """Reproduce the original client's 36 reward plots by allocation method.

    The supplied ``PlotRewardFunction.py`` overlays proportional, priority and
    contribution allocation for each of the 18 epsilon-kappa pairs. Here the
    supervisory controller is held fixed so only the allocation effect changes.
    """
    output = Path(output)
    cumulative_dir = output / "cumulative"
    hourly_dir = output / "hourly_first_48h"
    cumulative_dir.mkdir(parents=True, exist_ok=True)
    hourly_dir.mkdir(parents=True, exist_ok=True)
    pairs = sorted({(key[0], key[1]) for key in reward_curves if len(key) == 4})
    methods = ("proportional", "priority", "contribution")
    labels = {
        "proportional": "Proportional Allocation",
        "priority": "Priority-Based Allocation",
        "contribution": "Contribution-Based Allocation",
    }
    for epsilon, kappa in pairs:
        available = [method for method in methods
                     if (epsilon, kappa, controller, method) in reward_curves]
        if not available:
            continue
        stem = f"epsilon_{str(epsilon).replace('.', 'p')}_kappa_{str(kappa).replace('.', 'p')}"

        fig, ax = plt.subplots(figsize=(10, 5.5))
        for method in available:
            d = reward_curves[(epsilon, kappa, controller, method)]
            ax.plot(np.arange(len(d)), d["reward"].to_numpy().cumsum(),
                    label=labels[method])
        ax.set_title(f"Cumulative Rewards - epsilon={epsilon}, kappa={kappa}")
        ax.set_xlabel("Hour"); ax.set_ylabel("Cumulative reward")
        ax.grid(alpha=.25); ax.legend()
        fig.text(.5, .01, f"Tariff: {tariff} | Controller: {controller}",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .035, 1, 1))
        fig.savefig(cumulative_dir / f"cumulative_{stem}.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5.5))
        for method in available:
            d = reward_curves[(epsilon, kappa, controller, method)].iloc[:48]
            ax.plot(np.arange(len(d)), d["reward"].to_numpy(),
                    label=labels[method])
        ax.set_title(f"Hourly Rewards (First 48 h) - epsilon={epsilon}, kappa={kappa}")
        ax.set_xlabel("Hour"); ax.set_ylabel("Reward")
        ax.grid(alpha=.25); ax.legend()
        fig.text(.5, .01, f"Tariff: {tariff} | Controller: {controller}",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .035, 1, 1))
        fig.savefig(hourly_dir / f"hourly_first48_{stem}.png", dpi=180)
        plt.close(fig)


def plot_annual_tariffs(temporal: pd.DataFrame, output: str | Path):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    annual = temporal[temporal["frequency"] == "annual"]
    if annual.empty: return
    table = annual.pivot_table(index="tariff", columns="controller", values="net_cost_cad", aggfunc="sum")
    ax = table.plot.bar(figsize=(11, 5)); ax.set_ylabel("Net cost (CAD)"); ax.set_title("Annual Net Cost by Tariff and Controller")
    ax.grid(axis="y", alpha=.25); plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig(output / "06_tariff_controller_net_cost.png", dpi=180); plt.close()


def plot_temporal_metric_comparisons(temporal: pd.DataFrame, output: str | Path):
    """Monthly, seasonal and annual plots for every requested aggregate KPI."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    metrics = [
        "net_cost_cad", "grid_bill_savings_cad", "net_cost_savings_cad",
        "pv_utilization", "peak_load_reduction_kw",
        "SAIDI_hours", "SAIFI_interruptions", "CAIFI_interruptions_affected_customer",
        "cumulative_reward",
    ]
    for frequency in ("monthly", "seasonal", "annual"):
        d = temporal[temporal["frequency"] == frequency]
        if d.empty: continue
        for metric in metrics:
            if metric not in d: continue
            for tariff in d["tariff"].unique():
                x = d[d["tariff"] == tariff]
                table = x.pivot_table(index="period", columns="controller", values=metric, aggfunc="mean")
                if table.empty: continue
                if frequency == "monthly":
                    ax = table.plot(kind="line", marker="o", figsize=(11, 5))
                else:
                    ax = table.plot(kind="bar", figsize=(11, 5))
                ax.set_title(f"{metric.replace('_', ' ').title()} - {tariff} - {frequency}")
                ax.set_ylabel(metric); ax.set_xlabel("Period"); ax.grid(axis="y", alpha=.25)
                plt.xticks(rotation=45 if len(table) > 4 else 0); plt.tight_layout()
                name = f"10_{frequency}_{_safe_name(tariff)}_{metric}.png"
                plt.savefig(output / name, dpi=180); plt.close()


def plot_allocation_method_comparison(summary: pd.DataFrame, output: str | Path):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    metrics = ["net_cost_cad", "pv_utilization", "cumulative_reward", "CAIFI_interruptions_affected_customer"]
    for tariff in summary["tariff"].unique():
        d = summary[summary["tariff"] == tariff]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, metric in zip(axes.flat, metrics):
            table = d.pivot_table(index="allocation_method", columns="controller", values=metric, aggfunc="mean")
            table.plot.bar(ax=ax); ax.set_title(metric.replace("_", " ").title()); ax.set_xlabel("Allocation method")
            ax.tick_params(axis="x", rotation=0)
        fig.suptitle(f"Original Allocation Algorithms vs Controllers - {tariff}")
        fig.tight_layout(); fig.savefig(output / f"11_allocation_method_comparison_{_safe_name(tariff)}.png", dpi=180); plt.close(fig)
