"""Fail loudly when a full client run is missing required evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _count(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("--allow-preliminary", action="store_true")
    args = parser.parse_args()
    root = Path(args.results)
    tables = root / "tables"
    failures = []

    required = [
        root / "RUN_MANIFEST.csv", root / "RESULTS_SUMMARY.md",
        root / "VALIDATION_CHECKS.json", root / "models/d3qn_per_model.npz",
        root / "models/d3qn_uniform_replay_model.npz",
        tables / "d3qn_training_history.csv", tables / "d3qn_validation_summary.csv",
        tables / "d3qn_seed_validation_summary.csv", tables / "d3qn_seed_test_summary.csv",
        tables / "d3qn_per_ablation_statistics.csv", tables / "controller_tariff_summary.csv",
        tables / "forecast_accuracy_test.csv", tables / "temporal_metrics.csv",
        tables / "disaggregated_prosumer_metrics.csv", tables / "variation_metrics.csv",
        tables / "switching_window_summary.csv", tables / "allocation_method_comparison.csv",
        tables / "equity_epsilon_kappa_sweep.csv", tables / "battery_npv_replacement_summary.csv",
        tables / "hue_outage_events.csv", tables / "hue_rtp_diagnostics.csv",
        tables / "hue_augmentation_config.csv", tables / "hue_augmentation_validation.csv",
        tables / "hue_augmentation_yearly_energy.csv", tables / "outage_sensitivity_summary.csv",
    ]
    failures += [f"missing {p.relative_to(root)}" for p in required if not p.exists()]
    if failures:
        raise SystemExit("OUTPUT AUDIT FAILED\n- " + "\n- ".join(failures))

    manifest = pd.read_csv(root / "RUN_MANIFEST.csv").iloc[0]
    split = pd.read_csv(tables / "data_split_manifest.csv")
    switching = pd.read_csv(tables / "switching_window_summary.csv")
    allocation = pd.read_csv(tables / "allocation_method_comparison.csv")
    equity = pd.read_csv(tables / "equity_epsilon_kappa_sweep.csv")
    controller_tariff = pd.read_csv(tables / "controller_tariff_summary.csv")
    outage = pd.read_csv(tables / "outage_sensitivity_summary.csv")
    seed_validation = pd.read_csv(tables / "d3qn_seed_validation_summary.csv")
    yearly = pd.read_csv(tables / "hue_augmentation_yearly_energy.csv")
    validation = json.loads((root / "VALIDATION_CHECKS.json").read_text())
    full = str(manifest["full_mode"]).lower() == "true"

    if "HUE" not in str(manifest["data_source"]):
        failures.append("data source is not HUE")
    if int(manifest.get("augmentation_years", 0)) != 20 and not args.allow_preliminary:
        failures.append("augmentation does not span exactly 20 calendar years")
    if controller_tariff[["controller", "tariff"]].drop_duplicates().shape[0] != 20:
        failures.append("controller/tariff evaluation is not the complete 4 x 5 grid")
    if set(controller_tariff["tariff"]) != {"flat", "tiered", "tou", "rtp", "balanced"}:
        failures.append("pricing cases do not include flat/tiered/ToU/RTP/balanced")
    if seed_validation[["seed", "replay"]].drop_duplicates().shape[0] < 6:
        failures.append("PER/uniform ablation does not cover three matched seeds each")
    if len(outage) != 12 or set(outage.get("outage_scenario", [])) != {"normal", "moderate", "stress"}:
        failures.append("outage sensitivity is not the complete 3 scenarios x 4 controllers")
    expected_year_rows = int(manifest.get("augmentation_years", 0)) * int(manifest["n_prosumers"])
    if expected_year_rows and len(yearly) != expected_year_rows:
        failures.append(f"yearly augmented table has {len(yearly)} rows, expected {expected_year_rows}")
    if not args.allow_preliminary:
        if not full:
            failures.append("full_mode is false")
        expected_test = int(split.loc[split["split"] == "test", "hours"].iat[0])
        if expected_test not in (8760, 8784, 17520, 17544):
            failures.append("test split is not one or two complete calendar years")
        if int(manifest["test_hours"]) != expected_test:
            failures.append("evaluated test horizon does not equal the reserved complete-year split")
        if switching["switch_interval_hours"].nunique() != 5:
            failures.append("switching sweep does not contain five intervals")
        if len(equity[["atkinson_epsilon", "kappa"]].drop_duplicates()) != 18:
            failures.append("equity sweep does not contain 18 epsilon-kappa pairs")
        if len(equity) != 216:
            failures.append(f"equity sweep has {len(equity)} rows, expected 216")
        if len(allocation) != 24:
            failures.append(f"allocation comparison has {len(allocation)} rows, expected 24")
        families = {
            "original cumulative": root / "figures/atkinson_combinations/original_allocation_comparison_d3qn_per/cumulative",
            "original hourly": root / "figures/atkinson_combinations/original_allocation_comparison_d3qn_per/hourly_first_48h",
            "controller cumulative": root / "figures/atkinson_combinations/controller_comparison_proportional/cumulative",
            "controller hourly": root / "figures/atkinson_combinations/controller_comparison_proportional/hourly_first_48h",
        }
        for label, folder in families.items():
            count = _count(folder, "*.png")
            if count != 18:
                failures.append(f"{label} plot count is {count}, expected 18")

    for key in ("energy_balance_pass", "soc_limits_pass",
                "no_simultaneous_charge_discharge_pass"):
        if not validation.get(key, False):
            failures.append(f"physical validation failed: {key}")

    if failures:
        raise SystemExit("OUTPUT AUDIT FAILED\n- " + "\n- ".join(failures))

    print("OUTPUT AUDIT PASSED")
    print(f"Data source: {manifest['data_source']}")
    print(f"Evaluated test hours: {int(manifest['test_hours'])}")
    print(f"Switching intervals: {switching['switch_interval_hours'].nunique()}")
    print(f"Equity rows / unique pairs: {len(equity)} / "
          f"{len(equity[['atkinson_epsilon', 'kappa']].drop_duplicates())}")
    print(f"Allocation comparison rows: {len(allocation)}")
    print(f"Controller/tariff rows: {len(controller_tariff)}")
    print(f"Outage sensitivity rows: {len(outage)}")
    print(f"RL seed/replay validation rows: {len(seed_validation)}")
    print(f"Maximum energy-balance error: {validation['maximum_energy_balance_error_kwh']:.3e} kWh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
