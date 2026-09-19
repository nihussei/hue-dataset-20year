import numpy as np
import pandas as pd

from microgrid_opt.config import ExperimentConfig, TariffConfig
from microgrid_opt.controllers import MPCController, RuleBasedController
from microgrid_opt.core import Battery, MicrogridEnvironment, allocate
from microgrid_opt.data import generate_synthetic_community, validate_long_data
from microgrid_opt.equity import atkinson_index, ede
from microgrid_opt.evaluation import run_controller, run_equity_sweep
from microgrid_opt.metrics import disaggregated_metrics, equity_summary
from microgrid_opt.tariffs import tariff_prices


def test_battery_soc_constraints():
    cfg = ExperimentConfig().battery
    b = Battery(cfg)
    for _ in range(100): b.dispatch(-1e6, surplus_kwh=1e6, deficit_kwh=0)
    assert b.soc <= cfg.max_soc + 1e-9
    for _ in range(100): b.dispatch(1e6, surplus_kwh=0, deficit_kwh=1e6)
    assert b.soc >= cfg.min_soc - 1e-9


def test_allocation_is_physical():
    demand = np.array([1.0, 2.0, 7.0])
    got = allocate(demand, 4.0, "priority", weights=np.array([3.0, 2.0, 1.0]))
    assert np.all(got >= 0) and np.all(got <= demand + 1e-9)
    assert np.isclose(got.sum(), 4.0)


def test_tier_resets_monthly():
    ts = pd.date_range("2025-01-01", "2025-02-28 23:00", freq="h")
    imports = np.full(len(ts), 2.0)
    prices = tariff_prices(ts, "tiered", TariffConfig(), imports_kwh=imports)
    feb = np.flatnonzero(ts.month == 2)[0]
    assert prices[feb] == TariffConfig().tier1_cad_per_kwh


def test_rule_and_mpc_run():
    data = generate_synthetic_community(3, 2)
    cfg = ExperimentConfig()
    for controller in (RuleBasedController(), MPCController(6)):
        env = MicrogridEnvironment(data, cfg, "tou")
        result = run_controller(env, controller)
        assert len(result) == 48
        assert result["soc"].between(cfg.battery.min_soc-1e-9, cfg.battery.max_soc+1e-9).all()


def test_ede_equals_mean_at_perfect_equality():
    # Atkinson EDE of a constant vector must equal that constant for any epsilon.
    values = np.full(5, 0.7)
    for epsilon in (0.5, 1.0, 2.0):
        assert np.isclose(ede(values, epsilon), 0.7, atol=1e-6)
        assert np.isclose(atkinson_index(values, epsilon), 0.0, atol=1e-6)


def test_ede_penalizes_dispersion_more_as_epsilon_rises():
    values = np.array([0.9, 0.9, 0.9, 0.1])  # one disadvantaged prosumer
    ede_low = ede(values, 0.5)
    ede_high = ede(values, 2.0)
    assert ede_high < ede_low  # higher epsilon -> more equity-averse -> lower EDE
    assert atkinson_index(values, 2.0) > atkinson_index(values, 0.5)


def test_environment_produces_per_prosumer_utility_and_welfare_reward():
    data = generate_synthetic_community(4, 3)
    cfg = ExperimentConfig()
    cfg.reward.atkinson_epsilon = 2.0
    cfg.reward.kappa = 0.1
    env = MicrogridEnvironment(data, cfg, "tou")
    result = run_controller(env, RuleBasedController())
    for pid in env.ids:
        assert f"utility_{pid}" in result.columns
    assert {"welfare_cost_ede", "welfare_pv_ede", "welfare_reliability_ede"}.issubset(result.columns)
    summary = equity_summary(result, env.ids, cfg.reward.atkinson_epsilon, cfg.reward.kappa)
    assert set(["mean_prosumer_reward", "prosumer_reward_variance", "atkinson_index",
                "equity_adjusted_welfare"]).issubset(summary)
    assert np.isclose(summary["cumulative_reward"], result["reward"].sum())


def test_disaggregated_metrics_have_one_row_per_prosumer_per_period():
    data = generate_synthetic_community(3, 35)  # spans >1 month
    cfg = ExperimentConfig()
    env = MicrogridEnvironment(data, cfg, "tou")
    result = run_controller(env, RuleBasedController())
    dm = disaggregated_metrics(result, env.ids, "monthly")
    assert set(dm["prosumer_id"].unique()) == set(env.ids)
    assert dm.groupby("period")["prosumer_id"].nunique().eq(len(env.ids)).all()


def test_equity_sweep_covers_full_grid():
    data = generate_synthetic_community(3, 2)
    cfg = ExperimentConfig()
    controllers = [RuleBasedController(), MPCController(6)]
    swept = run_equity_sweep(data, cfg, controllers, tariff="tou",
                              epsilon_grid=(0.5, 1.0), kappa_grid=(0.01, 0.1))
    assert len(swept) == 2 * 2 * len(controllers)  # epsilon x kappa x controllers
    assert set(swept["controller"]) == {c.name for c in controllers}



def test_policy_switching_separates_switch_interval_from_next_hour_evaluation():
    from microgrid_opt.evaluation import run_policy_switching
    data = generate_synthetic_community(3, 1)
    cfg = ExperimentConfig()
    env = MicrogridEnvironment(data, cfg, "tou")
    controllers = [RuleBasedController(), MPCController(3)]
    _, decisions = run_policy_switching(env, controllers, switch_interval_hours=6, evaluation_horizon_hours=1)
    assert set(decisions["switch_interval_hours"]) == {6}
    assert set(decisions["evaluation_horizon_hours"]) == {1}
    assert len(decisions) == 4  # t = 0, 6, 12, 18


def test_policy_candidate_rollout_uses_forecast_not_realized_future():
    from microgrid_opt.evaluation import _rollout_value_forecast
    data = generate_synthetic_community(2, 1)
    data.loc[:, "load_kwh"] = 100.0
    data.loc[:, "load_forecast_kwh"] = 1.0
    cfg = ExperimentConfig()
    env = MicrogridEnvironment(data, cfg, "tou")

    class CaptureController:
        name = "capture"
        def __init__(self): self.seen = []
        def act(self, obs, rollout_env, explore=False):
            self.seen.append(float(rollout_env.load[rollout_env.t].sum()))
            return 0.0

    c = CaptureController()
    _rollout_value_forecast(env, c, 1)
    assert np.isclose(c.seen[0], 2.0)  # 2 prosumers x 1 kWh forecast, not 200 kWh realized


def test_allocation_method_sweep_covers_three_original_algorithms():
    from microgrid_opt.evaluation import run_allocation_method_sweep
    data = generate_synthetic_community(3, 1)
    cfg = ExperimentConfig()
    controllers = [RuleBasedController(), MPCController(3)]
    summary, _ = run_allocation_method_sweep(data, cfg, controllers, tariffs=("tou",),
                                             allocation_methods=("proportional", "priority", "contribution"))
    assert set(summary["allocation_method"]) == {"proportional", "priority", "contribution"}
    assert len(summary) == 3 * len(controllers)


def test_community_caifi_is_interruptions_per_affected_customer():
    from microgrid_opt.metrics import community_reliability
    data = generate_synthetic_community(2, 1)
    cfg = ExperimentConfig()
    env = MicrogridEnvironment(data, cfg, "tou")
    result = run_controller(env, RuleBasedController())
    # Build a controlled interruption pattern: P1 has two events, P2 unaffected.
    result["unserved_P1"] = 0.0
    result["unserved_P2"] = 0.0
    result.loc[[1, 2, 7], "unserved_P1"] = 1.0  # events [1-2] and [7]
    rel = community_reliability(result, ["P1", "P2"])
    assert np.isclose(rel["SAIFI_interruptions"], 1.0)  # 2 events / 2 customers
    assert np.isclose(rel["CAIFI_interruptions_affected_customer"], 2.0)  # 2 / 1 affected


def test_full_client_equity_grid_is_exactly_18_combinations():
    from microgrid_opt.equity import EPSILON_GRID, KAPPA_GRID
    assert EPSILON_GRID == (0.5, 1.0, 2.0)
    assert KAPPA_GRID == (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
    assert len(EPSILON_GRID) * len(KAPPA_GRID) == 18


def test_d3qn_model_save_load_roundtrip(tmp_path):
    from microgrid_opt.controllers import D3QNController
    agent = D3QNController(state_dim=10, seed=7)
    x = np.linspace(-1, 1, 10)
    q_before = agent.online.q(x[None, :])
    path = tmp_path / "agent.npz"
    agent.save(str(path))
    loaded = D3QNController.load(str(path))
    assert np.allclose(q_before, loaded.online.q(x[None, :]))


def test_default_2026_bc_hydro_rate_parameters():
    t = TariffConfig()
    assert np.isclose(t.flat_cad_per_kwh, 0.1270)
    assert np.isclose(t.flat_basic_charge_cad_per_day, 0.2500)
    assert np.isclose(t.tier1_cad_per_kwh, 0.1187)
    assert np.isclose(t.tier2_cad_per_kwh, 0.1408)
    assert np.isclose(t.tiered_basic_charge_cad_per_day, 0.2344)
    assert np.isclose(t.tou_discount_cad_per_kwh, 0.05)
    assert np.isclose(t.tou_surcharge_cad_per_kwh, 0.05)


def test_outage_dispatch_serves_critical_load_before_noncritical_load():
    from microgrid_opt.evaluation import run_controller
    ts = pd.Timestamp("2025-01-01 18:00")
    data = pd.DataFrame({
        "timestamp": [ts, ts],
        "prosumer_id": ["P1", "P2"],
        "load_kwh": [1.0, 1.0],
        "pv_kwh": [0.0, 0.0],
        "load_forecast_kwh": [1.0, 1.0],
        "pv_forecast_kwh": [0.0, 0.0],
        "critical_fraction": [1.0, 0.0],
        "grid_available": [0, 0],
        "rtp_cad_per_kwh": [0.10, 0.10],
        "rtp_forecast_cad_per_kwh": [0.10, 0.10],
    })
    cfg = ExperimentConfig()
    cfg.battery.max_discharge_kw = 1.0
    cfg.battery.capacity_kwh = 10.0
    cfg.battery.initial_soc = 0.8
    env = MicrogridEnvironment(data, cfg, "tou")
    result = run_controller(env, RuleBasedController())
    assert np.isclose(result.loc[0, "discharge_P1"], 1.0)
    assert np.isclose(result.loc[0, "discharge_P2"], 0.0)
    assert np.isclose(result.loc[0, "unserved_P1"], 0.0)


def test_prosumer_net_cost_includes_degradation_share():
    from microgrid_opt.metrics import prosumer_metrics
    from microgrid_opt.evaluation import run_controller
    data = generate_synthetic_community(3, 2)
    cfg = ExperimentConfig()
    env = MicrogridEnvironment(data, cfg, "tou")
    result = run_controller(env, RuleBasedController())
    pm = prosumer_metrics(result, env.ids).set_index("prosumer_id")
    for pid in env.ids:
        expected_bill = float(result[f"cost_{pid}"].sum())
        expected_deg = float(result[f"degradation_cost_{pid}"].sum())
        assert np.isclose(pm.loc[pid, "energy_bill_cad"], expected_bill)
        assert np.isclose(pm.loc[pid, "battery_degradation_cost_cad"], expected_deg)
        assert np.isclose(pm.loc[pid, "net_cost_cad"], expected_bill + expected_deg)
        assert np.isclose(pm.loc[pid, "net_cost_savings_cad"],
                          pm.loc[pid, "grid_bill_savings_cad"] - expected_deg)


def test_input_requires_identical_prosumer_set_and_hourly_continuity():
    base = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:00",
                                      "2026-01-01 02:00", "2026-01-01 02:00"]),
        "prosumer_id": ["P1", "P2", "P1", "P3"],
        "load_kwh": 1.0,
        "pv_kwh": 0.0,
    })
    try:
        validate_long_data(base)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid prosumer membership/hourly gap was accepted")


def test_forecast_view_rebuilds_rtp_prices_without_realized_price_leakage():
    data = generate_synthetic_community(2, 1)
    data["rtp_cad_per_kwh"] = 0.50
    data["rtp_forecast_cad_per_kwh"] = 0.10
    env = MicrogridEnvironment(data, ExperimentConfig(), "rtp")
    forecast = env.forecast_view()
    assert np.allclose(env.price_matrix, 0.50)
    assert np.allclose(forecast.price_matrix, 0.10)


def test_no_bess_baseline_has_independent_tier_accumulator():
    data = generate_synthetic_community(1, 2)
    data["load_kwh"] = 1.0
    data["load_forecast_kwh"] = 1.0
    data["pv_kwh"] = 0.0
    data["pv_forecast_kwh"] = 0.0
    cfg = ExperimentConfig()
    cfg.tariff.tier1_daily_kwh = 2.0 / 31.0  # 2 kWh January threshold
    env = MicrogridEnvironment(data, cfg, "tiered")
    result = run_controller(env, RuleBasedController(high_price_threshold=1.0))
    baseline = result["baseline_cost_P1"] - result["basic_charge_P1"]
    assert np.isclose(baseline.iloc[0], cfg.tariff.tier1_cad_per_kwh)
    assert np.isclose(baseline.iloc[1], cfg.tariff.tier1_cad_per_kwh)
    assert np.isclose(baseline.iloc[2], cfg.tariff.tier2_cad_per_kwh)


def test_runtime_energy_balance_including_outages():
    from microgrid_opt.validation import validate_hourly_result
    data = generate_synthetic_community(4, 25)
    cfg = ExperimentConfig()
    env = MicrogridEnvironment(data, cfg, "tou")
    result = run_controller(env, RuleBasedController())
    checks = validate_hourly_result(result, cfg)
    assert checks["energy_balance_pass"]
    assert checks["soc_limits_pass"]
    assert checks["no_simultaneous_charge_discharge_pass"]


def test_exact_36_client_atkinson_reward_plots(tmp_path):
    from microgrid_opt.equity import EPSILON_GRID, KAPPA_GRID
    from microgrid_opt.plots import plot_atkinson_reward_combinations
    controllers = ("rule_based", "mpc", "d3qn_uniform_replay", "d3qn_per")
    ts = pd.date_range("2026-01-01", periods=48, freq="h")
    curves = {}
    for epsilon in EPSILON_GRID:
        for kappa in KAPPA_GRID:
            for j, controller in enumerate(controllers):
                curves[(epsilon, kappa, controller)] = pd.DataFrame({
                    "timestamp": ts,
                    "reward": np.linspace(0.1 + j * 0.01, 0.2 + j * 0.01, len(ts)),
                })
    plot_atkinson_reward_combinations(curves, tmp_path)
    assert len(list((tmp_path / "cumulative").glob("*.png"))) == 18
    assert len(list((tmp_path / "hourly_first_48h").glob("*.png"))) == 18


def test_exact_36_original_allocation_atkinson_reward_plots(tmp_path):
    from microgrid_opt.equity import EPSILON_GRID, KAPPA_GRID
    from microgrid_opt.plots import plot_atkinson_allocation_combinations
    methods = ("proportional", "priority", "contribution")
    ts = pd.date_range("2026-01-01", periods=48, freq="h")
    curves = {}
    for epsilon in EPSILON_GRID:
        for kappa in KAPPA_GRID:
            for j, method in enumerate(methods):
                curves[(epsilon, kappa, "d3qn_per", method)] = pd.DataFrame({
                    "timestamp": ts,
                    "reward": np.linspace(0.1 + j * 0.01, 0.2 + j * 0.01, len(ts)),
                })
    plot_atkinson_allocation_combinations(curves, tmp_path)
    assert len(list((tmp_path / "cumulative").glob("*.png"))) == 18
    assert len(list((tmp_path / "hourly_first_48h").glob("*.png"))) == 18


def test_battery_npv_projection_ports_client_replacement_economics():
    from microgrid_opt.financial import battery_financial_projection, replacement_npv
    client_case = replacement_npv(5.0, 20.0, 16000.0, 0.035)
    assert client_case["replacement_count"] == 4
    assert client_case["replacement_present_value_cad"] > 0
    data = generate_synthetic_community(3, 2)
    cfg = ExperimentConfig()
    result = run_controller(MicrogridEnvironment(data, cfg, "tou"), RuleBasedController())
    projection = battery_financial_projection(result, cfg.battery)
    assert projection["estimated_operating_life_years"] > 0
    assert "incremental_replacement_npv_cad" in projection


def test_calendar_year_is_reserved_as_unseen_test_set():
    from microgrid_opt.data import chronological_split
    timestamps = pd.date_range("2015-02-28", "2018-01-29 23:00", freq="h")
    data = pd.DataFrame({
        "timestamp": np.repeat(timestamps, 2),
        "prosumer_id": np.tile(["P1", "P2"], len(timestamps)),
        "load_kwh": 1.0, "pv_kwh": 0.0,
    })
    train, validation, test, manifest = chronological_split(data, 180, 365)
    assert train["timestamp"].max() < validation["timestamp"].min()
    assert validation["timestamp"].max() < test["timestamp"].min()
    assert test["timestamp"].min() == pd.Timestamp("2017-01-01 00:00")
    assert test["timestamp"].max() == pd.Timestamp("2017-12-31 23:00")
    assert test["timestamp"].nunique() == 8760
    assert manifest.loc[manifest["split"] == "test", "hours"].iat[0] == 8760


def test_unplanned_outages_are_reproducible_but_not_leaked_to_forecast():
    from microgrid_opt.data import simulate_outages
    timestamps = pd.date_range("2020-01-01", periods=8760, freq="h")
    base = pd.DataFrame({
        "timestamp": np.repeat(timestamps, 2),
        "prosumer_id": np.tile(["P1", "P2"], len(timestamps)),
        "weather": "Clear",
    })
    first, events_first = simulate_outages(base, "moderate", seed=11)
    second, events_second = simulate_outages(base, "moderate", seed=11)
    assert np.array_equal(first["grid_available"], second["grid_available"])
    assert events_first.equals(events_second)
    assert (first["grid_available"] == 0).any()
    assert (first["grid_available_forecast"] == 1).all()


def test_dynamic_rtp_uses_real_grid_stress_without_forecast_oracle():
    from microgrid_opt.data import add_dynamic_rtp
    timestamps = pd.date_range("2026-01-01", periods=400, freq="h")
    load = 1.0 + 0.3 * np.sin(np.arange(400) * 2 * np.pi / 24)
    frame = pd.DataFrame({
        "timestamp": timestamps, "prosumer_id": "P1",
        "load_kwh": load, "pv_kwh": 0.2,
        "load_forecast_kwh": np.r_[load[0], load[:-1]],
        "pv_forecast_kwh": 0.2,
        "grid_available": 1, "grid_available_forecast": 1,
    })
    frame.loc[frame.index == 300, "grid_available"] = 0
    priced, _ = add_dynamic_rtp(frame, seed=7)
    row = priced.loc[priced["timestamp"] == timestamps[300]].iloc[0]
    assert row["rtp_cad_per_kwh"] > row["rtp_forecast_cad_per_kwh"] + 0.05
    assert priced["rtp_cad_per_kwh"].nunique() > 20


def test_causal_forecast_cannot_see_modified_future_values():
    from microgrid_opt.data import _causal_forecast
    values = pd.Series(np.linspace(1.0, 3.0, 500))
    baseline = _causal_forecast(values)
    changed = values.copy(); changed.iloc[350:] = 999.0
    alternative = _causal_forecast(changed)
    assert np.allclose(baseline.iloc[:350], alternative.iloc[:350], equal_nan=True)


def test_posthoc_reward_matches_environment_at_default_epsilon_kappa():
    from microgrid_opt.evaluation import posthoc_equity_sweep
    data = generate_synthetic_community(3, 2)
    cfg = ExperimentConfig()
    result = run_controller(MicrogridEnvironment(data, cfg, "tou"), RuleBasedController())
    summary = posthoc_equity_sweep(
        {("rule_based", "proportional"): result}, cfg,
        epsilon_grid=(cfg.reward.atkinson_epsilon,),
        kappa_grid=(cfg.reward.kappa,),
    )
    assert np.isclose(summary["cumulative_reward"].iat[0], result["reward"].sum())


def test_calendar_aging_is_shared_equally_and_cycle_aging_by_throughput():
    from microgrid_opt.evaluation import run_controller
    data = generate_synthetic_community(3, 1)
    # Force one household to have much larger surplus/deficit exposure.
    data.loc[data['prosumer_id'].eq('P1'), 'pv_kwh'] *= 4.0
    cfg = ExperimentConfig()
    env = MicrogridEnvironment(data, cfg, 'tou')
    result = run_controller(env, RuleBasedController())
    # Calendar aging is time-driven: every prosumer gets exactly 1/N each hour.
    cal = result[[f'calendar_degradation_cost_{pid}' for pid in env.ids]].to_numpy()
    assert np.allclose(cal, cal[:, [0]])
    assert np.allclose(cal.sum(axis=1), result['calendar_degradation_cost_cad'])
    # Cyclic shares conserve the community cyclic degradation cost.
    cyc = result[[f'cycle_degradation_cost_{pid}' for pid in env.ids]].to_numpy()
    assert np.allclose(cyc.sum(axis=1), result['cycle_degradation_cost_cad'])
    total = result[[f'degradation_cost_{pid}' for pid in env.ids]].to_numpy()
    assert np.allclose(total, cal + cyc)
    assert np.allclose(total.sum(axis=1), result['degradation_cost_cad'])


def test_balanced_tariff_assignment_represents_all_four_tariffs_and_is_fixed():
    from microgrid_opt.tariffs import fit_balanced_tariff_assignments
    data = generate_synthetic_community(10, 10)
    assignments, table = fit_balanced_tariff_assignments(data, TariffConfig())
    counts = table['selected_tariff'].value_counts()
    assert set(counts.index) == {'flat', 'tou', 'tiered', 'rtp'}
    assert counts.max() - counts.min() <= 1
    assert set(assignments) == set(table['prosumer_id'])
    assert table['selected_tariff'].notna().all()
