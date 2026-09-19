from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class BatteryConfig:
    capacity_kwh: float = 40.0
    initial_soc: float = 0.50
    min_soc: float = 0.20
    max_soc: float = 0.90
    max_charge_kw: float = 20.0
    max_discharge_kw: float = 20.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    replacement_cost_cad: float = 16_000.0
    cycle_life: float = 4_000.0
    calendar_fade_per_year: float = 0.015
    end_of_life_fade: float = 0.20


@dataclass
class TariffConfig:
    flat_cad_per_kwh: float = 0.1270
    flat_basic_charge_cad_per_day: float = 0.2500
    tier1_cad_per_kwh: float = 0.1187
    tiered_basic_charge_cad_per_day: float = 0.2344
    tier2_cad_per_kwh: float = 0.1408
    tier1_daily_kwh: float = 22.1918
    tou_discount_cad_per_kwh: float = 0.05
    tou_surcharge_cad_per_kwh: float = 0.05
    overnight_hours: Tuple[int, ...] = tuple(range(23, 24)) + tuple(range(0, 7))
    on_peak_hours: Tuple[int, ...] = tuple(range(16, 21))
    export_credit_cad_per_kwh: float = 0.0


@dataclass
class RewardConfig:
    """Community reward = Atkinson welfare (efficiency+equity) - kappa * degradation.

    `atkinson_epsilon` controls the efficiency/equity trade-off: <1 leans
    efficiency, >1 leans equity, =1 is the geometric-mean special case
    (client instruction, 2026-09-06). `kappa` is the battery-degradation
    weight, swept independently over `equity.KAPPA_GRID`.
    """
    weights: Dict[str, float] = field(default_factory=lambda: {
        "cost": 1 / 3,
        "pv": 1 / 3,
        "reliability": 1 / 3,
    })
    cost_scale_cad: float = 10.0
    kappa: float = 0.05
    atkinson_epsilon: float = 1.0
    unserved_energy_penalty_cad_per_kwh: float = 10.0

    @property
    def degradation_weight(self) -> float:  # backward-compatible alias
        return self.kappa


@dataclass
class ExperimentConfig:
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    tariff: TariffConfig = field(default_factory=TariffConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    allocation_method: str = "proportional"
    mpc_horizon_hours: int = 24
    switching_windows: Tuple[int, ...] = (1, 3, 6, 12, 24)
    switching_evaluation_horizon_hours: int = 1
    allocation_methods: Tuple[str, ...] = ("proportional", "priority", "contribution")
    seed: int = 42

