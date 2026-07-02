"""Typed loaders for engine config (configs/*.yaml) and pack metadata YAML.

Engine config holds tunables (K-factors, decays, windows); packs hold competition
facts (tier mappings, backtest windows) per CLAUDE.md rule 1.
"""

import datetime as dt
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class GoalMarginConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    two_goals: float
    three_goals: float
    per_extra_goal: float

    def multiplier(self, margin: int) -> float:
        if margin <= 1:
            return 1.0
        if margin == 2:
            return self.two_goals
        return self.three_goals + (margin - 3) * self.per_extra_goal


class EloConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    initial: float
    home_advantage: float
    k_factors: dict[str, float]
    goal_margin: GoalMarginConfig

    def k_for(self, tier: str) -> float:
        try:
            return self.k_factors[tier]
        except KeyError:
            known = ", ".join(sorted(self.k_factors))
            raise KeyError(f"no K-factor for tier {tier!r} (configured: {known})") from None


class FormConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    decay: float = Field(gt=0.0, lt=1.0)


class RestConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    cap_days: int = Field(gt=0)
    default_days: int = Field(gt=0)


class AttackDefenceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    alpha: float = Field(gt=0.0, lt=1.0)
    mu_alpha: float = Field(gt=0.0, lt=1.0)
    initial_mu: float = Field(gt=0.0)
    floor: float = Field(gt=0.0)


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    window_years: int = Field(gt=0)
    ece_bins: int = Field(gt=1)


class LogisticRungConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_iter: int = Field(gt=0)
    c: float = Field(gt=0.0)


# Backwards-friendly alias: rung 0 and rung 1 share the logistic config shape.
Rung0Config = LogisticRungConfig


class Rung2Config(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_trials: int = Field(gt=0)
    n_estimators: int = Field(gt=0)
    early_stopping_rounds: int = Field(gt=0)
    learning_rate: tuple[float, float]
    num_leaves: tuple[int, int]
    min_child_samples: tuple[int, int]


class MlflowConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    tracking_dir: str
    experiment: str


class ModelsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    rung0: LogisticRungConfig
    rung1: LogisticRungConfig
    rung2: Rung2Config


class SimulationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    runs: int = Field(gt=0)
    max_goals: int = Field(gt=0)
    lambda_floor: float = Field(gt=0.0)
    lambda_cap: float = Field(gt=0.0)


class EngineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    seed: int
    elo: EloConfig
    form: FormConfig
    rest: RestConfig
    attack_defence: AttackDefenceConfig
    calibration: CalibrationConfig
    models: ModelsConfig
    simulation: SimulationConfig
    mlflow: MlflowConfig


def load_engine_config(path: Path) -> EngineConfig:
    return EngineConfig.model_validate(yaml.safe_load(path.read_text()))


class CompetitionTiers(BaseModel):
    """Pack-declared mapping from tournament names to importance tiers."""

    model_config = ConfigDict(frozen=True)

    tiers: dict[str, list[str]]
    qualification_suffix: str
    qualification_tier: str
    default_tier: str

    def tier_of(self, competition: str) -> str:
        for tier, names in self.tiers.items():
            if competition in names:
                return tier
        if competition.endswith(self.qualification_suffix):
            return self.qualification_tier
        return self.default_tier


class BacktestWindow(BaseModel):
    """Train on matches < freeze; evaluate on `competition` matches in [freeze, end)."""

    model_config = ConfigDict(frozen=True)

    name: str
    freeze: dt.date
    end: dt.date
    competition: str


def load_competition_tiers(pack_dir: Path) -> CompetitionTiers:
    return CompetitionTiers.model_validate(
        yaml.safe_load((pack_dir / "competitions.yaml").read_text())
    )


def load_backtests(pack_dir: Path) -> list[BacktestWindow]:
    raw = yaml.safe_load((pack_dir / "backtests.yaml").read_text())
    return [BacktestWindow.model_validate(entry) for entry in raw["backtests"]]
