import datetime as dt
import math
from pathlib import Path

import numpy as np
import pytest

from engine.core.config import BacktestWindow, EngineConfig, load_engine_config
from engine.evaluation.metrics import brier, ece, log_loss
from engine.evaluation.walk_forward import evaluate_window, split_window
from engine.models.baselines import FrequencyBaseline
from engine.models.calibration import IsotonicCalibrator, fit_best_calibrator
from engine.models.rung0 import Rung0Model, rung0_features
from engine.ratings.store import MatchFeatures

REPO = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def config() -> EngineConfig:
    return load_engine_config(REPO / "configs/default.yaml")


def make_features(
    date: dt.date, elo_home: float, elo_away: float, outcome: str, competition: str = "Test Cup"
) -> MatchFeatures:
    return MatchFeatures(
        date=date,
        competition=competition,
        home="home_team",
        away="away_team",
        neutral=False,
        elo_home=elo_home,
        elo_away=elo_away,
        form_home=0.5,
        form_away=0.5,
        attack_home=1.0,
        attack_away=1.0,
        defence_home=1.0,
        defence_away=1.0,
        rest_days_home=7,
        rest_days_away=7,
        outcome=outcome,  # type: ignore[arg-type]
    )


def synthetic_dataset(n: int, start: dt.date, seed: int) -> list[MatchFeatures]:
    """Elo-separable synthetic matches: big elo_diff decides the outcome."""
    rng = np.random.default_rng(seed)
    features = []
    for i in range(n):
        diff = float(rng.normal(0.0, 200.0))
        noise = float(rng.normal(0.0, 60.0))
        effective = diff + noise
        outcome = "H" if effective > 40 else ("A" if effective < -40 else "D")
        features.append(make_features(start + dt.timedelta(days=i), 1500.0 + diff, 1500.0, outcome))
    return features


class TestMetrics:
    def test_log_loss_perfect_and_uniform(self) -> None:
        y = np.array([0, 1, 2], dtype=np.int64)
        perfect = np.eye(3)
        uniform = np.full((3, 3), 1.0 / 3.0)
        assert log_loss(y, perfect) == pytest.approx(0.0, abs=1e-10)
        assert log_loss(y, uniform) == pytest.approx(math.log(3.0))

    def test_brier_hand_computed(self) -> None:
        y = np.array([0], dtype=np.int64)
        uniform = np.full((1, 3), 1.0 / 3.0)
        # (1/3-1)^2 + (1/3)^2 + (1/3)^2 = 6/9
        assert brier(y, uniform) == pytest.approx(2.0 / 3.0)

    def test_ece_perfectly_calibrated_is_zero(self) -> None:
        # Confidence 0.75 and correct exactly 75% of the time.
        probs = np.tile([0.75, 0.15, 0.10], (100, 1))
        y = np.array([0] * 75 + [1] * 25, dtype=np.int64)
        assert ece(y, probs, bins=10) == pytest.approx(0.0, abs=1e-9)

    def test_ece_overconfident(self) -> None:
        probs = np.tile([0.95, 0.03, 0.02], (100, 1))
        y = np.array([0] * 50 + [1] * 50, dtype=np.int64)
        assert ece(y, probs, bins=10) == pytest.approx(0.45)


class TestFrequencyBaseline:
    def test_predicts_train_frequencies(self) -> None:
        y = np.array([0, 0, 1, 2], dtype=np.int64)
        baseline = FrequencyBaseline()
        baseline.fit(np.zeros((4, 3)), y)
        probs = baseline.predict_proba(np.zeros((2, 3)))
        assert probs.shape == (2, 3)
        assert probs[0] == pytest.approx([0.5, 0.25, 0.25])


class TestRung0:
    def test_learns_elo_signal_and_beats_baseline(self, config: EngineConfig) -> None:
        train = synthetic_dataset(800, dt.date(2000, 1, 1), seed=1)
        test = synthetic_dataset(200, dt.date(2010, 1, 1), seed=2)
        x_train, x_test = rung0_features(train), rung0_features(test)
        y_train = np.array([{"H": 0, "D": 1, "A": 2}[f.outcome] for f in train], dtype=np.int64)
        y_test = np.array([{"H": 0, "D": 1, "A": 2}[f.outcome] for f in test], dtype=np.int64)

        model = Rung0Model(config.models.rung0, config.seed)
        model.fit(x_train, y_train)
        probs = model.predict_proba(x_test)
        assert probs.shape == (200, 3)
        assert probs.sum(axis=1) == pytest.approx(np.ones(200))

        baseline = FrequencyBaseline()
        baseline.fit(x_train, y_train)
        assert log_loss(y_test, probs) < log_loss(y_test, baseline.predict_proba(x_test))

    def test_deterministic(self, config: EngineConfig) -> None:
        data = synthetic_dataset(300, dt.date(2000, 1, 1), seed=3)
        x, y = (
            rung0_features(data),
            np.array([{"H": 0, "D": 1, "A": 2}[f.outcome] for f in data], dtype=np.int64),
        )
        probs = []
        for _ in range(2):
            model = Rung0Model(config.models.rung0, config.seed)
            model.fit(x, y)
            probs.append(model.predict_proba(x))
        np.testing.assert_array_equal(probs[0], probs[1])


class TestCalibration:
    def test_isotonic_improves_overconfident_probs(self) -> None:
        rng = np.random.default_rng(0)
        n = 2000
        # True probability 0.6 for class 0, but the model claims 0.9.
        y = (rng.random(n) > 0.6).astype(np.int64)  # 0 with p=0.6
        probs = np.tile([0.9, 0.06, 0.04], (n, 1))
        probs += rng.normal(0, 0.01, probs.shape)  # isotonic needs input variation
        probs = np.clip(probs, 1e-6, None)
        probs /= probs.sum(axis=1, keepdims=True)

        calibrator = IsotonicCalibrator()
        calibrator.fit(probs, y)
        calibrated = calibrator.transform(probs)
        assert ece(y, calibrated, 10) < ece(y, probs, 10)

    def test_best_calibrator_never_worse_than_identity(self) -> None:
        rng = np.random.default_rng(1)
        n = 1000
        y = rng.integers(0, 3, n).astype(np.int64)
        probs = rng.dirichlet([2.0, 2.0, 2.0], n)
        calibrator = fit_best_calibrator(probs, y, ece_bins=10, seed=42)
        assert ece(y, calibrator.transform(probs), 10) <= ece(y, probs, 10) + 1e-12


class TestWalkForward:
    def test_split_boundaries(self, config: EngineConfig) -> None:
        features = synthetic_dataset(400, dt.date(2010, 1, 1), seed=4)
        window = BacktestWindow(
            name="w", freeze=dt.date(2010, 9, 1), end=dt.date(2010, 10, 1), competition="Test Cup"
        )
        train, val, test = split_window(features, window, config.calibration.window_years)
        assert all(f.date < dt.date(2006, 9, 1) for f in train)
        assert all(dt.date(2006, 9, 1) <= f.date < window.freeze for f in val)
        assert all(window.freeze <= f.date < window.end for f in test)
        assert train == []  # dataset starts 2010: nothing before the calibration window

    def test_evaluate_window_beats_baseline_on_synthetic(self, config: EngineConfig) -> None:
        features = synthetic_dataset(2500, dt.date(2000, 1, 1), seed=5)
        window = BacktestWindow(
            name="synthetic",
            freeze=dt.date(2006, 6, 1),
            end=dt.date(2006, 9, 1),
            competition="Test Cup",
        )
        result = evaluate_window(features, window, config)
        assert result.n_train > 0 and result.n_val > 0 and result.n_test > 0
        assert result.beats_baseline

    def test_empty_split_raises(self, config: EngineConfig) -> None:
        features = synthetic_dataset(100, dt.date(2010, 1, 1), seed=6)
        window = BacktestWindow(
            name="w", freeze=dt.date(1990, 1, 1), end=dt.date(1990, 2, 1), competition="Test Cup"
        )
        with pytest.raises(ValueError, match="empty split"):
            evaluate_window(features, window, config)
