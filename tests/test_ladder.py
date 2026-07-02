import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from engine.core.config import EngineConfig, load_engine_config
from engine.evaluation.metrics import log_loss
from engine.evaluation.promotion import decide, promoted_rung, record_promotion
from engine.evaluation.walk_forward import Metrics, WindowResult
from engine.explain.shap_explain import rung2_feature_importance
from engine.models.base import labels_array
from engine.models.features import FEATURE_NAMES, full_features
from engine.models.ladder import get_rung
from engine.models.rung2 import Rung2Model
from tests.test_models import synthetic_dataset

REPO = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def config() -> EngineConfig:
    return load_engine_config(REPO / "configs/default.yaml")


def window_result(name: str, ll: float, ece_value: float) -> WindowResult:
    metrics = Metrics(log_loss=ll, brier=0.6, ece=ece_value)
    baseline = Metrics(log_loss=1.09, brier=0.66, ece=0.05)
    return WindowResult(
        window=name,
        n_train=100,
        n_val=50,
        n_test=64,
        calibrator="isotonic",
        model=metrics,
        baseline=baseline,
    )


class TestFeatures:
    def test_shape_and_names(self) -> None:
        data = synthetic_dataset(10, dt.date(2020, 1, 1), seed=0)
        x = full_features(data)
        assert x.shape == (10, len(FEATURE_NAMES))

    def test_log_diff_antisymmetric(self) -> None:
        data = synthetic_dataset(5, dt.date(2020, 1, 1), seed=0)
        x = full_features(data)
        # synthetic data has equal counts both sides -> zero log diffs
        assert (x[:, 6] == 0.0).all()
        assert (x[:, 7] == 0.0).all()


class TestLadder:
    def test_rungs_construct_and_fit(self, config: EngineConfig) -> None:
        data = synthetic_dataset(400, dt.date(2000, 1, 1), seed=1)
        y = labels_array(data)
        for n in (0, 1):
            spec = get_rung(n, config)
            x = spec.extract(data)
            model = spec.make()
            model.fit(x, y)
            probs = model.predict_proba(x)
            assert probs.shape == (400, 3)
            assert probs.sum(axis=1) == pytest.approx(np.ones(400))

    def test_unknown_rung_raises(self, config: EngineConfig) -> None:
        with pytest.raises(ValueError, match="unknown rung"):
            get_rung(9, config)


class TestRung2:
    @pytest.fixture(scope="class")
    def fitted(self, config: EngineConfig) -> tuple[Rung2Model, object, object]:
        fast = config.models.rung2.model_copy(update={"n_trials": 2, "n_estimators": 40})
        train = synthetic_dataset(600, dt.date(2000, 1, 1), seed=2)
        val = synthetic_dataset(200, dt.date(2005, 1, 1), seed=3)
        x_train, y_train = full_features(train), labels_array(train)
        x_val, y_val = full_features(val), labels_array(val)
        model = Rung2Model(fast, seed=42)
        model.tune(x_train, y_train, x_val, y_val)
        model.fit(x_train, y_train)
        return model, x_val, y_val

    def test_probs_valid_and_better_than_uniform(
        self, fitted: tuple[Rung2Model, object, object]
    ) -> None:
        model, x_val, y_val = fitted
        probs = model.predict_proba(x_val)  # type: ignore[arg-type]
        assert probs.shape[1] == 3
        assert probs.sum(axis=1) == pytest.approx(np.ones(len(probs)))
        uniform = np.full_like(probs, 1.0 / 3.0)
        assert log_loss(y_val, probs) < log_loss(y_val, uniform)  # type: ignore[arg-type]

    def test_fit_before_tune_raises(self, config: EngineConfig) -> None:
        model = Rung2Model(config.models.rung2, seed=42)
        with pytest.raises(RuntimeError, match="tune"):
            model.fit(np.zeros((4, 8)), np.zeros(4, dtype=np.int64))

    def test_shap_importance(self, fitted: tuple[Rung2Model, object, object]) -> None:
        model, x_val, _ = fitted
        importance = rung2_feature_importance(model, x_val[:50])  # type: ignore[index]
        assert set(importance) == set(FEATURE_NAMES)
        # elo_diff drives the synthetic labels; it must dominate.
        assert importance["elo_diff"] == max(importance.values())


class TestPromotion:
    def test_promoted_when_better_on_both(self) -> None:
        candidate = [window_result("w1", 0.95, 0.05), window_result("w2", 0.99, 0.06)]
        incumbent = [window_result("w1", 1.00, 0.06), window_result("w2", 1.02, 0.07)]
        decision = decide(1, candidate, 0, incumbent)
        assert decision.promoted

    def test_not_promoted_on_worse_ece(self) -> None:
        candidate = [window_result("w1", 0.95, 0.20)]
        incumbent = [window_result("w1", 1.00, 0.06)]
        assert not decide(1, candidate, 0, incumbent).promoted

    def test_not_promoted_on_worse_log_loss(self) -> None:
        candidate = [window_result("w1", 1.05, 0.01)]
        incumbent = [window_result("w1", 1.00, 0.06)]
        assert not decide(1, candidate, 0, incumbent).promoted

    def test_registry_roundtrip(self, tmp_path: Path) -> None:
        data_dir = tmp_path
        (data_dir / "processed").mkdir()
        assert promoted_rung(data_dir) == 0

        good = decide(1, [window_result("w", 0.9, 0.05)], 0, [window_result("w", 1.0, 0.06)])
        record_promotion(data_dir, good)
        assert promoted_rung(data_dir) == 1

        # A failed rung-2 attempt must not demote the promoted rung 1.
        bad = decide(2, [window_result("w", 1.2, 0.30)], 1, [window_result("w", 0.9, 0.05)])
        record_promotion(data_dir, bad)
        assert promoted_rung(data_dir) == 1
