"""Walk-forward evaluation harness (SCOPE §2.12, rule 3).

For each backtest window: train on matches strictly before the calibration window,
fit the calibrator on [calib_start, freeze), test on the window's competition
matches in [freeze, end). No shuffled splits, ever.
"""

import datetime as dt
from dataclasses import dataclass

from engine.core.config import BacktestWindow, EngineConfig
from engine.evaluation.metrics import FloatArray, IntArray, brier, ece, log_loss
from engine.models.base import labels_array
from engine.models.baselines import FrequencyBaseline
from engine.models.calibration import Calibrator, fit_best_calibrator
from engine.models.rung0 import Rung0Model, rung0_features
from engine.ratings.store import MatchFeatures


@dataclass(frozen=True)
class Metrics:
    log_loss: float
    brier: float
    ece: float


@dataclass(frozen=True)
class WindowResult:
    window: str
    n_train: int
    n_val: int
    n_test: int
    calibrator: str
    model: Metrics
    baseline: Metrics

    @property
    def beats_baseline(self) -> bool:
        return self.model.log_loss < self.baseline.log_loss


def split_window(
    features: list[MatchFeatures], window: BacktestWindow, calibration_years: int
) -> tuple[list[MatchFeatures], list[MatchFeatures], list[MatchFeatures]]:
    calib_start = _years_before(window.freeze, calibration_years)
    train = [f for f in features if f.date < calib_start]
    val = [f for f in features if calib_start <= f.date < window.freeze]
    test = [
        f
        for f in features
        if window.freeze <= f.date < window.end and f.competition == window.competition
    ]
    return train, val, test


def evaluate_window(
    features: list[MatchFeatures], window: BacktestWindow, config: EngineConfig
) -> WindowResult:
    train, val, test = split_window(features, window, config.calibration.window_years)
    if not train or not val or not test:
        raise ValueError(
            f"window {window.name!r}: empty split "
            f"(train={len(train)}, val={len(val)}, test={len(test)})"
        )
    x_train, y_train = rung0_features(train), labels_array(train)
    x_val, y_val = rung0_features(val), labels_array(val)
    x_test, y_test = rung0_features(test), labels_array(test)

    model = Rung0Model(config.models.rung0, config.seed)
    model.fit(x_train, y_train)
    calibrator = fit_best_calibrator(
        model.predict_proba(x_val), y_val, ece_bins=config.calibration.ece_bins, seed=config.seed
    )
    probs = calibrator.transform(model.predict_proba(x_test))

    baseline = FrequencyBaseline()
    baseline.fit(x_train, y_train)
    baseline_probs = baseline.predict_proba(x_test)

    bins = config.calibration.ece_bins
    return WindowResult(
        window=window.name,
        n_train=len(train),
        n_val=len(val),
        n_test=len(test),
        calibrator=calibrator.name,
        model=_metrics(y_test, probs, bins),
        baseline=_metrics(y_test, baseline_probs, bins),
    )


def fit_model_asof(
    features: list[MatchFeatures], as_of: dt.date, config: EngineConfig
) -> tuple[Rung0Model, Calibrator]:
    """Train Rung 0 + its calibrator strictly on pre-``as_of`` data (rule 3)."""
    calib_start = _years_before(as_of, config.calibration.window_years)
    train = [f for f in features if f.date < calib_start]
    val = [f for f in features if calib_start <= f.date < as_of]
    if not train or not val:
        raise ValueError(f"as_of {as_of}: empty split (train={len(train)}, val={len(val)})")
    model = Rung0Model(config.models.rung0, config.seed)
    model.fit(rung0_features(train), labels_array(train))
    calibrator = fit_best_calibrator(
        model.predict_proba(rung0_features(val)),
        labels_array(val),
        ece_bins=config.calibration.ece_bins,
        seed=config.seed,
    )
    return model, calibrator


def _metrics(y: IntArray, probs: FloatArray, bins: int) -> Metrics:
    return Metrics(log_loss=log_loss(y, probs), brier=brier(y, probs), ece=ece(y, probs, bins))


def _years_before(date: dt.date, years: int) -> dt.date:
    try:
        return date.replace(year=date.year - years)
    except ValueError:  # Feb 29
        return date.replace(year=date.year - years, day=28)
