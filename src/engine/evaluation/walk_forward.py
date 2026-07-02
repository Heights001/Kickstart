"""Walk-forward evaluation harness (SCOPE §2.12, rule 3).

For each backtest window: train on matches strictly before the calibration window,
fit the calibrator on [calib_start, freeze), test on the window's competition
matches in [freeze, end). No shuffled splits, ever.
"""

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING

from engine.core.config import BacktestWindow, EngineConfig
from engine.evaluation.metrics import FloatArray, IntArray, brier, ece, log_loss
from engine.models.base import OutcomeModel, labels_array
from engine.models.baselines import FrequencyBaseline
from engine.models.calibration import Calibrator, fit_best_calibrator
from engine.ratings.store import MatchFeatures

if TYPE_CHECKING:
    from engine.models.ladder import RungSpec


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
    features: list[MatchFeatures],
    window: BacktestWindow,
    config: EngineConfig,
    rung: "RungSpec | None" = None,
) -> WindowResult:
    from engine.models.ladder import get_rung

    spec = rung if rung is not None else get_rung(0, config)
    train, val, test = split_window(features, window, config.calibration.window_years)
    if not train or not val or not test:
        raise ValueError(
            f"window {window.name!r}: empty split "
            f"(train={len(train)}, val={len(val)}, test={len(test)})"
        )
    x_train, y_train = spec.extract(train), labels_array(train)
    x_val, y_val = spec.extract(val), labels_array(val)
    x_test, y_test = spec.extract(test), labels_array(test)

    model = _fit(spec, x_train, y_train, x_val, y_val)
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


def _fit(
    spec: "RungSpec",
    x_train: FloatArray,
    y_train: IntArray,
    x_val: FloatArray,
    y_val: IntArray,
) -> "OutcomeModel":
    model = spec.make()
    if spec.needs_tuning:
        from engine.models.rung2 import Rung2Model

        assert isinstance(model, Rung2Model)
        model.tune(x_train, y_train, x_val, y_val)
    model.fit(x_train, y_train)
    return model


def fit_model_asof(
    features: list[MatchFeatures],
    as_of: dt.date,
    config: EngineConfig,
    rung: "RungSpec | None" = None,
) -> "tuple[OutcomeModel, Calibrator, RungSpec]":
    """Train a rung + its calibrator strictly on pre-``as_of`` data (rule 3)."""
    from engine.models.ladder import get_rung

    spec = rung if rung is not None else get_rung(0, config)
    calib_start = _years_before(as_of, config.calibration.window_years)
    train = [f for f in features if f.date < calib_start]
    val = [f for f in features if calib_start <= f.date < as_of]
    if not train or not val:
        raise ValueError(f"as_of {as_of}: empty split (train={len(train)}, val={len(val)})")
    x_train, y_train = spec.extract(train), labels_array(train)
    x_val, y_val = spec.extract(val), labels_array(val)
    model = _fit(spec, x_train, y_train, x_val, y_val)
    calibrator = fit_best_calibrator(
        model.predict_proba(x_val),
        y_val,
        ece_bins=config.calibration.ece_bins,
        seed=config.seed,
    )
    return model, calibrator, spec


def _metrics(y: IntArray, probs: FloatArray, bins: int) -> Metrics:
    return Metrics(log_loss=log_loss(y, probs), brier=brier(y, probs), ece=ece(y, probs, bins))


def _years_before(date: dt.date, years: int) -> dt.date:
    try:
        return date.replace(year=date.year - years)
    except ValueError:  # Feb 29
        return date.replace(year=date.year - years, day=28)
