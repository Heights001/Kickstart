"""Calibration layer (SCOPE §2.5): fit isotonic and Platt on a walk-forward
validation window, pick per model by ECE (Brier as tie-break), persist with the model.

Both calibrators operate one-vs-rest per class on the model's predicted
probabilities, then renormalise rows to sum to 1.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from engine.evaluation.metrics import FloatArray, IntArray, brier, ece

_EPS = 1e-9


class IdentityCalibrator:
    name = "identity"

    def fit(self, probs: FloatArray, y: IntArray) -> None:
        pass

    def transform(self, probs: FloatArray) -> FloatArray:
        return probs


class IsotonicCalibrator:
    name = "isotonic"

    def __init__(self) -> None:
        self._per_class: list[IsotonicRegression] = []

    def fit(self, probs: FloatArray, y: IntArray) -> None:
        self._per_class = []
        for cls in range(probs.shape[1]):
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(probs[:, cls], (y == cls).astype(np.float64))
            self._per_class.append(iso)

    def transform(self, probs: FloatArray) -> FloatArray:
        columns = [
            np.asarray(iso.predict(probs[:, cls]), dtype=np.float64)
            for cls, iso in enumerate(self._per_class)
        ]
        return _renormalise(np.column_stack(columns))


class PlattCalibrator:
    """Per-class logistic regression on the log-odds of the model probability."""

    name = "platt"

    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._per_class: list[LogisticRegression] = []

    def fit(self, probs: FloatArray, y: IntArray) -> None:
        self._per_class = []
        for cls in range(probs.shape[1]):
            lr = LogisticRegression(random_state=self._seed)
            lr.fit(_log_odds(probs[:, cls]), (y == cls).astype(np.int64))
            self._per_class.append(lr)

    def transform(self, probs: FloatArray) -> FloatArray:
        columns = [
            np.asarray(lr.predict_proba(_log_odds(probs[:, cls]))[:, 1], dtype=np.float64)
            for cls, lr in enumerate(self._per_class)
        ]
        return _renormalise(np.column_stack(columns))


type Calibrator = IdentityCalibrator | IsotonicCalibrator | PlattCalibrator


def fit_best_calibrator(
    probs_val: FloatArray, y_val: IntArray, *, ece_bins: int, seed: int
) -> Calibrator:
    """Fit all calibrators on the validation window; return the best by ECE."""
    candidates: list[Calibrator] = [
        IdentityCalibrator(),
        IsotonicCalibrator(),
        PlattCalibrator(seed),
    ]
    best: tuple[float, float] | None = None
    best_calibrator = candidates[0]
    for calibrator in candidates:
        calibrator.fit(probs_val, y_val)
        calibrated = calibrator.transform(probs_val)
        key = (ece(y_val, calibrated, ece_bins), brier(y_val, calibrated))
        if best is None or key < best:
            best = key
            best_calibrator = calibrator
    return best_calibrator


def _log_odds(p: FloatArray) -> FloatArray:
    clipped = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def _renormalise(probs: FloatArray) -> FloatArray:
    probs = np.clip(probs, _EPS, None)
    result: FloatArray = probs / probs.sum(axis=1, keepdims=True)
    return result
