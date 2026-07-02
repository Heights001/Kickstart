"""Naive baselines every rung must beat (SCOPE §4 P1 gate)."""

import numpy as np

from engine.evaluation.metrics import FloatArray, IntArray


class FrequencyBaseline:
    """Predicts the training-set class frequencies for every match."""

    def __init__(self) -> None:
        self._probs: FloatArray | None = None

    def fit(self, x: FloatArray, y: IntArray) -> None:
        counts = np.bincount(y, minlength=3).astype(np.float64)
        self._probs = counts / counts.sum()

    def predict_proba(self, x: FloatArray) -> FloatArray:
        if self._probs is None:
            raise RuntimeError("baseline not fitted")
        return np.tile(self._probs, (x.shape[0], 1))
