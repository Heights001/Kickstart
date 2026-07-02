"""Rung 0: the permanent baseline model (SCOPE §2.5).

Multinomial logistic regression on [elo_diff, neutral flag, rest-day diff]. Trains
in seconds; every higher rung must beat it out-of-time to ship.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from engine.core.config import Rung0Config
from engine.evaluation.metrics import FloatArray, IntArray
from engine.ratings.store import MatchFeatures


def rung0_features(features: list[MatchFeatures]) -> FloatArray:
    return np.array(
        [[f.elo_diff, float(f.neutral), float(f.rest_diff)] for f in features],
        dtype=np.float64,
    )


class Rung0Model:
    def __init__(self, config: Rung0Config, seed: int) -> None:
        self._model = LogisticRegression(C=config.c, max_iter=config.max_iter, random_state=seed)

    def fit(self, x: FloatArray, y: IntArray) -> None:
        self._model.fit(x, y)

    def predict_proba(self, x: FloatArray) -> FloatArray:
        raw = np.asarray(self._model.predict_proba(x), dtype=np.float64)
        # sklearn orders columns by self._model.classes_; map back to 0/1/2.
        probs = np.zeros((x.shape[0], 3), dtype=np.float64)
        for column, label in enumerate(self._model.classes_):
            probs[:, int(label)] = raw[:, column]
        return probs

    def home_win_contributions(self, x: FloatArray) -> list[float]:
        """Per-feature contribution to the home-win logit for one row."""
        row = list(self._model.classes_).index(0)
        return [float(c * v) for c, v in zip(self._model.coef_[row], x[0], strict=True)]
