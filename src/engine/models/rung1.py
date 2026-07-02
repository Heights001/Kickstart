"""Rung 1: multinomial logistic regression on the full engineered feature set."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from engine.core.config import LogisticRungConfig
from engine.evaluation.metrics import FloatArray, IntArray


class Rung1Model:
    def __init__(self, config: LogisticRungConfig, seed: int) -> None:
        self._pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(C=config.c, max_iter=config.max_iter, random_state=seed),
                ),
            ]
        )

    def fit(self, x: FloatArray, y: IntArray) -> None:
        self._pipeline.fit(x, y)

    def predict_proba(self, x: FloatArray) -> FloatArray:
        raw = np.asarray(self._pipeline.predict_proba(x), dtype=np.float64)
        probs = np.zeros((x.shape[0], 3), dtype=np.float64)
        for column, label in enumerate(self._pipeline.classes_):
            probs[:, int(label)] = raw[:, column]
        return probs

    def home_win_contributions(self, x: FloatArray) -> list[float]:
        """Per-feature contribution to the home-win logit (on the scaled input)."""
        scaled = self._pipeline.named_steps["scale"].transform(x)
        lr = self._pipeline.named_steps["lr"]
        row = list(lr.classes_).index(0)
        return [float(c * v) for c, v in zip(lr.coef_[row], scaled[0], strict=True)]
