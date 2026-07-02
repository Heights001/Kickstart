"""OutcomeModel protocol and the shared class ordering for outcome probabilities.

Every model rung is a drop-in ``OutcomeModel`` (SCOPE §2.5). Probability columns are
always ordered per ``CLASSES``: home win, draw, away win.
"""

from typing import Protocol

import numpy as np

from engine.evaluation.metrics import FloatArray, IntArray
from engine.ratings.store import MatchFeatures

CLASSES: tuple[str, str, str] = ("H", "D", "A")
_CLASS_INDEX = {label: index for index, label in enumerate(CLASSES)}


class OutcomeModel(Protocol):
    def fit(self, x: FloatArray, y: IntArray) -> None: ...

    def predict_proba(self, x: FloatArray) -> FloatArray:
        """Return an (n, 3) array of probabilities in ``CLASSES`` order."""
        ...


def outcome_index(features: MatchFeatures) -> int:
    return _CLASS_INDEX[features.outcome]


def labels_array(features: list[MatchFeatures]) -> IntArray:
    return np.array([outcome_index(f) for f in features], dtype=np.int64)
