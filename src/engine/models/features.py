"""Engineered feature extraction for Rung 1 and Rung 2 (SCOPE §4 P3).

All inputs come from the as-of feature store, so chronological integrity is
inherited. Diffs are home-minus-away throughout.
"""

import numpy as np

from engine.evaluation.metrics import FloatArray
from engine.ratings.store import MatchFeatures

FEATURE_NAMES: tuple[str, ...] = (
    "elo_diff",
    "neutral",
    "rest_diff",
    "form_diff",
    "attack_diff",
    "defence_diff",
    "matches_diff_log",
    "tier_matches_diff_log",
)


def full_features(features: list[MatchFeatures]) -> FloatArray:
    def row(f: MatchFeatures) -> list[float]:
        return [
            f.elo_diff,
            float(f.neutral),
            float(f.rest_diff),
            f.form_home - f.form_away,
            f.attack_home - f.attack_away,
            f.defence_home - f.defence_away,
            _log_diff(f.matches_home, f.matches_away),
            _log_diff(f.tier_matches_home, f.tier_matches_away),
        ]

    return np.array([row(f) for f in features], dtype=np.float64)


def _log_diff(home: int, away: int) -> float:
    """Signed log scale keeps century-scale count gaps from dominating."""
    return float(np.log1p(home) - np.log1p(away))
