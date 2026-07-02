"""Chronological Elo updater (SCOPE §2.6).

K scales with competition importance tier, the update scales with winning margin,
and the home-advantage term is suppressed entirely on neutral venues. Updates are
zero-sum. Callers drive matches through in date order; see ``store.py`` for the
as-of walk.
"""

import math

from engine.core.config import EloConfig
from engine.core.schema import Match


class EloRater:
    def __init__(self, config: EloConfig) -> None:
        self._config = config
        self._ratings: dict[str, float] = {}

    def rating(self, team_id: str) -> float:
        return self._ratings.get(team_id, self._config.initial)

    def expected_home_score(self, home: str, away: str, *, neutral: bool) -> float:
        """Expected match score for the home side (win=1, draw=0.5, loss=0)."""
        advantage = 0.0 if neutral else self._config.home_advantage
        diff = self.rating(away) - self.rating(home) - advantage
        return 1.0 / (1.0 + math.pow(10.0, diff / 400.0))

    def update(self, match: Match, tier: str) -> None:
        expected = self.expected_home_score(match.home, match.away, neutral=match.neutral)
        actual = _home_score(match)
        margin = abs(match.home_goals - match.away_goals)
        delta = (
            self._config.k_for(tier)
            * self._config.goal_margin.multiplier(margin)
            * (actual - expected)
        )
        self._ratings[match.home] = self.rating(match.home) + delta
        self._ratings[match.away] = self.rating(match.away) - delta

    def snapshot(self) -> dict[str, float]:
        return dict(self._ratings)


def _home_score(match: Match) -> float:
    if match.home_goals > match.away_goals:
        return 1.0
    if match.home_goals < match.away_goals:
        return 0.0
    return 0.5
