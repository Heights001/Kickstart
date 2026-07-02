"""Multiplicative attack/defence ratings vs an opponent-strength-adjusted baseline.

Expected goals for the home side are ``mu * attack[home] * defence[away]`` where
``mu`` is a slowly adapting global goals-per-side mean, attack > 1 means scores
more than average, defence > 1 means concedes more than average. Each match nudges
the four involved ratings toward the rates implied by the observed score, using
pre-match values throughout.
"""

from engine.core.config import AttackDefenceConfig
from engine.core.schema import Match


class AttackDefenceRater:
    def __init__(self, config: AttackDefenceConfig) -> None:
        self._config = config
        self._mu = config.initial_mu
        self._attack: dict[str, float] = {}
        self._defence: dict[str, float] = {}

    def attack(self, team_id: str) -> float:
        return self._attack.get(team_id, 1.0)

    def defence(self, team_id: str) -> float:
        return self._defence.get(team_id, 1.0)

    def expected_goals(self, for_team: str, against_team: str) -> float:
        expected = self._mu * self.attack(for_team) * self.defence(against_team)
        return max(self._config.floor, expected)

    def update(self, match: Match) -> None:
        cfg = self._config
        attack_h, attack_a = self.attack(match.home), self.attack(match.away)
        defence_h, defence_a = self.defence(match.home), self.defence(match.away)
        mu = self._mu

        def implied(goals: int, opponent_factor: float) -> float:
            return goals / max(cfg.floor, mu * opponent_factor)

        self._attack[match.home] = attack_h + cfg.alpha * (
            implied(match.home_goals, defence_a) - attack_h
        )
        self._defence[match.away] = defence_a + cfg.alpha * (
            implied(match.home_goals, attack_h) - defence_a
        )
        self._attack[match.away] = attack_a + cfg.alpha * (
            implied(match.away_goals, defence_h) - attack_a
        )
        self._defence[match.home] = defence_h + cfg.alpha * (
            implied(match.away_goals, attack_a) - defence_h
        )
        goals_per_side = (match.home_goals + match.away_goals) / 2.0
        self._mu = mu + cfg.mu_alpha * (goals_per_side - mu)

    @property
    def mu(self) -> float:
        return self._mu
