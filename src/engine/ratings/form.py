"""Exponentially weighted form rating: recent results on a 0-1 scale.

New teams start neutral (0.5). Each match moves a team's form toward its result
score (win 1, draw 0.5, loss 0) with configured decay.
"""

from engine.core.config import FormConfig
from engine.core.schema import Match

_NEUTRAL_FORM = 0.5


class FormRater:
    def __init__(self, config: FormConfig) -> None:
        self._decay = config.decay
        self._form: dict[str, float] = {}

    def form(self, team_id: str) -> float:
        return self._form.get(team_id, _NEUTRAL_FORM)

    def update(self, match: Match) -> None:
        if match.home_goals > match.away_goals:
            home_result, away_result = 1.0, 0.0
        elif match.home_goals < match.away_goals:
            home_result, away_result = 0.0, 1.0
        else:
            home_result = away_result = 0.5
        for team, result in ((match.home, home_result), (match.away, away_result)):
            self._form[team] = self._decay * self.form(team) + (1.0 - self._decay) * result

    def snapshot(self) -> dict[str, float]:
        return dict(self._form)
