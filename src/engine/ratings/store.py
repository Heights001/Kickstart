"""Chronological ratings walk producing per-match pre-match feature snapshots.

The as-of guarantee (CLAUDE.md rule 3): every ``MatchFeatures`` record is captured
*before* the raters see that match's result, so a match's features never leak its
own outcome — nor any later match's. ``ratings_as_of`` consumes only matches with
``date < as_of``.
"""

import datetime as dt
from collections.abc import Iterable, Iterator
from typing import Literal

from pydantic import BaseModel, ConfigDict

from engine.core.config import CompetitionTiers, EngineConfig
from engine.core.schema import Match
from engine.ratings.attack_defence import AttackDefenceRater
from engine.ratings.elo import EloRater
from engine.ratings.form import FormRater

Outcome = Literal["H", "D", "A"]


class MatchFeatures(BaseModel):
    """Pre-match feature snapshot plus the realised outcome label."""

    model_config = ConfigDict(frozen=True)

    date: dt.date
    competition: str
    home: str
    away: str
    neutral: bool
    elo_home: float
    elo_away: float
    form_home: float
    form_away: float
    attack_home: float
    attack_away: float
    defence_home: float
    defence_away: float
    rest_days_home: int
    rest_days_away: int
    outcome: Outcome

    @property
    def elo_diff(self) -> float:
        return self.elo_home - self.elo_away

    @property
    def rest_diff(self) -> int:
        return self.rest_days_home - self.rest_days_away


class RatingsWalker:
    """Feeds matches through all raters in date order, snapshotting before updates."""

    def __init__(self, config: EngineConfig, tiers: CompetitionTiers) -> None:
        self._config = config
        self._tiers = tiers
        self.elo = EloRater(config.elo)
        self.form = FormRater(config.form)
        self.attack_defence = AttackDefenceRater(config.attack_defence)
        self._last_played: dict[str, dt.date] = {}
        self._cursor: dt.date | None = None

    def observe(self, match: Match) -> MatchFeatures:
        if self._cursor is not None and match.date < self._cursor:
            raise ValueError(
                f"matches out of order: {match.date} after cursor {self._cursor} "
                f"({match.home} vs {match.away})"
            )
        self._cursor = match.date
        features = self._snapshot(match)
        self.elo.update(match, self._tiers.tier_of(match.competition))
        self.form.update(match)
        self.attack_defence.update(match)
        self._last_played[match.home] = match.date
        self._last_played[match.away] = match.date
        return features

    def _snapshot(self, match: Match) -> MatchFeatures:
        return MatchFeatures(
            date=match.date,
            competition=match.competition,
            home=match.home,
            away=match.away,
            neutral=match.neutral,
            elo_home=self.elo.rating(match.home),
            elo_away=self.elo.rating(match.away),
            form_home=self.form.form(match.home),
            form_away=self.form.form(match.away),
            attack_home=self.attack_defence.attack(match.home),
            attack_away=self.attack_defence.attack(match.away),
            defence_home=self.attack_defence.defence(match.home),
            defence_away=self.attack_defence.defence(match.away),
            rest_days_home=self._rest_days(match.home, match.date),
            rest_days_away=self._rest_days(match.away, match.date),
            outcome=_outcome(match),
        )

    def _rest_days(self, team_id: str, on: dt.date) -> int:
        last = self._last_played.get(team_id)
        if last is None:
            return self._config.rest.default_days
        return min((on - last).days, self._config.rest.cap_days)


def build_features(
    matches: Iterable[Match], config: EngineConfig, tiers: CompetitionTiers
) -> Iterator[MatchFeatures]:
    """Walk all matches chronologically, yielding pre-match snapshots."""
    walker = RatingsWalker(config, tiers)
    for match in sorted(matches, key=lambda m: m.date):
        yield walker.observe(match)


def ratings_as_of(
    matches: Iterable[Match],
    as_of: dt.date,
    config: EngineConfig,
    tiers: CompetitionTiers,
) -> RatingsWalker:
    """Ratings state from strictly-before-``as_of`` matches only (rule 3)."""
    walker = RatingsWalker(config, tiers)
    for match in sorted(matches, key=lambda m: m.date):
        if match.date >= as_of:
            break
        walker.observe(match)
    return walker


def _outcome(match: Match) -> Outcome:
    if match.home_goals > match.away_goals:
        return "H"
    if match.home_goals < match.away_goals:
        return "A"
    return "D"
