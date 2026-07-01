import datetime as dt

import pytest
from pydantic import ValidationError

from engine.core.schema import Match, Team


def make_match(**overrides: object) -> Match:
    base: dict[str, object] = {
        "date": dt.date(2026, 6, 11),
        "competition": "FIFA World Cup",
        "season": "2026",
        "stage": "group",
        "home": "mexico",
        "away": "south_africa",
        "neutral": False,
        "home_goals": 2,
        "away_goals": 0,
    }
    base.update(overrides)
    return Match.model_validate(base)


class TestTeam:
    def test_valid(self) -> None:
        team = Team(team_id="usa", name="United States", aliases=frozenset({"USA", "USMNT"}))
        assert team.team_id == "usa"
        assert "USMNT" in team.aliases

    def test_id_must_be_slug(self) -> None:
        with pytest.raises(ValidationError):
            Team(team_id="United States", name="United States")

    def test_frozen(self) -> None:
        team = Team(team_id="usa", name="United States")
        with pytest.raises(ValidationError):
            team.name = "renamed"  # type: ignore[misc]


class TestMatch:
    def test_valid(self) -> None:
        match = make_match()
        assert match.home_goals == 2
        assert match.extras == {}

    def test_same_team_rejected(self) -> None:
        with pytest.raises(ValidationError, match="same team"):
            make_match(away="mexico")

    def test_negative_goals_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_match(home_goals=-1)

    def test_stage_optional(self) -> None:
        assert make_match(stage=None).stage is None

    def test_extras_sparse_floats(self) -> None:
        match = make_match(extras={"home_xg": 1.7})
        assert match.extras["home_xg"] == 1.7

    def test_frozen(self) -> None:
        match = make_match()
        with pytest.raises(ValidationError):
            match.home_goals = 5  # type: ignore[misc]

    def test_roundtrip_json(self) -> None:
        match = make_match()
        assert Match.model_validate_json(match.model_dump_json()) == match
