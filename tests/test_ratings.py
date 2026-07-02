import datetime as dt
from pathlib import Path

import pytest

from engine.core.config import EngineConfig, load_competition_tiers, load_engine_config
from engine.core.schema import Match
from engine.ratings.attack_defence import AttackDefenceRater
from engine.ratings.elo import EloRater
from engine.ratings.form import FormRater
from engine.ratings.store import RatingsWalker, build_features, ratings_as_of

REPO = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def config() -> EngineConfig:
    return load_engine_config(REPO / "configs/default.yaml")


def make_match(
    home: str = "brazil",
    away: str = "germany",
    home_goals: int = 2,
    away_goals: int = 0,
    *,
    neutral: bool = False,
    competition: str = "FIFA World Cup",
    date: dt.date = dt.date(2022, 11, 25),
) -> Match:
    return Match(
        date=date,
        competition=competition,
        season=str(date.year),
        home=home,
        away=away,
        neutral=neutral,
        home_goals=home_goals,
        away_goals=away_goals,
    )


class TestElo:
    def test_hand_computed_home_win(self, config: EngineConfig) -> None:
        # Both 1500, home advantage 80: We = 1/(1+10^(-80/400)) = 0.61314.
        # 2-0 in a World Cup match: K=60, margin multiplier 1.5.
        # delta = 60 * 1.5 * (1 - 0.61314) = 34.8172
        elo = EloRater(config.elo)
        elo.update(make_match(), tier="world_cup")
        assert elo.rating("brazil") == pytest.approx(1534.8172, abs=1e-3)
        assert elo.rating("germany") == pytest.approx(1465.1828, abs=1e-3)

    def test_hand_computed_neutral_draw_friendly(self, config: EngineConfig) -> None:
        # Neutral venue: We = 0.5 exactly; a draw changes nothing.
        elo = EloRater(config.elo)
        elo.update(
            make_match(home_goals=1, away_goals=1, neutral=True, competition="Friendly"),
            tier="friendly",
        )
        assert elo.rating("brazil") == 1500.0
        assert elo.rating("germany") == 1500.0

    def test_neutral_flag_respected(self, config: EngineConfig) -> None:
        elo = EloRater(config.elo)
        assert elo.expected_home_score("brazil", "germany", neutral=True) == 0.5
        assert elo.expected_home_score("brazil", "germany", neutral=False) > 0.5

    def test_zero_sum(self, config: EngineConfig) -> None:
        elo = EloRater(config.elo)
        elo.update(make_match(home_goals=5, away_goals=1), tier="qualifier")
        assert elo.rating("brazil") + elo.rating("germany") == pytest.approx(3000.0)

    def test_k_scales_with_tier(self, config: EngineConfig) -> None:
        wc, friendly = EloRater(config.elo), EloRater(config.elo)
        wc.update(make_match(), tier="world_cup")
        friendly.update(make_match(competition="Friendly"), tier="friendly")
        wc_delta = wc.rating("brazil") - 1500.0
        friendly_delta = friendly.rating("brazil") - 1500.0
        assert wc_delta == pytest.approx(friendly_delta * 60.0 / 20.0)


class TestForm:
    def test_hand_computed_decay(self, config: EngineConfig) -> None:
        # decay 0.85: 0.5 -> win -> 0.575 -> loss -> 0.48875
        form = FormRater(config.form)
        form.update(make_match(home_goals=1, away_goals=0))
        assert form.form("brazil") == pytest.approx(0.575)
        form.update(make_match(home_goals=0, away_goals=3))
        assert form.form("brazil") == pytest.approx(0.48875)
        assert form.form("germany") == pytest.approx(0.85 * 0.425 + 0.15)


class TestAttackDefence:
    def test_hand_computed_update(self, config: EngineConfig) -> None:
        # mu=1.3, all ratings 1.0, home wins 4-0 with alpha 0.1:
        # attack_home = 1 + 0.1*(4/1.3 - 1) = 1.20769; defence_away same;
        # attack_away = 1 + 0.1*(0 - 1) = 0.9; defence_home = 0.9;
        # mu = 1.3 + 0.02*(2 - 1.3) = 1.314
        rater = AttackDefenceRater(config.attack_defence)
        rater.update(make_match(home_goals=4, away_goals=0))
        assert rater.attack("brazil") == pytest.approx(1.20769, abs=1e-4)
        assert rater.defence("germany") == pytest.approx(1.20769, abs=1e-4)
        assert rater.attack("germany") == pytest.approx(0.9)
        assert rater.defence("brazil") == pytest.approx(0.9)
        assert rater.mu == pytest.approx(1.314)

    def test_expected_goals_floor(self, config: EngineConfig) -> None:
        rater = AttackDefenceRater(config.attack_defence)
        for _ in range(50):
            rater.update(make_match(home_goals=0, away_goals=0))
        assert rater.expected_goals("brazil", "germany") >= config.attack_defence.floor


class TestStore:
    @pytest.fixture
    def walker(self, config: EngineConfig) -> RatingsWalker:
        tiers = load_competition_tiers(REPO / "packs/world_cup_2026")
        return RatingsWalker(config, tiers)

    def test_snapshot_precedes_update(self, walker: RatingsWalker, config: EngineConfig) -> None:
        # A match's own result must not appear in its features (rule 3).
        features = walker.observe(make_match(home_goals=7, away_goals=0))
        assert features.elo_home == config.elo.initial
        assert features.elo_away == config.elo.initial
        assert features.form_home == 0.5
        assert features.rest_days_home == config.rest.default_days
        assert features.outcome == "H"

    def test_second_match_sees_first_result(self, walker: RatingsWalker) -> None:
        walker.observe(make_match(date=dt.date(2022, 11, 25)))
        features = walker.observe(make_match(date=dt.date(2022, 11, 28)))
        assert features.elo_home > features.elo_away
        assert features.rest_days_home == 3
        assert features.rest_days_away == 3

    def test_out_of_order_raises(self, walker: RatingsWalker) -> None:
        walker.observe(make_match(date=dt.date(2022, 11, 25)))
        with pytest.raises(ValueError, match="out of order"):
            walker.observe(make_match(date=dt.date(2022, 11, 24)))

    def test_ratings_as_of_excludes_boundary(self, config: EngineConfig) -> None:
        tiers = load_competition_tiers(REPO / "packs/world_cup_2026")
        matches = [
            make_match(date=dt.date(2022, 11, 25)),
            make_match(date=dt.date(2022, 11, 28), home_goals=0, away_goals=4),
        ]
        walker = ratings_as_of(matches, dt.date(2022, 11, 28), config, tiers)
        # Only the first match (2-0 Brazil) is visible; the 0-4 on the as_of
        # date itself must be excluded.
        assert walker.elo.rating("brazil") > config.elo.initial

    def test_build_features_sorts_by_date(self, config: EngineConfig) -> None:
        tiers = load_competition_tiers(REPO / "packs/world_cup_2026")
        matches = [
            make_match(date=dt.date(2022, 11, 28)),
            make_match(date=dt.date(2022, 11, 25)),
        ]
        features = list(build_features(matches, config, tiers))
        assert [f.date for f in features] == [dt.date(2022, 11, 25), dt.date(2022, 11, 28)]
