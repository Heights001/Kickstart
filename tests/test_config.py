from pathlib import Path

import pytest

from engine.core.config import (
    GoalMarginConfig,
    load_backtests,
    load_competition_tiers,
    load_engine_config,
)

REPO = Path(__file__).parent.parent
PACK = REPO / "packs/world_cup_2026"


class TestEngineConfig:
    def test_default_config_loads(self) -> None:
        config = load_engine_config(REPO / "configs/default.yaml")
        assert config.seed == 42
        assert config.elo.k_for("world_cup") == 60.0

    def test_unknown_tier_raises_with_known_tiers(self) -> None:
        config = load_engine_config(REPO / "configs/default.yaml")
        with pytest.raises(KeyError, match="no K-factor for tier 'galactic'"):
            config.elo.k_for("galactic")


class TestGoalMargin:
    @pytest.mark.parametrize(
        ("margin", "expected"),
        [(0, 1.0), (1, 1.0), (2, 1.5), (3, 1.75), (4, 1.875), (7, 2.25)],
    )
    def test_multiplier(self, margin: int, expected: float) -> None:
        gm = GoalMarginConfig(two_goals=1.5, three_goals=1.75, per_extra_goal=0.125)
        assert gm.multiplier(margin) == pytest.approx(expected)


class TestCompetitionTiers:
    def test_pack_mapping(self) -> None:
        tiers = load_competition_tiers(PACK)
        assert tiers.tier_of("FIFA World Cup") == "world_cup"
        assert tiers.tier_of("Copa América") == "continental"
        assert tiers.tier_of("FIFA World Cup qualification") == "qualifier"
        assert tiers.tier_of("UEFA Euro qualification") == "qualifier"
        assert tiers.tier_of("UEFA Nations League") == "nations_league"
        assert tiers.tier_of("Friendly") == "friendly"
        assert tiers.tier_of("Merdeka Tournament") == "other"

    def test_every_pack_tier_has_a_k_factor(self) -> None:
        config = load_engine_config(REPO / "configs/default.yaml")
        tiers = load_competition_tiers(PACK)
        referenced = set(tiers.tiers) | {tiers.qualification_tier, tiers.default_tier}
        for tier in referenced:
            config.elo.k_for(tier)  # raises if missing


class TestBacktests:
    def test_pack_windows(self) -> None:
        windows = load_backtests(PACK)
        assert [w.name for w in windows] == ["wc2014", "wc2018", "wc2022"]
        assert all(w.freeze < w.end for w in windows)
        assert all(w.competition == "FIFA World Cup" for w in windows)
