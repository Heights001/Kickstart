import datetime as dt
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from engine.core.config import EngineConfig, load_engine_config
from engine.core.schema import Match
from engine.evaluation.walk_forward import fit_model_asof
from engine.simulation.format import load_format
from engine.simulation.monte_carlo import _rank_stable_complete_groups, simulate
from engine.simulation.sampler import MatchSampler
from engine.simulation.state import build_state
from tests.test_models import synthetic_dataset

REPO = Path(__file__).parent.parent
PACK = REPO / "packs/world_cup_2026"


@pytest.fixture(scope="module")
def config() -> EngineConfig:
    return load_engine_config(REPO / "configs/default.yaml")


@pytest.fixture(scope="module")
def sampler(config: EngineConfig) -> MatchSampler:
    """Sampler with a model fitted on synthetic data and hand-set ratings."""
    features = synthetic_dataset(3000, dt.date(2000, 1, 1), seed=11)
    model, calibrator = fit_model_asof(features, dt.date(2011, 1, 1), config)
    spec = load_format(PACK)
    elo = dict.fromkeys(spec.teams, 1600.0)
    elo["argentina"] = 2100.0
    elo["jordan"] = 1300.0
    return MatchSampler(
        elo=elo,
        attack=dict.fromkeys(spec.teams, 1.0),
        defence=dict.fromkeys(spec.teams, 1.0),
        mu=1.3,
        default_elo=config.elo.initial,
        model=model,
        calibrator=calibrator,
        hosts=frozenset(spec.hosts),
        config=config,
    )


class TestSampler:
    def test_probs_sum_to_one(self, sampler: MatchSampler) -> None:
        probs = sampler.outcome_probs("argentina", "jordan")
        assert probs.sum() == pytest.approx(1.0)
        assert probs[0] > probs[2]  # much stronger side should be favoured

    def test_orientation_symmetry_on_neutral(self, sampler: MatchSampler) -> None:
        ab = sampler.outcome_probs("argentina", "jordan")
        ba = sampler.outcome_probs("jordan", "argentina")
        assert ab[0] == pytest.approx(ba[2], abs=1e-9)
        assert ab[1] == pytest.approx(ba[1], abs=1e-9)

    def test_host_gets_home_advantage(self, sampler: MatchSampler) -> None:
        neutral_pair = sampler.outcome_probs("argentina", "england")
        host_home = sampler.outcome_probs("usa", "england")
        host_away = sampler.outcome_probs("england", "usa")
        # usa (host) vs england is not neutral: usa should be favoured relative
        # to the same elo gap on neutral ground, and orientation must not matter.
        assert host_home[0] == pytest.approx(host_away[2], abs=1e-9)
        assert neutral_pair[1] != host_home[1] or neutral_pair[0] != host_home[0]

    def test_knockout_renormalisation(self, sampler: MatchSampler) -> None:
        p = sampler.outcome_probs("argentina", "jordan")
        expected = p[0] + p[1] * p[0] / (p[0] + p[2])
        assert sampler.knockout_home_win_prob("argentina", "jordan") == pytest.approx(expected)

    def test_scores_respect_outcome(self, sampler: MatchSampler) -> None:
        rng = np.random.default_rng(3)
        outcomes = np.array([0] * 100 + [1] * 100 + [2] * 100)
        hg, ag = sampler.sample_scores("argentina", "jordan", outcomes, rng)
        assert (hg[:100] > ag[:100]).all()
        assert (hg[100:200] == ag[100:200]).all()
        assert (hg[200:] < ag[200:]).all()
        assert hg.max() <= 10


def wc_match(date: dt.date, home: str, away: str, hg: int, ag: int, **kw: object) -> Match:
    return Match(
        date=date,
        competition="FIFA World Cup",
        season="2026",
        home=home,
        away=away,
        neutral=True,
        home_goals=hg,
        away_goals=ag,
        **kw,  # type: ignore[arg-type]
    )


class TestSimulate:
    def test_rewind_full_simulation(self, config: EngineConfig, sampler: MatchSampler) -> None:
        spec = load_format(PACK)
        state = build_state([], spec, as_of=dt.date(2026, 6, 10))
        result = simulate(spec, state, sampler, runs=300, seed=config.seed)

        champion_total = sum(p["champion"] for p in result.reach.values())
        assert champion_total == pytest.approx(1.0)
        r32_total = sum(p["R32"] for p in result.reach.values())
        assert r32_total == pytest.approx(32.0)
        final_total = sum(p["F"] for p in result.reach.values())
        assert final_total == pytest.approx(2.0)
        # Argentina (elo 2100) should clearly beat Jordan (1300) over many runs.
        assert result.reach["argentina"]["champion"] > result.reach["jordan"]["champion"]

    def test_fixed_seed_bitwise_repeatable(
        self, config: EngineConfig, sampler: MatchSampler
    ) -> None:
        spec = load_format(PACK)
        state = build_state([], spec, as_of=dt.date(2026, 6, 10))
        # Fresh sampler caches are populated on first use; rerunning with the same
        # seed must give identical probabilities.
        a = simulate(spec, state, sampler, runs=150, seed=7)
        b = simulate(spec, state, sampler, runs=150, seed=7)
        assert a.reach == b.reach

    def test_eliminated_team_has_zero_champion_probability(
        self, config: EngineConfig, sampler: MatchSampler
    ) -> None:
        """Feed one completed group: the team mathematically eliminated must be 0.0."""
        spec = load_format(PACK)
        matches = [
            # Group A complete: mexico 9pts, south_africa 6, south_korea 3, czech 0
            wc_match(dt.date(2026, 6, 12), "mexico", "south_africa", 2, 0),
            wc_match(dt.date(2026, 6, 13), "mexico", "south_korea", 2, 0),
            wc_match(dt.date(2026, 6, 14), "mexico", "czech_republic", 2, 0),
            wc_match(dt.date(2026, 6, 15), "south_africa", "south_korea", 1, 0),
            wc_match(dt.date(2026, 6, 16), "south_africa", "czech_republic", 1, 0),
            wc_match(dt.date(2026, 6, 17), "south_korea", "czech_republic", 1, 0),
        ]
        state = build_state(matches, spec, as_of=dt.date(2026, 6, 20))
        result = simulate(spec, state, sampler, runs=200, seed=1)
        assert result.reach["czech_republic"]["champion"] == 0.0
        assert result.reach["czech_republic"]["R32"] == 0.0
        assert result.reach["mexico"]["R32"] == 1.0

    def test_knockout_facts_pin_outcomes(self, config: EngineConfig, sampler: MatchSampler) -> None:
        """With all 72 group matches + one R32 fact, that R32 result is pinned."""
        spec = load_format(PACK)
        matches = []
        for members in spec.groups.values():
            for i, (a, b) in enumerate(combinations(members, 2)):
                # First listed team wins 1-0, except fixture 4 drawn: strict
                # 9/4/3/1 tables, no random tiebreaks inside groups.
                hg, ag = (2, 2) if i == 4 else (1, 0)
                matches.append(wc_match(dt.date(2026, 6, 12), a, b, hg, ag))
        state = build_state(matches, spec, as_of=dt.date(2026, 6, 28))
        assert all(not f for f in state.remaining_group_fixtures.values())

        # Determine who meets in match 73 (2A vs 2B) and pin it with a fact.
        stable = _rank_stable_complete_groups(spec, state)
        assert set(stable) == set(spec.groups)
        second_a = stable["A"][0][1]
        second_b = stable["B"][0][1]
        matches.append(wc_match(dt.date(2026, 6, 29), second_a, second_b, 0, 0, winner=second_b))
        state = build_state(matches, spec, as_of=dt.date(2026, 7, 1))
        assert len(state.knockout_facts) == 1

        result = simulate(spec, state, sampler, runs=100, seed=2)
        # The shootout loser of match 73 can never reach R16.
        assert result.reach[second_a]["R16"] == 0.0
        assert result.reach[second_b]["R16"] == 1.0
