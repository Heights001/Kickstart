import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from engine.core.schema import Match
from engine.simulation.format import FormatSpec, PointsRule, load_format
from engine.simulation.interpreter import (
    TeamRecord,
    allocate_thirds,
    compute_records,
    rank_group,
    rank_thirds,
    resolve_slot,
)
from engine.simulation.state import build_state

REPO = Path(__file__).parent.parent
PACK = REPO / "packs/world_cup_2026"
POINTS = PointsRule(win=3, draw=1, loss=0)
CHAIN = ["points", "goal_difference", "goals_for", "head_to_head", "random"]


def rng() -> np.random.Generator:
    return np.random.default_rng(42)


class TestRankGroup:
    def test_golden_head_to_head(self) -> None:
        """Hand-computed: w/x tie on pts=6, gd=+2, gf=3; h2h (w beat x) decides."""
        teams = ["w", "x", "y", "z"]
        results = [
            ("w", "x", 1, 0),
            ("x", "y", 2, 0),
            ("y", "w", 1, 0),
            ("w", "z", 2, 0),
            ("x", "z", 1, 0),
            ("y", "z", 1, 0),
        ]
        records = compute_records(teams, results, POINTS)
        assert records["w"].key == (6, 2, 3)
        assert records["x"].key == (6, 2, 3)
        assert records["y"].key == (6, 0, 2)
        assert records["z"].key == (0, -4, 0)
        assert rank_group(teams, results, POINTS, CHAIN, rng()) == ["w", "x", "y", "z"]

    def test_golden_gd_then_gf(self) -> None:
        """All beat z; cycle w>x>y with scores making gd then gf decisive."""
        teams = ["w", "x", "y", "z"]
        results = [
            ("w", "x", 3, 0),  # w gd boost
            ("x", "y", 2, 1),
            ("y", "w", 1, 0),
            ("w", "z", 1, 0),
            ("x", "z", 2, 0),
            ("y", "z", 2, 0),
        ]
        records = compute_records(teams, results, POINTS)
        assert records["w"].key == (6, 3, 4)
        assert records["x"].key == (6, 0, 4)
        assert records["y"].key == (6, 1, 4)
        assert rank_group(teams, results, POINTS, CHAIN, rng()) == ["w", "y", "x", "z"]

    def test_full_tie_random_is_seeded(self) -> None:
        teams = ["w", "x", "y", "z"]
        results = [(a, b, 0, 0) for i, a in enumerate(teams) for b in teams[i + 1 :]]
        first = rank_group(teams, results, POINTS, CHAIN, np.random.default_rng(7))
        second = rank_group(teams, results, POINTS, CHAIN, np.random.default_rng(7))
        assert first == second
        assert sorted(first) == teams

    def test_incomplete_results_still_rank(self) -> None:
        teams = ["w", "x", "y", "z"]
        results = [("w", "x", 2, 0)]
        order = rank_group(teams, results, POINTS, CHAIN, rng())
        assert order[0] == "w"
        assert order[-1] == "x" or order[1] in ("y", "z")


class TestThirds:
    def test_rank_thirds_by_record(self) -> None:
        records = {
            "A": TeamRecord(points=0, goal_difference=0, goals_for=0),
            "B": TeamRecord(points=3, goal_difference=2, goals_for=2),
            "C": TeamRecord(points=3, goal_difference=1, goals_for=1),
        }
        assert rank_thirds(records, rng()) == ["B", "C", "A"]

    def test_allocation_lookup_live_combination(self) -> None:
        """The actual WC2026 combination, corroborated by real R32 results."""
        spec = load_format(PACK)
        slots = allocate_thirds(list("BDEFGJKL"), spec)
        assert slots == {
            "1A": "E",
            "1B": "G",
            "1D": "B",
            "1E": "D",  # Germany vs Paraguay — happened
            "1G": "J",
            "1I": "F",  # France vs Sweden — happened
            "1K": "L",
            "1L": "K",
        }

    def test_unknown_combination_raises(self) -> None:
        spec = load_format(PACK)
        with pytest.raises(KeyError, match="AAAAAAAA"):
            allocate_thirds(list("AAAAAAAA"), spec)


class TestResolveSlot:
    def test_grammar(self) -> None:
        group_positions = {"A": ["a1", "a2", "a3", "a4"], "B": ["b1", "b2", "b3", "b4"]}
        third_slots = {"1A": "B"}
        thirds_teams = {"B": "b3"}
        outcomes = {73: ("a2", "b2")}
        resolve = lambda s: resolve_slot(s, group_positions, third_slots, thirds_teams, outcomes)  # noqa: E731
        assert resolve("1A") == "a1"
        assert resolve("2B") == "b2"
        assert resolve("T1A") == "b3"
        assert resolve("W73") == "a2"
        assert resolve("L73") == "b2"


class TestFormatSpec:
    def test_real_pack_loads_and_validates(self) -> None:
        spec = load_format(PACK)
        assert len(spec.teams) == 48
        assert len(spec.groups) == 12
        assert len(spec.knockout) == 32
        assert len(spec.allocations) == 495
        assert spec.allocation_slots == ["1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L"]

    def test_duplicate_team_rejected(self) -> None:
        with pytest.raises(ValueError, match="more than one group"):
            FormatSpec(
                competition="X",
                season="1",
                start=dt.date(2026, 1, 1),
                end=dt.date(2026, 2, 1),
                hosts=[],
                points=POINTS,
                groups={"A": ["t1", "t2"], "B": ["t1", "t3"]},
                group_tiebreakers=CHAIN,
                third_place={"advance": 1, "tiebreakers": CHAIN, "allocation": "x.yaml"},
                knockout=[],
            )


class TestBuildState:
    @staticmethod
    def match(date: dt.date, home: str, away: str, hg: int, ag: int, **kw: object) -> Match:
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

    def test_group_vs_knockout_partition(self) -> None:
        spec = load_format(PACK)
        matches = [
            # group A meeting
            self.match(dt.date(2026, 6, 12), "mexico", "south_africa", 2, 0),
            # cross-group => knockout pool
            self.match(dt.date(2026, 6, 29), "canada", "south_africa", 1, 0),
            # drawn knockout with shootout winner
            self.match(dt.date(2026, 6, 30), "paraguay", "germany", 1, 1, winner="paraguay"),
            # after as_of: ignored
            self.match(dt.date(2026, 7, 10), "france", "spain", 1, 0),
        ]
        state = build_state(matches, spec, as_of=dt.date(2026, 7, 2))
        assert state.group_results["A"] == [("mexico", "south_africa", 2, 0)]
        assert len(state.knockout_facts) == 2
        assert state.knockout_facts[1].winner == "paraguay"
        assert len(state.remaining_group_fixtures["A"]) == 5
        assert state.completed_count == 3

    def test_drawn_knockout_without_winner_raises(self) -> None:
        spec = load_format(PACK)
        matches = [self.match(dt.date(2026, 6, 29), "canada", "germany", 1, 1)]
        with pytest.raises(ValueError, match="no shootout"):
            build_state(matches, spec, as_of=dt.date(2026, 7, 2))

    def test_rewind_has_no_facts(self) -> None:
        spec = load_format(PACK)
        matches = [self.match(dt.date(2026, 6, 12), "mexico", "south_africa", 2, 0)]
        state = build_state(matches, spec, as_of=dt.date(2026, 6, 10))
        assert state.completed_count == 0
        assert all(len(f) == 6 for f in state.remaining_group_fixtures.values())
