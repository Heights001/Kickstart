"""Live tournament state: completed matches are facts, the rest is simulated.

Group facts are same-group pairs' first meeting inside the tournament window;
any later tournament match (including a same-group rematch in the knockout) goes
to the knockout facts pool, matched to bracket slots during simulation.
"""

import datetime as dt
from dataclasses import dataclass, field
from itertools import combinations

from engine.core.schema import Match
from engine.simulation.format import FormatSpec
from engine.simulation.interpreter import Result


@dataclass(frozen=True)
class KnockoutFact:
    date: dt.date
    home: str
    away: str
    home_goals: int
    away_goals: int
    winner: str  # from goals, or the shootout annotation for draws

    @property
    def pair(self) -> frozenset[str]:
        return frozenset((self.home, self.away))


@dataclass
class TournamentState:
    as_of: dt.date
    group_results: dict[str, list[Result]]  # group letter -> completed results
    remaining_group_fixtures: dict[str, list[tuple[str, str]]]
    knockout_facts: list[KnockoutFact] = field(default_factory=list)

    @property
    def completed_count(self) -> int:
        group = sum(len(r) for r in self.group_results.values())
        return group + len(self.knockout_facts)


def build_state(matches: list[Match], spec: FormatSpec, as_of: dt.date) -> TournamentState:
    teams = spec.teams
    group_of = {team: g for g, members in spec.groups.items() for team in members}

    tournament = sorted(
        (
            m
            for m in matches
            if m.competition == spec.competition
            and spec.start <= m.date <= spec.end
            and m.date < as_of
            and m.home in teams
            and m.away in teams
        ),
        key=lambda m: m.date,
    )

    group_results: dict[str, list[Result]] = {g: [] for g in spec.groups}
    played_pairs: set[frozenset[str]] = set()
    knockout_facts: list[KnockoutFact] = []
    for m in tournament:
        pair = frozenset((m.home, m.away))
        same_group = group_of[m.home] == group_of[m.away]
        if same_group and pair not in played_pairs:
            played_pairs.add(pair)
            group_results[group_of[m.home]].append((m.home, m.away, m.home_goals, m.away_goals))
        else:
            knockout_facts.append(_knockout_fact(m))

    remaining: dict[str, list[tuple[str, str]]] = {}
    for g, members in spec.groups.items():
        remaining[g] = [
            (a, b) for a, b in combinations(members, 2) if frozenset((a, b)) not in played_pairs
        ]
    return TournamentState(
        as_of=as_of,
        group_results=group_results,
        remaining_group_fixtures=remaining,
        knockout_facts=knockout_facts,
    )


def _knockout_fact(m: Match) -> KnockoutFact:
    if m.home_goals > m.away_goals:
        winner = m.home
    elif m.home_goals < m.away_goals:
        winner = m.away
    elif m.winner is not None:
        winner = m.winner
    else:
        raise ValueError(
            f"drawn knockout match {m.home} vs {m.away} on {m.date} has no shootout "
            "winner — refresh the shootouts source"
        )
    return KnockoutFact(m.date, m.home, m.away, m.home_goals, m.away_goals, winner)
