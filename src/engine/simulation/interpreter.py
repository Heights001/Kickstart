"""Format interpreter: group ranking with the full tiebreaker chain, best-thirds
ranking, allocation-table lookup, and knockout slot resolution.

All randomness (final tiebreak step) flows through the caller's ``rng`` so a
seeded simulation is bit-for-bit reproducible (CLAUDE.md rule 5).
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from engine.simulation.format import FormatSpec, PointsRule

# (home, away, home_goals, away_goals)
Result = tuple[str, str, int, int]


@dataclass(frozen=True)
class TeamRecord:
    points: int
    goal_difference: int
    goals_for: int

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.points, self.goal_difference, self.goals_for)


def compute_records(
    teams: list[str], results: list[Result], points: PointsRule
) -> dict[str, TeamRecord]:
    pts = dict.fromkeys(teams, 0)
    gf = dict.fromkeys(teams, 0)
    ga = dict.fromkeys(teams, 0)
    for home, away, hg, ag in results:
        gf[home] += hg
        ga[home] += ag
        gf[away] += ag
        ga[away] += hg
        if hg > ag:
            pts[home] += points.win
            pts[away] += points.loss
        elif hg < ag:
            pts[away] += points.win
            pts[home] += points.loss
        else:
            pts[home] += points.draw
            pts[away] += points.draw
    return {t: TeamRecord(pts[t], gf[t] - ga[t], gf[t]) for t in teams}


def rank_group(
    teams: list[str],
    results: list[Result],
    points: PointsRule,
    tiebreakers: list[str],
    rng: np.random.Generator,
) -> list[str]:
    """Order a group best-first per the tiebreaker chain.

    ``points``/``goal_difference``/``goals_for`` sort on overall record;
    ``head_to_head`` re-ranks still-tied teams on matches among themselves only;
    ``random`` breaks anything left via ``rng``.
    """
    records = compute_records(teams, results, points)
    ordered = sorted(teams, key=lambda t: records[t].key, reverse=True)

    final: list[str] = []
    for tied in _tie_classes(ordered, lambda t: records[t].key):
        if len(tied) > 1 and "head_to_head" in tiebreakers:
            tied = _head_to_head(tied, results, points, rng)
        elif len(tied) > 1:
            tied = _shuffled(tied, rng)
        final.extend(tied)
    return final


def _head_to_head(
    tied: list[str], results: list[Result], points: PointsRule, rng: np.random.Generator
) -> list[str]:
    subset = set(tied)
    mutual = [r for r in results if r[0] in subset and r[1] in subset]
    records = compute_records(tied, mutual, points)
    ordered = sorted(tied, key=lambda t: records[t].key, reverse=True)
    final: list[str] = []
    for still_tied in _tie_classes(ordered, lambda t: records[t].key):
        final.extend(_shuffled(still_tied, rng) if len(still_tied) > 1 else still_tied)
    return final


def rank_thirds(group_thirds: dict[str, TeamRecord], rng: np.random.Generator) -> list[str]:
    """Order group letters by their third-placed team's record, best first."""
    ordered = sorted(group_thirds, key=lambda g: group_thirds[g].key, reverse=True)
    final: list[str] = []
    for tied in _tie_classes(ordered, lambda g: group_thirds[g].key):
        final.extend(_shuffled(tied, rng) if len(tied) > 1 else tied)
    return final


def allocate_thirds(qualified_groups: list[str], spec: FormatSpec) -> dict[str, str]:
    """Map allocation slots (e.g. ``1E``) to the qualified third's group letter."""
    combination = "".join(sorted(qualified_groups))
    try:
        return spec.allocations[combination]
    except KeyError:
        raise KeyError(f"combination {combination!r} missing from allocation table") from None


def resolve_slot(
    slot: str,
    group_positions: dict[str, list[str]],
    third_slots: dict[str, str],
    group_thirds_teams: dict[str, str],
    knockout_outcomes: dict[int, tuple[str, str]],
) -> str:
    """Resolve a slot expression to a team id.

    ``group_positions``: group letter -> teams best-first. ``third_slots``: slot
    label -> third's group. ``group_thirds_teams``: group letter -> third-placed
    team. ``knockout_outcomes``: match id -> (winner, loser).
    """
    if slot.startswith("T"):
        return group_thirds_teams[third_slots[slot[1:]]]
    if slot.startswith("W"):
        return knockout_outcomes[int(slot[1:])][0]
    if slot.startswith("L"):
        return knockout_outcomes[int(slot[1:])][1]
    position, group = int(slot[0]) - 1, slot[1:]
    return group_positions[group][position]


def _tie_classes[T](ordered: list[T], key: Callable[[T], object]) -> list[list[T]]:
    classes: list[list[T]] = []
    previous: object = object()
    for item in ordered:
        k = key(item)
        if classes and k == previous:
            classes[-1].append(item)
        else:
            classes.append([item])
        previous = k
    return classes


def _shuffled[T](items: list[T], rng: np.random.Generator) -> list[T]:
    order = rng.permutation(len(items))
    return [items[i] for i in order]
