"""Monte Carlo tournament simulation (SCOPE §2.9).

Completed matches are facts; only remaining matches are sampled. Fully seeded:
same config + same data = same output. Remaining group fixtures are sampled
vectorised across all runs; per-run logic handles tables, thirds allocation, and
the knockout tree. Completed groups whose ranking needs no random tiebreak are
ranked once, not per run. Reported probabilities carry MC standard errors, which
exclude model uncertainty.
"""

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

from engine.simulation.format import FormatSpec
from engine.simulation.interpreter import (
    TeamRecord,
    allocate_thirds,
    compute_records,
    rank_group,
    rank_thirds,
    resolve_slot,
)
from engine.simulation.sampler import MatchSampler
from engine.simulation.state import KnockoutFact, TournamentState

ROUNDS = ("R32", "R16", "QF", "SF", "F", "champion")

# fixture -> sampled per-run goal arrays
_SampledFixture = tuple[str, str, np.ndarray, np.ndarray]


@dataclass
class SimulationResult:
    as_of: dt.date
    runs: int
    seed: int
    calibrator: str
    reach: dict[str, dict[str, float]] = field(default_factory=dict)  # team -> round -> p

    def standard_error(self, p: float) -> float:
        return float(np.sqrt(p * (1.0 - p) / self.runs))


def simulate(
    spec: FormatSpec,
    state: TournamentState,
    sampler: MatchSampler,
    runs: int,
    seed: int,
) -> SimulationResult:
    rng = np.random.default_rng(seed)
    reach_counts = {team: np.zeros(len(ROUNDS), dtype=np.int64) for team in spec.teams}
    round_index = {r: i for i, r in enumerate(ROUNDS)}

    sampled = _sample_remaining_fixtures(spec, state, sampler, runs, rng)
    stable_orders = _rank_stable_complete_groups(spec, state)
    facts_by_pair = _index_knockout_facts(state)

    fact_check_done = False
    for run in tqdm(range(runs), desc="simulating", unit=" runs"):
        group_positions: dict[str, list[str]] = {}
        thirds_records: dict[str, TeamRecord] = {}
        thirds_teams: dict[str, str] = {}
        for g, members in spec.groups.items():
            stable = stable_orders.get(g)
            if stable is not None:
                order, records = stable
            else:
                results = list(state.group_results[g])
                results.extend(
                    (home, away, int(hg[run]), int(ag[run])) for home, away, hg, ag in sampled[g]
                )
                order = rank_group(members, results, spec.points, spec.group_tiebreakers, rng)
                records = compute_records(members, results, spec.points)
            group_positions[g] = order
            thirds_teams[g] = order[2]
            thirds_records[g] = records[order[2]]

        qualified = rank_thirds(thirds_records, rng)[: spec.third_place.advance]
        third_slots = allocate_thirds(qualified, spec)

        outcomes: dict[int, tuple[str, str]] = {}
        used_facts: set[frozenset[str]] = set()
        for m in spec.knockout:
            home = resolve_slot(m.home, group_positions, third_slots, thirds_teams, outcomes)
            away = resolve_slot(m.away, group_positions, third_slots, thirds_teams, outcomes)
            pair = frozenset((home, away))
            fact = facts_by_pair.get(pair)
            if fact is not None and pair not in used_facts:
                winner = fact.winner
                used_facts.add(pair)
            else:
                p_home = sampler.knockout_home_win_prob(home, away)
                winner = home if rng.random() < p_home else away
            outcomes[m.id] = (winner, away if winner == home else home)

            idx = round_index.get(m.round)
            if idx is not None:
                reach_counts[home][idx] += 1
                reach_counts[away][idx] += 1
            elif m.round == "Final":
                final_idx = round_index["F"]
                reach_counts[home][final_idx] += 1
                reach_counts[away][final_idx] += 1
                reach_counts[winner][round_index["champion"]] += 1

        if not fact_check_done:
            unused = set(facts_by_pair) - used_facts
            if unused:
                raise ValueError(
                    f"{len(unused)} completed knockout matches never mapped onto the "
                    f"bracket: {sorted(tuple(sorted(p)) for p in unused)}"
                )
            fact_check_done = True

    reach = {
        team: {r: float(counts[i] / runs) for i, r in enumerate(ROUNDS)}
        for team, counts in reach_counts.items()
    }
    return SimulationResult(
        as_of=state.as_of,
        runs=runs,
        seed=seed,
        calibrator=sampler.calibrator.name,
        reach=reach,
    )


def _sample_remaining_fixtures(
    spec: FormatSpec,
    state: TournamentState,
    sampler: MatchSampler,
    runs: int,
    rng: np.random.Generator,
) -> dict[str, list[_SampledFixture]]:
    """Sample every remaining group fixture across all runs at once."""
    sampled: dict[str, list[_SampledFixture]] = {g: [] for g in spec.groups}
    for g, fixtures in state.remaining_group_fixtures.items():
        for home, away in fixtures:
            probs = sampler.outcome_probs(home, away)
            outcomes = rng.choice(3, size=runs, p=probs)
            home_goals, away_goals = sampler.sample_scores(home, away, outcomes, rng)
            sampled[g].append((home, away, home_goals, away_goals))
    return sampled


def _rank_stable_complete_groups(
    spec: FormatSpec, state: TournamentState
) -> dict[str, tuple[list[str], dict[str, TeamRecord]]]:
    """Rank finished groups once when no random tiebreak is involved.

    A group qualifies if all fixtures are played and two rankings with different
    rngs agree (i.e. the tiebreaker chain never reached the random step).
    """
    stable: dict[str, tuple[list[str], dict[str, TeamRecord]]] = {}
    for g, members in spec.groups.items():
        if state.remaining_group_fixtures[g]:
            continue
        results = state.group_results[g]
        first = rank_group(
            members, results, spec.points, spec.group_tiebreakers, np.random.default_rng(0)
        )
        second = rank_group(
            members, results, spec.points, spec.group_tiebreakers, np.random.default_rng(1)
        )
        if first == second:
            stable[g] = (first, compute_records(members, results, spec.points))
    return stable


def _index_knockout_facts(state: TournamentState) -> dict[frozenset[str], KnockoutFact]:
    facts: dict[frozenset[str], KnockoutFact] = {}
    for fact in state.knockout_facts:
        if fact.pair in facts:
            raise ValueError(f"duplicate knockout fact for pair {sorted(fact.pair)}")
        facts[fact.pair] = fact
    return facts
