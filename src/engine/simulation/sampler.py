"""Match sampler for simulation (SCOPE §2.7).

Outcome comes from the calibrated classifier of whichever rung is promoted;
scorelines come from a double-Poisson grid (means from as-of attack/defence
ratings, clamped to configured bounds) conditioned on the sampled outcome.
Knockout draws resolve by renormalising the two win probabilities.

Venue approximation: a host playing a non-host gets home advantage; every other
pairing is neutral. Rest days for simulated matches are equal (diff 0).
"""

import datetime as dt
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from engine.core.config import CompetitionTiers, EngineConfig
from engine.core.schema import Match
from engine.evaluation.metrics import FloatArray
from engine.evaluation.walk_forward import fit_model_asof
from engine.models.base import OutcomeModel
from engine.models.calibration import Calibrator
from engine.models.ladder import RungSpec
from engine.ratings.store import MatchFeatures, ratings_as_of
from engine.simulation.format import FormatSpec

_SIM_REST_DAYS = 7


@dataclass(frozen=True)
class TeamState:
    """As-of rating components for one team."""

    elo: float
    form: float
    attack: float
    defence: float
    matches: int
    tier_matches: int


class MatchSampler:
    def __init__(
        self,
        teams: dict[str, TeamState],
        mu: float,
        default_state: TeamState,
        model: OutcomeModel,
        calibrator: Calibrator,
        rung: RungSpec,
        hosts: frozenset[str],
        config: EngineConfig,
        competition: str,
        as_of: dt.date,
    ) -> None:
        self._teams = teams
        self._mu = mu
        self._default = default_state
        self._model = model
        self.calibrator = calibrator
        self.rung = rung
        self._hosts = hosts
        self._sim = config.simulation
        self._competition = competition
        self._as_of = as_of
        self._probs_cache: dict[tuple[str, str], FloatArray] = {}
        self._score_cache: dict[tuple[str, str], list[FloatArray]] = {}

    @classmethod
    def build(
        cls,
        matches: list[Match],
        features: list[MatchFeatures],
        as_of: dt.date,
        spec: FormatSpec,
        config: EngineConfig,
        tiers: CompetitionTiers,
        rung: RungSpec | None = None,
    ) -> "MatchSampler":
        model, calibrator, used_rung = fit_model_asof(
            [f for f in features if f.date < as_of], as_of, config, rung
        )
        walker = ratings_as_of(matches, as_of, config, tiers)
        ad = walker.attack_defence
        tier = tiers.tier_of(spec.competition)
        team_states = {
            t: TeamState(
                elo=walker.elo.rating(t),
                form=walker.form.form(t),
                attack=ad.attack(t),
                defence=ad.defence(t),
                matches=walker.matches_played(t),
                tier_matches=walker.tier_matches(t, tier),
            )
            for t in spec.teams
        }
        default_state = TeamState(
            elo=config.elo.initial, form=0.5, attack=1.0, defence=1.0, matches=0, tier_matches=0
        )
        return cls(
            teams=team_states,
            mu=ad.mu,
            default_state=default_state,
            model=model,
            calibrator=calibrator,
            rung=used_rung,
            hosts=frozenset(spec.hosts),
            config=config,
            competition=spec.competition,
            as_of=as_of,
        )

    def outcome_probs(self, home: str, away: str) -> FloatArray:
        """(3,) probabilities [home win, draw, away win] with venue handling."""
        key = (home, away)
        cached = self._probs_cache.get(key)
        if cached is not None:
            return cached
        if away in self._hosts and home not in self._hosts:
            flipped = self.outcome_probs(away, home)
            probs = np.array([flipped[2], flipped[1], flipped[0]])
        else:
            neutral = not (home in self._hosts and away not in self._hosts)
            x = self.rung.extract([self._synthetic_features(home, away, neutral=neutral)])
            probs = self.calibrator.transform(self._model.predict_proba(x))[0]
        self._probs_cache[key] = probs
        return probs

    def _synthetic_features(self, home: str, away: str, *, neutral: bool) -> MatchFeatures:
        h, a = self._state(home), self._state(away)
        return MatchFeatures(
            date=self._as_of,
            competition=self._competition,
            home=home,
            away=away,
            neutral=neutral,
            elo_home=h.elo,
            elo_away=a.elo,
            form_home=h.form,
            form_away=a.form,
            attack_home=h.attack,
            attack_away=a.attack,
            defence_home=h.defence,
            defence_away=a.defence,
            rest_days_home=_SIM_REST_DAYS,
            rest_days_away=_SIM_REST_DAYS,
            matches_home=h.matches,
            matches_away=a.matches,
            tier_matches_home=h.tier_matches,
            tier_matches_away=a.tier_matches,
            outcome="D",  # placeholder; extractors never read the label
        )

    def knockout_home_win_prob(self, home: str, away: str) -> float:
        """P(home advances): draws split by renormalised win probabilities."""
        p = self.outcome_probs(home, away)
        return float(p[0] + p[1] * (p[0] / (p[0] + p[2])))

    def sample_scores(
        self, home: str, away: str, outcomes: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample (home_goals, away_goals) per run, conditioned on each run's outcome."""
        grids = self._conditional_grids(home, away)
        n = self._sim.max_goals + 1
        home_goals = np.zeros(len(outcomes), dtype=np.int64)
        away_goals = np.zeros(len(outcomes), dtype=np.int64)
        for outcome in (0, 1, 2):
            mask = outcomes == outcome
            count = int(mask.sum())
            if count == 0:
                continue
            flat_probs = grids[outcome]
            draws = rng.choice(len(flat_probs), size=count, p=flat_probs)
            home_goals[mask] = draws // n
            away_goals[mask] = draws % n
        return home_goals, away_goals

    def _conditional_grids(self, home: str, away: str) -> list[FloatArray]:
        key = (home, away)
        cached = self._score_cache.get(key)
        if cached is not None:
            return cached
        lam_home = self._lambda(home, away)
        lam_away = self._lambda(away, home)
        n = self._sim.max_goals + 1
        grid = np.outer(_poisson_pmf(lam_home, n), _poisson_pmf(lam_away, n))  # grid[h, a]
        hs, as_ = np.indices(grid.shape)
        masks = [hs > as_, hs == as_, hs < as_]  # H, D, A
        grids: list[FloatArray] = []
        for mask in masks:
            conditional = np.where(mask, grid, 0.0).ravel()
            grids.append(conditional / conditional.sum())
        self._score_cache[key] = grids
        return grids

    def _lambda(self, for_team: str, against_team: str) -> float:
        raw = self._mu * self._state(for_team).attack * self._state(against_team).defence
        return float(np.clip(raw, self._sim.lambda_floor, self._sim.lambda_cap))

    def _state(self, team: str) -> TeamState:
        return self._teams.get(team, self._default)


@lru_cache(maxsize=256)
def _poisson_pmf_tuple(lam: float, n: int) -> tuple[float, ...]:
    return tuple(math.exp(-lam) * lam**k / math.factorial(k) for k in range(n))


def _poisson_pmf(lam: float, n: int) -> FloatArray:
    return np.array(_poisson_pmf_tuple(round(lam, 6), n), dtype=np.float64)
