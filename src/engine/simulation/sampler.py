"""Match sampler for simulation (SCOPE §2.7).

Outcome comes from the calibrated classifier; scorelines come from a double-Poisson
grid (means from as-of attack/defence ratings, clamped to configured bounds)
conditioned on the sampled outcome. Knockout draws resolve by renormalising the two
win probabilities.

Venue approximation: a host playing a non-host gets home advantage; every other
pairing is neutral. Rest-day difference for simulated matches is 0.
"""

import datetime as dt
import math
from functools import lru_cache

import numpy as np

from engine.core.config import CompetitionTiers, EngineConfig
from engine.core.schema import Match
from engine.evaluation.metrics import FloatArray
from engine.evaluation.walk_forward import fit_model_asof
from engine.models.calibration import Calibrator
from engine.models.rung0 import Rung0Model
from engine.ratings.store import MatchFeatures, ratings_as_of
from engine.simulation.format import FormatSpec


class MatchSampler:
    def __init__(
        self,
        elo: dict[str, float],
        attack: dict[str, float],
        defence: dict[str, float],
        mu: float,
        default_elo: float,
        model: Rung0Model,
        calibrator: Calibrator,
        hosts: frozenset[str],
        config: EngineConfig,
    ) -> None:
        self._elo = elo
        self._attack = attack
        self._defence = defence
        self._mu = mu
        self._default_elo = default_elo
        self._model = model
        self.calibrator = calibrator
        self._hosts = hosts
        self._sim = config.simulation
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
    ) -> "MatchSampler":
        model, calibrator = fit_model_asof([f for f in features if f.date < as_of], as_of, config)
        walker = ratings_as_of(matches, as_of, config, tiers)
        ad = walker.attack_defence
        teams = spec.teams
        return cls(
            elo={t: walker.elo.rating(t) for t in teams},
            attack={t: ad.attack(t) for t in teams},
            defence={t: ad.defence(t) for t in teams},
            mu=ad.mu,
            default_elo=config.elo.initial,
            model=model,
            calibrator=calibrator,
            hosts=frozenset(spec.hosts),
            config=config,
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
            x = np.array(
                [[self._elo_of(home) - self._elo_of(away), float(neutral), 0.0]],
                dtype=np.float64,
            )
            probs = self.calibrator.transform(self._model.predict_proba(x))[0]
        self._probs_cache[key] = probs
        return probs

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
        pmf_home = _poisson_pmf(lam_home, n)
        pmf_away = _poisson_pmf(lam_away, n)
        grid = np.outer(pmf_home, pmf_away)  # grid[h, a]
        hs, as_ = np.indices(grid.shape)
        masks = [hs > as_, hs == as_, hs < as_]  # H, D, A
        grids: list[FloatArray] = []
        for mask in masks:
            conditional = np.where(mask, grid, 0.0).ravel()
            grids.append(conditional / conditional.sum())
        self._score_cache[key] = grids
        return grids

    def _lambda(self, for_team: str, against_team: str) -> float:
        raw = self._mu * self._attack.get(for_team, 1.0) * self._defence.get(against_team, 1.0)
        return float(np.clip(raw, self._sim.lambda_floor, self._sim.lambda_cap))

    def _elo_of(self, team: str) -> float:
        return self._elo.get(team, self._default_elo)


@lru_cache(maxsize=256)
def _poisson_pmf_tuple(lam: float, n: int) -> tuple[float, ...]:
    return tuple(math.exp(-lam) * lam**k / math.factorial(k) for k in range(n))


def _poisson_pmf(lam: float, n: int) -> FloatArray:
    return np.array(_poisson_pmf_tuple(round(lam, 6), n), dtype=np.float64)
