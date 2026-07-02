"""Orchestrates ``engine simulate``: load pack + data, build state, run MC, log history."""

import datetime as dt
import json
from pathlib import Path

from engine.core.config import load_competition_tiers, load_engine_config
from engine.ingestion.pipeline import load_matches
from engine.ratings.pipeline import load_features
from engine.simulation.format import FormatSpec, load_format
from engine.simulation.monte_carlo import ROUNDS, SimulationResult, simulate
from engine.simulation.sampler import MatchSampler
from engine.simulation.state import build_state
from engine.utils.logging import get_logger, log_with

logger = get_logger(__name__)

PROB_HISTORY = "prob_history.jsonl"


def run_simulation(
    pack_dir: Path,
    data_dir: Path,
    config_path: Path,
    as_of: dt.date,
    runs: int | None = None,
) -> tuple[SimulationResult, FormatSpec]:
    config = load_engine_config(config_path)
    spec = load_format(pack_dir)
    tiers = load_competition_tiers(pack_dir)
    matches = list(load_matches(data_dir / "processed" / "matches.jsonl"))
    features = load_features(data_dir / "processed" / "features.jsonl")

    state = build_state(matches, spec, as_of)
    sampler = MatchSampler.build(matches, features, as_of, spec, config, tiers)
    n_runs = runs if runs is not None else config.simulation.runs
    log_with(
        logger,
        "simulation starting",
        as_of=str(as_of),
        runs=n_runs,
        completed_matches=state.completed_count,
        calibrator=sampler.calibrator.name,
    )
    result = simulate(spec, state, sampler, n_runs, config.seed)
    _append_prob_history(data_dir / "processed" / PROB_HISTORY, result)
    return result, spec


def _append_prob_history(path: Path, result: SimulationResult) -> None:
    """SCOPE §2.10: every simulation run appends to the probability history."""
    generated_at = dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as fh:
        for team, probs in sorted(result.reach.items()):
            row = {
                "as_of": str(result.as_of),
                "generated_at": generated_at,
                "team": team,
                "model": f"rung0+{result.calibrator}",
                "runs": result.runs,
                "seed": result.seed,
                **{round_name: probs[round_name] for round_name in ROUNDS},
            }
            fh.write(json.dumps(row) + "\n")
