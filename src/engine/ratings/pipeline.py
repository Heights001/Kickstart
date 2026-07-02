"""Rebuild chronological ratings: processed matches → per-match features + final state."""

import json
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from engine.core.config import load_competition_tiers, load_engine_config
from engine.ingestion.pipeline import load_matches
from engine.ratings.store import RatingsWalker, build_features
from engine.utils.logging import get_logger, log_with

logger = get_logger(__name__)

PROCESSED_FEATURES = "features.jsonl"
PROCESSED_RATINGS = "ratings.json"


@dataclass
class RatingsReport:
    feature_count: int
    team_count: int
    features_path: Path
    ratings_path: Path


def rebuild_ratings(pack_dir: Path, data_dir: Path, config_path: Path) -> RatingsReport:
    config = load_engine_config(config_path)
    tiers = load_competition_tiers(pack_dir)
    matches = list(load_matches(data_dir / "processed" / "matches.jsonl"))

    processed_dir = data_dir / "processed"
    features_path = processed_dir / PROCESSED_FEATURES
    count = 0
    # build_features sorts and walks; re-walk manually to keep the final state.
    walker = RatingsWalker(config, tiers)
    with features_path.open("w", encoding="utf-8") as fh:
        for match in tqdm(sorted(matches, key=lambda m: m.date), desc="ratings", unit=" matches"):
            fh.write(walker.observe(match).model_dump_json() + "\n")
            count += 1

    ratings_path = processed_dir / PROCESSED_RATINGS
    elo = walker.elo.snapshot()
    form = walker.form.snapshot()
    final = {
        team: {
            "elo": elo[team],
            "form": form.get(team, 0.5),
            "attack": walker.attack_defence.attack(team),
            "defence": walker.attack_defence.defence(team),
        }
        for team in sorted(elo)
    }
    ratings_path.write_text(json.dumps(final, indent=2, sort_keys=True))

    log_with(logger, "ratings rebuilt", features=count, teams=len(final))
    return RatingsReport(count, len(final), features_path, ratings_path)


__all__ = ["RatingsReport", "build_features", "rebuild_ratings"]
