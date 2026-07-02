"""Ingest pipeline: fetch sources → parse → resolve names → write canonical matches.

Name resolution policy (approved 2026-07-01): a source marked ``authoritative_names``
in ``sources.yaml`` may introduce new canonical teams — each new name is
collision-checked against existing aliases and listed in the ingest report, never
added silently. All other sources raise on any unresolved name (CLAUDE.md rule 4).
"""

import datetime as dt
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from tqdm import tqdm

from engine.core.registry import TeamRegistry, UnresolvedTeamError, slugify
from engine.core.schema import Match, Team
from engine.ingestion.base import RawMatch, SourceAdapter, SourceConfig
from engine.ingestion.csv_adapter import CsvSourceAdapter
from engine.utils.logging import get_logger, log_with

logger = get_logger(__name__)

PROCESSED_MATCHES = "matches.jsonl"


def load_matches(path: Path) -> Iterator[Match]:
    """Stream canonical matches back from a processed JSONL file."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            yield Match.model_validate_json(line)


@dataclass
class IngestReport:
    match_count: int = 0
    date_min: dt.date | None = None
    date_max: dt.date | None = None
    skipped_rows: int = 0
    registered_names: list[str] = field(default_factory=list)


def build_adapter(config: SourceConfig) -> SourceAdapter:
    if config.kind == "csv":
        return CsvSourceAdapter(config)
    raise ValueError(f"unknown source kind {config.kind!r} (source {config.name!r})")


def load_source_configs(pack_dir: Path) -> list[SourceConfig]:
    raw = yaml.safe_load((pack_dir / "sources.yaml").read_text())
    return [SourceConfig.model_validate(entry) for entry in raw["sources"]]


def ingest_pack(pack_dir: Path, data_dir: Path) -> IngestReport:
    """Run the full raw→processed pipeline for one pack."""
    registry = TeamRegistry.from_yaml(pack_dir / "teams.yaml")
    report = IngestReport()
    matches: list[Match] = []

    for config in load_source_configs(pack_dir):
        adapter = build_adapter(config)
        raw_path = adapter.fetch(data_dir / "raw")
        for raw in tqdm(adapter.parse(raw_path), desc=config.name, unit=" rows"):
            matches.append(_to_match(raw, registry, config, report))
        report.skipped_rows += getattr(adapter, "skipped_rows", 0)

    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / PROCESSED_MATCHES
    with out_path.open("w", encoding="utf-8") as fh:
        for match in matches:
            fh.write(match.model_dump_json() + "\n")

    report.match_count = len(matches)
    if matches:
        report.date_min = min(m.date for m in matches)
        report.date_max = max(m.date for m in matches)
    log_with(
        logger,
        "ingest complete",
        matches=report.match_count,
        registered=len(report.registered_names),
        skipped=report.skipped_rows,
        output=str(out_path),
    )
    return report


def _to_match(
    raw: RawMatch, registry: TeamRegistry, config: SourceConfig, report: IngestReport
) -> Match:
    return Match(
        date=raw.date,
        competition=raw.competition,
        season=str(raw.date.year),
        stage=None,
        home=_resolve(raw.home_name, registry, config, report),
        away=_resolve(raw.away_name, registry, config, report),
        neutral=raw.neutral,
        home_goals=raw.home_goals,
        away_goals=raw.away_goals,
        extras=raw.extras,
    )


def _resolve(name: str, registry: TeamRegistry, config: SourceConfig, report: IngestReport) -> str:
    try:
        return registry.resolve(name)
    except UnresolvedTeamError:
        if not config.authoritative_names:
            raise
        # Explicit, reported registration from the declared-authoritative source.
        # Collisions with existing aliases still raise inside register().
        team = Team(team_id=slugify(name), name=name)
        registry.register(team)
        report.registered_names.append(name)
        return team.team_id
