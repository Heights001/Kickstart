"""CSV adapter for flat results files (v1: martj42/international_results).

Column names come from the pack's ``sources.yaml`` mapping — nothing
competition-specific is hardcoded here (CLAUDE.md rule 1).
"""

import csv
import datetime as dt
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from engine.ingestion.base import RawMatch, SourceConfig
from engine.utils.logging import get_logger, log_with

logger = get_logger(__name__)

_TRUTHY = {"true", "1", "yes"}
_FALSY = {"false", "0", "no"}
_MISSING = {"", "na", "nan", "null"}


class MalformedRowError(ValueError):
    def __init__(self, path: Path, line: int, problem: str) -> None:
        super().__init__(f"{path.name}:{line}: {problem}")


class CsvSourceAdapter:
    """Downloads a results CSV and parses it into :class:`RawMatch` records.

    Rows with missing scores (fixtures not yet played, abandoned matches) are skipped
    and counted in :attr:`skipped_rows`; anything else malformed raises.
    """

    def __init__(self, config: SourceConfig) -> None:
        self.config = config
        self.skipped_rows = 0

    def fetch(self, raw_dir: Path) -> Path:
        target = raw_dir / self.config.filename
        if target.exists():
            log_with(logger, "raw file present, not re-downloading", path=str(target))
            return target
        raw_dir.mkdir(parents=True, exist_ok=True)
        log_with(logger, "downloading", url=self.config.url, path=str(target))
        tmp = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(self.config.url, tmp)
        tmp.rename(target)
        return target

    def parse(self, path: Path) -> Iterator[RawMatch]:
        cols = self.config.columns
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for line, row in enumerate(reader, start=2):
                home_goals = row[cols["home_goals"]].strip()
                away_goals = row[cols["away_goals"]].strip()
                if home_goals.casefold() in _MISSING or away_goals.casefold() in _MISSING:
                    self.skipped_rows += 1
                    continue
                try:
                    yield RawMatch(
                        date=dt.date.fromisoformat(row[cols["date"]].strip()),
                        competition=row[cols["competition"]].strip(),
                        home_name=row[cols["home"]].strip(),
                        away_name=row[cols["away"]].strip(),
                        neutral=_parse_bool(row[cols["neutral"]], path, line),
                        home_goals=int(home_goals),
                        away_goals=int(away_goals),
                    )
                except (ValueError, KeyError) as exc:
                    raise MalformedRowError(path, line, str(exc)) from exc


def _parse_bool(value: str, path: Path, line: int) -> bool:
    lowered = value.strip().casefold()
    if lowered in _TRUTHY:
        return True
    if lowered in _FALSY:
        return False
    raise MalformedRowError(path, line, f"unparseable boolean {value!r}")
