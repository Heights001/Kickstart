import datetime as dt
from pathlib import Path

import pytest

from engine.ingestion.base import RawMatch, SourceConfig
from engine.ingestion.csv_adapter import CsvSourceAdapter, MalformedRowError

FIXTURES = Path(__file__).parent / "fixtures"

MARTJ42_COLUMNS = {
    "date": "date",
    "home": "home_team",
    "away": "away_team",
    "home_goals": "home_score",
    "away_goals": "away_score",
    "competition": "tournament",
    "neutral": "neutral",
}


def make_adapter(**overrides: object) -> CsvSourceAdapter:
    base: dict[str, object] = {
        "name": "mini_results",
        "kind": "csv",
        "url": "file:///unused",
        "filename": "results.csv",
        "authoritative_names": True,
        "columns": MARTJ42_COLUMNS,
    }
    base.update(overrides)
    return CsvSourceAdapter(SourceConfig.model_validate(base))


class TestParse:
    def test_exact_records(self) -> None:
        adapter = make_adapter()
        records = list(adapter.parse(FIXTURES / "mini_results.csv"))
        assert records[0] == RawMatch(
            date=dt.date(1872, 11, 30),
            competition="Friendly",
            home_name="Scotland",
            away_name="England",
            neutral=False,
            home_goals=0,
            away_goals=0,
        )
        assert records[1] == RawMatch(
            date=dt.date(2022, 11, 21),
            competition="FIFA World Cup",
            home_name="USA",
            away_name="Wales",
            neutral=True,
            home_goals=1,
            away_goals=1,
        )
        assert len(records) == 4

    def test_missing_scores_skipped_and_counted(self) -> None:
        adapter = make_adapter()
        list(adapter.parse(FIXTURES / "mini_results.csv"))
        assert adapter.skipped_rows == 1

    def test_malformed_row_raises_with_location(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.csv"
        bad.write_text(
            "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
            "2026-06-11,Mexico,South Africa,2,0,FIFA World Cup,Mexico City,Mexico,maybe\n"
        )
        adapter = make_adapter()
        with pytest.raises(MalformedRowError, match=r"bad\.csv:2.*maybe"):
            list(adapter.parse(bad))


class TestFetch:
    def test_downloads_via_url_and_never_overwrites(self, tmp_path: Path) -> None:
        source = FIXTURES / "mini_results.csv"
        adapter = make_adapter(url=source.as_uri())
        raw_dir = tmp_path / "raw"

        fetched = adapter.fetch(raw_dir)
        assert fetched == raw_dir / "results.csv"
        assert fetched.read_text() == source.read_text()

        # data/raw is immutable: a second fetch must not touch the existing file.
        fetched.write_text("locally frozen")
        assert adapter.fetch(raw_dir).read_text() == "locally frozen"
