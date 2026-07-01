import datetime as dt
from pathlib import Path

import pytest

from engine.core.registry import UnresolvedTeamError
from engine.core.schema import Match
from engine.ingestion.pipeline import PROCESSED_MATCHES, ingest_pack

FIXTURES = Path(__file__).parent / "fixtures"


def make_pack(tmp_path: Path, *, authoritative: bool = True) -> Path:
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "teams.yaml").write_text((FIXTURES / "mini_teams.yaml").read_text())
    sources = (FIXTURES / "mini_sources.yaml").read_text()
    sources = sources.replace("PLACEHOLDER", (FIXTURES / "mini_results.csv").as_uri())
    if not authoritative:
        sources = sources.replace("authoritative_names: true", "authoritative_names: false")
    (pack / "sources.yaml").write_text(sources)
    return pack


class TestIngestPack:
    def test_end_to_end(self, tmp_path: Path) -> None:
        report = ingest_pack(make_pack(tmp_path), tmp_path / "data")

        assert report.match_count == 4
        assert report.date_min == dt.date(1872, 11, 30)
        assert report.date_max == dt.date(2026, 6, 11)
        assert report.skipped_rows == 1
        # "South Africa" is absent from mini_teams.yaml and comes from the
        # authoritative backbone, so it is registered and reported.
        assert report.registered_names == ["South Africa"]

        lines = (tmp_path / "data/processed" / PROCESSED_MATCHES).read_text().splitlines()
        matches = [Match.model_validate_json(line) for line in lines]
        assert len(matches) == 4
        assert matches[1].home == "usa"  # alias "USA" resolved
        assert matches[2].away == "usa"  # canonical "United States" resolved
        assert matches[3].away == "south_africa"  # registered from authoritative source
        assert matches[3].season == "2026"
        assert matches[1].neutral is True

    def test_non_authoritative_source_raises(self, tmp_path: Path) -> None:
        pack = make_pack(tmp_path, authoritative=False)
        with pytest.raises(UnresolvedTeamError, match="South Africa"):
            ingest_pack(pack, tmp_path / "data")

    def test_deterministic_output(self, tmp_path: Path) -> None:
        pack = make_pack(tmp_path)
        ingest_pack(pack, tmp_path / "data1")
        ingest_pack(pack, tmp_path / "data2")
        out1 = (tmp_path / "data1/processed" / PROCESSED_MATCHES).read_bytes()
        out2 = (tmp_path / "data2/processed" / PROCESSED_MATCHES).read_bytes()
        assert out1 == out2
