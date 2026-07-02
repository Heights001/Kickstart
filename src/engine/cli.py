"""Thin CLI wrapper: ``engine <subcommand>`` (SCOPE.md §3)."""

import argparse
import sys
from pathlib import Path

from engine.ingestion.pipeline import IngestReport, ingest_pack
from engine.ratings.pipeline import rebuild_ratings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="engine", description="Football prediction engine")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Pull/refresh all sources for a pack")
    ratings = sub.add_parser("ratings", help="Rebuild chronological ratings for a pack")
    for sub_parser in (ingest, ratings):
        sub_parser.add_argument("--pack", type=Path, required=True, help="Path to a pack folder")
        sub_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    ratings.add_argument("--config", type=Path, default=Path("configs/default.yaml"))

    args = parser.parse_args(argv)
    if args.command == "ingest":
        report = ingest_pack(args.pack, args.data_dir)
        _print_ingest_report(report)
        return 0
    if args.command == "ratings":
        ratings_report = rebuild_ratings(args.pack, args.data_dir, args.config)
        print(f"Feature rows : {ratings_report.feature_count}")
        print(f"Teams rated  : {ratings_report.team_count}")
        print(f"Features     : {ratings_report.features_path}")
        print(f"Ratings      : {ratings_report.ratings_path}")
        return 0
    return 1  # pragma: no cover - argparse enforces the subcommand


def _print_ingest_report(report: IngestReport) -> None:
    print(f"Matches ingested : {report.match_count}")
    print(f"Date range       : {report.date_min} .. {report.date_max}")
    print(f"Rows skipped     : {report.skipped_rows} (missing scores)")
    print(f"Unresolved names : {len(report.registered_names)} registered from authoritative source")
    if report.registered_names:
        print("  (not in teams.yaml — add aliases there if any should map to an existing team)")
        for name in report.registered_names:
            print(f"  - {name}")


if __name__ == "__main__":
    sys.exit(main())
