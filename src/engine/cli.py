"""Thin CLI wrapper: ``engine <subcommand>`` (SCOPE.md §3)."""

import argparse
import sys
from pathlib import Path

from engine.ingestion.pipeline import IngestReport, ingest_pack


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="engine", description="Football prediction engine")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Pull/refresh all sources for a pack")
    ingest.add_argument("--pack", type=Path, required=True, help="Path to a pack folder")
    ingest.add_argument("--data-dir", type=Path, default=Path("data"))

    args = parser.parse_args(argv)
    if args.command == "ingest":
        report = ingest_pack(args.pack, args.data_dir)
        _print_ingest_report(report)
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
