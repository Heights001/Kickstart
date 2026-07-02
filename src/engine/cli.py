"""Thin CLI wrapper: ``engine <subcommand>`` (SCOPE.md §3)."""

import argparse
import datetime as dt
import json
import sys
from dataclasses import asdict
from pathlib import Path

from engine.core.config import load_backtests, load_engine_config
from engine.evaluation.walk_forward import evaluate_window
from engine.ingestion.pipeline import IngestReport, ingest_pack
from engine.ratings.pipeline import load_features, rebuild_ratings
from engine.simulation.pipeline import run_simulation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="engine", description="Football prediction engine")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Pull/refresh all sources for a pack")
    ratings = sub.add_parser("ratings", help="Rebuild chronological ratings for a pack")
    train = sub.add_parser("train", help="Train + calibrate a rung, walk-forward evaluated")
    simulate = sub.add_parser("simulate", help="Monte Carlo simulation of the competition")
    for sub_parser in (ingest, ratings, train, simulate):
        sub_parser.add_argument("--pack", type=Path, required=True, help="Path to a pack folder")
        sub_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    for sub_parser in (ratings, train, simulate):
        sub_parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    train.add_argument("--rung", type=int, choices=[0], default=0)
    group = simulate.add_mutually_exclusive_group()
    group.add_argument("--as-of", default="now", help='"now" or YYYY-MM-DD')
    group.add_argument("--freeze", default=None, help="Alias for --as-of (rewind backtests)")
    simulate.add_argument("--runs", type=int, default=None, help="Override configured run count")

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
    if args.command == "train":
        return _run_train(args.pack, args.data_dir, args.config)
    if args.command == "simulate":
        raw = args.freeze if args.freeze is not None else args.as_of
        as_of = dt.date.today() if raw == "now" else dt.date.fromisoformat(raw)
        return _run_simulate(args.pack, args.data_dir, args.config, as_of, args.runs)
    return 1  # pragma: no cover - argparse enforces the subcommand


def _run_simulate(
    pack_dir: Path, data_dir: Path, config_path: Path, as_of: dt.date, runs: int | None
) -> int:
    result, _spec = run_simulation(pack_dir, data_dir, config_path, as_of, runs)
    print(
        f"\nas-of {result.as_of} | {result.runs} runs | seed {result.seed} "
        f"| rung0+{result.calibrator}"
    )
    header = (
        f"{'team':24} {'champion':>9} {'±SE':>7} {'final':>7} {'SF':>7} "
        f"{'QF':>7} {'R16':>7} {'R32':>7}"
    )
    print(header)
    ranked = sorted(result.reach.items(), key=lambda kv: kv[1]["champion"], reverse=True)
    for team, p in ranked:
        se = result.standard_error(p["champion"])
        print(
            f"{team:24} {p['champion']:>9.4f} {se:>7.4f} {p['F']:>7.4f} {p['SF']:>7.4f} "
            f"{p['QF']:>7.4f} {p['R16']:>7.4f} {p['R32']:>7.4f}"
        )
    print("\nMC standard error only — model uncertainty is not included.")
    print(f"Probability history appended: {data_dir / 'processed' / 'prob_history.jsonl'}")
    return 0


def _run_train(pack_dir: Path, data_dir: Path, config_path: Path) -> int:
    config = load_engine_config(config_path)
    features = load_features(data_dir / "processed" / "features.jsonl")
    results = [evaluate_window(features, window, config) for window in load_backtests(pack_dir)]

    header = (
        f"{'window':8} {'n_test':>6} {'calib':>9} "
        f"{'logloss':>9} {'base_ll':>9} {'brier':>7} {'ece':>7} {'base_ece':>8}  gate"
    )
    print(header)
    for r in results:
        gate = "PASS" if r.beats_baseline else "FAIL"
        print(
            f"{r.window:8} {r.n_test:>6} {r.calibrator:>9} "
            f"{r.model.log_loss:>9.4f} {r.baseline.log_loss:>9.4f} "
            f"{r.model.brier:>7.4f} {r.model.ece:>7.4f} {r.baseline.ece:>8.4f}  {gate}"
        )

    artifact = data_dir / "processed" / "eval_rung0.json"
    artifact.write_text(json.dumps([asdict(r) for r in results], indent=2, default=str))
    print(f"\nEvaluation artifact: {artifact}")

    if all(r.beats_baseline for r in results):
        print("P1 gate: Rung 0 beats the frequency baseline on every window.")
        return 0
    print("P1 gate FAILED: Rung 0 does not beat the baseline on every window.")
    return 1


def _print_ingest_report(report: IngestReport) -> None:
    print(f"Matches ingested : {report.match_count}")
    print(f"Date range       : {report.date_min} .. {report.date_max}")
    print(f"Rows skipped     : {report.skipped_rows} (missing scores)")
    print(f"Shootout winners : {report.shootouts_annotated} drawn matches annotated")
    print(f"Unresolved names : {len(report.registered_names)} registered from authoritative source")
    if report.registered_names:
        print("  (not in teams.yaml — add aliases there if any should map to an existing team)")
        for name in report.registered_names:
            print(f"  - {name}")


if __name__ == "__main__":
    sys.exit(main())
