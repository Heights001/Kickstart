"""End-to-end CLI integration: ingest → ratings → train → simulate on one
synthetic-but-realistic pack, driving ``engine.cli.main`` directly."""

import datetime as dt
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from engine.cli import main
from engine.core.registry import TeamRegistry

REPO = Path(__file__).parent.parent
PACK = REPO / "packs/world_cup_2026"


def synthetic_results_csv(path: Path, seed: int) -> None:
    """Decades of results with a persistent strength ordering, so ratings carry
    real signal and Rung 0 can beat the frequency baseline."""
    registry = TeamRegistry.from_yaml(PACK / "teams.yaml")
    teams = sorted({e["id"] for e in yaml.safe_load((PACK / "teams.yaml").read_text())["teams"]})
    names = {t: registry.get(t).name for t in teams}
    strength = {t: i / len(teams) for i, t in enumerate(teams)}
    rng = np.random.default_rng(seed)

    rows = ["date,home_team,away_team,home_score,away_score,tournament,city,country,neutral"]
    date = dt.date(1995, 1, 10)
    for i, (home, away) in enumerate(itertools.cycle(itertools.combinations(teams, 2))):
        if date >= dt.date(2025, 12, 1):
            break
        edge = strength[home] - strength[away]
        home_goals = rng.poisson(0.9 + 3.0 * max(edge, 0.0))
        away_goals = rng.poisson(0.9 + 3.0 * max(-edge, 0.0))
        competition = "Friendly" if i % 3 else "FIFA World Cup qualification"
        neutral = "TRUE" if i % 2 else "FALSE"
        rows.append(
            f"{date},{names[home]},{names[away]},{home_goals},{away_goals},"
            f"{competition},City,Country,{neutral}"
        )
        date += dt.timedelta(days=2)
    path.write_text("\n".join(rows) + "\n")


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("integration")
    pack = root / "pack"
    pack.mkdir()
    for name in ("teams.yaml", "competitions.yaml", "format.yaml", "bracket_allocation.yaml"):
        shutil.copy(PACK / name, pack / name)

    results_csv = root / "source_results.csv"
    synthetic_results_csv(results_csv, seed=9)
    shootouts_csv = root / "source_shootouts.csv"
    shootouts_csv.write_text("date,home_team,away_team,winner,first_shooter\n")
    (pack / "sources.yaml").write_text(
        yaml.safe_dump(
            {
                "sources": [
                    {
                        "name": "synthetic_results",
                        "kind": "csv",
                        "url": results_csv.as_uri(),
                        "filename": "results.csv",
                        "authoritative_names": True,
                        "columns": {
                            "date": "date",
                            "home": "home_team",
                            "away": "away_team",
                            "home_goals": "home_score",
                            "away_goals": "away_score",
                            "competition": "tournament",
                            "neutral": "neutral",
                        },
                    },
                    {
                        "name": "synthetic_shootouts",
                        "kind": "shootouts_csv",
                        "url": shootouts_csv.as_uri(),
                        "filename": "shootouts.csv",
                        "columns": {
                            "date": "date",
                            "home": "home_team",
                            "away": "away_team",
                            "winner": "winner",
                        },
                    },
                ]
            }
        )
    )
    (pack / "backtests.yaml").write_text(
        yaml.safe_dump(
            {
                "backtests": [
                    {
                        "name": "synth2024",
                        "freeze": dt.date(2024, 1, 1),
                        "end": dt.date(2025, 12, 1),
                        "competition": "FIFA World Cup qualification",
                    }
                ]
            }
        )
    )

    config = yaml.safe_load((REPO / "configs/default.yaml").read_text())
    config["mlflow"]["tracking_dir"] = str(root / "mlruns")
    config["models"]["rung2"]["n_trials"] = 2
    config["models"]["rung2"]["n_estimators"] = 40
    config["simulation"]["runs"] = 300
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    return {"pack": pack, "data": root / "data", "config": config_path}


def run(env: dict[str, Path], *args: str) -> int:
    base = ["--pack", str(env["pack"]), "--data-dir", str(env["data"])]
    return main([args[0], *base, *args[1:]])


class TestPipelineEndToEnd:
    def test_1_ingest(self, env: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
        assert run(env, "ingest") == 0
        out = capsys.readouterr().out
        assert "Matches ingested" in out
        assert (env["data"] / "processed/matches.jsonl").exists()

    def test_2_ratings(self, env: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
        assert run(env, "ratings", "--config", str(env["config"])) == 0
        assert "Teams rated" in capsys.readouterr().out
        assert (env["data"] / "processed/features.jsonl").exists()

    def test_3_train_rung0_beats_baseline(
        self, env: dict[str, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert run(env, "train", "--config", str(env["config"]), "--rung", "0") == 0
        out = capsys.readouterr().out
        assert "PASS" in out
        assert (env["data"] / "processed/eval_rung0.json").exists()

    def test_4_simulate_appends_history(
        self, env: dict[str, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert (
            run(
                env,
                "simulate",
                "--config",
                str(env["config"]),
                "--as-of",
                "2025-11-01",
                "--runs",
                "250",
            )
            == 0
        )
        out = capsys.readouterr().out
        assert "champion" in out
        assert "model uncertainty" in out
        history = (env["data"] / "processed/prob_history.jsonl").read_text().splitlines()
        assert len(history) == 48
        row = json.loads(history[0])
        assert row["runs"] == 250 and row["as_of"] == "2025-11-01"

    def test_5_train_rung1_records_decision(
        self, env: dict[str, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert run(env, "train", "--config", str(env["config"]), "--rung", "1") == 0
        out = capsys.readouterr().out
        assert "Decision" in out
        registry = json.loads((env["data"] / "processed/model_registry.json").read_text())
        assert registry["last_decision"]["candidate_rung"] == 1
        # MLflow logged to the sandboxed sqlite backend.
        assert (env["config"].parent / "mlruns" / "mlflow.db").exists()
