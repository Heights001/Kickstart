import datetime as dt
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from engine.api.app import create_app
from engine.core.config import load_competition_tiers, load_engine_config
from engine.core.schema import Match
from engine.ratings.store import build_features
from engine.simulation.format import load_format

REPO = Path(__file__).parent.parent
PACK = REPO / "packs/world_cup_2026"


def synthetic_backbone(teams: list[str], seed: int) -> list[Match]:
    """Years of matches between pack teams so rung 0 has train + calibration data."""
    rng = np.random.default_rng(seed)
    matches = []
    date = dt.date(1995, 1, 10)
    for i, (home, away) in enumerate(itertools.cycle(itertools.combinations(teams, 2))):
        if date >= dt.date(2025, 12, 1):
            break
        goals = rng.integers(0, 4, size=2)
        matches.append(
            Match(
                date=date,
                competition="Friendly" if i % 3 else "FIFA World Cup qualification",
                season=str(date.year),
                home=home,
                away=away,
                neutral=bool(i % 2),
                home_goals=int(goals[0]),
                away_goals=int(goals[1]),
            )
        )
        date += dt.timedelta(days=2)
    return matches


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    data_dir = tmp_path_factory.mktemp("data")
    processed = data_dir / "processed"
    processed.mkdir()

    spec = load_format(PACK)
    config = load_engine_config(REPO / "configs/default.yaml")
    tiers = load_competition_tiers(PACK)
    matches = synthetic_backbone(sorted(spec.teams), seed=5)
    with (processed / "matches.jsonl").open("w") as fh:
        for m in matches:
            fh.write(m.model_dump_json() + "\n")
    with (processed / "features.jsonl").open("w") as fh:
        for f in build_features(matches, config, tiers):
            fh.write(f.model_dump_json() + "\n")

    # One fake prob_history generation for /champion-probabilities.
    with (processed / "prob_history.jsonl").open("w") as fh:
        for team, p in (("brazil", 0.3), ("france", 0.2)):
            fh.write(
                json.dumps(
                    {
                        "as_of": "2026-06-10",
                        "generated_at": "2026-06-10T12:00:00+00:00",
                        "team": team,
                        "model": "rung0+isotonic",
                        "runs": 1000,
                        "seed": 42,
                        "R32": 1.0,
                        "R16": 0.8,
                        "QF": 0.6,
                        "SF": 0.45,
                        "F": 0.35,
                        "champion": p,
                    }
                )
                + "\n"
            )

    app = create_app(PACK, data_dir, REPO / "configs/default.yaml")
    with TestClient(app) as test_client:  # context manager triggers lifespan
        yield test_client


class TestTeams:
    def test_all_48(self, client: TestClient) -> None:
        response = client.get("/teams")
        assert response.status_code == 200
        teams = response.json()
        assert len(teams) == 48
        usa = next(t for t in teams if t["team_id"] == "usa")
        assert "USMNT" in usa["aliases"]


class TestRatings:
    def test_pack_teams_rated(self, client: TestClient) -> None:
        response = client.get("/ratings")
        assert response.status_code == 200
        body = response.json()
        assert len(body["teams"]) == 48
        assert body["teams"][0]["elo"] >= body["teams"][-1]["elo"]


class TestPredict:
    def test_probs_sum_to_one(self, client: TestClient) -> None:
        response = client.get("/predict-match", params={"home": "Brazil", "away": "France"})
        assert response.status_code == 200
        body = response.json()
        total = body["p_home"] + body["p_draw"] + body["p_away"]
        assert total == pytest.approx(1.0)
        assert body["home"] == "brazil"
        assert body["neutral"] is True

    def test_alias_resolution(self, client: TestClient) -> None:
        response = client.get("/predict-match", params={"home": "USMNT", "away": "Wales"})
        # Wales is not a WC2026 team but resolution failure comes first for
        # unknown names; USMNT resolves, Wales is unresolved -> 404 + report.
        assert response.status_code == 404
        assert "Wales" in response.json()["detail"]

    def test_same_team_rejected(self, client: TestClient) -> None:
        response = client.get("/predict-match", params={"home": "Brazil", "away": "brazil"})
        assert response.status_code == 422


class TestSimulate:
    def test_small_run(self, client: TestClient) -> None:
        response = client.get("/simulate", params={"runs": 200})
        assert response.status_code == 200
        body = response.json()
        assert body["runs"] == 200
        champion_total = sum(r["champion"] for r in body["table"])
        assert champion_total == pytest.approx(1.0, abs=1e-9)
        assert "model uncertainty" in body["note"]

    def test_runs_capped(self, client: TestClient) -> None:
        assert client.get("/simulate", params={"runs": 10}).status_code == 422


class TestChampionProbabilities:
    def test_latest_generation(self, client: TestClient) -> None:
        response = client.get("/champion-probabilities")
        assert response.status_code == 200
        body = response.json()
        assert body["latest_as_of"] == "2026-06-10"
        assert body["table"][0]["team"] == "brazil"
        assert len(body["history"]) == 2


class TestExplain:
    def test_contributions_named(self, client: TestClient) -> None:
        response = client.get("/explain", params={"home": "Mexico", "away": "Japan"})
        assert response.status_code == 200
        body = response.json()
        assert set(body["features"]) == set(body["contributions"])
        assert "elo_diff" in body["features"]

    def test_unresolved_404_with_suggestions(self, client: TestClient) -> None:
        response = client.get("/explain", params={"home": "Mexcio", "away": "Japan"})
        assert response.status_code == 404
        assert "did you mean" in response.json()["detail"]
