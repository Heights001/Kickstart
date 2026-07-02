"""FastAPI service (SCOPE §4 P4).

The app context (data, registry, promoted-rung sampler) is built once in the
lifespan — Rung 2 retraining is minutes, not per-request work. All names resolve
through the registry; unresolved names return 404 with the suggested-alias report.

Deviation from SCOPE's ``/explain/{match_id}``: canonical matches carry no ids, so
``/explain`` takes a pairing and explains the hypothetical match instead.
"""

import datetime as dt
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request

from engine.api.schemas import (
    ChampionHistoryEntry,
    ChampionOut,
    ExplainOut,
    PredictOut,
    RatingOut,
    RatingsOut,
    ReachOut,
    SimulateOut,
    TeamOut,
)
from engine.core.config import EngineConfig, load_competition_tiers, load_engine_config
from engine.core.registry import TeamRegistry, UnresolvedTeamError
from engine.evaluation.metrics import FloatArray
from engine.evaluation.promotion import promoted_rung
from engine.ingestion.pipeline import load_matches
from engine.models.ladder import get_rung
from engine.ratings.pipeline import load_features
from engine.simulation.format import FormatSpec, load_format
from engine.simulation.monte_carlo import SimulationResult, simulate
from engine.simulation.pipeline import PROB_HISTORY
from engine.simulation.sampler import MatchSampler
from engine.simulation.state import build_state

MC_NOTE = "MC standard error only — model uncertainty is not included."


@dataclass
class AppContext:
    config: EngineConfig
    spec: FormatSpec
    registry: TeamRegistry
    sampler: MatchSampler
    matches: list  # type: ignore[type-arg]
    data_dir: Path
    model_name: str


def build_context(pack_dir: Path, data_dir: Path, config_path: Path) -> AppContext:
    config = load_engine_config(config_path)
    spec = load_format(pack_dir)
    tiers = load_competition_tiers(pack_dir)
    registry = TeamRegistry.from_yaml(pack_dir / "teams.yaml")
    matches = list(load_matches(data_dir / "processed" / "matches.jsonl"))
    features = load_features(data_dir / "processed" / "features.jsonl")
    as_of = max(m.date for m in matches) + dt.timedelta(days=1)
    rung = get_rung(promoted_rung(data_dir), config)
    sampler = MatchSampler.build(matches, features, as_of, spec, config, tiers, rung)
    return AppContext(
        config=config,
        spec=spec,
        registry=registry,
        sampler=sampler,
        matches=matches,
        data_dir=data_dir,
        model_name=f"{sampler.rung.name}+{sampler.calibrator.name}",
    )


def create_app(pack_dir: Path, data_dir: Path, config_path: Path) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.ctx = build_context(pack_dir, data_dir, config_path)
        yield

    app = FastAPI(title="predictor-engine", lifespan=lifespan)
    _register_routes(app)
    return app


def _ctx(request: Request) -> AppContext:
    ctx: AppContext = request.app.state.ctx
    return ctx


def _resolve(ctx: AppContext, name: str) -> str:
    try:
        return ctx.registry.resolve(name)
    except UnresolvedTeamError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _reach_table(result: SimulationResult) -> list[ReachOut]:
    rows = []
    for team, p in sorted(result.reach.items(), key=lambda kv: -kv[1]["champion"]):
        rows.append(
            ReachOut(
                team=team,
                champion=p["champion"],
                champion_se=result.standard_error(p["champion"]),
                final=p["F"],
                sf=p["SF"],
                qf=p["QF"],
                r16=p["R16"],
                r32=p["R32"],
            )
        )
    return rows


def _register_routes(app: FastAPI) -> None:
    @app.get("/teams", response_model=list[TeamOut])
    def teams(request: Request) -> list[TeamOut]:
        ctx = _ctx(request)
        out = []
        for team_id in sorted(ctx.spec.teams):
            team = ctx.registry.get(team_id)
            out.append(TeamOut(team_id=team.team_id, name=team.name, aliases=sorted(team.aliases)))
        return out

    @app.get("/ratings", response_model=RatingsOut)
    def ratings(request: Request) -> RatingsOut:
        ctx = _ctx(request)
        states = ctx.sampler.team_states()
        rows = [
            RatingOut(team_id=t, elo=s.elo, form=s.form, attack=s.attack, defence=s.defence)
            for t, s in sorted(states.items(), key=lambda kv: -kv[1].elo)
        ]
        return RatingsOut(as_of=ctx.sampler.as_of, teams=rows)

    @app.get("/predict-match", response_model=PredictOut)
    def predict_match(request: Request, home: str, away: str) -> PredictOut:
        ctx = _ctx(request)
        home_id, away_id = _resolve(ctx, home), _resolve(ctx, away)
        if home_id == away_id:
            raise HTTPException(status_code=422, detail="home and away are the same team")
        probs = ctx.sampler.outcome_probs(home_id, away_id)
        return PredictOut(
            home=home_id,
            away=away_id,
            neutral=ctx.sampler.is_neutral(home_id, away_id),
            p_home=float(probs[0]),
            p_draw=float(probs[1]),
            p_away=float(probs[2]),
            model=ctx.model_name,
            as_of=ctx.sampler.as_of,
        )

    @app.get("/simulate", response_model=SimulateOut)
    def run_simulate(
        request: Request, runs: int = Query(default=10_000, ge=100, le=1_000_000)
    ) -> SimulateOut:
        ctx = _ctx(request)
        state = build_state(ctx.matches, ctx.spec, ctx.sampler.as_of)
        result = simulate(ctx.spec, state, ctx.sampler, runs, ctx.config.seed)
        return SimulateOut(
            as_of=result.as_of,
            runs=result.runs,
            seed=result.seed,
            model=ctx.model_name,
            note=MC_NOTE,
            table=_reach_table(result),
        )

    @app.get("/champion-probabilities", response_model=ChampionOut)
    def champion_probabilities(request: Request) -> ChampionOut:
        ctx = _ctx(request)
        path = ctx.data_dir / "processed" / PROB_HISTORY
        if not path.exists():
            raise HTTPException(status_code=404, detail="no simulation history yet")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        latest_gen = max(r["generated_at"] for r in rows)
        latest = [r for r in rows if r["generated_at"] == latest_gen]
        table = [
            ReachOut(
                team=r["team"],
                champion=r["champion"],
                champion_se=float(np.sqrt(r["champion"] * (1 - r["champion"]) / r["runs"])),
                final=r["F"],
                sf=r["SF"],
                qf=r["QF"],
                r16=r["R16"],
                r32=r["R32"],
            )
            for r in sorted(latest, key=lambda r: -r["champion"])
        ]
        history = [
            ChampionHistoryEntry(
                as_of=r["as_of"],
                generated_at=r["generated_at"],
                team=r["team"],
                model=r["model"],
                champion=r["champion"],
            )
            for r in rows
        ]
        return ChampionOut(
            latest_as_of=dt.date.fromisoformat(latest[0]["as_of"]),
            model=latest[0]["model"],
            table=table,
            history=history,
        )

    @app.get("/explain", response_model=ExplainOut)
    def explain(request: Request, home: str, away: str) -> ExplainOut:
        ctx = _ctx(request)
        home_id, away_id = _resolve(ctx, home), _resolve(ctx, away)
        if home_id == away_id:
            raise HTTPException(status_code=422, detail="home and away are the same team")
        x = ctx.sampler.feature_row(home_id, away_id)
        names = list(ctx.sampler.rung.feature_names)
        contributions = _contributions(ctx, x)
        return ExplainOut(
            home=home_id,
            away=away_id,
            model=ctx.model_name,
            note="Contributions toward the home-win class, log-odds scale (uncalibrated).",
            features=dict(zip(names, (float(v) for v in x[0]), strict=True)),
            contributions=dict(zip(names, contributions, strict=True)),
        )


def _contributions(ctx: AppContext, x: FloatArray) -> list[float]:
    from engine.models.rung2 import Rung2Model

    model = ctx.sampler.model
    if isinstance(model, Rung2Model):
        import shap

        explainer = shap.TreeExplainer(model.booster)
        values = explainer.shap_values(x)
        stacked = np.stack(values, axis=-1) if isinstance(values, list) else np.asarray(values)
        arr = stacked if stacked.ndim == 3 else stacked[None, ...]
        # (n, features, classes) vs (classes, n, features) — take class 0, row 0.
        row = arr[0, :, 0] if arr.shape[-1] == 3 else arr[0, 0, :]
        return [float(v) for v in row]
    contributions = getattr(model, "home_win_contributions", None)
    if contributions is None:
        raise HTTPException(status_code=501, detail="explanation unavailable for this model")
    result: list[float] = contributions(x)
    return result
