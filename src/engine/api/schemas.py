"""API response models (Pydantic v2, SCOPE §4 P4)."""

import datetime as dt

from pydantic import BaseModel


class TeamOut(BaseModel):
    team_id: str
    name: str
    aliases: list[str]


class RatingOut(BaseModel):
    team_id: str
    elo: float
    form: float
    attack: float
    defence: float


class RatingsOut(BaseModel):
    as_of: dt.date
    teams: list[RatingOut]


class PredictOut(BaseModel):
    home: str
    away: str
    neutral: bool
    p_home: float
    p_draw: float
    p_away: float
    model: str
    as_of: dt.date


class ReachOut(BaseModel):
    team: str
    champion: float
    champion_se: float
    final: float
    sf: float
    qf: float
    r16: float
    r32: float


class SimulateOut(BaseModel):
    as_of: dt.date
    runs: int
    seed: int
    model: str
    note: str
    table: list[ReachOut]


class ChampionHistoryEntry(BaseModel):
    as_of: dt.date
    generated_at: str
    team: str
    model: str
    champion: float


class ChampionOut(BaseModel):
    latest_as_of: dt.date
    model: str
    table: list[ReachOut]
    history: list[ChampionHistoryEntry]


class ExplainOut(BaseModel):
    home: str
    away: str
    model: str
    note: str
    features: dict[str, float]
    contributions: dict[str, float]
