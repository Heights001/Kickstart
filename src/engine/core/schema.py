"""Canonical data models. Everything downstream of ingestion consumes only these.

SCOPE.md §2.2: source-specific quirks die inside adapters; a sparse ``extras`` dict
carries whatever optional numerics a source happens to provide (xG, shots, ...).
"""

import datetime as dt

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Team(BaseModel):
    """A canonical team. ``team_id`` is the stable slug all match records reference."""

    model_config = ConfigDict(frozen=True)

    team_id: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    name: str = Field(min_length=1)
    aliases: frozenset[str] = frozenset()


class Match(BaseModel):
    """One completed match, teams already resolved to canonical ids."""

    model_config = ConfigDict(frozen=True)

    date: dt.date
    competition: str = Field(min_length=1)
    season: str = Field(min_length=1)
    stage: str | None = None
    home: str = Field(min_length=1, description="Canonical team_id")
    away: str = Field(min_length=1, description="Canonical team_id")
    neutral: bool
    home_goals: int = Field(ge=0)
    away_goals: int = Field(ge=0)
    # Set only for drawn knockout matches decided by a shootout.
    winner: str | None = None
    extras: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self) -> "Match":
        if self.home == self.away:
            raise ValueError(f"home and away are the same team: {self.home!r}")
        if self.winner is not None:
            if self.winner not in (self.home, self.away):
                raise ValueError(f"winner {self.winner!r} is not a participant")
            if self.home_goals != self.away_goals:
                raise ValueError("winner is only for drawn matches decided by shootout")
        return self
