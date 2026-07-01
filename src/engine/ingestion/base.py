"""Source adapter protocol and pre-resolution record type.

Adapters own everything source-specific (URLs, column names, encodings). They yield
:class:`RawMatch` records with team names still unresolved; the ingest pipeline
resolves names through the registry and emits canonical ``Match`` records.
"""

import datetime as dt
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class RawMatch(BaseModel):
    """A parsed source row, before team-name resolution."""

    model_config = ConfigDict(frozen=True)

    date: dt.date
    competition: str
    home_name: str
    away_name: str
    neutral: bool
    home_goals: int = Field(ge=0)
    away_goals: int = Field(ge=0)
    extras: dict[str, float] = Field(default_factory=dict)


class SourceConfig(BaseModel):
    """One entry in a pack's ``sources.yaml``."""

    model_config = ConfigDict(frozen=True)

    name: str
    kind: str
    url: str
    filename: str
    authoritative_names: bool = False
    columns: dict[str, str]


class SourceAdapter(Protocol):
    """Fetches a source into ``data/raw/`` and parses it into raw match records."""

    config: SourceConfig

    def fetch(self, raw_dir: Path) -> Path:
        """Download into ``raw_dir`` and return the local path.

        ``data/raw/`` is immutable: an existing file is never overwritten.
        """
        ...

    def parse(self, path: Path) -> Iterator[RawMatch]: ...
