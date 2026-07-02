"""Typed loader for a pack's ``format.yaml`` + third-place allocation table.

The engine interprets these declaratively — no competition specifics in code
(CLAUDE.md rule 1). Slot grammar: ``1A``/``2A`` group winner/runner-up, ``T1E``
third-placed team allocated to slot 1E, ``W74``/``L101`` winner/loser of a match.
"""

import datetime as dt
import itertools
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class PointsRule(BaseModel):
    model_config = ConfigDict(frozen=True)

    win: int
    draw: int
    loss: int


class KnockoutMatch(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    round: str
    home: str
    away: str


class ThirdPlaceRule(BaseModel):
    model_config = ConfigDict(frozen=True)

    advance: int
    tiebreakers: list[str]
    allocation: str  # filename of the allocation YAML within the pack


class FormatSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    competition: str
    season: str
    start: dt.date
    end: dt.date
    hosts: list[str]
    points: PointsRule
    groups: dict[str, list[str]]
    group_tiebreakers: list[str]
    third_place: ThirdPlaceRule
    knockout: list[KnockoutMatch]
    allocation_slots: list[str] = Field(default_factory=list)
    # combination (sorted group letters) -> slot -> third's group
    allocations: dict[str, dict[str, str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> "FormatSpec":
        seen: set[str] = set()
        for group, members in self.groups.items():
            for team in members:
                if team in seen:
                    raise ValueError(f"team {team!r} appears in more than one group")
                seen.add(team)
            if len(members) < 2:
                raise ValueError(f"group {group!r} has fewer than two teams")
        ids = [m.id for m in self.knockout]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate knockout match ids")
        return self

    @property
    def teams(self) -> set[str]:
        return {team for members in self.groups.values() for team in members}

    def expected_combinations(self) -> set[str]:
        count = self.third_place.advance
        letters = sorted(self.groups)
        return {"".join(c) for c in itertools.combinations(letters, count)}


def load_format(pack_dir: Path) -> FormatSpec:
    raw = yaml.safe_load((pack_dir / "format.yaml").read_text())
    knockout = [
        KnockoutMatch(id=m["id"], round=stage["round"], home=m["home"], away=m["away"])
        for stage in raw.pop("knockout")
        for m in stage["matches"]
    ]
    spec = FormatSpec(knockout=knockout, **raw)

    allocation_raw = yaml.safe_load((pack_dir / spec.third_place.allocation).read_text())
    spec = spec.model_copy(
        update={
            "allocation_slots": allocation_raw["slots"],
            "allocations": allocation_raw["allocations"],
        }
    )
    missing = spec.expected_combinations() - set(spec.allocations)
    if missing:
        raise ValueError(
            f"allocation table incomplete: {len(missing)} combinations missing "
            f"(e.g. {sorted(missing)[:3]})"
        )
    return spec
