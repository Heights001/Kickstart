"""Team registry with alias resolution.

Every ingested team name resolves through this registry (CLAUDE.md rule 4). Resolution
is exact (after normalisation) — never fuzzy. Unresolved names raise
:class:`UnresolvedTeamError` carrying a suggested-alias report; teams are only ever
added explicitly via :meth:`TeamRegistry.register`, which surfaces collisions loudly.
"""

import difflib
import re
from collections.abc import Iterable
from pathlib import Path

import yaml

from engine.core.schema import Team

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    """Derive a canonical team_id from a display name."""
    slug = _SLUG_RE.sub("_", name.strip().casefold()).strip("_")
    if not slug:
        raise ValueError(f"cannot derive a team_id from {name!r}")
    return slug


def _normalise(name: str) -> str:
    return " ".join(name.strip().casefold().split())


class UnresolvedTeamError(KeyError):
    """A team name not known to the registry, with a suggested-alias report."""

    def __init__(self, name: str, suggestions: list[tuple[str, str]]) -> None:
        self.name = name
        self.suggestions = suggestions
        report = (
            "; ".join(
                f"did you mean {alias!r} (team_id={team_id!r})? "
                f"If so, add {name!r} to that team's aliases in teams.yaml"
                for alias, team_id in suggestions
            )
            or "no close matches in the registry"
        )
        super().__init__(f"unresolved team name {name!r}: {report}")


class AliasCollisionError(ValueError):
    """A name or alias maps to more than one team."""

    def __init__(self, alias: str, existing_id: str, new_id: str) -> None:
        self.alias = alias
        super().__init__(
            f"alias {alias!r} of team {new_id!r} already resolves to team {existing_id!r}"
        )


class TeamRegistry:
    """Maps display names and aliases to canonical :class:`Team` records."""

    def __init__(self, teams: Iterable[Team] = ()) -> None:
        self._teams: dict[str, Team] = {}
        self._alias_to_id: dict[str, str] = {}
        for team in teams:
            self.register(team)

    @classmethod
    def from_yaml(cls, path: Path) -> "TeamRegistry":
        """Load a registry seed (a pack's ``teams.yaml``)."""
        raw = yaml.safe_load(path.read_text())
        teams = [
            Team(
                team_id=entry["id"],
                name=entry["name"],
                aliases=frozenset(entry.get("aliases", [])),
            )
            for entry in raw["teams"]
        ]
        return cls(teams)

    def register(self, team: Team) -> None:
        """Explicitly add a team; collisions with existing names/aliases raise."""
        if team.team_id in self._teams:
            raise AliasCollisionError(team.team_id, team.team_id, team.team_id)
        keys = {_normalise(team.name), *(_normalise(a) for a in team.aliases)}
        for key in keys:
            existing = self._alias_to_id.get(key)
            if existing is not None:
                raise AliasCollisionError(key, existing, team.team_id)
        self._teams[team.team_id] = team
        for key in keys:
            self._alias_to_id[key] = team.team_id

    def resolve(self, name: str) -> str:
        """Resolve a display name/alias to a canonical team_id, or raise loudly."""
        team_id = self._alias_to_id.get(_normalise(name))
        if team_id is None:
            raise UnresolvedTeamError(name, self.suggest(name))
        return team_id

    def suggest(self, name: str) -> list[tuple[str, str]]:
        """Close alias matches for an unknown name, as (alias, team_id) pairs."""
        close = difflib.get_close_matches(_normalise(name), self._alias_to_id, n=3, cutoff=0.75)
        return [(alias, self._alias_to_id[alias]) for alias in close]

    def get(self, team_id: str) -> Team:
        return self._teams[team_id]

    def __contains__(self, team_id: str) -> bool:
        return team_id in self._teams

    def __len__(self) -> int:
        return len(self._teams)
