from pathlib import Path

import pytest

from engine.core.registry import (
    AliasCollisionError,
    TeamRegistry,
    UnresolvedTeamError,
    slugify,
)
from engine.core.schema import Team

USA = Team(
    team_id="usa",
    name="United States",
    aliases=frozenset({"USA", "USMNT", "United States of America"}),
)
SOUTH_KOREA = Team(team_id="south_korea", name="South Korea", aliases=frozenset({"Korea Republic"}))


@pytest.fixture
def registry() -> TeamRegistry:
    return TeamRegistry([USA, SOUTH_KOREA])


class TestSlugify:
    def test_basic(self) -> None:
        assert slugify("South Korea") == "south_korea"
        assert slugify("Côte d'Ivoire".replace("ô", "o")) == "cote_d_ivoire"

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            slugify("---")


class TestResolve:
    def test_canonical_name(self, registry: TeamRegistry) -> None:
        assert registry.resolve("United States") == "usa"

    def test_alias(self, registry: TeamRegistry) -> None:
        assert registry.resolve("USMNT") == "usa"
        assert registry.resolve("Korea Republic") == "south_korea"

    def test_case_and_whitespace_insensitive(self, registry: TeamRegistry) -> None:
        assert registry.resolve("  usa ") == "usa"
        assert registry.resolve("SOUTH  KOREA") == "south_korea"

    def test_unresolved_raises_with_suggestions(self, registry: TeamRegistry) -> None:
        with pytest.raises(UnresolvedTeamError) as excinfo:
            registry.resolve("Korea Repblic")
        err = excinfo.value
        assert err.name == "Korea Repblic"
        assert ("korea republic", "south_korea") in err.suggestions
        assert "teams.yaml" in str(err)

    def test_unresolved_without_close_match(self, registry: TeamRegistry) -> None:
        with pytest.raises(UnresolvedTeamError) as excinfo:
            registry.resolve("Brazil")
        assert excinfo.value.suggestions == []
        assert "no close matches" in str(excinfo.value)


class TestRegister:
    def test_explicit_registration(self, registry: TeamRegistry) -> None:
        registry.register(Team(team_id="brazil", name="Brazil"))
        assert registry.resolve("Brazil") == "brazil"
        assert len(registry) == 3

    def test_alias_collision_raises(self, registry: TeamRegistry) -> None:
        clash = Team(team_id="us_soccer", name="US Soccer", aliases=frozenset({"USA"}))
        with pytest.raises(AliasCollisionError, match="usa"):
            registry.register(clash)
        # Failed registration must not leave partial state behind.
        assert "us_soccer" not in registry

    def test_duplicate_id_raises(self, registry: TeamRegistry) -> None:
        with pytest.raises(AliasCollisionError):
            registry.register(Team(team_id="usa", name="Somewhere Else"))


class TestFromYaml(TestResolve):
    """Run the resolve suite against a registry loaded from YAML."""

    @pytest.fixture
    def registry(self, tmp_path: Path) -> TeamRegistry:
        yaml_path = tmp_path / "teams.yaml"
        yaml_path.write_text(
            """
teams:
  - id: usa
    name: United States
    aliases: [USA, USMNT, United States of America]
  - id: south_korea
    name: South Korea
    aliases: [Korea Republic]
"""
        )
        return TeamRegistry.from_yaml(yaml_path)
