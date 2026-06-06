"""Unit tests for the name resolver (src/ingest/names.py).

Tests:
  - Known aliases resolve correctly
  - Canonical names pass through unchanged
  - Unmapped names raise NameResolutionError
  - resolve_all propagates errors
"""

from __future__ import annotations

import pytest

from src.ingest.names import CANONICAL_TEAMS, NameResolutionError, resolve, resolve_all


class TestKnownAliases:
    def test_usa_resolves(self) -> None:
        assert resolve("USA") == "United States"

    def test_us_resolves(self) -> None:
        assert resolve("US") == "United States"

    def test_united_states_of_america_resolves(self) -> None:
        assert resolve("United States of America") == "United States"

    def test_korea_republic_resolves(self) -> None:
        assert resolve("Korea Republic") == "South Korea"

    def test_republic_of_korea_resolves(self) -> None:
        assert resolve("Republic of Korea") == "South Korea"

    def test_ir_iran_resolves(self) -> None:
        assert resolve("IR Iran") == "Iran"

    def test_islamic_republic_of_iran_resolves(self) -> None:
        assert resolve("Islamic Republic of Iran") == "Iran"

    def test_ivory_coast_resolves(self) -> None:
        assert resolve("Ivory Coast") == "Côte d'Ivoire"

    def test_dr_congo_variant_resolves(self) -> None:
        assert resolve("Congo DR") == "DR Congo"

    def test_curacao_ascii_resolves(self) -> None:
        assert resolve("Curacao") == "Curaçao"


class TestNewAliases:
    def test_czech_republic_resolves(self) -> None:
        assert resolve("Czech Republic") == "Czechia"

    def test_turkey_resolves(self) -> None:
        assert resolve("Turkey") == "Türkiye"

    def test_cape_verde_resolves(self) -> None:
        assert resolve("Cape Verde") == "Cabo Verde"

    def test_bosnia_hyphen_resolves(self) -> None:
        assert resolve("Bosnia-Herzegovina") == "Bosnia and Herzegovina"

    def test_congo_comma_resolves(self) -> None:
        assert resolve("Congo, DR") == "DR Congo"


class TestPassthrough:
    def test_canonical_name_passes_through(self) -> None:
        assert resolve("United States") == "United States"

    def test_south_korea_canonical_passes(self) -> None:
        assert resolve("South Korea") == "South Korea"

    def test_turkiye_canonical_passes(self) -> None:
        assert resolve("Türkiye") == "Türkiye"

    def test_cabo_verde_canonical_passes(self) -> None:
        assert resolve("Cabo Verde") == "Cabo Verde"


class TestCanonicalRegistry:
    def test_registry_has_48_teams(self) -> None:
        assert len(CANONICAL_TEAMS) == 48

    def test_all_three_hosts_in_registry(self) -> None:
        for host in ("Mexico", "Canada", "United States"):
            assert host in CANONICAL_TEAMS

    def test_all_12_groups_represented(self) -> None:
        import json

        with open("data/groups.json") as f:
            groups = json.load(f)["groups"]
        for group_label, teams in groups.items():
            for team in teams:
                assert team in CANONICAL_TEAMS, f"{team} (Group {group_label}) not in registry"


class TestErrors:
    def test_unmapped_name_raises(self) -> None:
        with pytest.raises(NameResolutionError, match="not in canonical registry"):
            resolve("Nonexistentland FC")

    def test_resolve_all_propagates_error(self) -> None:
        result = resolve_all(["USA", "Korea Republic"])
        assert result == ["United States", "South Korea"]

    def test_resolve_all_raises_on_bad_name(self) -> None:
        with pytest.raises(NameResolutionError):
            resolve_all(["Brazil", "NotATeam"])
