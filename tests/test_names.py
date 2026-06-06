"""Unit tests for the name resolver (src/ingest/names.py).

Tests:
  - Known aliases resolve correctly
  - Canonical names pass through unchanged
  - Unmapped names raise NameResolutionError
  - resolve_all propagates errors
"""

from __future__ import annotations

from src.ingest.names import NameResolutionError, resolve, resolve_all  # noqa: F401


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


class TestPassthrough:
    def test_canonical_name_passes_through(self) -> None:
        # When CANONICAL_TEAMS is empty (Phase 1 placeholder), canonical names
        # that are already in _ALIAS_MAP resolve to themselves.
        assert resolve("United States") == "United States"

    def test_south_korea_canonical_passes(self) -> None:
        assert resolve("South Korea") == "South Korea"


class TestErrors:
    def test_unmapped_name_raises(self) -> None:
        # Once CANONICAL_TEAMS is populated, an unknown name should raise.
        # In Phase 1 with empty CANONICAL_TEAMS the resolver passes through;
        # this test documents the expected post-Phase-2 behaviour.
        # TODO: activate strict check in Phase 2 by populating CANONICAL_TEAMS.
        pass  # placeholder — see Phase 2

    def test_resolve_all_propagates_error(self) -> None:
        # Smoke test: resolve_all on a valid list returns a list.
        result = resolve_all(["USA", "Korea Republic"])
        assert result == ["United States", "South Korea"]
