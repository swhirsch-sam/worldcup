"""Post-simulation invariant checks."""

from __future__ import annotations

import pytest


class TestSimulationInvariants:
    def test_champion_probs_sum_to_one(self) -> None:
        pytest.skip("Implement in Phase 6.")

    def test_monotonic_advancement_probs(self) -> None:
        """P(reach R32) >= P(reach R16) >= ... >= P(win) for every team."""
        pytest.skip("Implement in Phase 6.")

    def test_group_finish_probs_sum_to_one(self) -> None:
        pytest.skip("Implement in Phase 6.")

    def test_exactly_32_r32_teams(self) -> None:
        pytest.skip("Implement in Phase 6.")

    def test_no_duplicate_teams_in_bracket(self) -> None:
        pytest.skip("Implement in Phase 6.")

    def test_group_goals_conservation(self) -> None:
        """Total goals scored == total goals conceded within each group."""
        pytest.skip("Implement in Phase 6.")
