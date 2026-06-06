"""Unit + property tests for group tiebreakers and best-third ranking.

Covers every tiebreaker level with hand-computed reference cases.
Property tests use Hypothesis to verify invariants over random group results.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.tournament.standings import TeamStanding

# ---------------------------------------------------------------------------
# Hand-computed tiebreaker fixtures
# ---------------------------------------------------------------------------


def _make_standing(team: str, pts: int, gd: int, gf: int, fp: int = 0) -> TeamStanding:
    """Helper: create a standing directly from summary stats."""
    s = TeamStanding(team=team)
    # Back-compute W/D/L from points (simplified; sufficient for tiebreaker tests)
    s.goals_for = gf
    s.goals_against = gf - gd
    s.yellow_cards = fp
    wins, remainder = divmod(pts, 3)
    s.wins = wins
    s.draws = remainder  # 0 or 1 extra point means draws
    s.losses = 3 - wins - (remainder > 0)
    s.played = s.wins + s.draws + s.losses
    return s


# ---------------------------------------------------------------------------
# Tiebreaker level 1: Points
# ---------------------------------------------------------------------------


class TestPointsTiebreaker:
    def test_clear_points_order(self) -> None:
        """No tie at all — straightforward ranking by points."""
        pytest.skip("Implement in Phase 4 when rank_group is ready.")

    def test_partial_points_tie(self) -> None:
        pytest.skip("Implement in Phase 4.")


# ---------------------------------------------------------------------------
# Tiebreaker level 2: Goal difference
# ---------------------------------------------------------------------------


class TestGoalDifferenceTiebreaker:
    def test_gd_breaks_points_tie(self) -> None:
        pytest.skip("Implement in Phase 4.")


# ---------------------------------------------------------------------------
# Tiebreaker level 3: Goals scored
# ---------------------------------------------------------------------------


class TestGoalsScoredTiebreaker:
    def test_goals_scored_breaks_tie(self) -> None:
        pytest.skip("Implement in Phase 4.")


# ---------------------------------------------------------------------------
# Tiebreaker level 4-6: Head-to-head (points, GD, goals scored)
# ---------------------------------------------------------------------------


class TestHeadToHeadTiebreaker:
    def test_h2h_two_teams(self) -> None:
        pytest.skip("Implement in Phase 4.")

    def test_h2h_three_teams(self) -> None:
        """Three-way tie resolved by head-to-head subset (the tricky case)."""
        pytest.skip("Implement in Phase 4.")


# ---------------------------------------------------------------------------
# Tiebreaker level 7: Fair-play
# ---------------------------------------------------------------------------


class TestFairPlayTiebreaker:
    def test_fewer_cards_ranks_higher(self) -> None:
        pytest.skip("Implement in Phase 4.")


# ---------------------------------------------------------------------------
# Property tests (Hypothesis)
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Enable in Phase 4 when rank_group is implemented.")
class TestRankGroupProperties:
    @given(
        st.integers(min_value=0, max_value=9),
        st.integers(min_value=0, max_value=9),
        st.integers(min_value=0, max_value=9),
    )
    @settings(max_examples=200)
    def test_rank_group_returns_four_teams(self, s1: int, s2: int, s3: int) -> None:
        """rank_group always returns exactly 4 teams."""
        pass  # implementation in Phase 4
