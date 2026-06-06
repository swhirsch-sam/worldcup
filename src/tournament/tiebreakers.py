"""FIFA group stage tiebreaker rules (strict priority order).

Priority order per FIFA 2026 regulations:
  1. Points
  2. Goal difference (all group matches)
  3. Goals scored (all group matches)
  4. Head-to-head: points among tied teams only
  5. Head-to-head: goal difference among tied teams only
  6. Head-to-head: goals scored among tied teams only
  7. Fair-play (disciplinary record - fewer cards)
  8. Drawing of lots (seeded random, passed as rng argument)

These are pure functions. All tiebreaker logic is unit-tested in tests/test_tiebreakers.py
including hand-computed cases for every tiebreaker level.
"""

from __future__ import annotations

from numpy.random import Generator

from src.tournament.standings import MatchResult, TeamStanding


def rank_group(
    standings: list[TeamStanding],
    all_results: list[MatchResult],
    rng: Generator,
) -> list[TeamStanding]:
    """Return standings sorted 1st through 4th, applying tiebreakers as needed.

    Args:
        standings: Unsorted list of 4 TeamStanding objects.
        all_results: All 6 matches in the group (for head-to-head subset).
        rng: Seeded Generator for drawing of lots (tiebreaker 8).

    Returns:
        Standings sorted from 1st to 4th place.
    """
    raise NotImplementedError("rank_group: implement in Phase 4.")


def _head_to_head_subset(
    tied_teams: list[TeamStanding],
    all_results: list[MatchResult],
) -> list[TeamStanding]:
    """Recompute standings using only the matches among *tied_teams*."""
    raise NotImplementedError("_head_to_head_subset: implement in Phase 4.")


def _sort_key_primary(s: TeamStanding) -> tuple[int, int, int]:
    """(points, GD, goals_scored) — higher is better; negate for sort."""
    return (s.points, s.goal_difference, s.goals_for)


def _sort_key_h2h(s: TeamStanding) -> tuple[int, int, int]:
    """Same as primary but computed on head-to-head results."""
    return (s.points, s.goal_difference, s.goals_for)


def _sort_key_fairplay(s: TeamStanding) -> int:
    """Fair-play score — lower is better (negate for descending sort)."""
    return -s.fair_play_score
