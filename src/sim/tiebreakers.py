"""FIFA group-stage tiebreaker procedure (8 levels).

Criteria order per FIFA WC 2026 Regulations:
  1. Points
  2. Goal difference (all group matches)
  3. Goals scored (all group matches)
  4. Head-to-head points (matches only between the tied teams)
  5. Head-to-head goal difference
  6. Head-to-head goals scored
  7. Fair-play points (yellow=-1, direct-red or 2nd-yellow=-3)
  8. Drawing of lots (random permutation via rng)

H2H (criteria 4-6) is applied once across the full tied group.
If sub-groups remain tied after H2H, criteria 7-8 resolve them —
H2H is never applied recursively.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.random import Generator

if TYPE_CHECKING:
    from src.sim.standings import MatchResult, TeamRecord


def rank_group(
    records: dict[str, "TeamRecord"],
    match_results: list["MatchResult"],
    rng: Generator,
) -> list[str]:
    """Rank all teams in a group using the full FIFA 8-level tiebreaker.

    Returns a list of team names ordered [1st, 2nd, 3rd, 4th].
    """
    teams = sorted(records.keys(), key=lambda t: _overall_key(records[t]))

    result: list[str] = []
    i = 0
    while i < len(teams):
        j = i + 1
        while j < len(teams) and _same_overall(records[teams[i]], records[teams[j]]):
            j += 1
        tied = teams[i:j]
        if len(tied) == 1:
            result.append(tied[0])
        else:
            result.extend(_resolve_tie(tied, records, match_results, rng))
        i = j
    return result


def _overall_key(r: "TeamRecord") -> tuple[int, int, int]:
    return (-r.points, -r.goal_difference, -r.goals_for)


def _same_overall(a: "TeamRecord", b: "TeamRecord") -> bool:
    return (
        a.points == b.points
        and a.goal_difference == b.goal_difference
        and a.goals_for == b.goals_for
    )


def _resolve_tie(
    tied: list[str],
    records: dict[str, "TeamRecord"],
    match_results: list["MatchResult"],
    rng: Generator,
) -> list[str]:
    """Apply H2H once, then fair-play, then lots for any remaining sub-ties."""
    tied_set = set(tied)

    # Build H2H records from matches exclusively between these teams
    h2h: dict[str, dict[str, int]] = {
        t: {"wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0}
        for t in tied
    }
    for mr in match_results:
        if mr.home in tied_set and mr.away in tied_set:
            h, a = h2h[mr.home], h2h[mr.away]
            h["gf"] += mr.home_goals
            h["ga"] += mr.away_goals
            a["gf"] += mr.away_goals
            a["ga"] += mr.home_goals
            if mr.home_goals > mr.away_goals:
                h["wins"] += 1
                a["losses"] += 1
            elif mr.home_goals == mr.away_goals:
                h["draws"] += 1
                a["draws"] += 1
            else:
                a["wins"] += 1
                h["losses"] += 1

    def h2h_pts(t: str) -> int:
        return 3 * h2h[t]["wins"] + h2h[t]["draws"]

    def h2h_gd(t: str) -> int:
        return h2h[t]["gf"] - h2h[t]["ga"]

    def h2h_gf(t: str) -> int:
        return h2h[t]["gf"]

    h2h_sorted = sorted(
        tied, key=lambda t: (-h2h_pts(t), -h2h_gd(t), -h2h_gf(t))
    )

    result: list[str] = []
    i = 0
    while i < len(h2h_sorted):
        j = i + 1
        while (
            j < len(h2h_sorted)
            and h2h_pts(h2h_sorted[i]) == h2h_pts(h2h_sorted[j])
            and h2h_gd(h2h_sorted[i]) == h2h_gd(h2h_sorted[j])
            and h2h_gf(h2h_sorted[i]) == h2h_gf(h2h_sorted[j])
        ):
            j += 1
        sub = h2h_sorted[i:j]
        if len(sub) == 1:
            result.append(sub[0])
        else:
            result.extend(_fair_play_then_lots(sub, records, rng))
        i = j
    return result


def _fair_play_then_lots(
    tied: list[str],
    records: dict[str, "TeamRecord"],
    rng: Generator,
) -> list[str]:
    """Sort by fair-play (fewer penalties = better), then random lots."""
    # fp_points is non-positive; -fp_points in ascending order puts 0 first (best)
    fp_sorted = sorted(tied, key=lambda t: -records[t].fp_points)

    result: list[str] = []
    i = 0
    while i < len(fp_sorted):
        j = i + 1
        while j < len(fp_sorted) and (
            records[fp_sorted[i]].fp_points == records[fp_sorted[j]].fp_points
        ):
            j += 1
        sub = fp_sorted[i:j]
        if len(sub) == 1:
            result.append(sub[0])
        else:
            shuffled = list(sub)
            rng.shuffle(shuffled)
            result.extend(shuffled)
        i = j
    return result
