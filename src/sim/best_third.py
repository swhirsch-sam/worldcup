"""Select the best n third-place teams from all group results.

FIFA ranking criteria for best-third comparison (WC 2026):
  1. Points
  2. Goal difference (all group matches)
  3. Goals scored (all group matches)
  4. Fair-play points (yellow=-1, red/2nd-yellow=-3)
  5. Drawing of lots

All third-place teams played exactly 3 group matches, so comparisons
are on equal footing without any match-exclusion complication.
"""

from __future__ import annotations

from typing import Any

from numpy.random import Generator

from src.sim.standings import GroupResult, TeamRecord


def pick_best_thirds(
    group_results: list[GroupResult],
    *,
    n: int = 8,
    rng: Generator | None = None,
    cfg: dict[str, Any] | None = None,
) -> list[str]:
    """Return the n best third-place teams ranked by FIFA criteria.

    Args:
        group_results: Results of all groups (must have >= n groups).
        n: Number of best thirds to qualify (8 for WC 2026).
        rng: Generator for lots; required only when a boundary tie occurs.
        cfg: Config dict (unused; reserved for future criteria changes).

    Returns:
        Unordered list of n qualifying team names.
    """
    if len(group_results) < n:
        raise ValueError(f"pick_best_thirds requires at least {n} groups; got {len(group_results)}")

    thirds = [(gr.third, gr.records[gr.third]) for gr in group_results]

    def sort_key(item: tuple[str, TeamRecord]) -> tuple[int, int, int, int]:
        _, r = item
        # Ascending sort: lower tuple = better rank.
        # fp_points is non-positive; -fp_points puts 0 (no cards) first.
        return (-r.points, -r.goal_difference, -r.goals_for, -r.fp_points)

    thirds_sorted = sorted(thirds, key=sort_key)

    if n >= len(thirds_sorted):
        return [t[0] for t in thirds_sorted]

    bkey_last_in = sort_key(thirds_sorted[n - 1])
    bkey_first_out = sort_key(thirds_sorted[n])

    if bkey_last_in != bkey_first_out:
        return [t[0] for t in thirds_sorted[:n]]

    # Boundary tie: split into secure (strictly better) and the boundary group
    secure = [t[0] for t in thirds_sorted if sort_key(t) < bkey_last_in]
    boundary = [t[0] for t in thirds_sorted if sort_key(t) == bkey_last_in]
    needed = n - len(secure)

    if needed == len(boundary):
        return secure + boundary

    if rng is None:
        raise ValueError(
            "rng is required when a boundary tie in best-third selection "
            "must be resolved by drawing of lots"
        )

    chosen_idx = rng.choice(len(boundary), size=needed, replace=False)
    return secure + [boundary[i] for i in sorted(chosen_idx)]
