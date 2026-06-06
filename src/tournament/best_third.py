"""Best third-place team selection and bracket allocation.

After all 12 groups complete, rank the 12 third-placed teams by:
  1. Points
  2. Goal difference
  3. Goals scored
  4. Fair-play (disciplinary record)
  5. Drawing of lots

Select the top 8 and look up their bracket slots in bracket_map.json.
"""

from __future__ import annotations

from typing import Any

from numpy.random import Generator

from src.tournament.standings import TeamStanding

# The 12 valid third-place group combinations → bracket slot mapping.
# Populated from bracket_map.json in Phase 5 once the file is confirmed.
THIRD_PLACE_GROUP_COMBINATIONS: dict[str, Any] = {}


def rank_third_place_teams(
    third_place_standings: list[TeamStanding],
    rng: Generator,
) -> list[TeamStanding]:
    """Rank all 12 third-placed teams and return the top 8 in order.

    Args:
        third_place_standings: One TeamStanding per group, the third-placed team.
        rng: For drawing of lots if still tied after all criteria.

    Returns:
        List of exactly 8 TeamStanding objects in ranked order.
    """
    raise NotImplementedError("rank_third_place_teams: implement in Phase 4.")


def assign_best_third_to_bracket(
    best_third: list[TeamStanding],
    bracket_map: dict[str, Any],
) -> dict[str, str]:
    """Return a mapping of bracket slot → team name for best-third teams.

    The combination of which groups produced the 8 best thirds determines
    which bracket slots they fill (per FIFA allocation table).
    """
    raise NotImplementedError("assign_best_third_to_bracket: implement in Phase 5.")
