"""Group stage standings calculation.

Pure functions: take match results, return standing tables.
No I/O; no global state.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MatchResult:
    """Single group stage match outcome."""

    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    home_yellow_cards: int = 0
    away_yellow_cards: int = 0
    home_red_cards: int = 0
    away_red_cards: int = 0


@dataclass
class TeamStanding:
    """Accumulated standing for one team in a group."""

    team: str
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    yellow_cards: int = 0
    red_cards: int = 0

    @property
    def points(self) -> int:
        return self.wins * 3 + self.draws

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against

    @property
    def fair_play_score(self) -> int:
        """Lower is better (fewer infractions)."""
        from src.tournament import _load_config

        cfg = _load_config()
        tb = cfg["tiebreaker"]
        return self.yellow_cards * tb["yellow_card_weight"] + self.red_cards * tb["red_card_weight"]


def compute_group_standings(results: list[MatchResult]) -> list[TeamStanding]:
    """Return standing rows for all teams in a group (unsorted).

    Args:
        results: All matches played in the group so far.
    """
    raise NotImplementedError("compute_group_standings: implement in Phase 4.")


def apply_match(standing: TeamStanding, result: MatchResult, is_home: bool) -> TeamStanding:
    """Return a new TeamStanding with *result* applied for the given side."""
    raise NotImplementedError("apply_match: implement in Phase 4.")
