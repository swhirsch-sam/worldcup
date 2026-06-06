"""Group-stage simulation: round-robin matches and standings.

simulate_group() plays out all 6 pairings for a 4-team group, builds per-team
records, and returns a GroupResult with teams ranked 1st-4th via FIFA
tiebreaker rules.

All WC 2026 group matches are treated as neutral (no home advantage additive
on top of the host-bump already baked into strength ratings).
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.random import Generator

if TYPE_CHECKING:
    from src.model.poisson import MatchSimulator

from src.sim.tiebreakers import rank_group

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    home: str
    away: str
    home_goals: int
    away_goals: int
    home_fp: int = 0  # fair-play penalty for home team (yellow=-1, red=-3)
    away_fp: int = 0


@dataclass
class TeamRecord:
    team: str
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    fp_points: int = 0  # sum of fair-play penalties (non-positive)

    @property
    def points(self) -> int:
        return 3 * self.wins + self.draws

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against


@dataclass
class GroupResult:
    group_id: str
    first: str
    second: str
    third: str
    fourth: str
    records: dict[str, TeamRecord] = field(default_factory=dict)
    match_results: list[MatchResult] = field(default_factory=list)


def simulate_group(
    group_id: str,
    teams: list[str],
    strength: dict[str, float],
    model: MatchSimulator,
    rng: Generator,
    cfg: dict[str, Any],
) -> GroupResult:
    """Simulate one 4-team round-robin and return ranked standings.

    Args:
        group_id: Group label ('A'-'L').
        teams: The 4 team names.
        strength: team to composite Elo-scale strength (from strength.py).
        model: MatchSimulator (DixonColesModel or BivariatePoissonModel).
        rng: NumPy random Generator.
        cfg: Parsed config dict.

    Returns:
        GroupResult with first/second/third/fourth and all supporting records.
    """
    alpha = float(cfg["goals_model"]["intercept"])
    beta = float(cfg["goals_model"]["slope"])
    lam_cap = float(cfg["goals_model"].get("lambda_cap", 6.0))
    lam_floor = float(cfg["goals_model"].get("lambda_floor", 0.1))

    # itertools.combinations gives C(4,2)=6 pairings; first element is "home"
    pairings = list(itertools.combinations(teams, 2))

    match_results: list[MatchResult] = []
    for home_team, away_team in pairings:
        # Neutral match: no home-advantage term (host bump already in strength)
        elo_diff = strength[home_team] - strength[away_team]
        lam_h = float(np.clip(np.exp(alpha + beta * elo_diff), lam_floor, lam_cap))
        lam_a = float(np.clip(np.exp(alpha - beta * elo_diff), lam_floor, lam_cap))
        h_goals, a_goals = model.simulate_match(lam_h, lam_a, rng)
        match_results.append(MatchResult(home_team, away_team, int(h_goals), int(a_goals)))

    records: dict[str, TeamRecord] = {t: TeamRecord(team=t) for t in teams}
    for mr in match_results:
        h_rec = records[mr.home]
        a_rec = records[mr.away]
        h_rec.played += 1
        a_rec.played += 1
        h_rec.goals_for += mr.home_goals
        h_rec.goals_against += mr.away_goals
        a_rec.goals_for += mr.away_goals
        a_rec.goals_against += mr.home_goals
        h_rec.fp_points += mr.home_fp
        a_rec.fp_points += mr.away_fp
        if mr.home_goals > mr.away_goals:
            h_rec.wins += 1
            a_rec.losses += 1
        elif mr.home_goals == mr.away_goals:
            h_rec.draws += 1
            a_rec.draws += 1
        else:
            a_rec.wins += 1
            h_rec.losses += 1

    ranking = rank_group(records, match_results, rng)
    logger.debug(
        "Group %s: %s > %s > %s > %s",
        group_id,
        ranking[0],
        ranking[1],
        ranking[2],
        ranking[3],
    )
    return GroupResult(
        group_id=group_id,
        first=ranking[0],
        second=ranking[1],
        third=ranking[2],
        fourth=ranking[3],
        records=records,
        match_results=match_results,
    )
