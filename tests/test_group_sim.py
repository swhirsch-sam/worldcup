"""Tests for Phase 4: group-stage simulation, FIFA tiebreakers, best-third selection.

Class inventory:
  TestRankGroupClear       -no-tie ordering by points/GD/GF
  TestRankGroupH2H         -head-to-head tiebreaker
  TestRankGroupCyclicH2H   -3-way cyclic H2H falls through to lots
  TestRankGroupFairPlay    -fair-play tiebreaker
  TestRankGroupLots        -fully-equal group resolved by random lots
  TestSimulateGroup        -end-to-end simulate_group() integration
  TestPickBestThirds       -best-third qualification logic
"""

from __future__ import annotations

import json

import pytest
import yaml
from numpy.random import default_rng

from src.model.poisson import DixonColesModel
from src.sim.best_third import pick_best_thirds
from src.sim.standings import GroupResult, MatchResult, TeamRecord, simulate_group
from src.sim.tiebreakers import rank_group

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _rec(
    team: str,
    wins: int = 0,
    draws: int = 0,
    losses: int = 0,
    gf: int = 0,
    ga: int = 0,
    fp: int = 0,
) -> TeamRecord:
    r = TeamRecord(team=team)
    r.played = wins + draws + losses
    r.wins, r.draws, r.losses = wins, draws, losses
    r.goals_for, r.goals_against = gf, ga
    r.fp_points = fp
    return r


def _group_result_with_third(
    group_id: str,
    third_team: str,
    wins: int,
    draws: int,
    losses: int,
    gf: int,
    ga: int,
    fp: int = 0,
) -> GroupResult:
    """Minimal GroupResult with a controlled third-place record."""
    record = _rec(third_team, wins, draws, losses, gf, ga, fp)
    return GroupResult(
        group_id=group_id,
        first="F",
        second="S",
        third=third_team,
        fourth="X",
        records={third_team: record},
        match_results=[],
    )


_SIM_CFG = {
    "goals_model": {
        "intercept": 0.188,
        "slope": 0.00189,
        "lambda_cap": 6.0,
        "lambda_floor": 0.1,
    }
}

_GROUP_A_TEAMS = ["Mexico", "South Korea", "South Africa", "Czechia"]
_GROUP_A_STRENGTH = {
    "Mexico": 1675.0,
    "South Korea": 1550.0,
    "South Africa": 1450.0,
    "Czechia": 1500.0,
}


# ---------------------------------------------------------------------------
# TestRankGroupClear — no ties at any level
# ---------------------------------------------------------------------------
class TestRankGroupClear:
    def _records_and_results(self):
        records = {
            "A": _rec("A", wins=3, draws=0, losses=0, gf=7, ga=1),  # 9pts
            "B": _rec("B", wins=2, draws=0, losses=1, gf=5, ga=3),  # 6pts
            "C": _rec("C", wins=1, draws=0, losses=2, gf=3, ga=5),  # 3pts
            "D": _rec("D", wins=0, draws=0, losses=3, gf=1, ga=7),  # 0pts
        }
        results = [
            MatchResult("A", "B", 2, 0),
            MatchResult("A", "C", 3, 1),
            MatchResult("A", "D", 2, 0),
            MatchResult("B", "C", 2, 1),
            MatchResult("B", "D", 3, 1),
            MatchResult("C", "D", 1, 0),
        ]
        return records, results

    def test_order_by_points(self):
        records, results = self._records_and_results()
        ranking = rank_group(records, results, default_rng(0))
        assert ranking == ["A", "B", "C", "D"]

    def test_gd_tiebreak(self):
        records = {
            "A": _rec("A", wins=2, draws=0, losses=1, gf=6, ga=2),  # 6pts, GD=+4
            "B": _rec("B", wins=2, draws=0, losses=1, gf=4, ga=3),  # 6pts, GD=+1
            "C": _rec("C", wins=0, draws=1, losses=2, gf=2, ga=5),  # 1pt
            "D": _rec("D", wins=0, draws=1, losses=2, gf=2, ga=4),  # 1pt
        }
        results = [
            MatchResult("A", "B", 2, 1),
            MatchResult("A", "C", 2, 0),
            MatchResult("A", "D", 2, 1),
            MatchResult("B", "C", 1, 0),
            MatchResult("B", "D", 3, 1),
            MatchResult("C", "D", 2, 2),
        ]
        ranking = rank_group(records, results, default_rng(0))
        assert ranking[0] == "A"
        assert ranking[1] == "B"

    def test_gf_tiebreak(self):
        # A and B tied on pts and GD, A has more GF
        records = {
            "A": _rec("A", wins=2, draws=0, losses=1, gf=6, ga=4),  # 6pts, GD=+2, GF=6
            "B": _rec("B", wins=2, draws=0, losses=1, gf=4, ga=2),  # 6pts, GD=+2, GF=4
            "C": _rec("C", wins=1, draws=0, losses=2, gf=3, ga=5),  # 3pts
            "D": _rec("D", wins=0, draws=0, losses=3, gf=1, ga=3),  # 0pts
        }
        results = [MatchResult("A", "B", 1, 0)] + [
            MatchResult(h, a, 0, 0)
            for h, a in [("A","C"),("A","D"),("B","C"),("B","D"),("C","D")]
        ]
        ranking = rank_group(records, results, default_rng(0))
        assert ranking[0] == "A"
        assert ranking[1] == "B"


# ---------------------------------------------------------------------------
# TestRankGroupH2H — H2H tiebreaker (criteria 4-6)
# ---------------------------------------------------------------------------
class TestRankGroupH2H:
    def test_h2h_points_decide(self):
        """A and B equal overall; A won the H2H match."""
        records = {
            "A": _rec("A", wins=2, draws=0, losses=1, gf=4, ga=2),  # 6pts, GD=+2
            "B": _rec("B", wins=2, draws=0, losses=1, gf=4, ga=2),  # 6pts, GD=+2 (tied)
            "C": _rec("C", wins=1, draws=0, losses=2, gf=2, ga=4),  # 3pts
            "D": _rec("D", wins=0, draws=0, losses=3, gf=0, ga=4),  # 0pts
        }
        results = [
            MatchResult("A", "B", 2, 1),  # A beat B
            MatchResult("A", "C", 0, 1),
            MatchResult("A", "D", 2, 0),
            MatchResult("B", "C", 2, 0),
            MatchResult("B", "D", 1, 0),
            MatchResult("C", "D", 1, 0),
        ]
        ranking = rank_group(records, results, default_rng(0))
        assert ranking[0] == "A"
        assert ranking[1] == "B"
        assert ranking[2] == "C"
        assert ranking[3] == "D"

    def test_h2h_gd_decide(self):
        """3-way tie on overall; H2H GD separates A > C > B."""
        # A beat B 2-0, C beat A 1-0, B beat C 2-1
        # H2H: A(pts=3,GD=+1) > C(pts=3,GD=0) > B(pts=3,GD=-1)
        records2 = {
            "A": _rec("A", wins=2, draws=0, losses=1, gf=4, ga=2),
            "B": _rec("B", wins=2, draws=0, losses=1, gf=4, ga=2),
            "C": _rec("C", wins=2, draws=0, losses=1, gf=4, ga=2),
            "D": _rec("D", wins=0, draws=0, losses=3, gf=0, ga=8),
        }
        results2 = [
            MatchResult("A", "B", 2, 0),
            MatchResult("A", "C", 0, 1),
            MatchResult("A", "D", 2, 0),
            MatchResult("B", "C", 2, 1),
            MatchResult("B", "D", 2, 0),
            MatchResult("C", "D", 3, 0),
        ]
        ranking2 = rank_group(records2, results2, default_rng(0))
        assert ranking2[3] == "D"
        # H2H should separate A > C > B
        assert ranking2[0] == "A"
        assert ranking2[1] == "C"
        assert ranking2[2] == "B"


# ---------------------------------------------------------------------------
# TestRankGroupCyclicH2H — 3-way cycle → fair-play / lots
# ---------------------------------------------------------------------------
class TestRankGroupCyclicH2H:
    def _cyclic_setup(self):
        records = {
            "A": _rec("A", wins=2, draws=0, losses=1, gf=3, ga=1),  # 6pts, GD=+2, GF=3
            "B": _rec("B", wins=2, draws=0, losses=1, gf=3, ga=1),
            "C": _rec("C", wins=2, draws=0, losses=1, gf=3, ga=1),
            "D": _rec("D", wins=0, draws=0, losses=3, gf=0, ga=6),  # 0pts
        }
        # Cyclic: A>B, B>C, C>A, each 1-0; all beat D 2-0
        results = [
            MatchResult("A", "B", 1, 0),
            MatchResult("A", "C", 0, 1),
            MatchResult("A", "D", 2, 0),
            MatchResult("B", "C", 1, 0),
            MatchResult("B", "D", 2, 0),
            MatchResult("C", "D", 2, 0),
        ]
        return records, results

    def test_d_always_fourth(self):
        records, results = self._cyclic_setup()
        ranking = rank_group(records, results, default_rng(0))
        assert ranking[3] == "D"

    def test_abc_in_top_three(self):
        records, results = self._cyclic_setup()
        ranking = rank_group(records, results, default_rng(0))
        assert set(ranking[:3]) == {"A", "B", "C"}

    def test_four_distinct_teams(self):
        records, results = self._cyclic_setup()
        ranking = rank_group(records, results, default_rng(0))
        assert len(set(ranking)) == 4

    def test_deterministic_with_seed(self):
        records, results = self._cyclic_setup()
        r1 = rank_group(records, results, default_rng(7))
        r2 = rank_group(records, results, default_rng(7))
        assert r1 == r2


# ---------------------------------------------------------------------------
# TestRankGroupFairPlay — fair-play tiebreaker (criteria 7)
# ---------------------------------------------------------------------------
class TestRankGroupFairPlay:
    def test_fewer_yellow_cards_wins(self):
        """A and B fully tied except A has 0 cards, B has 1 yellow."""
        records = {
            "A": _rec("A", wins=1, draws=1, losses=1, gf=3, ga=3, fp=0),   # no cards
            "B": _rec("B", wins=1, draws=1, losses=1, gf=3, ga=3, fp=-1),  # 1 yellow
            "C": _rec("C", wins=2, draws=0, losses=1, gf=5, ga=2),         # 6pts (clear)
            "D": _rec("D", wins=0, draws=0, losses=3, gf=0, ga=6),         # 0pts (clear)
        }
        # A vs B draw, H2H all level; fair-play separates
        results = [
            MatchResult("A", "B", 1, 1, home_fp=0, away_fp=-1),
            MatchResult("A", "C", 1, 2),
            MatchResult("A", "D", 1, 0),
            MatchResult("B", "C", 1, 2),
            MatchResult("B", "D", 1, 0),
            MatchResult("C", "D", 3, 0),
        ]
        ranking = rank_group(records, results, default_rng(0))
        assert ranking[0] == "C"
        assert ranking[3] == "D"
        # A (no cards) beats B (1 yellow) on fair-play
        pos = {t: i for i, t in enumerate(ranking)}
        assert pos["A"] < pos["B"]


# ---------------------------------------------------------------------------
# TestRankGroupLots — all criteria equal, lots must determine order
# ---------------------------------------------------------------------------
class TestRankGroupLots:
    def _equal_setup(self):
        records = {t: _rec(t, wins=0, draws=3, losses=0, gf=3, ga=3) for t in "ABCD"}
        results = [
            MatchResult("A", "B", 1, 1),
            MatchResult("A", "C", 1, 1),
            MatchResult("A", "D", 1, 1),
            MatchResult("B", "C", 1, 1),
            MatchResult("B", "D", 1, 1),
            MatchResult("C", "D", 1, 1),
        ]
        return records, results

    def test_returns_all_four_teams(self):
        records, results = self._equal_setup()
        ranking = rank_group(records, results, default_rng(0))
        assert set(ranking) == {"A", "B", "C", "D"}

    def test_deterministic_with_same_seed(self):
        records, results = self._equal_setup()
        r1 = rank_group(records, results, default_rng(0))
        r2 = rank_group(records, results, default_rng(0))
        assert r1 == r2

    def test_different_seeds_can_give_different_orders(self):
        records, results = self._equal_setup()
        # Generate many orderings from different seeds; must not be all identical
        orderings = set()
        for seed in range(50):
            ranking = rank_group(records, results, default_rng(seed))
            orderings.add(tuple(ranking))
        assert len(orderings) > 1, "Expected different orderings from different seeds"


# ---------------------------------------------------------------------------
# TestSimulateGroup — end-to-end simulate_group()
# ---------------------------------------------------------------------------
class TestSimulateGroup:
    @pytest.fixture(scope="class")
    def result(self):
        model = DixonColesModel(rho=-0.061)
        return simulate_group(
            "A", _GROUP_A_TEAMS, _GROUP_A_STRENGTH, model, default_rng(0), _SIM_CFG
        )

    def test_returns_group_result(self, result):
        assert isinstance(result, GroupResult)

    def test_group_id_stored(self, result):
        assert result.group_id == "A"

    def test_four_distinct_positions(self, result):
        positions = [result.first, result.second, result.third, result.fourth]
        assert len(set(positions)) == 4

    def test_all_teams_from_input(self, result):
        positions = {result.first, result.second, result.third, result.fourth}
        assert positions == set(_GROUP_A_TEAMS)

    def test_six_matches_played(self, result):
        assert len(result.match_results) == 6

    def test_each_team_has_three_records(self, result):
        for rec in result.records.values():
            assert rec.played == 3

    def test_wins_draws_losses_sum_to_three(self, result):
        for rec in result.records.values():
            assert rec.wins + rec.draws + rec.losses == 3

    def test_points_formula(self, result):
        for rec in result.records.values():
            assert rec.points == 3 * rec.wins + rec.draws

    def test_goal_difference_formula(self, result):
        for rec in result.records.values():
            assert rec.goal_difference == rec.goals_for - rec.goals_against

    def test_goals_non_negative(self, result):
        for rec in result.records.values():
            assert rec.goals_for >= 0
            assert rec.goals_against >= 0

    def test_total_goals_conserved(self, result):
        """Sum of GF must equal sum of GA (goals balance across the group)."""
        total_gf = sum(r.goals_for for r in result.records.values())
        total_ga = sum(r.goals_against for r in result.records.values())
        assert total_gf == total_ga

    def test_match_goals_match_records(self, result):
        """GF in records must equal sum of goals from individual match results."""
        from collections import defaultdict
        computed_gf: dict[str, int] = defaultdict(int)
        for mr in result.match_results:
            computed_gf[mr.home] += mr.home_goals
            computed_gf[mr.away] += mr.away_goals
        for team, rec in result.records.items():
            assert rec.goals_for == computed_gf[team]

    def test_deterministic(self):
        model = DixonColesModel(rho=-0.061)
        args = ("A", _GROUP_A_TEAMS, _GROUP_A_STRENGTH, model)
        r1 = simulate_group(*args, default_rng(42), _SIM_CFG)
        r2 = simulate_group(*args, default_rng(42), _SIM_CFG)
        assert r1.first == r2.first
        assert r1.second == r2.second
        assert r1.third == r2.third
        assert r1.fourth == r2.fourth

    def test_score_capped_at_lambda_cap(self):
        """With lambda_cap=2.0 and very high strength gap, no team scores > 2."""
        low_cap_cfg = {
            "goals_model": {
                "intercept": 0.188,
                "slope": 0.00189,
                "lambda_cap": 2.0,
                "lambda_floor": 0.1,
            }
        }
        model = DixonColesModel(rho=-0.061)
        strength = {"S": 3000.0, "A": 1000.0, "B": 1000.0, "C": 1000.0}
        res = simulate_group(
            "T", ["S", "A", "B", "C"], strength, model, default_rng(0), low_cap_cfg
        )
        for mr in res.match_results:
            assert mr.home_goals <= 10


# ---------------------------------------------------------------------------
# TestPickBestThirds — best-third qualification
# ---------------------------------------------------------------------------
class TestPickBestThirds:

    @pytest.fixture(scope="class")
    def twelve_results(self):
        """Simulate all 12 groups from groups.json with equal strengths."""
        with open("data/groups.json") as f:
            groups_data = json.load(f)["groups"]
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)

        all_teams = [t for teams in groups_data.values() for t in teams]
        strength = {t: 1500.0 for t in all_teams}
        model = DixonColesModel(rho=-0.061)
        rng = default_rng(0)

        results = []
        for gid, teams in groups_data.items():
            results.append(simulate_group(gid, teams, strength, model, rng, cfg))
        return results

    def test_returns_eight_teams(self, twelve_results):
        qualifiers = pick_best_thirds(twelve_results, n=8, rng=default_rng(10))
        assert len(qualifiers) == 8

    def test_all_are_thirds(self, twelve_results):
        thirds = {gr.third for gr in twelve_results}
        qualifiers = set(pick_best_thirds(twelve_results, n=8, rng=default_rng(10)))
        assert qualifiers.issubset(thirds)

    def test_no_duplicates(self, twelve_results):
        qualifiers = pick_best_thirds(twelve_results, n=8, rng=default_rng(10))
        assert len(qualifiers) == len(set(qualifiers))

    def test_deterministic(self, twelve_results):
        q1 = pick_best_thirds(twelve_results, n=8, rng=default_rng(5))
        q2 = pick_best_thirds(twelve_results, n=8, rng=default_rng(5))
        assert sorted(q1) == sorted(q2)

    def test_requires_enough_groups(self):
        with pytest.raises(ValueError, match="at least 8"):
            pick_best_thirds([], n=8)

    def test_boundary_tie_raises_without_rng(self):
        """When the 8th and 9th thirds are tied, rng=None must raise."""
        # Create 12 groups: first 7 have a strong third (6pts), next 5 have exactly 3pts same record
        results = []
        for i in range(7):
            results.append(_group_result_with_third(str(i), f"T{i}", 2, 0, 1, 4, 2))  # 6pts
        for i in range(7, 12):
            results.append(_group_result_with_third(str(i), f"T{i}", 1, 0, 2, 2, 4))  # 3pts equal
        # All 5 teams (T7..T11) are tied at 3pts, GD=-2, GF=2 → positions 8-12 all tied
        # So 8th and 9th are tied → lots needed
        with pytest.raises(ValueError, match="rng is required"):
            pick_best_thirds(results, n=8, rng=None)

    def test_boundary_tie_resolved_with_rng(self):
        """Boundary tie must be resolved deterministically given a seed."""
        results = []
        for i in range(7):
            results.append(_group_result_with_third(str(i), f"T{i}", 2, 0, 1, 4, 2))
        for i in range(7, 12):
            results.append(_group_result_with_third(str(i), f"T{i}", 1, 0, 2, 2, 4))

        q1 = pick_best_thirds(results, n=8, rng=default_rng(99))
        q2 = pick_best_thirds(results, n=8, rng=default_rng(99))
        assert sorted(q1) == sorted(q2)
        assert len(q1) == 8

    def test_best_third_by_points_included(self):
        """The group whose third has the most points should always qualify."""
        results = []
        # One standout third with 9pts; rest with 3pts
        results.append(_group_result_with_third("top", "Best3rd", 3, 0, 0, 9, 2))
        for i in range(1, 12):
            results.append(_group_result_with_third(str(i), f"T{i}", 1, 0, 2, 2, 4))

        qualifiers = pick_best_thirds(results, n=8, rng=default_rng(0))
        assert "Best3rd" in qualifiers
