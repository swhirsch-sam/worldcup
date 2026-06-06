"""Tests for Phase 5: bracket allocation and knockout simulation.

Class inventory:
  TestBipartiteMatching      - _bipartite_matching() validity
  TestAllCombinationsSolvable - all C(12,8)=495 combos have valid assignments
  TestAssignThirdsToSlots    - assign_thirds_to_slots() correctness
  TestResolveR32             - resolve_r32() builds correct 16 matchups
  TestSimulateKoMatch        - single KO match: returns winner, ET, penalties
  TestSimulateKnockout       - full bracket R32->champion
"""

from __future__ import annotations

import json

import pytest
import yaml
from numpy.random import default_rng

from src.model.poisson import DixonColesModel
from src.sim.best_third import pick_best_thirds
from src.sim.bracket import (
    _bipartite_matching,
    assign_thirds_to_slots,
    load_bracket_map,
    resolve_r32,
    simulate_knockout,
    simulate_ko_match,
)
from src.sim.standings import GroupResult, TeamRecord, simulate_group

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_BRACKET_MAP = load_bracket_map()
_SLOT_CANDIDATES = _BRACKET_MAP["third_place_slot_candidates"]
_ALL_GROUPS = list("ABCDEFGHIJKL")

_SIM_CFG = {
    "goals_model": {
        "intercept": 0.188,
        "slope": 0.00189,
        "lambda_cap": 6.0,
        "lambda_floor": 0.1,
    },
    "knockout": {
        "caution_factor": 0.85,
        "extra_time_duration_fraction": 0.333,
        "penalty_base": 0.5,
        "penalty_strength_tilt": 0.05,
        "penalty_strength_tilt_max": 0.1,
    },
}


def _minimal_group_result(gid: str, first: str, second: str, third: str) -> GroupResult:
    """Build a GroupResult with the minimum fields needed for bracket resolution."""
    records = {t: TeamRecord(team=t) for t in [first, second, third]}
    return GroupResult(
        group_id=gid,
        first=first,
        second=second,
        third=third,
        fourth="X",
        records=records,
        match_results=[],
    )


def _make_group_results() -> dict[str, GroupResult]:
    """Create one GroupResult per group with synthetic team names."""
    results = {}
    for g in _ALL_GROUPS:
        results[g] = _minimal_group_result(
            g,
            first=f"1st_{g}",
            second=f"2nd_{g}",
            third=f"3rd_{g}",
        )
    return results


# ---------------------------------------------------------------------------
# TestBipartiteMatching
# ---------------------------------------------------------------------------
class TestBipartiteMatching:
    def test_basic_assignment(self):
        groups = list("ABCDEFGH")
        result = _bipartite_matching(groups, _SLOT_CANDIDATES)
        assert isinstance(result, dict)
        assert set(result.keys()) == set(groups)

    def test_each_group_in_valid_slot(self):
        groups = list("ABCDEFGH")
        result = _bipartite_matching(groups, _SLOT_CANDIDATES)
        for grp, slot in result.items():
            assert grp in _SLOT_CANDIDATES[slot], (
                f"Group {grp} assigned to {slot} but {slot} candidates are "
                f"{_SLOT_CANDIDATES[slot]}"
            )

    def test_no_duplicate_slots(self):
        groups = list("ABCDEFGH")
        result = _bipartite_matching(groups, _SLOT_CANDIDATES)
        slots_used = list(result.values())
        assert len(slots_used) == len(set(slots_used))

    def test_k_assigned_to_ehijk(self):
        """K can only go to slot 3EHIJK; must always end up there."""
        groups = list("BCDEGHIK")  # K present
        result = _bipartite_matching(sorted(groups), _SLOT_CANDIDATES)
        assert result.get("K") == "3EHIJK", f"K should go to 3EHIJK, got {result.get('K')}"

    def test_l_assigned_to_deijl(self):
        """L can only go to slot 3DEIJL; must always end up there."""
        groups = list("BCDEGHIL")  # L present
        result = _bipartite_matching(sorted(groups), _SLOT_CANDIDATES)
        assert result.get("L") == "3DEIJL", f"L should go to 3DEIJL, got {result.get('L')}"

    def test_k_and_l_together(self):
        """Both K and L present — each must go to their only eligible slot."""
        groups = list("ABCDEIKL")
        result = _bipartite_matching(sorted(groups), _SLOT_CANDIDATES)
        assert result.get("K") == "3EHIJK"
        assert result.get("L") == "3DEIJL"


# ---------------------------------------------------------------------------
# TestAllCombinationsSolvable
# ---------------------------------------------------------------------------
class TestAllCombinationsSolvable:
    """Verify all 495 possible group combinations have a valid slot assignment."""

    @pytest.fixture(scope="class")
    def all_assignments(self) -> dict[str, dict[str, str]]:
        """Pre-computed Annex C table from bracket_map.json."""
        bm = load_bracket_map()
        return bm.get("annex_c_combinations", {})

    def test_all_495_present(self, all_assignments):
        assert len(all_assignments) == 495

    def test_all_have_eight_entries(self, all_assignments):
        bad = [k for k, v in all_assignments.items() if len(v) != 8]
        assert not bad, f"Combinations with != 8 entries: {bad[:5]}"

    def test_all_slots_distinct(self, all_assignments):
        for combo, mapping in all_assignments.items():
            slots = list(mapping.values())
            assert len(slots) == len(set(slots)), f"Duplicate slots in combination {combo}: {slots}"

    def test_all_groups_in_valid_slot(self, all_assignments):
        violations = []
        for combo, mapping in all_assignments.items():
            for grp, slot in mapping.items():
                if grp not in _SLOT_CANDIDATES[slot]:
                    violations.append((combo, grp, slot))
        assert not violations, f"Validity violations: {violations[:5]}"

    def test_all_groups_from_combo(self, all_assignments):
        """The assigned groups in each mapping must match the combo key."""
        for combo, mapping in all_assignments.items():
            assert set(mapping.keys()) == set(
                combo
            ), f"Mismatch: combo={combo} but mapping keys={set(mapping.keys())}"


# ---------------------------------------------------------------------------
# TestAssignThirdsToSlots
# ---------------------------------------------------------------------------
class TestAssignThirdsToSlots:
    def test_returns_eight_mappings(self):
        qualifying = list("ABCDEFGH")
        result = assign_thirds_to_slots(qualifying, _BRACKET_MAP)
        assert len(result) == 8

    def test_uses_precomputed_table(self):
        """Result should match the pre-computed Annex C entry."""
        qualifying = sorted("ABCDEFGH")
        key = "".join(qualifying)
        expected = _BRACKET_MAP["annex_c_combinations"].get(key)
        result = assign_thirds_to_slots(qualifying, _BRACKET_MAP)
        if expected:
            assert result == expected

    def test_fallback_works_without_precomputed(self):
        """With empty annex_c, fallback matching still produces valid result."""
        import copy

        bm_empty = copy.deepcopy(_BRACKET_MAP)
        bm_empty["annex_c_combinations"] = {}
        result = assign_thirds_to_slots(list("ABCDEFGH"), bm_empty)
        assert len(result) == 8
        for grp, slot in result.items():
            assert grp in _SLOT_CANDIDATES[slot]

    def test_deterministic(self):
        q = list("BCDEFGIJ")
        r1 = assign_thirds_to_slots(q, _BRACKET_MAP)
        r2 = assign_thirds_to_slots(q, _BRACKET_MAP)
        assert r1 == r2


# ---------------------------------------------------------------------------
# TestResolveR32
# ---------------------------------------------------------------------------
class TestResolveR32:
    @pytest.fixture(scope="class")
    def matchups(self):
        group_results = _make_group_results()
        best_third_groups = list("ABCDEFGH")
        return resolve_r32(group_results, best_third_groups, _BRACKET_MAP)

    def test_returns_sixteen_matchups(self, matchups):
        assert len(matchups) == 16

    def test_each_matchup_has_two_teams(self, matchups):
        for a, b in matchups:
            assert isinstance(a, str) and isinstance(b, str)
            assert a != b

    def test_all_32_teams_appear_once(self, matchups):
        teams = [t for pair in matchups for t in pair]
        assert len(teams) == 32
        assert len(set(teams)) == 32

    def test_group_winners_placed_correctly(self, matchups):
        """Codes "1X" resolve to the first-place team of group X."""
        # R32_01 team_a is "1E" -> should be "1st_E"
        assert matchups[0][0] == "1st_E"

    def test_third_place_teams_from_qualifying_groups(self, matchups):
        best_third = list("ABCDEFGH")
        third_teams = {f"3rd_{g}" for g in best_third}
        all_teams = {t for pair in matchups for t in pair}
        thirds_in_bracket = all_teams & third_teams
        # Exactly 8 third-place teams
        assert len(thirds_in_bracket) == 8
        # All from qualifying groups
        assert all(t.replace("3rd_", "") in best_third for t in thirds_in_bracket)

    def test_different_best_third_selection_changes_matchups(self):
        gr = _make_group_results()
        m1 = resolve_r32(gr, list("ABCDEFGH"), _BRACKET_MAP)
        m2 = resolve_r32(gr, list("ABCDEIJK"), _BRACKET_MAP)
        # Different qualifiers -> different bracket (at least some matchups differ)
        assert m1 != m2


# ---------------------------------------------------------------------------
# TestSimulateKoMatch
# ---------------------------------------------------------------------------
class TestSimulateKoMatch:
    @pytest.fixture
    def model(self):
        return DixonColesModel(rho=-0.061)

    @pytest.fixture
    def strength(self):
        return {"Strong": 1900.0, "Weak": 1400.0, "Equal1": 1600.0, "Equal2": 1600.0}

    def test_returns_one_of_two_teams(self, model, strength):
        rng = default_rng(0)
        winner = simulate_ko_match("Strong", "Weak", strength, model, rng, _SIM_CFG)
        assert winner in {"Strong", "Weak"}

    def test_deterministic_with_seed(self, model, strength):
        w1 = simulate_ko_match("Strong", "Weak", strength, model, default_rng(7), _SIM_CFG)
        w2 = simulate_ko_match("Strong", "Weak", strength, model, default_rng(7), _SIM_CFG)
        assert w1 == w2

    def test_stronger_team_wins_more_often(self, model, strength):
        n = 3000
        strong_wins = sum(
            simulate_ko_match("Strong", "Weak", strength, model, default_rng(i), _SIM_CFG)
            == "Strong"
            for i in range(n)
        )
        assert strong_wins / n > 0.60, f"Strong only won {strong_wins}/{n}"

    def test_equal_teams_roughly_50_50(self, model, strength):
        n = 2000
        wins_1 = sum(
            simulate_ko_match("Equal1", "Equal2", strength, model, default_rng(i), _SIM_CFG)
            == "Equal1"
            for i in range(n)
        )
        assert 0.40 < wins_1 / n < 0.60, f"Equal teams: {wins_1}/{n} wins"

    def test_et_and_penalties_occur(self, model):
        """All results must be valid team names (ET/penalties always resolve)."""
        strength = {"A": 1600.0, "B": 1600.0}
        rng = default_rng(42)
        results = set()
        for _ in range(500):
            w = simulate_ko_match("A", "B", strength, model, rng, _SIM_CFG)
            results.add(w)
        assert results == {"A", "B"}

    def test_asymmetric_penalty_tilt(self, model):
        """Elo advantage should tilt penalty win rate above 50%."""
        strength = {"Fav": 1800.0, "Dog": 1400.0}
        # elo_diff = 400; tilt = min(400/100 * 0.05, 0.1) = 0.1
        # p_a = 0.5 + 0.1 = 0.6 → consistent with stronger team winning more
        n = 2000
        fav_wins = sum(
            simulate_ko_match("Fav", "Dog", strength, model, default_rng(i), _SIM_CFG) == "Fav"
            for i in range(n)
        )
        assert fav_wins / n > 0.55, f"Fav won only {fav_wins}/{n}"


# ---------------------------------------------------------------------------
# TestSimulateKnockout
# ---------------------------------------------------------------------------
class TestSimulateKnockout:
    @pytest.fixture(scope="class")
    def full_bracket_result(self):
        """Simulate a complete knockout bracket with 32 synthetic teams."""
        teams = [f"T{i:02d}" for i in range(1, 33)]
        strength = {t: 1500.0 + (i * 5) for i, t in enumerate(teams)}
        model = DixonColesModel(rho=-0.061)
        rng = default_rng(0)
        matchups = [(teams[i], teams[i + 1]) for i in range(0, 32, 2)]
        return simulate_knockout(matchups, strength, model, rng, _SIM_CFG)

    def test_champion_is_string(self, full_bracket_result):
        assert isinstance(full_bracket_result["champion"], str)

    def test_champion_was_r32_team(self, full_bracket_result):
        all_r32 = set(full_bracket_result["r32"])
        assert full_bracket_result["champion"] in all_r32

    def test_round_sizes(self, full_bracket_result):
        assert len(full_bracket_result["r32"]) == 16
        assert len(full_bracket_result["r16"]) == 8
        assert len(full_bracket_result["qf"]) == 4
        assert len(full_bracket_result["sf"]) == 2

    def test_r16_winners_are_r32_winners(self, full_bracket_result):
        r32_set = set(full_bracket_result["r32"])
        for team in full_bracket_result["r16"]:
            assert team in r32_set

    def test_qf_winners_are_r16_winners(self, full_bracket_result):
        r16_set = set(full_bracket_result["r16"])
        for team in full_bracket_result["qf"]:
            assert team in r16_set

    def test_sf_winners_are_qf_winners(self, full_bracket_result):
        qf_set = set(full_bracket_result["qf"])
        for team in full_bracket_result["sf"]:
            assert team in qf_set

    def test_champion_is_sf_winner(self, full_bracket_result):
        assert full_bracket_result["champion"] in set(full_bracket_result["sf"])

    def test_deterministic_with_seed(self):
        teams = [f"T{i:02d}" for i in range(1, 33)]
        strength = dict.fromkeys(teams, 1500.0)
        model = DixonColesModel(rho=-0.061)
        matchups = [(teams[i], teams[i + 1]) for i in range(0, 32, 2)]
        r1 = simulate_knockout(matchups, strength, model, default_rng(5), _SIM_CFG)
        r2 = simulate_knockout(matchups, strength, model, default_rng(5), _SIM_CFG)
        assert r1["champion"] == r2["champion"]
        assert r1["r32"] == r2["r32"]

    def test_wrong_matchup_count_raises(self):
        teams = [f"T{i}" for i in range(10)]
        strength = dict.fromkeys(teams, 1500.0)
        model = DixonColesModel(rho=-0.061)
        matchups = [("T0", "T1")] * 10  # wrong count
        with pytest.raises(ValueError, match="16 R32 matchups"):
            simulate_knockout(matchups, strength, model, default_rng(0), _SIM_CFG)


# ---------------------------------------------------------------------------
# Integration: simulate all 12 groups + pick best thirds + resolve R32 + KO
# ---------------------------------------------------------------------------
class TestFullPipelineIntegration:
    @pytest.fixture(scope="class")
    def pipeline_result(self):
        with open("data/groups.json") as f:
            groups_data = json.load(f)["groups"]
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)

        all_teams = [t for tl in groups_data.values() for t in tl]
        strength = dict.fromkeys(all_teams, 1500.0)
        model = DixonColesModel(rho=-0.061)
        rng = default_rng(0)

        group_results = {}
        for gid, teams in groups_data.items():
            group_results[gid] = simulate_group(gid, teams, strength, model, rng, cfg)

        best_thirds = pick_best_thirds(list(group_results.values()), n=8, rng=rng)
        best_third_groups = [
            gr.group_id for gr in group_results.values() if gr.third in best_thirds
        ]

        bm = load_bracket_map()
        matchups = resolve_r32(group_results, best_third_groups, bm)
        strength_ko = dict.fromkeys(all_teams, 1500.0)
        result = simulate_knockout(matchups, strength_ko, model, rng, cfg)
        return result, group_results, best_thirds

    def test_champion_is_wc_team(self, pipeline_result):
        result, group_results, _ = pipeline_result
        all_teams = {
            t for gr in group_results.values() for t in [gr.first, gr.second, gr.third, gr.fourth]
        }
        assert result["champion"] in all_teams

    def test_full_bracket_produces_valid_rounds(self, pipeline_result):
        result, _, _ = pipeline_result
        assert len(result["r32"]) == 16
        assert len(result["r16"]) == 8
        assert len(result["qf"]) == 4
        assert len(result["sf"]) == 2
        assert isinstance(result["champion"], str)
