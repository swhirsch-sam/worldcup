"""Post-simulation invariant checks."""

from __future__ import annotations

import itertools
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.model.montecarlo import check_invariants, run_simulation
from src.model.poisson import DixonColesModel
from src.model.strength import build_strength_table

_N = 300  # small but enough to exercise all invariants


@pytest.fixture(scope="module")
def sim_results() -> dict:
    with open("config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    model = DixonColesModel(rho=float(cfg["dixon_coles"]["rho"]))
    elo_df = pd.read_csv("data/raw/elo_ratings.csv")
    strength_df, _ = build_strength_table(elo_df, None, None)
    with tempfile.TemporaryDirectory() as tmp:
        cfg["output"]["simulation_summary"] = str(Path(tmp) / "sim.json")
        return run_simulation(strength_df, model, n_iterations=_N, seed=42, cfg=cfg)


class TestSimulationInvariants:
    def test_champion_probs_sum_to_one(self, sim_results: dict) -> None:
        probs = sim_results["probabilities"]
        total = sum(p["champion"] for p in probs.values())
        assert abs(total - 1.0) < 1e-9

    def test_monotonic_advancement_probs(self, sim_results: dict) -> None:
        """P(reach R32) >= P(reach R16) >= ... >= P(win) for every team."""
        chain = ("r32", "r16", "qf", "sf", "final", "champion")
        for team, p in sim_results["probabilities"].items():
            for a, b in itertools.pairwise(chain):
                assert p[a] >= p[b], f"{team}: {a}={p[a]:.4f} < {b}={p[b]:.4f}"

    def test_group_finish_probs_sum_to_one(self, sim_results: dict) -> None:
        """Sum of group_first probabilities = 12/48 per team on average."""
        counts = sim_results["counts"]
        n = _N
        assert sum(c["group_first"] for c in counts.values()) == 12 * n
        assert sum(c["group_second"] for c in counts.values()) == 12 * n

    def test_exactly_32_r32_teams(self, sim_results: dict) -> None:
        counts = sim_results["counts"]
        assert sum(c["r32"] for c in counts.values()) == 32 * _N

    def test_no_duplicate_teams_in_bracket(self, sim_results: dict) -> None:
        """r32 count per team = group_first + group_second + third_qualify."""
        for team, c in sim_results["counts"].items():
            assert c["r32"] == c["group_first"] + c["group_second"] + c["third_qualify"], team

    def test_group_goals_conservation(self, sim_results: dict) -> None:
        """Verify check_invariants passes in full."""
        check_invariants(sim_results, _N)

    def test_round_sizes(self, sim_results: dict) -> None:
        """Winner counts for each KO round must equal round_size x n_iterations."""
        expected = {"r16": 16, "qf": 8, "sf": 4, "final": 2, "champion": 1}
        counts = sim_results["counts"]
        n = _N
        for stage, size in expected.items():
            total = sum(c[stage] for c in counts.values())
            assert total == size * n, f"{stage}: {total} != {size * n}"

    def test_probabilities_in_unit_interval(self, sim_results: dict) -> None:
        for team, p in sim_results["probabilities"].items():
            for stage, val in p.items():
                assert 0.0 <= val <= 1.0, f"{team}.{stage}={val}"
