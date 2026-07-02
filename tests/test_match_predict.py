"""Tests for closed-form match prediction (src/model/scoreline.py) and the
precompute assembly (src/model/match_predict.py).

Class inventory:
  TestScoreMatrix          - dixon_coles_matrix / outcome_probabilities math
  TestMatchesSimulator     - analytic distribution matches the DC simulator
  TestMatchLambdas         - match_lambdas strength → expected-goals mapping
  TestPredictAssembly      - predict_match / build_group_matches / build_payload
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.stats import poisson

from src.model.match_predict import build_group_matches, build_payload, predict_match
from src.model.poisson import DixonColesModel
from src.model.scoreline import (
    dixon_coles_matrix,
    match_lambdas,
    outcome_probabilities,
    predict_outcome,
)

PARAMS = {
    "intercept": 0.188,
    "slope": 0.00189,
    "rho": -0.061,
    "lambda_floor": 0.1,
    "lambda_cap": 6.0,
}

STRENGTH = {"A": 1850.0, "B": 1600.0, "C": 1500.0, "D": 1400.0}


def _model_grid(lh: float, la: float, rho: float) -> np.ndarray:
    """Reconstruct DixonColesModel's normalized grid using scipy (reference)."""
    grid = np.arange(11)
    joint = np.outer(poisson.pmf(grid, lh), poisson.pmf(grid, la))
    joint[0, 0] *= 1.0 - lh * la * rho
    joint[1, 0] *= 1.0 + la * rho
    joint[0, 1] *= 1.0 + lh * rho
    joint[1, 1] *= 1.0 - rho
    return joint / joint.sum()


# ===========================================================================
class TestScoreMatrix:
    def test_matrix_sums_to_one(self) -> None:
        m = dixon_coles_matrix(1.5, 1.1, -0.061)
        assert abs(float(m.sum()) - 1.0) < 1e-12
        assert (m >= 0).all()

    def test_outcome_probs_sum_to_one(self) -> None:
        stats = outcome_probabilities(dixon_coles_matrix(1.7, 0.9, -0.061))
        total = stats["p_home"] + stats["p_draw"] + stats["p_away"]
        assert abs(total - 1.0) < 1e-9

    def test_symmetric_when_equal(self) -> None:
        stats = outcome_probabilities(dixon_coles_matrix(1.3, 1.3, -0.061))
        assert abs(stats["p_home"] - stats["p_away"]) < 1e-9

    def test_stronger_side_favored(self) -> None:
        stats = outcome_probabilities(dixon_coles_matrix(2.1, 0.7, -0.061))
        assert stats["p_home"] > stats["p_away"]
        assert stats["p_home"] > stats["p_draw"]

    def test_expected_goals_close_to_lambda(self) -> None:
        stats = outcome_probabilities(dixon_coles_matrix(1.4, 1.0, -0.061))
        # Truncation at 10 and the tau correction shift things only slightly.
        assert abs(stats["exp_home"] - 1.4) < 0.06
        assert abs(stats["exp_away"] - 1.0) < 0.06

    def test_top_scores_sorted(self) -> None:
        stats = outcome_probabilities(dixon_coles_matrix(1.6, 1.0, -0.061))
        probs = [s["prob"] for s in stats["top_scores"]]
        assert probs == sorted(probs, reverse=True)
        assert len(stats["top_scores"]) == 3


# ===========================================================================
class TestMatchesSimulator:
    """The analytic distribution must reproduce what the DC simulator draws."""

    def test_matrix_matches_model_grid(self) -> None:
        for lh, la in [(1.7, 0.9), (1.0, 1.0), (2.5, 0.6)]:
            ours = dixon_coles_matrix(lh, la, -0.061)
            ref = _model_grid(lh, la, -0.061)
            assert np.allclose(ours, ref, atol=1e-12)

    def test_analytic_matches_simulation(self) -> None:
        rho = -0.061
        lam_h, lam_a = 1.7, 0.9
        n = 80_000
        model = DixonColesModel(rho=rho)
        h, a = model.simulate_match_batch(np.full(n, lam_h), np.full(n, lam_a), default_rng(123))
        emp_home = float((h > a).mean())
        emp_draw = float((h == a).mean())
        emp_away = float((a > h).mean())

        stats = outcome_probabilities(dixon_coles_matrix(lam_h, lam_a, rho))
        assert abs(stats["p_home"] - emp_home) < 0.02
        assert abs(stats["p_draw"] - emp_draw) < 0.02
        assert abs(stats["p_away"] - emp_away) < 0.02
        assert abs(stats["exp_home"] - float(h.mean())) < 0.05
        assert abs(stats["exp_away"] - float(a.mean())) < 0.05


# ===========================================================================
class TestMatchLambdas:
    def test_equal_strength_equal_lambdas(self) -> None:
        lh, la = match_lambdas(1500.0, 1500.0, intercept=0.188, slope=0.00189)
        assert lh == la
        assert abs(lh - np.exp(0.188)) < 1e-9

    def test_stronger_team_scores_more(self) -> None:
        lh, la = match_lambdas(1800.0, 1500.0, intercept=0.188, slope=0.00189)
        assert lh > la

    def test_clip_bounds(self) -> None:
        lh, la = match_lambdas(
            5000.0, 1000.0, intercept=0.188, slope=0.01, lambda_floor=0.1, lambda_cap=6.0
        )
        assert lh == 6.0
        assert la == 0.1

    def test_predict_outcome_matches_pieces(self) -> None:
        direct = predict_outcome(1800.0, 1500.0, **PARAMS)
        lh, la = match_lambdas(
            1800.0,
            1500.0,
            intercept=PARAMS["intercept"],
            slope=PARAMS["slope"],
            lambda_floor=PARAMS["lambda_floor"],
            lambda_cap=PARAMS["lambda_cap"],
        )
        manual = outcome_probabilities(dixon_coles_matrix(lh, la, PARAMS["rho"]))
        assert direct["p_home"] == manual["p_home"]


# ===========================================================================
class TestPredictAssembly:
    def test_predict_match_structure(self) -> None:
        m = predict_match("A", "D", STRENGTH, PARAMS)
        for key in ("p_home", "p_draw", "p_away", "exp_home", "exp_away", "top_scores"):
            assert key in m
        assert m["favorite"] == "A"
        assert abs(m["p_home"] + m["p_draw"] + m["p_away"] - 1.0) < 1e-9

    def test_favorite_is_stronger_team(self) -> None:
        m = predict_match("D", "A", STRENGTH, PARAMS)  # weaker listed "home"
        assert m["favorite"] == "A"
        assert m["favorite_prob"] == m["p_away"]

    def test_build_group_matches_count(self) -> None:
        groups = {"X": ["A", "B", "C", "D"]}
        matches = build_group_matches(groups, STRENGTH, PARAMS)
        assert len(matches) == 6  # C(4,2)
        assert all(m["group"] == "X" for m in matches)

    def test_build_payload_shape(self) -> None:
        groups = {"X": ["A", "B", "C", "D"]}
        strength_df = pd.DataFrame({"team": list(STRENGTH), "strength": list(STRENGTH.values())})
        cfg = {
            "goals_model": {"intercept": 0.188, "slope": 0.00189},
            "dixon_coles": {"rho": -0.061},
        }
        payload = build_payload(strength_df, groups, cfg)
        assert set(payload) == {
            "metadata",
            "model_params",
            "team_strength",
            "groups",
            "group_matches",
        }
        assert len(payload["group_matches"]) == 6
        assert len(payload["team_strength"]) == 4
        assert payload["model_params"]["rho"] == -0.061
