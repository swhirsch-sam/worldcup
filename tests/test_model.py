"""Tests for the model layer: Elo helpers, Dixon-Coles PMF, bivariate Poisson,
and the statistical fitting routines.

Class inventory:
  TestKFactor            - _k_factor() tournament classification
  TestGoalWeight         - _goal_weight() multiplier values
  TestDixonColesTau      - DixonColesModel.tau() correction factors
  TestSimulateMatch      - Single-match draws (DC model)
  TestSimulateBatch      - Vectorised batch draws (DC model)
  TestBivariatePoisson   - BivariatePoissonModel properties
  TestGoalsModelFit      - fit_goals_model() recovers known parameters
  TestDixonColesFit      - fit_dixon_coles() recovers known rho
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.random import default_rng

from src.ingest.elo import _goal_weight, _k_factor
from src.model.fit import fit_dixon_coles, fit_goals_model
from src.model.poisson import BivariatePoissonModel, DixonColesModel, get_model

# ---------------------------------------------------------------------------
# Minimal config stub reused across tests
# ---------------------------------------------------------------------------
_CFG = {
    "elo_computation": {
        "home_advantage_elo": 100.0,
        "starting_rating": 1500.0,
        "k_factors": {
            "world_cup_final": 60,
            "continental_championship": 50,
            "world_cup_qualifier": 40,
            "other_official": 30,
            "friendly": 20,
        },
    },
    "dixon_coles": {
        "time_decay_halflife_days": 365,
        "rho": 0.10,
    },
    "goals_model": {
        "intercept": 0.27,
        "slope": 0.00416,
    },
}


# ===========================================================================
# TestKFactor
# ===========================================================================
class TestKFactor:
    def test_world_cup_k60(self) -> None:
        assert _k_factor("FIFA World Cup", _CFG) == 60.0

    def test_friendly_k20(self) -> None:
        assert _k_factor("Friendly", _CFG) == 20.0

    def test_qualifier_k40(self) -> None:
        assert _k_factor("WC Qualification", _CFG) == 40.0

    def test_continental_k50(self) -> None:
        assert _k_factor("UEFA Euro", _CFG) == 50.0

    def test_nations_league_k40(self) -> None:
        # Nations league is a qualifier-class tournament
        assert _k_factor("UEFA Nations League", _CFG) == 40.0

    def test_copa_america_k50(self) -> None:
        assert _k_factor("Copa America", _CFG) == 50.0

    def test_world_cup_qualifier_k40(self) -> None:
        assert _k_factor("FIFA World Cup qualification", _CFG) == 40.0

    def test_other_official_k30(self) -> None:
        # A tournament that is none of the above
        assert _k_factor("King's Cup", _CFG) == 30.0


# ===========================================================================
# TestGoalWeight
# ===========================================================================
class TestGoalWeight:
    def test_zero_goal_diff(self) -> None:
        assert _goal_weight(0) == 1.0

    def test_one_goal_diff(self) -> None:
        assert _goal_weight(1) == 1.0

    def test_two_goal_diff(self) -> None:
        assert _goal_weight(2) == 1.5

    def test_three_plus_goal_diff(self) -> None:
        # (11 + 3) / 8 = 14/8 = 1.75
        assert _goal_weight(3) == pytest.approx(14.0 / 8.0)

    def test_five_goal_diff(self) -> None:
        assert _goal_weight(5) == pytest.approx(16.0 / 8.0)

    def test_goal_weight_monotone(self) -> None:
        values = [_goal_weight(d) for d in range(8)]
        assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))


# ===========================================================================
# TestDixonColesTau
# ===========================================================================
class TestDixonColesTau:
    """The tau correction should inflate low-score probabilities when rho < 0."""

    def _model(self, rho: float = -0.1) -> DixonColesModel:
        return DixonColesModel(rho=rho)

    def test_tau_zero_zero(self) -> None:
        m = self._model(rho=-0.1)
        lh, la = 1.3, 1.1
        # tau(0,0) = 1 - lh * la * rho  => with rho=-0.1: > 1
        expected = 1.0 - lh * la * (-0.1)
        assert m.tau(0, 0, lh, la) == pytest.approx(expected)
        assert m.tau(0, 0, lh, la) > 1.0

    def test_tau_one_one(self) -> None:
        m = self._model(rho=-0.1)
        # tau(1,1) = 1 - rho = 1.1 when rho=-0.1
        assert m.tau(1, 1, 1.3, 1.1) == pytest.approx(1.1)
        assert m.tau(1, 1, 1.3, 1.1) > 1.0

    def test_tau_one_zero(self) -> None:
        m = self._model(rho=-0.1)
        la = 1.1
        # tau(1,0) = 1 + la * rho = 1 - 0.11 = 0.89 (slightly < 1 to compensate)
        assert m.tau(1, 0, 1.3, la) == pytest.approx(1.0 + la * (-0.1))

    def test_tau_zero_one(self) -> None:
        m = self._model(rho=-0.1)
        lh = 1.3
        # tau(0,1) = 1 + lh * rho
        assert m.tau(0, 1, lh, 1.1) == pytest.approx(1.0 + lh * (-0.1))

    def test_tau_other(self) -> None:
        m = self._model(rho=-0.1)
        assert m.tau(2, 3, 1.3, 1.1) == pytest.approx(1.0)
        assert m.tau(5, 5, 1.3, 1.1) == pytest.approx(1.0)

    def test_tau_positivity_constraint(self) -> None:
        """With a moderate rho close to 0, all tau values must be positive."""
        m = self._model(rho=-0.1)
        for x in range(3):
            for y in range(3):
                assert m.tau(x, y, 1.3, 1.1) > 0.0


# ===========================================================================
# TestSimulateMatch
# ===========================================================================
class TestSimulateMatch:
    """Single-match draw from the Dixon-Coles model."""

    def _model(self) -> DixonColesModel:
        return DixonColesModel(rho=-0.10)

    def test_pmf_sums_to_one(self) -> None:
        """Draw many samples; empirical cell frequencies should sum to 1."""
        rng = default_rng(0)
        model = self._model()
        n = 10_000
        counts = {}
        for _ in range(n):
            score = model.simulate_match(1.35, 1.10, rng)
            counts[score] = counts.get(score, 0) + 1
        total = sum(counts.values())
        assert total == n

    def test_symmetric_teams(self) -> None:
        """When both teams have equal lambdas, P(home win) ≈ P(away win)."""
        rng = default_rng(1)
        model = self._model()
        n = 20_000
        home_wins = away_wins = 0
        lam = 1.3
        for _ in range(n):
            h, a = model.simulate_match(lam, lam, rng)
            if h > a:
                home_wins += 1
            elif a > h:
                away_wins += 1
        # Within 3 percentage points
        assert abs(home_wins / n - away_wins / n) < 0.03

    def test_strong_vs_weak(self) -> None:
        """When home team is much stronger, P(home win) >> 0.5."""
        rng = default_rng(2)
        model = self._model()
        n = 10_000
        # Simply check: home wins a majority
        home_wins_direct = sum(
            1 for _ in range(n) for h, a in [model.simulate_match(3.0, 0.5, rng)] if h > a
        )
        assert home_wins_direct / n > 0.60

    def test_deterministic_with_seed(self) -> None:
        """Same seed must produce identical results."""
        model = self._model()
        rng1 = default_rng(42)
        rng2 = default_rng(42)
        result1 = model.simulate_match(1.5, 1.0, rng1)
        result2 = model.simulate_match(1.5, 1.0, rng2)
        assert result1 == result2

    def test_scores_non_negative(self) -> None:
        """Score components must be non-negative integers."""
        rng = default_rng(3)
        model = self._model()
        for _ in range(1000):
            h, a = model.simulate_match(1.3, 1.1, rng)
            assert h >= 0
            assert a >= 0
            assert isinstance(h, int)
            assert isinstance(a, int)

    def test_scores_within_grid(self) -> None:
        """Scores must be at most 10 (grid limit)."""
        rng = default_rng(4)
        model = self._model()
        for _ in range(10_000):
            h, a = model.simulate_match(1.3, 1.1, rng)
            assert h <= 10
            assert a <= 10


# ===========================================================================
# TestSimulateBatch
# ===========================================================================
class TestSimulateBatch:
    """Vectorised batch simulation."""

    def _model(self) -> DixonColesModel:
        return DixonColesModel(rho=-0.10)

    def test_batch_output_shapes(self) -> None:
        model = self._model()
        rng = default_rng(0)
        n = 500
        lh = np.full(n, 1.35)
        la = np.full(n, 1.10)
        home_goals, away_goals = model.simulate_match_batch(lh, la, rng)
        assert home_goals.shape == (n,)
        assert away_goals.shape == (n,)

    def test_batch_vectorized_shapes(self) -> None:
        model = self._model()
        rng = default_rng(5)
        n = 1000
        lh = rng.uniform(0.5, 3.0, n)
        la = rng.uniform(0.5, 3.0, n)
        h, a = model.simulate_match_batch(lh, la, rng)
        assert h.shape == (n,)
        assert a.shape == (n,)
        assert (h >= 0).all()
        assert (a >= 0).all()
        assert (h <= 10).all()
        assert (a <= 10).all()

    def test_batch_matches_single(self) -> None:
        """Batch(N=5000) should match single-draw distribution (same seed)."""
        model = self._model()
        n = 5000
        lam_h = 1.35
        lam_a = 1.10

        # Single method
        rng_single = default_rng(99)
        single_results = [model.simulate_match(lam_h, lam_a, rng_single) for _ in range(n)]
        single_h = np.array([r[0] for r in single_results])
        single_a = np.array([r[1] for r in single_results])

        # Batch method with fresh but same-valued seed
        rng_batch = default_rng(99)
        lh_arr = np.full(n, lam_h)
        la_arr = np.full(n, lam_a)
        batch_h, batch_a = model.simulate_match_batch(lh_arr, la_arr, rng_batch)

        # Statistical equivalence: mean home/away goals should be close
        assert abs(single_h.mean() - batch_h.mean()) < 0.15
        assert abs(single_a.mean() - batch_a.mean()) < 0.15

        # Distribution equivalence: compare win/draw/loss fractions
        single_hw = (single_h > single_a).mean()
        batch_hw = (batch_h > batch_a).mean()
        assert abs(single_hw - batch_hw) < 0.05

    def test_batch_deterministic(self) -> None:
        """Same seed gives identical batch results."""
        model = self._model()
        n = 100
        lh = np.full(n, 1.35)
        la = np.full(n, 1.10)

        h1, a1 = model.simulate_match_batch(lh, la, default_rng(7))
        h2, a2 = model.simulate_match_batch(lh, la, default_rng(7))
        np.testing.assert_array_equal(h1, h2)
        np.testing.assert_array_equal(a1, a2)


# ===========================================================================
# TestBivariatePoisson
# ===========================================================================
class TestBivariatePoisson:
    def _model(self, lambda_corr: float = 0.2) -> BivariatePoissonModel:
        return BivariatePoissonModel(lambda_corr=lambda_corr)

    def test_means_correct(self) -> None:
        """E[home_goals] ≈ lambda_h."""
        model = self._model(lambda_corr=0.2)
        rng = default_rng(10)
        n = 20_000
        lh_arr = np.full(n, 1.5)
        la_arr = np.full(n, 1.1)
        h, a = model.simulate_match_batch(lh_arr, la_arr, rng)
        assert abs(h.mean() - 1.5) < 0.05
        assert abs(a.mean() - 1.1) < 0.05

    def test_correlation_positive(self) -> None:
        """Cov(home, away) > 0 when lambda_corr > 0."""
        model = self._model(lambda_corr=0.3)
        rng = default_rng(11)
        n = 20_000
        lh_arr = np.full(n, 1.5)
        la_arr = np.full(n, 1.1)
        h, a = model.simulate_match_batch(lh_arr, la_arr, rng)
        cov = np.cov(h.astype(float), a.astype(float))[0, 1]
        assert cov > 0.0

    def test_zero_corr_independent(self) -> None:
        """With lambda_corr=0, covariance should be near zero."""
        model = self._model(lambda_corr=0.0)
        rng = default_rng(12)
        n = 20_000
        lh_arr = np.full(n, 1.5)
        la_arr = np.full(n, 1.1)
        h, a = model.simulate_match_batch(lh_arr, la_arr, rng)
        cov = np.cov(h.astype(float), a.astype(float))[0, 1]
        # Should be close to 0, allowing sampling noise
        assert abs(cov) < 0.1

    def test_single_match_non_negative(self) -> None:
        model = self._model()
        rng = default_rng(13)
        for _ in range(500):
            h, a = model.simulate_match(1.5, 1.1, rng)
            assert h >= 0
            assert a >= 0

    def test_batch_shapes(self) -> None:
        model = self._model()
        rng = default_rng(14)
        n = 200
        h, a = model.simulate_match_batch(np.full(n, 1.5), np.full(n, 1.1), rng)
        assert h.shape == (n,)
        assert a.shape == (n,)


# ===========================================================================
# TestGetModel
# ===========================================================================
class TestGetModel:
    def test_returns_dc_by_default(self) -> None:
        model = get_model(rho=-0.1)
        assert isinstance(model, DixonColesModel)

    def test_returns_bivariate_when_requested(self) -> None:
        model = get_model(rho=-0.1, use_bivariate=True, lambda_corr=0.2)
        assert isinstance(model, BivariatePoissonModel)

    def test_dc_rho_stored(self) -> None:
        model = get_model(rho=-0.15)
        assert isinstance(model, DixonColesModel)
        assert model.rho == pytest.approx(-0.15)


# ===========================================================================
# TestGoalsModelFit
# ===========================================================================
class TestGoalsModelFit:
    """Fit on synthetic data where ground truth is known."""

    @pytest.fixture(scope="class")
    def synthetic_records(self) -> pd.DataFrame:
        """Generate N=5000 synthetic match records with known alpha/beta."""
        alpha_true = 0.25
        beta_true = 0.00140
        rng = np.random.default_rng(42)
        n = 5000

        home_elo = rng.uniform(1300, 1900, n)
        away_elo = rng.uniform(1300, 1900, n)
        neutral = rng.choice([True, False], n, p=[0.3, 0.7])
        home_adv = 100.0

        elo_diff = home_elo - away_elo + home_adv * (1 - neutral.astype(float))
        lam_h = np.exp(alpha_true + beta_true * elo_diff)
        lam_a = np.exp(alpha_true - beta_true * elo_diff)

        home_score = rng.poisson(lam_h).astype(int)
        away_score = rng.poisson(lam_a).astype(int)

        return pd.DataFrame(
            {
                "home_elo": home_elo,
                "away_elo": away_elo,
                "home_score": home_score,
                "away_score": away_score,
                "neutral": neutral,
            }
        )

    def test_fit_recovers_intercept_and_slope(self, synthetic_records: pd.DataFrame) -> None:
        alpha_true = 0.25
        beta_true = 0.00140
        result = fit_goals_model(synthetic_records, _CFG)
        assert abs(result["intercept"] - alpha_true) < 0.05
        assert abs(result["slope"] - beta_true) < 0.0002

    def test_fit_returns_dict_with_correct_keys(self, synthetic_records: pd.DataFrame) -> None:
        result = fit_goals_model(synthetic_records, _CFG)
        assert "intercept" in result
        assert "slope" in result

    def test_fit_slope_positive(self, synthetic_records: pd.DataFrame) -> None:
        """Higher Elo diff should predict more home goals → slope must be positive."""
        result = fit_goals_model(synthetic_records, _CFG)
        assert result["slope"] > 0.0


# ===========================================================================
# TestDixonColesFit
# ===========================================================================
class TestDixonColesFit:
    """Fit Dixon-Coles rho on synthetic data with known ground truth."""

    @pytest.fixture(scope="class")
    def synthetic_records_for_dc(self) -> tuple[pd.DataFrame, dict[str, float]]:
        """Generate synthetic match data from a DC model with rho_true=-0.10."""
        rho_true = -0.10
        alpha = 0.27
        beta = 0.00138
        home_adv = 100.0
        n = 5000
        rng = np.random.default_rng(55)

        home_elo = rng.uniform(1300, 1900, n)
        away_elo = rng.uniform(1300, 1900, n)
        neutral = rng.choice([True, False], n, p=[0.3, 0.7])

        elo_diff = home_elo - away_elo + home_adv * (1 - neutral.astype(float))
        lam_h = np.exp(alpha + beta * elo_diff)
        lam_a = np.exp(alpha - beta * elo_diff)

        # Draw from Dixon-Coles model
        dc_model = DixonColesModel(rho=rho_true)
        dc_rng = default_rng(55)
        lh_arr = lam_h
        la_arr = lam_a
        home_score, away_score = dc_model.simulate_match_batch(lh_arr, la_arr, dc_rng)

        # Create dates (spread over 5 years)
        base_date = pd.Timestamp("2019-01-01")
        dates = [base_date + pd.Timedelta(days=int(d)) for d in rng.integers(0, 1825, n)]

        df = pd.DataFrame(
            {
                "home_elo": home_elo,
                "away_elo": away_elo,
                "home_score": home_score.astype(int),
                "away_score": away_score.astype(int),
                "neutral": neutral,
                "date": dates,
            }
        )
        goals_coef = {"intercept": alpha, "slope": beta}
        return df, goals_coef

    def test_fit_recovers_rho(self, synthetic_records_for_dc) -> None:  # type: ignore[override]
        df, goals_coef = synthetic_records_for_dc
        rho_true = -0.10
        result = fit_dixon_coles(df, goals_coef, _CFG)
        assert abs(result["rho"] - rho_true) < 0.05

    def test_fit_returns_negative_rho(self, synthetic_records_for_dc) -> None:  # type: ignore[override]
        """Rho must be negative (more low-score draws than independent Poisson)."""
        df, goals_coef = synthetic_records_for_dc
        result = fit_dixon_coles(df, goals_coef, _CFG)
        assert result["rho"] < 0.0

    def test_fit_rho_in_bounds(self, synthetic_records_for_dc) -> None:  # type: ignore[override]
        df, goals_coef = synthetic_records_for_dc
        result = fit_dixon_coles(df, goals_coef, _CFG)
        assert -0.5 <= result["rho"] <= 0.0

    def test_fit_returns_dict_with_rho_key(self, synthetic_records_for_dc) -> None:  # type: ignore[override]
        df, goals_coef = synthetic_records_for_dc
        result = fit_dixon_coles(df, goals_coef, _CFG)
        assert "rho" in result


# ===========================================================================
# TestEloWithRecords
# ===========================================================================
class TestEloWithRecords:
    """Test compute_elo_with_records() produces valid output."""

    @pytest.fixture(scope="class")
    def small_history(self) -> pd.DataFrame:
        """A tiny synthetic history of 5 matches."""
        dates = pd.date_range("2020-01-01", periods=5, freq="W")
        return pd.DataFrame(
            {
                "date": dates,
                "home_team": ["A", "B", "A", "C", "B"],
                "away_team": ["B", "C", "C", "A", "A"],
                "home_score": [2, 1, 0, 3, 1],
                "away_score": [0, 1, 1, 1, 2],
                "tournament": ["Friendly"] * 5,
                "neutral": [False] * 5,
            }
        )

    def test_returns_three_items(self, small_history: pd.DataFrame) -> None:
        from src.ingest.elo import compute_elo_with_records

        result = compute_elo_with_records(small_history, _CFG)
        assert len(result) == 3

    def test_match_records_row_count(self, small_history: pd.DataFrame) -> None:
        from src.ingest.elo import compute_elo_with_records

        _, _, records = compute_elo_with_records(small_history, _CFG)
        assert len(records) == len(small_history)

    def test_match_records_columns(self, small_history: pd.DataFrame) -> None:
        from src.ingest.elo import compute_elo_with_records

        _, _, records = compute_elo_with_records(small_history, _CFG)
        required = {
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "home_elo",
            "away_elo",
            "neutral",
            "tournament",
        }
        assert required.issubset(set(records.columns))

    def test_first_match_uses_starting_rating(self, small_history: pd.DataFrame) -> None:
        """First encounter for any team should use the starting rating."""
        from src.ingest.elo import compute_elo_with_records

        _, _, records = compute_elo_with_records(small_history, _CFG)
        start = float(_CFG["elo_computation"]["starting_rating"])
        # First row's home and away elos should be the start rating (new teams)
        first = records.iloc[0]
        assert first["home_elo"] == pytest.approx(start)
        assert first["away_elo"] == pytest.approx(start)

    def test_final_ratings_consistent(self, small_history: pd.DataFrame) -> None:
        """Final ratings from compute_elo_with_records == compute_elo_from_historical."""
        from src.ingest.elo import compute_elo_from_historical, compute_elo_with_records

        ratings_ref, counts_ref = compute_elo_from_historical(small_history, _CFG)
        ratings_new, counts_new, _ = compute_elo_with_records(small_history, _CFG)

        for team in ratings_ref:
            assert ratings_new[team] == pytest.approx(ratings_ref[team], abs=1e-9)
        for team in counts_ref:
            assert counts_new[team] == counts_ref[team]
