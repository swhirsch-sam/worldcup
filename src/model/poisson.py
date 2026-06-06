"""Dixon-Coles adjusted Poisson model (default) and bivariate Poisson (alternative).

The active model is selected by config: model.bivariate_poisson.enabled.

Dixon-Coles features:
  - Low-score correction via rho dependence term (0-0, 1-0, 0-1, 1-1)
  - Time decay (xi) so recent matches weigh more during fitting

Both models expose the same interface: simulate_match(lambda_h, lambda_a, rng) -> (int, int)
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.random import Generator


class MatchSimulator(Protocol):
    """Interface every scoreline model must satisfy."""

    def simulate_match(
        self,
        lambda_h: float,
        lambda_a: float,
        rng: Generator,
    ) -> tuple[int, int]:
        """Draw a scoreline (home_goals, away_goals)."""
        ...

    def simulate_match_batch(
        self,
        lambda_h: np.ndarray,
        lambda_a: np.ndarray,
        rng: Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized batch draw; arrays must have the same shape."""
        ...


class DixonColesModel:
    """Dixon-Coles adjusted Poisson with low-score correction and time decay."""

    def __init__(self, rho: float) -> None:
        """
        Args:
            rho: Low-score correction parameter (fitted; typically small positive).
        """
        self.rho = rho

    def tau(self, x: int, y: int, lambda_h: float, lambda_a: float) -> float:
        """Dixon-Coles correction factor τ(x, y) for low scores."""
        if x == 0 and y == 0:
            return 1.0 - lambda_h * lambda_a * self.rho
        if x == 1 and y == 0:
            return 1.0 + lambda_a * self.rho
        if x == 0 and y == 1:
            return 1.0 + lambda_h * self.rho
        if x == 1 and y == 1:
            return 1.0 - self.rho
        return 1.0

    def simulate_match(
        self,
        lambda_h: float,
        lambda_a: float,
        rng: Generator,
    ) -> tuple[int, int]:
        """Draw a single scoreline using the Dixon-Coles correction.

        Uses rejection sampling for the low-score cells (0-0, 1-0, 0-1, 1-1).
        """
        raise NotImplementedError("DixonColesModel.simulate_match: implement in Phase 3.")

    def simulate_match_batch(
        self,
        lambda_h: np.ndarray,
        lambda_a: np.ndarray,
        rng: Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized batch simulation for Monte Carlo use."""
        raise NotImplementedError("DixonColesModel.simulate_match_batch: implement in Phase 3.")


class BivariatePoissonModel:
    """Bivariate Poisson model (alternative to Dixon-Coles)."""

    def __init__(self, lambda_corr: float) -> None:
        """
        Args:
            lambda_corr: Covariance parameter λ₃ in the bivariate Poisson.
        """
        self.lambda_corr = lambda_corr

    def simulate_match(
        self,
        lambda_h: float,
        lambda_a: float,
        rng: Generator,
    ) -> tuple[int, int]:
        raise NotImplementedError("BivariatePoissonModel.simulate_match: implement in Phase 3.")

    def simulate_match_batch(
        self,
        lambda_h: np.ndarray,
        lambda_a: np.ndarray,
        rng: Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "BivariatePoissonModel.simulate_match_batch: implement in Phase 3."
        )


def get_model(
    rho: float, *, use_bivariate: bool = False, lambda_corr: float = 0.0
) -> MatchSimulator:
    """Factory: return the configured model instance."""
    if use_bivariate:
        return BivariatePoissonModel(lambda_corr=lambda_corr)
    return DixonColesModel(rho=rho)
