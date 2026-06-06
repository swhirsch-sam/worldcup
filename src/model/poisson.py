"""Dixon-Coles adjusted Poisson model (default) and bivariate Poisson (alternative).

The active model is selected by config: model.bivariate_poisson.enabled.

Dixon-Coles features:
  - Low-score correction via rho dependence term (0-0, 1-0, 0-1, 1-1)
  - Time decay (xi) so recent matches weigh more during fitting

Both models expose the same interface: simulate_match(lambda_h, lambda_a, rng) -> (int, int)
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from scipy.stats import poisson as scipy_poisson


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
        lambda_h: NDArray[Any],
        lambda_a: NDArray[Any],
        rng: Generator,
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """Vectorized batch draw; arrays must have the same shape."""
        ...


# Precomputed grid for PMF evaluation: 0..10
_GRID = np.arange(11, dtype=np.int32)
_LOG_FACTORIAL = np.array([0.0, *list(np.cumsum(np.log(np.arange(1, 11))))])


class DixonColesModel:
    """Dixon-Coles adjusted Poisson with low-score correction and time decay."""

    def __init__(self, rho: float) -> None:
        """
        Args:
            rho: Low-score correction parameter (fitted; negative for football).
        """
        self.rho = rho

    def tau(self, x: int, y: int, lambda_h: float, lambda_a: float) -> float:
        """Dixon-Coles correction factor tau(x, y) for low scores."""
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
        """Draw a single scoreline using the Dixon-Coles PMF on a 0..10 grid.

        Computes joint PMF (11x11 = 121 cells), applies tau correction,
        normalizes, then samples via multinomial draw.
        """
        # PMF for each team: shape (11,)
        p_h = scipy_poisson.pmf(_GRID, lambda_h)
        p_a = scipy_poisson.pmf(_GRID, lambda_a)

        # Joint PMF: shape (11, 11)
        joint = np.outer(p_h, p_a)

        # Apply tau correction to low-score cells
        joint[0, 0] *= 1.0 - lambda_h * lambda_a * self.rho
        joint[1, 0] *= 1.0 + lambda_a * self.rho
        joint[0, 1] *= 1.0 + lambda_h * self.rho
        joint[1, 1] *= 1.0 - self.rho

        # Normalize (truncation at 10 causes tiny error)
        flat = joint.ravel()
        flat = flat / flat.sum()

        # Draw from flattened categorical
        cumsum = flat.cumsum()
        u = rng.random()
        idx = int(np.searchsorted(cumsum, u))
        idx = min(idx, 120)  # clamp to valid range
        home_goals = idx // 11
        away_goals = idx % 11
        return int(home_goals), int(away_goals)

    def simulate_match_batch(
        self,
        lambda_h: NDArray[Any],
        lambda_a: NDArray[Any],
        rng: Generator,
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """Vectorized batch simulation for Monte Carlo use.

        Args:
            lambda_h: Array of shape (N,) with home expected goals.
            lambda_a: Array of shape (N,) with away expected goals.
            rng: NumPy random Generator.

        Returns:
            (home_goals, away_goals) arrays of shape (N,).
        """
        n = len(lambda_h)

        # log PMF per team using broadcasting: (N, 1) op (1, 11) -> (N, 11)
        lh = lambda_h[:, None]  # (N, 1)
        la = lambda_a[:, None]  # (N, 1)
        k = _GRID[None, :]  # (1, 11)

        # log PMF per team: (N, 11)
        log_pmf_h = -lh + k * np.log(np.maximum(lh, 1e-300)) - _LOG_FACTORIAL[_GRID][None, :]
        log_pmf_a = -la + k * np.log(np.maximum(la, 1e-300)) - _LOG_FACTORIAL[_GRID][None, :]

        # Joint log PMF: (N, 11, 11)
        log_joint = log_pmf_h[:, :, None] + log_pmf_a[:, None, :]

        # Exponentiate to get probability
        joint = np.exp(log_joint)  # (N, 11, 11)

        # Apply tau correction (vectorized for low-score cells)
        rho = self.rho
        lh_flat = lambda_h  # (N,)
        la_flat = lambda_a  # (N,)

        joint[:, 0, 0] *= 1.0 - lh_flat * la_flat * rho
        joint[:, 1, 0] *= 1.0 + la_flat * rho
        joint[:, 0, 1] *= 1.0 + lh_flat * rho
        joint[:, 1, 1] *= 1.0 - rho

        # Flatten to (N, 121) and normalize each row
        flat = joint.reshape(n, 121)
        flat = flat / flat.sum(axis=1, keepdims=True)

        # Vectorized categorical sampling
        cumsum = flat.cumsum(axis=1)  # (N, 121)
        u = rng.random(n)  # (N,)
        indices = (cumsum < u[:, None]).sum(axis=1)  # (N,)
        indices = np.minimum(indices, 120)  # clamp

        home_goals = (indices // 11).astype(np.int32)
        away_goals = (indices % 11).astype(np.int32)
        return home_goals, away_goals


class BivariatePoissonModel:
    """Bivariate Poisson model (alternative to Dixon-Coles)."""

    def __init__(self, lambda_corr: float) -> None:
        """
        Args:
            lambda_corr: Covariance parameter lambda_3 in the bivariate Poisson.
        """
        self.lambda_corr = lambda_corr

    def simulate_match(
        self,
        lambda_h: float,
        lambda_a: float,
        rng: Generator,
    ) -> tuple[int, int]:
        """Draw scoreline from bivariate Poisson.

        Decompose:
          lam3 = min(lambda_corr, lambda_h, lambda_a)
          lam1 = lambda_h - lam3
          lam2 = lambda_a - lam3
        Draw x1 ~ Poisson(lam1), x2 ~ Poisson(lam2), x3 ~ Poisson(lam3)
        Return (x1 + x3, x2 + x3)
        """
        lam3 = min(self.lambda_corr, lambda_h, lambda_a)
        lam1 = lambda_h - lam3
        lam2 = lambda_a - lam3

        x1 = rng.poisson(lam1)
        x2 = rng.poisson(lam2)
        x3 = rng.poisson(lam3)
        return int(x1 + x3), int(x2 + x3)

    def simulate_match_batch(
        self,
        lambda_h: NDArray[Any],
        lambda_a: NDArray[Any],
        rng: Generator,
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """Fully vectorized bivariate Poisson batch draw."""
        lam3 = np.minimum(self.lambda_corr, np.minimum(lambda_h, lambda_a))
        lam1 = lambda_h - lam3
        lam2 = lambda_a - lam3

        x1 = rng.poisson(lam1)
        x2 = rng.poisson(lam2)
        x3 = rng.poisson(lam3)
        return (x1 + x3).astype(np.int32), (x2 + x3).astype(np.int32)


def get_model(
    rho: float, *, use_bivariate: bool = False, lambda_corr: float = 0.0
) -> MatchSimulator:
    """Factory: return the configured model instance."""
    if use_bivariate:
        return BivariatePoissonModel(lambda_corr=lambda_corr)
    return DixonColesModel(rho=rho)
