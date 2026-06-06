"""Small-N determinism test: same seed → identical results."""

from __future__ import annotations

import pytest


class TestDeterminism:
    def test_same_seed_same_output(self) -> None:
        """Two runs with the same seed must produce bit-for-bit identical results."""
        pytest.skip("Implement in Phase 6 when montecarlo.run_simulation is ready.")

    def test_different_seed_different_output(self) -> None:
        pytest.skip("Implement in Phase 6.")
