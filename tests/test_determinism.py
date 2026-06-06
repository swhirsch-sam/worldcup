"""Small-N determinism test: same seed → identical results."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import yaml

from src.model.montecarlo import check_invariants, run_simulation
from src.model.poisson import DixonColesModel

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


def _cfg_and_model() -> tuple[dict, DixonColesModel]:
    with open("config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    model = DixonColesModel(rho=float(cfg["dixon_coles"]["rho"]))
    return cfg, model


def _run(seed: int, n: int = 200) -> dict:
    import pandas as pd

    cfg, model = _cfg_and_model()
    elo_df = pd.read_csv("data/raw/elo_ratings.csv")
    from src.model.strength import build_strength_table

    strength_df, _ = build_strength_table(elo_df, None, None)
    with tempfile.TemporaryDirectory() as tmp:
        cfg["output"]["simulation_summary"] = str(Path(tmp) / "sim.json")
        return run_simulation(strength_df, model, n_iterations=n, seed=seed, cfg=cfg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_output(self) -> None:
        """Two runs with the same seed must produce bit-for-bit identical results."""
        r1 = _run(seed=0)
        r2 = _run(seed=0)
        assert r1["counts"] == r2["counts"]

    def test_different_seed_different_output(self) -> None:
        """Different seeds must (with overwhelming probability) yield different champions."""
        r1 = _run(seed=1)
        r2 = _run(seed=999)
        # At 200 iterations champions won't be identical unless pathologically unlucky
        assert r1["counts"] != r2["counts"]

    def test_invariants_pass_on_small_run(self) -> None:
        """check_invariants must not raise for a 200-iteration run."""
        r = _run(seed=42)
        check_invariants(r, 200)

    def test_output_file_written(self) -> None:
        """run_simulation must write a valid JSON file to the configured path."""
        import pandas as pd

        cfg, model = _cfg_and_model()
        elo_df = pd.read_csv("data/raw/elo_ratings.csv")
        from src.model.strength import build_strength_table

        strength_df, _ = build_strength_table(elo_df, None, None)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sim.json"
            cfg["output"]["simulation_summary"] = str(out)
            run_simulation(strength_df, model, n_iterations=50, seed=7, cfg=cfg)
            assert out.exists()
            data = json.loads(out.read_text())
            assert "probabilities" in data
            assert "metadata" in data
            assert data["metadata"]["iterations"] == 50
