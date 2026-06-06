"""Sensitivity analysis: perturb key parameters and measure probability shifts.

Produces a tornado-style plot showing which parameters most affect champion
and advancement probabilities.
"""

from __future__ import annotations

import logging
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _load_config() -> dict[str, Any]:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def run_sensitivity_analysis(
    strength_df: object,
    model: object,
    base_results: dict[str, Any],
    cfg: dict[str, Any],
) -> None:
    """Perturb each parameter in cfg["sensitivity"]["parameters"] ± delta.

    For each perturbation:
      - Re-run simulation at cfg["sensitivity"]["iterations"]
      - Record champion and top-6 advancement probability shifts

    Saves tornado plot to cfg["output"]["sensitivity_plot"].
    """
    raise NotImplementedError("run_sensitivity_analysis: implement in Phase 7.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit("sensitivity.py: full implementation in Phase 7.")
