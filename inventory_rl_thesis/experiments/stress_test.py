# experiments/stress_test.py
# STATUS: Stress test — elevated disruption regime experiments
# DEPENDS ON: experiments/phase2_extend.py
# TEST: python -c "from experiments.stress_test import run_stress_test"

"""
Stress Test: Same as Phase 2 but with the "stress_test" disruption regime
(p_start=0.15, mean_duration=10, mu1=0.3, mu2=0.4).

This pushes the supply chain to its limits to highlight the value
of disruption awareness and LLM risk signals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.phase2_extend import run_phase2


def run_stress_test(
    total_timesteps: int | None = None,
    eval_only: bool = False,
    quick_test: bool = False,
) -> dict[str, Any]:
    """
    Run the stress test experiment (stress_test regime only).

    Args:
        total_timesteps: Override training budget.
        eval_only: If True, load saved models instead of training.
        quick_test: Use reduced timesteps.

    Returns:
        Results dict with stress_test regime results.
    """
    print("\n" + "=" * 70)
    print("  STRESS TEST: ELEVATED DISRUPTION SCENARIO")
    print("  p_start=0.15, mean_duration=10, mu1=0.3, mu2=0.4")
    print("=" * 70)

    return run_phase2(
        regimes=["stress_test"],
        total_timesteps=total_timesteps,
        eval_only=eval_only,
        quick_test=quick_test,
    )
