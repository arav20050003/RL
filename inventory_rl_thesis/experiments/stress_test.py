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

    results = run_phase2(
        regimes=["stress_test"],
        total_timesteps=total_timesteps,
        eval_only=eval_only,
        quick_test=quick_test,
    )
    check_policy_collapse(results)
    return results


def check_policy_collapse(results: dict[str, Any]) -> None:
    import numpy as np
    from config import MODELS_DIR
    from envs.disruption_env import DisruptionEnv
    from policies.agents import load_agent
    
    for name, key in [("PPO Blind", "ppo_blind"), ("PPO Disrupt-Aware", "ppo_aware"), ("PPO LLM-Augmented", "ppo_llm")]:
        if "stress_test" not in results or key not in results["stress_test"]:
            continue
        model_path = MODELS_DIR / f"phase2_stress_test_{key}_final"
        if not model_path.with_suffix(".zip").exists():
            continue
            
        env = DisruptionEnv(disruption_regime="stress_test", include_true_flag=(key == "ppo_aware"), include_risk_score=(key == "ppo_llm"))
        agent = load_agent(model_path, env, "ppo")
        
        all_actions = []
        for ep in range(5):
            obs, _ = env.reset(seed=42 + ep)
            done = False
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                all_actions.append(action)
                obs, _, term, trunc, _ = env.step(action)
                done = term or trunc
                
        mean_act = np.mean(all_actions, axis=0)
        if np.all(mean_act > 0.95):
            print(f"    NOTE: {name} converged to [1.0, 1.0] saturation")
            print("     policy — consistent with policy collapse under extreme")
            print("     distributional shift (see thesis Section X.X)")
            results["stress_test"][key]["collapse_detected"] = True
        else:
            results["stress_test"][key]["collapse_detected"] = False
